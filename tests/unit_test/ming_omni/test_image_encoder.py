# SPDX-License-Identifier: Apache-2.0
"""Distributed backend policy tests for the Ming image encoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt import server_args
from sglang.srt.distributed import parallel_state
from sglang.srt.layers import dp_attention
from sglang.srt.layers import linear as linear_layers
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from sglang_omni.models.ming_omni.components import image_encoder, npu_tensor_parallel
from sglang_omni.models.ming_omni.components.npu_tensor_parallel import (
    CanonicalVisionLinearMethod,
    OutputParallelVisionLinear,
    configure_npu_tensor_parallel,
)
from sglang_omni.models.ming_omni.components.projectors import VisionProjector


@pytest.mark.parametrize("tensor_parallel_size", [2, 4, 8])
@pytest.mark.parametrize("has_bias", [False, True])
def test_canonical_column_checkpoint_shards(
    tensor_parallel_size: int, has_bias: bool
) -> None:
    weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8) / 16
    bias = torch.arange(16, dtype=torch.float32) / 8 if has_bias else None
    x = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8) / 4
    outputs = []
    for rank in range(tensor_parallel_size):
        layer = ColumnParallelLinear(
            8,
            16,
            bias=has_bias,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=tensor_parallel_size,
        )
        layer.weight.weight_loader(layer.weight, weight)
        if bias is None:
            local_bias = None
        else:
            layer.bias.weight_loader(layer.bias, bias)
            local_bias = layer.bias
        outputs.append(CanonicalVisionLinearMethod().apply(layer, x, local_bias))
        assert layer.weight.numel() == weight.numel() // tensor_parallel_size
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1),
        torch.nn.functional.linear(x, weight, bias),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tensor_parallel_size", [2, 4, 8])
def test_canonical_packed_qkv_checkpoint_shards(tensor_parallel_size: int) -> None:
    weight = torch.arange(48 * 8, dtype=torch.float32).reshape(48, 8) / 16
    bias = torch.arange(48, dtype=torch.float32) / 8
    x = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8) / 4
    outputs = []
    for rank in range(tensor_parallel_size):
        layer = QKVParallelLinear(
            8,
            2,
            8,
            bias=True,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=tensor_parallel_size,
        )
        layer.weight.weight_loader(layer.weight, weight)
        layer.bias.weight_loader(layer.bias, bias)
        outputs.append(CanonicalVisionLinearMethod().apply(layer, x, layer.bias))
        assert layer.weight.numel() == weight.numel() // tensor_parallel_size
    joined = torch.cat(
        [
            torch.cat([output.chunk(3, dim=-1)[part] for output in outputs], dim=-1)
            for part in range(3)
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        joined, torch.nn.functional.linear(x, weight, bias), rtol=0, atol=0
    )


@pytest.mark.parametrize("tensor_parallel_size", [2, 4, 8])
@pytest.mark.parametrize("has_bias", [False, True])
def test_output_parallel_checkpoint_owns_complete_input_rows(
    tensor_parallel_size: int, has_bias: bool
) -> None:
    weight = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) / 16
    bias = torch.arange(8, dtype=torch.float32) / 8 if has_bias else None
    x = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16) / 4
    outputs = []
    for rank in range(tensor_parallel_size):
        source = RowParallelLinear(
            16,
            8,
            bias=has_bias,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=tensor_parallel_size,
        )
        layer = OutputParallelVisionLinear(source)
        layer.weight.weight_loader(layer.weight, weight)
        if bias is None:
            local_bias = None
        else:
            layer.bias.weight_loader(layer.bias, bias)
            local_bias = layer.bias
        outputs.append(layer.quant_method.apply(layer, x, local_bias))
        assert layer.weight.numel() == weight.numel() // tensor_parallel_size
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1),
        torch.nn.functional.linear(x, weight, bias),
        rtol=0,
        atol=0,
    )


def test_single_rank_layer_configuration_preserves_original_modules() -> None:
    model = torch.nn.Sequential(
        ColumnParallelLinear(4, 8, tp_rank=0, tp_size=1),
        torch.nn.GELU(),
        RowParallelLinear(8, 4, tp_rank=0, tp_size=1),
    )
    layers = list(model.children())
    methods = [model[0].quant_method, model[2].quant_method]
    configure_npu_tensor_parallel(model)
    assert list(model.children()) == layers
    assert [model[0].quant_method, model[2].quant_method] == methods


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_projector_loads_checkpoint_and_unpacks_output_column_layers(
    monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    monkeypatch.setattr(
        linear_layers, "get_parallel", lambda: SimpleNamespace(tp_rank=0, tp_size=1)
    )
    monkeypatch.setattr(
        linear_layers, "get_forward", lambda: SimpleNamespace(sp_active=False)
    )

    # note (Codex): Rank-one identity isolates the projector tuple contract.
    def gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        return input_

    monkeypatch.setattr(linear_layers, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(npu_tensor_parallel, "tensor_model_parallel_all_gather", gather)
    projector = VisionProjector(vision_dim=4, llm_dim=4, mlp_depth=depth)
    for index, layer in enumerate(projector.proj):
        if isinstance(layer, RowParallelLinear):
            projector.proj[index] = OutputParallelVisionLinear(layer)
        elif isinstance(layer, ColumnParallelLinear):
            layer.quant_method = CanonicalVisionLinearMethod()
        else:
            pass
    checkpoint = {
        name: torch.arange(parameter.numel(), dtype=torch.float32).reshape(
            parameter.shape
        )
        / 16
        for name, parameter in projector.named_parameters()
    }
    assert projector.load_weights(checkpoint.items()) == set(checkpoint)
    reference_layers = []
    for index in range(depth):
        layer = torch.nn.Linear(4, 4)
        with torch.no_grad():
            layer.weight.copy_(checkpoint[f"proj.{index * 2}.weight"])
            layer.bias.copy_(checkpoint[f"proj.{index * 2}.bias"])
        reference_layers.append(layer)
        if index < depth - 1:
            reference_layers.append(torch.nn.GELU())
        else:
            pass
    x = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 4
    torch.testing.assert_close(
        projector(x), torch.nn.Sequential(*reference_layers)(x), rtol=0, atol=0
    )


def test_tp_initialization_uses_platform_backend(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        image_encoder,
        "current_platform",
        SimpleNamespace(get_torch_distributed_backend_str=lambda: "hccl"),
    )
    monkeypatch.setattr(dp_attention, "_ATTN_TP_SIZE", None, raising=False)
    monkeypatch.setattr(dp_attention, "_ATTN_TP_RANK", None, raising=False)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        lambda **kwargs: calls.setdefault("distributed", kwargs),
    )
    monkeypatch.setattr(
        parallel_state,
        "initialize_model_parallel",
        lambda **kwargs: calls.setdefault("model_parallel", kwargs),
    )
    monkeypatch.setattr(
        server_args,
        "ServerArgs",
        lambda **kwargs: calls.setdefault("server_args", kwargs),
    )
    monkeypatch.setattr(
        server_args, "set_global_server_args_for_scheduler", lambda args: None
    )
    monkeypatch.setattr(image_encoder.MingImageEncoder, "did_init_tp", False)

    image_encoder.MingImageEncoder.init_sglang_tp(tp_rank=1, tp_size=2)

    distributed = calls["distributed"]
    assert distributed["backend"] == "hccl"
    assert distributed["world_size"] == 2
    assert distributed["rank"] == 1
    assert calls["model_parallel"] == {"tensor_model_parallel_size": 2}
    assert calls["server_args"] == {"model_path": "dummy", "tp_size": 2}
