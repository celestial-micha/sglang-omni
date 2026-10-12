"""Ming vision projections with sharded weights and canonical NPU GEMM shapes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

QKV_PROJECTION_COUNT = 3


class CanonicalVisionLinearMethod(UnquantizedLinearMethod):
    """Compute owned output rows at the single-rank GEMM shape."""

    def apply(
        self,
        layer: ColumnParallelLinear,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_rows = layer.weight.shape[0]
        if isinstance(layer, QKVParallelLinear):
            global_projection_rows = layer.output_size // QKV_PROJECTION_COUNT
            local_projection_rows = local_rows // QKV_PROJECTION_COUNT
            owned_rows = torch.cat(
                [
                    torch.arange(
                        projection * global_projection_rows
                        + layer.tp_rank * local_projection_rows,
                        projection * global_projection_rows
                        + (layer.tp_rank + 1) * local_projection_rows,
                        device=x.device,
                    )
                    for projection in range(QKV_PROJECTION_COUNT)
                ]
            )
        else:
            owned_rows = torch.arange(
                layer.tp_rank * local_rows,
                (layer.tp_rank + 1) * local_rows,
                device=x.device,
            )
        weight = layer.weight.new_zeros((layer.output_size, layer.input_size))
        weight.index_copy_(0, owned_rows, layer.weight)
        if bias is None:
            full_bias = None
        else:
            full_bias = bias.new_zeros(layer.output_size)
            full_bias.index_copy_(0, owned_rows, bias)
        output = F.linear(x, weight, full_bias)
        return output.index_select(-1, owned_rows)


class OutputParallelVisionLinear(ColumnParallelLinear):
    """Gather real input shards and concatenate independently owned outputs."""

    input_is_parallel: bool

    def __init__(self, source: RowParallelLinear) -> None:
        assert source.reduce_results and not source.skip_bias_add
        assert not source.use_dp_attention_reduce
        super().__init__(
            source.input_size,
            source.output_size,
            bias=source.bias is not None,
            gather_output=True,
            params_dtype=source.weight.dtype,
            tp_rank=source.tp_rank,
            tp_size=source.tp_size,
        )
        self.input_is_parallel = source.input_is_parallel
        self.quant_method = CanonicalVisionLinearMethod()

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.input_is_parallel:
            input_ = tensor_model_parallel_all_gather(input_, dim=-1)
        else:
            pass
        return super().forward(input_)


def configure_npu_tensor_parallel(model: nn.Module) -> None:
    """Configure the vision/projector layers before checkpoint loading."""
    for parent in list(model.modules()):
        for name, layer in list(parent.named_children()):
            if isinstance(layer, RowParallelLinear) and layer.tp_size > 1:
                assert isinstance(layer.quant_method, UnquantizedLinearMethod)
                parent.add_module(name, OutputParallelVisionLinear(layer))
            elif isinstance(layer, ColumnParallelLinear) and layer.tp_size > 1:
                assert isinstance(layer.quant_method, UnquantizedLinearMethod)
                layer.quant_method = CanonicalVisionLinearMethod()
            else:
                pass
