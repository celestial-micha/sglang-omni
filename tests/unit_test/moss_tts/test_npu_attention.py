# SPDX-License-Identifier: Apache-2.0
"""Parameter registration and configuration of the MOSS NPU attention adapter."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.models.qwen3 import Qwen3Attention

from sglang_omni.models.moss_tts.sglang_model import MossNPUAttention


@pytest.mark.parametrize("rope_scaling", [None, {"rope_type": "linear", "factor": 2.0}])
def test_npu_attention_reuses_weights_and_rotary_configuration(
    rope_scaling: dict[str, str | float] | None,
) -> None:
    original = Qwen3Attention.__new__(Qwen3Attention)
    torch.nn.Module.__init__(original)
    original.qkv_proj = torch.nn.Linear(4, 12, bias=False)
    original.o_proj = torch.nn.Linear(4, 4, bias=False)
    original.q_norm = torch.nn.LayerNorm(4)
    original.k_norm = torch.nn.LayerNorm(4)
    original.rotary_emb = torch.nn.Identity()
    original.rotary_emb.rope_scaling = rope_scaling
    original.attn = torch.nn.Identity()
    original.head_dim = 4
    original.scaling = 0.5
    original.use_fused_qk_norm_mrope = rope_scaling is not None
    original.register_buffer("position_cache", torch.ones(4), persistent=False)
    expected_parameters = dict(original.named_parameters())
    expected_state = original.state_dict()

    adapted = MossNPUAttention(original)

    assert dict(adapted.named_parameters()).keys() == expected_parameters.keys()
    assert adapted.state_dict().keys() == expected_state.keys()
    for name, parameter in adapted.named_parameters():
        assert parameter is expected_parameters[name]
    for name, tensor in adapted.state_dict().items():
        assert tensor.data_ptr() == expected_state[name].data_ptr()
    for name in ("qkv_proj", "o_proj", "q_norm", "k_norm", "rotary_emb", "attn"):
        assert getattr(adapted, name) is getattr(original, name)
    assert adapted.use_fused_qk_norm_mrope == original.use_fused_qk_norm_mrope
    assert adapted.rotary_emb.rope_scaling == rope_scaling
    assert adapted.scaling == original.scaling
    assert adapted.position_cache is original.position_cache

    adapted.register_buffer("extra_cache", torch.zeros(1), persistent=False)
    assert "extra_cache" not in original._buffers  # noqa: leading-underscore
    assert (
        "extra_cache" not in original._non_persistent_buffers_set
    )  # noqa: leading-underscore
