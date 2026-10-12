# SPDX-License-Identifier: Apache-2.0
"""Correctness and routing tests for MOSS-TTS Delay sampling graphs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.sampler import (
    DelayGraphBatch,
    MossTTSDelayAudioGraphSampler,
)
from sglang_omni.models.moss_tts.sampling_cuda_graph import (
    MossTTSDelaySamplingCudaGraphRunner,
)
from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel
from sglang_omni.platforms import current_platform

_INT64_MAX = torch.iinfo(torch.int64).max
_MOSS_DELAY_N_VQ = 32
_MOSS_DELAY_AUDIO_VOCAB = 1025


def graph_model_config(*, n_vq: int = 2, audio_vocab: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        n_vq=n_vq,
        channels=n_vq + 1,
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=audio_vocab - 1,
        im_end_token_id=14,
        vocab_size_list=[20] + [audio_vocab] * n_vq,
    )


def sampling_data(
    *,
    profile: str,
    repetition_penalty: float = 1.0,
) -> SimpleNamespace:
    if profile == "default":
        values = (1.5, 1.0, 50, 1.7, 0.8, 25)
    elif profile == "greedy":
        values = (0.0, 0.7, 7, 0.0, 0.6, 5)
    elif profile == "custom":
        values = (1.0, 1.0, 50, 1.6, 0.8, 25)
    else:
        raise ValueError(profile)
    return SimpleNamespace(
        text_temperature=values[0],
        text_top_p=values[1],
        text_top_k=values[2],
        audio_temperature=values[3],
        audio_top_p=values[4],
        audio_top_k=values[5],
        audio_repetition_penalty=repetition_penalty,
    )


class CudaGraphModel:
    def __init__(self, device: torch.device) -> None:
        self.config = graph_model_config(
            n_vq=_MOSS_DELAY_N_VQ,
            audio_vocab=_MOSS_DELAY_AUDIO_VOCAB,
        )
        self.device = device
        self.delay_graph_sampler = MossTTSDelayAudioGraphSampler(self.config).to(device)

    def sample_delay_fixed_shape(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        *,
        audio_sample_output=None,
    ):
        return self.delay_graph_sampler(
            control_logits,
            audio_logits,
            batch,
            audio_sample_output=audio_sample_output,
        )


def request_data(
    delay_state: torch.Tensor,
    *,
    seed: int,
    generation_steps: int,
) -> SimpleNamespace:
    profile = sampling_data(profile="default")
    return SimpleNamespace(
        delay_state=delay_state,
        audio_length=int(delay_state[0]),
        delayed_length=int(delay_state[1]),
        is_audio=bool(int(delay_state[2])),
        sampling_seed=seed,
        generation_steps=generation_steps,
        text_temperature=profile.text_temperature,
        text_top_p=profile.text_top_p,
        text_top_k=profile.text_top_k,
        audio_temperature=profile.audio_temperature,
        audio_top_p=profile.audio_top_p,
        audio_top_k=profile.audio_top_k,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
    )


def test_npu_sampling_graph_setup_rejects_missing_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "is_npu", lambda: True)
    monkeypatch.setattr(
        MossTTSDelaySamplingCudaGraphRunner,
        "capture",
        lambda **kwargs: SimpleNamespace(graphs={1: object()}),
    )
    model = SimpleNamespace(
        device=torch.device("cpu"), sampling_graph_support_reason=lambda: None
    )
    with pytest.raises(RuntimeError, match="NPU sampling graph capture is incomplete"):
        MossTTSDelaySGLangModel.init_sampling_graphs(model, [1, 2])


@pytest.mark.accelerator
@pytest.mark.parametrize("device_type", ["npu"])
def test_fixed_shape_sampling_matches_current_compacted_eager(device_type: str) -> None:
    if device_type == "npu" and not current_platform.is_npu():
        pytest.skip("Ascend NPU is required")
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device(device_type)
    config = graph_model_config(
        n_vq=_MOSS_DELAY_N_VQ,
        audio_vocab=_MOSS_DELAY_AUDIO_VOCAB,
    )
    sampler = MossTTSDelayAudioGraphSampler(config).to(device)
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=config,
        text_control_token_ids=torch.tensor([12, 13], device=device),
    )
    initial_states = torch.tensor(
        [
            [1, _INT64_MAX, 1],
            [17, _INT64_MAX, 1],
            [_MOSS_DELAY_N_VQ + 1, _INT64_MAX, 1],
            [_MOSS_DELAY_N_VQ + 1, 0, 1],
            [_MOSS_DELAY_N_VQ + 1, _MOSS_DELAY_N_VQ - 1, 1],
            [_MOSS_DELAY_N_VQ + 1, _MOSS_DELAY_N_VQ, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    seeds = torch.tensor(
        [101, 202, 303, 404, 505, 606], dtype=torch.long, device=device
    )
    steps = torch.tensor([1, 7, 8, 9, 10, 11], dtype=torch.long, device=device)
    datas = [
        request_data(
            initial_states[index].clone(),
            seed=int(seeds[index]),
            generation_steps=int(steps[index]),
        )
        for index in range(int(initial_states.shape[0]))
    ]
    generator = torch.Generator(device=device).manual_seed(20260722)
    batch_size = int(initial_states.shape[0])
    channel_logits = [
        torch.randn(batch_size, 2, device=device, generator=generator),
        *[
            torch.randn(
                batch_size,
                _MOSS_DELAY_AUDIO_VOCAB,
                device=device,
                generator=generator,
            )
            for _ in range(_MOSS_DELAY_N_VQ)
        ],
    ]

    eager_rows = runner.sample_rows(
        [logits.clone() for logits in channel_logits],
        datas,
        n_vq=_MOSS_DELAY_N_VQ,
        is_audio=True,
    )
    eager_state = torch.stack([data.delay_state for data in datas])
    graph_batch = DelayGraphBatch(
        delay_state=initial_states,
        seeds=seeds,
        generation_steps=steps,
    )
    fixed = sampler(
        channel_logits[0],
        torch.stack(channel_logits[1:], dim=1),
        graph_batch,
    )
    torch.get_device_module(device).synchronize()

    assert torch.equal(eager_rows, fixed.rows)
    assert torch.equal(eager_state, fixed.next_delay_state)


@pytest.mark.accelerator
@pytest.mark.parametrize("device_type", ["npu"])
def test_sampling_cuda_graph_replay_matches_fixed_shape_eager(device_type: str) -> None:
    if device_type == "npu" and not current_platform.is_npu():
        pytest.skip("Ascend NPU is required")
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device(device_type)
    model = CudaGraphModel(device)
    runner = MossTTSDelaySamplingCudaGraphRunner.capture(
        model=model,
        capture_bs=(2,),
    )
    assert set(runner.graphs) == {2}
    assert runner.can_replay(1)
    assert runner.can_replay(2)

    generator = torch.Generator(device=device).manual_seed(20260722)
    delay_states = [
        [1, _INT64_MAX, 1],
        [17, _INT64_MAX, 1],
        [_MOSS_DELAY_N_VQ + 1, _INT64_MAX, 1],
        [_MOSS_DELAY_N_VQ + 1, 0, 1],
        [_MOSS_DELAY_N_VQ + 1, _MOSS_DELAY_N_VQ - 1, 1],
        [_MOSS_DELAY_N_VQ + 1, _MOSS_DELAY_N_VQ, 1],
        [0, _INT64_MAX, 0],
    ]
    for replay_index, batch_size in enumerate((2, 1, 2, 2, 2)):
        control_logits = torch.randn(
            batch_size,
            2,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        audio_logits = torch.randn(
            batch_size,
            _MOSS_DELAY_N_VQ,
            _MOSS_DELAY_AUDIO_VOCAB,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        batch = DelayGraphBatch(
            delay_state=torch.tensor(
                [
                    delay_states[(replay_index * 2 + index) % len(delay_states)]
                    for index in range(batch_size)
                ],
                device=device,
            ),
            seeds=torch.arange(batch_size, device=device, dtype=torch.long)
            + 1_234_567
            + replay_index * 100,
            generation_steps=torch.arange(batch_size, device=device, dtype=torch.long)
            + 7
            + replay_index,
        )

        expected = model.sample_delay_fixed_shape(
            control_logits,
            audio_logits,
            batch,
        )
        actual = runner.replay(
            control_logits,
            audio_logits,
            batch,
        )
        actual_rows = actual.rows.clone()
        actual_state = actual.next_delay_state.clone()
        torch.get_device_module(device).synchronize()

        assert torch.equal(expected.rows, actual_rows)
        assert torch.equal(expected.next_delay_state, actual_state)
        if batch_size == 1:
            static_inputs = runner.require_inputs()
            assert not torch.count_nonzero(static_inputs.control_logits[1:2])
            assert not torch.count_nonzero(static_inputs.audio_logits[1:2])
            assert not torch.count_nonzero(static_inputs.delay_state[1:2])
            assert not torch.count_nonzero(static_inputs.seeds[1:2])
            assert not torch.count_nonzero(static_inputs.generation_steps[1:2])
    runner.clear()
