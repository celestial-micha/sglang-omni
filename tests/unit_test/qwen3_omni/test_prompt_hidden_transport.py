# SPDX-License-Identifier: Apache-2.0
"""Prompt-layer transport through the actual output processor and talker builder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from torch import nn

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_scheduler_adapters,
    make_thinker_stream_output_builder,
    should_generate_audio_output,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer

HIDDEN_LAYER = 24
HIDDEN_WIDTH = 4
VOCAB_SIZE = 128
PROMPT_TOKENS = (10, 20, 30, 32, 33, 31, 99, 10, 40, 41)


def make_transport_request(
    prompt_rows: int,
    *,
    request_id: str = "prompt",
    audio_output: bool = True,
    speech_enabled: bool = True,
) -> SGLangARRequestData:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=[],
            params={"stream": True, "max_new_tokens": 4},
            metadata={"output_modalities": ["audio"] if audio_output else ["text"]},
        ),
        data={"prompt": {"input_ids": torch.arange(prompt_rows, dtype=torch.long)}},
    )
    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(vocab_size=VOCAB_SIZE),
        vocab_size=VOCAB_SIZE,
        speech_enabled=speech_enabled,
    )
    return request_builder(payload)


def set_prompt_chunk(
    request_data: SGLangARRequestData, start: int, end: int, *, middle: bool
) -> None:
    request_data.req.set_extend_range(start, end)
    request_data.req.inflight_middle_chunks = int(middle)


def should_capture_prompt(request: SchedulerRequest) -> bool:
    return not request.data.req.output_ids and should_generate_audio_output(
        request.data.stage_payload
    )


def test_output_processor_splits_a_full_batch_by_actual_request_ranges() -> None:
    first, second = make_transport_request(2), make_transport_request(
        3, request_id="other"
    )
    set_prompt_chunk(first, 0, 2, middle=False)
    set_prompt_chunk(second, 0, 3, middle=False)
    requests = [
        SchedulerRequest("prompt", data=first),
        SchedulerRequest("other", data=second),
    ]
    hidden = torch.arange(5 * HIDDEN_WIDTH, dtype=torch.float32).reshape(
        5, HIDDEN_WIDTH
    )
    outputs = SGLangOutputProcessor(capture_hidden=True).process(
        SimpleNamespace(
            next_token_ids=torch.tensor([11, 12]),
            logits_output=SimpleNamespace(hidden_states=hidden),
        ),
        SchedulerOutput(
            requests,
            SimpleNamespace(
                reqs=[first.req, second.req], forward_mode=ForwardMode.EXTEND
            ),
        ),
    )
    assert torch.equal(outputs["prompt"].extra["hidden_states"], hidden[:2])
    assert torch.equal(outputs["other"].extra["hidden_states"], hidden[2:])
    stream_builder = make_thinker_stream_output_builder(
        speech_enabled=True, thinker_hidden_layer=HIDDEN_LAYER
    )
    first_messages = stream_builder("prompt", first, outputs["prompt"])
    second_messages = stream_builder("other", second, outputs["other"])
    assert torch.equal(first_messages[1].data, hidden[:2])
    assert torch.equal(second_messages[1].data, hidden[2:])
    assert first_messages[0].data.tolist() == [11]
    assert second_messages[0].data.tolist() == [12]


def test_mixed_batch_does_not_capture_or_replace_decode_rows() -> None:
    prefill, decode = make_transport_request(3), make_transport_request(
        1, request_id="decode"
    )
    set_prompt_chunk(prefill, 0, 3, middle=False)
    set_prompt_chunk(decode, 0, 1, middle=False)
    decode.req.output_ids.append(7)
    requests = [
        SchedulerRequest("prompt", data=prefill),
        SchedulerRequest("decode", data=decode),
    ]
    hidden = torch.arange(4 * HIDDEN_WIDTH, dtype=torch.float32).reshape(
        4, HIDDEN_WIDTH
    )
    processor = SGLangOutputProcessor(
        capture_hidden=True, should_emit_hidden=should_capture_prompt
    )
    outputs = processor.process(
        SimpleNamespace(
            next_token_ids=torch.tensor([11, 12]),
            logits_output=SimpleNamespace(hidden_states=hidden),
        ),
        SchedulerOutput(
            requests,
            SimpleNamespace(
                reqs=[prefill.req, decode.req], forward_mode=ForwardMode.EXTEND
            ),
        ),
    )
    assert torch.equal(outputs["prompt"].extra["hidden_states"], hidden[:3])
    assert outputs["decode"].extra is None
    stream_builder = make_thinker_stream_output_builder(
        speech_enabled=True, thinker_hidden_layer=HIDDEN_LAYER
    )
    messages = stream_builder("decode", decode, outputs["decode"])
    assert [message.data.tolist() for message in messages] == [[12], [12]]
    assert all(message.metadata == {"token_id": 12} for message in messages)


def test_capture_mode_requests_full_only_for_a_speech_prompt() -> None:
    prompt = SchedulerRequest("prompt", data=make_transport_request(3))
    text = SchedulerRequest("text", data=make_transport_request(3, audio_output=False))
    decode = SchedulerRequest("decode", data=make_transport_request(3))
    decode.data.req.output_ids.append(7)
    runner = object.__new__(ThinkerModelRunner)
    runner.output_processor = SGLangOutputProcessor(
        capture_hidden=True, should_emit_hidden=should_capture_prompt
    )
    assert (
        runner.requested_capture_hidden_mode_prefill(None, [prompt, text, decode])
        == CaptureHiddenMode.FULL
    )
    assert (
        runner.requested_capture_hidden_mode_prefill(None, [text, decode])
        == CaptureHiddenMode.NULL
    )
    assert (
        runner.requested_capture_hidden_mode_decode(None, [prompt])
        == CaptureHiddenMode.NULL
    )
    runner.output_processor = SGLangOutputProcessor(capture_hidden=False)
    assert (
        runner.requested_capture_hidden_mode_prefill(None, [prompt])
        == CaptureHiddenMode.NULL
    )


def test_replayed_chunks_replace_old_rows_and_emit_the_complete_prompt() -> None:
    request_data = make_transport_request(7)
    stream_builder = make_thinker_stream_output_builder(
        speech_enabled=True, thinker_hidden_layer=HIDDEN_LAYER
    )
    stale = torch.full((7, HIDDEN_WIDTH), -100.0)
    expected = torch.arange(7 * HIDDEN_WIDTH, dtype=torch.float32).reshape(
        7, HIDDEN_WIDTH
    )
    for start, end, hidden, middle in (
        (0, 3, stale[:3], True),
        (0, 2, expected[:2], True),
        (2, 5, expected[2:5], True),
        (4, 7, expected[4:], False),
    ):
        set_prompt_chunk(request_data, start, end, middle=middle)
        messages = stream_builder(
            "prompt",
            request_data,
            RequestOutput("prompt", data=11, extra={"hidden_states": hidden}),
        )
        if middle:
            assert messages == []
        else:
            assert torch.equal(messages[1].data, expected)
            assert messages[1].metadata == {
                "token_id": 11,
                "thinker_hidden_layer": HIDDEN_LAYER,
            }
    request_data.req.output_ids.append(11)
    messages = stream_builder("prompt", request_data, RequestOutput("prompt", data=12))
    assert [message.data.tolist() for message in messages] == [[12], [12]]


def test_interleaved_requests_keep_independent_prompt_snapshots() -> None:
    first, second = make_transport_request(4), make_transport_request(
        5, request_id="other"
    )
    stream_builder = make_thinker_stream_output_builder(
        speech_enabled=True, thinker_hidden_layer=HIDDEN_LAYER
    )
    first_hidden = torch.full((4, HIDDEN_WIDTH), 101.0)
    second_hidden = torch.full((5, HIDDEN_WIDTH), 202.0)
    borrowed = first_hidden[:2].clone()
    set_prompt_chunk(first, 0, 2, middle=True)
    assert (
        stream_builder(
            "prompt",
            first,
            RequestOutput("prompt", data=None, extra={"hidden_states": borrowed}),
        )
        == []
    )
    borrowed.fill_(-500.0)
    set_prompt_chunk(second, 0, 3, middle=True)
    assert (
        stream_builder(
            "other",
            second,
            RequestOutput(
                "other", data=None, extra={"hidden_states": second_hidden[:3]}
            ),
        )
        == []
    )
    set_prompt_chunk(first, 2, 4, middle=False)
    first_messages = stream_builder(
        "prompt",
        first,
        RequestOutput("prompt", data=11, extra={"hidden_states": first_hidden[2:]}),
    )
    set_prompt_chunk(second, 3, 5, middle=False)
    second_messages = stream_builder(
        "other",
        second,
        RequestOutput("other", data=12, extra={"hidden_states": second_hidden[3:]}),
    )
    assert torch.equal(first_messages[1].data, first_hidden)
    assert torch.equal(second_messages[1].data, second_hidden)


@pytest.mark.parametrize(
    "failure", ["missing", "incomplete", "gap", "too_many_rows", "wrong_width"]
)
def test_invalid_prompt_capture_never_emits_a_fabricated_complete_tensor(
    failure: Literal["missing", "incomplete", "gap", "too_many_rows", "wrong_width"],
) -> None:
    request_data = make_transport_request(4)
    stream_builder = make_thinker_stream_output_builder(
        speech_enabled=True, thinker_hidden_layer=HIDDEN_LAYER
    )
    set_prompt_chunk(request_data, 0, 2, middle=True)
    if failure in ("gap", "wrong_width"):
        assert (
            stream_builder(
                "prompt",
                request_data,
                RequestOutput(
                    "prompt",
                    data=None,
                    extra={"hidden_states": torch.ones(2, HIDDEN_WIDTH)},
                ),
            )
            == []
        )
    else:
        pass
    if failure == "missing":
        set_prompt_chunk(request_data, 0, 4, middle=False)
        output = RequestOutput("prompt", data=11)
    elif failure == "incomplete":
        set_prompt_chunk(request_data, 0, 2, middle=False)
        output = RequestOutput(
            "prompt", data=11, extra={"hidden_states": torch.ones(2, HIDDEN_WIDTH)}
        )
    elif failure == "gap":
        set_prompt_chunk(request_data, 3, 4, middle=False)
        output = RequestOutput(
            "prompt", data=11, extra={"hidden_states": torch.ones(1, HIDDEN_WIDTH)}
        )
    elif failure == "too_many_rows":
        set_prompt_chunk(request_data, 0, 4, middle=False)
        output = RequestOutput(
            "prompt", data=11, extra={"hidden_states": torch.ones(5, HIDDEN_WIDTH)}
        )
    else:
        set_prompt_chunk(request_data, 2, 4, middle=False)
        output = RequestOutput(
            "prompt", data=11, extra={"hidden_states": torch.ones(2, 1)}
        )
    with pytest.raises((AssertionError, RuntimeError)):
        stream_builder("prompt", request_data, output)


def test_speech_prefix_ownership_is_unique_even_when_request_ids_are_reused() -> None:
    first = make_transport_request(4, request_id="reused")
    second = make_transport_request(4, request_id="reused")
    assert first.req.extra_key and second.req.extra_key
    assert first.req.extra_key != second.req.extra_key
    assert make_transport_request(4, audio_output=False).req.extra_key is None
    assert make_transport_request(4, speech_enabled=False).req.extra_key is None


class SmallProjectionTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(accept_hidden_layer=HIDDEN_LAYER)
        self.activation_dtype = torch.float32
        self.model = nn.Module()
        self.model.codec_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_WIDTH)
        self.text_projection = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        self.hidden_projection = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        with torch.no_grad():
            self.text_projection.weight.copy_(torch.eye(HIDDEN_WIDTH) * 2)
            self.hidden_projection.weight.copy_(torch.eye(HIDDEN_WIDTH) * 3)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.codec_embedding


def make_talker_prefill(
    tmp_path: Path,
) -> tuple[TalkerPrefillBuilder, StagePayload, torch.Tensor]:
    embedding_table = torch.arange(
        VOCAB_SIZE * HIDDEN_WIDTH, dtype=torch.float32
    ).reshape(VOCAB_SIZE, HIDDEN_WIDTH)
    save_file(
        {"thinker.model.embed_tokens.weight": embedding_table},
        str(tmp_path / "model.safetensors"),
    )
    builder = TalkerPrefillBuilder(
        model=SmallProjectionTalker(),
        model_path=str(tmp_path),
        audio_token_id=30,
        image_token_id=32,
        video_token_id=33,
        tts_bos_token_id=1,
        tts_eos_token_id=2,
        tts_pad_token_id=3,
        im_start_token_id=10,
        im_end_token_id=99,
        system_token_id=19,
        user_token_id=20,
        assistant_token_id=40,
        codec_bos_id=5,
        codec_nothink_id=1,
        codec_think_bos_id=2,
        codec_think_eos_id=3,
        codec_pad_id=4,
    )
    payload = StagePayload(
        request_id="prompt",
        request=OmniRequest(inputs=[], params={}),
        data={
            "prompt": {"input_ids": torch.tensor(PROMPT_TOKENS)},
            "thinker_inputs": {
                "model_inputs": {
                    "audio_embeds": torch.full((1, HIDDEN_WIDTH), 1000.0),
                    "image_embeds": torch.full((1, HIDDEN_WIDTH), 2000.0),
                    "video_embeds": torch.full((1, HIDDEN_WIDTH), 3000.0),
                }
            },
        },
    )
    return builder, payload, embedding_table


def test_talker_uses_captured_layer_only_at_multimodal_positions(
    tmp_path: Path,
) -> None:
    builder, payload, embedding_table = make_talker_prefill(tmp_path)
    hidden = (
        torch.arange(len(PROMPT_TOKENS) * HIDDEN_WIDTH, dtype=torch.float32).reshape(
            len(PROMPT_TOKENS), HIDDEN_WIDTH
        )
        + 500
    )
    chunk = StreamItem(
        0, hidden, "thinker", {"token_id": 11, "thinker_hidden_layer": HIDDEN_LAYER}
    )
    result = builder.build_prompt_prefill(payload, [chunk], thinker_done=False)
    expected_user = embedding_table[torch.tensor(PROMPT_TOKENS[:7])] * 2
    expected_user[2:5] = hidden[2:5] * 3
    assert torch.equal(result["input_embeds"][:7], expected_user)


@pytest.mark.parametrize(
    "failure",
    ["missing_layer", "wrong_layer", "missing_hidden", "wrong_rows", "wrong_width"],
)
def test_multimodal_talker_rejects_missing_or_incompatible_prompt_states(
    tmp_path: Path,
    failure: Literal[
        "missing_layer", "wrong_layer", "missing_hidden", "wrong_rows", "wrong_width"
    ],
) -> None:
    builder, payload, _ = make_talker_prefill(tmp_path)
    hidden = torch.ones(len(PROMPT_TOKENS), HIDDEN_WIDTH)
    metadata = {"token_id": 11, "thinker_hidden_layer": HIDDEN_LAYER}
    if failure == "missing_layer":
        metadata.pop("thinker_hidden_layer")
    elif failure == "wrong_layer":
        metadata["thinker_hidden_layer"] = HIDDEN_LAYER - 1
    elif failure == "missing_hidden":
        hidden = torch.tensor([11])
    elif failure == "wrong_rows":
        hidden = hidden[:-1]
    else:
        hidden = hidden[:, :1]
    with pytest.raises((AssertionError, KeyError)):
        builder.build_prompt_prefill(
            payload, [StreamItem(0, hidden, "thinker", metadata)], thinker_done=False
        )


def test_text_only_talker_prefill_does_not_require_prompt_hidden(
    tmp_path: Path,
) -> None:
    builder, payload, embedding_table = make_talker_prefill(tmp_path)
    prompt = torch.tensor(PROMPT_TOKENS)
    prompt[2:5] = 31
    payload.data["prompt"]["input_ids"] = prompt
    chunk = StreamItem(0, torch.tensor([11]), "thinker", {"token_id": 11})
    result = builder.build_prompt_prefill(payload, [chunk], thinker_done=False)
    assert torch.equal(result["input_embeds"][:7], embedding_table[prompt[:7]] * 2)
