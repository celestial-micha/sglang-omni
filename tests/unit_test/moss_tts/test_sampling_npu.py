import math

import pytest
import torch

from sglang_omni.models.moss_tts.sampling_kernels import (
    multinomial_with_seed_and_token_ids,
)

_UINT32_MAX_HASH_POSITION = 1_707_985_137

_UINT32_MASK = (1 << 32) - 1


def murmur_hash32_scalar(seed: int, position: int, column: int) -> int:
    """Independent scalar reference for SGLang's four-block MurmurHash32."""

    def rotl32(value: int, bits: int) -> int:
        return ((value << bits) | (value >> (32 - bits))) & _UINT32_MASK

    def mix(hash_value: int, key: int) -> int:
        key = (key * 0xCC9E2D51) & _UINT32_MASK
        key = rotl32(key, 15)
        key = (key * 0x1B873593) & _UINT32_MASK
        hash_value = rotl32(hash_value ^ key, 13)
        return (hash_value * 5 + 0xE6546B64) & _UINT32_MASK

    seed &= (1 << 64) - 1
    hash_value = 0
    for key in (
        seed & _UINT32_MASK,
        (seed >> 32) & _UINT32_MASK,
        position & _UINT32_MASK,
        column & _UINT32_MASK,
    ):
        hash_value = mix(hash_value, key)
    hash_value ^= 16
    hash_value ^= hash_value >> 16
    hash_value = (hash_value * 0x85EBCA6B) & _UINT32_MASK
    hash_value ^= hash_value >> 13
    hash_value = (hash_value * 0xC2B2AE35) & _UINT32_MASK
    return (hash_value ^ (hash_value >> 16)) & _UINT32_MASK


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_model_hash_matches_scalar_reference(device: str) -> None:
    from sglang_omni.models.moss_tts.sampling_npu import moss_hash32
    from sglang_omni.platforms import current_platform

    if device == "npu" and not current_platform.is_npu():
        pytest.skip("requires Ascend NPU")

    seeds = torch.tensor([0, 1, -1, -(2**63), 2**63 - 1, 2**32, 2**40])
    positions = torch.tensor([0, 17, 2**31, 2**32 - 1, 2**32, -1, 2**40])
    columns = torch.tensor([0, 1, 2**31, 2**32 - 1, 2**32, 2**40])
    expected = [
        [murmur_hash32_scalar(int(seed), int(position), int(col)) for col in columns]
        for seed, position in zip(seeds, positions)
    ]
    actual = moss_hash32(seeds.to(device), positions.to(device), columns.to(device))
    assert actual.device.type == device
    assert actual.tolist() == expected


def sample_seeded_scalar(
    scores: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor | None = None,
) -> list[int]:
    """Independent scalar reference for the GPU sampler's Gumbel-max step."""
    columns = (
        list(range(scores.shape[1]))
        if token_ids is None
        else [int(value) for value in token_ids.tolist()]
    )
    result = []
    for row in range(scores.shape[0]):
        values = []
        for column_index, hash_column in enumerate(columns):
            hashed = murmur_hash32_scalar(
                int(seeds[row]), int(positions[row]), hash_column
            )
            uniform = hashed / _UINT32_MASK
            log_uniform = math.log(uniform) if uniform else -math.inf
            log_uniform = min(log_uniform, -(2.0**-32))
            log_uniform = max(log_uniform, -float.fromhex("0x1.fffffffffffffp+1023"))
            gumbel = -math.log(-log_uniform)
            values.append(float(scores[row, column_index]) + gumbel)
        result.append(max(range(len(values)), key=values.__getitem__))
    return result


def test_npu_hash_graph_replay_matches_scalar_reference() -> None:
    from sglang_omni.models.moss_tts.sampling_npu import moss_hash32
    from sglang_omni.platforms import current_platform

    if not current_platform.is_npu():
        pytest.skip("requires Ascend NPU")
    seeds = torch.tensor([0, 2**40, -1, -(2**63)], device="npu")
    positions = torch.tensor([0, 17, 2**32 - 1, 2**40], device="npu")
    columns = torch.tensor([0, 1, 2**31, 2**32 - 1, 2**40], device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        moss_hash32(seeds, positions, columns)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="thread_local"):
        captured = moss_hash32(seeds, positions, columns)
    for step in range(3):
        positions.add_(7)
        graph.replay()
        torch.npu.synchronize()
        expected = [
            [
                murmur_hash32_scalar(int(seed), int(position), int(column))
                for column in columns.cpu()
            ]
            for seed, position in zip(seeds.cpu(), positions.cpu())
        ]
        assert captured.tolist() == expected


def test_npu_compact_sampler_handles_request_batch_shapes() -> None:
    from sglang_omni.platforms import current_platform

    if not current_platform.is_npu():
        pytest.skip("requires Ascend NPU")
    token_ids = torch.tensor([12, 13], device="npu")
    for rows in range(1, 17):
        scores = torch.randn(rows, 2)
        seeds = torch.arange(rows, dtype=torch.long) + 1234
        positions = torch.arange(rows, dtype=torch.long) + 17
        expected = sample_seeded_scalar(scores, seeds, positions, token_ids.cpu())
        for repetition in range(2):
            actual = multinomial_with_seed_and_token_ids(
                scores.npu(), seeds.npu(), positions.npu(), token_ids
            )
            assert actual.shape == (rows,)
            assert actual.tolist() == expected


@pytest.mark.parametrize("compact", [False, True], ids=["full", "compact"])
def test_npu_seeded_sampler_matches_scalar_reference(compact: bool) -> None:
    from sglang_omni.models.moss_tts.sampling_npu import multinomial_with_seed_npu
    from sglang_omni.platforms import current_platform

    if not current_platform.is_npu():
        pytest.skip("requires Ascend NPU")

    generator = torch.Generator().manual_seed(20260929)
    scores = torch.randn(8, 257, generator=generator)
    scores[0].fill_(-torch.inf)
    scores[1, ::3] = -torch.inf
    seeds = torch.tensor([0, 1, 1234, -1, -(2**63), 2**63 - 1, 2**32, 2**40])
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION, 0, 17, 2**32 - 1, 2**32, 2**33, -1, 2**63 - 1]
    )
    token_ids = torch.arange(257) * 65537 if compact else None
    expected = sample_seeded_scalar(scores, seeds, positions, token_ids)
    if compact:
        actual = multinomial_with_seed_and_token_ids(
            scores.npu(), seeds.npu(), positions.npu(), token_ids.npu()
        )
    else:
        actual = multinomial_with_seed_npu(scores.npu(), seeds.npu(), positions.npu())
    assert actual.device.type == "npu"
    assert actual.tolist() == expected


def test_npu_seeded_sampler_masks_max_hash_endpoint() -> None:
    from sglang_omni.models.moss_tts.sampling_npu import multinomial_with_seed_npu
    from sglang_omni.platforms import current_platform

    if not current_platform.is_npu():
        pytest.skip("requires Ascend NPU")

    scores = torch.tensor([[-torch.inf, 0.0]], device="npu")
    seeds = torch.tensor([0], dtype=torch.long, device="npu")
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION], dtype=torch.long, device="npu"
    )
    assert multinomial_with_seed_npu(scores, seeds, positions).item() == 1
    compact = multinomial_with_seed_and_token_ids(
        scores.flip(1),
        seeds,
        positions,
        torch.tensor([1, 0], dtype=torch.long, device="npu"),
    )
    assert compact.item() == 0
