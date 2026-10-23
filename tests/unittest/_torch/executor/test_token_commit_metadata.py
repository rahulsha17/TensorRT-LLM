# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA token-commit metadata reuse, ownership and publication regressions."""

import gc
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_device import DeviceExecutionAuthority, _PendingUse
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class CountingAuthority(DeviceExecutionAuthority):
    def __init__(self) -> None:
        super().__init__(torch.cuda.current_device())
        self.uploads: list[tuple[tuple[int, ...], torch.dtype]] = []
        self.input_calls = 0

    def _upload(self, values: list[int], dtype: torch.dtype, use: _PendingUse) -> torch.Tensor:
        self.uploads.append((tuple(values), dtype))
        return super()._upload(values, dtype, use)

    def _input_values(self, versions: list[ExecutionVersion | None]) -> tuple[list[int], list[int]]:
        self.input_calls += 1
        return super()._input_values(versions)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("compact,steps", [(True, 1), (False, 1), (False, 5)])
def test_metadata_three_uploads_then_zero_with_fresh_outputs(
    dtype: torch.dtype, compact: bool, steps: int
) -> None:
    """Also run this unchanged against the old source: its hit uploads three."""
    authority = CountingAuthority()
    version = ExecutionVersion("tenant", "scope", (1 << 64) - 1)
    authority.publish(version)
    authority.uploads.clear()  # Publication upload is not commit metadata.
    candidates = torch.arange(steps * 4, device=authority.device, dtype=dtype).reshape(steps, 4, 1)
    first = authority.enqueue_token_commit([version, None], [2, 0], candidates, compact)
    first.wait()
    assert authority.uploads == [
        ((0, -1), torch.int32),
        ((-1, 0), torch.int64),
        ((2, 0), torch.int32),
    ]
    authority.uploads.clear()
    candidates.add_(17)
    second = authority.enqueue_token_commit([version, None], [2, 0], candidates, compact)
    result = second.wait()
    assert authority.uploads == [], "identical ready metadata uploaded again"
    assert authority.input_calls == 2
    assert first.tokens_device.data_ptr() != second.tokens_device.data_ptr()
    assert first.tokens_host.data_ptr() != second.tokens_host.data_ptr()
    assert first.allowed_host.data_ptr() != second.allowed_host.data_ptr()
    assert result.allowed == [True, True]
    expected = torch.zeros_like(candidates)
    expected[:, [0, 2], :] = candidates[:, [0, 2], :]
    assert torch.equal(result.tokens_device, expected)
    expected_host = expected[0, [2, 0], 0] if compact else expected
    assert torch.equal(result.tokens_host, expected_host.cpu())


def test_metadata_hit_observes_new_publication_and_unknown_scope() -> None:
    authority = CountingAuthority()
    old = ExecutionVersion("tenant", "scope", 7)
    new = replace(old, epoch=8)
    authority.publish(old)
    candidates = torch.full((1, 2, 1), 41, dtype=torch.int32, device=authority.device)
    assert authority.enqueue_token_commit([old], [1], candidates, True).wait().allowed == [True]
    metadata = authority._token_commit_metadata
    authority.publish(new)
    authority.uploads.clear()
    stale = authority.enqueue_token_commit([old], [1], candidates, True).wait()
    assert authority._token_commit_metadata is metadata
    assert authority.uploads == []
    assert stale.allowed == [False]
    assert stale.tokens_host.tolist() == [0]
    current = authority.enqueue_token_commit([new], [1], candidates, True).wait()
    assert len(authority.uploads) == 3
    assert current.allowed == [True] and current.tokens_host.tolist() == [41]
    with pytest.raises(RuntimeError, match="no device publication"):
        authority.enqueue_token_commit([replace(new, scope_id="unknown")], [1], candidates, True)
    with pytest.raises(RuntimeError, match="restart required"):
        authority.enqueue_token_commit([new], [1], candidates, True)


def test_metadata_reuse_across_candidate_dtype_and_step_count() -> None:
    authority = CountingAuthority()
    metadata = None
    for dtype, steps in [(torch.int32, 1), (torch.int64, 3), (torch.int32, 5)]:
        authority.uploads.clear()
        candidates = torch.arange(steps * 2, dtype=dtype, device=authority.device).reshape(
            steps, 2, 1
        )
        result = authority.enqueue_token_commit([None], [1], candidates, False).wait()
        assert len(authority.uploads) == (3 if metadata is None else 0)
        if metadata is not None:
            assert authority._token_commit_metadata is metadata
        metadata = authority._token_commit_metadata
        expected = torch.zeros_like(candidates)
        expected[:, 1, :] = candidates[:, 1, :]
        assert result.allowed == [True]
        assert result.tokens_host.dtype == dtype
        assert torch.equal(result.tokens_host, expected.cpu())


def test_metadata_key_preserves_order_and_recycled_sequence_identity() -> None:
    authority = CountingAuthority()
    first = ExecutionVersion("tenant", "one", 4)
    second = replace(first, scope_id="two")
    for version in [first, second]:
        authority.publish(version)
    candidates = torch.tensor([[[11], [22], [33]]], dtype=torch.int64, device=authority.device)
    cases = [
        ([first, second], [0, 2], [11, 33]),
        ([second, first], [0, 2], [11, 33]),
        ([second, first], [2, 0], [33, 11]),
        ([None, first], [2, 0], [33, 11]),
    ]
    for versions, slots, expected in cases:
        authority.uploads.clear()
        ticket = authority.enqueue_token_commit(versions, slots, candidates, True)
        assert ticket.wait().tokens_host.tolist() == expected
        assert len(authority.uploads) == 3
        metadata = authority._token_commit_metadata
        assert metadata.key[2] == tuple(slots)
    assert authority._slots[("tenant", "one")] != authority._slots[("tenant", "two")]


def test_metadata_not_ready_reuploads_without_reusing_old_storage() -> None:
    authority = CountingAuthority()
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    candidates = torch.full((1, 2, 1), 19, device=authority.device, dtype=torch.int32)
    # Warm the exact kernel before testing a deliberately backlogged stream.
    authority.enqueue_token_commit([None], [0], candidates, True).wait()
    slow, other = torch.cuda.Stream(), torch.cuda.Stream()
    slow.wait_stream(torch.cuda.current_stream())
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(slow):
        torch.cuda._sleep(1_000_000_000)
        first = authority.enqueue_token_commit([version], [0], candidates, True)
    old = authority._token_commit_metadata
    try:
        assert not old.ready.query(), "test requires an outstanding metadata copy"
        authority.uploads.clear()
        with torch.cuda.stream(other):
            second = authority.enqueue_token_commit([version], [0], candidates, True)
        assert len(authority.uploads) == 3
        assert authority._token_commit_metadata is not old
        assert authority._token_commit_metadata.slots.data_ptr() != old.slots.data_ptr()
        assert first.wait().tokens_host.tolist() == second.wait().tokens_host.tolist() == [19]
    finally:
        slow.synchronize()
        other.synchronize()


def test_metadata_replacement_preserves_old_outstanding_ticket() -> None:
    authority = CountingAuthority()
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    candidates = torch.tensor([[[13], [27]]], device=authority.device, dtype=torch.int32)
    authority.enqueue_token_commit([version], [0], candidates, True).wait()
    old = authority._token_commit_metadata
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    authority.uploads.clear()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        pending = authority.enqueue_token_commit([version], [0], candidates, True)
    try:
        assert not pending.event.query()
        assert authority.uploads == []
        for tensor in [old.slots, old.expected, old.sequence_slots]:
            assert any(value is tensor for value in pending._use.tensors)
        replacement = authority.enqueue_token_commit([version], [1], candidates, True)
        replacement.wait()
        assert authority._token_commit_metadata is not old
        del old
        gc.collect()
        assert pending.wait().tokens_host.tolist() == [13]
    finally:
        stream.synchronize()


def test_metadata_cross_stream_early_authority_release() -> None:
    authority = CountingAuthority()
    candidates = torch.tensor([[[73], [29]]], device=authority.device, dtype=torch.int32)
    authority.enqueue_token_commit([None], [1], candidates, True).wait()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        pending = authority.enqueue_token_commit([None], [1], candidates, True)
    event, host = pending.event, pending.tokens_host
    assert not event.query()
    del pending, authority
    gc.collect()
    # Exercise allocator reuse on the original allocation stream while the hit runs elsewhere.
    scratch = [
        torch.full((1,), -9, device="cuda", dtype=dtype)
        for _ in range(256)
        for dtype in [torch.int32, torch.int64]
    ]
    try:
        event.synchronize()
        assert host.tolist() == [29]
        assert scratch[0].item() == -9
    finally:
        stream.synchronize()


def test_metadata_cache_does_not_allow_graph_capture() -> None:
    authority = CountingAuthority()
    candidates = torch.ones((1, 2, 1), device=authority.device, dtype=torch.int32)
    authority.enqueue_token_commit([None], [0], candidates, True).wait()
    authority.uploads.clear()
    calls = authority.input_calls
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="outside capture"):
            authority.enqueue_token_commit([None], [0], candidates, True)
        candidates.add_(1)
    assert authority.input_calls == calls
    assert authority.uploads == []
    authority._healthy()


def test_metadata_query_error_poisoning_retains_pending_use() -> None:
    authority = CountingAuthority()
    candidates = torch.ones((1, 1, 1), device=authority.device, dtype=torch.int32)
    authority.enqueue_token_commit([None], [0], candidates, True).wait()

    def fail_query() -> bool:
        raise RuntimeError("injected ready query failure")

    authority._token_commit_metadata = replace(
        authority._token_commit_metadata, ready=SimpleNamespace(query=fail_query)
    )
    with pytest.raises(RuntimeError, match="injected ready query failure"):
        authority.enqueue_token_commit([None], [0], candidates, True)
    assert any(use.event is None and use.tensors[0] is candidates for use in authority._pending)
    with pytest.raises(RuntimeError, match="restart required"):
        authority.enqueue_token_commit([None], [0], candidates, True)
