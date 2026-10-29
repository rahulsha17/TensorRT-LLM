# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA authority kernels. These tests do not need model weights."""

from dataclasses import replace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_device import DeviceExecutionAuthority
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("captured", [False, True])
def test_stable_attention_rows_reject_revisions_and_refresh_reordered_padding(captured):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "attention-batch", 7)
    current = replace(old, epoch=8)
    neighbor = replace(old, scope_id="neighbor")
    authority.publish(old)
    authority.publish(neighbor)
    rows = authority.create_attention_rows(4)
    allowed = torch.empty(4, dtype=torch.bool, device=authority.device)

    def check():
        module._check_versions[(1,)](authority._table, rows.slots, rows.expected, allowed, 4, 128)

    authority.prepare_attention_rows(rows, [old, neighbor])
    check()
    assert allowed.tolist() == [True, True, True, True]
    graph = torch.cuda.CUDAGraph() if captured else None
    if graph is not None:
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            check()
    evaluate = graph.replay if graph is not None else check
    identities = rows._layout
    authority.publish(current)
    authority.prepare_attention_rows(rows, [old, neighbor])
    assert rows._layout is identities
    evaluate()
    assert allowed.tolist() == [False, True, True, True]
    for versions, verdict in (
        ([neighbor, old, current], [True, False, True, True]),
        ([current], [True, True, True, True]),
        ([None, old], [True, False, True, True]),
    ):
        authority.prepare_attention_rows(rows, versions)
        evaluate()
        assert allowed.tolist() == verdict
    assert rows.slots.tolist() == [-1, authority._slots[("tenant", "attention-batch")], -1, -1]


def test_device_epoch_preserves_all_uint64_bits_and_scope_isolation() -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    previous = None
    for epoch in [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]:
        version = ExecutionVersion("tenant", "scope", epoch)
        authority.publish(version)
        assert authority.check([version, None]) == [True, True]
        if previous is not None:
            assert authority.check([previous]) == [False]
        previous = version
    other_scope = replace(previous, scope_id="other", epoch=0)
    other_tenant = replace(other_scope, tenant_id="other")
    authority.publish(other_scope)
    authority.publish(other_tenant)
    authority.publish(replace(other_scope, epoch=1))
    assert authority.check([previous, other_scope, other_tenant]) == [True, False, True]


def test_async_context_receipt_follows_model_work_without_waiting_for_it():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "context-receipt-order", 7)
    new = replace(old, epoch=8)
    authority.publish(old)
    assert authority.check([old]) == [True]
    stream = torch.cuda.Stream()
    ticket = None
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            ticket = authority.enqueue_check([old])
        assert not ticket.query(), "receipt submission waited for the model stream"
        authority.publish(new)
        assert authority.check([new]) == [True]
        assert ticket.wait() == [False]
        assert ticket.query()
    finally:
        stream.synchronize()


def test_async_context_receipt_retains_an_earlier_verdict_after_publication():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "context-receipt-retention", 7)
    new = replace(old, epoch=8)
    authority.publish(old)
    before = authority.enqueue_check([old, None, old])
    assert before.wait() == [True, True, True]
    authority.publish(new)
    after = authority.enqueue_check([old, new, None])
    assert after.wait() == [False, True, True]
    assert before.wait() == [True, True, True]
    assert before._allowed_host.data_ptr() != after._allowed_host.data_ptr()
    assert before.event is not after.event


def test_async_context_receipt_snapshots_input_identity_and_row_order():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "context-receipt-identity", (1 << 64) - 2)
    new = replace(old, epoch=(1 << 64) - 1)
    neighbor = replace(old, scope_id="neighbor")
    authority.publish(new)
    authority.publish(neighbor)
    identities = [neighbor, old, None, new]
    ticket = authority.enqueue_check(identities)
    identities[:] = [new]
    assert ticket.versions == (neighbor, old, None, new)
    assert ticket.wait() == [True, False, True, True]


def test_async_context_receipt_requires_rows_and_accepts_untagged_rows():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    with pytest.raises(ValueError, match="at least one row"):
        authority.enqueue_check([])
    assert authority.check([]) == []
    assert authority.enqueue_check([None, None]).wait() == [True, True]


def test_async_context_receipt_retains_buffers_on_copy_submission_failure(monkeypatch):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "context-receipt-copy-failure", 7)
    authority.publish(version)
    original = torch.Tensor.copy_
    error = RuntimeError("injected context receipt copy failure")
    copies = []

    def fail_after_copy(destination, source, *args, **kwargs):
        result = original(destination, source, *args, **kwargs)
        if destination.device.type == "cpu" and source.device.type == "cuda":
            copies.append((destination, source))
            raise error
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", fail_after_copy)
    try:
        with pytest.raises(RuntimeError) as raised:
            authority.enqueue_check([version, None])
        assert raised.value is error and authority._failure is error
        assert len(copies) == 1
        pending = [use for use in authority._pending if use.event is None]
        assert len(pending) == 1
        for tensor in copies[0]:
            assert any(
                owned.device == tensor.device
                and owned.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                for owned in pending[0].tensors
            )
        with pytest.raises(RuntimeError, match="restart required"):
            authority.enqueue_check([version])
    finally:
        torch.cuda.synchronize()


def test_async_context_receipt_query_failure_poisoning_preserves_ownership(monkeypatch):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "context-receipt-query-failure", 7)
    authority.publish(version)
    ticket = authority.enqueue_check([version])
    assert ticket.wait() == [True]
    retained = tuple(ticket._use.tensors)
    error = RuntimeError("injected context receipt completion failure")

    def fail_query():
        raise error

    monkeypatch.setattr(ticket.event, "query", fail_query)
    with pytest.raises(RuntimeError) as raised:
        ticket.query()
    assert raised.value is error and authority._failure is error
    assert all(left is right for left, right in zip(retained, ticket._use.tensors))
    with pytest.raises(RuntimeError, match="restart required"):
        authority.enqueue_check([version])


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_token_commit_rejects_stale_rows_without_changing_candidate_scratch(compact, dtype) -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", 7)
    current = replace(old, epoch=8)
    other = replace(old, scope_id="other")
    authority.publish(old)
    authority.publish(other)
    authority.publish(current)
    candidates = torch.tensor([[[11], [22], [33], [44]]], dtype=dtype, device=authority.device)
    commit = authority.commit_tokens([old, other, None], [2, 0, 3], candidates, compact)
    assert commit.allowed == [False, True, True]
    assert commit.tokens_device.data_ptr() != candidates.data_ptr()
    assert candidates.flatten().tolist() == [11, 22, 33, 44]
    assert commit.tokens_device.flatten().tolist() == [11, 0, 0, 44]
    assert commit.tokens_host.flatten().tolist() == ([0, 11, 44] if compact else [11, 0, 0, 44])
    # A recycled sequence slot does not recycle its old scope's authority.
    successor = authority.commit_tokens([current], [2], candidates, compact)
    assert successor.allowed == [True]
    assert successor.tokens_device[0, 2, 0].item() == 33


@pytest.mark.parametrize("count", [1, 3, 17])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compact_token_commit_preserves_integer_width_and_request_order(count, dtype):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "compact-width", 7)
    current = replace(old, epoch=8)
    authority.publish(current)
    width = count + 3
    base = 1 << (40 if dtype == torch.int64 else 29)
    values = [(-1 if i % 2 else 1) * (base + i) for i in range(width)]
    candidates = torch.tensor([[[v] for v in values]], dtype=dtype, device=authority.device)
    slots = list(range(width - 1, width - count - 1, -1))
    versions = [None if i % 3 == 0 else old if i % 3 == 1 else current for i in range(count)]
    expected_allowed = [v != old for v in versions]
    result = authority.enqueue_token_commit(versions, slots, candidates, compact=True).wait()

    assert result.allowed == expected_allowed
    assert all(type(value) is bool for value in result.allowed)
    assert result.tokens_host.dtype == dtype
    assert result.tokens_host.shape == (count,)
    assert result.tokens_host.tolist() == [
        values[slot] if allowed else 0 for slot, allowed in zip(slots, expected_allowed)
    ]
    expected_device = [0] * width
    for slot, allowed in zip(slots, expected_allowed):
        if allowed:
            expected_device[slot] = values[slot]
    assert result.tokens_device.flatten().tolist() == expected_device
    assert candidates.flatten().tolist() == values


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compact_token_commit_keeps_outstanding_results_across_revision(dtype):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "compact-overlap", 7)
    current = replace(old, epoch=8)
    authority.publish(old)
    values = torch.tensor([[[101], [-202], [303], [-404]]], dtype=dtype, device=authority.device)
    first = authority.enqueue_token_commit([old, None], [3, 1], values, compact=True)
    # Complete the earlier admitted commit before publication, but defer host
    # consumption until two later tickets have exercised the same allocator.
    first.event.synchronize()
    authority.publish(current)
    stale = authority.enqueue_token_commit([old], [3], values, compact=True)
    successor = authority.enqueue_token_commit([current, None], [1, 2], values, compact=True)
    latest = successor.wait()
    assert latest.allowed == [True, True]
    assert latest.tokens_host.tolist() == [-202, 303]
    denied = stale.wait()
    assert denied.allowed == [False]
    assert denied.tokens_host.tolist() == [0]
    original = first.wait()
    assert original.allowed == [True, True]
    assert original.tokens_host.tolist() == [-404, -202]
    assert values.flatten().tolist() == [101, -202, 303, -404]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compact_token_commit_retains_buffers_after_copy_submission_failure(monkeypatch, dtype):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "compact-copy-failure", 7)
    authority.publish(version)
    candidates = torch.ones((1, 4, 1), dtype=dtype, device=authority.device)
    original = torch.Tensor.copy_
    error = RuntimeError("injected failure after D2H submission")
    copies = []

    def fail_after_copy(destination, source, *args, **kwargs):
        result = original(destination, source, *args, **kwargs)
        if destination.device.type == "cpu" and source.device.type == "cuda":
            copies.append((destination, source))
            raise error
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", fail_after_copy)
    try:
        with pytest.raises(RuntimeError) as raised:
            authority.enqueue_token_commit([version, None], [3, 1], candidates, compact=True)
        assert raised.value is error
        assert authority._failure is error
        assert len(copies) == 1
        pending = [use for use in authority._pending if use.event is None]
        assert len(pending) == 1
        for tensor in copies[0]:
            assert any(
                owned.device == tensor.device
                and owned.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                for owned in pending[0].tensors
            ), "partially submitted transfer lost its owning buffer"
        with pytest.raises(RuntimeError, match="restart required"):
            authority.check([version])
    finally:
        torch.cuda.synchronize()


@pytest.mark.parametrize("steps", [2, 5, 9])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("captured", [False, True])
def test_multi_token_commit_has_one_authority_verdict_for_every_step(steps, dtype, captured):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "speculative", 7)
    current = replace(old, epoch=8)
    neighbor = replace(old, scope_id="neighbor")
    authority.publish(old)
    authority.publish(neighbor)
    source = torch.arange(steps * 4, dtype=dtype, device=authority.device).reshape(steps, 4, 1)
    candidates = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph() if captured else None
    if graph is not None:
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            candidates.copy_(source)
    for version, live in ((old, True), (current, False)):
        authority.publish(version)
        if graph is not None:
            graph.replay()
        else:
            candidates.copy_(source)
        ticket = authority.enqueue_token_commit([old, neighbor, None], [2, 0, 3], candidates, False)
        committed = ticket.wait()
        assert committed.allowed == [live, True, True]
        expected = torch.zeros_like(source)
        expected[:, [0, 3], :] = source[:, [0, 3], :]
        if live:
            expected[:, 2, :] = source[:, 2, :]
        assert torch.equal(committed.tokens_device, expected)
        assert torch.equal(committed.tokens_host, expected.cpu())
        assert torch.equal(candidates, source)


def test_multi_token_commit_cannot_use_compact_single_step_layout():
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    candidates = torch.zeros((3, 4, 1), dtype=torch.int32, device=authority.device)
    with pytest.raises(ValueError, match="exactly one step"):
        authority.enqueue_token_commit([None], [2], candidates, True)


@pytest.mark.parametrize("compact", [False, True])
def test_cached_token_metadata_rechecks_authority_and_survives_batch_replacement(compact):
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "cached-decode", 7)
    current = replace(old, epoch=8)
    authority.publish(old)
    candidates = torch.tensor([[[11], [22], [33], [44]]], device=authority.device)
    first = authority.enqueue_token_commit([old, None], [2, 0], candidates, compact)
    metadata = authority._token_commit_metadata
    assert first.wait().allowed == [True, True]
    authority.publish(current)
    stale = authority.enqueue_token_commit([old, None], [2, 0], candidates + 1, compact)
    assert authority._token_commit_metadata is metadata
    # Change both the expected version and sequence slots before consuming the
    # stale ticket. The old batch's metadata must stay immutable and alive.
    successor = authority.enqueue_token_commit([current, None], [1, 3], candidates + 2, compact)
    assert authority._token_commit_metadata is not metadata
    denied = stale.wait()
    assert denied.allowed == [False, True]
    assert denied.tokens_host.flatten().tolist() == ([0, 12] if compact else [12, 0, 0, 0])
    admitted = successor.wait()
    assert admitted.allowed == [True, True]
    assert admitted.tokens_host.flatten().tolist() == ([24, 46] if compact else [0, 24, 0, 46])
    assert first.tokens_host.flatten().tolist() == ([33, 11] if compact else [11, 0, 33, 0])


def test_unknown_scopes_and_capacity_fail_instead_of_aliasing_epoch_zero() -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device(), capacity=1)
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    authority.publish(version)
    with pytest.raises(RuntimeError, match="capacity"):
        authority.publish(replace(version, scope_id="other"))
    assert authority.check([version]) == [True]
    with pytest.raises(RuntimeError, match="no device publication"):
        authority.check([replace(version, scope_id="unknown")])
    with pytest.raises(RuntimeError, match="restart required"):
        authority.check([version])


def test_publication_does_not_join_an_occupied_model_stream() -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", 0)
    new = replace(old, epoch=1)
    authority.publish(old)
    assert authority.check([old, old]) == [True, True]  # Warm the same check shape.
    model_stream = torch.cuda.Stream()
    model_done = torch.cuda.Event()
    with torch.cuda.stream(model_stream):
        torch.cuda._sleep(1_000_000_000)
        model_done.record()
    try:
        authority.publish(new)
        # The check uses the caller stream; the publisher uses its own stream.
        assert authority.check([old, new]) == [False, True]
        assert not model_done.query(), "publication unexpectedly waited for model execution"
    finally:
        model_done.synchronize()


@pytest.mark.parametrize("captured", [False, True])
def test_publication_during_non_capturing_submission_rejects_old_tokens(captured):
    from threading import Thread

    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "concurrent-submit", 0)
    new = replace(old, epoch=1)
    authority.publish(old)
    candidates = torch.ones((1, 2, 1), dtype=torch.int32, device=authority.device)
    authority.enqueue_token_commit([old, None], [0, 1], candidates, compact=True).wait()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with authority.submission(), torch.cuda.graph(graph):
            candidates.fill_(7)

    errors = []

    def publish():
        try:
            authority.publish(new)
        except BaseException as error:
            errors.append(error)

    publisher = Thread(target=publish)
    try:
        with authority.submission(capture_possible=False):
            if graph is None:
                candidates.fill_(7)
            else:
                graph.replay()
            publisher.start()
            publisher.join(timeout=2)
            assert not publisher.is_alive(), "publication waited for CPU submission to finish"
            assert not errors
            result = authority.enqueue_token_commit(
                [old, None], [0, 1], candidates, compact=True
            ).wait()
            assert result.allowed == [False, True]
            assert result.tokens_host.tolist() == [0, 7]
    finally:
        if publisher.ident is not None:
            publisher.join(timeout=2)
        torch.cuda.synchronize()


@pytest.mark.parametrize("revoked", [False, True])
def test_context_frontier_requires_gpu_completion_before_host_acceptance(revoked):
    from threading import Thread
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable

    old = ExecutionVersion("tenant", "scope", 0)
    guard = object.__new__(ExecutionDeviceGuard)
    guard._rank_consensus = None
    guard.device = DeviceExecutionAuthority(torch.cuda.current_device())
    guard.table = WorkerVersionTable("session")
    guard.table.bind_device_publisher(guard.device)
    guard.table.apply(VersionUpdate("session", 1, old))
    assert guard.device.check([old]) == [True]
    guard._native_kv_guards = True
    guard._early_exit_enabled = False
    guard._uses = {}
    guard._kv_frontiers = {}
    request = SimpleNamespace(
        py_execution_version=old,
        py_request_id=42,
        py_client_id=142,
        prepopulated_prompt_len=0,
        context_current_position=0,
        context_chunk_size=64,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_decoding_iter=0,
        is_finished=False,
    )

    def finish(reason):
        request.is_finished = True
        request.state = LlmRequestState.GENERATION_COMPLETE

    request.finish_by_reason = finish
    batch = ScheduledRequests()
    batch.context_requests_chunking = [request]
    stream = torch.cuda.Stream()
    marker = torch.zeros(1, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), guard.forward(batch):
        torch.cuda._sleep(1_000_000_000)
        marker.fill_(1)
    completion = guard._uses[id(request)].event
    assert not completion.query()
    wait = guard.device.wait_event

    def wait_with_revision(event):
        publisher = Thread(
            target=lambda: guard.table.apply(VersionUpdate("session", 2, replace(old, epoch=1)))
        )
        publisher.start()
        publisher.join(timeout=2)
        assert not publisher.is_alive(), "GPU wait prevented publication"
        wait(event)

    if revoked:
        guard.device.wait_event = wait_with_revision
    try:
        with guard.context_commit(batch) as accepted:
            assert completion.query()
            assert marker.item() == 1
            assert accepted.context_requests == ([] if revoked else [request])
            if not revoked:
                request.context_current_position = 64
        assert guard._kv_frontiers[id(request)] == (0 if revoked else 64)
    finally:
        stream.synchronize()


def test_unknown_completion_keeps_buffers_and_fails_future_guards() -> None:
    from types import SimpleNamespace

    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    assert authority.check([version]) == [True]
    buffer = torch.ones(1, device=authority.device)
    use = SimpleNamespace(tensors=[buffer], event=SimpleNamespace(query=lambda: False))
    authority._pending.append(use)
    with pytest.raises(TimeoutError, match="did not complete"):
        authority._wait(use, timeout=0)
    assert use in authority._pending
    assert use.tensors[0] is buffer
    with pytest.raises(RuntimeError, match="restart required"):
        authority.check([version])


@pytest.mark.parametrize("compact", [False, True])
def test_overlapping_tickets_own_tokens_and_metadata(compact) -> None:
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 0)
    authority.publish(version)
    candidates = torch.zeros((1, 4, 1), dtype=torch.int32, device=authority.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Warm this stream's allocator pool as well as the kernels before timing submission.
    with torch.cuda.stream(stream):
        warmed = [
            authority.enqueue_token_commit([version], [2], candidates, compact) for _ in range(2)
        ]
    for ticket in warmed:
        ticket.wait()
    versions, slots = [version], [2]
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        candidates.fill_(11)
        first = authority.enqueue_token_commit(versions, slots, candidates, compact)
        candidates.fill_(22)
        second = authority.enqueue_token_commit(versions, slots, candidates, compact)
    try:
        assert not first.event.query(), "token submission unexpectedly waited for the GPU"
        versions[0], slots[0] = replace(version, epoch=1), 0
        assert first.versions == second.versions == (version,)
        assert first.sequence_slots == second.sequence_slots == (2,)
        assert first.tokens_device.data_ptr() != second.tokens_device.data_ptr()
        assert first.tokens_host.data_ptr() != second.tokens_host.data_ptr()
        for ticket, expected in [(first, 11), (second, 22)]:
            commit = ticket.wait()
            assert commit.allowed == [True]
            assert commit.tokens_host.flatten().tolist() == (
                [expected] if compact else [0, 0, expected, 0]
            )
    finally:
        stream.synchronize()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_live_authority_after_graph_replay_and_compiled_output(compiled, captured) -> None:
    """The same candidate graph is replayed across revisions without capturing an epoch."""
    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    old = ExecutionVersion("tenant", "scope", (1 << 63) - 1)
    new = replace(old, epoch=1 << 63)
    other = replace(old, scope_id="other")
    authority.publish(old)
    authority.publish(other)
    logits = torch.eye(6, device=authority.device)[[1, 2, 3, 4]]
    candidates = torch.empty((1, 4, 1), device=authority.device, dtype=torch.int32)

    def sample(scores, destination):
        destination.copy_(scores.argmax(dim=-1).reshape(1, -1, 1))

    produce = torch.compile(sample, fullgraph=True) if compiled else sample
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with authority.submission(), torch.cuda.stream(warmup):
        for _ in range(3):
            produce(logits, candidates)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with authority.submission(), torch.cuda.graph(graph):
            produce(logits, candidates)

    def run(versions):
        with authority.submission():
            if graph is not None:
                graph.replay()
            else:
                produce(logits, candidates)
        return authority.enqueue_token_commit(versions, [2, 0, 3], candidates, compact=True).wait()

    before = run([old, other, None])
    assert before.allowed == [True, True, True]
    assert before.tokens_host.tolist() == [3, 1, 4]
    authority.publish(new)
    after = run([old, other, None])
    assert after.allowed == [False, True, True]
    assert after.tokens_device.flatten().tolist() == [1, 0, 0, 4]
    assert after.tokens_host.tolist() == [0, 1, 4]
    # Reusing the same sequence slot with the current version doesn't inherit rejection.
    assert run([new, other, None]).tokens_host.tolist() == [3, 1, 4]
    assert candidates.flatten().tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("logprobs", [False, True])
def test_native_sampler_consumes_overlap_batch_containing_canceled_request(logprobs) -> None:
    """Previous-result cancellation must not change current sampler row alignment."""
    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
    from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable

    version = ExecutionVersion("tenant", "scope", 7)
    guard = object.__new__(ExecutionDeviceGuard)
    guard._rank_consensus = None
    guard.device = DeviceExecutionAuthority(torch.cuda.current_device())
    guard.table = WorkerVersionTable("session")
    guard.table.bind_device_publisher(guard.device)
    guard.table.apply(VersionUpdate("session", 1, version))
    guard._uses = {}
    guard._kv_manager = None
    guard._v1_manager = None
    guard._kv_frontiers = {}
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=32,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=1,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=False,
        )
    )
    requests = [
        LlmRequest(
            request_id=slot,
            seq_slot=slot,
            input_tokens=[1, 2],
            max_new_tokens=8,
            end_id=7,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            return_log_probs=logprobs,
            execution_version=tag,
        )
        for slot, tag in [(2, version), (0, None)]
    ]
    logits = torch.full((2, 8), -100.0, device="cuda")
    logits[0, 3] = logits[1, 5] = 100.0
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = requests
    with guard.sampling(batch):
        previous = sampler.sample_async(batch, {"logits": logits}, [0, 1, 2])
        guard.commit_sample(batch, previous)
    # The current forward's batch is fixed before the previous result is consumed.
    for request in requests:
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
    batch = ScheduledRequests()
    batch.generation_requests = requests
    guard.table.apply(VersionUpdate("session", 2, replace(version, epoch=8)))
    with guard.token_commit(previous):
        sampler.update_requests(previous)
    assert requests[0].state == LlmRequestState.GENERATION_COMPLETE
    with guard.sampling(batch):
        current = sampler.sample_async(batch, {"logits": logits}, [0])
        guard.commit_sample(batch, current)
    with guard.token_commit(current):
        sampler.update_requests(current)
    assert requests[0].get_tokens(0) == [1, 2]
    assert requests[1].get_tokens(0) == [1, 2, 5, 5]
    for request in requests:
        guard.before_release(request)


def test_native_in_graph_sampling_is_guarded_after_replay() -> None:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
    from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.sampling_params import SamplingParams

    authority = DeviceExecutionAuthority(torch.cuda.current_device())
    version = ExecutionVersion("tenant", "scope", 7)
    authority.publish(version)
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=32,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=1,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=False,
        )
    )
    params = SamplingParams(temperature=0.8, top_k=2)
    batch = ScheduledRequests()
    batch.generation_requests = [
        LlmRequest(
            request_id=slot,
            seq_slot=slot,
            input_tokens=[1, 2],
            max_new_tokens=8,
            sampling_config=SamplingConfig(params._get_sampling_config()),
            is_streaming=False,
            execution_version=tag,
        )
        for slot, tag in [(2, version), (0, None)]
    ]
    sample_type = sampler.get_sample_type(batch.all_requests())
    assert sample_type is SampleType.FAST
    sampler.stage_in_graph_sampling(batch, sample_type)
    logits = torch.full((2, 8), -1000.0, device="cuda")
    logits[0, 3] = logits[1, 5] = 1000.0
    outputs = {"logits": logits}
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with authority.submission(), torch.cuda.stream(warmup):
        for _ in range(3):
            sampler.sample_in_graph(outputs, sample_type)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with authority.submission(), torch.cuda.graph(graph):
        sampler.sample_in_graph(outputs, sample_type)
    graph.replay()
    accepted = authority.commit_tokens([version, None], [2, 0], sampler.store.new_tokens, True)
    assert accepted.tokens_host.tolist() == [3, 5]
    authority.publish(replace(version, epoch=8))
    graph.replay()
    rejected = authority.commit_tokens([version, None], [2, 0], sampler.store.new_tokens, True)
    assert rejected.allowed == [False, True]
    assert rejected.tokens_host.tolist() == [0, 5]
    assert sampler.store.new_tokens[0, 2, 0].item() == 3


@pytest.mark.parametrize("cached", [False, True])
def test_native_v1_alignment_release_never_publishes_unexecuted_blocks(cached, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.bindings import DataType, LayerType
    from tensorrt_llm.bindings import ModelConfig as NativeModelConfig
    from tensorrt_llm.bindings.executor import SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType, LlmRequest
    from tensorrt_llm.execution_alignment import AlignmentInput
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = NativeModelConfig(
        vocab_size=64,
        num_layers=1,
        num_attention_layers=1,
        num_rnn_layers=0,
        num_heads=2,
        hidden_size=64,
        data_type=DataType.HALF,
    )
    config.layer_types = [LayerType.ATTENTION]
    config.set_num_kv_heads(2)
    manager = KVCacheManager(
        kv_cache_config=KvCacheConfig(max_tokens=256, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        tokens_per_block=16,
        max_seq_len=64,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        model_config=config,
        max_beam_width=1,
    )
    tokens = [1] * 33
    if cached:
        seed = LlmRequest(
            request_id=10,
            max_new_tokens=1,
            input_tokens=tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            cache_salt="shared",
        )
        manager.impl.add_sequence_batch([(10, len(tokens), 1)], [seed])
        manager.impl.refresh_blocks()
        manager._stream.synchronize()
        # Seed native ownership metadata; no model reads these synthetic values.
        manager.impl.store_blocks_for_reuse(10, seed, False)
    executor = SimpleNamespace(
        kv_cache_manager=manager,
        max_seq_len=64,
        _is_kv_manager_v2=False,
        executor_request_queue=Mock(),
    )
    executor.executor_request_queue.reserve_request_id.return_value = 11
    probe = manager.probe_prefix_match_length
    # Exercise an advisory hit that has disappeared before native allocation.
    monkeypatch.setattr(manager, "probe_prefix_match_length", lambda *args: 32)
    hold = KVAlignmentAdapter(executor).reserve(AlignmentInput(tuple(tokens), cache_salt="shared"))
    assert hold.num_tokens == (32 if cached else 0)
    if cached:
        seed.is_dummy_request = True
        manager.impl.remove_sequence(10, seed, False)
    hold.settle()
    hold.close()
    hold.close()
    assert probe(tokens, None, "shared") == (32 if cached else 0)
    assert probe(tokens, None, "other") == 0


def test_native_v2_isolated_handoff_preserves_prefix_while_source_cleanup_is_pending() -> None:
    """Keep native ownership across early RESUME and a host-controlled CUDA gate."""
    import faulthandler
    from threading import Event
    from types import SimpleNamespace

    import cuda.bindings.runtime as cudart

    from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm.bindings.internal import runtime as runtime_bindings
    from tensorrt_llm.execution_alignment import (
        AlignmentInput,
        AlignmentRequest,
        WorkerAlignmentState,
    )
    from tensorrt_llm.execution_authority import Revision
    from tensorrt_llm.execution_fence import FenceRequest, RequestDrainTracker
    from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        ReuseScope,
    )

    torch.empty(1, device="cuda")
    producer_stream, successor_stream = torch.cuda.Stream(), torch.cuda.Stream()
    entered, release_gate, timed_out = Event(), Event(), Event()
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(4 << 20), HostCacheTierConfig(4 << 20)],
            layers=[AttentionLayerConfig(0, [BufferConfig("key", 4096)])],
            enable_partial_reuse=True,
        )
    )
    old = ExecutionVersion("tenant", "native-isolated-handoff", 7)
    new = replace(old, epoch=8)
    tokens, prefix = list(range(12)), 8
    scope = ReuseScope(salt=11)
    owner = successor = state = None
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        owner = core.create_kv_cache(scope, expected_prompt_length=len(tokens))
        assert owner.resume(producer_stream.cuda_stream) and owner.resize(len(tokens))
        (desc,) = core.get_aggregated_pages(list(core.all_buffer_ids))

        def indices(cache):
            return list(cache.get_aggregated_page_indices(desc.layer_group_id))

        def fill(index, value, stream):
            (error,) = cudart.cudaMemsetAsync(
                int(desc.base) + index * desc.stride, value, desc.size, stream.cuda_stream
            )
            assert int(error) == 0

        def read(cache_indices, output, stream):
            for ordinal, index in enumerate(cache_indices):
                (error,) = cudart.cudaMemcpyAsync(
                    output.data_ptr() + ordinal * desc.size,
                    int(desc.base) + index * desc.stride,
                    desc.size,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream.cuda_stream,
                )
                assert int(error) == 0

        for index in indices(owner):
            fill(index, 37, producer_stream)
        producer_stream.synchronize()
        owner.commit(tokens[:prefix])
        original = indices(owner)
        assert len(original) == 3 and owner.num_committed_tokens == prefix
        source_output = torch.empty((2, desc.size), dtype=torch.uint8, device="cuda")
        successor_output = torch.empty((3, desc.size), dtype=torch.uint8, device="cuda")

        manager = object.__new__(KVCacheManagerV2)
        manager.impl, manager._stream = core, successor_stream
        manager.enable_block_reuse = manager._can_publish_block_reuse = True
        manager.tokens_per_block, manager.max_seq_len = 4, 32
        manager.kv_cache_map = {1: owner}
        manager._derive_reuse_salt = lambda salt: 11
        adapter = KVAlignmentAdapter(
            SimpleNamespace(kv_cache_manager=manager, max_seq_len=32, _is_kv_manager_v2=True),
            keep_resident=True,
        )
        guard = object.__new__(ExecutionDeviceGuard)
        guard.table = table = WorkerVersionTable("session")
        guard.device = DeviceExecutionAuthority(torch.cuda.current_device())
        table.bind_device_publisher(guard.device)
        table.apply(VersionUpdate("session", 1, old))
        assert guard.device.check([old]) == [True]
        guard._kv_manager, guard._v1_manager = manager, None
        guard._rank_consensus, guard._async_context = None, True
        guard._alignment_adapter = adapter
        guard._uses, guard._kv_frontiers = {}, {}
        guard._context_owners, guard._context_chunks = {}, {}
        guard._isolated_requests, guard._pending_releases = {}, {}
        guard._release_error = None
        table.drain_tracker = RequestDrainTracker()
        table.drain_tracker.enqueue(old, lambda: 1)
        table.alignment_state = state = WorkerAlignmentState(
            adapter.reserve,
            lambda: None,
            prepare=guard._prepare_prefix,
            reserve_isolated=guard._reserve_isolated_prefix,
        )
        request = SimpleNamespace(
            py_request_id=1,
            py_execution_version=old,
            lora_task_id=None,
            is_dummy_request=False,
            is_finished=False,
            state=LlmRequestState.GENERATION_IN_PROGRESS,
            py_decoding_iter=0,
            context_current_position=prefix,
            context_remaining_length=0,
            get_num_tokens=lambda beam: prefix + 1,
        )

        def finish(reason):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.is_finished = True

        request.finish_by_reason = finish
        guard._kv_frontiers[id(request)] = prefix
        torch.cuda.synchronize()

        def gate():
            # CUDA APIs must not be called inside a CUDA host callback.
            entered.set()
            if not release_gate.wait(30):
                timed_out.set()

        with (
            torch.cuda.stream(producer_stream),
            guard.sampling(SimpleNamespace(all_requests=lambda: [request])),
        ):
            runtime_bindings.launch_hostfunc(producer_stream.cuda_stream, True, False, gate)
            read(original[:2], source_output, producer_stream)
            fill(original[2], 71, producer_stream)
        completion = guard._uses[id(request)].event
        assert entered.wait(30), "CUDA host gate did not start"
        assert not completion.query()

        revised = AlignmentInput(tuple(tokens[:prefix] + [99, 98, 97, 96]), cache_salt="scope")
        alignment = AlignmentRequest("session", Revision("edit", old, new), revised)
        table.apply(VersionUpdate("session", 2, new, alignment))
        guard.process_alignment()
        receipt = table.poll_alignment(alignment)
        assert receipt is not None and receipt.reusable_prefix_length == prefix
        assert not completion.query(), "early ALIGN waited for obsolete GPU work"
        assert not timed_out.is_set()
        assert not adapter.has_residency()
        assert owner.is_active and owner.num_committed_tokens == prefix
        fence = FenceRequest("session", alignment.revision)
        assert table.poll_fence(fence) is None
        table.apply(VersionUpdate("session", 3, new, alignment, resume=receipt))
        assert state.claim(new, 2, revised.input_token_ids, revised.cache_salt, None) == prefix

        successor = core.create_kv_cache(
            scope, list(revised.input_token_ids[:prefix]), expected_prompt_length=12
        )
        assert successor.num_committed_tokens == prefix
        assert successor.resume(successor_stream.cuda_stream) and successor.resize(12)
        inherited = indices(successor)
        assert inherited[:2] == original[:2] and inherited[2] != original[2]
        fill(inherited[2], 93, successor_stream)
        read(inherited, successor_output, successor_stream)
        state.confirm_attachment(new, 2, prefix)
        assert adapter._reserved_v2_blocks == 0
        assert not completion.query(), "successor attachment drained the old owner"

        released = []

        def reclaim(retired):
            guard.before_release(retired)
            manager.kv_cache_map.pop(retired.py_request_id).close()
            guard.terminal(retired.py_request_id, retired.py_execution_version)
            released.append(retired)

        assert guard.defer_release(request)
        guard.process_releases(reclaim)
        assert not released and guard.pending_releases == 1
        assert table.poll_fence(fence) is None
        release_gate.set()
        producer_stream.synchronize()
        successor_stream.synchronize()
        assert not timed_out.is_set() and completion.query()
        assert torch.all(source_output == 37).item()
        assert torch.all(successor_output[:2] == 37).item()
        assert torch.all(successor_output[2] == 93).item()
        guard.process_releases(reclaim, retained=(request,))
        assert not released and table.poll_fence(fence) is None
        guard.process_releases(reclaim)
        owner = None
        assert released == [request] and guard.pending_releases == 0
        assert table.poll_fence(fence) is not None
        assert successor.is_active and indices(successor) == inherited
    finally:
        release_gate.set()
        producer_stream.synchronize()
        successor_stream.synchronize()
        if state is not None:
            state.close()
            state.process()
        for cache in (successor, owner):
            if cache is not None:
                cache.close()
        core.shutdown()
        faulthandler.cancel_dump_traceback_later()


def test_breakable_graph_revocation_preserves_unsubmitted_segments() -> None:
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import BreakableCUDAGraph
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        early_exit_scope,
    )

    values = torch.zeros(3, device="cuda")
    views = list(values.unbind())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = BreakableCUDAGraph()
    revoked = False

    def revoke():
        nonlocal revoked
        revoked = True

    with torch.cuda.stream(stream):
        for value in views:
            value.add_(1)
        stream.synchronize()
        for value in views:
            segment = torch.cuda.CUDAGraph()
            with torch.cuda.graph(segment, stream=stream):
                value.add_(1)
            graph._segments.append(segment)
        graph._break_functions = [revoke, lambda: None]
        values.zero_()
        with early_exit_scope(lambda: revoked):
            with pytest.raises(ExecutionEarlyExit):
                graph.replay()
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert values.tolist() == [1, 0, 0]
    with torch.cuda.stream(stream):
        graph.replay()
        done.record()
    done.synchronize()
    assert values.tolist() == [2, 1, 1]
