# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native execution guard wiring with a controlled device verdict."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from threading import Condition, Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.execution_device import DeviceExecutionAuthority
from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.execution_consensus import ExecutionStepConsensus
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
NEXT = replace(VERSION, epoch=8)


def test_short_device_wait_yields_to_publication_without_millisecond_delay(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority._lock = Lock()
    authority._failure = None
    clock = [0.0]
    sleeps = []

    def yield_to_publisher(delay):
        assert authority._lock.acquire(blocking=False)
        authority._lock.release()
        sleeps.append(delay)
        clock[0] += max(delay, 0.00005)

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", yield_to_publisher)
    event = SimpleNamespace(query=lambda: clock[0] >= 0.0001)

    authority.wait_event(event)

    assert event.query()
    assert 0 < clock[0] < 0.001
    assert sleeps and max(sleeps) < 0.001
    assert authority._failure is None


def test_long_device_wait_backs_off_and_keeps_ownership_on_timeout(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority._lock = Lock()
    authority._failure = None
    clock = [0.0]
    sleeps = []

    def advance_time(delay):
        sleeps.append(delay)
        clock[0] += max(delay, 0.00005)

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", advance_time)
    buffer = object()
    use = module._PendingUse([buffer], SimpleNamespace(query=lambda: False))
    authority._pending = [use]

    with pytest.raises(TimeoutError, match="did not complete"):
        authority._wait(use, timeout=0.003)

    assert any(sleeps) and len(sleeps) < 32
    assert authority._pending == [use]
    assert use.tensors == [buffer]
    assert isinstance(authority._failure, TimeoutError)
    with pytest.raises(RuntimeError, match="restart required"):
        authority.wait_event(SimpleNamespace(query=lambda: True))


@pytest.mark.parametrize("wait_kind", ["model", "token_commit"])
def test_slow_model_completion_keeps_ownership_and_publisher_access(monkeypatch, wait_kind):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority._lock = Lock()
    authority._failure = None
    clock = [0.0]
    buffer = object()
    event = SimpleNamespace(query=lambda: clock[0] >= 6.0)
    use = module._PendingUse([buffer], event)
    authority._pending = [use]
    reads = []

    def read_allowed():
        assert event.query()
        reads.append(clock[0])
        return [True, False]

    def advance_model(delay):
        assert authority._lock.acquire(blocking=False)
        authority._lock.release()
        assert authority._pending == [use]
        assert use.tensors == [buffer]
        assert not reads
        clock[0] += 0.5

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", advance_model)
    if wait_kind == "model":
        authority.wait_event(event)
    else:
        ticket = module.DeviceTokenCommitTicket(
            authority,
            (VERSION, NEXT),
            (0, 1),
            object(),
            object(),
            SimpleNamespace(tolist=read_allowed),
            use,
        )
        result = ticket.wait()
        assert result.allowed == [True, False]
        assert result.tokens_device is ticket.tokens_device
        assert result.tokens_host is ticket.tokens_host
        assert reads == [6.0]
    assert event.query()
    assert clock[0] == 6.0
    assert authority._failure is None


@pytest.mark.parametrize("failure_source", ["publisher", "event"])
def test_slow_device_wait_still_fails_closed_on_failure(monkeypatch, failure_source):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority._lock = Lock()
    authority._failure = None
    clock = [0.0]
    buffer = object()
    error = RuntimeError("injected device failure")

    def query():
        if failure_source == "event" and clock[0] >= 6.0:
            raise error
        return False

    use = module._PendingUse([buffer], SimpleNamespace(query=query))
    authority._pending = [use]

    def advance_model(delay):
        assert authority._lock.acquire(blocking=False)
        try:
            clock[0] += 0.5
            if failure_source == "publisher" and clock[0] >= 6.0:
                authority._failure = error
        finally:
            authority._lock.release()

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", advance_model)
    message = "restart required" if failure_source == "publisher" else "injected device failure"
    with pytest.raises(RuntimeError, match=message):
        authority._wait(use, timeout=None)
    assert clock[0] == 6.0
    assert authority._pending == [use]
    assert use.tensors == [buffer]
    assert authority._failure is not None
    with pytest.raises(RuntimeError, match="restart required"):
        authority.wait_event(SimpleNamespace(query=lambda: True))


@pytest.fixture
def attention_metadata(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority.device = "cuda:0"
    authority._lock = Lock()
    authority._failure = None
    authority._ready = object()
    authority._published = {("tenant", "scope"): (VERSION.epoch, object())}
    authority._slots = {("tenant", "scope"): 0}
    authority._start_use = Mock(side_effect=lambda tensors: module._PendingUse(tensors))
    authority._record = Mock()
    stream = SimpleNamespace(cuda_stream=17, wait_event=Mock())
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(module.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(module.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(module, "prefer_pinned", lambda: False)
    uploads = Mock(side_effect=lambda values, **kwargs: tuple(values))
    monkeypatch.setattr(module.torch, "tensor", uploads)
    monkeypatch.setattr(module, "_row_storage", lambda tensor: (id(tensor),))
    slots = SimpleNamespace(numel=lambda: 4, dtype=module.torch.int32, copy_=Mock())
    expected = SimpleNamespace(dtype=module.torch.int64, copy_=Mock())
    rows = module.DeviceAttentionRows(authority, slots, expected, object(), stream.cuda_stream)
    return authority, rows, stream, uploads


def test_stable_attention_metadata_joins_publication_without_another_upload(attention_metadata):
    authority, rows, stream, uploads = attention_metadata
    authority.prepare_attention_rows(rows, [VERSION, None])
    rows.slots.copy_.assert_called_once_with((0, -1, -1, -1), non_blocking=True)
    publication = object()
    authority._published[("tenant", "scope")] = (NEXT.epoch, publication)
    authority.prepare_attention_rows(rows, [VERSION, None])
    assert uploads.call_count == 2
    assert authority._record.call_count == 2
    stream.wait_event.assert_called_with(publication)
    # Expected identities stay stale so the kernel can reject the old request.
    assert rows._layout.key == ((0, -1, -1, -1), (VERSION.epoch, 0, 0, 0))
    authority.prepare_attention_rows(rows, [None, NEXT])
    rows.slots.copy_.assert_called_with((-1, 0, -1, -1), non_blocking=True)
    rows.expected.copy_.assert_called_with((0, NEXT.epoch, 0, 0), non_blocking=True)
    authority.prepare_attention_rows(rows, [None])
    rows.slots.copy_.assert_called_with((-1, -1, -1, -1), non_blocking=True)
    rows.expected.copy_.assert_called_with((0, 0, 0, 0), non_blocking=True)
    assert uploads.call_count == 6


@pytest.mark.parametrize("invalid_submission", ["stream", "capture"])
def test_cached_attention_metadata_still_checks_submission_context(
    attention_metadata, monkeypatch, invalid_submission
):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority, rows, stream, uploads = attention_metadata
    authority.prepare_attention_rows(rows, [VERSION])
    if invalid_submission == "stream":
        stream.cuda_stream += 1
        message = "cannot move between"
    else:
        monkeypatch.setattr(module.torch.cuda, "is_current_stream_capturing", lambda: True)
        message = "outside capture"
    with pytest.raises(RuntimeError, match=message):
        authority.prepare_attention_rows(rows, [VERSION])
    assert uploads.call_count == 2


def test_stable_token_metadata_keeps_publication_waits_and_inflight_ownership(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority.device = "cuda:0"
    authority._token_commit_metadata = None
    authority._ready = object()
    publication = object()
    authority._published = {("tenant", "scope"): (VERSION.epoch, publication)}
    authority._slots = {("tenant", "scope"): 0}
    stream = SimpleNamespace(cuda_stream=17, wait_event=Mock())
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda device: stream)
    uploads = []

    def upload(values, dtype, use):
        tensor = Mock()
        uploads.append((values, dtype))
        use.tensors.append(tensor)
        return tensor

    authority._upload = upload
    monkeypatch.setattr(module.torch.cuda, "Event", lambda: Mock(query=Mock(return_value=True)))
    first = module._PendingUse([])
    original = authority._token_commit_inputs([VERSION, None], [2, 0], first)
    assert len(uploads) == 3
    next_publication = object()
    authority._published[("tenant", "scope")] = (NEXT.epoch, next_publication)
    second = module._PendingUse([])
    reused = authority._token_commit_inputs([VERSION, None], [2, 0], second)
    assert all(before is after for before, after in zip(original, reused))
    assert len(uploads) == 3
    stream.wait_event.assert_any_call(next_publication)
    assert stream.wait_event.call_args.args[0] is authority._token_commit_metadata.ready
    assert all(tensor in second.tensors for tensor in original)
    # A successor has a new expected epoch even when it recycles the same slots.
    successor = authority._token_commit_inputs([NEXT, None], [2, 0], module._PendingUse([]))
    assert successor is not original
    assert len(uploads) == 6
    assert all(tensor in first.tensors and tensor in second.tensors for tensor in original)


@pytest.mark.parametrize("change", ["slots", "row_order", "stream", "tenant", "scope"])
def test_token_metadata_does_not_alias_other_rows_or_streams(monkeypatch, change):
    from tensorrt_llm._torch.pyexecutor import execution_device as module

    authority = object.__new__(DeviceExecutionAuthority)
    authority.device = "cuda:0"
    authority._token_commit_metadata = None
    authority._input_values = Mock(return_value=([0, -1], [VERSION.epoch, 0]))
    authority._upload = Mock(side_effect=lambda values, dtype, use: Mock())
    stream = SimpleNamespace(cuda_stream=17, wait_event=Mock())
    monkeypatch.setattr(module.torch.cuda, "Event", lambda: Mock(query=Mock(return_value=False)))
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda device: stream)
    versions, slots = [VERSION, None], [2, 0]
    before = authority._token_commit_inputs(versions, slots, module._PendingUse([]))
    if change == "slots":
        slots.reverse()
    elif change == "row_order":
        versions.reverse()
    elif change == "stream":
        stream.cuda_stream = 18
    elif change == "tenant":
        versions[0] = replace(VERSION, tenant_id="other")
    else:
        versions[0] = replace(VERSION, scope_id="other")
    if change in ("tenant", "scope"):
        authority._input_values.return_value = ([1, -1], [VERSION.epoch, 0])
    after = authority._token_commit_inputs(versions, slots, module._PendingUse([]))
    assert after is not before
    assert authority._upload.call_count == 6


def _request(version=VERSION):
    request = SimpleNamespace(
        py_request_id=42,
        py_client_id=142,
        prepopulated_prompt_len=0,
        context_current_position=0,
        context_chunk_size=64,
        context_remaining_length=64,
        is_finished=False,
        is_context_init_state=False,
        py_execution_version=version,
        py_seq_slot=2,
        py_decoding_iter=3,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_beam_width=1,
        py_draft_tokens=None,
        py_multimodal_data=None,
        py_disaggregated_params=None,
        py_logits_post_processors=None,
        num_tokens=64,
    )
    request.get_num_tokens = lambda beam: request.num_tokens
    request.get_token = lambda beam, position: 7

    def finish(reason):
        assert reason == FinishReason.CANCELLED
        request.state = LlmRequestState.GENERATION_COMPLETE
        request.is_finished = True

    request.finish_by_reason = Mock(side_effect=finish)
    return request


@pytest.fixture
def guard(monkeypatch):
    result = object.__new__(ExecutionDeviceGuard)
    result._rank_consensus = None
    result.table = WorkerVersionTable("session")
    result.table.apply(VersionUpdate("session", 1, VERSION))
    result.device = Mock()
    result.device.submission.side_effect = lambda *, capture_possible=True: nullcontext()
    result._uses = {}
    result._kv_frontiers = {}
    result._early_exit_enabled = True
    result._native_kv_guards = False
    result._v1_manager = None
    result._kv_manager = Mock()
    result._kv_manager.needs_execution_prefix_commit.return_value = False
    monkeypatch.setattr("torch.cuda.Event", Mock)
    return result


def _batch(*requests, context=False):
    batch = ScheduledRequests()
    if context:
        batch.context_requests_last_chunk = list(requests)
    else:
        batch.generation_requests = list(requests)
    return batch


def _async_context_fixture(guard, *, prompt_len=12, chunk_size=4):
    request = _request()
    request.prompt_len = prompt_len
    request.context_remaining_length = prompt_len
    request.context_chunk_size = chunk_size
    request.is_dummy_request = False
    request.num_tokens = prompt_len
    cache = SimpleNamespace(is_active=True, is_commit_allowed=True, num_committed_tokens=0)
    guard._async_context = guard._native_kv_guards = True
    guard._context_owners = {}
    guard._context_chunks = {}
    guard._kv_manager.kv_cache_map = {request.py_request_id: cache}
    receipts = []

    def enqueue(versions):
        ticket = Mock(versions=tuple(versions))
        ticket.query.return_value = False
        ticket.wait.return_value = [True] * len(versions)
        receipts.append(ticket)
        return ticket

    guard.device.enqueue_check.side_effect = enqueue
    return request, cache, receipts


def _advance_async_context(guard, request):
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    with guard.context_commit(batch) as accepted:
        assert accepted.context_requests == [request]
        request.context_current_position += request.context_chunk_size
        request.context_remaining_length = request.prompt_len - request.context_current_position
    return batch


@pytest.mark.parametrize(
    "exit_kind", ["success", "early_exit", "partial_failure", "receipt_failure"]
)
def test_async_context_guard_retains_readers_and_only_receipts_full_forwards(guard, exit_kind):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import ExecutionEarlyExit

    request, _, receipts = _async_context_fixture(guard)
    if exit_kind == "receipt_failure":
        guard.device.enqueue_check.side_effect = RuntimeError("receipt submission failed")
    error = None
    try:
        with guard.forward(_batch(request, context=True)):
            if exit_kind == "early_exit":
                raise ExecutionEarlyExit("layer", 1, 4)
            if exit_kind == "partial_failure":
                raise ValueError("partial forward")
    except (ExecutionEarlyExit, ValueError, RuntimeError) as caught:
        error = caught
    assert (error is None) is (exit_kind == "success")
    assert len(receipts) == int(exit_kind == "success")
    assert len(guard._context_chunks.get(id(request), [])) == int(exit_kind == "success")
    assert guard._uses[id(request)].event is not None
    assert guard._kv_frontiers[id(request)] == 0
    guard.device.wait_event.assert_not_called()


def test_async_context_guard_advances_cpu_cursor_without_waiting_or_publishing(guard):
    request, cache, receipts = _async_context_fixture(guard)
    batch = _advance_async_context(guard, request)
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.kv_cache_manager = guard._kv_manager
    executor.enable_joint_kv_cache_reuse = False
    executor._update_v2_context_resources(batch)
    accepted = guard._kv_manager.update_context_resources.call_args.args[0]
    assert accepted.context_requests == []
    assert batch.context_requests == [request]
    assert request.context_current_position == 4
    assert guard._kv_frontiers[id(request)] == cache.num_committed_tokens == 0
    receipts[0].wait.assert_not_called()
    guard.device.wait_event.assert_not_called()
    guard._kv_manager.preserve_execution_context.assert_not_called()


def test_async_context_guard_keeps_completed_receipt_until_cursor_update(guard):
    request, _, receipts = _async_context_fixture(guard)
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    receipts[0].query.return_value = True
    guard._settle_context_receipts(request)
    assert len(guard._context_chunks[id(request)]) == 1
    assert guard._kv_frontiers[id(request)] == 0
    with guard.context_commit(batch):
        request.context_current_position = 4
    guard._settle_context_receipts(request)
    assert guard._kv_frontiers[id(request)] == 4
    assert id(request) not in guard._context_chunks


def test_async_context_guard_settles_chunks_in_order_without_later_reader_rebase(guard):
    request, cache, receipts = _async_context_fixture(guard)
    _advance_async_context(guard, request)
    batch = _advance_async_context(guard, request)
    receipts[1].query.return_value = True
    guard._settle_context_receipts(request)
    assert guard._kv_frontiers[id(request)] == 0
    receipts[1].wait.assert_not_called()
    receipts[0].query.return_value = True
    guard._uses[id(request)].event.query.return_value = False
    with guard.context_commit(batch, publish=True) as accepted:
        assert accepted.context_requests == []
    assert guard._kv_frontiers[id(request)] == 8
    assert id(request) not in guard._context_chunks
    guard._kv_manager.preserve_execution_context.assert_not_called()
    guard._uses[id(request)].event.query.return_value = True
    guard._publish_context_frontier(request)
    guard._kv_manager.preserve_execution_context.assert_called_once_with(request, 8, cache)


@pytest.mark.parametrize("fault", ["gpu_reject", "gap", "identity", "query", "copy"])
def test_async_context_guard_rejects_unproven_or_failed_receipts(guard, fault):
    request, _, receipts = _async_context_fixture(guard)
    _advance_async_context(guard, request)
    _advance_async_context(guard, request)
    for ticket in receipts:
        ticket.query.return_value = True
    if fault == "gpu_reject":
        receipts[0].wait.return_value = [False]
        guard._settle_context_receipts(request)
        assert request.is_finished
        assert guard._context_owners[id(request)].rejected
        assert id(request) not in guard._context_chunks
    else:
        if fault == "gap":
            guard._context_chunks[id(request)][0].start = 1
        elif fault == "identity":
            receipts[0].versions = (NEXT,)
        elif fault == "query":
            receipts[0].query.side_effect = RuntimeError("query failed")
        else:
            receipts[0].wait.side_effect = RuntimeError("copy failed")
        with pytest.raises(RuntimeError):
            guard._settle_context_receipts(request)
        assert len(guard._context_chunks[id(request)]) == 2
    assert guard._kv_frontiers[id(request)] == 0
    guard._kv_manager.preserve_execution_context.assert_not_called()


@pytest.mark.parametrize("changed", ["allocation", "request_id", "version"])
def test_async_context_guard_never_retargets_an_old_receipt(guard, changed):
    request, _, receipts = _async_context_fixture(guard)
    _advance_async_context(guard, request)
    old_owner = guard._context_owners[id(request)]
    if changed == "request_id":
        request.py_request_id += 1
    elif changed == "version":
        request.py_execution_version = NEXT
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    replacement = SimpleNamespace(is_active=True, is_commit_allowed=True, num_committed_tokens=0)
    guard._kv_manager.kv_cache_map[request.py_request_id] = replacement
    request.context_current_position = request.prepopulated_prompt_len = 0
    _advance_async_context(guard, request)
    assert guard._context_owners[id(request)] is not old_owner
    receipts[0].query.return_value = True
    guard._settle_context_receipts(request)
    assert guard._kv_frontiers[id(request)] == 0
    assert len(guard._context_chunks[id(request)]) == 1
    guard._publish_context_frontier(request)
    guard._kv_manager.preserve_execution_context.assert_not_called()


@pytest.mark.parametrize("revoke_at", ["before_cursor", "after_cursor"])
def test_async_context_guard_revision_retains_only_accepted_completed_progress(guard, revoke_at):
    request, cache, receipts = _async_context_fixture(guard, chunk_size=7)
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    if revoke_at == "before_cursor":
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.context_commit(batch) as accepted:
        if accepted.context_requests:
            request.context_current_position = 7
    if revoke_at == "after_cursor":
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    receipts[0].query.return_value = True
    guard.before_release(request)
    if revoke_at == "after_cursor":
        guard._kv_manager.preserve_execution_context.assert_called_once_with(request, 7, cache)
    else:
        guard._kv_manager.preserve_execution_context.assert_not_called()
    assert not guard._context_chunks and not guard._context_owners
    assert id(request) not in guard._uses and id(request) not in guard._kv_frontiers


@pytest.mark.parametrize("failure", ["reader", "receipt", "publication"])
def test_async_context_guard_failed_release_retains_owner_and_receipts(guard, failure):
    request, _, receipts = _async_context_fixture(guard)
    _advance_async_context(guard, request)
    if failure == "reader":
        guard.device.wait_event.side_effect = RuntimeError("reader incomplete")
    elif failure == "receipt":
        receipts[0].wait.side_effect = RuntimeError("receipt incomplete")
    else:
        guard._kv_manager.preserve_execution_context.side_effect = RuntimeError("commit failed")
    with pytest.raises(RuntimeError):
        guard.before_release(request)
    assert id(request) in guard._uses and id(request) in guard._context_owners
    assert id(request) in guard._kv_frontiers
    assert (id(request) in guard._context_chunks) is (failure != "publication")
    guard._kv_manager.stop_request_committing.assert_not_called()


@pytest.mark.parametrize("proof", ["ready", "pending", "rejected", "released"])
def test_async_context_guard_requires_context_proof_before_accepting_first_token(guard, proof):
    request, _, receipts = _async_context_fixture(guard, prompt_len=4, chunk_size=4)
    _advance_async_context(guard, request)
    if proof == "released":
        guard.before_release(request)
        guard._kv_manager.kv_cache_map.clear()
    else:
        receipts[0].query.return_value = proof != "pending"
        receipts[0].wait.return_value = [proof != "rejected"]
    if proof == "pending":
        with pytest.raises(RuntimeError, match="proven completed context"):
            with guard.token_commit(_state(request, allowed=[True])):
                pytest.fail("must reject before token mutation")
        assert request.num_tokens == 4
    else:
        with guard.token_commit(_state(request, allowed=[True])):
            if not request.is_finished:
                request.num_tokens += 1
        if proof == "ready":
            assert guard._kv_frontiers[id(request)] == 4
        elif proof == "released":
            assert id(request) not in guard._kv_frontiers
        else:
            assert request.is_finished and request.num_tokens == 4


@pytest.mark.parametrize("reason", ["disabled", "untagged", "dummy", "multimodal"])
def test_async_context_guard_keeps_unsupported_request_fallbacks(guard, reason):
    request, _, receipts = _async_context_fixture(guard)
    if reason == "disabled":
        guard._async_context = False
    elif reason == "untagged":
        request.py_execution_version = None
    elif reason == "dummy":
        request.is_dummy_request = True
    else:
        request.py_multimodal_data = {"input": object()}
    _advance_async_context(guard, request)
    assert not receipts
    assert guard.device.wait_event.call_count == int(reason != "untagged")


def test_async_context_guard_cleanup_never_publishes_or_freezes_a_replacement(guard):
    request, _, receipts = _async_context_fixture(guard)
    _advance_async_context(guard, request)
    receipts[0].query.return_value = True
    guard._settle_context_receipts(request)
    assert guard._kv_frontiers[id(request)] == 4
    replacement = SimpleNamespace(is_active=True, is_commit_allowed=True, num_committed_tokens=0)
    guard._kv_manager.kv_cache_map[request.py_request_id] = replacement
    guard.before_release(request)
    guard._kv_manager.preserve_execution_context.assert_not_called()
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()
    assert id(request) not in guard._uses


def test_async_context_guard_multimodal_metadata_cleanup_cannot_change_publication_mode(guard):
    request, _, receipts = _async_context_fixture(guard)
    request.py_multimodal_data = {"input": object()}
    batch = _advance_async_context(guard, request)
    request.py_multimodal_data = None
    with guard.context_commit(batch, publish=True) as accepted:
        assert accepted.context_requests == [request]
    assert not receipts


def test_async_context_guard_executor_advances_last_chunk_before_device_completion(guard):
    request, _, receipts = _async_context_fixture(guard, prompt_len=4, chunk_size=4)
    request.will_complete_next_iteration = lambda: False

    def move():
        request.context_current_position = 4
        request.context_remaining_length = 0

    request.move_to_next_context_chunk = move
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.dist = SimpleNamespace(cp_config={})
    executor.active_requests = []
    executor.disable_overlap_scheduler = False
    executor.kv_cache_manager = guard._kv_manager
    executor.enable_joint_kv_cache_reuse = False
    executor._update_request_states(batch)
    executor._update_v2_context_resources(batch)
    assert request.state == LlmRequestState.GENERATION_IN_PROGRESS
    assert request.context_current_position == 4
    assert request.py_last_context_chunk == (0, 4)
    assert guard._kv_frontiers[id(request)] == 0
    receipts[0].wait.assert_not_called()
    guard.device.wait_event.assert_not_called()


@pytest.mark.parametrize("fallback", [None, "native", "overlap", "joint", "layout"])
def test_async_context_guard_constructor_selects_only_supported_executor(fallback, monkeypatch):
    executor = _executor()
    executor.device_id = 0
    executor._is_kv_manager_v2 = True
    executor.disable_overlap_scheduler = fallback == "overlap"
    executor.enable_joint_kv_cache_reuse = fallback == "joint"
    executor.kv_cache_manager.supports_execution_context_overlap = lambda: fallback != "layout"
    executor.executor_request_queue = Mock()
    executor.model_engine.execution_attention = (
        None if fallback == "native" else SimpleNamespace(authority=Mock())
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.execution_guard.DeviceExecutionAuthority", Mock()
    )
    guard = ExecutionDeviceGuard(executor, WorkerVersionTable("session"))
    assert guard._async_context is (fallback is None)


@pytest.mark.parametrize("failed", [False, True])
def test_async_context_guard_fence_requires_receipt_and_resource_drain(guard, failed):
    request, _, receipts = _async_context_fixture(guard)
    tracker = guard.table.drain_tracker = RequestDrainTracker()
    tracker.enqueue(VERSION, lambda: request.py_request_id)
    _advance_async_context(guard, request)
    if failed:
        receipts[0].wait.side_effect = RuntimeError("receipt completion failed")
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_coordinator = Mock()
    executor.gather_all_responses = True
    executor.result_wait_queues = {request.py_request_id: object()}

    def free(req):
        assert req is request
        assert not tracker.drained(VERSION)

    executor.resource_manager.free_resources.side_effect = free
    if failed:
        with pytest.raises(RuntimeError, match="receipt completion failed"):
            executor._do_terminate_request(request)
        executor.resource_manager.free_resources.assert_not_called()
        assert request.py_request_id in executor.result_wait_queues
    else:
        executor._do_terminate_request(request)
        executor.resource_manager.free_resources.assert_called_once_with(request)
    assert tracker.drained(VERSION) is (not failed)


def test_async_context_guard_alignment_drains_then_publishes_accepted_partial_context(guard):
    request, cache, receipts = _async_context_fixture(guard, chunk_size=7)
    _advance_async_context(guard, request)
    order = []
    guard.device.wait_event.side_effect = lambda event: order.append("reader")
    receipts[0].wait.side_effect = lambda: order.append("receipt") or [True]
    guard._kv_manager.preserve_execution_context.side_effect = lambda *args: order.append("publish")
    guard._prepare_prefix(SimpleNamespace(revision=SimpleNamespace(expected_version=VERSION)))
    assert order == ["reader", "receipt", "publish"]
    guard._kv_manager.preserve_execution_context.assert_called_once_with(request, 7, cache)
    guard._kv_manager.stop_request_committing.assert_called_once_with(request)


@pytest.mark.parametrize("failed_receipt", [False, True])
def test_async_context_guard_suspension_settles_receipt_before_preserving_prefix(
    guard, failed_receipt
):
    request, cache, receipts = _async_context_fixture(guard, chunk_size=7)
    _advance_async_context(guard, request)
    use = guard._uses[id(request)]
    order = []
    guard.device.wait_event.side_effect = lambda event: order.append("reader")

    def receipt():
        order.append("receipt")
        if failed_receipt:
            raise RuntimeError("receipt completion failed")
        return [True]

    receipts[0].wait.side_effect = receipt
    guard._kv_manager.preserve_execution_context.side_effect = lambda *args: order.append("publish")
    if failed_receipt:
        with pytest.raises(RuntimeError, match="receipt completion failed"):
            guard.before_suspend(request)
        assert order == ["reader", "receipt"]
        assert guard._kv_frontiers[id(request)] == 0
        assert guard._context_chunks[id(request)]
        guard._kv_manager.preserve_execution_context.assert_not_called()
    else:
        guard.before_suspend(request)
        assert order == ["reader", "receipt", "publish"]
        assert guard._kv_frontiers[id(request)] == 7
        assert id(request) not in guard._context_chunks
        guard._kv_manager.preserve_execution_context.assert_called_once_with(request, 7, cache)
    assert guard._uses[id(request)] is use
    assert guard._kv_manager.kv_cache_map[request.py_request_id] is cache
    assert cache.is_active
    guard._kv_manager.stop_request_committing.assert_not_called()


@pytest.mark.parametrize(
    "binding_failure,has_transceiver",
    [
        (None, False),
        (None, True),
        ("manager", False),
        ("manager", True),
        ("session", True),
    ],
)
def test_authority_installation_binds_transfer_drain_or_fails_closed(
    monkeypatch,
    binding_failure,
    has_transceiver,
):
    table = WorkerVersionTable("session")
    tracker = RequestDrainTracker()
    guard = SimpleNamespace(
        table=table,
        _rank_consensus=None,
        execution_failed=Mock(side_effect=tracker.fail),
        before_suspend=Mock(),
    )

    def make_guard(executor, installed_table, consensus):
        assert installed_table is table and consensus is None
        table.drain_tracker = tracker
        return guard

    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.execution_guard.ExecutionDeviceGuard", make_guard
    )
    transfers = Mock()
    transceiver = Mock() if has_transceiver else None
    executor = SimpleNamespace(
        _execution_guard=None,
        async_transfer_manager=transfers,
        kv_cache_transceiver=transceiver,
        scheduler=Mock(),
    )
    if binding_failure:
        owner = transfers if binding_failure == "manager" else transceiver
        binding = (
            owner.bind_execution_drain
            if binding_failure == "manager"
            else owner.bind_execution_authority
        )
        binding.side_effect = RuntimeError("bind failed")
        with pytest.raises(RuntimeError, match="bind failed"):
            PyExecutor.configure_execution_authority(executor, table)
        assert executor._execution_guard is None
        with pytest.raises(RuntimeError, match="unknown"):
            tracker.drained(VERSION)
    else:
        PyExecutor.configure_execution_authority(executor, table)
        assert executor._execution_guard is guard
        PyExecutor.configure_execution_authority(executor, table)
        guard.execution_failed.assert_not_called()
    transfers.bind_execution_drain.assert_called_once_with(tracker)
    if transceiver is not None:
        if binding_failure == "manager":
            transceiver.bind_execution_authority.assert_not_called()
        else:
            transceiver.bind_execution_authority.assert_called_once_with(table)


def test_residency_rpc_preserves_cache_namespace_and_page_tiers(monkeypatch) -> None:
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2 = True
    executor.device_id = 0
    page = SimpleNamespace(
        block_ordinal=0,
        life_cycle_id=0,
        token_start=0,
        matched_token_count=4,
        recorded_token_count=4,
        cache_level=1,
        pool_group=0,
        slot_id=7,
        ready=True,
    )
    manager = Mock()
    manager._derive_reuse_salt.return_value = 1234
    manager.impl.probe_reuse_residency.return_value = SimpleNamespace(num_tokens=4, pages=[page])
    manager.impl.cache_tier_list = [
        SimpleNamespace(name="GPU_MEM"),
        SimpleNamespace(name="HOST_MEM"),
    ]
    executor.kv_cache_manager = manager
    monkeypatch.setattr("torch.cuda.device", lambda _: nullcontext())
    result = executor.get_kv_cache_residency([1, 2, 3, 4, 5], "tenant-salt", 7)
    scope, tokens = manager.impl.probe_reuse_residency.call_args.args
    assert scope.lora_id == 7 and scope.salt == 1234
    assert tokens == (1, 2, 3, 4, 5)
    manager._derive_reuse_salt.assert_called_once_with("tenant-salt")
    assert result["matched_tokens"] == 4
    assert result["cache_tiers"] == ["GPU_MEM", "HOST_MEM"]
    assert result["sample_start_ns"] <= result["sample_end_ns"]
    assert result["pages"] == [vars(page)]
    manager.impl.probe_reuse_residency.reset_mock()
    with pytest.raises(ValueError, match="nonnegative int32"):
        executor.get_kv_cache_residency([-1])
    manager.impl.probe_reuse_residency.assert_not_called()
    executor._is_kv_manager_v2 = False
    with pytest.raises(NotImplementedError, match="require KV cache V2"):
        executor.get_kv_cache_residency([1])


def _state(*requests, allowed=None):
    state = SimpleNamespace(
        requests=list(requests),
        sampler_event=Mock(),
        execution_ticket=None,
        device=SimpleNamespace(new_tokens=object()),
        host=SimpleNamespace(new_tokens=object(), single_step_greedy=True),
    )
    if allowed is not None:
        state.execution_ticket = Mock(
            versions=tuple(r.py_execution_version for r in requests),
            sequence_slots=tuple(r.py_seq_slot for r in requests),
        )
        state.execution_ticket.wait.return_value = SimpleNamespace(allowed=allowed)
    return state


def test_entry_rejects_old_and_unknown_without_waiting_for_gpu(guard) -> None:
    old, untagged = _request(), _request(None)
    unknown = _request(replace(VERSION, scope_id="unknown"))
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    assert guard.revoke_obsolete_requests([old, untagged, unknown]) == [old, unknown]
    assert guard.device.mock_calls == []
    assert old.state == unknown.state == LlmRequestState.GENERATION_COMPLETE
    assert old.py_seq_slot == 2
    guard._kv_manager.stop_request_committing.assert_not_called()


def test_commit_submission_is_async_and_preserves_sample_order(guard) -> None:
    old, untagged = _request(), _request(None)
    untagged.py_seq_slot = 0
    state = _state(old, untagged)
    candidate = state.device.new_tokens
    ticket = Mock(versions=(VERSION, None), sequence_slots=(2, 0))
    ticket.wait.return_value = SimpleNamespace(allowed=[True, True])
    guard.device.enqueue_token_commit.return_value = ticket
    guard.commit_sample(_batch(old, untagged), state)
    guard.device.enqueue_token_commit.assert_called_once_with(
        [VERSION, None], [2, 0], candidate, compact=True
    )
    state.sampler_event.synchronize.assert_not_called()
    ticket.wait.assert_not_called()
    assert state.requests == [old, untagged]
    assert state.device.new_tokens is ticket.tokens_device
    assert state.host.new_tokens is ticket.tokens_host
    assert state.execution_ticket is ticket
    # Publication after the GPU accepted still rejects the host commit.
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.token_commit(state):
        assert old.state == LlmRequestState.GENERATION_COMPLETE
        assert old.decoding_iter == 3
        assert untagged.state == LlmRequestState.GENERATION_IN_PROGRESS
    untagged.finish_by_reason.assert_not_called()


def test_device_rejection_blocks_host_commit_even_if_host_still_current(guard) -> None:
    old = _request()
    with guard.token_commit(_state(old, allowed=[False])):
        old.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)


def test_missing_ticket_cannot_commit(guard) -> None:
    with pytest.raises(RuntimeError, match="device commit ticket"):
        with guard.token_commit(_state(_request())):
            pytest.fail("missing ticket admitted a token")


def test_short_device_verdict_cannot_commit(guard) -> None:
    with pytest.raises(RuntimeError, match="complete device commit verdict"):
        with guard.token_commit(_state(_request(), allowed=[])):
            pytest.fail("short verdict admitted a token")


def test_inflight_request_cannot_be_retagged_to_bypass_guard(guard) -> None:
    request = _request()
    state = _state(request, allowed=[True])
    request.py_execution_version = None
    with pytest.raises(RuntimeError, match="identity changed"):
        with guard.token_commit(state):
            pytest.fail("retagged request committed")


def test_inflight_request_cannot_read_tokens_from_a_reassigned_slot(guard) -> None:
    request = _request()
    state = _state(request, allowed=[True])
    request.py_seq_slot = 0
    with pytest.raises(RuntimeError, match="sequence slots changed"):
        with guard.token_commit(state):
            pytest.fail("reassigned slot committed another request's token")


def test_revision_during_sampler_wait_is_observed_without_locking_publisher(guard) -> None:
    old = _request()
    state = _state(old, allowed=[True])
    state.sampler_event.synchronize.side_effect = lambda: guard.table.apply(
        VersionUpdate("session", 2, NEXT)
    )
    with guard.token_commit(state):
        assert old.state == LlmRequestState.GENERATION_COMPLETE
        assert state.sampler_event is None  # Native update must not repeat a wait under the lock.


def test_context_only_chunk_rejects_progress_and_preserves_sampler_row_order(guard) -> None:
    old, other = _request(), _request(None)
    batch = _batch(other, context=True)
    batch.context_requests_chunking = [old]
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.context_commit(batch) as accepted:
        assert accepted.context_requests == [other]
        assert old.state == LlmRequestState.GENERATION_COMPLETE
        assert batch.context_requests == [old, other]
    guard.device.wait_event.assert_not_called()


@pytest.mark.parametrize("last_chunk", [False, True])
def test_native_context_waits_before_accepting_progress(guard, last_chunk):
    guard._native_kv_guards = True
    first, second, untagged = _request(), _request(), _request(None)
    batch = _batch(untagged, context=True)
    if last_chunk:
        batch.context_requests_last_chunk += [first, second]
    else:
        batch.context_requests_chunking = [first, second]
    with guard.forward(batch):
        pass
    completed = []
    guard.device.wait_event.side_effect = lambda event: completed.append(event)
    with guard.context_commit(batch) as accepted:
        assert completed == [guard._uses[id(first)].event]
        assert accepted.context_requests == batch.context_requests
        first.context_current_position = second.context_current_position = 64
    assert guard._kv_frontiers[id(first)] == guard._kv_frontiers[id(second)] == 64
    # Both tagged requests share the same completion event; wait once per batch.
    guard.device.wait_event.assert_called_once()


def test_native_context_revision_can_publish_during_completion_wait(guard):
    guard._native_kv_guards = True
    old, live = _request(), _request(None)
    batch = _batch(live, context=True)
    batch.context_requests_chunking = [old]
    with guard.forward(batch):
        pass
    published = Event()

    def publish():
        guard.table.apply(VersionUpdate("session", 2, NEXT))
        published.set()

    publisher = Thread(target=publish)

    def wait(event):
        publisher.start()
        assert published.wait(2), "context completion held the publication lock"

    guard.device.wait_event.side_effect = wait
    try:
        with guard.context_commit(batch) as accepted:
            assert published.is_set()
            assert accepted.context_requests == [live]
            assert old.is_finished
        assert guard._kv_frontiers[id(old)] == 0
    finally:
        if publisher.ident is not None:
            publisher.join(timeout=2)


@pytest.mark.parametrize("failure", ["missing", "unrecorded", "identity", "timeout"])
def test_native_context_unknown_completion_does_not_publish_prefix(guard, failure):
    guard._native_kv_guards = True
    request = _request()
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    if failure == "missing":
        guard._uses.clear()
    elif failure == "unrecorded":
        guard._uses[id(request)].event = None
    elif failure == "identity":
        guard._uses[id(request)].request = _request()
    else:
        guard.device.wait_event.side_effect = TimeoutError("incomplete chunk")
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.kv_cache_manager = Mock()
    executor.enable_joint_kv_cache_reuse = False
    error = TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(error):
        executor._update_v2_context_resources(batch)
    executor.kv_cache_manager.update_context_resources.assert_not_called()
    assert guard._kv_frontiers[id(request)] == 0


@pytest.mark.parametrize("native", [False, True])
def test_context_wait_does_not_serialize_untagged_or_decode_batches(guard, native):
    guard._native_kv_guards = native
    for batch in (_batch(_request(None), context=True), _batch(_request())):
        with guard.context_commit(batch) as accepted:
            assert accepted is batch
    guard.device.wait_event.assert_not_called()


def test_canceled_prefill_does_not_reenter_generation() -> None:
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    request = SimpleNamespace(
        state=LlmRequestState.GENERATION_COMPLETE,
        context_remaining_length=0,
        move_to_next_context_chunk=Mock(),
    )
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    executor._update_request_states_tp(batch)
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    request.move_to_next_context_chunk.assert_not_called()


def test_untagged_batch_skips_commit_and_readback(guard) -> None:
    untagged = _request(None)
    assert guard.revoke_obsolete_requests([untagged]) == []
    state = _state(untagged)
    batch = _batch(untagged)
    guard.commit_sample(batch, state)
    with guard.token_commit(state):
        pass
    with guard.context_commit(batch) as accepted:
        assert accepted is batch
    assert guard.device.mock_calls == []


def test_release_waits_for_latest_overlap_reader_before_freezing_v2(guard) -> None:
    request = _request()
    batch = _batch(request)
    with guard.forward(batch):
        pass
    previous = guard._uses[id(request)].event
    # The next forward is already submitted when the old token is revoked.
    with guard.forward(batch):
        pass
    latest = guard._uses[id(request)].event
    assert latest is not previous
    order = []
    guard.device.wait_event.side_effect = lambda event: order.append(("wait", event))
    guard._kv_manager.stop_request_committing.side_effect = lambda req: order.append(
        ("freeze", req)
    )
    guard.before_release(request)
    assert order == [("wait", latest), ("freeze", request)]
    assert id(request) not in guard._uses


def test_lossless_suspension_waits_for_latest_use_and_preserves_ownership(guard) -> None:
    request = _request()
    with guard.forward(_batch(request)):
        pass
    previous = guard._uses[id(request)]
    with guard.sampling(_batch(request)):
        pass
    latest = guard._uses[id(request)]
    assert latest.event is not previous.event
    guard._kv_frontiers[id(request)] = 64
    alignment = Mock(reservation=object())
    tracker = RequestDrainTracker()
    tracker.enqueue(VERSION, lambda: request.py_request_id)
    guard.table.alignment_state = alignment
    guard.table.drain_tracker = tracker

    guard.before_suspend(request)

    guard.device.wait_event.assert_called_once_with(latest.event)
    assert guard._uses[id(request)] is latest
    assert guard._kv_frontiers[id(request)] == 64
    assert guard.table.alignment_state is alignment
    assert guard.table.drain_tracker is tracker
    assert alignment.mock_calls == []
    assert not tracker.drained(VERSION)
    assert request.py_execution_version == VERSION
    assert request.py_seq_slot == 2
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()
    # Event ownership survives a pause, including a revision while suspended.
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    guard.before_release(request)
    assert id(request) not in guard._uses
    assert id(request) not in guard._kv_frontiers
    guard._kv_manager.preserve_execution_prefix.assert_called_once_with(request, 64)
    guard._kv_manager.stop_request_committing.assert_called_once_with(request)
    assert not tracker.drained(VERSION)


@pytest.mark.parametrize("failure", ["missing", "wrong_request", "eventless", "wait"])
def test_lossless_suspension_fails_closed_without_completion(guard, failure: str) -> None:
    request = _request()
    with guard.forward(_batch(request)):
        pass
    guard._kv_frontiers[id(request)] = 64
    if failure == "missing":
        guard._uses.clear()
    elif failure == "wrong_request":
        guard._uses[id(request)].request = _request()
    elif failure == "eventless":
        guard._uses[id(request)].event = None
    else:
        guard.device.wait_event.side_effect = TimeoutError("still running")
    uses = dict(guard._uses)

    with pytest.raises((RuntimeError, TimeoutError)):
        guard.before_suspend(request)

    assert guard._uses == uses
    assert guard._kv_frontiers == {id(request): 64}
    assert guard._kv_manager.mock_calls == []


@pytest.mark.parametrize("unsupported", ["v1", "distributed"])
def test_lossless_suspension_rejects_unsupported_owner(guard, unsupported: str) -> None:
    if unsupported == "v1":
        guard._kv_manager = None
    else:
        guard._rank_consensus = Mock()
    with pytest.raises(RuntimeError, match="single-rank KV cache V2"):
        guard.before_suspend(_request())
    guard.device.wait_event.assert_not_called()


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
def test_authority_binds_lossless_suspension_only_to_single_rank_v2(
    monkeypatch, v2: bool, distributed: bool
) -> None:
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    consensus = Mock() if distributed else None
    owner = SimpleNamespace(table=table, _rank_consensus=consensus, before_suspend=Mock())
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.execution_guard.ExecutionDeviceGuard",
        lambda *_: owner,
    )
    scheduler = object.__new__(KVCacheV2Scheduler) if v2 else SimpleNamespace()
    scheduler._execution_suspend_guard = None
    executor = SimpleNamespace(
        _execution_guard=None,
        async_transfer_manager=Mock(),
        kv_cache_transceiver=None,
        scheduler=scheduler,
    )

    PyExecutor.configure_execution_authority(executor, table, consensus)
    PyExecutor.configure_execution_authority(executor, table, consensus)

    assert executor._execution_guard is owner
    assert scheduler._execution_suspend_guard is (
        owner.before_suspend if v2 and not distributed else None
    )


def test_wait_failure_retains_ownership_and_never_closes_cache(guard) -> None:
    request = _request()
    with guard.forward(_batch(request)):
        pass
    guard.device.wait_event.side_effect = TimeoutError("still running")
    with pytest.raises(TimeoutError):
        guard.before_release(request)
    assert id(request) in guard._uses
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()


def test_partial_forward_failure_still_records_gpu_readers(guard) -> None:
    request = _request()
    with pytest.raises(ValueError, match="forward failed"):
        with guard.forward(_batch(request)):
            raise ValueError("forward failed after kernel launch")
    guard.before_release(request)
    guard.device.wait_event.assert_called_once()


def test_partial_sampling_failure_covers_work_submitted_after_forward(guard) -> None:
    request = _request()
    batch = _batch(request)
    with guard.forward(batch):
        pass
    forward_done = guard._uses[id(request)].event
    with pytest.raises(ValueError, match="sampling failed"):
        with guard.sampling(batch):
            raise ValueError("sampling failed after partial submission")
    sampling_done = guard._uses[id(request)].event
    assert sampling_done is not forward_done
    guard.before_release(request)
    guard.device.wait_event.assert_called_once_with(sampling_done)


def test_context_only_sampling_tracks_completion_without_a_token_ticket(guard) -> None:
    request = _request()
    batch = ScheduledRequests()
    batch.context_requests_chunking = [request]
    with guard.sampling(batch):
        guard.commit_sample(batch, _state())
    guard.device.enqueue_token_commit.assert_not_called()
    guard.before_release(request)
    guard.device.wait_event.assert_called_once()


def test_retired_sample_does_not_decrement_reused_sequence_slot() -> None:
    sampler = object.__new__(TorchSampler)
    sampler._track_pending_steps = True
    sampler._pending_steps = [0, 0, 4]  # Slot 2 now belongs to a different request.
    old = _request()
    old.py_is_draft = False
    old.state = LlmRequestState.GENERATION_COMPLETE
    state = _state(old)
    state.sampler_event = None
    sampler.update_requests(state)
    assert sampler._pending_steps == [0, 0, 4]


@pytest.mark.parametrize("capture_possible", [False, True])
def test_model_submission_excludes_publisher_cuda_calls(monkeypatch, capture_possible) -> None:
    authority = object.__new__(DeviceExecutionAuthority)
    authority._lock = Lock()
    authority._failure = None
    authority.device = None
    authority._published = {(VERSION.tenant_id, VERSION.scope_id): (VERSION.epoch, None)}
    cuda_calls = Mock(side_effect=nullcontext)
    monkeypatch.setattr("torch.cuda.device", cuda_calls)
    started, finished = Event(), Event()
    errors = []

    def publish():
        started.set()
        try:
            authority.publish(VERSION)  # Duplicate needs no actual device allocation.
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    worker = Thread(target=publish)
    try:
        with authority.submission(capture_possible=capture_possible):
            worker.start()
            assert started.wait(2)
            if capture_possible:
                assert not finished.wait(0.05)
                cuda_calls.assert_not_called()
            else:
                assert finished.wait(2)
                cuda_calls.assert_called_once_with(None)
    finally:
        if worker.ident is not None:
            worker.join(timeout=2)
    assert finished.is_set()
    assert not errors
    cuda_calls.assert_called_once_with(None)


@pytest.mark.parametrize("may_capture", [False, True, None])
def test_forward_submission_uses_engine_capture_contract(guard, may_capture):
    guard._model_engine = (
        SimpleNamespace()
        if may_capture is None
        else SimpleNamespace(forward_may_capture=may_capture)
    )
    with guard.forward(_batch(_request())):
        pass
    guard.device.submission.assert_called_once_with(
        capture_possible=True if may_capture is None else may_capture
    )


@pytest.mark.parametrize(
    "capture_source",
    [
        None,
        "warmup",
        "compile",
        "backend",
        "breakable",
        "missing_graph",
        "graph",
        "callback",
        "engine",
        "runner",
    ],
)
def test_model_engine_capture_contract_is_conservative(capture_source):
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    engine = object.__new__(PyTorchModelEngine)
    runner = object.__new__(DecoderRunner)
    engine._runner = runner
    engine._is_warmup = capture_source == "warmup"
    engine.torch_compile_config = object() if capture_source == "compile" else None
    runner._torch_compile_backend = object() if capture_source == "backend" else None
    runner.breakable_cuda_graph_runner = object() if capture_source == "breakable" else None
    runner.cuda_graph_runner = SimpleNamespace(_capture_allowed=capture_source == "graph")
    if capture_source == "missing_graph":
        runner.cuda_graph_runner = None
    runner.forward_pass_callable = object() if capture_source == "callback" else None
    if capture_source == "engine":

        class CustomEngine(PyTorchModelEngine):
            pass

        custom = object.__new__(CustomEngine)
        custom.__dict__.update(engine.__dict__)
        engine = custom
    elif capture_source == "runner":

        class CustomRunner(DecoderRunner):
            pass

        custom = object.__new__(CustomRunner)
        custom.__dict__.update(runner.__dict__)
        engine._runner = custom

    assert engine.forward_may_capture is (capture_source is not None)
    # The same engine must re-enter exclusion when capture is enabled later.
    runner.cuda_graph_runner = SimpleNamespace(_capture_allowed=True)
    assert engine.forward_may_capture


def test_event_record_failure_prevents_release(guard, monkeypatch) -> None:
    event = Mock()
    event.record.side_effect = RuntimeError("CUDA event failed")
    monkeypatch.setattr("torch.cuda.Event", lambda: event)
    request = _request()
    with pytest.raises(RuntimeError, match="CUDA event failed"):
        with guard.forward(_batch(request)):
            pass
    with pytest.raises(RuntimeError, match="completion is unknown"):
        guard.before_release(request)
    guard._kv_manager.stop_request_committing.assert_not_called()


def test_identical_native_ids_and_sequence_slots_do_not_alias_ownership(guard) -> None:
    first, second = _request(), _request()
    with guard.forward(_batch(first)):
        pass
    with guard.forward(_batch(second)):
        pass
    second_use = guard._uses[id(second)]
    guard.before_release(first)
    assert guard._uses[id(second)] is second_use


def test_v2_freeze_preserves_other_owners_and_never_appends_tokens() -> None:
    manager = object.__new__(KVCacheManagerV2)
    old_cache, shared_owner = Mock(), Mock()
    manager.kv_cache_map = {42: old_cache, 43: shared_owner}
    request = _request()
    manager.stop_request_committing(request)
    old_cache.stop_committing.assert_called_once_with()
    old_cache.commit.assert_not_called()
    old_cache.close.assert_not_called()
    assert manager.kv_cache_map == {42: old_cache, 43: shared_owner}
    assert shared_owner.mock_calls == []
    manager.kv_cache_map.pop(42)
    manager.stop_request_committing(request)  # Already released is harmless.


def test_v2_context_commit_rechecks_revision_after_context_progress(guard) -> None:
    old, other = _request(), _request(None)
    batch = _batch(old, other, context=True)
    with guard.context_commit(batch) as accepted:
        assert accepted.context_requests == [old, other]
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.enable_joint_kv_cache_reuse = False
    executor.kv_cache_manager = Mock()
    executor._update_v2_context_resources(batch)
    committed_batch = executor.kv_cache_manager.update_context_resources.call_args.args[0]
    assert committed_batch.context_requests == [other]
    assert batch.context_requests == [old, other]


@pytest.mark.parametrize("frontier", [0, 16, 35])
def test_v2_retains_only_accepted_context_after_drain(guard, frontier) -> None:
    request = _request()
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    with guard.context_commit(batch):
        request.context_current_position = frontier
    with guard.forward(batch):
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.context_commit(batch) as accepted:
        assert not accepted.context_requests
    order = []
    guard.device.wait_event.side_effect = lambda event: order.append("drain")
    guard._kv_manager.preserve_execution_prefix.side_effect = lambda req, count: order.append(count)
    guard._kv_manager.stop_request_committing.side_effect = lambda req: order.append("freeze")
    guard.before_release(request)
    assert order == ["drain", frontier, "freeze"]


@pytest.mark.parametrize(
    "frontier,start,expected_end", [(0, 0, 0), (15, 0, 0), (35, 16, 32), (35, 24, 32), (35, 35, 35)]
)
def test_v2_preserves_complete_blocks_without_publishing_unfinished_tail(
    frontier, start, expected_end
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = manager._can_publish_block_reuse = True
    manager.tokens_per_block = 16
    request = _request()
    request.is_dummy_request = False
    request.context_current_position = 48
    cache = Mock(is_active=True, capacity=64, num_committed_tokens=start)
    other = Mock()
    manager.kv_cache_map = {42: cache, 43: other}
    manager._reuse_token_source = Mock(return_value=list(range(64)))
    manager._augment_tokens_for_block_reuse = lambda tokens, req, start, end: tokens[start:end]

    def commit(tokens):
        cache.num_committed_tokens += len(tokens)

    cache.commit.side_effect = commit
    manager.preserve_execution_prefix(request, frontier)
    manager.preserve_execution_prefix(request, frontier)
    assert cache.num_committed_tokens == expected_end
    if expected_end > start:
        cache.commit.assert_called_once_with(list(range(start, expected_end)))
    else:
        cache.commit.assert_not_called()
    cache.stop_committing.assert_not_called()
    cache.close.assert_not_called()
    assert other.mock_calls == []


@pytest.mark.parametrize(
    "unavailable", ["disabled", "not_publishable", "dummy", "suspended", "absent", "stopped"]
)
def test_v2_prefix_preservation_does_not_restore_or_publish_unavailable_cache(unavailable) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = unavailable != "disabled"
    manager._can_publish_block_reuse = unavailable != "not_publishable"
    manager.tokens_per_block = 16
    request = _request()
    request.context_current_position = 32
    request.is_dummy_request = unavailable == "dummy"
    cache = Mock(is_active=unavailable != "suspended", num_committed_tokens=0, capacity=32)
    cache.is_commit_allowed = unavailable != "stopped"
    manager.kv_cache_map = {} if unavailable == "absent" else {42: cache}
    manager.preserve_execution_prefix(request, 32)
    cache.commit.assert_not_called()
    cache.resume.assert_not_called()


@pytest.mark.parametrize("frontier", [-1, True, 49])
def test_v2_prefix_preservation_rejects_invalid_frontier(frontier) -> None:
    manager = object.__new__(KVCacheManagerV2)
    request = _request()
    request.context_current_position = 48
    with pytest.raises(ValueError, match="completed frontier"):
        manager.preserve_execution_prefix(request, frontier)


@pytest.mark.parametrize("capacity,source_length", [(16, 64), (64, 16)])
def test_v2_prefix_preservation_rejects_missing_allocation(capacity, source_length) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = manager._can_publish_block_reuse = True
    manager.tokens_per_block = 16
    request = _request()
    request.is_dummy_request = False
    request.context_current_position = 32
    cache = Mock(is_active=True, capacity=capacity, num_committed_tokens=0)
    manager.kv_cache_map = {42: cache}
    manager._reuse_token_source = Mock(return_value=list(range(source_length)))
    with pytest.raises(RuntimeError, match="retained KV allocation"):
        manager.preserve_execution_prefix(request, 32)
    cache.commit.assert_not_called()


def test_executor_release_failure_does_not_free_any_resources(guard) -> None:
    request = _request()
    with guard.forward(_batch(request)):
        pass
    guard.device.wait_event.side_effect = TimeoutError("still running")
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.resource_manager = Mock()
    with pytest.raises(TimeoutError):
        executor._free_request_resources(request)
    executor.resource_manager.free_resources.assert_not_called()


def _executor():
    return SimpleNamespace(
        dist=SimpleNamespace(world_size=1, rank=0),
        disable_overlap_scheduler=True,
        max_beam_width=1,
        sampler=object.__new__(TorchSampler),
        drafter=None,
        model_engine=SimpleNamespace(
            is_spec_decode=False,
            cuda_graph_config=None,
            torch_compile_config=None,
            enable_in_graph_sampling=False,
        ),
        _is_kv_manager_v2=False,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        is_encoder_decoder=False,
        guided_decoder=None,
        kv_cache_manager=SimpleNamespace(),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_guard_binds_exact_membership_and_only_source_wakes(monkeypatch, rank):
    executor = _executor()
    executor.dist = SimpleNamespace(world_size=2, tp_size=2, pp_size=1, cp_size=1, rank=rank)
    executor.enable_attention_dp = False
    executor._is_kv_manager_v2 = True
    executor.executor_request_queue = Mock()
    executor.model_engine.execution_attention = SimpleNamespace(authority=Mock())
    table = WorkerVersionTable("session")
    consensus = ExecutionStepConsensus("session", rank, (0, 1), Mock())
    guard = ExecutionDeviceGuard(executor, table, consensus)
    assert guard.is_distributed
    assert not hasattr(executor.kv_cache_manager, "_execution_resident_recovery")
    table.alignment_state._wake()
    assert executor.executor_request_queue.wake_alignment.call_count == int(rank == 0)


def test_resident_owner_query_is_bound_to_the_installed_worker_state():
    executor = _executor()
    executor._is_kv_manager_v2 = True
    executor.executor_request_queue = Mock()
    executor.model_engine.execution_attention = SimpleNamespace(authority=Mock())
    table = WorkerVersionTable("session")
    installed = ExecutionDeviceGuard(executor, table)
    request = SimpleNamespace(py_execution_version=VERSION, py_request_id=19)
    table.alignment_state.has_resident_recovery = Mock(return_value=True)
    manager = executor.kv_cache_manager
    assert KVCacheManagerV2.has_execution_resident_recovery(manager, request)
    table.alignment_state.has_resident_recovery.assert_called_once_with(VERSION, 19)
    request.py_execution_version = None
    assert not KVCacheManagerV2.has_execution_resident_recovery(manager, request)
    assert table.alignment_state.has_resident_recovery.call_count == 1
    assert installed.table is table


def test_resident_owner_query_is_false_without_an_execution_authority():
    manager = SimpleNamespace()
    request = SimpleNamespace(py_execution_version=VERSION, py_request_id=19)
    assert not KVCacheManagerV2.has_execution_resident_recovery(manager, request)


@pytest.mark.parametrize(
    "fault", ["pp", "cp", "adp", "no_coordinator", "session", "rank", "membership"]
)
def test_tp_guard_rejects_incomplete_topology_before_binding(fault):
    executor = _executor()
    executor.dist = SimpleNamespace(world_size=2, tp_size=2, pp_size=1, cp_size=1, rank=0)
    executor.enable_attention_dp = fault == "adp"
    if fault == "pp":
        executor.dist.pp_size = 2
    if fault == "cp":
        executor.dist.cp_size = 2
    table = WorkerVersionTable("session")
    consensus = ExecutionStepConsensus(
        "other" if fault == "session" else "session",
        1 if fault == "rank" else 0,
        (0, 1, 2) if fault == "membership" else (0, 1),
        Mock(),
    )
    with pytest.raises((NotImplementedError, ValueError)):
        ExecutionDeviceGuard(executor, table, None if fault == "no_coordinator" else consensus)
    assert table.drain_tracker is None


@pytest.mark.parametrize("early_exit", [None, "0", "1"])
def test_controller_adopts_authority_already_captured_by_attention(monkeypatch, early_exit):
    monkeypatch.delenv("TRTLLM_RETIRE_EARLY_EXIT", raising=False)
    if early_exit is not None:
        monkeypatch.setenv("TRTLLM_RETIRE_EARLY_EXIT", early_exit)
    executor = _executor()
    executor._is_kv_manager_v2 = True
    executor.executor_request_queue = Mock()
    startup_authority = Mock()
    executor.model_engine.execution_attention = SimpleNamespace(authority=startup_authority)
    constructor = Mock(side_effect=AssertionError("must not replace graph authority"))
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.execution_guard.DeviceExecutionAuthority", constructor
    )
    table = WorkerVersionTable("session")
    guard = ExecutionDeviceGuard(executor, table)
    table.apply(VersionUpdate("session", 1, VERSION))
    assert guard.device is startup_authority
    assert guard._early_exit_enabled is (early_exit != "0")
    startup_authority.publish.assert_called_once_with(VERSION)
    constructor.assert_not_called()


@pytest.mark.parametrize("value", ["", "yes", "2"])
def test_invalid_early_exit_setting_fails_before_executor_setup(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_RETIRE_EARLY_EXIT", value)
    table = WorkerVersionTable("session")
    with pytest.raises(ValueError, match="TRTLLM_RETIRE_EARLY_EXIT must be 0 or 1"):
        ExecutionDeviceGuard(None, table)
    assert table.drain_tracker is None
    assert table.alignment_state is None


@pytest.mark.parametrize(
    "target, field, value",
    [
        ("", "max_beam_width", 2),
        ("", "kv_cache_manager", None),
        ("", "kv_cache_transceiver", object()),
        ("", "kv_connector_manager", object()),
        ("", "drafter", object()),
        ("", "guided_decoder", object()),
        ("", "is_encoder_decoder", True),
        ("", "sampler", object()),
        ("dist", "world_size", 2),
        ("model_engine", "is_spec_decode", True),
        ("kv_cache_manager", "is_linear_attention", True),
        ("kv_cache_manager", "is_vswa", True),
    ],
)
def test_unsupported_execution_modes_fail_before_device_allocation(target, field, value) -> None:
    executor = _executor()
    ExecutionDeviceGuard.validate_executor(executor)
    setattr(getattr(executor, target) if target else executor, field, value)
    with pytest.raises(NotImplementedError, match="RETIRE device guards"):
        ExecutionDeviceGuard(executor, WorkerVersionTable("session"))


@pytest.mark.parametrize(
    "field, value",
    [
        ("py_beam_width", 2),
        ("py_draft_tokens", [1]),
        ("py_multimodal_data", {}),
        ("py_disaggregated_params", object()),
        ("py_logits_post_processors", [object()]),
    ],
)
def test_unsupported_requests_fail_before_scheduling(field, value) -> None:
    request = _request()
    ExecutionDeviceGuard.validate_request(request)
    setattr(request, field, value)
    with pytest.raises(ValueError, match="text-only"):
        ExecutionDeviceGuard.validate_request(request)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("mode", ["eager", "graph", "compiled", "compiled-graph"])
@pytest.mark.parametrize("in_graph_sampling", [False, True])
def test_optimized_mode_matrix_is_accepted(v2, overlap, mode, in_graph_sampling) -> None:
    executor = _executor()
    executor._is_kv_manager_v2 = v2
    executor.disable_overlap_scheduler = not overlap
    executor.model_engine.cuda_graph_config = object() if "graph" in mode else None
    executor.model_engine.torch_compile_config = object() if "compiled" in mode else None
    executor.model_engine.enable_in_graph_sampling = in_graph_sampling
    ExecutionDeviceGuard.validate_executor(executor)


@pytest.mark.parametrize("early_exit", [True, False])
def test_fence_covers_enqueue_until_terminal_cleanup(guard, early_exit):
    from tensorrt_llm.execution_authority import Revision
    from tensorrt_llm.execution_fence import FenceRequest, RequestDrainTracker

    guard._early_exit_enabled = early_exit
    guard.table.drain_tracker = RequestDrainTracker()
    request = _request()
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.enqueue_request.return_value = request.py_request_id
    with guard.table.admission(VERSION):
        assert executor.enqueue_request(request) == request.py_request_id
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    query = FenceRequest("session", Revision("edit", VERSION, NEXT))
    assert guard.table.poll_fence(query) is None
    with guard.forward(_batch(request)):
        pass
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = set()
    executor._disagg_coordinator = Mock()
    executor.gather_all_responses = True
    executor.result_wait_queues = {}
    # Missing GPU completion cannot become a successful fence.
    event = guard._uses[id(request)].event
    guard._uses[id(request)].event = None
    with pytest.raises(RuntimeError, match="completion is unknown"):
        executor._do_terminate_request(request)
    assert guard.table.poll_fence(query) is None
    guard._uses[id(request)].event = event
    # Resource-only release can be used for recomputation and is not terminal.
    executor._free_request_resources(request)
    assert guard.table.poll_fence(query) is None
    # A failed cleanup must not certify completion.
    executor.resource_manager.free_resources.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        executor._do_terminate_request(request)
    assert guard.table.poll_fence(query) is None
    executor.resource_manager.free_resources.side_effect = None
    executor._do_terminate_request(request)
    assert guard.table.poll_fence(query).request == query
    guard.device.wait_event.assert_called_once()


def test_fence_enqueue_rejects_late_obsolete_work(guard):
    from tensorrt_llm.execution_fence import RequestDrainTracker
    from tensorrt_llm.execution_gate import ExecutionAuthorityError

    guard.table.drain_tracker = RequestDrainTracker()
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    submit = Mock(return_value=42)
    with pytest.raises(ExecutionAuthorityError):
        guard.enqueue(VERSION, submit)
    submit.assert_not_called()
    assert guard.table.drain_tracker.drained(VERSION)


def test_waiting_cancellation_completes_only_removed_items(guard):
    from tensorrt_llm.execution_fence import RequestDrainTracker

    guard.table.drain_tracker = RequestDrainTracker()
    for request_id in (1, 2):
        guard.enqueue(VERSION, lambda request_id=request_id: request_id)
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.canceled_req_ids = [1, 2]
    # Request 2 may still be in an earlier ingress queue; cancellation alone
    # cannot certify its removal or absence of future resource access.
    executor.waiting_queue = Mock()
    executor.waiting_queue.__iter__ = Mock(
        return_value=iter([SimpleNamespace(id=1, request=_request())])
    )
    executor.active_requests = []
    executor._handle_canceled_requests()
    assert not guard.table.drain_tracker.drained(VERSION)
    guard.terminal(2, VERSION)
    assert guard.table.drain_tracker.drained(VERSION)


def test_child_registration_keeps_fence_pending_after_parent_cleanup(guard):
    from tensorrt_llm.execution_fence import RequestDrainTracker

    guard.table.drain_tracker = RequestDrainTracker()
    guard.enqueue(VERSION, lambda: 42)
    child = _request()
    child.is_child = True
    child.parent_request_id = 42
    child.py_request_id = 43
    guard.register_children([child])
    guard.terminal(42, VERSION)
    assert not guard.table.drain_tracker.drained(VERSION)
    guard.terminal(43, VERSION)
    assert guard.table.drain_tracker.drained(VERSION)


def test_alignment_hold_acquired_before_old_resources_are_freed(guard):
    from tensorrt_llm.execution_alignment import (
        AlignmentInput,
        AlignmentRequest,
        WorkerAlignmentState,
    )
    from tensorrt_llm.execution_authority import Revision

    order = []
    hold = Mock(num_tokens=2)

    def reserve(revised):
        order.append("acquire hold")
        return hold

    state = guard.table.alignment_state = WorkerAlignmentState(reserve, lambda: None)
    request = _request()
    revised = AlignmentRequest(
        "session", Revision("edit", VERSION, NEXT), AlignmentInput((1, 2, 3))
    )
    guard.table.apply(VersionUpdate("session", 2, NEXT, revised))
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.resource_manager = Mock()
    executor.resource_manager.free_resources.side_effect = lambda request: order.append("free old")
    executor._prefetched_request_ids = set()
    executor._disagg_coordinator = Mock()
    executor._free_request_resources(request)
    assert order == ["acquire hold", "free old"]
    hold.close.assert_not_called()
    state.close()
    guard.process_alignment()
    hold.close.assert_called_once()


def test_alignment_wake_is_consumed_without_becoming_inference(guard):
    from queue import Queue

    from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue

    queue = object.__new__(ExecutorRequestQueue)
    queue.enqueue_lock = Lock()
    queue.active = True
    queue.request_queue = Queue()
    queue._get_request_id = Mock(return_value=99)
    assert queue.reserve_request_id() == 99
    assert queue.request_queue.empty()
    queue.wake_alignment()
    item = queue.request_queue.get_nowait()
    assert item.is_alignment_wake_request
    assert not item.is_normal_request
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    guard.process_alignment = Mock()
    assert executor._handle_special_queue_items([item]) == []
    guard.process_alignment.assert_called_once()
    queue.active = False
    queue.wake_alignment()
    assert queue.request_queue.empty()


def _isolated_handoff(guard):
    from tensorrt_llm.execution_alignment import (
        AlignmentInput,
        AlignmentRequest,
        WorkerAlignmentState,
    )
    from tensorrt_llm.execution_authority import Revision

    request = _request()
    cache = SimpleNamespace(is_active=True)
    guard._async_context = True
    guard._isolated_requests = {}
    guard._pending_releases = {}
    guard._release_error = None
    guard._kv_manager.kv_cache_map = {request.py_request_id: cache}
    guard.table.drain_tracker = RequestDrainTracker()
    guard.table.drain_tracker.enqueue(VERSION, lambda: request.py_request_id)
    with guard.forward(_batch(request)):
        pass
    guard._uses[id(request)].event.query.return_value = False
    guard._alignment_adapter = Mock()
    guard._alignment_adapter.reserve_isolated.return_value = Mock(num_tokens=2)
    guard.table.alignment_state = WorkerAlignmentState(
        guard._alignment_adapter.reserve,
        lambda: None,
        prepare=guard._prepare_prefix,
        reserve_isolated=guard._reserve_isolated_prefix,
    )
    alignment = AlignmentRequest(
        "session", Revision("edit", VERSION, NEXT), AlignmentInput((1, 2, 3))
    )
    guard.table.apply(VersionUpdate("session", 2, NEXT, alignment))
    return request, cache, alignment


def _retiring_executor(guard, request):
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_coordinator = Mock()
    executor.gather_all_responses = True
    executor.result_wait_queues = {request.py_request_id: object()}
    return executor


def test_isolated_alignment_resumes_while_old_gpu_work_and_cleanup_remain(guard):
    from tensorrt_llm.execution_fence import FenceRequest

    request, cache, alignment = _isolated_handoff(guard)
    guard.process_alignment()
    receipt = guard.table.poll_alignment(alignment)
    assert receipt.reusable_prefix_length == 2
    fence = FenceRequest("session", alignment.revision)
    assert guard.table.poll_fence(fence) is None
    assert request.is_finished
    guard.device.wait_event.assert_not_called()
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()

    guard.table.apply(VersionUpdate("session", 3, NEXT, alignment, resume=receipt))
    successor = _request(NEXT)
    successor.py_request_id = 43
    successor.get_tokens = lambda beam: [1, 2, 3]
    successor.cache_salt = successor.lora_task_id = None
    guard.validate_successor(successor)
    assert successor.execution_reuse_limit == 2
    assert guard._kv_manager.kv_cache_map[request.py_request_id] is cache

    executor = _retiring_executor(guard, request)
    executor._do_terminate_request(request)
    executor._do_terminate_request(request)  # duplicate retirement is idempotent
    assert guard.pending_releases == 1
    guard.process_releases(executor._finish_terminated_request)
    executor.resource_manager.free_resources.assert_not_called()
    assert request.py_request_id in executor.result_wait_queues
    assert guard.table.poll_fence(fence) is None
    # The old completion need not depend on successor attachment or success.
    guard._uses[id(request)].event.query.return_value = True
    guard.process_releases(executor._finish_terminated_request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_request_id not in executor.result_wait_queues
    assert guard.pending_releases == 0
    assert guard.table.poll_fence(fence) is not None


@pytest.mark.parametrize("invalid", ["event", "allocation", "native_cleanup"])
def test_isolated_alignment_and_retirement_fail_closed(guard, invalid):
    from tensorrt_llm.execution_alignment import AlignmentSafetyError

    request, _, _ = _isolated_handoff(guard)
    if invalid == "event":
        guard._uses[id(request)].event = None
        with pytest.raises(AlignmentSafetyError, match="ownership is unknown"):
            guard.process_alignment()
        assert not guard.table.drain_tracker.drained(VERSION)
        guard._alignment_adapter.reserve_isolated.assert_not_called()
        return
    guard.process_alignment()
    executor = _retiring_executor(guard, request)
    if invalid == "allocation":
        guard._kv_manager.kv_cache_map[request.py_request_id] = object()
        with pytest.raises(RuntimeError, match="allocation changed"):
            executor._do_terminate_request(request)
        executor.resource_manager.free_resources.assert_not_called()
        assert not guard.table.drain_tracker.drained(VERSION)
        return
    executor._do_terminate_request(request)
    guard._uses[id(request)].event.query.return_value = True
    executor.resource_manager.free_resources.side_effect = RuntimeError("native cleanup failed")
    with pytest.raises(RuntimeError, match="native cleanup failed"):
        guard.process_releases(executor._finish_terminated_request)
    assert guard.pending_releases == 1
    assert request.py_request_id in executor.result_wait_queues
    with pytest.raises(RuntimeError, match="restart required"):
        guard.table.drain_tracker.drained(VERSION)
    with pytest.raises(RuntimeError, match="cleanup is unknown"):
        guard.process_releases(executor._finish_terminated_request)
    executor.resource_manager.free_resources.assert_called_once()


def test_isolated_retirement_waits_for_latest_sampling_event_and_context_ticket(guard):
    request, _, _ = _isolated_handoff(guard)
    guard.process_alignment()
    owner = guard._isolated_requests[id(request)]
    # Sampling already queued by the overlap loop can extend the completion
    # record. Isolation is tied to the allocation, not the superseded event.
    with guard.sampling(_batch(request)):
        pass
    assert guard._isolated_requests[id(request)] is owner
    event = guard._uses[id(request)].event
    event.query.return_value = True
    ticket = Mock()
    ticket.query.return_value = False
    guard._context_chunks = {id(request): [SimpleNamespace(ticket=ticket)]}
    executor = _retiring_executor(guard, request)
    executor._do_terminate_request(request)
    guard.process_releases(executor._finish_terminated_request)
    executor.resource_manager.free_resources.assert_not_called()
    # Receipt consumption itself is covered by the existing context tests.
    guard._context_chunks.clear()
    guard.process_releases(executor._finish_terminated_request)
    executor.resource_manager.free_resources.assert_called_once()


def test_rapid_revision_keeps_old_owner_but_reclaims_ready_newer_owner(guard):
    from tensorrt_llm.execution_alignment import AlignmentRequest
    from tensorrt_llm.execution_authority import Revision
    from tensorrt_llm.execution_fence import FenceRequest

    old_request, _, first = _isolated_handoff(guard)
    guard.process_alignment()
    first_receipt = guard.table.poll_alignment(first)
    guard.table.apply(VersionUpdate("session", 3, NEXT, first, resume=first_receipt))
    next_request = _request(NEXT)
    next_request.py_request_id = 43
    next_request.get_tokens = lambda beam: [1, 2, 3]
    next_request.cache_salt = next_request.lora_task_id = None
    guard.validate_successor(next_request)
    guard._kv_manager.kv_cache_map[43] = SimpleNamespace(is_active=True)
    guard.table.drain_tracker.enqueue(NEXT, lambda: 43)
    guard.table.alignment_state.confirm_attachment(NEXT, 43, 2)
    with guard.forward(_batch(next_request)):
        pass
    guard._uses[id(next_request)].event.query.return_value = True
    old_owner = guard._isolated_requests[id(old_request)]
    later = replace(NEXT, epoch=NEXT.epoch + 1)
    second = AlignmentRequest("session", Revision("again", NEXT, later), first.revised_input)
    guard._alignment_adapter.reserve_isolated.return_value = Mock(num_tokens=2)
    guard.table.apply(VersionUpdate("session", 4, later, second))
    guard.process_alignment()
    second_receipt = guard.table.poll_alignment(second)
    assert second_receipt is not None
    assert guard._isolated_requests[id(old_request)] is old_owner
    guard.table.apply(VersionUpdate("session", 5, later, second, resume=second_receipt))
    executor = _retiring_executor(guard, old_request)
    executor._do_terminate_request(old_request)
    executor._do_terminate_request(next_request)
    guard.process_releases(executor._finish_terminated_request)
    executor.resource_manager.free_resources.assert_called_once_with(next_request)
    assert guard.pending_releases == 1
    assert guard.table.poll_fence(FenceRequest("session", second.revision)) is None
    guard._uses[id(old_request)].event.query.return_value = True
    guard.process_releases(executor._finish_terminated_request)
    assert guard.pending_releases == 0
    assert guard.table.poll_fence(FenceRequest("session", first.revision)) is not None
    assert guard.table.poll_fence(FenceRequest("session", second.revision)) is not None


@pytest.mark.parametrize("unsupported", ["no_overlap", "distributed", "no_prefix", "lora_source"])
def test_isolated_alignment_falls_back_to_existing_drain_path(guard, unsupported):
    request, _, alignment = _isolated_handoff(guard)
    if unsupported == "no_overlap":
        guard._async_context = False
    elif unsupported == "distributed":
        guard._rank_consensus = object()
    elif unsupported == "lora_source":
        request.lora_task_id = 11
    else:
        guard._alignment_adapter.reserve_isolated.return_value = None
    assert guard._reserve_isolated_prefix(alignment) is None
    assert not request.is_finished
    assert not guard.defer_release(request)
    assert not guard.table.drain_tracker.isolated(VERSION)


def test_retired_slots_survive_pending_cpu_sample_even_after_gpu_completion(guard):
    request, _, _ = _isolated_handoff(guard)
    guard.process_alignment()
    executor = _retiring_executor(guard, request)
    executor._do_terminate_request(request)
    guard._uses[id(request)].event.query.return_value = True
    executor.previous_batch = SimpleNamespace(scheduled_requests=_batch(request))
    executor._process_execution_releases()
    executor.resource_manager.free_resources.assert_not_called()
    assert guard.pending_releases == 1
    # The normal overlap consumer replaces/clears previous_batch only after
    # it has read the old sampler rows. Only the following safe point reclaims.
    executor.previous_batch = None
    executor._process_execution_releases()
    executor.resource_manager.free_resources.assert_called_once_with(request)


@pytest.mark.parametrize("ready", [True, False])
def test_rebalance_rechecks_owners_retired_while_consuming_previous_batch(
    guard, monkeypatch, ready
):
    request, _, _ = _isolated_handoff(guard)
    guard.process_alignment()
    guard._uses[id(request)].event.query.return_value = ready
    executor = _retiring_executor(guard, request)
    executor._agreed_need_adjustment = lambda: True
    executor._consume_previous_batch_for_rebalance = lambda: executor._do_terminate_request(request)
    executor._rebalance_kv_pools_now = Mock()
    monkeypatch.setattr("torch.cuda.current_stream", Mock(return_value=Mock()))
    assert guard.pending_releases == 0
    executor._maybe_rebalance_kv_pools()
    assert guard.pending_releases == (0 if ready else 1)
    assert executor._rebalance_kv_pools_now.call_count == int(ready)
    assert executor.resource_manager.free_resources.call_count == int(ready)


@pytest.mark.parametrize("draining", [True, False])
def test_retired_owners_block_shutdown_rebalance_and_control(guard, draining):
    request, _, _ = _isolated_handoff(guard)
    guard.process_alignment()
    executor = _retiring_executor(guard, request)
    executor._do_terminate_request(request)
    executor.is_shutdown = True
    executor.active_requests = []
    executor.waiting_queue = []
    executor.kv_connector_manager = None
    executor.enable_kv_pool_rebalance = True
    executor.control_requests = [SimpleNamespace(control_requires_drain=draining)]
    assert executor._num_execution_pending_releases() == 1
    assert not executor.should_stop_processing
    assert not executor._can_pause_for_rebalance()
    executor._handle_control_request()
    assert len(executor.control_requests) == 1
    guard._uses[id(request)].event.query.return_value = True
    guard.process_releases(executor._finish_terminated_request)
    assert executor.should_stop_processing


def test_retired_slots_count_against_fetch_admission_and_prevent_idle_wait(guard, monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor as module

    request, _, _ = _isolated_handoff(guard)
    guard.process_alignment()
    executor = _retiring_executor(guard, request)
    executor._do_terminate_request(request)
    executor.enable_attention_dp = False
    executor.enable_iter_perf_stats = False
    executor.num_fetch_requests = 0
    executor.dist = SimpleNamespace(cp_config=None, cp_rank=0, cp_size=1)
    executor._fetch_and_enqueue_requests = Mock()
    executor._pop_from_waiting_queue = Mock(return_value=[])
    executor._should_exclude_last_generation_logits = lambda: False
    monkeypatch.setattr(module, "merge_requests", lambda requests, **kwargs: requests)
    waiting = object()
    assert executor._fetch_new_requests(waiting, []) == []
    executor._fetch_and_enqueue_requests.assert_called_once_with(waiting, 1)
    executor._pop_from_waiting_queue.assert_called_once_with(waiting, 1, None)


def test_shutdown_closes_alignment_before_blocking_retired_cleanup():
    order = []
    executor = object.__new__(PyExecutor)
    executor.response_cv = Condition()
    executor.shutdown_event = Event()
    executor.num_micro_batches = 0
    executor._execution_guard = Mock()
    executor._execution_guard.close_alignment.side_effect = lambda: order.append("invalidate")
    executor._execution_guard.process_releases.side_effect = lambda *args, **kw: order.append(
        "drain"
    )
    executor._executor_loop_cleanup()
    assert order == ["invalidate", "drain"]
    executor._execution_guard.process_releases.assert_called_once_with(
        executor._finish_terminated_request, block=True
    )


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("mode", ["eager", "graph", "compiled", "compiled-graph"])
@pytest.mark.parametrize("common_prefix", [None, 0, 2])
def test_resume_handoff_before_common_forward_boundary(guard, v2, overlap, mode, common_prefix):
    from tensorrt_llm.execution_alignment import (
        AlignmentInput,
        AlignmentRequest,
        WorkerAlignmentState,
    )
    from tensorrt_llm.execution_authority import Revision
    from tensorrt_llm.execution_fence import RequestDrainTracker

    executor = _executor()
    executor._is_kv_manager_v2 = v2
    executor.disable_overlap_scheduler = not overlap
    executor.model_engine.cuda_graph_config = object() if "graph" in mode else None
    executor.model_engine.torch_compile_config = object() if "compiled" in mode else None
    ExecutionDeviceGuard.validate_executor(executor)
    native_hold = Mock(num_tokens=3)
    state = guard.table.alignment_state = WorkerAlignmentState(lambda _: native_hold, lambda: None)
    guard.table.drain_tracker = RequestDrainTracker()
    revised = AlignmentInput((1, 2, 3, 4, 5), cache_salt="cache")
    alignment = AlignmentRequest("session", Revision("edit", VERSION, NEXT), revised)
    guard.table.apply(VersionUpdate("session", 2, NEXT, alignment))
    state.process()
    receipt = guard.table.poll_alignment(alignment)
    guard.table.apply(VersionUpdate("session", 3, NEXT, alignment, receipt, common_prefix))
    request = _request(NEXT)
    request.get_tokens = Mock(return_value=list(revised.input_token_ids))
    request.cache_salt = revised.cache_salt
    request.lora_task_id = None
    guard.validate_successor(request)
    expected = 3 if common_prefix is None else common_prefix
    if common_prefix is not None:
        assert request.execution_reuse_limit == expected
    request.prepopulated_prompt_len = 2 if common_prefix is None else common_prefix + 1
    with pytest.raises(RuntimeError, match="KV prefix"):
        with guard.forward(_batch(request, context=True)):
            pytest.fail("must not submit the model without its prefix")
    guard.device.submission.assert_not_called()
    request.prepopulated_prompt_len = expected
    with guard.forward(_batch(request, context=True)):
        assert next(iter(state._records.values())).attached
    native_hold.settle.assert_called_once()
    native_hold.close.assert_called_once()
    # A later revision still cancels this resumed execution.
    later = replace(NEXT, epoch=9)
    guard.table.apply(VersionUpdate("session", 4, later))
    assert guard.revoke_obsolete_requests([request]) == [request]
    # A stale row racing submission is handled by normal fences, not a
    # batch-wide alignment exception that would poison unrelated requests.
    with guard.forward(_batch(request)):
        pass


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("prefix", [0, 4])
def test_confirmed_attachment_releases_hold_without_changing_reuse_limit(guard, v2, prefix):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings.executor import SamplingConfig
    from tensorrt_llm.execution_alignment import (
        AlignmentInput,
        AlignmentRequest,
        WorkerAlignmentState,
    )
    from tensorrt_llm.execution_authority import Revision

    if not v2:
        guard._kv_manager = None
    hold = Mock(num_tokens=prefix)
    state = guard.table.alignment_state = WorkerAlignmentState(lambda _: hold, lambda: None)
    guard.table.drain_tracker = RequestDrainTracker()
    revised = AlignmentInput(tuple(range(13)), cache_salt="cache")
    alignment = AlignmentRequest("session", Revision("edit", VERSION, NEXT), revised)
    guard.table.apply(VersionUpdate("session", 2, NEXT, alignment))
    state.process()
    receipt = guard.table.poll_alignment(alignment)
    guard.table.apply(VersionUpdate("session", 3, NEXT, alignment, receipt, prefix))
    request = LlmRequest(
        request_id=42,
        client_id=142,
        max_new_tokens=4,
        input_tokens=list(revised.input_token_ids),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        cache_salt="cache",
        execution_version=NEXT,
    )
    guard.validate_successor(request)
    assert request.execution_reuse_limit == prefix
    assert request.py_execution_recovery_pending
    assert request.py_execution_recovery_tokens == prefix
    # A mismatched first attachment must preserve the original constraint.
    request.set_prepopulated_prompt_len(prefix + 1, 4)
    with pytest.raises(RuntimeError, match="agreed KV prefix"):
        with guard.forward(_batch(request, context=True)):
            pytest.fail("the initial agreed depth must still be enforced")
    assert request.execution_reuse_limit == prefix
    hold.close.assert_not_called()

    request.set_prepopulated_prompt_len(prefix, 4)
    with guard.forward(_batch(request, context=True)):
        assert request.execution_reuse_limit == prefix
    hold.close.assert_called_once()
    # The C++ contract stays immutable, including a zero-token agreement.
    with pytest.raises(TypeError, match="incompatible function arguments"):
        request.execution_reuse_limit = None
    with pytest.raises(RuntimeError, match="Execution reuse limit cannot change"):
        request.execution_reuse_limit = 1 if prefix == 0 else 0
    # Later scheduling keeps the runtime bound without reacquiring the hold.
    request.set_prepopulated_prompt_len(prefix, 4)
    with guard.forward(_batch(request, context=True)):
        pass
    hold.close.assert_called_once()
    assert state.pending_prefix_length(NEXT, request.py_request_id) == 0
    assert state.pending_recovery_prefix(NEXT, request.py_request_id) is None
    guard.table.apply(VersionUpdate("session", 4, replace(NEXT, epoch=9)))
    assert guard.revoke_obsolete_requests([request]) == [request]


@pytest.mark.parametrize("v2", [False, True])
def test_eager_idle_iteration_drains_finished_revision(monkeypatch, v2) -> None:
    """The final revoked request must not need another GPU batch to drain."""
    from unittest.mock import MagicMock

    for name in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{name}", Mock())
    request = SimpleNamespace(is_finished=True, py_execution_version=VERSION)
    executor = SimpleNamespace(
        device_id=0,
        _profiler=lambda: nullcontext(Mock()),
        hang_detector=MagicMock(),
        enable_iter_perf_stats=False,
        is_benchmark_disagg=False,
        _resource_governor_enabled=False,
        _is_kv_manager_v2=v2,
        _can_pause_for_rebalance=lambda: False,
        _mm_encoder_item_scheduling_enabled=False,
        kv_connector_manager=None,
        active_requests=[request],
        _execution_guard=object(),
        iter_counter=0,
        disagg=Mock(),
        perf_manager=Mock(enabled=False),
    )
    batch = ScheduledRequests()
    executor._prepare_and_schedule_batch = Mock(side_effect=[(batch, None), (None, None)])
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._can_queue = Mock(return_value=(False, False))
    for name in (
        "_terminate_requests",
        "_pause_requests",
        "_release_unused_connector_reservations",
        "_revert_gen_alloc",
        "_finalize_adp_dummy_allocation",
        "_kv_connector_terminate_requests",
        "_flush_pending_transfer_responses",
        "_flush_iter_stats_synced",
        "_terminate_recompute_paused_requests",
        "_pause_recompute_paused_requests",
    ):
        setattr(executor, name, Mock())

    def drain(*, terminal_only):
        assert terminal_only
        executor.active_requests.clear()
        return [request]

    executor._handle_responses = Mock(side_effect=drain)
    PyExecutor._executor_loop(executor)
    assert not executor.active_requests
    executor._handle_responses.assert_called_once_with(terminal_only=True)
    assert executor._event_loop_completed


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("revoked", [False, True])
@pytest.mark.parametrize("early_emit", [False, True])
def test_overlap_idle_iteration_drains_terminal_request(monkeypatch, v2, revoked, early_emit):
    """Cancellation must free memory without waiting for a new model batch."""
    executor, terminal, live = _idle_overlap_executor(monkeypatch, v2, early_emit)
    terminal.is_finished = revoked
    executor._execution_guard = object() if revoked else None

    def cancel():
        terminal.is_finished = True

    def drain(*, terminal_only):
        assert terminal_only and terminal.is_finished
        executor.active_requests.remove(terminal)
        executor._enqueue_responses([terminal])
        return [terminal]

    executor._handle_canceled_requests.side_effect = cancel
    executor._handle_responses.side_effect = drain
    PyExecutor._executor_loop_overlap(executor)

    assert executor.active_requests == [live]
    executor._handle_canceled_requests.assert_called_once_with()
    executor._handle_responses.assert_called_once_with(terminal_only=True)
    expected = [([terminal],)]
    if early_emit:
        expected.insert(0, ([],))
    assert [call.args for call in executor._enqueue_responses.call_args_list] == expected
    executor._process_previous_batch.assert_not_called()
    assert executor._event_loop_completed


@pytest.mark.parametrize("pending_previous", [False, True])
def test_overlap_idle_peer_does_not_release_local_batch(monkeypatch, pending_previous):
    """A rank waiting for its peers still owns its scheduled resources."""
    executor, terminal, live = _idle_overlap_executor(monkeypatch, True, False)
    previous = object() if pending_previous else None
    executor.previous_batch = previous
    executor._can_queue.return_value = (False, True)

    PyExecutor._executor_loop_overlap(executor)

    assert executor.previous_batch is previous
    assert executor.active_requests == [terminal, live]
    executor._handle_canceled_requests.assert_not_called()
    executor._handle_responses.assert_not_called()
    executor._process_previous_batch.assert_not_called()
    executor._enqueue_responses.assert_called_once_with([])


def test_overlap_empty_batch_consumes_final_sample_before_release(monkeypatch):
    """Capacity deferral must drain the final sample without another forward."""
    executor, terminal, live = _idle_overlap_executor(monkeypatch, True, False)
    terminal.is_finished = False
    sample = object()
    scheduled = ScheduledRequests()
    scheduled.generation_requests = [terminal]
    previous = SimpleNamespace(
        sample_state=sample,
        scheduled_requests=scheduled,
    )
    executor.previous_batch = previous
    executor.speculation_gate = None
    executor._send_kv_async = Mock()
    executor._commit_kv_cache_stats = Mock()
    order = []

    def accept(state):
        assert state is sample and terminal in executor.active_requests
        terminal.is_finished = True
        order.append("accepted")

    def release():
        assert terminal.is_finished and order == ["accepted"]
        executor.active_requests.remove(terminal)
        order.append("released")

    executor._update_requests = Mock(side_effect=accept)
    executor._process_previous_batch.side_effect = release

    PyExecutor._executor_loop_overlap(executor)

    assert order == ["accepted", "released"]
    assert executor.active_requests == [live]
    assert executor.previous_batch is None
    executor._update_requests.assert_called_once_with(sample)
    executor._process_previous_batch.assert_called_once_with()
    executor._handle_responses.assert_not_called()
    assert executor._event_loop_completed


def _idle_overlap_executor(monkeypatch, v2, early_emit):
    from unittest.mock import MagicMock

    for name in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{name}", Mock())
    terminal = SimpleNamespace(is_finished=True)
    live = SimpleNamespace(is_finished=False)
    executor = SimpleNamespace(
        device_id=0,
        _profiler=lambda: nullcontext(Mock()),
        hang_detector=MagicMock(),
        enable_iter_perf_stats=False,
        is_benchmark_disagg=False,
        _resource_governor_enabled=False,
        _is_kv_manager_v2=v2,
        _can_pause_for_rebalance=lambda: False,
        _mm_encoder_item_scheduling_enabled=False,
        kv_connector_manager=None,
        active_requests=[terminal, live],
        iter_counter=0,
        disagg=Mock(),
        perf_manager=Mock(enabled=False),
        previous_batch=None,
        drafter=None,
        enable_early_first_token_response=early_emit,
    )
    executor._prepare_and_schedule_batch = Mock(
        side_effect=[(ScheduledRequests(), None), (None, None)]
    )
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._can_queue = Mock(return_value=(False, False))
    for name in (
        "_wait_for_model_engine_input_copy",
        "_terminate_requests",
        "_pause_requests",
        "_release_unused_connector_reservations",
        "_revert_gen_alloc",
        "_finalize_adp_dummy_allocation",
        "_kv_connector_terminate_requests",
        "_flush_pending_transfer_responses",
        "_flush_iter_stats_synced",
        "_terminate_recompute_paused_requests",
        "_pause_recompute_paused_requests",
        "_handle_canceled_requests",
        "_handle_responses",
        "_enqueue_responses",
        "_process_previous_batch",
    ):
        setattr(executor, name, Mock())
    return executor, terminal, live


def test_terminal_only_response_pass_preserves_live_request() -> None:
    live = SimpleNamespace(py_request_id=1, is_finished=False)
    executor = SimpleNamespace(
        active_requests=[live],
        dist=SimpleNamespace(rank=0),
        perf_manager=Mock(),
        disagg=Mock(),
        _enqueue_responses=Mock(),
        _terminate_request=Mock(),
    )
    assert PyExecutor._handle_responses(executor, terminal_only=True) == []
    assert executor.active_requests == [live]
    executor._enqueue_responses.assert_called_once_with([])
    executor._terminate_request.assert_not_called()
    executor.perf_manager.append_step_metrics.assert_not_called()


@pytest.mark.parametrize("neighbor", [None, "live", "untagged"])
def test_operator_exit_requires_every_row_to_lose_authority(guard, neighbor):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        execution_early_exit_point,
    )

    request = _request()
    requests = [request]
    if neighbor == "live":
        version = replace(VERSION, scope_id="neighbor")
        guard.table.apply(VersionUpdate("session", 1, version))
        requests.append(_request(version))
    elif neighbor == "untagged":
        requests.append(_request(None))
    with guard.forward(_batch(*requests, context=True)):
        execution_early_exit_point()
        guard.table.apply(VersionUpdate("session", 2, NEXT))
        if neighbor is None:
            with pytest.raises(ExecutionEarlyExit):
                execution_early_exit_point()
        else:
            execution_early_exit_point()
    # An early exit is not a release: every submitted use still needs a wait.
    assert guard._uses[id(request)].event is not None
    guard.device.wait_event.assert_not_called()
    execution_early_exit_point()  # No batch callback leaks into the next forward.


def test_operator_exit_unwinds_and_records_partial_submission(guard):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        execution_early_exit_point,
    )

    request = _request()
    with pytest.raises(ExecutionEarlyExit):
        with guard.forward(_batch(request, context=True)):
            guard.table.apply(VersionUpdate("session", 2, NEXT))
            execution_early_exit_point()
    assert guard._uses[id(request)].event is not None
    assert guard._kv_frontiers[id(request)] == 0
    execution_early_exit_point()


def test_disabling_early_exit_keeps_stale_commit_gates_and_completion_wait(guard):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        execution_early_exit_point,
        get_layer_early_exit_point,
    )

    guard._early_exit_enabled = False
    request = _request()
    batch = _batch(request, context=True)
    with guard.forward(batch):
        guard.table.apply(VersionUpdate("session", 2, NEXT))
        assert get_layer_early_exit_point() is None
        execution_early_exit_point("eager_layer", 4, 64)
    event = guard._uses[id(request)].event
    assert event is not None
    guard.device.wait_event.assert_not_called()
    # A previously accepted device result still cannot commit under old host authority.
    with guard.token_commit(_state(request, allowed=[True])):
        assert request.is_finished
    with guard.context_commit(batch) as accepted:
        assert not accepted.context_requests
    guard.before_release(request)
    guard.device.wait_event.assert_called_once_with(event)


def test_boundary_observer_tracks_verdict_and_restores_after_unwind() -> None:
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        early_exit_scope,
        execution_early_exit_point,
    )

    outer, inner = Mock(), Mock()
    with early_exit_scope(lambda: False, observer=outer):
        execution_early_exit_point("eager_layer", 4, 16)
        with pytest.raises(ExecutionEarlyExit):
            with early_exit_scope(lambda: True, observer=inner):
                execution_early_exit_point("eager_layer", 8, 16)
        execution_early_exit_point("eager_layer", 12, 16)
    execution_early_exit_point()
    assert [c.args for c in outer.call_args_list] == [
        ("eager_layer", 4, 16, False),
        ("eager_layer", 12, 16, False),
    ]
    inner.assert_called_once_with("eager_layer", 8, 16, True)


@pytest.mark.parametrize("phase", ["schedule", "context_commit", "token_commit"])
def test_remote_rejection_prevents_local_progress(guard, phase):
    request = _request()
    calls = []

    def gather(vote):
        calls.append(vote)
        if phase == "token_commit":
            # GPU completion must precede the host collective.
            state.execution_ticket.wait.assert_called_once()
            assert state.sampler_event is None
        return [vote, replace(vote, rank=1, allowed=(False,))]

    guard._rank_consensus = ExecutionStepConsensus("session", 0, (0, 1), gather)
    if phase == "schedule":
        assert guard.revoke_obsolete_requests([request]) == [request]
    elif phase == "context_commit":
        with guard.context_commit(_batch(request, context=True)) as accepted:
            assert not accepted.context_requests
    else:
        state = _state(request, allowed=[True])
        with guard.token_commit(state):
            assert request.is_finished
    assert request.is_finished
    assert not guard._kv_frontiers
    expected = [phase, phase + "_done"]
    if phase == "token_commit":
        expected.append("token_prefix_done")
    assert [v.phase for v in calls] == expected
    assert calls[0].rows[0].version == VERSION
    assert guard.table.current_version("tenant", "scope") == VERSION


def test_remote_early_exit_uses_exact_boundary_and_records_local_readers(guard):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        execution_early_exit_point,
        get_layer_early_exit_point,
    )

    calls = []

    def gather(vote):
        calls.append(vote)
        return [vote, replace(vote, rank=1, allowed=(False,))]

    guard._rank_consensus = ExecutionStepConsensus("session", 0, (0, 1), gather)
    request = _request()
    with pytest.raises(ExecutionEarlyExit):
        with guard.forward(_batch(request, context=True)):
            assert get_layer_early_exit_point() is not None
            execution_early_exit_point("decoder_layer", 3, 64)
    assert [v.phase for v in calls] == [
        "forward_entry:early_exit=1",
        "early_exit:decoder_layer:3:64",
        "forward_done:early_exit:decoder_layer:3:64",
    ]
    assert guard._uses[id(request)].event is not None
    guard.device.wait_event.assert_not_called()
    assert get_layer_early_exit_point() is None
    execution_early_exit_point()


@pytest.mark.parametrize("untagged", [False, True])
def test_coordinated_entry_includes_empty_and_untagged_batches(guard, untagged):
    votes = []

    def gather(vote):
        votes.append(vote)
        return [vote, replace(vote, rank=1)]

    guard._rank_consensus = ExecutionStepConsensus("session", 0, (0, 1), gather)
    requests = [_request(None)] if untagged else []
    assert guard.revoke_obsolete_requests(requests) == []
    assert [v.phase for v in votes] == ["schedule", "schedule_done"]
    assert all(len(v.rows) == len(requests) for v in votes)


def test_boundary_poll_restores_outer_authority_on_exception() -> None:
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        early_exit_scope,
        execution_early_exit_point,
    )

    outer = Mock(return_value=False)
    inner = Mock(return_value=True)
    with early_exit_scope(None, boundary_poll=outer):
        execution_early_exit_point("outer", 0, 1)
        with pytest.raises(ExecutionEarlyExit):
            with early_exit_scope(None, boundary_poll=inner):
                execution_early_exit_point("inner", 2, 4)
        execution_early_exit_point("outer", 1, 1)
        with pytest.raises(ValueError, match="one early-exit"):
            with early_exit_scope(lambda: False, boundary_poll=inner):
                pytest.fail("ambiguous authority accepted")
    assert [call.args for call in outer.call_args_list] == [("outer", 0, 1), ("outer", 1, 1)]
    inner.assert_called_once_with("inner", 2, 4)


def test_prefill_trace_carries_request_identity_and_partial_submission(guard, monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        execution_early_exit_point,
    )

    path = "tensorrt_llm._torch.pyexecutor.execution_guard"
    monkeypatch.setattr(path + ".execution_trace_enabled", lambda: True)
    trace = Mock()
    monkeypatch.setattr(path + ".trace_execution", trace)
    request = _request()
    with pytest.raises(ExecutionEarlyExit):
        with guard.forward(_batch(request, context=True)):
            guard.table.apply(VersionUpdate("session", 2, NEXT))
            execution_early_exit_point("eager_layer", 4, 16)
    assert [c.args[0] for c in trace.call_args_list] == ["forward_begin", "boundary", "forward_end"]
    assert trace.call_args_list[1].kwargs["obsolete"]
    assert trace.call_args.kwargs["completed"] is False
    metadata = trace.call_args.kwargs["requests"][0]
    assert metadata["client_id"] == request.py_client_id
    assert metadata["epoch"] == VERSION.epoch


@pytest.mark.parametrize("completed", [0, 16, 35])
def test_v1_reserves_completed_chunks_before_probing_tree(guard, completed):
    from tensorrt_llm.execution_alignment import AlignmentInput

    request = _request()
    request.freeze_retire_kv_frontier = Mock()
    guard._v1_manager = Mock(tokens_per_block=16, enable_block_reuse=True)
    guard._kv_manager = None
    guard._alignment_adapter = Mock()
    batch = _batch(request, context=True)
    with guard.forward(batch):
        pass
    with guard.context_commit(batch):
        request.context_current_position = completed
    # A later chunk was submitted but only partially executed at revocation.
    with guard.forward(batch):
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.context_commit(batch):
        pass
    order = []
    guard.device.wait_event.side_effect = lambda event: order.append("drain")
    request.freeze_retire_kv_frontier.side_effect = lambda count: order.append(("freeze", count))
    guard._v1_manager.store_blocks_for_reuse.side_effect = lambda req: order.append("publish")
    from tensorrt_llm.execution_alignment import AlignmentRequest, WorkerAlignmentState
    from tensorrt_llm.execution_authority import Revision

    alignment = AlignmentRequest(
        "session", Revision("edit", VERSION, NEXT), AlignmentInput(tuple(range(64)))
    )
    guard._alignment_adapter.reserve.side_effect = lambda revised: (
        order.append("reserve") or Mock(num_tokens=completed // 16 * 16)
    )
    state = WorkerAlignmentState(
        guard._alignment_adapter.reserve, lambda: None, prepare=guard._prepare_prefix
    )
    state.observe(NEXT, alignment)
    state.process()
    expected = ["drain", ("freeze", completed // 16 * 16)]
    if completed >= 16:
        expected.append("publish")
    assert order == expected + ["reserve"]
    assert id(request) in guard._uses  # Old ownership survives until cleanup.


def test_failed_drain_cannot_publish_or_reserve_prefill(guard):
    from tensorrt_llm.execution_alignment import AlignmentInput

    request = _request()
    request.freeze_retire_kv_frontier = Mock()
    guard._v1_manager = Mock(tokens_per_block=16)
    guard._alignment_adapter = Mock()
    with guard.forward(_batch(request)):
        pass
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    guard.device.wait_event.side_effect = RuntimeError("unknown completion")
    from tensorrt_llm.execution_alignment import AlignmentRequest
    from tensorrt_llm.execution_authority import Revision

    alignment = AlignmentRequest("session", Revision("edit", VERSION, NEXT), AlignmentInput((1, 2)))
    with pytest.raises(RuntimeError, match="source KV preparation failed"):
        guard._prepare_prefix(alignment)
    request.freeze_retire_kv_frontier.assert_not_called()
    guard._v1_manager.store_blocks_for_reuse.assert_not_called()
    guard._alignment_adapter.reserve.assert_not_called()
    assert id(request) in guard._uses


def test_early_exit_does_not_call_sampler_update(guard):
    executor = object.__new__(PyExecutor)
    executor._execution_guard = guard
    executor.sampler = Mock()
    executor._update_requests(None)
    executor.sampler.update_requests.assert_not_called()


@pytest.mark.parametrize("compiled,capturing", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("family", ["qwen3", "qwen2", "llama"])
@pytest.mark.parametrize("armed", [True, False])
def test_layer_hook_skips_only_unsubmitted_eager_layers(
    monkeypatch, compiled, capturing, family, armed
):
    from tensorrt_llm._torch.models import modeling_llama, modeling_qwen, modeling_qwen3
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        early_exit_scope,
    )

    calls = []
    monkeypatch.setattr("torch.compiler.is_compiling", lambda: compiled)
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(modeling_qwen3, "maybe_allgather_for_helix_cp", lambda h, *args: h)

    def layer(**kwargs):
        calls.append(len(calls))
        return kwargs["hidden_states"], kwargs["residual"]

    model = SimpleNamespace(
        layers=[layer] * 12,
        norm=lambda h, r: (h, r),
        mapping_with_cp=None,
        num_hidden_layers=12,
        skip_norm=False,
    )
    forward = {
        "qwen3": modeling_qwen3.Qwen3Model.forward,
        "qwen2": modeling_qwen.QwenModel.forward,
        "llama": modeling_llama.LlamaModel.forward,
    }[family]
    if family == "llama":
        model.norm = lambda h: h
        # Configured depth can be smaller than the module list.
        model.layers.append(layer)
    with early_exit_scope(lambda: bool(calls), layer_checks=armed):
        if compiled or capturing or not armed:
            forward(model, None, inputs_embeds=object())
            assert len(calls) == 12
        else:
            with pytest.raises(ExecutionEarlyExit) as stopped:
                forward(model, None, inputs_embeds=object())
            assert len(calls) == 4
            assert stopped.value.boundary == "eager_layer"
            assert stopped.value.submitted == 4
            assert stopped.value.total == 12


def test_breakable_graph_exit_skips_remaining_segments_and_restores_context(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
        BreakableCUDAGraph,
        get_current_replay_token,
    )
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        early_exit_scope,
    )

    monkeypatch.setattr("torch.cuda.current_stream", Mock())
    graph = BreakableCUDAGraph()
    graph._segments = [Mock(), Mock(), Mock()]
    graph._break_functions = [Mock(), Mock()]
    with early_exit_scope(lambda: graph._segments[0].replay.called):
        with pytest.raises(ExecutionEarlyExit):
            graph.replay()
    graph._segments[0].replay.assert_called_once()
    graph._segments[1].replay.assert_not_called()
    graph._segments[2].replay.assert_not_called()
    assert get_current_replay_token() is None
    graph.replay()  # Later unarmed replays must run the entire graph.
    graph._segments[2].replay.assert_called_once()


def test_alignment_does_not_wait_for_another_scope(guard):
    from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentRequest
    from tensorrt_llm.execution_authority import Revision

    other_version = replace(VERSION, scope_id="other")
    other = _request(other_version)
    guard.table.apply(VersionUpdate("session", 1, other_version))
    with guard.forward(_batch(other)):
        pass
    guard.table.apply(VersionUpdate("session", 2, replace(other_version, epoch=8)))
    alignment = AlignmentRequest("session", Revision("edit", VERSION, NEXT), AlignmentInput((1, 2)))
    guard._prepare_prefix(alignment)
    guard.device.wait_event.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()


def test_executor_treats_early_exit_as_revocation_not_model_error(guard, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import execution_early_exit_point

    request = _request()
    batch = _batch(request, context=True)
    stream = Mock()
    monkeypatch.setattr("torch.cuda.stream", lambda stream: nullcontext())
    monkeypatch.setattr("torch.cuda.current_stream", lambda: stream)

    def forward(*args, **kwargs):
        guard.table.apply(VersionUpdate("session", 2, NEXT))
        execution_early_exit_point()
        pytest.fail("revoked model continued")

    executor = SimpleNamespace(
        _execution_guard=guard,
        model_engine=SimpleNamespace(forward=forward),
        sampler=Mock(),
        execution_stream=stream,
        resource_manager=Mock(),
        iter_counter=1,
        _iter_adp_dummy_ctx_tokens=0,
        _iter_adp_dummy_gen_tokens=0,
        _compute_adp_dummy_tokens=lambda batch: (0, 0),
        _attach_encoder_output_to_execution_stream=Mock(),
        _maybe_record_hang_diagnostic_phase=Mock(),
        _handle_errors=Mock(),
    )
    assert PyExecutor._forward_step(executor, batch) is None
    assert request.is_finished
    assert guard._uses[id(request)].event is not None
    executor._handle_errors.assert_not_called()
    guard.device.wait_event.assert_not_called()  # Cleanup must still drain it.


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("finished", [False, True])
def test_overlap_accepts_missing_sample_only_for_fully_revoked_batch(monkeypatch, v2, finished):
    from unittest.mock import MagicMock

    class ReachedCommit(Exception):
        pass

    for name in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{name}", Mock())
    request = _request()
    request.is_finished = finished
    batch = _batch(request, context=True)
    executor = MagicMock()
    for flag in (
        "enable_iter_perf_stats",
        "is_benchmark_disagg",
        "_resource_governor_enabled",
        "_mm_encoder_item_scheduling_enabled",
        "use_spec_decode",
        "has_previous_draft_tokens",
        "enable_early_first_token_response",
    ):
        setattr(executor, flag, False)
    for field in (
        "kv_connector_manager",
        "kv_cache_transceiver",
        "drafter",
        "guided_decoder",
        "previous_batch",
        "speculation_gate",
    ):
        setattr(executor, field, None)
    executor._is_kv_manager_v2 = v2
    executor._can_pause_for_rebalance.return_value = False
    executor._profiler.return_value = nullcontext(Mock())
    executor._prepare_and_schedule_batch.return_value = (batch, None)
    executor._check_benchmark_disagg_gate.return_value = (True, False)
    executor._can_queue.return_value = (True, True)
    executor._forward_step.return_value = None
    executor._sample_async.return_value = None
    executor.perf_manager.create_timing_events.return_value = (None, None, None)
    executor._update_request_states.side_effect = ReachedCommit
    with pytest.raises(ReachedCommit if finished else AssertionError):
        PyExecutor._executor_loop_overlap(executor)
    executor._sample_async.assert_called_once_with(batch, None)


def test_v1_old_native_wheel_rejected_before_device_setup(monkeypatch):
    import tensorrt_llm._torch.pyexecutor.execution_guard as module

    monkeypatch.setattr(module, "LlmRequest", type("OldRequest", (), {}))
    device = Mock()
    monkeypatch.setattr(module, "DeviceExecutionAuthority", device)
    with pytest.raises(RuntimeError, match="rebuilt native bindings"):
        ExecutionDeviceGuard(_executor(), WorkerVersionTable("session"))
    device.assert_not_called()


def test_normal_v1_completion_keeps_native_generated_prefix_reuse(guard):
    request = _request()
    request.freeze_retire_kv_frontier = Mock()
    guard._v1_manager = Mock(tokens_per_block=16)
    guard._kv_manager = None
    with guard.forward(_batch(request)):
        pass
    request.num_tokens = 81
    guard._kv_frontiers[id(request)] = 80
    request.is_finished = True
    request.context_remaining_length = 0
    guard.before_release(request)
    guard.device.wait_event.assert_called_once()
    request.freeze_retire_kv_frontier.assert_called_once_with(80)
    guard._v1_manager.store_blocks_for_reuse.assert_called_once_with(request)


@pytest.mark.parametrize("before", [64, 79, 80, 81])
@pytest.mark.parametrize("device_ok,host_ok", [(True, True), (True, False), (False, True)])
def test_decode_frontier_requires_both_verdicts_and_excludes_sample(
    guard, before, device_ok, host_ok
):
    request = _request()
    request.num_tokens = before
    request.context_current_position = 64
    request.context_remaining_length = 0
    guard._kv_frontiers[id(request)] = 64
    state = _state(request, allowed=[device_ok])
    if not host_ok:
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.token_commit(state):
        if not request.is_finished:
            request.num_tokens += 1
    assert guard._kv_frontiers[id(request)] == (before if device_ok and host_ok else 64)
    assert guard._kv_frontiers[id(request)] < request.num_tokens or not (device_ok and host_ok)


@pytest.mark.parametrize("delta,remaining", [(2, 0), (-1, 0), (1, 16)])
def test_unsupported_token_progress_cannot_enlarge_frontier(guard, delta, remaining):
    request = _request()
    request.num_tokens = 80
    request.context_remaining_length = remaining
    guard._kv_frontiers[id(request)] = 64
    with pytest.raises(RuntimeError, match="one accepted token"):
        with guard.token_commit(_state(request, allowed=[True])):
            request.num_tokens += delta
    assert guard._kv_frontiers[id(request)] == 64
    guard._kv_manager.preserve_execution_prefix.assert_not_called()


def test_sampler_exception_does_not_accept_partially_mutated_token_state(guard):
    request = _request()
    request.num_tokens = 80
    request.context_remaining_length = 0
    guard._kv_frontiers[id(request)] = 64
    with pytest.raises(ValueError, match="sampler failed"):
        with guard.token_commit(_state(request, allowed=[True])):
            request.num_tokens += 1
            raise ValueError("sampler failed")
    assert guard._kv_frontiers[id(request)] == 64


@pytest.mark.parametrize("matched", [0, 1, 3, 4])
@pytest.mark.parametrize("revoked", [False, True])
def test_ngram_frontier_excludes_rejected_drafts_and_bonus_sample(guard, matched, revoked):
    guard._speculative_tokens_per_step = 5
    request = _request()
    request.num_tokens = 80
    request.context_remaining_length = 0
    request.py_draft_tokens = [101, 102, 103, 104]
    request.get_token = lambda beam, position: ([7] * 80 + [101, 102, 103, 104, 999])[position]
    guard._kv_frontiers[id(request)] = 64
    if revoked:
        guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.token_commit(_state(request, allowed=[True])):
        if not request.is_finished:
            request.py_num_accepted_draft_tokens = matched
            request.num_tokens += matched + 1
    assert guard._kv_frontiers[id(request)] == (64 if revoked else 80 + matched)
    if not revoked:
        assert guard._uses[id(request)].request is request
        guard._uses[id(request)].event.record.assert_called_once()


@pytest.mark.parametrize(
    "fault", ["wrong_token", "wrong_count", "too_many", "sampler_error", "event_error"]
)
def test_ngram_verification_failure_cannot_preserve_candidate_kv(guard, fault, monkeypatch):
    guard._speculative_tokens_per_step = 5
    request = _request()
    request.num_tokens = 80
    request.context_remaining_length = 0
    request.py_draft_tokens = [101, 102, 103, 104]
    request.get_token = lambda beam, position: (
        999 if fault == "wrong_token" else 101 + position - 80
    )
    guard._kv_frontiers[id(request)] = 64
    if fault == "event_error":
        monkeypatch.setattr("torch.cuda.Event", Mock(side_effect=RuntimeError("event failed")))
    with pytest.raises(RuntimeError):
        with guard.token_commit(_state(request, allowed=[True])):
            request.py_num_accepted_draft_tokens = 3 if fault == "wrong_count" else 2
            request.num_tokens += 6 if fault == "too_many" else 3
            if fault == "sampler_error":
                raise RuntimeError("verification failed")
    assert guard._kv_frontiers[id(request)] == 64
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    if fault == "event_error":
        assert guard._uses[id(request)].event is None


@pytest.mark.parametrize(
    "fault", [None, "public_pool", "draft_engine", "forced_acceptance", "overlap"]
)
def test_only_private_ngram_without_draft_model_can_enable_retire(fault):
    from tensorrt_llm._torch.speculative.ngram import NGramDrafter

    executor = _executor()
    executor.disable_overlap_scheduler = fault != "overlap"
    executor.draft_model_engine = object() if fault == "draft_engine" else None
    executor.model_engine.is_spec_decode = True
    executor.model_engine.spec_config = SimpleNamespace(
        decoding_type="NGram", is_public_pool=fault == "public_pool"
    )
    executor.drafter = object.__new__(NGramDrafter)
    executor.drafter.spec_resource_manager = SimpleNamespace(is_public_pool=fault == "public_pool")
    executor.sampler._force_num_accepted_tokens = 1 if fault == "forced_acceptance" else 0
    if fault is None:
        ExecutionDeviceGuard.validate_executor(executor)
    else:
        with pytest.raises(NotImplementedError, match="speculative"):
            ExecutionDeviceGuard.validate_executor(executor)


@pytest.mark.parametrize("matched", [0, 1, 4])
@pytest.mark.parametrize("stop_after", [None, 1, 2, 4])
def test_ngram_frontier_with_real_verification_and_early_finish(guard, matched, stop_after):
    from tensorrt_llm._torch.pyexecutor.sampler.two_model_spec_dec import TwoModelSpecDecHandler

    guard._speculative_tokens_per_step = 5
    request = _request()
    request.context_remaining_length = 0
    request.py_draft_tokens = [101, 102, 103, 104]
    tokens = [7] * 80
    request.get_num_tokens = lambda beam: len(tokens)
    request.get_token = lambda beam, position: tokens[position]
    request.add_new_token = lambda token, beam: tokens.append(token)
    samples = [101, 102, 103, 104, 999]
    samples[matched] = 999

    def finish(*args, **kwargs):
        if stop_after is not None and len(tokens) - 80 == stop_after:
            request.is_finished = True
            request.state = LlmRequestState.GENERATION_COMPLETE
            return True
        return False

    sampler = SimpleNamespace(_force_num_accepted_tokens=0, finish_if_reason=finish)
    handler = TwoModelSpecDecHandler(sampler)
    with guard.forward(_batch(request)):
        pass
    with guard.token_commit(_state(request, allowed=[True])):
        request.py_num_accepted_draft_tokens = handler._process_draft_tokens_greedy(
            request,
            new_tokens=[[[token]] * 3 for token in samples],
            finish_reasons=[],
        )
    count = min(matched + 1, stop_after or 5)
    assert len(tokens) == 80 + count
    assert guard._kv_frontiers[id(request)] == 80 + count - 1
    assert tokens[80:] == samples[:count]


@pytest.mark.parametrize("settle", ["revision", "release"])
def test_v2_decode_publication_defers_without_losing_the_accepted_frontier(guard, settle):
    request = _request()
    request.context_current_position = 64
    request.context_remaining_length = 0
    request.num_tokens = 80
    with guard.forward(_batch(request)):
        pass
    latest = guard._uses[id(request)].event
    latest.query.return_value = False
    guard._kv_manager.needs_execution_prefix_commit.return_value = True

    with guard.token_commit(_state(request, allowed=[True])):
        request.num_tokens += 1

    assert guard._kv_frontiers[id(request)] == 80
    guard.device.wait_event.assert_not_called()
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    if settle == "revision":
        guard.table.apply(VersionUpdate("session", 2, NEXT))
        guard._prepare_prefix(SimpleNamespace(revision=SimpleNamespace(expected_version=VERSION)))
    else:
        guard.before_release(request)
    guard.device.wait_event.assert_called_once_with(latest)
    guard._kv_manager.preserve_execution_prefix.assert_called_once_with(request, 80)
    guard._kv_manager.stop_request_committing.assert_called_once_with(request)


@pytest.mark.parametrize("cache_state", ["absent", "suspended"])
def test_v2_recompute_cleanup_drains_without_publishing_an_old_frontier(guard, cache_state):
    request = _request()
    request.context_current_position = 64
    request.context_remaining_length = 0
    request.num_tokens = 81
    with guard.forward(_batch(request)):
        pass
    latest = guard._uses[id(request)].event
    guard._kv_frontiers[id(request)] = 80
    cache = Mock(is_active=False)
    other = Mock()
    guard._kv_manager.kv_cache_map = {43: other}
    if cache_state == "suspended":
        guard._kv_manager.kv_cache_map[request.py_request_id] = cache
    # The scheduler has rolled back the first context admission before the
    # executor performs the remaining resource cleanup for recompute.
    request.context_current_position = 0
    request.context_remaining_length = 81

    guard.before_release(request)

    guard.device.wait_event.assert_called_once_with(latest)
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_called_once_with(request)
    assert id(request) not in guard._uses
    assert id(request) not in guard._kv_frontiers
    assert other.mock_calls == []
    assert cache.mock_calls == []
    request.prepopulated_prompt_len = 16
    with guard.forward(_batch(request)):
        pass
    assert guard._kv_frontiers[id(request)] == 16


@pytest.mark.parametrize("cancel_after_consume", [False, True])
def test_pending_token_after_recompute_release_cannot_restore_old_frontier(
    guard, cancel_after_consume
):
    request = _request()
    request.context_current_position = 64
    request.context_remaining_length = 0
    request.num_tokens = 80
    with guard.forward(_batch(request)):
        pass
    with guard.sampling(_batch(request)):
        state = _state(request, allowed=[True])
    latest = guard._uses[id(request)].event
    guard._kv_frontiers[id(request)] = 79
    guard._kv_manager.kv_cache_map = {request.py_request_id: Mock(is_active=False)}

    # The overlap executor frees a recompute victim before consuming the
    # previous batch's sample, then resets the request's context cursor.
    guard.before_release(request)
    guard._kv_manager.kv_cache_map.clear()
    guard.device.wait_event.assert_called_once_with(latest)
    guard._kv_manager.needs_execution_prefix_commit.return_value = True
    with guard.token_commit(state):
        request.num_tokens += 1
    assert request.num_tokens == 81
    assert id(request) not in guard._kv_frontiers
    assert id(request) not in guard._uses
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    request.context_current_position = 0
    request.context_remaining_length = 81
    if cancel_after_consume:
        request.finish_by_reason(FinishReason.CANCELLED)
    guard.before_release(request)
    guard.device.wait_event.assert_called_once_with(latest)
    if not cancel_after_consume:
        request.prepopulated_prompt_len = 16
        with guard.forward(_batch(request, context=True)):
            pass
        assert guard._kv_frontiers[id(request)] == 16


def test_v2_active_cache_cleanup_still_rejects_an_invalid_frontier(guard):
    request = _request()
    request.context_current_position = 0
    request.context_remaining_length = 64
    with guard.forward(_batch(request)):
        pass
    guard._kv_frontiers[id(request)] = 64
    guard._kv_manager.kv_cache_map = {request.py_request_id: Mock(is_active=True)}
    guard._kv_manager.preserve_execution_prefix.side_effect = (
        KVCacheManagerV2.preserve_execution_prefix.__get__(guard._kv_manager)
    )

    with pytest.raises(ValueError, match="completed frontier"):
        guard.before_release(request)

    guard._kv_manager.stop_request_committing.assert_not_called()
    assert id(request) in guard._uses
    assert guard._kv_frontiers[id(request)] == 64


def test_v2_decode_commit_drains_latest_overlap_reader_without_locking_publisher(guard):
    request = _request()
    request.context_current_position = 64
    request.context_remaining_length = 0
    request.num_tokens = 80
    with guard.forward(_batch(request)):
        pass
    with guard.forward(_batch(request)):
        pass
    latest = guard._uses[id(request)].event
    guard._kv_manager.needs_execution_prefix_commit.return_value = True
    order = []

    def wait(event):
        assert event is latest
        order.append("drain")
        published = Event()

        def publish():
            guard.table.apply(VersionUpdate("session", 2, NEXT))
            published.set()

        worker = Thread(target=publish, daemon=True)
        worker.start()
        assert published.wait(2), "GPU wait held the host authority lock"
        worker.join(2)

    guard.device.wait_event.side_effect = wait
    guard._kv_manager.preserve_execution_prefix.side_effect = lambda req, end: order.append(end)
    with guard.token_commit(_state(request, allowed=[True])):
        request.num_tokens += 1
    assert order == ["drain", 80]
    assert guard._kv_frontiers[id(request)] == 80
    # The next sample cannot turn drained-but-revoked work into accepted KV.
    with guard.token_commit(_state(request, allowed=[True])):
        assert request.is_finished
    assert guard._kv_frontiers[id(request)] == 80


@pytest.mark.parametrize("failure", ["missing", "unknown", "wrong_owner", "wait"])
def test_v2_decode_commit_requires_owned_gpu_completion(guard, failure):
    request = _request()
    request.context_remaining_length = 0
    request.num_tokens = 80
    with guard.forward(_batch(request)):
        pass
    guard._kv_manager.needs_execution_prefix_commit.return_value = True
    if failure == "missing":
        guard._uses.clear()
    elif failure == "unknown":
        guard._uses[id(request)].event = None
    elif failure == "wrong_owner":
        guard._uses[id(request)].request = _request()
    else:
        guard.device.wait_event.side_effect = RuntimeError("incomplete")
    with pytest.raises(RuntimeError):
        with guard.token_commit(_state(request, allowed=[True])):
            request.num_tokens += 1
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()


@pytest.mark.parametrize("frontier,expected", [(64, 64), (79, 64), (80, 80)])
def test_v2_decode_prefix_preserves_full_blocks_after_prompt(frontier, expected):
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = manager._can_publish_block_reuse = True
    manager.tokens_per_block = 16
    request = _request()
    request.context_current_position = 64
    request.context_remaining_length = 0
    request.num_tokens = 81
    request.is_dummy_request = False
    cache = Mock(is_active=True, is_commit_allowed=True, capacity=96, num_committed_tokens=64)
    manager.kv_cache_map = {42: cache}
    manager._reuse_token_source = Mock(return_value=list(range(81)))
    manager._augment_tokens_for_block_reuse = lambda tokens, req, start, end: tokens[start:end]

    def commit(tokens):
        cache.num_committed_tokens += len(tokens)

    cache.commit.side_effect = commit
    manager.preserve_execution_prefix(request, frontier)
    manager.preserve_execution_prefix(request, frontier)
    assert cache.num_committed_tokens == expected
    if expected > 64:
        cache.commit.assert_called_once_with(list(range(64, expected)))
    else:
        cache.commit.assert_not_called()
    with pytest.raises(ValueError, match="completed frontier"):
        manager.preserve_execution_prefix(request, 81)


@pytest.mark.parametrize("version", [None, VERSION])
def test_v2_context_end_leaves_only_tagged_request_open_for_decode_commits(version):
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = manager._can_publish_block_reuse = True
    request = _request(version)
    request.is_dummy_request = False
    request.context_current_position = 64
    request.context_remaining_length = 0
    cache = Mock(num_committed_tokens=64)
    manager.kv_cache_map = {42: cache}
    manager.try_commit_blocks(request)
    assert cache.stop_committing.call_count == int(version is None)
    cache.commit.assert_not_called()


def test_context_publication_does_not_reopen_cache_stopped_by_decode_commit():
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = manager._can_publish_block_reuse = True
    request = _request()
    request.is_dummy_request = False
    request.context_current_position = 65
    request.context_remaining_length = 0
    cache = Mock(num_committed_tokens=64, is_commit_allowed=False)
    manager.kv_cache_map = {42: cache}
    manager.try_commit_blocks(request)
    cache.commit.assert_not_called()
    cache.stop_committing.assert_not_called()


def test_v1_retirement_uses_accepted_generated_frontier_after_drain(guard):
    request = _request()
    request.context_current_position = 64
    request.context_remaining_length = 0
    request.num_tokens = 80
    request.freeze_retire_kv_frontier = Mock()
    guard._kv_manager = None
    guard._v1_manager = Mock(tokens_per_block=16, enable_block_reuse=True)
    with guard.forward(_batch(request)):
        pass
    with guard.token_commit(_state(request, allowed=[True])):
        request.num_tokens += 1
    with guard.forward(_batch(request)):
        pass
    guard.table.apply(VersionUpdate("session", 2, NEXT))
    with guard.token_commit(_state(request, allowed=[True])):
        assert request.is_finished
    guard.before_release(request)
    guard.device.wait_event.assert_called_once()
    request.freeze_retire_kv_frontier.assert_called_once_with(80)
    guard._v1_manager.store_blocks_for_reuse.assert_called_once_with(request)


def test_frontier_without_completion_record_cannot_be_released(guard):
    request = _request()
    guard._kv_frontiers[id(request)] = 64
    with pytest.raises(RuntimeError, match="completion is unknown"):
        guard.before_release(request)
    assert guard._kv_frontiers[id(request)] == 64
    guard._kv_manager.preserve_execution_prefix.assert_not_called()
    guard._kv_manager.stop_request_committing.assert_not_called()


def _guard_group(template, size):
    """Separate guard state and bounded host transport for each simulated rank."""
    condition = Condition()
    rounds = {}
    calls = [0] * size
    guards = []

    def endpoint(rank):
        def gather(vote):
            with condition:
                step = calls[rank]
                calls[rank] += 1
                values = rounds.setdefault(step, {})
                values[rank] = vote
                condition.notify_all()
                if not condition.wait_for(lambda: len(values) == size, timeout=3):
                    raise TimeoutError("test rank skipped an execution boundary")
                return [values[r] for r in range(size)]

        return gather

    for rank in range(size):
        local = copy(template)
        local.table = WorkerVersionTable("session")
        local.table.apply(VersionUpdate("session", 1, VERSION))
        local.table.drain_tracker = RequestDrainTracker()
        local.device = Mock()
        local.device.submission.side_effect = lambda *, capture_possible=True: nullcontext()
        local._uses = {}
        local._kv_frontiers = {}
        local._kv_manager = Mock()
        local._kv_manager.needs_execution_prefix_commit.return_value = False
        local._rank_consensus = ExecutionStepConsensus(
            "session", rank, tuple(range(size)), endpoint(rank)
        )
        guards.append(local)
    return guards


def _all_rank_failure(guards, run):
    def attempt(rank):
        with pytest.raises(RuntimeError, match="invalid local input|disagree"):
            run(rank, guards[rank])
        with pytest.raises(RuntimeError, match="restart required"):
            guards[rank].revoke_obsolete_requests([])
        # Local terminal cleanup must not erase the distributed failure.
        guards[rank].table.drain_tracker.complete(42, VERSION)
        with pytest.raises(RuntimeError, match="unknown"):
            guards[rank].table.drain_tracker.drained(VERSION)

    with ThreadPoolExecutor(max_workers=len(guards)) as pool:
        list(pool.map(attempt, range(len(guards))))


def test_multi_rank_handoff_keeps_common_reuse_cap(guard):
    guards = _guard_group(guard, 2)

    def run(rank):
        local = guards[rank]
        local.table.alignment_state = Mock()
        local.table.alignment_state.confirm_attachment.return_value = True
        request = _request()
        request.execution_reuse_limit = request.prepopulated_prompt_len = 4
        with local.forward(_batch(request, context=True)):
            pass
        assert request.execution_reuse_limit == 4

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))


@pytest.mark.parametrize("size", [2, 4])
def test_rank_steps_preserve_equal_progress_with_different_physical_slots(guard, size):
    guards = _guard_group(guard, size)

    def run(rank):
        local = guards[rank]
        request = _request()
        request.py_seq_slot = rank + 7
        local.revoke_obsolete_requests([request])
        with local.forward(_batch(request, context=True)):
            pass
        with local.context_commit(_batch(request, context=True)):
            request.context_current_position = 64
            request.context_remaining_length = 0
        with local.token_commit(_state(request, allowed=[True])):
            request.num_tokens += 1
        return local._kv_frontiers[id(request)]

    with ThreadPoolExecutor(max_workers=size) as pool:
        assert list(pool.map(run, range(size))) == [64] * size


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("kind", ["none", "empty", "untagged"])
def test_rank_commits_include_absent_empty_and_untagged_sampler_states(guard, size, kind):
    guards = _guard_group(guard, size)

    def run(rank):
        local = guards[rank]
        requests = [_request(None)] if kind == "untagged" else []
        state = None if kind == "none" else _state(*requests)
        event = None if state is None else state.sampler_event
        with local.token_commit(state):
            if state is not None:
                assert state.sampler_event is None
        with local.context_commit(_batch(*requests, context=True)):
            pass
        with local.forward(_batch(*requests)):
            pass
        if event is not None:
            event.synchronize.assert_called_once()
        assert local.table.drain_tracker.drained(VERSION)

    with ThreadPoolExecutor(max_workers=size) as pool:
        list(pool.map(run, range(size)))


@pytest.mark.parametrize("fault_rank", [0, 1])
@pytest.mark.parametrize(
    "fault",
    [
        "missing_ticket",
        "retagged",
        "slot_changed",
        "sampler_wait",
        "ticket_wait",
        "short_verdict",
        "table_closed",
        "absent_state",
    ],
)
def test_rank_token_preparation_failure_reaches_peers(guard, fault, fault_rank):
    guards = _guard_group(guard, 2)

    def run(rank, local):
        request = _request()
        state = _state(request, allowed=[True])
        if rank == fault_rank:
            if fault == "missing_ticket":
                state.execution_ticket = None
            elif fault == "retagged":
                request.py_execution_version = None
            elif fault == "slot_changed":
                request.py_seq_slot += 1
            elif fault == "sampler_wait":
                state.sampler_event.synchronize.side_effect = RuntimeError("CUDA wait failed")
            elif fault == "ticket_wait":
                state.execution_ticket.wait.side_effect = RuntimeError("commit wait failed")
            elif fault == "short_verdict":
                state.execution_ticket.wait.return_value.allowed = []
            elif fault == "table_closed":
                local.table.close()
            elif fault == "absent_state":
                state = None
        with local.token_commit(state):
            pytest.fail("failed preparation admitted host tokens")

    _all_rank_failure(guards, run)


@pytest.mark.parametrize("fault", ["missing_use", "missing_event", "wait_failed", "table_closed"])
def test_rank_context_preparation_failure_reaches_peers(guard, fault):
    guards = _guard_group(guard, 2)

    def run(rank, local):
        request = _request()
        local._native_kv_guards = True
        event = Mock()
        local._uses[id(request)] = SimpleNamespace(request=request, event=event)
        if rank == 1:
            if fault == "missing_use":
                local._uses.clear()
            elif fault == "missing_event":
                local._uses[id(request)].event = None
            elif fault == "wait_failed":
                local.device.wait_event.side_effect = RuntimeError("context wait failed")
            else:
                local.table.close()
        with local.context_commit(_batch(request, context=True)):
            pytest.fail("failed preparation admitted context progress")

    _all_rank_failure(guards, run)


@pytest.mark.parametrize("phase", ["token", "context", "prefix"])
def test_rank_commit_body_failure_keeps_every_drain_unknown(guard, phase):
    guards = _guard_group(guard, 2)

    def run(rank, local):
        request = _request()
        if phase == "prefix":
            request.context_remaining_length = 0
            request.num_tokens = 80
            with local.forward(_batch(request)):
                pass
            local._kv_manager.needs_execution_prefix_commit.return_value = True
            if rank == 1:
                local._kv_manager.preserve_execution_prefix.side_effect = RuntimeError(
                    "rebase failed"
                )
        commit = (
            local.context_commit(_batch(request, context=True))
            if phase == "context"
            else local.token_commit(_state(request, allowed=[True]))
        )
        with commit:
            if phase == "prefix":
                request.num_tokens += 1
            elif rank == 1:
                raise RuntimeError("host update failed")

    _all_rank_failure(guards, run)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "order",
        "attachment",
        "mode",
        "forward",
        "record",
        "exit",
    ],
)
def test_rank_forward_entry_and_completion_fail_together(guard, fault):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import ExecutionEarlyExit

    guards = _guard_group(guard, 2)
    entered = []

    def run(rank, local):
        local._early_exit_enabled = False
        first, second = _request(), _request(None)
        second.py_request_id = 43
        requests = [first, second]
        if rank == 1:
            if fault == "version":
                first.py_execution_version = NEXT
            elif fault == "order":
                requests.reverse()
            elif fault == "attachment":
                local.table.alignment_state = Mock()
                local.table.alignment_state.confirm_attachment.side_effect = RuntimeError(
                    "KV hold lost"
                )
            elif fault == "mode":
                local._early_exit_enabled = True
            elif fault == "record":
                local._record_uses = Mock(side_effect=RuntimeError("event record failed"))
        with local.forward(_batch(*requests, context=True)):
            entered.append(rank)
            if rank == 1 and fault == "forward":
                raise RuntimeError("model failed after partial submission")
            if rank == 1 and fault == "exit":
                raise ExecutionEarlyExit("decoder_layer", 3, 64)

    _all_rank_failure(guards, run)
    if fault in ("version", "order", "attachment", "mode"):
        assert not entered
    else:
        assert sorted(entered) == [0, 1]


@pytest.mark.parametrize("with_guard", [False, True])
def test_executor_empty_sampler_reaches_guard_without_mutating_sampler(with_guard):
    executor = SimpleNamespace(
        sampler=Mock(),
        _accumulate_spec_dec_stats=Mock(),
        _handle_errors=Mock(),
        _execution_guard=None,
    )
    if with_guard:
        executor._execution_guard = Mock()
        executor._execution_guard.token_commit.side_effect = lambda state: nullcontext()
    PyExecutor._update_requests(executor, None)
    executor.sampler.update_requests.assert_not_called()
    executor._accumulate_spec_dec_stats.assert_not_called()
    executor._handle_errors.assert_not_called()
    if with_guard:
        executor._execution_guard.token_commit.assert_called_once_with(None)


@pytest.mark.parametrize("size", [2, 4])
def test_rank_early_exit_agrees_and_preserves_every_submitted_reader(guard, size):
    from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
        ExecutionEarlyExit,
        execution_early_exit_point,
    )

    guards = _guard_group(guard, size)
    guards[-1].table.apply(VersionUpdate("session", 2, NEXT))

    def run(rank):
        local = guards[rank]
        request = _request()
        with pytest.raises(ExecutionEarlyExit) as stopped:
            with local.forward(_batch(request, context=True)):
                execution_early_exit_point("decoder_layer", 3, 64)
        assert (stopped.value.boundary, stopped.value.submitted) == ("decoder_layer", 3)
        assert local._uses[id(request)].event is not None
        # A successful early exit remains a usable coordinator. The next empty
        # consume point must still participate instead of skipping a vote.
        with local.token_commit(None):
            pass
        local.device.wait_event.assert_not_called()

    with ThreadPoolExecutor(max_workers=size) as pool:
        list(pool.map(run, range(size)))


@pytest.mark.parametrize("fault", ["table_closed", "identity", "cancel"])
def test_rank_scheduling_errors_reach_peers_before_next_forward(guard, fault):
    guards = _guard_group(guard, 2)
    for local in guards:
        local.table.apply(VersionUpdate("session", 2, NEXT))

    def run(rank, local):
        request = _request()
        if rank == 1:
            if fault == "table_closed":
                local.table.close()
            elif fault == "identity":
                del request.py_execution_version
            else:
                request.finish_by_reason.side_effect = RuntimeError("cancel failed")
        local.revoke_obsolete_requests([request])

    _all_rank_failure(guards, run)


@pytest.mark.parametrize("state", ["CONTEXT_INIT", "GENERATION_IN_PROGRESS", "GENERATION_COMPLETE"])
def test_step_snapshot_obeys_native_context_accessor_lifetime(guard, state):
    class NativeContextContract(SimpleNamespace):
        @property
        def context_chunk_size(self):
            assert self.is_context_init_state, "native chunk accessor is invalid after context"
            return 64

    fields = vars(_request()).copy()
    fields.pop("context_chunk_size")
    request = NativeContextContract(**fields)
    request.state = SimpleNamespace(name=state)
    request.is_context_init_state = state == "CONTEXT_INIT"
    votes = []

    def gather(vote):
        votes.append(vote)
        return [vote, replace(vote, rank=1)]

    guard._rank_consensus = ExecutionStepConsensus("session", 0, (0, 1), gather)
    guard._coordinate("snapshot", [request], [True])
    row = votes[0].rows[0]
    assert row.context_chunk_size == (64 if request.is_context_init_state else 0)
    assert row.last_token_id == 7


def test_equal_token_counts_do_not_hide_divergent_accepted_tokens(guard):
    guards = _guard_group(guard, 2)

    def run(rank, local):
        request = _request()
        request.context_remaining_length = 0
        with local.token_commit(_state(request, allowed=[True])):
            request.num_tokens += 1
            request.get_token = lambda beam, position: 8 + rank

    _all_rank_failure(guards, run)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("kind", ["empty", "tagged", "untagged"])
def test_loop_branches_agree_for_idle_and_active_ranks(guard, size, kind):
    guards = _guard_group(guard, size)

    def run(rank):
        local = guards[rank]
        requests = [] if kind == "empty" else [_request(None if kind == "untagged" else VERSION)]
        batch = _batch(*requests)
        local.check_loop_branch("stop", False, requests)
        local.check_dispatch(batch, bool(requests), None)
        # An empty new batch can still have a previous sampler result to drain.
        previous = SimpleNamespace(scheduled_requests=batch)
        local.check_dispatch(_batch(), False, previous)
        local.check_loop_branch("stop", True, [])

    with ThreadPoolExecutor(max_workers=size) as pool:
        list(pool.map(run, range(size)))


@pytest.mark.parametrize(
    "fault",
    [
        "stop",
        "non_boolean",
        "dispatch",
        "scheduled_rows",
        "previous",
        "previous_rows",
        "scheduled_access",
        "previous_access",
    ],
)
def test_asymmetric_loop_branches_fail_before_work_or_cleanup(guard, fault):
    guards = _guard_group(guard, 2)

    def run(rank, local):
        request = _request()
        if fault in ("stop", "non_boolean"):
            selected = bool(rank) if fault == "stop" else (None if rank else False)
            local.check_loop_branch("stop", selected, [])
            return
        batch = _batch(request)
        previous = SimpleNamespace(scheduled_requests=_batch(request))
        can_queue = True
        if rank:
            if fault == "dispatch":
                can_queue = False
            elif fault == "scheduled_rows":
                batch = _batch()
            elif fault == "previous":
                previous = None
            elif fault == "previous_rows":
                previous.scheduled_requests = _batch(_request(None))
            elif fault == "scheduled_access":
                batch = SimpleNamespace(all_requests=Mock(side_effect=ValueError("broken batch")))
            elif fault == "previous_access":
                previous = SimpleNamespace()
        local.check_dispatch(batch, can_queue, previous)

    _all_rank_failure(guards, run)


def test_single_rank_branch_checks_do_not_inspect_distributed_state(guard):
    assert not guard.is_distributed
    guard.check_dispatch(object(), True, object())
    guard.check_loop_branch("stop", False, [object()])
    guard.execution_failed(RuntimeError("local request error"))
    assert guard.table.current_version("tenant", "scope") == VERSION


@pytest.mark.parametrize("with_guard", [False, True])
@pytest.mark.parametrize("has_batch", [False, True])
def test_executor_dispatch_checks_before_returning_branch(with_guard, has_batch):
    guard = Mock() if with_guard else None
    previous = object()
    batch = _batch(_request()) if has_batch else _batch()
    executor = SimpleNamespace(
        enable_attention_dp=False, _execution_guard=guard, previous_batch=previous
    )
    assert PyExecutor._can_queue(executor, batch) == (has_batch, has_batch)
    if guard is not None:
        guard.check_dispatch.assert_called_once_with(batch, has_batch, previous)


def test_executor_checks_stop_before_skipping_schedule():
    guard = Mock()
    executor = SimpleNamespace(
        _process_execution_releases=Mock(),
        _release_unused_connector_reservations=Mock(),
        _poll_encoder_steps=Mock(),
        _fetch_and_activate_new_requests=Mock(return_value=[]),
        should_stop_processing=True,
        _execution_guard=guard,
        active_requests=[],
        _handle_control_request=Mock(),
    )
    assert PyExecutor._prepare_and_schedule_batch(executor) == (None, None)
    executor._process_execution_releases.assert_called_once()
    guard.check_loop_branch.assert_called_once_with("stop", True, [])
    executor._handle_control_request.assert_not_called()


@pytest.mark.parametrize("charge_budget", [False, True])
def test_distributed_request_error_cannot_enter_local_recovery(guard, charge_budget):
    guard = _guard_group(guard, 2)[0]
    executor = SimpleNamespace(_execution_guard=guard, _error_budget=Mock())
    with pytest.raises(RuntimeError, match="distributed RETIRE execution failed: model error"):
        PyExecutor._handle_errors(executor, "model error", requests=[], charge_budget=charge_budget)
    executor._error_budget.consume.assert_not_called()


@pytest.mark.parametrize("fault", ["model", "invalidation", "cleanup", "after_loop"])
def test_distributed_error_invalidates_evidence_with_watchdog_armed(guard, monkeypatch, fault):
    from tensorrt_llm._torch.pyexecutor import py_executor as executor_module

    local = _guard_group(guard, 2)[0]
    events = []
    original = RuntimeError("model failed")
    watchdog = SimpleNamespace(deadline=17, cancel=lambda: events.append("cancel watchdog"))

    def arm(*args, **kwargs):
        events.append("arm watchdog")
        return watchdog

    def kill(*args, **kwargs):
        assert kwargs["deadline"] == 17
        events.append("kill")

    monkeypatch.setattr(executor_module, "host_profiler_context", lambda **kw: nullcontext())
    monkeypatch.setattr(executor_module, "customized_gc_thresholds", lambda *args: nullcontext())
    monkeypatch.setattr(executor_module, "start_rank_crash_kill_watchdog", arm)
    monkeypatch.setattr(executor_module, "hard_kill_on_rank_crash", kill)

    def loop():
        if fault == "after_loop":
            executor._event_loop_completed = True
        raise original

    def invalidate(error):
        assert error is original
        if fault != "after_loop":
            assert events == ["arm watchdog"]
        events.append("invalidate")
        ExecutionDeviceGuard.execution_failed(local, error)
        if fault == "invalidation":
            raise RuntimeError("invalidation failed")

    def cleanup():
        events.append("cleanup")
        local.table.drain_tracker.complete(42, VERSION)
        with pytest.raises(RuntimeError, match="unknown"):
            local.table.drain_tracker.drained(VERSION)
        with pytest.raises(RuntimeError, match="closed"):
            local.table.current_version("tenant", "scope")
        with pytest.raises(RuntimeError, match="restart required"):
            local._rank_consensus.agree("next", (), ())
        if fault == "cleanup":
            raise RuntimeError("cleanup failed")

    local.execution_failed = invalidate
    executor = SimpleNamespace(
        llm_args=SimpleNamespace(torch_blas_preference=None),
        _execution_guard=local,
        _event_loop_error=None,
        _event_loop_error_delivered=Event(),
        is_warmup=False,
        garbage_collection_gen0_threshold=None,
        event_loop=loop,
        dist=SimpleNamespace(world_size=2),
        _executor_loop_cleanup=cleanup,
    )
    with pytest.raises(RuntimeError, match="failed"):
        PyExecutor._event_loop_wrapper(executor)
    assert executor._event_loop_error is original
    if fault == "after_loop":
        assert events == ["invalidate", "cleanup"]
    else:
        assert events == ["arm watchdog", "invalidate", "cleanup", "cancel watchdog", "kill"]
