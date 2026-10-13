# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native execution guard wiring with a controlled device verdict."""

from contextlib import nullcontext
from dataclasses import replace
from threading import Event, Lock, Thread
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
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
NEXT = replace(VERSION, epoch=8)


def _request(version=VERSION):
    request = SimpleNamespace(
        py_request_id=42,
        py_execution_version=version,
        py_seq_slot=2,
        py_decoding_iter=3,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_beam_width=1,
        py_draft_tokens=None,
        py_multimodal_data=None,
        py_disaggregated_params=None,
        py_logits_post_processors=None,
    )

    def finish(reason):
        assert reason == FinishReason.CANCELLED
        request.state = LlmRequestState.GENERATION_COMPLETE

    request.finish_by_reason = Mock(side_effect=finish)
    return request


@pytest.fixture
def guard(monkeypatch):
    result = object.__new__(ExecutionDeviceGuard)
    result.table = WorkerVersionTable("session")
    result.table.apply(VersionUpdate("session", 1, VERSION))
    result.device = Mock()
    result.device.submission.side_effect = nullcontext
    result._uses = {}
    result._kv_manager = Mock()
    monkeypatch.setattr("torch.cuda.Event", Mock)
    return result


def _batch(*requests, context=False):
    batch = ScheduledRequests()
    if context:
        batch.context_requests_last_chunk = list(requests)
    else:
        batch.generation_requests = list(requests)
    return batch


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


def test_wait_failure_retains_ownership_and_never_closes_cache(guard) -> None:
    request = _request()
    with guard.forward(_batch(request)):
        pass
    guard.device.wait_event.side_effect = TimeoutError("still running")
    with pytest.raises(TimeoutError):
        guard.before_release(request)
    assert id(request) in guard._uses
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


def test_model_submission_excludes_publisher_cuda_calls(monkeypatch) -> None:
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
        with authority.submission():
            worker.start()
            assert started.wait(2)
            assert not finished.wait(0.05)
            cuda_calls.assert_not_called()
    finally:
        if worker.ident is not None:
            worker.join(timeout=2)
    assert finished.is_set()
    assert not errors
    cuda_calls.assert_called_once_with(None)


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
        dist=SimpleNamespace(world_size=1),
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


def test_fence_covers_enqueue_until_terminal_cleanup(guard):
    from tensorrt_llm.execution_authority import Revision
    from tensorrt_llm.execution_fence import FenceRequest, RequestDrainTracker

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


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("mode", ["eager", "graph", "compiled", "compiled-graph"])
def test_resume_handoff_before_common_forward_boundary(guard, v2, overlap, mode):
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
    guard.table.apply(VersionUpdate("session", 3, NEXT, alignment, receipt))
    request = _request(NEXT)
    request.get_tokens = Mock(return_value=list(revised.input_token_ids))
    request.cache_salt = revised.cache_salt
    request.lora_task_id = None
    guard.validate_successor(request)
    request.prepopulated_prompt_len = 2
    with pytest.raises(RuntimeError, match="aligned KV prefix"):
        with guard.forward(_batch(request, context=True)):
            pytest.fail("must not submit the model without its prefix")
    guard.device.submission.assert_not_called()
    request.prepopulated_prompt_len = 3
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
