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
        py_client_id=142,
        prepopulated_prompt_len=0,
        context_current_position=0,
        context_chunk_size=64,
        context_remaining_length=64,
        is_finished=False,
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

    def finish(reason):
        assert reason == FinishReason.CANCELLED
        request.state = LlmRequestState.GENERATION_COMPLETE
        request.is_finished = True

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
