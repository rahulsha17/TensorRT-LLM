# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local CPU/QA checks of actual policy and guard methods with device doubles."""

from concurrent.futures import Future
from copy import copy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from test_execution_device_guard import NEXT, VERSION, _batch, _request
from test_execution_device_guard import guard as guard

from tensorrt_llm._torch.pyexecutor.execution_attention import ExecutionAttentionBinding
from tensorrt_llm._torch.pyexecutor.execution_early_exit import (
    early_exit_scope,
    get_layer_early_exit_point,
)
from tensorrt_llm._torch.pyexecutor.execution_rate_policy import (
    _PolicyStep,
    _RatePolicyConfig,
    _RatePolicyState,
    _startup_config,
    _startup_policy,
    current_policy_step,
    policy_step_scope,
)
from tensorrt_llm.execution_mailbox import VersionUpdate

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only


def request(
    version: "ExecutionVersion | None" = VERSION,
    *,
    slot: int = 2,
    request_id: int = 42,
    context_tokens: int = 32,
) -> SimpleNamespace:
    value = _request(version)
    value.py_seq_slot = slot
    value.py_request_id = request_id
    value.context_chunk_size = context_tokens
    value.is_cuda_graph_dummy = value.is_dummy_request = False
    return value


def policy(horizon: int = 4, minimum: int = 32) -> _RatePolicyState:
    return _RatePolicyState(_RatePolicyConfig(horizon, minimum))


def finalized(owner: _RatePolicyState, batch: "ScheduledRequests") -> _PolicyStep:
    step = _PolicyStep(owner, batch)
    total = sum(r.context_chunk_size for r in batch.context_requests) + len(
        batch.generation_requests
    )
    step.finalize(batch, total, False)
    return step


def sample_state(requests: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        requests=list(requests),
        host=SimpleNamespace(new_tokens=object(), single_step_greedy=True),
        device=SimpleNamespace(new_tokens=object()),
        execution_ticket=None,
        execution_policy_receipt=None,
        sampler_event=SimpleNamespace(synchronize=Mock(), worker_futures=[]),
    )


def dispatch(
    guard: "ExecutionDeviceGuard",
    *,
    horizon: int = 0,
    minimum: int = 1024,
    context: bool = False,
    requests: list[SimpleNamespace] | None = None,
) -> tuple["ScheduledRequests", SimpleNamespace]:
    guard._rate_policy = policy(horizon, minimum)
    guard._native_kv_guards = True
    requests = requests or [request()]
    batch = _batch(*requests, context=context)
    with guard.forward(batch):
        step = current_policy_step()
        total = sum(r.context_chunk_size for r in batch.context_requests) + len(
            batch.generation_requests
        )
        step.finalize(batch, total, False)
        step.attention_prepared = True
    state = sample_state(batch.context_requests_last_chunk + batch.generation_requests)
    with guard.sampling(batch):
        guard.commit_sample(batch, state)
    return batch, state


def test_absent_policy_preserves_startup_without_reading_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_RETIRE_BATCH_POLICY", raising=False)
    assert _startup_policy(None) is None


@pytest.mark.parametrize(
    "value",
    [
        "",
        "[]",
        "{}",
        '{"layer_horizon":4,"min_scheduled_tokens":true}',
        '{"layer_horizon":2,"min_scheduled_tokens":32}',
        '{"layer_horizon":0,"min_scheduled_tokens":2}',
        '{"layer_horizon":4,"min_scheduled_tokens":32,"unknown":1}',
    ],
)
def test_invalid_startup_policy_fails_closed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TRTLLM_RETIRE_BATCH_POLICY", value)
    with pytest.raises(ValueError):
        _startup_config()


@pytest.mark.parametrize(
    "fault",
    [None, "tp", "overlap", "graph", "compile", "speculation", "model", "native_off", "early_off"],
)
def test_startup_restricts_opt_in_without_expanding_engine_support(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    monkeypatch.setenv(
        "TRTLLM_RETIRE_BATCH_POLICY", '{"layer_horizon":4,"min_scheduled_tokens":32}'
    )
    monkeypatch.setenv("TRTLLM_RETIRE_NATIVE_GUARDS", "0" if fault == "native_off" else "1")
    monkeypatch.setenv("TRTLLM_RETIRE_EARLY_EXIT", "0" if fault == "early_off" else "1")
    model_type = type(
        "Qwen3ForCausalLM", (), {"__module__": "tensorrt_llm._torch.models.modeling_qwen3"}
    )
    runner = SimpleNamespace(
        model=object() if fault == "model" else model_type(),
        mapping=SimpleNamespace(world_size=2 if fault == "tp" else 1),
        _torch_compile_backend=None,
        sparse_attention_config=None,
        _config=SimpleNamespace(
            disable_overlap_scheduler=fault != "overlap",
            cuda_graph_config={} if fault == "graph" else None,
            torch_compile_enabled=fault == "compile",
            is_spec_decode=fault == "speculation",
            spec_config=None,
            is_multimodal=False,
            is_encode_only=False,
            enable_attention_dp=False,
            max_beam_width=1,
        ),
    )
    if fault:
        with pytest.raises((ValueError, NotImplementedError)):
            _startup_policy(runner)
    else:
        assert _startup_policy(runner).config == _RatePolicyConfig(4, 32)


@pytest.mark.parametrize("total", [0, 1, 31, 32, 1023, 1024])
@pytest.mark.parametrize("minimum", [1, 32, 1024])
def test_threshold_uses_actual_finalized_tokens(total: int, minimum: int) -> None:
    batch = _batch(request(context_tokens=total), context=True) if total else _batch()
    step = finalized(policy(4, minimum), batch)
    assert step.plan.predicates is bool(total >= minimum)
    assert step.plan.context_tokens + step.plan.decode_tokens == total
    with pytest.raises(FrozenInstanceError):
        step.plan.predicates = False


def test_total_tokens_include_decode_and_mixed_untagged_rows_cannot_exit() -> None:
    context = request(context_tokens=31)
    neighbor = request(None, slot=3, request_id=43)
    batch = _batch(context, context=True)
    batch.generation_requests = [neighbor]
    step = finalized(policy(4, 32), batch)
    assert step.plan.predicates and not step.plan.layer_probes
    assert (step.plan.context_tokens, step.plan.decode_tokens) == (31, 1)


@pytest.mark.parametrize("concurrency", [4, 8])
def test_always_armed_decode_is_distinct_from_token_adaptive(concurrency: int) -> None:
    batch = _batch(*(request(slot=i, request_id=i) for i in range(concurrency)))
    assert finalized(policy(4, 1), batch).plan.layer_probes
    assert not finalized(policy(4, 32), batch).plan.predicates


@pytest.mark.parametrize(
    "fault", ["token_count", "graph", "duplicate", "dummy", "draft", "second_finalize"]
)
def test_finalized_batch_does_not_accept_padding_or_ambiguous_work(fault: str | None) -> None:
    row = request()
    batch = _batch(row, context=True)
    step = _PolicyStep(policy(), batch)
    if fault == "duplicate":
        batch.generation_requests = [row]
    if fault == "dummy":
        row.is_cuda_graph_dummy = True
    if fault == "draft":
        row.py_draft_tokens = [1]
    if fault == "second_finalize":
        step.finalize(batch, 32, False)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        step.finalize(batch, 33 if fault == "token_count" else 32, fault == "graph")


def test_layer_zero_is_only_omitted_in_opt_in_and_context_is_restored() -> None:
    seen = []
    step = finalized(policy(4, 1), _batch(request()))
    with early_exit_scope(lambda: False, observer=lambda *args: seen.append(args[1])):
        legacy = get_layer_early_exit_point()
        legacy("eager_layer", 0, 8)
        with pytest.raises(RuntimeError, match="model failed"):
            with policy_step_scope(step):
                hook = get_layer_early_exit_point()
                for layer in (0, 4, 8):
                    hook("eager_layer", layer, 12)
                raise RuntimeError("model failed")
        assert current_policy_step() is None
        get_layer_early_exit_point()("eager_layer", 0, 8)
    assert seen == [0, 4, 8, 0]


@pytest.mark.parametrize(
    "horizon,minimum,chunk,armed", [(0, 1, 1024, False), (4, 1024, 32, False), (4, 1, 1, True)]
)
def test_attention_clears_all_predicates_without_releasing_old_rows(
    horizon: int, minimum: int, chunk: int, armed: bool
) -> None:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    owner = policy(horizon, minimum)
    authority = SimpleNamespace(
        _table=object(),
        create_attention_rows=Mock(
            return_value=SimpleNamespace(slots=object(), expected=object(), counters=object())
        ),
        prepare_attention_rows=Mock(),
    )
    binding = ExecutionAttentionBinding(authority, rate_policy=owner)
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata.is_cross = False
    metadata.kv_cache_manager = object()
    metadata.kv_lens_cuda_runtime = SimpleNamespace(numel=lambda: 1)
    metadata.request_ids = [42]
    metadata.max_num_requests = 1
    for name in (
        "retire_current_versions",
        "retire_sequence_slots",
        "retire_expected_versions",
        "retire_counters",
    ):
        setattr(metadata, name, object())
    batch = _batch(request(context_tokens=chunk), context=True)
    step = finalized(owner, batch)
    with policy_step_scope(step):
        binding.prepare(metadata, batch)
    assert step.attention_prepared
    assert (metadata.retire_current_versions is not None) is armed
    if armed:
        retained = dict(binding._rows)
        cold_batch = _batch(request(None, context_tokens=chunk), context=True)
        with policy_step_scope(finalized(owner, cold_batch)):
            binding.prepare(metadata, cold_batch)
        assert binding._rows == retained
    assert all(
        getattr(metadata, name) is None
        for name in (
            "retire_current_versions",
            "retire_sequence_slots",
            "retire_expected_versions",
            "retire_counters",
        )
    )
    if armed:
        with policy_step_scope(finalized(owner, batch)):
            binding.prepare(metadata, batch)
        assert metadata.retire_current_versions is authority._table
        assert binding._rows == retained
        assert authority.create_attention_rows.call_count == 1


def test_boundary_commit_waits_outside_authority_and_preserves_native_use(
    guard: "ExecutionDeviceGuard",
) -> None:
    batch, state = dispatch(guard)
    receipt = state.execution_policy_receipt
    use = guard._uses[id(state.requests[0])]
    order = []

    def completed() -> None:
        guard.table.apply(VersionUpdate("session", 2, NEXT))
        order.append("sampler")

    state.sampler_event.synchronize.side_effect = completed
    guard.device.wait_event.side_effect = lambda event: order.append("device")
    with guard.token_commit(state):
        order.append("host")
        assert state.requests[0].is_finished
    assert order == ["sampler", "device", "host"]
    assert guard._uses[id(state.requests[0])] is use
    assert receipt.completion_event.record.called
    guard.device.enqueue_token_commit.assert_not_called()


def test_boundary_acceptance_advances_only_completed_input_frontier_and_is_single_use(
    guard: "ExecutionDeviceGuard",
) -> None:
    row = request()
    row.context_remaining_length = 0
    _, state = dispatch(guard, requests=[row])
    with guard.token_commit(state):
        row.num_tokens += 1
    assert guard._kv_frontiers[id(row)] == row.num_tokens - 1
    assert state.execution_policy_receipt is None and state.sampler_event is None
    with pytest.raises(RuntimeError, match="policy receipt"):
        with guard.token_commit(state):
            pytest.fail("receipt accepted twice")


@pytest.mark.parametrize(
    "fault", [None, "version_none", "reorder", "post_ticket_change", "missing_receipt"]
)
def test_armed_sample_keeps_native_ticket_and_policy_identity(
    guard: "ExecutionDeviceGuard", fault: str | None
) -> None:
    rows = [request(slot=i, request_id=i) for i in range(2)]
    ticket = SimpleNamespace(
        versions=tuple(row.py_execution_version for row in rows),
        sequence_slots=tuple(row.py_seq_slot for row in rows),
        tokens_device=object(),
        tokens_host=object(),
        wait=Mock(return_value=SimpleNamespace(allowed=[True, True])),
    )
    guard.device.enqueue_token_commit.return_value = ticket
    _, state = dispatch(guard, horizon=4, minimum=1, requests=rows)
    assert (
        state.execution_ticket is ticket and state.execution_policy_receipt.completion_event is None
    )
    assert (
        state.device.new_tokens is ticket.tokens_device
        and state.host.new_tokens is ticket.tokens_host
    )
    if fault == "version_none":
        rows[0].py_execution_version = None
    if fault == "reorder":
        state.requests.reverse()
    if fault == "missing_receipt":
        state.execution_policy_receipt = None
    if fault == "post_ticket_change":

        def change_slot() -> SimpleNamespace:
            rows[0].py_seq_slot += 1
            return SimpleNamespace(allowed=[True, True])

        ticket.wait.side_effect = change_slot
    if fault:
        with pytest.raises(RuntimeError):
            with guard.token_commit(state):
                pytest.fail("armed identity changed")
    else:
        with guard.token_commit(state):
            pass
        ticket.wait.assert_called_once()
        with pytest.raises(RuntimeError, match="policy receipt"):
            with guard.token_commit(state):
                pytest.fail("armed receipt accepted twice")


def test_intermediate_context_empty_sampling_consumes_plan_without_ticket(
    guard: "ExecutionDeviceGuard",
) -> None:
    guard._rate_policy = policy(4, 32)
    row = request(context_tokens=32)
    batch = _batch()
    batch.context_requests_chunking = [row]
    with guard.forward(batch):
        current_policy_step().finalize(batch, 32, False)
        current_policy_step().attention_prepared = True
    state = sample_state([])
    with guard.sampling(batch):
        guard.commit_sample(batch, state)
    with guard.token_commit(state):
        pass
    assert guard._pending_policy_step is None and state.execution_policy_receipt is None
    assert id(row) in guard._uses
    guard.device.enqueue_token_commit.assert_not_called()


@pytest.mark.parametrize(
    "fault",
    [
        "request",
        "request_id",
        "version_none",
        "slot",
        "host_tokens",
        "device_tokens",
        "sampler_event",
        "missing_receipt",
    ],
)
def test_boundary_commit_rejects_changed_sampling_identity(
    guard: "ExecutionDeviceGuard", fault: str | None
) -> None:
    _, state = dispatch(guard)
    if fault == "request":
        state.requests[0] = copy(state.requests[0])
    if fault == "request_id":
        state.requests[0].py_request_id += 1
    if fault == "version_none":
        state.requests[0].py_execution_version = None
    if fault == "slot":
        state.requests[0].py_seq_slot += 1
    if fault == "host_tokens":
        state.host.new_tokens = object()
    if fault == "device_tokens":
        state.device.new_tokens = object()
    if fault == "sampler_event":
        state.sampler_event = None
    if fault == "missing_receipt":
        state.execution_policy_receipt = None
    with pytest.raises(RuntimeError):
        with guard.token_commit(state):
            pytest.fail("changed identity was accepted")


@pytest.mark.parametrize("fault", ["future_error", "future_mutation", "completion_error"])
def test_boundary_completion_failure_never_reaches_host_mutation(
    guard: "ExecutionDeviceGuard", fault: str | None
) -> None:
    _, state = dispatch(guard)
    if fault == "future_error":
        future = Future()
        future.set_exception(RuntimeError("D2H failed"))
        state.sampler_event.worker_futures = [future]
    elif fault == "future_mutation":
        state.sampler_event.worker_futures = [
            SimpleNamespace(result=lambda: setattr(state.requests[0], "py_seq_slot", 99))
        ]
    else:
        guard.device.wait_event.side_effect = RuntimeError("completion failed")
    with pytest.raises(RuntimeError):
        with guard.token_commit(state):
            pytest.fail("failed copy/event was accepted")


def test_exact_sampling_order_excludes_intermediate_context_and_receipt_is_single_use(
    guard: "ExecutionDeviceGuard",
) -> None:
    guard._rate_policy = policy(0, 1)
    chunk = request(slot=0, request_id=1)
    final = request(slot=1, request_id=2)
    decode = request(slot=2, request_id=3)
    batch = _batch(final, context=True)
    batch.context_requests_chunking = [chunk]
    batch.generation_requests = [decode]
    with guard.forward(batch):
        current_policy_step().finalize(batch, 65, False)
        current_policy_step().attention_prepared = True
    state = sample_state([decode, final])
    with pytest.raises(RuntimeError, match="identity"):
        guard.commit_sample(batch, state)
    with pytest.raises(RuntimeError, match="completed"):
        guard.commit_sample(batch, sample_state([final, decode]))


@pytest.mark.parametrize("failure", ["model", "missing_finalize", "event_record"])
def test_failed_forward_cannot_reuse_previous_batch_policy(
    guard: "ExecutionDeviceGuard", monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    dispatch(guard)
    batch = _batch(request())
    if failure == "event_record":
        monkeypatch.setattr(
            "torch.cuda.Event",
            lambda: SimpleNamespace(record=Mock(side_effect=RuntimeError("record failed"))),
        )
    with pytest.raises(RuntimeError):
        with guard.forward(batch):
            if failure != "missing_finalize":
                current_policy_step().finalize(batch, 1, False)
                current_policy_step().attention_prepared = True
            if failure == "model":
                raise RuntimeError("model failed")
    assert current_policy_step() is None and guard._pending_policy_step is None
    with pytest.raises(RuntimeError, match="completed"):
        guard.commit_sample(batch, sample_state(batch.generation_requests))


def test_legacy_missing_device_ticket_still_fails(guard: "ExecutionDeviceGuard") -> None:
    state = sample_state([request()])
    with pytest.raises(RuntimeError, match="no device commit ticket"):
        with guard.token_commit(state):
            pytest.fail("legacy missing ticket was accepted")


def test_bounded_telemetry_retains_counts_after_trace_eviction() -> None:
    owner = policy()
    for _ in range(200):
        finalized(owner, _batch(request(context_tokens=32), context=True))
    result = owner.snapshot()
    assert result["counters"]["forward_plans"] == 200
    assert len(result["recent_decisions"]) == 128
    assert result["recent_decisions"][0]["step"] == 73
