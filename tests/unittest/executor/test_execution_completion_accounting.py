# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion credit must retain the exact native owner that proved sharing."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only


def request(request_id=1, *, finished=False):
    return SimpleNamespace(
        py_request_id=request_id,
        prompt_len=7,
        max_new_tokens=6,
        is_finished=finished,
        is_first_context_chunk=True,
        lora_task_id=3,
        cache_salt="tenant",
        execution_reuse_limit=7,
        is_dummy=False,
        return_perf_metrics=True,
    )


@pytest.fixture
def manager():
    value = SimpleNamespace(
        impl=Mock(),
        kv_cache_map={},
        conversation_manager=None,
        _stream=SimpleNamespace(cuda_stream=123),
        enable_block_reuse=True,
        tokens_per_block=4,
        _context_reuse_tokens=Mock(return_value=list(range(7))),
    )
    value.impl.get_life_cycle_pool_group_indices.return_value = [0, 0]
    value.impl.get_completion_demand.return_value = [9]
    value.execution_completion_demand = MethodType(
        KVCacheManagerV2.execution_completion_demand, value
    )
    value.execution_completion_try_admit = MethodType(
        KVCacheManagerV2.execution_completion_try_admit, value
    )
    value.execution_completion_try_bounded_prefill = MethodType(
        KVCacheManagerV2.execution_completion_try_bounded_prefill, value
    )
    value.execution_admission_demand_snapshot = MethodType(
        KVCacheManagerV2.execution_admission_demand_snapshot, value
    )

    def create(request_id, *args, **kwargs):
        cache = SimpleNamespace(capacity=0, num_committed_tokens=0)
        value.kv_cache_map[request_id] = cache
        return cache

    value._create_kv_cache = Mock(side_effect=create)
    value.free_resources = Mock(side_effect=lambda req: value.kv_cache_map.pop(req.py_request_id))
    return value


def test_demand_uses_exact_native_owners_and_rounds_physical_slots_up(manager):
    live, obsolete = request(), request(2, finished=True)
    live_cache, old_cache = object(), object()
    manager.kv_cache_map = {1: live_cache, 2: old_cache}
    assert (
        manager.execution_completion_demand([(live, live_cache, 999), (obsolete, old_cache, 999)])
        == 5
    )
    manager.impl.get_completion_demand.assert_called_once_with([live_cache, old_cache], [12, 0])
    manager.kv_cache_map[2] = object()
    with pytest.raises(RuntimeError, match="identity changed"):
        manager.execution_completion_demand([(obsolete, old_cache, 0)])


@pytest.mark.parametrize("groups", [[], [0, 1], [1]])
def test_demand_rejects_unqualified_pool_layout(manager, groups):
    manager.impl.get_life_cycle_pool_group_indices.return_value = groups
    with pytest.raises(ValueError, match="one dense GPU pool"):
        manager.execution_completion_demand([])
    manager.impl.get_completion_demand.assert_not_called()


@pytest.mark.parametrize("demand", [[-1], [True], [1.5], [], [1, 2]])
def test_demand_rejects_invalid_native_receipt(manager, demand):
    manager.impl.get_completion_demand.return_value = demand
    with pytest.raises(RuntimeError, match="Invalid native completion demand"):
        manager.execution_completion_demand([])


def test_missing_native_demand_api_fails_clearly(manager):
    manager.impl.get_completion_demand = None
    with pytest.raises(RuntimeError, match="requires rebuilt native KV demand API"):
        manager.execution_completion_demand([])


def test_admission_retains_the_exact_match_until_context_adopts_it(manager):
    req = request()
    assert manager.execution_completion_try_admit([], req, 5)
    acquired = manager.kv_cache_map[1]
    assert acquired.cuda_stream == 123
    manager.impl.get_completion_demand.assert_called_once_with([acquired], [12])
    manager._create_kv_cache.assert_called_once_with(
        1,
        3,
        list(range(7)),
        cache_salt="tenant",
        max_reuse_tokens=7,
        is_dummy=False,
        enable_request_stats=True,
        expected_prompt_length=7,
    )
    manager.free_resources.assert_not_called()
    with pytest.raises(RuntimeError, match="already owns KV"):
        manager.execution_completion_try_admit([], req, 5)


@pytest.mark.parametrize("failure", [False, True])
def test_failed_admission_releases_only_new_owner(manager, failure):
    old, old_cache = request(2), object()
    manager.kv_cache_map[2] = old_cache
    if failure:
        manager.impl.get_completion_demand.side_effect = RuntimeError("native proof failed")
        with pytest.raises(RuntimeError, match="native proof failed"):
            manager.execution_completion_try_admit([(old, old_cache, 3)], request(), 4)
    else:
        assert not manager.execution_completion_try_admit([(old, old_cache, 3)], request(), 4)
    assert manager.kv_cache_map == {2: old_cache}
    manager.free_resources.assert_called_once()
    assert manager.free_resources.call_args.args[0].py_request_id == 1


def test_no_native_request_slot_does_not_claim_or_release_an_owner(manager):
    manager._create_kv_cache.side_effect = None
    manager._create_kv_cache.return_value = None
    assert not manager.execution_completion_try_admit([], request(), 5)
    manager.impl.get_completion_demand.assert_not_called()
    manager.free_resources.assert_not_called()


@pytest.mark.parametrize("conversation", [True, False])
def test_admission_rejects_nonfresh_or_conversation_requests(manager, conversation):
    req = request()
    if conversation:
        manager.conversation_manager = object()
    else:
        req.is_first_context_chunk = False
    with pytest.raises(ValueError, match="fresh text context"):
        manager.execution_completion_try_admit([], req, 5)
    manager._create_kv_cache.assert_not_called()


def test_bounded_demand_keeps_decoder_completion_and_exact_prefix_owner(manager):
    decoder, prefill = request(2), request(1)
    decoder_cache = SimpleNamespace(capacity=8)
    prefill_cache = SimpleNamespace(capacity=3)
    manager.kv_cache_map = {2: decoder_cache, 1: prefill_cache}
    claims = [(decoder, decoder_cache, 3), (prefill, prefill_cache, 3)]
    manager.execution_completion_demand(claims, bounded_contexts={1: 6})
    manager.impl.get_completion_demand.assert_called_once_with(
        [decoder_cache, prefill_cache], [12, 6]
    )
    prefill.is_finished = True
    manager.execution_completion_demand(claims, bounded_contexts={1: 6})
    assert manager.impl.get_completion_demand.call_args.args[1] == [12, 0]


@pytest.mark.parametrize("limit", [-1, 7, True])
def test_bounded_demand_rejects_limits_that_allow_unreserved_decode(manager, limit):
    req, cache = request(), SimpleNamespace(capacity=0)
    manager.kv_cache_map[1] = cache
    with pytest.raises(ValueError, match="before prompt completion"):
        manager.execution_completion_demand([(req, cache, 0)], bounded_contexts={1: limit})
    manager.impl.get_completion_demand.assert_not_called()


def test_bounded_demand_fails_closed_for_growth_or_missing_owner(manager):
    req, cache = request(), SimpleNamespace(capacity=5)
    manager.kv_cache_map[1] = cache
    with pytest.raises(RuntimeError, match="exceeded its reservation"):
        manager.execution_completion_demand([(req, cache, 0)], bounded_contexts={1: 4})
    with pytest.raises(RuntimeError, match="no exact native owner"):
        manager.execution_completion_demand([], bounded_contexts={1: 4})
    manager.impl.get_completion_demand.assert_not_called()


def test_bounded_admission_retains_only_unreserved_capacity(manager):
    decoder = request(2)
    old_cache = SimpleNamespace(capacity=8)
    manager.kv_cache_map[2] = old_cache
    req = request()
    req.prompt_len = 21
    manager.impl.get_completion_demand.side_effect = lambda caches, bounds: [
        2 * sum((max(cache.capacity, bound) + 3) // 4 for cache, bound in zip(caches, bounds))
    ]
    limit = manager.execution_completion_try_bounded_prefill([(decoder, old_cache, 3)], req, 5, 4)
    assert limit == 8
    candidate = manager.kv_cache_map[1]
    assert manager.impl.get_completion_demand.call_args.args == ([old_cache, candidate], [12, 8])
    assert manager.kv_cache_map[2] is old_cache
    assert candidate.cuda_stream == 123
    manager.free_resources.assert_not_called()


def test_bounded_partial_match_reserves_private_tail_and_native_copy_source(manager):
    decoder, req = request(2), request()
    req.prompt_len = 21
    old_cache = SimpleNamespace(capacity=8)
    manager.kv_cache_map[2] = old_cache

    def create(*args, **kwargs):
        cache = SimpleNamespace(capacity=6, num_committed_tokens=6)
        manager.kv_cache_map[1] = cache
        return cache

    manager._create_kv_cache.side_effect = create
    manager.impl.get_completion_demand.side_effect = lambda caches, bounds: [
        2 * (1 + sum((max(cache.capacity, bound) + 3) // 4 for cache, bound in zip(caches, bounds)))
    ]
    assert (
        manager.execution_completion_try_bounded_prefill([(decoder, old_cache, 3)], req, 8, 4) == 16
    )
    assert manager.impl.get_completion_demand.call_args.args[1] == [12, 16]
    manager.free_resources.assert_not_called()


@pytest.mark.parametrize("failure", ["no_slack", "copy_source", "exception"])
def test_bounded_admission_releases_candidate_if_native_proof_fails(manager, failure):
    req = request()
    req.prompt_len = 21
    old_cache = SimpleNamespace(capacity=8)
    old = request(2)
    manager.kv_cache_map[2] = old_cache
    if failure == "exception":
        manager.impl.get_completion_demand.side_effect = RuntimeError("native failure")
        with pytest.raises(RuntimeError, match="native failure"):
            manager.execution_completion_try_bounded_prefill([(old, old_cache, 3)], req, 5, 4)
    else:
        # A copy-source charge can consume the slack before any GPU work starts.
        manager.impl.get_completion_demand.return_value = [10 if failure == "no_slack" else 12]
        assert (
            manager.execution_completion_try_bounded_prefill([(old, old_cache, 3)], req, 5, 4)
            is None
        )
    assert manager.kv_cache_map == {2: old_cache}
    manager.free_resources.assert_called_once_with(req)


def test_bounded_admission_cannot_finish_prompt_without_completion_claim(manager):
    req = request()
    manager.impl.get_completion_demand.return_value = [0]
    assert manager.execution_completion_try_bounded_prefill([], req, 100, 1) == req.prompt_len - 1


def test_full_admission_preserves_exact_bounded_and_obsolete_owners(manager):
    partial, obsolete = request(2), request(3, finished=True)
    partial_cache = SimpleNamespace(capacity=4)
    old_cache = SimpleNamespace(capacity=8)
    manager.kv_cache_map = {2: partial_cache, 3: old_cache}
    claims = [(partial, partial_cache, 3), (obsolete, old_cache, 3)]
    assert manager.execution_completion_try_admit(claims, request(), 5, bounded_contexts={2: 6})
    candidate = manager.kv_cache_map[1]
    manager.impl.get_completion_demand.assert_called_once_with(
        [partial_cache, old_cache, candidate], [6, 0, 12]
    )
    manager.free_resources.assert_not_called()


@pytest.mark.parametrize("demand", [[11], RuntimeError("native failure")])
def test_failed_full_admission_with_partial_releases_only_candidate(manager, demand):
    partial, partial_cache = request(2), SimpleNamespace(capacity=4)
    manager.kv_cache_map[2] = partial_cache
    if isinstance(demand, Exception):
        manager.impl.get_completion_demand.side_effect = demand
        with pytest.raises(RuntimeError, match="native failure"):
            manager.execution_completion_try_admit(
                [(partial, partial_cache, 3)], request(), 5, bounded_contexts={2: 6}
            )
    else:
        manager.impl.get_completion_demand.return_value = demand
        assert not manager.execution_completion_try_admit(
            [(partial, partial_cache, 3)], request(), 5, bounded_contexts={2: 6}
        )
    assert manager.kv_cache_map == {2: partial_cache}
    manager.free_resources.assert_called_once()


@pytest.mark.parametrize(
    "reason", ["no_slot", "complete_match", "matched_capacity", "minimum_chunk", "final_capacity"]
)
def test_bounded_rejection_counters_distinguish_native_capacity_causes(manager, reason):
    req = request()
    req.prompt_len = 20
    counters = {}
    manager.impl.get_completion_demand.return_value = [4]
    if reason == "no_slot":
        manager._create_kv_cache.side_effect = None
        manager._create_kv_cache.return_value = None
    elif reason == "complete_match":
        create = manager._create_kv_cache.side_effect

        def complete(*args, **kwargs):
            cache = create(*args, **kwargs)
            cache.capacity = 20
            return cache

        manager._create_kv_cache.side_effect = complete
    elif reason == "matched_capacity":
        manager.impl.get_completion_demand.return_value = [100]
    elif reason == "minimum_chunk":
        manager.impl.get_completion_demand.return_value = [10]
    else:
        manager.impl.get_completion_demand.side_effect = [[4], [100]]
    assert (
        manager.execution_completion_try_bounded_prefill([], req, 5, 4, rejection_counters=counters)
        is None
    )
    assert counters == {"bounded_prefill_" + reason: 1}
    assert manager.kv_cache_map == {}


def test_admission_diagnostic_queries_keep_retired_capacity_and_decoder_promises(manager):
    decoder, prefill, retired, candidate = [request(index) for index in range(1, 5)]
    for req in (decoder, prefill, retired, candidate):
        req.state_value = LlmRequestState.CONTEXT_INIT.value
    decoder.state_value = LlmRequestState.GENERATION_IN_PROGRESS.value
    prefill.prompt_len, prefill.max_new_tokens = 20, 1
    retired.is_finished = True
    candidate.prompt_len, candidate.max_new_tokens = 12, 1
    caches = [SimpleNamespace(capacity=capacity, is_active=False) for capacity in (8, 4, 8, 4)]
    manager.kv_cache_map = dict(zip(range(1, 5), caches))
    manager.impl.get_storage_statistics.return_value = [
        SimpleNamespace(total=100, free=16, evictable=20, available=36, unavailable=64)
    ]
    queries = []

    def native_union(selected, capacities):
        queries.append((list(selected), list(capacities)))
        # A native-like shared prefix: the first four tokens are immutable and
        # common to every owner. Distinct futures therefore are not additive.
        lengths = [max(cache.capacity, final) for cache, final in zip(selected, capacities)]
        union = 4 + sum(length - 4 for length in lengths) if lengths else 0
        return [2 * union]

    manager.impl.get_completion_demand.side_effect = native_union
    before = [(cache.capacity, cache.is_active) for cache in caches]
    result = manager.execution_admission_demand_snapshot(
        [(decoder, caches[0], 0), (prefill, caches[1], 0), (retired, caches[2], 0)],
        candidate,
        caches[3],
        40,
    )
    assert [capacities for _, capacities in queries] == [
        [0, 0, 0],
        [12, 0, 0],
        [12, 20, 0],
        [12],
        [12, 0, 0, 12],
        [12, 0, 0, 12],
    ]
    assert result["unions"] == dict(
        allocated_capacity_union_blocks=12,
        allocated_plus_decoder_completion_union_blocks=16,
        current_commitment_union_blocks=32,
        candidate_alone_completion_union_blocks=12,
        current_plus_candidate_completion_union_blocks=40,
        allocated_plus_decoder_completion_plus_candidate_union_blocks=24,
        one_prefill_future_removed_plus_candidate_union_blocks=24,
    )
    assert result["owners"][2]["allocated_capacity_tokens"] == 8
    assert result["owners"][2]["requested_completion_tokens"] == 0
    assert result["one_prefill_future_removed_request_id"] == 2
    assert "unknown" in result["protected_align_hold_attribution"]
    assert result["allocated_capacity_is_hbm_occupancy"] is False
    assert result["gpu_pool_storage_slots"] == [
        dict(total=100, free=16, evictable=20, available=36, unavailable=64)
    ]
    assert [(cache.capacity, cache.is_active) for cache in caches] == before
    assert manager.kv_cache_map == dict(zip(range(1, 5), caches))
    manager._create_kv_cache.assert_not_called()
    manager.free_resources.assert_not_called()


@pytest.mark.parametrize("duplicate", [False, True])
def test_diagnostic_rejects_reused_id_or_changed_allocation_without_query(manager, duplicate):
    old, candidate = request(1), request(1 if duplicate else 2)
    old_cache, new_cache = object(), object()
    manager.kv_cache_map = {1: object(), candidate.py_request_id: new_cache}
    with pytest.raises(RuntimeError, match="distinct exact owners|identity changed"):
        manager.execution_admission_demand_snapshot([(old, old_cache, 0)], candidate, new_cache, 4)
    manager.impl.get_completion_demand.assert_not_called()
    manager._create_kv_cache.assert_not_called()
    manager.free_resources.assert_not_called()


def test_diagnostic_owner_limit_does_not_query_a_truncated_union(manager):
    result = manager.execution_admission_demand_snapshot(
        [(request(index), object(), 0) for index in range(65)], request(66), object(), 99
    )
    assert result == dict(unavailable="owner_limit", owner_count=65)
    manager.impl.get_completion_demand.assert_not_called()


@pytest.mark.parametrize("accepted", [False, True])
def test_diagnostic_callback_runs_before_candidate_cleanup_and_cannot_change_decision(
    manager, accepted
):
    req = request()
    observed = []

    def observe(cache, demand):
        assert manager.kv_cache_map[1] is cache
        observed.append((cache, demand))
        raise RuntimeError("diagnostic failure")

    assert (
        manager.execution_completion_try_admit(
            [], req, 5 if accepted else 4, admission_observer=observe
        )
        is accepted
    )
    assert observed[0][1] == 5
    assert manager.kv_cache_map == ({1: observed[0][0]} if accepted else {})
    assert manager.free_resources.call_count == int(not accepted)
    assert manager.impl.get_completion_demand.call_count == 1


def test_diagnostic_reports_no_slot_without_inventing_a_candidate(manager):
    manager._create_kv_cache.side_effect = None
    manager._create_kv_cache.return_value = None
    observer = Mock()
    assert not manager.execution_completion_try_admit([], request(), 5, admission_observer=observer)
    observer.assert_called_once_with(None, None)
    manager.impl.get_completion_demand.assert_not_called()
    manager.free_resources.assert_not_called()
