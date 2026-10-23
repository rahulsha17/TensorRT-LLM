# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual V1/V2 reservation adapters with controlled native APIs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
from tensorrt_llm.execution_alignment import (
    AlignmentInput,
    AlignmentRequest,
    AlignmentSafetyError,
    WorkerAlignmentState,
)
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
INPUT = AlignmentInput(tuple(range(17)), cache_salt="shared", lora_task_id=3)


def assert_v1_release(manager):
    request = manager.impl.add_sequence_batch.call_args.args[1][0]
    assert request.is_dummy_request
    manager.impl.remove_sequence.assert_called_once_with(99, request, False)


@pytest.fixture
def engine():
    manager = Mock(enable_block_reuse=True, tokens_per_block=4)
    manager.impl.tokens_per_block = 4
    manager.impl.cache_tier_list = [0, 1]
    manager.impl.get_life_cycle_pool_group_indices.return_value = [0]
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=16)]
    manager._derive_reuse_salt.return_value = 42
    manager.probe_prefix_match_length.return_value = 12
    executor = SimpleNamespace(
        kv_cache_manager=manager,
        max_seq_len=32,
        _is_kv_manager_v2=True,
        executor_request_queue=Mock(),
    )
    executor.executor_request_queue.reserve_request_id.return_value = 99
    return executor


def test_v2_owns_a_suspended_cache_with_exact_namespace_and_last_token_excluded(engine):
    cache = Mock(num_committed_tokens=7)
    engine.kv_cache_manager.impl.create_kv_cache.return_value = cache
    hold = KVAlignmentAdapter(engine).reserve(INPUT)
    assert hold.num_tokens == 7
    args, kwargs = engine.kv_cache_manager.impl.create_kv_cache.call_args
    assert args[0].lora_id == 3 and args[0].salt == 42
    assert args[1] == list(INPUT.input_token_ids[:-1])
    assert kwargs["expected_prompt_length"] == 17
    cache.resume.assert_not_called()
    cache.close.assert_not_called()
    hold.close()
    hold.close()
    cache.close.assert_called_once()


@pytest.mark.parametrize("v2", [False, True])
def test_zero_limit_never_acquires_cached_pages(engine, v2):
    engine._is_kv_manager_v2 = v2
    hold = KVAlignmentAdapter(engine).reserve(replace(INPUT, max_reuse_tokens=0))
    assert hold.num_tokens == 0
    assert engine.kv_cache_manager.mock_calls == []
    hold.settle()
    hold.close()


@pytest.mark.parametrize("limit", [1, 7, 8, 16])
def test_v2_lookup_cannot_cross_runtime_boundary(engine, limit):
    manager = engine.kv_cache_manager
    manager.impl.create_kv_cache.return_value = Mock(num_committed_tokens=0)
    hold = KVAlignmentAdapter(engine).reserve(replace(INPUT, max_reuse_tokens=limit))
    args, kwargs = manager.impl.create_kv_cache.call_args
    assert args[1] == list(INPUT.input_token_ids[:limit])
    assert kwargs["expected_prompt_length"] == len(INPUT.input_token_ids)
    hold.close()


@pytest.mark.parametrize("limit,expected", [(1, 0), (7, 4), (8, 8), (16, 12)])
def test_v1_rounds_runtime_boundary_down_to_complete_blocks(engine, limit, expected):
    engine._is_kv_manager_v2 = False
    manager = engine.kv_cache_manager

    def allocate(infos, requests):
        assert infos == [(99, expected + 1, 1)]
        assert requests[0].get_tokens(0) == list(INPUT.input_token_ids[: expected + 1])
        requests[0].set_prepopulated_prompt_len(expected, 4)

    manager.impl.add_sequence_batch.side_effect = allocate
    hold = KVAlignmentAdapter(engine).reserve(replace(INPUT, max_reuse_tokens=limit))
    assert hold.num_tokens == expected
    if expected == 0:
        manager.impl.add_sequence_batch.assert_not_called()
    hold.close()


def test_v1_acquires_native_sequence_and_reports_actual_match(engine):
    engine._is_kv_manager_v2 = False
    manager = engine.kv_cache_manager
    order = []

    def allocate(infos, requests):
        assert infos == [(99, 13, 1)]
        req = requests[0]
        assert not req.is_dummy_request
        assert req.cache_salt == "shared" and req.lora_task_id == 3
        # Native allocation may acquire less than the advisory probe predicted.
        req.set_prepopulated_prompt_len(8, 4)
        order.append("allocate")

    manager.impl.add_sequence_batch.side_effect = allocate
    manager._stream.synchronize.side_effect = lambda: order.append("wait copies")
    hold = KVAlignmentAdapter(engine).reserve(INPUT)
    assert hold.num_tokens == 8
    assert order == ["allocate", "wait copies"]
    hold.close()
    assert_v1_release(manager)
    manager.impl.store_blocks_for_reuse.assert_not_called()


def test_v1_failure_settles_copies_before_releasing_partial_allocation(engine):
    engine._is_kv_manager_v2 = False
    manager = engine.kv_cache_manager
    manager.impl.add_sequence_batch.side_effect = RuntimeError("OOM")
    with pytest.raises(RuntimeError, match="OOM"):
        KVAlignmentAdapter(engine).reserve(INPUT)
    manager.impl.refresh_blocks.assert_called_once()
    manager._stream.synchronize.assert_called_once()
    assert_v1_release(manager)


def test_v1_unknown_copy_completion_retains_native_sequence(engine):
    engine._is_kv_manager_v2 = False
    manager = engine.kv_cache_manager
    manager.impl.add_sequence_batch.side_effect = RuntimeError("OOM")
    manager._stream.synchronize.side_effect = RuntimeError("device failure")
    with pytest.raises(AlignmentSafetyError):
        KVAlignmentAdapter(engine).reserve(INPUT)
    manager.impl.remove_sequence.assert_not_called()


@pytest.mark.parametrize("v2", [False, True])
def test_reuse_disabled_yields_zero_without_native_allocation(engine, v2):
    engine._is_kv_manager_v2 = v2
    engine.kv_cache_manager.enable_block_reuse = False
    assert KVAlignmentAdapter(engine).reserve(INPUT).num_tokens == 0
    assert engine.kv_cache_manager.impl.mock_calls == []


def test_v1_cache_miss_does_not_allocate(engine):
    engine._is_kv_manager_v2 = False
    engine.kv_cache_manager.probe_prefix_match_length.return_value = 0
    assert KVAlignmentAdapter(engine).reserve(INPUT).num_tokens == 0
    engine.executor_request_queue.reserve_request_id.assert_not_called()


def test_input_length_rejected_before_native_allocation(engine):
    engine.max_seq_len = 4
    with pytest.raises(ValueError, match="max_seq_len"):
        KVAlignmentAdapter(engine).reserve(INPUT)
    assert engine.kv_cache_manager.impl.mock_calls == []


def test_manager_limit_used_when_executor_limit_is_unset(engine):
    engine.max_seq_len = None
    engine.kv_cache_manager.max_seq_len = 4
    with pytest.raises(ValueError, match="max_seq_len"):
        KVAlignmentAdapter(engine).reserve(INPUT)


def test_v1_reclaim_waits_on_manager_copy_stream(engine: SimpleNamespace) -> None:
    engine._is_kv_manager_v2 = False
    manager = engine.kv_cache_manager
    hold = KVAlignmentAdapter(engine).reserve(INPUT)
    manager._stream.synchronize.reset_mock()
    hold.settle()
    manager._stream.synchronize.assert_called_once()
    hold.close()
    hold.close()
    assert_v1_release(manager)


@pytest.mark.parametrize("count", [0, 7, 8])
def test_v2_settle_does_not_wait_on_unrelated_manager_work(
    engine: SimpleNamespace, count: int
) -> None:
    manager = engine.kv_cache_manager
    cache = Mock(num_committed_tokens=count)
    manager.impl.create_kv_cache.return_value = cache
    manager._stream.synchronize.side_effect = RuntimeError("unrelated manager work")
    adapter = KVAlignmentAdapter(engine)
    hold = adapter.reserve(INPUT)
    hold.settle()
    hold.settle()
    manager._stream.synchronize.assert_not_called()
    cache.resume.assert_not_called()
    cache.close.assert_not_called()
    assert hold.num_tokens == count
    assert adapter._reserved_v2_blocks == (count + 3) // 4
    hold.close()
    hold.close()
    cache.close.assert_called_once()
    assert adapter._reserved_v2_blocks == 0


def test_v2_reservation_pressure_limits_new_promise_and_releases_budget(engine):
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=4)]
    caches = [Mock(num_committed_tokens=n) for n in (7, 8, 8)]
    manager.impl.create_kv_cache.side_effect = caches
    adapter = KVAlignmentAdapter(engine)
    first = adapter.reserve(INPUT)
    second = adapter.reserve(INPUT)
    assert (first.num_tokens, second.num_tokens) == (7, 8)
    assert manager.impl.create_kv_cache.call_args.args[1] == list(INPUT.input_token_ids[:8])
    assert adapter.reserve(INPUT).num_tokens == 0
    assert manager.impl.create_kv_cache.call_count == 2
    caches[0].close.assert_not_called()
    caches[1].close.assert_not_called()
    first.close()
    first.close()
    third = adapter.reserve(INPUT)
    assert third.num_tokens == 8
    second.close()
    third.close()
    assert adapter._reserved_v2_blocks == 0
    for cache in caches:
        cache.close.assert_called_once()


def test_v2_capacity_accounts_for_lifecycles_sharing_each_pool(engine):
    manager = engine.kv_cache_manager
    manager.impl.get_life_cycle_pool_group_indices.return_value = [0, 0, 1]
    manager.impl.get_storage_statistics.return_value = [
        SimpleNamespace(total=6),
        SimpleNamespace(total=5),
    ]
    manager.impl.create_kv_cache.return_value = Mock(num_committed_tokens=12)
    adapter = KVAlignmentAdapter(engine)
    hold = adapter.reserve(INPUT)
    assert hold.num_tokens == 12
    assert manager.impl.create_kv_cache.call_args.args[1] == list(INPUT.input_token_ids[:12])
    assert adapter.reserve(INPUT).num_tokens == 0
    hold.close()


@pytest.mark.parametrize("limit", [1, 3, 4, 7, 8, 16])
def test_v2_capacity_and_runtime_limit_both_bound_reservation(engine, limit):
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=2)]
    expected = min(limit, 8)
    cache = Mock(num_committed_tokens=expected)
    manager.impl.create_kv_cache.return_value = cache
    adapter = KVAlignmentAdapter(engine)
    hold = adapter.reserve(replace(INPUT, max_reuse_tokens=limit))
    assert hold.num_tokens == expected
    assert manager.impl.create_kv_cache.call_args.args[1] == list(INPUT.input_token_ids[:expected])
    assert adapter._reserved_v2_blocks == (expected + 3) // 4
    hold.close()
    assert adapter._reserved_v2_blocks == 0


@pytest.mark.parametrize("capacity", [0, 1])
def test_v2_required_prefix_is_not_silently_weakened_under_pressure(engine, capacity):
    engine.kv_cache_manager.impl.get_storage_statistics.return_value = [
        SimpleNamespace(total=capacity)
    ]
    with pytest.raises(RuntimeError, match="required prefix exceeds"):
        KVAlignmentAdapter(engine).reserve(replace(INPUT, min_reuse_tokens=8))
    engine.kv_cache_manager.impl.create_kv_cache.assert_not_called()


def test_v2_without_backing_tier_does_not_pin_recovery_workspace(engine):
    manager = engine.kv_cache_manager
    manager.impl.cache_tier_list = [0]
    hold = KVAlignmentAdapter(engine).reserve(INPUT)
    assert hold.num_tokens == 0
    manager.impl.create_kv_cache.assert_not_called()
    manager.impl.get_storage_statistics.assert_not_called()
    hold.settle()
    hold.close()
    manager._stream.synchronize.assert_not_called()


@pytest.mark.parametrize("totals", [[0], []])
def test_v2_empty_backing_capacity_does_not_create_a_hold(engine, totals):
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=n) for n in totals]
    manager.impl.get_life_cycle_pool_group_indices.return_value = list(range(len(totals)))
    assert KVAlignmentAdapter(engine).reserve(INPUT).num_tokens == 0
    manager.impl.create_kv_cache.assert_not_called()


def test_v2_colder_tier_does_not_expand_intermediate_capacity(engine):
    manager = engine.kv_cache_manager
    manager.impl.cache_tier_list = [0, 1, 2]
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=1)]
    manager.impl.create_kv_cache.return_value = Mock(num_committed_tokens=4)
    hold = KVAlignmentAdapter(engine).reserve(INPUT)
    assert hold.num_tokens == 4
    manager.impl.get_storage_statistics.assert_called_once_with(1)
    manager.impl.get_life_cycle_pool_group_indices.assert_called_once_with(1)
    hold.close()


def test_v2_cache_miss_does_not_consume_reservation_capacity(engine):
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=1)]
    caches = [Mock(num_committed_tokens=n) for n in (0, 4)]
    manager.impl.create_kv_cache.side_effect = caches
    adapter = KVAlignmentAdapter(engine)
    miss = adapter.reserve(INPUT)
    hit = adapter.reserve(INPUT)
    assert (miss.num_tokens, hit.num_tokens) == (0, 4)
    miss.close()
    assert adapter._reserved_v2_blocks == 1
    hit.close()
    assert adapter._reserved_v2_blocks == 0


def test_v2_failed_native_close_does_not_return_capacity(engine: SimpleNamespace) -> None:
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=2)]
    cache = Mock(num_committed_tokens=8)
    manager.impl.create_kv_cache.return_value = cache
    adapter = KVAlignmentAdapter(engine)
    hold = adapter.reserve(INPUT)
    hold.settle()
    cache.close.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        hold.close()
    assert adapter.reserve(INPUT).num_tokens == 0
    assert adapter._reserved_v2_blocks == 2
    cache.close.assert_called_once()
    manager._stream.synchronize.assert_not_called()


def test_v2_settle_retains_capacity_until_native_close(engine: SimpleNamespace) -> None:
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=2)]
    cache = Mock(num_committed_tokens=8)
    manager.impl.create_kv_cache.return_value = cache
    adapter = KVAlignmentAdapter(engine)
    hold = adapter.reserve(INPUT)
    manager._stream.synchronize.side_effect = RuntimeError("copy completion unknown")
    hold.settle()
    manager._stream.synchronize.assert_not_called()
    cache.resume.assert_not_called()
    cache.close.assert_not_called()
    assert adapter.reserve(INPUT).num_tokens == 0
    assert adapter._reserved_v2_blocks == 2
    hold.close()
    assert adapter._reserved_v2_blocks == 0
    cache.close.assert_called_once()


@pytest.mark.parametrize("close_fails", [False, True])
def test_v2_reclaim_preserves_transfer_drain_and_close_failure_ownership(
    engine: SimpleNamespace, close_fails: bool
) -> None:
    manager = engine.kv_cache_manager
    cache = Mock(num_committed_tokens=8)
    manager.impl.create_kv_cache.return_value = cache
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=2)]
    adapter = KVAlignmentAdapter(engine)
    state = WorkerAlignmentState(adapter.reserve, lambda: None)
    old = ExecutionVersion("tenant", "scope", 0)
    new = ExecutionVersion("tenant", "scope", 1)
    request = AlignmentRequest("session", Revision("edit", old, new), INPUT)
    state.observe(new, request)
    state.process()
    receipt = state.poll(request)
    drained = Mock(return_value=False)
    state.retain_transfer(request, drained)
    state.authorize_resume(receipt, 8)
    assert state.claim(new, 42, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id) == 8
    state.confirm_attachment(new, 42, 8)
    cache.close.assert_not_called()
    assert adapter._reserved_v2_blocks == 2
    drained.return_value = True
    if close_fails:
        cache.close.side_effect = RuntimeError("native close completion unknown")
        with pytest.raises(AlignmentSafetyError, match="RECLAIM completion is unknown"):
            state.confirm_attachment(new, 42, 8)
        assert adapter._reserved_v2_blocks == 2
        assert adapter.reserve(INPUT).num_tokens == 0
        with pytest.raises(AlignmentSafetyError):
            state.process()
    else:
        state.confirm_attachment(new, 42, 8)
        state.confirm_attachment(new, 42, 8)
        assert adapter._reserved_v2_blocks == 0
    cache.close.assert_called_once()
    manager._stream.synchronize.assert_not_called()
    cache.resume.assert_not_called()


def test_v2_backing_capacity_shrink_keeps_existing_holds_charged(engine):
    manager = engine.kv_cache_manager
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=4)]
    cache = Mock(num_committed_tokens=12)
    manager.impl.create_kv_cache.return_value = cache
    adapter = KVAlignmentAdapter(engine)
    hold = adapter.reserve(INPUT)
    manager.impl.get_storage_statistics.return_value = [SimpleNamespace(total=2)]
    assert adapter.reserve(INPUT).num_tokens == 0
    assert adapter._reserved_v2_blocks == 3
    cache.close.assert_not_called()
    hold.close()
    manager.impl.create_kv_cache.return_value = Mock(num_committed_tokens=8)
    smaller = adapter.reserve(INPUT)
    assert smaller.num_tokens == 8
    assert manager.impl.create_kv_cache.call_args.args[1] == list(INPUT.input_token_ids[:8])
    smaller.close()
    assert adapter._reserved_v2_blocks == 0


@pytest.mark.parametrize("count", [-1, 5])
def test_v2_invalid_native_prefix_is_released_without_charging_capacity(engine, count):
    manager = engine.kv_cache_manager
    cache = Mock(num_committed_tokens=count)
    manager.impl.create_kv_cache.return_value = cache
    adapter = KVAlignmentAdapter(engine)
    with pytest.raises(RuntimeError, match="reservation limit"):
        adapter.reserve(replace(INPUT, max_reuse_tokens=4))
    cache.close.assert_called_once()
    assert adapter._reserved_v2_blocks == 0


def test_v2_failed_partial_acquisition_cleanup_is_fatal(engine):
    manager = engine.kv_cache_manager
    cache = Mock(num_committed_tokens=5)
    cache.close.side_effect = RuntimeError("native close failed")
    manager.impl.create_kv_cache.return_value = cache
    with pytest.raises(AlignmentSafetyError, match="V2 reservation cleanup is unknown"):
        KVAlignmentAdapter(engine).reserve(replace(INPUT, max_reuse_tokens=4))
