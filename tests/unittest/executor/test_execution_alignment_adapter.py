# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual V1/V2 reservation adapters with controlled native APIs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentSafetyError

pytestmark = pytest.mark.cpu_only
INPUT = AlignmentInput(tuple(range(17)), cache_salt="shared", lora_task_id=3)


def assert_v1_release(manager):
    request = manager.impl.add_sequence_batch.call_args.args[1][0]
    assert request.is_dummy_request
    manager.impl.remove_sequence.assert_called_once_with(99, request, False)


@pytest.fixture
def engine():
    manager = Mock(enable_block_reuse=True, tokens_per_block=4)
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


@pytest.mark.parametrize("v2", [False, True])
def test_reclaim_waits_on_manager_copy_stream(engine, v2):
    engine._is_kv_manager_v2 = v2
    manager = engine.kv_cache_manager
    manager.impl.create_kv_cache.return_value = Mock(num_committed_tokens=7)
    hold = KVAlignmentAdapter(engine).reserve(INPUT)
    manager._stream.synchronize.reset_mock()
    hold.settle()
    manager._stream.synchronize.assert_called_once()
    hold.close()
    hold.close()
    if v2:
        manager.impl.create_kv_cache.return_value.close.assert_called_once()
    else:
        assert_v1_release(manager)
