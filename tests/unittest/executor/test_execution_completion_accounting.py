# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion credit must retain the exact native owner that proved sharing."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

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

    def create(request_id, *args, **kwargs):
        cache = SimpleNamespace()
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
