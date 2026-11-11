# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse bounds must protect the physical suffix after commit, not just lookup."""

from types import SimpleNamespace

import cuda.bindings.runtime as cudart
import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_alignment import _Reservation
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentRequest, WorkerAlignmentState
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CudaStream,
    GpuCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def cache_fixture(request):
    torch.empty(1, device="cuda")
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
            layers=[
                AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])
            ],
            enable_partial_reuse=True,
            reuse_match_backoff=getattr(request, "param", 0),
        )
    )
    caches = []
    stream = CudaStream(torch.cuda.current_stream().cuda_stream)

    def create(tokens=None, **kwargs):
        cache = core.create_kv_cache(None, tokens, expected_prompt_length=12, **kwargs)
        caches.append(cache)
        assert cache.resume(stream)
        assert cache.resize(12)
        return cache

    try:
        yield core, create, stream
    finally:
        for cache in reversed(caches):
            cache.close()
        core.shutdown()


def _pages(core, cache):
    (desc,) = core.get_aggregated_pages(list(core.all_buffer_ids))
    indices = list(cache.get_aggregated_page_indices(desc.layer_group_id))
    return desc, indices


def _fill(core, cache, value, start_token=0):
    torch.cuda.synchronize()
    desc, indices = _pages(core, cache)
    for ordinal, index in enumerate(indices):
        offset = max(0, min(4, start_token - 4 * ordinal)) * (desc.size // 4)
        if offset < desc.size:
            (err,) = cudart.cudaMemset(
                int(desc.base) + index * desc.stride + offset, value, desc.size - offset
            )
            assert int(err) == 0


def _read(core, cache):
    torch.cuda.synchronize()
    desc, indices = _pages(core, cache)
    result = np.empty((len(indices), desc.size), dtype=np.uint8)
    for ordinal, index in enumerate(indices):
        (err,) = cudart.cudaMemcpy(
            result[ordinal].ctypes.data,
            int(desc.base) + index * desc.stride,
            desc.size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert int(err) == 0
    return result


@pytest.mark.parametrize("limit", [None, 0, 4, 7, 8])
@pytest.mark.parametrize("lookup", [False, True])
@pytest.mark.parametrize("suspend", [False, True])
def test_commit_respects_execution_reuse_limit(cache_fixture, limit, lookup, suspend):
    core, create, stream = cache_fixture
    tokens = list(range(12))
    owner = create()
    _fill(core, owner, 11)
    owner.commit(tokens)
    _, original = _pages(core, owner)

    successor = create(tokens if lookup else None, max_reuse_tokens=limit)
    initial = (12 if limit is None else limit) if lookup else 0
    assert successor.num_committed_tokens == initial
    _fill(core, successor, 29, initial)
    _, fresh = _pages(core, successor)
    if suspend:
        successor.suspend()
        assert successor.resume(stream)
    successor.commit(tokens[initial:])
    # Partial lookup copies exactly its prefix into a private boundary block.
    # Commit may adopt full blocks only when they are wholly inside the bound.
    adopted = 12 if limit is None else limit // 4 * 4
    _, after = _pages(core, successor)
    assert after[: adopted // 4] == original[: adopted // 4]
    assert after[adopted // 4 :] == fresh[adopted // 4 :]
    assert all(a != b for a, b in zip(after[adopted // 4 :], original[adopted // 4 :]))
    actual = _read(core, successor).reshape(-1)
    bytes_per_token = actual.size // 12
    inherited = max(initial, adopted)
    assert np.all(actual[: inherited * bytes_per_token] == 11)
    assert np.all(actual[inherited * bytes_per_token :] == 29)
    assert np.all(_read(core, owner) == 11)
    assert core.probe_reuse(None, tokens) == 12
    assert successor.is_commit_allowed is (limit is None)


@pytest.mark.parametrize("limit", [0, 4])
def test_reuse_limit_still_publishes_fresh_blocks(cache_fixture, limit):
    core, create, _ = cache_fixture
    tokens = list(range(12))
    owner = create(tokens, max_reuse_tokens=limit)
    assert owner.num_committed_tokens == 0
    _fill(core, owner, 29)
    owner.commit(tokens)
    assert owner.is_commit_allowed
    successor = create(tokens)
    assert successor.num_committed_tokens == 12
    assert np.all(_read(core, successor) == 29)


def test_negative_reuse_limit_is_rejected(cache_fixture):
    core, _, _ = cache_fixture
    with pytest.raises(ValueError, match="max_reuse_tokens must be nonnegative"):
        core.create_kv_cache(max_reuse_tokens=-1)


@pytest.mark.parametrize("cache_fixture", [2], indirect=True)
@pytest.mark.parametrize("limit", [0, 4])
def test_reuse_limit_preserves_backoff_evidence(cache_fixture, limit):
    _, create, _ = cache_fixture
    tokens = list(range(12))
    owner = create()
    owner.commit(tokens)
    successor = create(tokens, max_reuse_tokens=limit)
    assert successor.num_committed_tokens == limit


@pytest.mark.parametrize("limit", [0, 8, 10, 12])
@pytest.mark.parametrize("exact", [False, True], ids=["runtime-bound", "exact-common-prefix"])
@pytest.mark.parametrize("isolated", [False, True])
def test_alignment_can_acquire_later_committed_pages_within_runtime_bound(
    cache_fixture, limit, exact, isolated
):
    core, create, _ = cache_fixture
    tokens = list(range(12))
    owner = create()
    _fill(core, owner, 11)
    owner.commit(tokens[:8])
    old = ExecutionVersion("tenant", "scope", 0)
    new = ExecutionVersion("tenant", "scope", 1)
    revised = AlignmentInput(tuple(range(13)), max_reuse_tokens=limit)
    request = AlignmentRequest("session", Revision("edit", old, new), revised)

    def reserve(revised_input):
        held = create(tokens[: revised_input.reuse_limit], max_reuse_tokens=limit)
        return _Reservation(int(held.num_committed_tokens), held.close, torch.cuda.synchronize)

    state = WorkerAlignmentState(
        reserve,
        lambda: None,
        reserve_isolated=(lambda alignment: reserve(alignment.revised_input)) if isolated else None,
    )
    try:
        state.observe(new, request)
        state.process()
        receipt = state.poll(request)
        acquired = min(limit, 8)
        assert receipt.reusable_prefix_length == acquired

        # Another native owner completes additional compatible pages while the
        # ALIGN hold remains live. Its bytes are committed before successor lookup.
        producer = create(tokens[:8])
        _fill(core, producer, 17, start_token=8)
        producer.commit(tokens[8:])
        _, produced_pages = _pages(core, producer)
        state.authorize_resume(receipt, acquired if exact else None)
        permitted = state.claim(new, 42, revised.input_token_ids, None, None)
        expected = acquired if exact else limit
        assert permitted == expected
        successor = create(tokens, max_reuse_tokens=permitted)
        assert successor.num_committed_tokens == expected
        state.confirm_attachment(new, 42, successor.num_committed_tokens)
        assert state.snapshot()["held_count"] == 0

        _fill(core, successor, 29, start_token=expected)
        _, private_pages = _pages(core, successor)
        successor.commit(tokens[expected:])
        _, actual_pages = _pages(core, successor)
        complete_blocks = expected // 4
        assert actual_pages[:complete_blocks] == produced_pages[:complete_blocks]
        assert actual_pages[complete_blocks:] == private_pages[complete_blocks:]
        assert all(
            page != produced
            for page, produced in zip(
                actual_pages[complete_blocks:], produced_pages[complete_blocks:]
            )
        )
        actual = _read(core, successor).reshape(-1)
        bytes_per_token = actual.size // 12
        first = min(expected, 8) * bytes_per_token
        end = expected * bytes_per_token
        assert np.all(actual[:first] == 11)
        assert np.all(actual[first:end] == 17)
        assert np.all(actual[end:] == 29)
        original = _read(core, producer).reshape(-1)
        assert np.all(original[: 8 * bytes_per_token] == 11)
        assert np.all(original[8 * bytes_per_token :] == 17)
    finally:
        state.close()
        state.process()


def test_isolated_zero_alignment_keeps_later_committed_prefix_and_future_publication(cache_fixture):
    core, create, _ = cache_fixture
    tokens = list(range(12))
    owner = create()
    _fill(core, owner, 11)
    old = ExecutionVersion("tenant", "scope", 0)
    new = ExecutionVersion("tenant", "scope", 1)
    revised = AlignmentInput(tuple(range(13)))
    request = AlignmentRequest("session", Revision("edit", old, new), revised)

    def reserve(alignment):
        held = create(tokens)
        return _Reservation(int(held.num_committed_tokens), held.close, torch.cuda.synchronize)

    state = WorkerAlignmentState(
        lambda _: pytest.fail("isolated handoff must not use drained preparation"),
        lambda: None,
        reserve_isolated=reserve,
    )
    try:
        state.observe(new, request)
        state.process()
        receipt = state.poll(request)
        assert receipt.reusable_prefix_length == 0
        state.authorize_resume(receipt)
        permitted = state.claim(new, 42, revised.input_token_ids, None, None)
        assert permitted is None

        # Accepted source work becomes canonical after RESUME is authorized.
        # Its remaining allocation stays private while the source remains live.
        owner.commit(tokens[:8])
        assert owner.is_active and owner.num_committed_tokens == 8
        successor = create(tokens, max_reuse_tokens=permitted)
        assert successor.num_committed_tokens == 8
        state.confirm_attachment(new, 42, successor.num_committed_tokens)
        assert state.snapshot()["held_count"] == 0

        _fill(core, successor, 29, start_token=8)
        successor.commit(tokens[8:])
        assert successor.num_committed_tokens == 12
        following = create(tokens)
        assert following.num_committed_tokens == 12
        actual = _read(core, following).reshape(-1)
        bytes_per_token = actual.size // 12
        assert np.all(actual[: 8 * bytes_per_token] == 11)
        assert np.all(actual[8 * bytes_per_token :] == 29)
        assert np.all(_read(core, owner) == 11)
    finally:
        state.close()
        state.process()


def test_bounded_prefill_preserves_anchor_completion_and_promotes_after_release(cache_fixture):
    core, create, _ = cache_fixture
    total_blocks = core.get_storage_statistics(0)[0].total
    assert total_blocks >= 8
    anchor_end = (total_blocks - 2) * 4
    anchor_start = anchor_end - 8
    anchor = create()
    pending = create()
    assert pending.resize(8)
    # Resume both owners before filling the pool: native admission refuses new
    # contexts above its 97% utilization threshold. The invariant under test is
    # reserved growth of already-admitted owners, which may fill the pool.
    assert anchor.resize(anchor_start)
    _fill(core, anchor, 11)
    _fill(core, pending, 29)

    # Charge the entire physical pool: two blocks hold unfinished prefill, and
    # the decoder retains enough capacity for its last two blocks of growth.
    assert core.get_completion_demand([anchor, pending], [anchor_end, 8]) == [total_blocks]
    assert core.get_completion_demand([anchor, pending], [anchor_end, 20]) == [total_blocks + 3]
    assert anchor.resize(anchor_end)
    _fill(core, anchor, 11, start_token=anchor_start)
    assert core.get_completion_demand([anchor, pending], [anchor_end, 8]) == [total_blocks]
    assert np.all(_read(core, anchor) == 11)
    assert np.all(_read(core, pending) == 29)

    anchor.close()
    assert core.get_completion_demand([pending], [20]) == [5]
    assert pending.resize(20)
    _fill(core, pending, 17, start_token=8)
    data = _read(core, pending)
    assert np.all(data[:2] == 29)
    assert np.all(data[2:] == 17)


def test_bounded_prefill_demand_never_undercharges_allocated_pages(cache_fixture):
    core, create, _ = cache_fixture
    anchor = create()
    pending = create()
    assert pending.resize(8)
    assert core.get_completion_demand([anchor, pending], [20, 0]) == [7]
    assert pending.resize(12)
    assert core.get_completion_demand([anchor, pending], [20, 8]) == [8]
    pending.close()
    assert core.get_completion_demand([anchor], [20]) == [5]


@pytest.mark.parametrize("resume", [False, True])
def test_bounded_prefill_counts_partial_match_source_during_private_copy(cache_fixture, resume):
    core, create, stream = cache_fixture
    anchor = create()
    tokens = list(range(12))
    _fill(core, anchor, 11)
    anchor.commit(tokens)
    pending = core.create_kv_cache(None, tokens[:7], expected_prompt_length=20)
    try:
        assert pending.num_committed_tokens == 7
        # The full shared block saves one destination; the partial block needs
        # its own destination even though its source is owned by the anchor.
        assert core.get_completion_demand([anchor, pending], [20, 8]) == [6]
        # Without another owner in the query, the immutable partial source must
        # also be charged while the suspended cache still holds that source.
        assert core.get_completion_demand([pending], [8]) == [3]
        if resume:
            assert pending.resume(stream)
            assert pending.resize(8)
            _fill(core, pending, 29, start_token=7)
            assert core.get_completion_demand([anchor, pending], [20, 8]) == [6]
            actual = _read(core, pending).reshape(-1)
            bytes_per_token = actual.size // 8
            assert np.all(actual[: 7 * bytes_per_token] == 11)
            assert np.all(actual[7 * bytes_per_token :] == 29)
            assert np.all(_read(core, anchor) == 11)
    finally:
        pending.close()


def test_bounded_prefill_full_bypass_preserves_all_native_reservations(cache_fixture):
    core, create, _ = cache_fixture
    total = core.get_storage_statistics(0)[0].total
    anchor, partial, bypass = create(), create(), create()
    assert partial.resize(8)
    assert bypass.resize(4)
    anchor_end = (total - 3) * 4
    assert anchor.resize(anchor_end - 8)
    _fill(core, anchor, 11)
    _fill(core, partial, 29)
    _fill(core, bypass, 37)
    # Both the partial prefix and the separately completing bypass coexist
    # without reducing the anchor's protected growth or sharing writable pages.
    assert core.get_completion_demand([anchor, partial, bypass], [anchor_end, 8, 4]) == [total]
    assert core.get_completion_demand([anchor, partial, bypass], [anchor_end, 20, 4]) == [total + 3]
    assert anchor.resize(anchor_end)
    _fill(core, anchor, 11, start_token=anchor_end - 8)
    assert np.all(_read(core, anchor) == 11)
    assert np.all(_read(core, partial) == 29)
    assert np.all(_read(core, bypass) == 37)
    # Finishing an owner releases its future growth, never its actual pages.
    assert core.get_completion_demand([anchor, partial, bypass], [0, 8, 4]) == [total]
    bypass.close()
    assert core.get_completion_demand([anchor, partial], [anchor_end, 8]) == [total - 1]
    anchor.close()
    assert partial.resize(20)
    _fill(core, partial, 17, start_token=8)
    data = _read(core, partial)
    assert np.all(data[:2] == 29)
    assert np.all(data[2:] == 17)


def test_admission_diagnostic_counterfactuals_preserve_real_shared_pages(cache_fixture):
    core, create, _ = cache_fixture
    tokens = list(range(12))
    anchor = create()
    _fill(core, anchor, 11)
    anchor.commit(tokens)
    prefill, retired = create(), create()
    assert prefill.resize(8)
    assert retired.resize(4)
    _fill(core, prefill, 29)
    _fill(core, retired, 37)
    successor = create(tokens)
    caches = [anchor, prefill, retired, successor]
    requests = [
        SimpleNamespace(
            py_request_id=index,
            prompt_len=prompt,
            max_new_tokens=1,
            is_finished=index == 2,
            state_value=(
                LlmRequestState.GENERATION_IN_PROGRESS.value
                if index == 0
                else LlmRequestState.CONTEXT_INIT.value
            ),
        )
        for index, prompt in enumerate((20, 24, 12, 20))
    ]
    wrapper = SimpleNamespace(impl=core, kv_cache_map=dict(enumerate(caches)))
    before = [(cache.capacity, cache.is_active, _read(core, cache)) for cache in caches]
    full_union = core.get_completion_demand(caches, [20, 24, 0, 20])[0]
    result = KVCacheManagerV2.execution_admission_demand_snapshot(
        wrapper,
        list(zip(requests[:3], caches[:3], (0, 0, 0))),
        requests[3],
        successor,
        full_union,
    )
    unions = result["unions"]
    assert (
        unions["allocated_capacity_union_blocks"]
        == core.get_completion_demand(caches[:3], [0, 0, 0])[0]
    )
    assert (
        unions["allocated_plus_decoder_completion_plus_candidate_union_blocks"]
        == core.get_completion_demand(caches, [20, 0, 0, 20])[0]
    )
    assert (
        unions["one_prefill_future_removed_plus_candidate_union_blocks"]
        == core.get_completion_demand(caches, [20, 0, 0, 20])[0]
    )
    assert unions["current_plus_candidate_completion_union_blocks"] == full_union
    # The retired owner retains its physical four-token allocation even though
    # the diagnostic requests zero future growth for it.
    assert (
        core.get_completion_demand(caches, [20, 0, 0, 20])[0]
        > core.get_completion_demand([anchor, prefill, successor], [20, 0, 20])[0]
    )
    for cache, (capacity, active, content) in zip(caches, before):
        assert cache.capacity == capacity and cache.is_active is active
        assert np.array_equal(_read(core, cache), content)
    assert successor.num_committed_tokens == 12
    assert result["gpu_pool_storage_slots"][0]["available"] is not None
