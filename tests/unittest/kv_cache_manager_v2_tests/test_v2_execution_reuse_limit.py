# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse bounds must protect the physical suffix after commit, not just lookup."""

import cuda.bindings.runtime as cudart
import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_alignment import _Reservation
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
def test_alignment_can_acquire_later_committed_pages_within_runtime_bound(
    cache_fixture, limit, exact
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

    state = WorkerAlignmentState(reserve, lambda: None)
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
