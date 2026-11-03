# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A suspended ALIGN hold retains native pages without draining another stream."""

import faulthandler
import threading
from types import SimpleNamespace

import cuda.bindings.runtime as cudart
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.execution_alignment import KVAlignmentAdapter
from tensorrt_llm.bindings.internal import runtime as runtime_bindings
from tensorrt_llm.execution_alignment import AlignmentInput
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CudaStream,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    _introspection,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("num_tokens", [4, 8])
def test_suspended_hold_reclaims_while_successor_reader_is_pending(num_tokens: int) -> None:
    """Use a host-controlled stream gate; no elapsed-time performance assertion."""
    torch.empty(1, device="cuda")
    manager_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()
    entered = threading.Event()
    release = threading.Event()
    timed_out = threading.Event()
    sentinel = torch.cuda.Event()
    reader_done = torch.cuda.Event()
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(quota=4 << 20), HostCacheTierConfig(quota=4 << 20)],
            layers=[
                AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])
            ],
            enable_partial_reuse=True,
        )
    )
    owner = successor = hold = None
    # A C-level watchdog also bounds failures that hold the GIL inside a native API.
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        tokens = list(range(num_tokens))
        owner = core.create_kv_cache(None, None, expected_prompt_length=num_tokens)
        assert owner.resume(CudaStream(producer_stream.cuda_stream))
        assert owner.resize(num_tokens)
        (desc,) = core.get_aggregated_pages(list(core.all_buffer_ids))
        original = list(owner.get_aggregated_page_indices(desc.layer_group_id))
        assert len(original) == num_tokens // 4
        for index in original:
            (error,) = cudart.cudaMemsetAsync(
                int(desc.base) + index * desc.stride, 37, desc.size, producer_stream.cuda_stream
            )
            assert int(error) == 0
        producer_stream.synchronize()
        owner.commit(tokens)
        manager = SimpleNamespace(
            impl=core,
            enable_block_reuse=True,
            _stream=manager_stream,
            _derive_reuse_salt=lambda _: None,
            _supports_dense_resume_capacity=lambda: True,
        )
        adapter = KVAlignmentAdapter(
            SimpleNamespace(kv_cache_manager=manager, max_seq_len=32, _is_kv_manager_v2=True),
            keep_resident=False,
        )
        hold = adapter.reserve(AlignmentInput(tuple(tokens + [99])))
        assert hold.num_tokens == num_tokens
        assert adapter._reserved_v2_blocks == len(original)
        owner.close()
        assert not _introspection.all_tree_pages_droppable(core)
        # The ALIGN reservation is the only cache owner until this lookup.
        successor = core.create_kv_cache(None, tokens, expected_prompt_length=num_tokens)
        assert successor.num_committed_tokens == num_tokens
        assert successor.resume(CudaStream(manager_stream.cuda_stream))
        assert successor.resize(num_tokens)
        assert list(successor.get_aggregated_page_indices(desc.layer_group_id)) == original
        output = torch.empty((len(original), desc.size), dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()

        def gate() -> None:
            # CUDA calls are forbidden inside a CUDA host function.
            entered.set()
            if not release.wait(30):
                timed_out.set()

        runtime_bindings.launch_hostfunc(manager_stream.cuda_stream, True, False, gate)
        sentinel.record(manager_stream)
        for ordinal, index in enumerate(original):
            (error,) = cudart.cudaMemcpyAsync(
                output.data_ptr() + ordinal * desc.size,
                int(desc.base) + index * desc.stride,
                desc.size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                manager_stream.cuda_stream,
            )
            assert int(error) == 0
        reader_done.record(manager_stream)
        assert entered.wait(30), "CUDA host gate did not start"
        assert not sentinel.query()
        assert not reader_done.query()
        hold.settle()
        assert not sentinel.query(), "settle drained unrelated manager-stream work"
        assert not reader_done.query()
        assert not timed_out.is_set()
        assert adapter._reserved_v2_blocks == len(original)
        hold.close()
        hold.close()
        assert adapter._reserved_v2_blocks == 0
        assert not sentinel.query(), "native suspended-cache close drained the manager stream"
        assert not reader_done.query()
        assert successor.num_committed_tokens == num_tokens
        assert not _introspection.all_tree_pages_droppable(core)
        release.set()
        manager_stream.synchronize()
        assert not timed_out.is_set()
        assert reader_done.query()
        assert torch.all(output == 37).item()
        assert list(successor.get_aggregated_page_indices(desc.layer_group_id)) == original
        successor.close()
        assert _introspection.all_tree_pages_droppable(core)
    finally:
        release.set()
        manager_stream.synchronize()
        producer_stream.synchronize()
        if hold is not None:
            hold.close()
        if successor is not None:
            successor.close()
        if owner is not None:
            owner.close()
        core.shutdown()
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("snapshot", [False, True])
def test_strict_partial_isolation_copies_private_tail_while_old_work_is_pending(
    monkeypatch, snapshot
) -> None:
    """Strict seven-token promise; old and new writers never share a tail."""
    monkeypatch.setenv("TRTLLM_RETIRE_ISOLATED_PARTIAL", "1")
    torch.empty(1, device="cuda")
    old_stream, next_stream = torch.cuda.Stream(), torch.cuda.Stream()
    entered, release, timed_out = threading.Event(), threading.Event(), threading.Event()
    pending = torch.cuda.Event()
    output = torch.empty((2, 4096), dtype=torch.uint8, device="cuda")
    old_output = torch.empty((4096,), dtype=torch.uint8, device="cuda")
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(quota=4 << 20), HostCacheTierConfig(quota=4 << 20)],
            layers=[
                AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])
            ],
            enable_partial_reuse=True,
            commit_min_snapshot=snapshot,
        )
    )
    owner = successor = hold = None
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        owner = core.create_kv_cache(None, None, expected_prompt_length=8)
        assert owner.resume(CudaStream(old_stream.cuda_stream))
        assert owner.resize(8)
        (desc,) = core.get_aggregated_pages(list(core.all_buffer_ids))
        original = list(owner.get_aggregated_page_indices(desc.layer_group_id))
        for index in original:
            (error,) = cudart.cudaMemsetAsync(
                int(desc.base) + index * desc.stride, 37, desc.size, old_stream.cuda_stream
            )
            assert int(error) == 0
        old_stream.synchronize()
        owner.commit(list(range(7 if snapshot else 8)))
        old_stream.synchronize()
        manager = SimpleNamespace(
            impl=core,
            enable_block_reuse=True,
            _stream=next_stream,
            _derive_reuse_salt=lambda _: None,
            _supports_dense_resume_capacity=lambda: True,
        )
        adapter = KVAlignmentAdapter(
            SimpleNamespace(
                kv_cache_manager=manager,
                max_seq_len=32,
                _is_kv_manager_v2=True,
            )
        )
        revised = AlignmentInput(tuple(range(8)), min_reuse_tokens=7, max_reuse_tokens=7)
        torch.cuda.synchronize()

        def gate():
            entered.set()
            if not release.wait(30):
                timed_out.set()

        runtime_bindings.launch_hostfunc(old_stream.cuda_stream, True, False, gate)
        # A full old prefix is read-only. A partial snapshot has a separate old
        # private tail, so that old writer may overwrite its own entire page.
        if snapshot:
            (error,) = cudart.cudaMemsetAsync(
                int(desc.base) + original[1] * desc.stride, 99, desc.size, old_stream.cuda_stream
            )
        else:
            (error,) = cudart.cudaMemcpyAsync(
                old_output.data_ptr(),
                int(desc.base) + original[1] * desc.stride,
                desc.size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                old_stream.cuda_stream,
            )
        assert int(error) == 0
        pending.record(old_stream)
        assert entered.wait(30) and not pending.query()
        hold = adapter.reserve_isolated(revised)
        assert hold is not None and hold.num_tokens == 7
        assert not pending.query() and not timed_out.is_set()
        successor = core.create_kv_cache(None, list(range(7)), expected_prompt_length=8)
        assert successor.num_committed_tokens == 7
        assert successor.isolated_prefix_ready(allow_partial=True)
        assert successor.resume(CudaStream(next_stream.cuda_stream))
        assert successor.resize(8)
        inherited = list(successor.get_aggregated_page_indices(desc.layer_group_id))
        assert inherited[0] == original[0]
        assert inherited[1] != original[1]
        # Writing token eight must leave the first seven inherited token bytes intact.
        (error,) = cudart.cudaMemsetAsync(
            int(desc.base) + inherited[1] * desc.stride + 3072, 88, 1024, next_stream.cuda_stream
        )
        assert int(error) == 0
        for ordinal, index in enumerate(inherited):
            (error,) = cudart.cudaMemcpyAsync(
                output.data_ptr() + ordinal * desc.size,
                int(desc.base) + index * desc.stride,
                desc.size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                next_stream.cuda_stream,
            )
            assert int(error) == 0
        next_stream.synchronize()
        assert not pending.query() and not timed_out.is_set()
        # Release the gate before inspecting tensors on PyTorch's default stream.
        release.set()
        old_stream.synchronize()
        assert not timed_out.is_set()
        assert torch.all(output[0] == 37).item()
        assert torch.all(output[1, :3072] == 37).item()
        assert torch.all(output[1, 3072:] == 88).item()
        if not snapshot:
            assert torch.all(old_output == 37).item()
    finally:
        release.set()
        old_stream.synchronize()
        next_stream.synchronize()
        if hold is not None:
            hold.close()
        if successor is not None:
            successor.close()
        if owner is not None:
            owner.close()
        core.shutdown()
        faulthandler.cancel_dump_traceback_later()


def test_partial_isolation_rejects_snapshot_until_copy_completion(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_RETIRE_ISOLATED_PARTIAL", "1")
    torch.empty(1, device="cuda")
    stream = torch.cuda.Stream()
    entered, release, timed_out = threading.Event(), threading.Event(), threading.Event()
    pending = torch.cuda.Event()
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(quota=4 << 20), HostCacheTierConfig(quota=4 << 20)],
            layers=[
                AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])
            ],
            enable_partial_reuse=True,
            commit_min_snapshot=True,
        )
    )
    owner = hold = None
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        owner = core.create_kv_cache(None, None, expected_prompt_length=8)
        assert owner.resume(CudaStream(stream.cuda_stream)) and owner.resize(8)
        torch.cuda.synchronize()

        def gate():
            entered.set()
            if not release.wait(30):
                timed_out.set()

        runtime_bindings.launch_hostfunc(stream.cuda_stream, True, False, gate)
        pending.record(stream)
        assert entered.wait(30) and not pending.query()
        owner.commit(list(range(7)))
        manager = SimpleNamespace(
            impl=core, enable_block_reuse=True, _stream=stream, _derive_reuse_salt=lambda _: None
        )
        adapter = KVAlignmentAdapter(
            SimpleNamespace(kv_cache_manager=manager, max_seq_len=32, _is_kv_manager_v2=True)
        )
        revised = AlignmentInput(tuple(range(8)), min_reuse_tokens=7, max_reuse_tokens=7)
        assert adapter.reserve_isolated(revised) is None
        assert adapter._reserved_v2_blocks == 0
        assert not pending.query() and not timed_out.is_set()
        release.set()
        stream.synchronize()
        hold = adapter.reserve_isolated(revised)
        assert hold is not None and hold.num_tokens == 7
    finally:
        release.set()
        stream.synchronize()
        if hold is not None:
            hold.close()
        if owner is not None:
            owner.close()
        core.shutdown()
        faulthandler.cancel_dump_traceback_later()
