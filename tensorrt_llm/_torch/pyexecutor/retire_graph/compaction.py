# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-only MLP compaction; attention and KV sequence layouts stay fixed."""

import torch
import triton
import triton.language as tl


@triton.jit
def _snapshot_live_rows(
    table,
    slots,
    expected,
    real,
    indices,
    count,
    ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    slot = tl.load(slots + row, row < ROWS, other=-1)
    epoch = tl.load(expected + row, row < ROWS, other=0)
    is_real = tl.load(real + row, row < ROWS, other=False)
    current = tl.atomic_add(
        table + tl.maximum(slot, 0),
        0,
        mask=(row < ROWS) & (slot >= 0) & (slot < CAPACITY),
        sem="acquire",
        scope="gpu",
    )
    live = is_real & ((slot < 0) | (slot >= CAPACITY) | (current == epoch))
    offset = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(indices + offset, row, live)
    tl.store(count, tl.sum(live.to(tl.int32), 0))


@triton.jit
def _gather_rows(
    source, indices, count, packed, WIDTH: tl.constexpr, BUCKET: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed_row = offset // WIDTH
    column = offset % WIDTH
    live = (packed_row < tl.load(count)) & (packed_row < BUCKET)
    source_row = tl.load(indices + packed_row, live, other=0)
    value = tl.load(source + source_row * WIDTH + column, live, other=0)
    tl.store(packed + offset, value, offset < BUCKET * WIDTH)


@triton.jit
def _scatter_rows(
    packed,
    indices,
    count,
    destination,
    WIDTH: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed_row = offset // WIDTH
    column = offset % WIDTH
    live = (packed_row < tl.load(count)) & (packed_row < BUCKET)
    destination_row = tl.load(indices + packed_row, live, other=0)
    value = tl.load(packed + offset, live, other=0)
    tl.store(destination + destination_row * WIDTH + column, value, live)


def compact_mlp(capture, function, hidden_states, metadata):
    """Select exactly one bucket from one authority snapshot per MLP.

    Publication after the snapshot can make extra rows obsolete, but cannot
    change the bucket midway through its alternatives. Downstream authority
    guards still reject those rows. No request is newly admitted inside a
    forward, and all live rows retain their original positions after scatter.
    """
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("RETIRE compact MLP requires contiguous [rows, hidden] activations")
    rows, width = hidden_states.shape
    if rows < 2:
        return function(hidden_states)
    if metadata.num_contexts or rows > metadata.retire_sequence_slots.numel():
        raise ValueError("RETIRE compact MLP requires one decode token per authority row")
    indices = torch.empty(rows, dtype=torch.int32, device=hidden_states.device)
    count = torch.empty(1, dtype=torch.int32, device=hidden_states.device)
    result = torch.zeros_like(hidden_states)
    _snapshot_live_rows[(1,)](
        metadata.retire_current_versions,
        metadata.retire_sequence_slots,
        metadata.retire_expected_versions,
        metadata.retire_real_rows,
        indices,
        count,
        rows,
        metadata.retire_current_versions.numel(),
        triton.next_power_of_2(rows),
    )
    lower, bucket = 0, 1
    while bucket * 2 <= rows:
        capture.boundary(count, lower, bucket)
        packed = torch.empty(
            (bucket, width), dtype=hidden_states.dtype, device=hidden_states.device
        )
        _gather_rows[(triton.cdiv(bucket * width, 256),)](
            hidden_states,
            indices,
            count,
            packed,
            width,
            bucket,
            256,
        )
        output = function(packed)
        _scatter_rows[(triton.cdiv(bucket * width, 256),)](
            output,
            indices,
            count,
            result,
            width,
            bucket,
            256,
        )
        lower = bucket
        bucket *= 2
    capture.boundary(count, lower, rows)
    # Near-full batches retain the original GEMM shape and avoid gathering.
    result.copy_(function(hidden_states))
    capture.boundary()
    return result
