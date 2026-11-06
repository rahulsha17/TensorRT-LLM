# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in device-conditional model segments with fixed PyTorch graph storage."""

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import torch

_capture: ContextVar["ConditionalCapture | None"] = ContextVar("retire_graph_capture", default=None)


def conditional_graph_enabled() -> bool:
    setting = os.getenv("TRTLLM_RETIRE_CONDITIONAL_GRAPHS", "0")
    if setting not in ("0", "1"):
        raise ValueError("TRTLLM_RETIRE_CONDITIONAL_GRAPHS must be 0 or 1")
    return setting == "1"


def layer_group_size() -> int:
    value = int(os.getenv("TRTLLM_RETIRE_GRAPH_LAYER_GROUP", "4"))
    if not 1 <= value <= 128:
        raise ValueError("TRTLLM_RETIRE_GRAPH_LAYER_GROUP must be between 1 and 128")
    return value


def compact_mlp_enabled() -> bool:
    setting = os.getenv("TRTLLM_RETIRE_COMPACT_MLP", "0")
    if setting not in ("0", "1"):
        raise ValueError("TRTLLM_RETIRE_COMPACT_MLP must be 0 or 1")
    if setting == "1" and not conditional_graph_enabled():
        raise ValueError("RETIRE compact MLP requires conditional CUDA graphs")
    return setting == "1"


@lru_cache(maxsize=1)
def load_native():
    """Compile before capture; source identity prevents cross-revision cache reuse."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("native.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 3):
        raise RuntimeError("RETIRE conditional graphs require CUDA 12.3 or newer")
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        raise RuntimeError("RETIRE conditional graphs require PyTorch raw CUDA graph access")
    return load(
        name=f"retire_conditional_graph_{digest}",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def layer_boundary(index: int) -> None:
    capture = _capture.get()
    if capture is not None and index and index % capture.group_size == 0:
        capture.boundary()


class ConditionalCapture:
    """Capture serial layer groups, then compose them into one device-driven graph.

    Segment graphs retain their allocator pools and fixed-address outputs. Only
    the composed graph may replay them. Native predicates read fresh authority
    before each segment; private outputs of skipped segments are unspecified.
    The executor's existing token and prefix guards reject those obsolete rows,
    and its unconditional use events still protect all allocations until drain.
    """

    def __init__(self, metadata, *, pool=None, warmup: bool = False):
        self.metadata = metadata
        self.group_size = layer_group_size()
        self._pool = pool
        self.warmup = warmup
        self.compact_mlp = compact_mlp_enabled()
        self.segments = []
        self.predicates = []
        self.logits = []
        self._predicate = (None, -1, -1)
        self.native = None
        self._segment_context = None
        self._token = None
        if not warmup:
            self._native_module = load_native()

    def __enter__(self):
        if _capture.get() is not None:
            raise RuntimeError("RETIRE conditional capture cannot nest")
        self._token = _capture.set(self)
        try:
            if not self.warmup:
                self._begin()
        except Exception:
            _capture.reset(self._token)
            raise
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if self._segment_context is not None:
                self._segment_context.__exit__(error_type, error, traceback)
                self._segment_context = None
            if error_type is None and not self.warmup:
                metadata = self.metadata
                self.native = self._native_module.ConditionalGraph(
                    [segment.raw_cuda_graph() for segment in self.segments],
                    metadata.retire_current_versions,
                    metadata.retire_sequence_slots,
                    metadata.retire_expected_versions,
                    metadata.retire_real_rows,
                    [
                        p[0] if p[0] is not None else metadata.retire_sequence_slots
                        for p in self.predicates
                    ],
                    [p[1] for p in self.predicates],
                    [p[2] for p in self.predicates],
                    self.logits,
                )
        finally:
            _capture.reset(self._token)
        return False

    def bind_logits(self, logits):
        """Make skipped rows safe for sampling before token authority is checked."""
        if (
            self.metadata.num_contexts
            or logits.ndim != 2
            or logits.shape[0] != len(self.metadata.request_ids)
            or logits.dtype != torch.float32
            or not logits.is_contiguous()
        ):
            raise ValueError(
                "RETIRE conditional graphs require contiguous FP32 decode logits in request-row order"
            )
        self.logits = [logits]

    def _begin(self):
        segment = torch.cuda.CUDAGraph(keep_graph=True)
        self._segment_context = torch.cuda.graph(segment, pool=self._pool)
        self._segment_context.__enter__()
        self.segments.append(segment)
        self.predicates.append(self._predicate)

    def boundary(self, count=None, lower: int = -1, upper: int = -1):
        if self.warmup:
            return
        self._segment_context.__exit__(None, None, None)
        self._segment_context = None
        if self._pool is None:
            self._pool = self.segments[0].pool()
        self._predicate = (count, lower, upper)
        self._begin()

    def replay(self):
        if self.native is None:
            raise RuntimeError("RETIRE conditional graph is not captured")
        self.native.replay()

    def pool(self):
        return self.segments[0].pool()

    def reset(self):
        if self.native is not None:
            self.native.close()
            self.native = None
        for segment in self.segments:
            segment.reset()
        self.segments.clear()
        self.predicates.clear()
        self.logits.clear()

    def __del__(self):
        # Native close waits for the last replay before segment allocator pools
        # can disappear. Keep teardown ordering explicit across Python GC.
        native = getattr(self, "native", None)
        if native is not None:
            native.close()


@contextmanager
def conditional_warmup(metadata):
    with ConditionalCapture(metadata, warmup=True):
        yield


def run_compact_mlp(function, hidden_states, metadata):
    capture = _capture.get()
    if capture is None or not capture.compact_mlp or metadata.num_contexts:
        return function(hidden_states)
    from .compaction import compact_mlp

    return compact_mlp(capture, function, hidden_states, metadata)
