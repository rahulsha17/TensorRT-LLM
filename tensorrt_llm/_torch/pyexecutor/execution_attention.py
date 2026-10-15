# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind native attention predicates before warmup and refresh them before replay."""

import os
import weakref
from typing import TYPE_CHECKING

import torch

from ...execution_version import ExecutionVersion
from .execution_device import DeviceAttentionRows, DeviceExecutionAuthority
from .execution_speculation import is_private_ngram

if TYPE_CHECKING:
    from ..attention.backends.trtllm import TrtllmAttentionMetadata
    from .engine.runners.decoder.runner import DecoderRunner
    from .llm_request import LlmRequest
    from .scheduler import ScheduledRequests


def attention_versions(
    request_ids: list[int], requests: list["LlmRequest"]
) -> list[ExecutionVersion | None]:
    """Use the attention packer's order, including repeated padding dummies."""
    by_id = {}
    for request in requests:
        if request.is_cuda_graph_dummy and request.py_execution_version is not None:
            raise ValueError("graph padding must not carry execution authority")
        previous = by_id.get(request.py_request_id)
        if previous is not None and (
            previous is not request
            or not request.is_cuda_graph_dummy
            or request.py_execution_version is not None
        ):
            raise ValueError("duplicate real request in attention authority rows")
        by_id[request.py_request_id] = request
    if any(request_id not in by_id for request_id in request_ids):
        raise ValueError("attention row has no execution request")
    seen = set()
    for request_id in request_ids:
        if request_id in seen and not by_id[request_id].is_cuda_graph_dummy:
            raise ValueError("duplicate real request in packed attention rows")
        seen.add(request_id)
    return [by_id[request_id].py_execution_version for request_id in request_ids]


class ExecutionAttentionBinding:
    """Own the authority used by both startup graphs and the later controller.

    Each metadata object has separate stable row storage. Copies made for graph
    shapes cannot share a mutable row buffer with their source metadata.
    """

    def __init__(self, authority: DeviceExecutionAuthority) -> None:
        self.authority = authority
        self._rows: dict[int, tuple[weakref.ReferenceType, DeviceAttentionRows]] = {}

    @classmethod
    def from_runner(cls, runner: "DecoderRunner") -> "ExecutionAttentionBinding | None":
        setting = os.environ.get("TRTLLM_RETIRE_NATIVE_GUARDS", "0")
        if setting == "0":
            return None
        if setting != "1":
            raise ValueError("TRTLLM_RETIRE_NATIVE_GUARDS must be 0 or 1")
        from ..attention.backends.trtllm import TrtllmAttentionMetadata

        config = runner._config
        if (
            torch.cuda.get_device_capability() != (9, 0)
            or config.dtype != torch.bfloat16
            or config.attention_backend.Metadata is not TrtllmAttentionMetadata
            or config.max_beam_width != 1
            or config.is_spec_decode
            and not is_private_ngram(config.spec_config)
            or config.is_encode_only
            or config.is_multimodal
            or runner.mapping.world_size != runner.mapping.tp_size
            or config.enable_attention_dp
            or runner.sparse_attention_config is not None
        ):
            raise NotImplementedError(
                "RETIRE native attention binding requires SM90 BF16 TRTLLM decoder attention "
                "with TP only, without beams, attention DP, sparse attention or speculation other than private n-gram"
            )
        if os.environ.get("TLLM_FMHA_LIBS") != "fallback":
            raise ValueError("RETIRE native guards require TLLM_FMHA_LIBS=fallback before startup")
        return cls(DeviceExecutionAuthority(torch.cuda.current_device()))

    def prepare(self, metadata: "TrtllmAttentionMetadata", batch: "ScheduledRequests") -> None:
        from ..attention.backends.trtllm import TrtllmAttentionMetadata

        if not isinstance(metadata, TrtllmAttentionMetadata) or metadata.is_cross:
            raise NotImplementedError("RETIRE native binding requires TRTLLM self-attention")
        if metadata.kv_cache_manager is None:
            raise NotImplementedError("RETIRE native binding requires paged KV attention")
        metadata.use_paged_context_fmha = True
        count = metadata.kv_lens_cuda_runtime.numel()
        if metadata.request_ids is None or len(metadata.request_ids) != count:
            raise ValueError("attention identities do not match physical sequence rows")
        versions = attention_versions(metadata.request_ids, list(batch.all_requests()))
        key = id(metadata)
        entry = self._rows.get(key)
        if entry is None:
            rows = self.authority.create_attention_rows(metadata.max_num_requests)
            self._rows[key] = (weakref.ref(metadata, lambda _: self._rows.pop(key, None)), rows)
        else:
            owner, rows = entry
            if owner() is not metadata:
                raise RuntimeError("attention metadata identity was reused before release")
        self.authority.prepare_attention_rows(rows, versions)
        metadata.retire_current_versions = self.authority._table
        # Prefill graphs can replay the same token bucket with a different
        # number of sequences. Capture capacity, not the warmup row count.
        metadata.retire_sequence_slots = rows.slots
        metadata.retire_expected_versions = rows.expected
        metadata.retire_counters = rows.counters
