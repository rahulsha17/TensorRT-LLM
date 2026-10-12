# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution authority at scheduling, GPU token commit, and KV ownership boundaries."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...bindings.executor import FinishReason
from ...execution_mailbox import WorkerVersionTable
from .execution_device import DeviceExecutionAuthority
from .kv_cache.mamba_cache_manager import BaseMambaCacheManager
from .llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .sampler import SampleStateTorch, TorchSampler
from .scheduler import ScheduledRequests

if TYPE_CHECKING:
    from .py_executor import PyExecutor

__all__ = ["ExecutionDeviceGuard"]


@dataclass
class _RequestUse:
    # Retain identity even if a native request or sequence slot is recycled.
    request: LlmRequest
    event: torch.cuda.Event | None


class ExecutionDeviceGuard:
    """One-rank eager/compiled/graph execution, with V1 or V2 and overlap.

    The model (including sampling inside its graph) produces candidates. The
    device commit runs outside capture and reads live authority on every replay.
    Submitted forwards may finish after revocation; this is not kernel preemption
    or permission to hand their KV to a successor.
    """

    @staticmethod
    def validate_executor(executor: "PyExecutor") -> None:
        engine = executor.model_engine
        unsupported = {
            "multiple ranks": executor.dist.world_size != 1,
            "beam search": executor.max_beam_width != 1,
            "non-TorchSampler": type(executor.sampler) is not TorchSampler,
            "speculative decoding": executor.drafter is not None or engine.is_spec_decode,
            "missing KV manager": executor.kv_cache_manager is None,
            "KV transfers": executor.kv_cache_transceiver is not None,
            "KV connectors": executor.kv_connector_manager is not None,
            "encoder-decoder models": executor.is_encoder_decoder,
            "guided decoding": executor.guided_decoder is not None,
            "recurrent KV state": isinstance(executor.kv_cache_manager, BaseMambaCacheManager)
            or getattr(executor.kv_cache_manager, "is_linear_attention", False),
            "variable sliding windows": getattr(executor.kv_cache_manager, "is_vswa", False),
        }
        rejected = [name for name, enabled in unsupported.items() if enabled]
        if rejected:
            raise NotImplementedError("RETIRE device guards do not support: " + ", ".join(rejected))

    def __init__(self, executor: "PyExecutor", table: WorkerVersionTable) -> None:
        self.validate_executor(executor)
        self.table = table
        self.device = DeviceExecutionAuthority(executor.device_id)
        self._kv_manager = executor.kv_cache_manager if executor._is_kv_manager_v2 else None
        self._uses: dict[int, _RequestUse] = {}
        table.bind_device_publisher(self.device)

    @staticmethod
    def validate_request(request: LlmRequest) -> None:
        if (
            request.py_beam_width != 1
            or get_draft_token_length(request) != 0
            or request.py_multimodal_data is not None
            or request.py_disaggregated_params is not None
            or request.py_logits_post_processors
        ):
            raise ValueError(
                "RETIRE device guards require text-only, single-beam, non-speculative requests without logits callbacks"
            )

    def revoke_obsolete_requests(self, requests: list[LlmRequest]) -> list[LlmRequest]:
        """Stop scheduling old work, keeping slots until the normal result drain.

        Host observation precedes device publication. No GPU readback is needed
        to reject an obsolete admission; GPU authority independently gates tokens.
        """
        versions = [request.py_execution_version for request in requests]
        if not any(version is not None for version in versions):
            return []
        with self.table.guard_versions(versions) as allowed:
            obsolete = [request for request, accepted in zip(requests, allowed) if not accepted]
            for request in obsolete:
                self._cancel(request)
            return obsolete

    @contextmanager
    def forward(self, batch: ScheduledRequests) -> Iterator[None]:
        """Record all submitted readers, including a forward that raises midway.

        The executor orders each forward after the previous sampler stream. A
        later event for a request therefore covers its earlier device uses too.
        Calls run on the engine's execution stream, outside model graph capture.
        """
        requests = [r for r in batch.all_requests() if r.py_execution_version is not None]
        with self.device.submission():
            for request in requests:
                self._uses[id(request)] = _RequestUse(request, None)
            try:
                yield
            finally:
                if requests:
                    self._record_uses(requests)

    def _record_uses(self, requests: list[LlmRequest]) -> None:
        # Publish a completion proof only after event.record succeeds. On a
        # record failure, a None entry prevents the owning request from freeing.
        for request in requests:
            self._uses[id(request)] = _RequestUse(request, None)
        event = torch.cuda.Event()
        event.record()
        for request in requests:
            self._uses[id(request)] = _RequestUse(request, event)

    @contextmanager
    def sampling(self, batch: ScheduledRequests) -> Iterator[None]:
        """Cover sampling and token copies even when either submits only partially."""
        requests = [r for r in batch.all_requests() if r.py_execution_version is not None]
        for request in requests:
            self._uses[id(request)] = _RequestUse(request, None)
        try:
            yield
        finally:
            if requests:
                self._record_uses(requests)

    def commit_sample(self, batch: ScheduledRequests, state: SampleStateTorch) -> None:
        """Queue the device guard; leave host waits to the normal result-consume step."""
        versions = [request.py_execution_version for request in state.requests]
        if any(version is not None for version in versions):
            assert state.device is not None and state.host is not None
            ticket = self.device.enqueue_token_commit(
                versions,
                [request.py_seq_slot for request in state.requests],
                state.device.new_tokens,
                compact=state.host.single_step_greedy,
            )
            # The ordinary sampler's asynchronous D2H copies own their original
            # destinations. Replacing these fields does not retarget those copies.
            state.device.new_tokens = ticket.tokens_device
            state.host.new_tokens = ticket.tokens_host
            state.execution_ticket = ticket

    @contextmanager
    def token_commit(self, state: SampleStateTorch | None) -> Iterator[None]:
        """Order host token mutation with publication, at the usual overlap consume point."""
        if state is None:
            yield
            return
        ticket = state.execution_ticket
        versions = [request.py_execution_version for request in state.requests]
        if ticket is None:
            if any(version is not None for version in versions):
                raise RuntimeError("tagged sampling step has no device commit ticket")
            yield
            return
        if tuple(versions) != ticket.versions:
            raise RuntimeError("execution identity changed while sampling was in flight")
        if tuple(r.py_seq_slot for r in state.requests) != ticket.sequence_slots:
            raise RuntimeError("sequence slots changed while sampling was in flight")
        if state.sampler_event is not None:
            state.sampler_event.synchronize()
            # The sampler must not repeat this wait while holding host authority.
            state.sampler_event = None
        commit = ticket.wait()
        if len(commit.allowed) != len(state.requests):
            raise RuntimeError("sampling step has no complete device commit verdict")
        with self.table.guard_versions(list(ticket.versions)) as host_allowed:
            for request, host_ok, device_ok in zip(state.requests, host_allowed, commit.allowed):
                if not (host_ok and device_ok):
                    self._cancel(request)
            yield

    @contextmanager
    def context_commit(self, batch: ScheduledRequests) -> Iterator[ScheduledRequests]:
        """Guard context progress and V2 prefix publication without changing sampler row order."""
        requests = batch.context_requests
        versions = [request.py_execution_version for request in requests]
        if not any(version is not None for version in versions):
            yield batch
            return
        with self.table.guard_versions(versions) as allowed:
            rejected = {id(r) for r, ok in zip(requests, allowed) if not ok}
            for request in requests:
                if id(request) in rejected:
                    self._cancel(request)
            accepted = copy(batch)
            accepted.context_requests_chunking = [
                r for r in batch.context_requests_chunking if id(r) not in rejected
            ]
            accepted.context_requests_last_chunk = [
                r for r in batch.context_requests_last_chunk if id(r) not in rejected
            ]
            yield accepted

    def before_release(self, request: LlmRequest) -> None:
        """Settle this request's latest submitted use before ordinary cleanup.

        A failed wait leaves the ownership record and KV intact. V2 freezes only
        this request's commit frontier; it keeps the tree and other owners' holds.
        This local cleanup check does not implement RETIRE's successor FENCE API.
        """
        use = self._uses.get(id(request))
        if use is not None:
            if use.event is None:
                raise RuntimeError("request GPU completion is unknown; refusing resource release")
            self.device.wait_event(use.event)
        if request.py_execution_version is not None and self._kv_manager is not None:
            self._kv_manager.stop_request_committing(request)
        self._uses.pop(id(request), None)

    @staticmethod
    def _cancel(request: LlmRequest) -> None:
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            request.finish_by_reason(FinishReason.CANCELLED)
            request.decoding_iter = request.py_decoding_iter
