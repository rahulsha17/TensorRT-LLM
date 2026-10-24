# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution authority at scheduling, GPU token commit, and KV ownership boundaries."""

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...bindings.executor import FinishReason
from ...execution_alignment import AlignmentRequest, AlignmentSafetyError, WorkerAlignmentState
from ...execution_consensus import ExecutionRow, ExecutionStepConsensus
from ...execution_fence import RequestDrainTracker
from ...execution_gate import ExecutionAuthorityError
from ...execution_mailbox import WorkerVersionTable
from ...execution_version import ExecutionVersion
from .execution_alignment import KVAlignmentAdapter
from .execution_device import DeviceExecutionAuthority, DeviceVersionCheckTicket
from .execution_early_exit import (
    ExecutionEarlyExit,
    early_exit_scope,
    execution_trace_enabled,
    trace_execution,
)
from .execution_speculation import is_private_ngram
from .kv_cache.mamba_cache_manager import BaseMambaCacheManager
from .llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .sampler import SampleStateTorch, TorchSampler
from .scheduler import ScheduledRequests

if TYPE_CHECKING:
    from ...runtime.kv_cache_manager_v2 import _KVCache
    from .py_executor import BatchState, PyExecutor

__all__ = ["ExecutionDeviceGuard"]


@dataclass
class _RequestUse:
    # Retain identity even if a native request or sequence slot is recycled.
    request: LlmRequest
    event: torch.cuda.Event | None


@dataclass
class _ContextOwner:
    request: LlmRequest
    request_id: int
    version: ExecutionVersion
    allocation: "_KVCache"
    frontier: int
    rejected: bool = False


@dataclass
class _ContextChunk:
    owner: _ContextOwner
    start: int
    end: int
    ticket: DeviceVersionCheckTicket
    row: int
    cursor_advanced: bool = False


class ExecutionDeviceGuard:
    """Eager/compiled/graph execution, with V1 or V2 and overlap.

    The model (including sampling inside its graph) produces candidates. The
    device commit runs outside capture and reads live authority on every replay.
    Submitted forwards may finish after revocation; this is not kernel preemption
    or permission to hand their KV to a successor.
    Multiple TP ranks require a coordinator installed while all engines pause.
    """

    @staticmethod
    def validate_executor(executor: "PyExecutor", *, tensor_parallel: bool = False) -> None:
        engine = executor.model_engine
        private_ngram = is_private_ngram(getattr(engine, "spec_config", None))
        if private_ngram:
            from ..speculative.ngram import NGramDrafter

            private_ngram = (
                type(executor.drafter) is NGramDrafter
                and executor.drafter.spec_resource_manager.is_public_pool is False
                and getattr(executor, "draft_model_engine", None) is None
                and getattr(executor.sampler, "_force_num_accepted_tokens", 0) == 0
                and executor.disable_overlap_scheduler
            )
        unsupported = {
            "multiple ranks without coordination": executor.dist.world_size != 1
            and not tensor_parallel,
            "beam search": executor.max_beam_width != 1,
            "non-TorchSampler": type(executor.sampler) is not TorchSampler,
            "speculative decoding other than private n-gram verification": (
                executor.drafter is not None or engine.is_spec_decode
            )
            and not private_ngram,
            "missing KV manager": executor.kv_cache_manager is None,
            "KV transfers": executor.kv_cache_transceiver is not None,
            "KV connectors": executor.kv_connector_manager is not None,
            "encoder-decoder models": executor.is_encoder_decoder,
            "guided decoding": executor.guided_decoder is not None,
            "recurrent KV state": isinstance(executor.kv_cache_manager, BaseMambaCacheManager)
            or getattr(executor.kv_cache_manager, "is_linear_attention", False),
            "variable sliding windows": getattr(executor.kv_cache_manager, "is_vswa", False),
        }
        if tensor_parallel:
            unsupported["non-lockstep tensor parallel topology"] = (
                executor.dist.world_size < 2
                or executor.dist.tp_size != executor.dist.world_size
                or executor.dist.pp_size != 1
                or executor.dist.cp_size != 1
                or executor.enable_attention_dp
            )
        rejected = [name for name, enabled in unsupported.items() if enabled]
        if rejected:
            raise NotImplementedError("RETIRE device guards do not support: " + ", ".join(rejected))

    def __init__(
        self,
        executor: "PyExecutor",
        table: WorkerVersionTable,
        rank_consensus: ExecutionStepConsensus | None = None,
    ) -> None:
        early_exit = os.environ.get("TRTLLM_RETIRE_EARLY_EXIT", "1")
        if early_exit not in ("0", "1"):
            raise ValueError("TRTLLM_RETIRE_EARLY_EXIT must be 0 or 1")
        self._early_exit_enabled = early_exit == "1"
        self.validate_executor(executor, tensor_parallel=rank_consensus is not None)
        self._model_engine = executor.model_engine
        self._speculative_tokens_per_step = (
            executor.model_engine.original_max_draft_len + 1
            if is_private_ngram(getattr(executor.model_engine, "spec_config", None))
            else 1
        )
        if rank_consensus is not None:
            rank_consensus.validate_membership(
                table.session_id, executor.dist.rank, tuple(range(executor.dist.world_size))
            )
        if not executor._is_kv_manager_v2 and not hasattr(LlmRequest, "freeze_retire_kv_frontier"):
            raise RuntimeError("RETIRE V1 prefill inheritance requires rebuilt native bindings")
        self.table = table
        self._rank_consensus = rank_consensus
        self._admission_source = executor.dist.rank == 0
        attention_binding = getattr(executor.model_engine, "execution_attention", None)
        self._native_kv_guards = attention_binding is not None
        self.device = (
            attention_binding.authority
            if attention_binding is not None
            else DeviceExecutionAuthority(executor.device_id)
        )
        self._kv_manager = executor.kv_cache_manager if executor._is_kv_manager_v2 else None
        self._v1_manager = None if executor._is_kv_manager_v2 else executor.kv_cache_manager
        self._alignment_adapter = KVAlignmentAdapter(executor, keep_resident=rank_consensus is None)
        self._uses: dict[int, _RequestUse] = {}
        self._kv_frontiers: dict[int, int] = {}
        self._async_context = bool(
            self._native_kv_guards
            and rank_consensus is None
            and not executor.disable_overlap_scheduler
            and not executor.enable_joint_kv_cache_reuse
            and executor.drafter is None
            and not executor.model_engine.is_spec_decode
            and self._kv_manager is not None
            and self._kv_manager.supports_execution_context_overlap()
        )
        self._context_owners: dict[int, _ContextOwner] = {}
        self._context_chunks: dict[int, list[_ContextChunk]] = {}
        table.bind_device_publisher(self.device)
        table.drain_tracker = RequestDrainTracker()
        table.alignment_state = WorkerAlignmentState(
            self._alignment_adapter.reserve,
            executor.executor_request_queue.wake_alignment
            if self._admission_source
            else lambda: None,
            prepare=self._prepare_prefix,
        )
        if self._kv_manager is not None and rank_consensus is None:
            self._kv_manager._execution_resident_recovery = self._has_resident_recovery

    def _has_resident_recovery(self, request: LlmRequest) -> bool:
        state = self.table.alignment_state
        version = request.py_execution_version
        return (
            version is not None
            and state is not None
            and state.has_resident_recovery(version, request.py_request_id)
        )

    def _prepare_prefix(self, alignment: AlignmentRequest) -> None:
        # Prepare only this scope's obsolete sources. Unrelated work must not
        # delay the handoff. Old owners stay alive throughout tree acquisition.
        old = alignment.revision.expected_version
        for use in list(self._uses.values()):
            version = use.request.py_execution_version
            if (
                version is not None
                and (version.tenant_id, version.scope_id) == (old.tenant_id, old.scope_id)
                and version.epoch <= old.epoch
            ):
                try:
                    self._settle_prefix(use.request)
                except Exception as error:
                    raise AlignmentSafetyError(
                        "source KV preparation failed; restart required"
                    ) from error

    def process_alignment(self) -> None:
        """Drive native prefix reservations on the engine thread outside capture."""
        if self.table.alignment_state is not None:
            self.table.alignment_state.process()

    @contextmanager
    def releasing_resources(self) -> Iterator[None]:
        """Keep the old KV owner until an observed revision has acquired its hold."""
        state = self.table.alignment_state
        if state is None:
            yield
        else:
            with state.releasing_owner():
                yield

    def close_alignment(self) -> None:
        """Invalidate and release reservations before model/cache teardown."""
        if self.table.alignment_state is not None:
            self.table.alignment_state.close()
            self.table.alignment_state.process()

    def enqueue(self, version: ExecutionVersion | None, submit: Callable[[], int]) -> int:
        """Track admission before a queued request can become visible to FENCE."""
        if version is None:
            return submit()
        with self.table.guard_versions([version]) as allowed:
            if not allowed[0]:
                raise ExecutionAuthorityError("cannot enqueue an obsolete execution")
            assert self.table.drain_tracker is not None
            return self.table.drain_tracker.enqueue(version, submit)

    def register_children(self, requests: list[LlmRequest]) -> None:
        """Include extra returned sequences before activation or parent cleanup."""
        tracker = self.table.drain_tracker
        assert tracker is not None
        for request in requests:
            if request.is_child and request.py_execution_version is not None:
                tracker.add_child(
                    request.parent_request_id, request.py_request_id, request.py_execution_version
                )

    def register_broadcast_requests(self, items: list) -> None:
        """Record every mirrored enqueue before any rank can activate or finish it."""
        if self._rank_consensus is None:
            return
        tracker = self.table.drain_tracker
        rows = []
        local_error = None
        try:
            if tracker is None:
                raise RuntimeError("distributed admission requires drain tracking")
            for item in items:
                if not item.is_normal_request or item.request is None:
                    raise ValueError("broadcast admission expects only normal requests")
                version = getattr(item.request, "py_execution_version", None)
                rows.append(ExecutionRow(item.id, version, 0, 0, 0, "queued"))
            tracker.observe_broadcast(
                tuple((row.request_id, row.version) for row in rows if row.version is not None),
                source_rank=self._admission_source,
            )
        except Exception as error:
            local_error = error
        try:
            self._rank_consensus.agree(
                "queue_admission",
                tuple(rows),
                (True,) * len(rows),
                local_error=local_error,
            )
        except Exception as error:
            if tracker is not None:
                tracker.fail(error)
            raise

    def terminal(self, request_id: int, version: ExecutionVersion | None) -> None:
        """Complete only after terminal cleanup or removal before activation."""
        if version is not None and self.table.alignment_state is not None:
            self.table.alignment_state.terminal(version, request_id)
        if self.table.drain_tracker is not None:
            self.table.drain_tracker.complete(request_id, version)

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

    def validate_successor(self, request: LlmRequest) -> None:
        """Bind revised input before scheduling can allocate KV or submit a graph."""
        state = self.table.alignment_state
        version = request.py_execution_version
        if state is not None and version is not None:
            prefix_length = state.claim(
                version,
                request.py_request_id,
                tuple(request.get_tokens(0)),
                request.cache_salt,
                request.lora_task_id,
            )
            if prefix_length is not None:
                request.execution_reuse_limit = prefix_length
            pending_prefix = state.pending_recovery_prefix(version, request.py_request_id)
            request.py_execution_recovery_pending = pending_prefix is not None
            request.py_execution_recovery_tokens = pending_prefix or 0

    def revoke_obsolete_requests(self, requests: list[LlmRequest]) -> list[LlmRequest]:
        """Stop scheduling old work, keeping slots until the normal result drain.

        Host observation precedes device publication. No GPU readback is needed
        to reject an obsolete admission; GPU authority independently gates tokens.
        """
        if self._rank_consensus is None and not any(
            request.py_execution_version is not None for request in requests
        ):
            return []
        with ExitStack() as authority:
            with self._preparing_step("schedule"):
                versions = [request.py_execution_version for request in requests]
                allowed = authority.enter_context(self.table.guard_versions(versions))
            allowed = self._coordinate("schedule", requests, allowed)
            with self._completing_step("schedule_done", requests):
                obsolete = [request for request, accepted in zip(requests, allowed) if not accepted]
                for request in obsolete:
                    self._cancel(request)
            return obsolete

    def _coordinate(
        self,
        phase: str,
        requests: list[LlmRequest],
        allowed: list[bool] | tuple[bool, ...],
        *,
        local_error: Exception | None = None,
    ) -> tuple[bool, ...]:
        """Agree before changing logical progress; physical allocation stays rank-local."""
        if self._rank_consensus is None:
            if local_error is not None:
                raise local_error
            return tuple(allowed)
        rows = ()
        try:
            snapshot = []
            for request in requests:
                token_count = request.get_num_tokens(0)
                snapshot.append(
                    ExecutionRow(
                        request.py_request_id,
                        request.py_execution_version,
                        token_count,
                        request.context_current_position,
                        request.context_chunk_size if request.is_context_init_state else 0,
                        request.state.name,
                        request.get_token(0, token_count - 1) if token_count else None,
                        tuple(
                            request.get_token(0, position)
                            for position in range(
                                max(0, token_count - self._speculative_tokens_per_step), token_count
                            )
                        )
                        if getattr(self, "_speculative_tokens_per_step", 1) > 1
                        else (),
                    )
                )
            rows = tuple(snapshot)
        except Exception as error:
            local_error = error
        try:
            return self._rank_consensus.agree(phase, rows, tuple(allowed), local_error=local_error)
        except Exception as error:
            if self.table.drain_tracker is not None:
                self.table.drain_tracker.fail(error)
            raise

    @property
    def is_distributed(self) -> bool:
        """Whether model steps require agreement across ranks."""
        return self._rank_consensus is not None

    def execution_failed(self, error: Exception) -> None:
        """Invalidate distributed drain evidence before the engine's fatal cleanup."""
        if self._rank_consensus is not None:
            self._rank_consensus.fail(error)
            if self.table.drain_tracker is not None:
                self.table.drain_tracker.fail(error)
            self.table.close()

    def check_loop_branch(self, boundary: str, selected: bool, requests: list[LlmRequest]) -> None:
        """Require equal loop decisions before ranks take different call paths."""
        if self._rank_consensus is not None:
            with self._preparing_step(f"loop:{boundary}"):
                if type(selected) is not bool:
                    raise TypeError("execution branch requires a boolean decision")
            self._coordinate(f"loop:{boundary}:{int(selected)}", requests, [True] * len(requests))

    def check_dispatch(
        self, batch: ScheduledRequests, can_queue: bool, previous_batch: "BatchState | None"
    ) -> None:
        """Check scheduled and pending sampler rows before either forward branch."""
        if self._rank_consensus is None:
            return
        with self._preparing_step("dispatch"):
            requests = list(batch.all_requests())
        self.check_loop_branch("dispatch", can_queue, requests)
        with self._preparing_step("previous_batch"):
            previous_requests = (
                []
                if previous_batch is None
                else list(previous_batch.scheduled_requests.all_requests())
            )
        self.check_loop_branch("previous_batch", previous_batch is not None, previous_requests)

    @contextmanager
    def _preparing_step(self, phase: str) -> Iterator[None]:
        """Report failed local preparation at the vote peers are about to enter."""
        try:
            yield
        except Exception as error:
            self._coordinate(phase, [], [], local_error=error)
            raise

    @contextmanager
    def _completing_step(
        self, phase: str, requests: list[LlmRequest], *, allow_early_exit: bool = False
    ) -> Iterator[None]:
        """Agree on completed mutation or forward exit before the next model step."""
        failure = None
        early_exit = None
        try:
            yield
        except ExecutionEarlyExit as error:
            if allow_early_exit:
                early_exit = error
                phase += f":early_exit:{error.boundary}:{error.submitted}:{error.total}"
            else:
                failure = error
        except Exception as error:
            failure = error
        if self._rank_consensus is not None:
            self._coordinate(phase, requests, [True] * len(requests), local_error=failure)
        if failure is not None:
            raise failure
        if early_exit is not None:
            raise early_exit

    @contextmanager
    def forward(self, batch: ScheduledRequests) -> Iterator[None]:
        """Record all submitted readers, including a forward that raises midway.

        The executor orders each forward after the previous sampler stream. A
        later event for a request therefore covers its earlier device uses too.
        Calls run on the engine's execution stream, outside model graph capture.
        """
        entry_phase = f"forward_entry:early_exit={int(self._early_exit_enabled)}"
        with self._preparing_step(entry_phase):
            # Include untagged rows: one live neighbor prohibits a whole-batch exit.
            all_requests = list(batch.all_requests())
            requests = [r for r in all_requests if r.py_execution_version is not None]
            versions = tuple(r.py_execution_version for r in all_requests)
            state = self.table.alignment_state
            if state is not None:
                for request in requests:
                    attached = state.confirm_attachment(
                        request.py_execution_version,
                        request.py_request_id,
                        request.prepopulated_prompt_len,
                    )
                    if attached and self._rank_consensus is None and self._kv_manager is not None:
                        # The agreed depth constrains the initial handoff, not
                        # later local recomputation. Keeping it (including 0)
                        # discards reuse of this successor's newly committed KV
                        # after preemption. Native radix matching still checks
                        # every token and the salt/adapter namespace. Retain the
                        # common cap for distributed workers until they can
                        # coordinate a new depth on recompute admission.
                        # Keep the immutable native agreement for auditing;
                        # only the local V2 lookup adapter consumes this flag.
                        request.py_execution_reuse_attached = True
            for request in requests:
                self._kv_frontiers.setdefault(id(request), request.prepopulated_prompt_len)
            context_chunks = self._snapshot_context(batch)
        self._coordinate(entry_phase, all_requests, [True] * len(all_requests))

        def wholly_obsolete() -> bool:
            return bool(versions) and not any(self.table.observe_allowed(versions))

        def coordinated_obsolete(boundary: str, submitted: int, total: int) -> bool:
            allowed = self._coordinate(
                f"early_exit:{boundary}:{submitted}:{total}",
                all_requests,
                self.table.observe_allowed(versions),
            )
            return bool(versions) and not any(allowed)

        traced = bool(requests) and bool(batch.context_requests) and execution_trace_enabled()
        trace_requests = (
            [
                {
                    "request_id": r.py_request_id,
                    "client_id": r.py_client_id,
                    "tenant_id": r.py_execution_version.tenant_id,
                    "scope_id": r.py_execution_version.scope_id,
                    "epoch": r.py_execution_version.epoch,
                    "prefill_position": r.context_current_position,
                }
                for r in requests
            ]
            if traced
            else []
        )

        def observe(boundary: str, submitted: int, total: int, obsolete: bool) -> None:
            trace_execution(
                "boundary",
                requests=trace_requests,
                boundary=boundary,
                submitted=submitted,
                total=total,
                obsolete=obsolete,
            )

        with (
            self._completing_step("forward_done", all_requests, allow_early_exit=True),
            self.device.submission(
                capture_possible=getattr(
                    getattr(self, "_model_engine", None), "forward_may_capture", True
                )
            ),
            early_exit_scope(
                wholly_obsolete
                if self._rank_consensus is None
                and self._early_exit_enabled
                and versions
                and all(v is not None for v in versions)
                else None,
                layer_checks=sum(r.context_chunk_size for r in batch.context_requests) >= 32,
                observer=observe if traced else None,
                boundary_poll=coordinated_obsolete
                if self._rank_consensus is not None and self._early_exit_enabled
                else None,
            ),
        ):
            for request in requests:
                self._uses[id(request)] = _RequestUse(request, None)
            if traced:
                trace_execution("forward_begin", requests=trace_requests)
            completed = False
            try:
                yield
                completed = True
                if context_chunks:
                    ticket = self.device.enqueue_check(
                        tuple(owner.version for owner, _, _ in context_chunks)
                    )
                    for row, (owner, start, end) in enumerate(context_chunks):
                        self._context_chunks.setdefault(id(owner.request), []).append(
                            _ContextChunk(owner, start, end, ticket, row)
                        )
            finally:
                if requests:
                    self._record_uses(requests)
                if traced:
                    trace_execution("forward_end", requests=trace_requests, completed=completed)

    def _record_uses(self, requests: list[LlmRequest]) -> None:
        # Publish a completion proof only after event.record succeeds. On a
        # record failure, a None entry prevents the owning request from freeing.
        for request in requests:
            self._uses[id(request)] = _RequestUse(request, None)
        event = torch.cuda.Event()
        event.record()
        for request in requests:
            self._uses[id(request)] = _RequestUse(request, event)

    def _uses_async_context(self, request: LlmRequest) -> bool:
        return bool(
            getattr(self, "_async_context", False)
            and request.py_execution_version is not None
            and not request.is_dummy_request
            and getattr(request, "py_multimodal_data", None) is None
        )

    def _current_context_owner(self, request: LlmRequest) -> _ContextOwner | None:
        owner = getattr(self, "_context_owners", {}).get(id(request))
        if (
            owner is not None
            and owner.request is request
            and owner.request_id == request.py_request_id
            and owner.version == request.py_execution_version
            and self._kv_manager.kv_cache_map.get(owner.request_id) is owner.allocation
        ):
            return owner
        return None

    def _owns_context_allocation(self, request: LlmRequest) -> bool:
        return (
            id(request) not in getattr(self, "_context_owners", {})
            or self._current_context_owner(request) is not None
        )

    def _snapshot_context(self, batch: ScheduledRequests) -> list[tuple[_ContextOwner, int, int]]:
        snapshots = []
        for request in batch.context_requests:
            if not self._uses_async_context(request):
                continue
            cache = self._kv_manager.kv_cache_map.get(request.py_request_id)
            if cache is None or not cache.is_active:
                raise RuntimeError("context forward has no active KV allocation")
            owner = self._current_context_owner(request)
            if owner is None:
                owner = _ContextOwner(
                    request,
                    request.py_request_id,
                    request.py_execution_version,
                    cache,
                    request.prepopulated_prompt_len,
                )
                self._context_owners[id(request)] = owner
                # A new allocation only owns its acquired prefix. Old receipts
                # retain their own owner until completion and cannot retarget it.
                self._kv_frontiers[id(request)] = owner.frontier
            start = request.context_current_position
            end = start + request.context_chunk_size
            if owner.rejected or not (owner.frontier <= start < end <= request.prompt_len):
                raise RuntimeError("context submission has no valid contiguous frontier")
            snapshots.append((owner, start, end))
        return snapshots

    def _settle_context_receipts(
        self, request: LlmRequest, *, block: bool = False, abandon: bool = False
    ) -> None:
        chunks = getattr(self, "_context_chunks", {}).get(id(request), [])
        while chunks:
            chunk = chunks[0]
            # A completed receipt can precede the host cursor update. Keep it
            # until that update succeeds, or until drained terminal cleanup.
            if not chunk.cursor_advanced and not abandon:
                break
            if not block and not chunk.ticket.query():
                break
            allowed = chunk.ticket.wait()
            if (
                len(allowed) != len(chunk.ticket.versions)
                or chunk.ticket.versions[chunk.row] != chunk.owner.version
            ):
                raise RuntimeError("context receipt has no complete identity verdict")
            owner = self._current_context_owner(request)
            if owner is chunk.owner and chunk.cursor_advanced:
                if not allowed[chunk.row]:
                    owner.rejected = True
                    self._cancel(request)
                elif not owner.rejected:
                    if chunk.start != owner.frontier:
                        raise RuntimeError("context receipt leaves an unproven KV gap")
                    owner.frontier = chunk.end
                    self._kv_frontiers[id(request)] = max(
                        self._kv_frontiers.get(id(request), 0), chunk.end
                    )
            chunks.pop(0)
        if not chunks:
            getattr(self, "_context_chunks", {}).pop(id(request), None)

    def _publish_context_frontier(self, request: LlmRequest, *, drained: bool = False) -> None:
        owner = self._current_context_owner(request)
        if (
            owner is None
            or not owner.allocation.is_active
            or not owner.allocation.is_commit_allowed
        ):
            return
        if owner.frontier <= owner.allocation.num_committed_tokens:
            return
        # A receipt covers its own forward only. A newer overlap forward may
        # still read this allocation, so receipt completion alone cannot rebase.
        if drained or self._wait_for_use(request, required=True, block=False):
            self._kv_manager.preserve_execution_context(request, owner.frontier, owner.allocation)

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
        if self._rank_consensus is None and (
            state is None
            or state.execution_ticket is None
            and not any(r.py_execution_version is not None for r in state.requests)
        ):
            yield
            return
        accepted = []
        with ExitStack() as authority:
            with self._preparing_step("token_commit"):
                requests = [] if state is None else state.requests
                versions = [r.py_execution_version for r in requests]
                ticket = None if state is None else state.execution_ticket
                device_allowed = [True] * len(requests)
                if ticket is None:
                    if any(version is not None for version in versions):
                        raise RuntimeError("tagged sampling step has no device commit ticket")
                else:
                    if tuple(versions) != ticket.versions:
                        raise RuntimeError(
                            "execution identity changed while sampling was in flight"
                        )
                    if tuple(r.py_seq_slot for r in requests) != ticket.sequence_slots:
                        raise RuntimeError("sequence slots changed while sampling was in flight")
                if state is not None and state.sampler_event is not None:
                    state.sampler_event.synchronize()
                    # The sampler must not repeat this wait under host authority.
                    state.sampler_event = None
                if ticket is not None:
                    device_allowed = ticket.wait().allowed
                    if len(device_allowed) != len(requests):
                        raise RuntimeError("sampling step has no complete device commit verdict")
                for request in requests:
                    self._settle_context_receipts(request)
                host_allowed = authority.enter_context(self.table.guard_versions(versions))
            allowed = self._coordinate(
                "token_commit",
                requests,
                [host and device for host, device in zip(host_allowed, device_allowed)],
            )
            with self._completing_step("token_commit_done", requests):
                for request, allowed_row in zip(requests, allowed):
                    if not allowed_row:
                        self._cancel(request)
                    elif request.py_execution_version is not None and not request.is_finished:
                        owner = self._current_context_owner(request)
                        if owner is not None and owner.frontier < request.prompt_len:
                            raise RuntimeError("token acceptance requires proven completed context")
                        drafts = tuple(request.py_draft_tokens or ())
                        if len(drafts) >= getattr(self, "_speculative_tokens_per_step", 1):
                            raise RuntimeError("draft proposal exceeds the configured RETIRE step")
                        accepted.append((request, request.get_num_tokens(0), drafts))
                try:
                    yield
                finally:
                    # Speculative verification can enqueue penalty-state writes
                    # during host acceptance, after the earlier sampling event.
                    if accepted and getattr(self, "_speculative_tokens_per_step", 1) > 1:
                        self._record_uses([request for request, _, _ in accepted])
                for request, before, drafts in accepted:
                    after = request.get_num_tokens(0)
                    if after == before:
                        continue
                    count = after - before
                    if not 1 <= count <= len(drafts) + 1 or request.context_remaining_length != 0:
                        raise RuntimeError(
                            "RETIRE requires one accepted token per verified position after completed context"
                        )
                    if drafts:
                        matched = count - 1
                        if request.py_num_accepted_draft_tokens != matched or any(
                            request.get_token(0, before + offset) != drafts[offset]
                            for offset in range(matched)
                        ):
                            raise RuntimeError(
                                "accepted tokens do not match the verified draft prefix"
                            )
                    # Recompute can drain and release KV before the overlapped
                    # sampler result is consumed. Accept that token, but do not
                    # recreate a frontier for the released allocation. A later
                    # forward starts its own frontier from the acquired prefix.
                    if id(request) in self._kv_frontiers and self._owns_context_allocation(request):
                        # The new sample has no KV yet; its accepted input does.
                        self._kv_frontiers[id(request)] = after - 1
        # V2 rebasing requires a drained reader. A single rank can defer
        # publication while an overlapped forward runs: ALIGN and release
        # drain and publish its accepted frontier before acquiring or freeing KV.
        # Multi-rank publication keeps a common, blocking completion boundary.
        with self._completing_step("token_prefix_done", requests):
            for request, _, _ in accepted:
                frontier = self._kv_frontiers.get(id(request))
                if frontier is None or not self._owns_context_allocation(request):
                    continue
                self._publish_context_frontier(request)
                if self._kv_manager is not None and self._kv_manager.needs_execution_prefix_commit(
                    request, frontier
                ):
                    if self._wait_for_use(
                        request, required=True, block=self._rank_consensus is not None
                    ):
                        self._kv_manager.preserve_execution_prefix(request, frontier)

    @contextmanager
    def context_commit(
        self, batch: ScheduledRequests, *, publish: bool = False
    ) -> Iterator[ScheduledRequests]:
        """Guard context progress and V2 prefix publication without changing sampler row order."""
        if self._rank_consensus is None and not any(
            r.py_execution_version is not None for r in batch.context_requests
        ):
            yield batch
            return
        with ExitStack() as authority:
            with self._preparing_step("context_commit"):
                requests = batch.context_requests
                versions = [request.py_execution_version for request in requests]
                # Eligibility belongs to the captured allocation. The ordinary
                # cursor update can clear multimodal metadata or finish a row.
                asynchronous = {
                    id(r) for r in requests if id(r) in getattr(self, "_context_owners", {})
                }
                for request in requests:
                    if id(request) in asynchronous:
                        self._settle_context_receipts(request)
                        if publish:
                            self._publish_context_frontier(request)
                if self._native_kv_guards:
                    # Complete native KV writes before accepting their frontier;
                    # publication stays unlocked during every device wait.
                    events = {}
                    for request, version in zip(requests, versions):
                        if version is None or id(request) in asynchronous:
                            continue
                        use = self._uses.get(id(request))
                        if use is None or use.request is not request or use.event is None:
                            raise RuntimeError(
                                "context GPU completion is unknown; refusing prefix commit"
                            )
                        events[id(use.event)] = use.event
                    for event in events.values():
                        self.device.wait_event(event)
                host_allowed = authority.enter_context(self.table.guard_versions(versions))
            allowed = self._coordinate("context_commit", requests, host_allowed)
            with self._completing_step("context_commit_done", requests):
                rejected = {id(r) for r, ok in zip(requests, allowed) if not ok}
                for request in requests:
                    if id(request) in rejected:
                        self._cancel(request)
                advancing = []
                if publish:
                    # Ordinary V2 updates derive their commit end from the
                    # scheduling cursor. Async rows publish explicit receipts.
                    rejected |= asynchronous
                else:
                    for request in requests:
                        if (
                            id(request) not in asynchronous
                            or id(request) in rejected
                            or request.is_finished
                        ):
                            continue
                        owner = self._current_context_owner(request)
                        chunks = [
                            chunk
                            for chunk in self._context_chunks.get(id(request), [])
                            if chunk.owner is owner
                            and not chunk.cursor_advanced
                            and chunk.start == request.context_current_position
                        ]
                        if len(chunks) != 1:
                            raise RuntimeError("context progress has no unique submitted receipt")
                        advancing.append(chunks[0])
                accepted = copy(batch)
                accepted.context_requests_chunking = [
                    r for r in batch.context_requests_chunking if id(r) not in rejected
                ]
                accepted.context_requests_last_chunk = [
                    r for r in batch.context_requests_last_chunk if id(r) not in rejected
                ]
                yield accepted
                for chunk in advancing:
                    if chunk.owner.request.context_current_position != chunk.end:
                        raise RuntimeError("context cursor did not advance to its submitted end")
                    chunk.cursor_advanced = True
                for request in accepted.context_requests:
                    if (
                        request.py_execution_version is not None
                        and not request.is_finished
                        and id(request) not in asynchronous
                    ):
                        self._kv_frontiers[id(request)] = max(
                            self._kv_frontiers.get(id(request), 0), request.context_current_position
                        )

    def _wait_for_use(
        self, request: LlmRequest, *, required: bool = False, block: bool = True
    ) -> bool:
        use = self._uses.get(id(request))
        if use is None and not required:
            return True
        if use is None or use.request is not request or use.event is None:
            raise RuntimeError("request GPU completion is unknown; refusing KV mutation or release")
        if not block and not use.event.query():
            return False
        self.device.wait_event(use.event)
        return True

    def _settle_prefix(self, request: LlmRequest) -> None:
        self._wait_for_use(request, required=id(request) in self._kv_frontiers)
        self._settle_context_receipts(request, block=True, abandon=True)
        self._publish_context_frontier(request, drained=True)
        if not self._owns_context_allocation(request):
            # The old GPU use has drained, but its accepted frontier does not
            # authorize publishing or freezing a replacement cache.
            return
        if request.py_execution_version is None:
            return
        if self._v1_manager is not None:
            block_size = self._v1_manager.tokens_per_block
            frontier = self._kv_frontiers.get(id(request), 0) // block_size * block_size
            request.freeze_retire_kv_frontier(frontier)
            if frontier and self._v1_manager.enable_block_reuse:
                self._v1_manager.store_blocks_for_reuse(request)
        elif self._kv_manager is not None:
            # Retain accepted progress if revocation won the race against the
            # ordinary V2 commit. The latest GPU use has completed above.
            cache = self._kv_manager.kv_cache_map.get(request.py_request_id)
            if cache is not None and cache.is_active:
                self._kv_manager.preserve_execution_prefix(
                    request, self._kv_frontiers.get(id(request), 0)
                )
            # Recompute admission can already have dropped this cache and
            # rewound its context cursor. Its old accepted frontier is not
            # evidence for a new allocation. Suspended caches likewise retain
            # only their existing committed prefix; do not restore them here.
            self._kv_manager.stop_request_committing(request)

    def before_release(self, request: LlmRequest) -> None:
        """Settle this request's latest submitted use before ordinary cleanup.

        A failed wait leaves the ownership record and KV intact. V2 freezes only
        this request's commit frontier; it keeps the tree and other owners' holds.
        Terminal cleanup reports completion to FENCE after resource release succeeds.
        """
        if request.py_execution_version is not None:
            self._settle_prefix(request)
        else:
            self._wait_for_use(request)
        self._uses.pop(id(request), None)
        self._kv_frontiers.pop(id(request), None)
        getattr(self, "_context_owners", {}).pop(id(request), None)
        getattr(self, "_context_chunks", {}).pop(id(request), None)

    @staticmethod
    def _cancel(request: LlmRequest) -> None:
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            request.finish_by_reason(FinishReason.CANCELLED)
            request.decoding_iter = request.py_decoding_iter
