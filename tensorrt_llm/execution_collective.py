# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate RETIRE acknowledgements over an executor's fixed worker set."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from threading import Event
from typing import TypeVar
from uuid import uuid4

from .execution_alignment import AlignmentReceipt, AlignmentRequest
from .execution_fence import FenceReceipt, FenceRequest
from .execution_mailbox import PublicationReceipt, PublicationTransport, VersionUpdate

__all__ = ["ExecutionParticipant", "CollectivePublicationTransport"]

_T = TypeVar("_T")


@dataclass(frozen=True, order=True)
class ExecutionParticipant:
    """One required TP/PP worker in a runtime-selected KV group."""

    tp_rank: int
    pp_rank: int
    kv_group: int = 0

    def __post_init__(self) -> None:
        for value in (self.tp_rank, self.pp_rank, self.kv_group):
            if type(value) is not int or value < 0:
                raise ValueError("participant coordinates must be nonnegative integers")


@dataclass(frozen=True)
class _CollectiveAlignment:
    receipt: AlignmentReceipt
    workers: tuple[AlignmentReceipt, ...]


class CollectivePublicationTransport:
    """Require every worker's proof before acknowledging the owner.

    Args:
        participants: Complete membership from the executor topology, determined
            independently of which workers answer. Membership cannot shrink.
        transports: One authenticated endpoint per participant, in the same
            executor lifetime. Bootstrap must bind endpoints to distinct workers.
        publication_source: Worker that broadcasts scheduler wakes. Publish to
            it last so every peer has recorded ALIGN/reclamation before the wake.

    The controller's publisher thread owns this transport and all children.
    Child operations must have bounded waits and honor stop. Failure on one
    worker does not prevent attempting publication to the remaining workers.
    A failed attempt never establishes completion or rolls back authority.

    ALIGN chooses the shortest reserved prefix. Each local hold stays with its
    worker; RESUME translates the aggregate receipt back to those exact local
    receipts and supplies the common length as a cache-lookup limit. Every worker
    must acknowledge that limit before the frontend can admit the successor.
    Failure retains holds until a newer revision or worker teardown reclaims them.
    Closing channels alone does not prove FENCE or release native resources.

    This control transport does not coordinate model collectives or early exit.
    The executor must provide that coordination before enabling multi-rank work.
    Only the frontend may admit work, after all RESUME acknowledgements arrive.
    """

    def __init__(
        self,
        participants: tuple[ExecutionParticipant, ...],
        transports: Mapping[ExecutionParticipant, PublicationTransport],
        *,
        publication_source: ExecutionParticipant | None = None,
    ) -> None:
        if not isinstance(participants, tuple) or not participants:
            raise ValueError("participants must be a nonempty tuple")
        if any(not isinstance(p, ExecutionParticipant) for p in participants):
            raise TypeError("invalid execution participant")
        if len(set(participants)) != len(participants):
            raise ValueError("duplicate execution participant")
        if set(transports) != set(participants):
            raise ValueError("transports must cover exactly the required participants")
        if len({id(t) for t in transports.values()}) != len(participants):
            raise ValueError("each participant requires a distinct transport")
        if publication_source is not None and publication_source not in participants:
            raise ValueError("publication source must be a required participant")
        self._publication_source = publication_source
        self._participants = tuple(sorted(participants))
        self._transports = tuple(transports[p] for p in self._participants)
        self._alignments: dict[tuple[str, str], _CollectiveAlignment] = {}
        self._closed = False

    @property
    def participants(self) -> tuple[ExecutionParticipant, ...]:
        """Fixed required membership, including workers that have not answered."""
        return self._participants

    def _require_open(self, stop: Event) -> None:
        if self._closed or stop.is_set():
            raise RuntimeError("execution collective is closed")

    def _call_all(
        self,
        operation: str,
        call: Callable[[PublicationTransport], _T],
        stop: Event,
        *,
        last: ExecutionParticipant | None = None,
    ) -> list[_T]:
        results = {}
        failures = []
        endpoints = list(zip(self._participants, self._transports))
        endpoints.sort(key=lambda item: item[0] == last)
        for participant, transport in endpoints:
            if stop.is_set():
                raise RuntimeError("execution collective stopped before all acknowledgements")
            try:
                results[participant] = call(transport)
            except Exception as error:
                # Endpoint failure must not prevent other workers from seeing a
                # revocation or hide a later worker's failed drain behind None.
                failures.append((participant, error))
        if failures:
            failed = tuple(p for p, _ in failures)
            raise RuntimeError(f"{operation} failed for participants {failed}") from failures[0][1]
        if stop.is_set():
            raise RuntimeError("execution collective stopped before acknowledgement")
        return [results[p] for p in self._participants]

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        """Publish to every worker; RESUME cannot complete on a partial acknowledgement."""
        self._require_open(stop)
        key = (update.version.tenant_id, update.version.scope_id)
        aligned = self._alignments.get(key)
        if aligned is not None and (
            aligned.receipt.request.revision.successor_version.epoch < update.version.epoch
        ):
            del self._alignments[key]
            aligned = None
        if update.resume is not None:
            if aligned is None or aligned.receipt != update.resume:
                raise ValueError("RESUME requires this collective's live ALIGN receipt")
            if self.poll_alignment(update.resume.request, stop) != update.resume:
                raise RuntimeError("RESUME requires every worker's live reservation and FENCE")
            prefix_length = update.resume_prefix_length
            if prefix_length is None and len(self._participants) > 1:
                prefix_length = aligned.receipt.reusable_prefix_length
            local_updates = {
                id(t): replace(
                    update,
                    resume=r,
                    resume_prefix_length=prefix_length,
                )
                for t, r in zip(self._transports, aligned.workers)
            }
        else:
            local_updates = {id(t): update for t in self._transports}

        def publish_one(transport: PublicationTransport) -> PublicationReceipt:
            local = local_updates[id(transport)]
            receipt = transport.publish(local, stop)
            if not isinstance(receipt, PublicationReceipt) or receipt.update != local:
                raise ValueError("worker acknowledgement does not match publication")
            if update.resume is not None and receipt.observed_version != update.version:
                raise RuntimeError("worker superseded RESUME")
            return receipt

        # The source broadcasts the scheduler wake. Peers must arm ALIGN or
        # reclamation before that wake can reach their engine threads.
        receipts = self._call_all("publication", publish_one, stop, last=self._publication_source)
        observed = min((r.observed_version for r in receipts), key=lambda v: v.epoch)
        return PublicationReceipt(update, observed)

    def poll_fence(self, request: FenceRequest, stop: Event) -> FenceReceipt | None:
        """Require drain through the replaced epoch on every participant."""
        self._require_open(stop)

        def poll_one(transport: PublicationTransport) -> FenceReceipt | None:
            receipt = transport.poll_fence(request, stop)
            if receipt is not None and (
                not isinstance(receipt, FenceReceipt) or receipt.request != request
            ):
                raise ValueError("worker acknowledgement does not match FENCE")
            return receipt

        receipts = self._call_all("FENCE", poll_one, stop)
        return None if any(r is None for r in receipts) else FenceReceipt(request)

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None:
        """Keep local receipts distinct and require a consistent execution prefix."""
        self._require_open(stop)

        def poll_one(transport: PublicationTransport) -> AlignmentReceipt | None:
            receipt = transport.poll_alignment(request, stop)
            if receipt is not None and (
                not isinstance(receipt, AlignmentReceipt) or receipt.request != request
            ):
                raise ValueError("worker acknowledgement does not match ALIGN")
            return receipt

        receipts = self._call_all("ALIGN", poll_one, stop)
        if any(r is None for r in receipts):
            return None
        workers = tuple(r for r in receipts if r is not None)
        version = request.revision.successor_version
        key = (version.tenant_id, version.scope_id)
        aligned = self._alignments.get(key)
        if aligned is not None and aligned.receipt.request == request:
            if aligned.workers != workers:
                raise RuntimeError("worker reservation changed after collective ALIGN")
            return aligned.receipt
        receipt = AlignmentReceipt(
            request, uuid4().hex, min(r.reusable_prefix_length for r in workers)
        )
        self._alignments[key] = _CollectiveAlignment(receipt, workers)
        return receipt

    def close(self) -> None:
        """Close all child channels even when one close fails; this is not RECLAIM."""
        if not self._closed:
            self._closed = True
            self._alignments.clear()
            self._call_all("close", lambda transport: transport.close(), Event())
