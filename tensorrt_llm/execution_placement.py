# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route successor ownership while retaining global revocation and drain membership."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import Event
from uuid import uuid4

from .execution_alignment import AlignmentReceipt, AlignmentRequest, ExecutionPlacement
from .execution_collective import CollectivePublicationTransport, ExecutionParticipant
from .execution_fence import FenceRequest
from .execution_mailbox import PublicationReceipt, PublicationTransport, VersionUpdate

__all__ = ["PlacementPublicationTransport", "placement_update"]


def placement_update(update: VersionUpdate, placement: ExecutionPlacement) -> VersionUpdate:
    """A non-destination observes the new epoch but cannot reserve or execute its work."""
    request = update.alignment
    destination = None if request is None else request.revised_input.destination
    if destination is None or destination == placement:
        return update
    return replace(
        update, alignment=None, resume=None, resume_prefix_length=None, admit_requests=False
    )


@dataclass(frozen=True)
class _PlacementAlignment:
    receipt: AlignmentReceipt
    local: AlignmentReceipt


class PlacementPublicationTransport(CollectivePublicationTransport):
    """Select a whole frontend group, never an individual rank within that group.

    Every participant still receives authority and contributes to FENCE. ALIGN
    reserves only at its explicit destination, whose child collective enforces
    the common prefix across its TP/PP ranks. Untargeted input retains the normal
    all-placement common-prefix behavior. The publisher thread owns this object.
    """

    def __init__(
        self,
        participants: tuple[ExecutionParticipant, ...],
        transports: Mapping[ExecutionParticipant, PublicationTransport],
        placements: Mapping[ExecutionParticipant, ExecutionPlacement],
    ) -> None:
        if set(placements) != set(participants):
            raise ValueError("placements must cover exactly the required frontends")
        if any(not isinstance(p, ExecutionPlacement) for p in placements.values()):
            raise TypeError("invalid frontend placement")
        if len(set(placements.values())) != len(participants):
            raise ValueError("frontend placements must be distinct")
        if len({p.session_id for p in placements.values()}) != 1:
            raise ValueError("frontend placements must share one authority session")
        super().__init__(participants, transports)
        self._placement_by_transport = {id(transports[p]): placements[p] for p in participants}
        self._transport_by_placement = {placements[p]: transports[p] for p in participants}
        self._routed_alignments: dict[tuple[str, str], _PlacementAlignment] = {}

    def _destination(self, request: AlignmentRequest) -> PublicationTransport | None:
        placement = request.revised_input.destination
        if placement is None:
            return None
        try:
            return self._transport_by_placement[placement]
        except KeyError as error:
            raise ValueError("ALIGN destination is not in this frontend cohort") from error

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        self._require_open(stop)
        key = (update.version.tenant_id, update.version.scope_id)
        aligned = self._routed_alignments.get(key)
        if (
            aligned is not None
            and aligned.receipt.request.revision.successor_version.epoch < update.version.epoch
        ):
            del self._routed_alignments[key]
            aligned = None
        destination = None if update.alignment is None else self._destination(update.alignment)
        if destination is None:
            return super().publish(update, stop)
        local = update
        if update.resume is not None:
            if aligned is None or aligned.receipt != update.resume:
                raise ValueError("RESUME requires this placement cohort's live ALIGN receipt")
            if self.poll_alignment(update.resume.request, stop) != update.resume:
                raise RuntimeError("RESUME requires global FENCE and the destination reservation")
            local = replace(update, resume=aligned.local)

        def publish_one(transport: PublicationTransport) -> PublicationReceipt:
            routed = placement_update(local, self._placement_by_transport[id(transport)])
            receipt = transport.publish(routed, stop)
            if not isinstance(receipt, PublicationReceipt) or receipt.update != routed:
                raise ValueError("worker acknowledgement does not match placement publication")
            if update.resume is not None and receipt.observed_version != update.version:
                raise RuntimeError("worker superseded placement RESUME")
            return receipt

        receipts = self._call_all("placement publication", publish_one, stop)
        return PublicationReceipt(
            update, min((r.observed_version for r in receipts), key=lambda v: v.epoch)
        )

    def poll_alignment(self, request: AlignmentRequest, stop: Event) -> AlignmentReceipt | None:
        self._require_open(stop)
        destination = self._destination(request)
        if destination is None:
            return super().poll_alignment(request, stop)
        fence = self.poll_fence(FenceRequest(request.session_id, request.revision), stop)
        local = destination.poll_alignment(request, stop)
        if local is not None and (
            not isinstance(local, AlignmentReceipt) or local.request != request
        ):
            raise ValueError("destination acknowledgement does not match ALIGN")
        self._require_open(stop)
        if fence is None or local is None:
            return None
        version = request.revision.successor_version
        key = (version.tenant_id, version.scope_id)
        previous = self._routed_alignments.get(key)
        if previous is not None and previous.receipt.request == request:
            if previous.local != local:
                raise RuntimeError("destination reservation changed after ALIGN")
            return previous.receipt
        receipt = AlignmentReceipt(request, uuid4().hex, local.reusable_prefix_length)
        self._routed_alignments[key] = _PlacementAlignment(receipt, local)
        return receipt

    def close(self) -> None:
        self._routed_alignments.clear()
        super().close()
