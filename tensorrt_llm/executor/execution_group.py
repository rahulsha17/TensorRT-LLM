# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind a fixed set of executor frontends to one revision authority."""

from contextlib import ExitStack
from uuid import uuid4

from ..execution_alignment import ExecutionPlacement
from ..execution_collective import CollectivePublicationTransport, ExecutionParticipant
from ..execution_mailbox import ExecutionController, LocalPublicationTransport, PublicationTransport
from ..execution_placement import PlacementPublicationTransport
from .execution_control import ExecutionControlMixin
from .execution_mailbox import IpcPublicationTransport

__all__ = ["create_shared_execution_controller"]


def _worker_keys(transport: PublicationTransport, session_id: str) -> frozenset[str]:
    """Reject repeated worker tables/endpoints, including nested TP cohorts."""
    if isinstance(transport, LocalPublicationTransport):
        identity = transport._table.identity
        if identity.session_id != session_id:
            raise ValueError("worker belongs to a different authority session")
        return frozenset((identity.worker_id,))
    if isinstance(transport, IpcPublicationTransport):
        address, key = transport._address
        if not isinstance(address, str) or not address or not isinstance(key, bytes) or not key:
            raise ValueError("shared authority requires authenticated worker endpoints")
        return frozenset((transport.identify(session_id).worker_id,))
    if isinstance(transport, CollectivePublicationTransport):
        keys: set[str] = set()
        for child in transport._transports:
            members = _worker_keys(child, session_id)
            if keys.intersection(members):
                raise ValueError("shared authority contains a repeated worker endpoint")
            keys.update(members)
        return frozenset(keys)
    raise TypeError("shared authority requires a supported worker publication transport")


def create_shared_execution_controller(
    frontends: tuple[ExecutionControlMixin, ...],
) -> ExecutionController:
    """Install one authority before any member admits work or creates a controller.

    Args:
        frontends: Distinct, running frontends owned by this runtime process.
            Each retains its own worker topology and TP consensus. Membership
            is fixed for the lifetime of the returned controller.

    All frontends share the same registry, admission gate and result revocation.
    Publication and FENCE require every member. ALIGN currently requires a local
    reservation on every member and chooses their common minimum prefix; this
    does not route or transfer KV between placements. Existing device/topology
    guards still apply, including the rejection of KV transfers and connectors.

    Bootstrap holds all frontend admission locks. A partial failure leaves every
    member unavailable until restart, rather than allowing separate authorities.
    AlignmentInput.destination can select one frontend's get_execution_placement().
    Only that group reserves and admits the successor; every member still
    participates in revocation, publication and FENCE. No KV transfer is implied.
    Closing any member's control path closes the shared controller and prevents
    further submissions on every member. Close each executor normally to stop
    its workers; closing authority alone is not proof of physical reclamation.
    """
    if not isinstance(frontends, tuple) or len(frontends) < 2:
        raise ValueError("shared authority requires a tuple of at least two frontends")
    if any(not isinstance(frontend, ExecutionControlMixin) for frontend in frontends):
        raise TypeError("shared authority requires executor frontends")
    if len({id(frontend) for frontend in frontends}) != len(frontends):
        raise ValueError("shared authority requires distinct frontends")

    with ExitStack() as locks:
        # Consistent ordering also covers concurrent, overlapping bootstrap calls.
        for frontend in sorted(frontends, key=id):
            locks.enter_context(frontend._execution_control_lock)
        for frontend in frontends:
            frontend._require_execution_control_open()
            if (
                frontend._execution_submit_started
                or frontend._execution_controller is not None
                or frontend._execution_version_table is not None
                or frontend._execution_mailbox_server is not None
                or frontend._execution_frontend_gate is not None
            ):
                raise RuntimeError("shared authority must start before admission or worker binding")

        session = uuid4().hex
        transports: list[PublicationTransport] = []
        try:
            workers: set[str] = set()
            for frontend in frontends:
                transport = frontend._create_execution_transport(session)
                transports.append(transport)
                members = _worker_keys(transport, session)
                if workers.intersection(members):
                    raise ValueError("shared authority contains a repeated worker endpoint")
                workers.update(members)
            participants = tuple(
                ExecutionParticipant(0, 0, group) for group in range(len(frontends))
            )
            placements = tuple(ExecutionPlacement(session, uuid4().hex) for _ in frontends)
            publication = PlacementPublicationTransport(
                participants,
                dict(zip(participants, transports)),
                dict(zip(participants, placements)),
            )
            controller = ExecutionController(session, publication, placements=frozenset(placements))
            for frontend, placement in zip(frontends, placements):
                frontend._execution_controller = controller
                frontend._execution_shared = True
                frontend._execution_placement = placement
            return controller
        except Exception as error:
            for frontend in frontends:
                frontend._execution_control_error = error
            cleanup_failed = False
            for transport in reversed(transports):
                try:
                    transport.close()
                except Exception:
                    # Still attempt every prepared channel; no admission or
                    # reclamation can rely on this failed bootstrap.
                    cleanup_failed = True
            if cleanup_failed:
                raise RuntimeError(
                    "shared authority bootstrap and cleanup failed; restart every member"
                ) from error
            raise
