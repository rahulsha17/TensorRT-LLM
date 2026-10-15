# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placement receipts, non-destination drain, and publication failure coverage."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_alignment import (
    AlignmentInput,
    AlignmentRequest,
    ExecutionPlacement,
    WorkerAlignmentState,
)
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_collective import ExecutionParticipant
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    VersionUpdate,
    WorkerVersionTable,
)
from tensorrt_llm.execution_placement import PlacementPublicationTransport, placement_update
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 0)
NEW = replace(OLD, epoch=1)
REVISION = Revision("edit", OLD, NEW)
PLACEMENTS = (ExecutionPlacement("session", "source"), ExecutionPlacement("session", "destination"))
INPUT = AlignmentInput((1, 2, 3, 4, 5), min_reuse_tokens=3, destination=PLACEMENTS[1])


@pytest.fixture
def cluster():
    participants = tuple(ExecutionParticipant(0, 0, i) for i in range(2))
    tables = [WorkerVersionTable("session") for _ in participants]
    holds = [Mock(num_tokens=0), Mock(num_tokens=3)]
    for table, hold in zip(tables, holds):
        table.drain_tracker = RequestDrainTracker()
        table.alignment_state = WorkerAlignmentState(Mock(return_value=hold), lambda: None)
    children = [Mock(wraps=LocalPublicationTransport(t)) for t in tables]
    transport = PlacementPublicationTransport(
        participants, dict(zip(participants, children)), dict(zip(participants, PLACEMENTS))
    )
    controller = ExecutionController("session", transport, placements=frozenset(PLACEMENTS))
    yield SimpleNamespace(
        tables=tables, holds=holds, children=children, transport=transport, controller=controller
    )
    controller.close()
    for table in tables:
        table.close()
        table.alignment_state.process()


def align(cluster):
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    return result.alignment.wait()


def test_source_does_not_limit_destination_and_never_acquires_a_hold(cluster):
    receipt = align(cluster)
    assert receipt.reusable_prefix_length == 3
    cluster.children[0].poll_alignment.assert_not_called()
    assert not cluster.tables[0].alignment_state._records
    cluster.controller.resume(receipt).wait()
    with pytest.raises(ExecutionAuthorityError):
        with cluster.tables[0].admission(NEW):
            pytest.fail("source admitted destination work")
    with cluster.tables[1].admission(NEW):
        pass
    cluster.holds[0].close.assert_not_called()


@pytest.mark.parametrize("failed_member", [0, 1])
def test_partial_resume_failure_retains_hold_and_cannot_admit_at_owner(cluster, failed_member):
    receipt = align(cluster)
    cluster.children[failed_member].publish.side_effect = TimeoutError("lost ACK")
    with pytest.raises(RuntimeError, match="placement publication"):
        cluster.controller.resume(receipt).wait()
    for placement in PLACEMENTS:
        with pytest.raises(ExecutionAuthorityError):
            with cluster.controller.admission(NEW, placement=placement):
                pytest.fail("partial publication opened admission")
    cluster.holds[1].close.assert_not_called()
    cluster.children[1 - failed_member].publish.assert_called()


@pytest.mark.parametrize("failed_member", [0, 1])
def test_resume_requires_every_members_live_fence(cluster, failed_member):
    receipt = align(cluster)
    cluster.children[failed_member].poll_fence.return_value = None
    with pytest.raises(RuntimeError, match="global FENCE"):
        cluster.controller.resume(receipt).wait()
    cluster.holds[1].close.assert_not_called()


def test_changed_destination_reservation_cannot_be_reused(cluster):
    receipt = align(cluster)
    local = cluster.children[1].poll_alignment(receipt.request, Event())
    cluster.children[1].poll_alignment.return_value = replace(local, reservation_id="replacement")
    with pytest.raises(RuntimeError, match="reservation changed"):
        cluster.controller.resume(receipt).wait()


def test_foreign_receipt_is_rejected_even_if_prefix_length_matches(cluster):
    receipt = align(cluster)
    bad = replace(receipt, reservation_id="forged")
    with pytest.raises(ValueError, match="completed ALIGN"):
        cluster.controller.resume(bad)


def test_destination_failure_is_not_hidden_by_pending_source_drain(cluster):
    cluster.controller.register(OLD).wait()
    cluster.tables[0].drain_tracker.enqueue(OLD, lambda: 7)
    result = cluster.controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    cluster.children[1].poll_alignment.side_effect = RuntimeError("unsafe destination")
    with pytest.raises(RuntimeError, match="unsafe destination"):
        result.alignment.wait()


def test_old_completion_can_drain_after_source_stops_admitting(cluster):
    cluster.controller.register(OLD).wait()
    tracker = cluster.tables[0].drain_tracker
    tracker.enqueue(OLD, lambda: 7)
    result = cluster.controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    assert cluster.tables[0].current_version("tenant", "scope") == NEW
    assert cluster.tables[0].observe_allowed((OLD, NEW)) == (False, False)
    tracker.complete(7, OLD)
    result.fence.wait()


def test_deferred_revision_blocks_worker_without_requesting_a_reservation(cluster):
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION)
    result.publication.wait()
    for table in cluster.tables:
        assert table.current_version("tenant", "scope") == NEW
        with pytest.raises(ExecutionAuthorityError):
            with table.admission(NEW):
                pytest.fail("pending revision admitted at worker")
        assert not table.alignment_state._records


def test_disabling_same_epoch_wakes_existing_consumers_and_blocks_commit():
    table = WorkerVersionTable("session")
    try:
        table.apply(VersionUpdate("session", 1, OLD))
        with table.admission(OLD) as ticket:
            ticket._wake = wake = Mock()
        table.apply(VersionUpdate("session", 2, OLD, admit_requests=False))
        wake.assert_called_once()
        with pytest.raises(ExecutionAuthorityError):
            ticket.commit(lambda: pytest.fail("disabled worker published output"))
        with table.guard_versions([OLD]) as allowed:
            assert allowed == [False]
        assert table.observe_allowed((OLD,)) == (False,)
    finally:
        table.close()


@pytest.mark.parametrize("session,placement", [("", "id"), ("session", ""), (1, "id"), ("s", None)])
def test_placement_requires_full_identity(session, placement):
    with pytest.raises(ValueError):
        ExecutionPlacement(session, placement)


def test_destination_session_is_part_of_alignment_identity():
    with pytest.raises(ValueError, match="another authority"):
        AlignmentRequest("foreign", REVISION, INPUT)
    with pytest.raises(TypeError, match="destination"):
        replace(INPUT, destination="destination")


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_admission_permission_is_not_truthiness(value):
    with pytest.raises(TypeError):
        VersionUpdate("session", 1, OLD, admit_requests=value)


def test_routing_cannot_turn_a_retired_member_into_a_resume_recipient(cluster):
    receipt = align(cluster)
    update = VersionUpdate("session", 3, NEW, receipt.request, receipt)
    routed = placement_update(update, PLACEMENTS[0])
    assert not routed.admit_requests
    assert routed.alignment is routed.resume is routed.resume_prefix_length is None
    assert routed.version == update.version and routed.sequence == update.sequence
    with pytest.raises(ValueError):
        replace(update, admit_requests=False)
