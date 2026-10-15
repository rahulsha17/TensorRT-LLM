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


@pytest.fixture(params=["local", "ipc", "tp2"])
def copy_cluster(request):
    from tensorrt_llm.execution_alignment import PrefixTransferPlan
    from tensorrt_llm.execution_collective import CollectivePublicationTransport
    from tensorrt_llm.executor.execution_mailbox import (
        ExecutionMailboxServer,
        IpcPublicationTransport,
    )

    groups, servers = [], []
    children = []
    ranks = 2 if request.param == "tp2" else 1
    for _ in PLACEMENTS:
        tables = [WorkerVersionTable("session") for _ in range(ranks)]
        endpoints = []
        for table in tables:
            table.drain_tracker = RequestDrainTracker()
            table.alignment_state = WorkerAlignmentState(
                Mock(side_effect=lambda _: Mock(num_tokens=3)), Mock()
            )
            if request.param == "ipc":
                server = ExecutionMailboxServer(table)
                servers.append(server)
                endpoints.append(IpcPublicationTransport(server.address))
            else:
                endpoints.append(LocalPublicationTransport(table))
        participants = tuple(ExecutionParticipant(i, 0) for i in range(ranks))
        child = CollectivePublicationTransport(participants, dict(zip(participants, endpoints)))
        children.append(Mock(wraps=child))
        groups.append(tables)
    participants = tuple(ExecutionParticipant(0, 0, i) for i in range(2))
    transport = PlacementPublicationTransport(
        participants, dict(zip(participants, children)), dict(zip(participants, PLACEMENTS))
    )
    controller = ExecutionController("session", transport, placements=frozenset(PLACEMENTS))
    revised = replace(INPUT, transfer=PrefixTransferPlan(PLACEMENTS[0], 3))
    yield SimpleNamespace(
        groups=groups,
        children=children,
        controller=controller,
        transport=transport,
        revised=revised,
    )
    controller.close()
    for server in servers:
        server.close()
    for group in groups:
        for table in group:
            for permit in table._alignment_transfers:
                if table.drain_tracker.has_resource(permit.claim):
                    table.drain_tracker.release_resource(permit.claim)
            table.close()
            if table.alignment_state._unsafe_error is None:
                table.alignment_state.process()


def start_copy(cluster):
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION, alignment=cluster.revised)
    result.publication.wait()
    for group in cluster.groups:
        for table in group:
            table.alignment_state.process()
    return result


def prepare_copies(cluster, result):
    fence = result.fence.wait()
    permits = []
    for index, group in enumerate(cluster.groups):
        for table in group:
            permit = table.admit_alignment_transfer(
                result.alignment.request,
                fence,
                request_id=7,
                transfer_id=99,
                direction="send" if index == 0 else "receive",
                prefix_length=3,
            )
            permits.append((table, permit))
    return permits


def accept_copies(permits):
    for table, permit in permits:
        table.drain_tracker.release_resource(permit.claim)
        table.accept_alignment_transfer(permit)


def test_source_copy_role_keeps_hold_without_generation_and_reclaims_after_resume(copy_cluster):
    c = copy_cluster
    result = start_copy(c)
    with pytest.raises(TimeoutError):
        result.alignment.wait(0.01)  # A pre-existing prefix is not copy acceptance.
    for group in c.groups:
        for table in group:
            assert table.alignment_state.snapshot()["held_count"] == 1
            with pytest.raises(ExecutionAuthorityError):
                with table.admission(NEW):
                    pytest.fail("pending copy admitted generation")
    permits = prepare_copies(c, result)
    accept_copies(permits)
    receipt = result.alignment.wait()
    c.controller.resume(receipt).wait()
    for source in c.groups[0]:
        with pytest.raises(ExecutionAuthorityError):
            with source.admission(NEW):
                pytest.fail("source received successor authority")
        source.alignment_state.process()
        assert source.alignment_state.snapshot()["held_count"] == 0
        assert source.poll_alignment(receipt.request) is not None
        with pytest.raises(ExecutionAuthorityError, match="RESUME"):
            source.admit_alignment_transfer(
                receipt.request,
                result.fence.wait(),
                request_id=7,
                transfer_id=100,
                direction="send",
                prefix_length=3,
            )
    for destination in c.groups[1]:
        assert destination.alignment_state.snapshot()["held_count"] == 1
        with destination.admission(NEW) as execution:
            execution.finish()
        destination.alignment_state.claim(NEW, 9, INPUT.input_token_ids, None, None)
        destination.alignment_state.confirm_attachment(NEW, 9, 3)
        assert destination.alignment_state.snapshot()["held_count"] == 0


def test_source_hold_is_armed_before_old_owner_release(copy_cluster):
    c = copy_cluster
    c.controller.register(OLD).wait()
    source = c.groups[0][0]
    source.drain_tracker.enqueue(OLD, lambda: 7)
    result = c.controller.advance(REVISION, alignment=c.revised)
    result.publication.wait()
    source.alignment_state._reserve.assert_not_called()
    with source.alignment_state.releasing_owner():
        source.alignment_state._reserve.assert_called_once_with(c.revised)
        source.drain_tracker.complete(7, OLD)
    assert source.alignment_state.snapshot()["held_count"] == 1
    result.fence.wait()


@pytest.mark.parametrize("endpoint", [0, 1])
def test_copy_plan_rejects_wrong_endpoint_direction(copy_cluster, endpoint):
    result = start_copy(copy_cluster)
    table = copy_cluster.groups[endpoint][0]
    with pytest.raises(ValueError, match="endpoint"):
        table.admit_alignment_transfer(
            result.alignment.request,
            result.fence.wait(),
            request_id=7,
            transfer_id=99,
            direction="receive" if endpoint == 0 else "send",
            prefix_length=3,
        )
    assert not table.drain_tracker._resources


@pytest.mark.parametrize("length", [1, 2, 4])
def test_copy_plan_rejects_another_prefix_length(copy_cluster, length):
    result = start_copy(copy_cluster)
    source = copy_cluster.groups[0][0]
    with pytest.raises(ValueError, match="endpoint"):
        source.admit_alignment_transfer(
            result.alignment.request,
            result.fence.wait(),
            request_id=7,
            transfer_id=99,
            direction="send",
            prefix_length=length,
        )
    assert not source.drain_tracker._resources


@pytest.mark.parametrize("missing_endpoint", [0, 1])
def test_global_alignment_waits_for_every_endpoint_copy_acceptance(copy_cluster, missing_endpoint):
    result = start_copy(copy_cluster)
    permits = prepare_copies(copy_cluster, result)
    missing_table = copy_cluster.groups[missing_endpoint][-1]
    accept_copies([(t, p) for t, p in permits if t is not missing_table])
    with pytest.raises(TimeoutError):
        result.alignment.wait(0.01)
    accept_copies([(t, p) for t, p in permits if t is missing_table])
    assert result.alignment.wait().reusable_prefix_length == 3


def test_source_release_cannot_substitute_for_copy_completion(copy_cluster):
    result = start_copy(copy_cluster)
    prepare_copies(copy_cluster, result)
    source = copy_cluster.groups[0][0]
    with pytest.raises(RuntimeError, match="completed ALIGN"):
        source.apply(
            VersionUpdate(
                "session",
                3,
                NEW,
                result.alignment.request,
                admit_requests=False,
                source_only=True,
                release_source=True,
            )
        )
    source.alignment_state.process()
    assert source.alignment_state.snapshot()["held_count"] == 1


def test_supersession_retains_source_memory_until_physical_copy_drains(copy_cluster):
    c = copy_cluster
    result = start_copy(c)
    permits = prepare_copies(c, result)
    second = c.controller.advance(Revision("second", NEW, replace(NEW, epoch=2)))
    second.publication.wait()
    for group in c.groups:
        for table in group:
            table.alignment_state.process()
            assert table.alignment_state.snapshot()["held_count"] == 1
    with pytest.raises(TimeoutError):
        second.fence.wait(0.01)
    for table, permit in permits:
        table.drain_tracker.release_resource(permit.claim)
        table.alignment_state.process()
        assert table.alignment_state.snapshot()["held_count"] == 0
        with pytest.raises(ExecutionAuthorityError):
            table.accept_alignment_transfer(permit)
    second.fence.wait()


@pytest.mark.parametrize("lost_endpoint", [0, 1])
def test_lost_resume_ack_retry_preserves_source_release_and_destination_authority(
    copy_cluster, lost_endpoint
):
    c = copy_cluster
    result = start_copy(c)
    accept_copies(prepare_copies(c, result))
    receipt = result.alignment.wait()
    child = c.children[lost_endpoint]
    original = child._mock_wraps.publish

    def lost_ack(update, stop):
        original(update, stop)
        for table in c.groups[lost_endpoint]:
            table.alignment_state.process()
        raise TimeoutError("lost resume ACK")

    child.publish.side_effect = lost_ack
    with pytest.raises(RuntimeError, match="placement publication"):
        c.controller.resume(receipt).wait()
    with pytest.raises(ExecutionAuthorityError):
        with c.controller.admission(NEW, placement=PLACEMENTS[1]):
            pytest.fail("lost acknowledgement opened frontend")
    child.publish.side_effect = None
    c.controller.resume(receipt).wait()
    with c.controller.admission(NEW, placement=PLACEMENTS[1]) as execution:
        pass
    execution.finish()
    for source in c.groups[0]:
        source.alignment_state.process()
        assert source.alignment_state.snapshot()["held_count"] == 0
        assert source.observe_allowed((NEW,)) == (False,)


def test_source_routing_is_idempotent_and_observers_do_not_acquire_copy_roles():
    from tensorrt_llm.execution_alignment import AlignmentReceipt, PrefixTransferPlan

    revised = replace(INPUT, transfer=PrefixTransferPlan(PLACEMENTS[0], 3))
    request = AlignmentRequest("session", REVISION, revised)
    for resume in (None, AlignmentReceipt(request, "ready", 3)):
        update = VersionUpdate(
            "session", 3, NEW, request, resume, admit_requests=resume is not None
        )
        source = placement_update(update, PLACEMENTS[0])
        assert source.source_only and source.release_source == (resume is not None)
        assert not source.admit_requests and source.resume is None
        assert placement_update(source, PLACEMENTS[0]) == source
        observer = placement_update(update, ExecutionPlacement("session", "observer"))
        assert (
            observer.alignment is None and not observer.source_only and not observer.admit_requests
        )


@pytest.mark.parametrize(
    "change",
    [
        "absent_destination",
        "same_endpoint",
        "foreign_session",
        "short_prefix",
        "full_input",
        "bool_length",
        "foreign_type",
    ],
)
def test_invalid_prefix_transfer_plan_is_rejected(change):
    from tensorrt_llm.execution_alignment import PrefixTransferPlan

    kwargs = dict(transfer=PrefixTransferPlan(PLACEMENTS[0], 3))
    if change == "absent_destination":
        kwargs["destination"] = None
    elif change == "same_endpoint":
        kwargs["destination"] = PLACEMENTS[0]
    elif change == "foreign_session":
        kwargs["transfer"] = PrefixTransferPlan(ExecutionPlacement("foreign", "source"), 3)
    elif change == "short_prefix":
        kwargs["transfer"] = PrefixTransferPlan(PLACEMENTS[0], 2)
    elif change == "full_input":
        kwargs["transfer"] = PrefixTransferPlan(PLACEMENTS[0], 5)
    elif change == "foreign_type":
        kwargs["transfer"] = "source"
    else:
        with pytest.raises(ValueError):
            PrefixTransferPlan(PLACEMENTS[0], True)
        return
    with pytest.raises((ValueError, TypeError)):
        replace(INPUT, **kwargs)


def test_source_must_be_in_prepared_runtime_cohort(cluster):
    from tensorrt_llm.execution_alignment import PrefixTransferPlan

    cluster.controller.register(OLD).wait()
    revised = replace(
        INPUT, transfer=PrefixTransferPlan(ExecutionPlacement("session", "unknown"), 3)
    )
    with pytest.raises(ValueError, match="source is not a member"):
        cluster.controller.advance(REVISION, alignment=revised)
    assert cluster.controller.snapshot("tenant", "scope").current_version == OLD


def test_source_copy_cannot_start_before_engine_acquires_the_hold(copy_cluster):
    c = copy_cluster
    c.controller.register(OLD).wait()
    result = c.controller.advance(REVISION, alignment=c.revised)
    result.publication.wait()
    source = c.groups[0][0]
    with pytest.raises(RuntimeError, match="prepared source"):
        source.admit_alignment_transfer(
            result.alignment.request,
            result.fence.wait(),
            request_id=7,
            transfer_id=99,
            direction="send",
            prefix_length=3,
        )
    assert not source.drain_tracker._resources


def test_failed_source_reservation_does_not_wait_forever_for_copy_acceptance(copy_cluster):
    c = copy_cluster
    c.groups[0][0].alignment_state._reserve.side_effect = RuntimeError("source cache unavailable")
    result = start_copy(c)
    with pytest.raises(RuntimeError, match="source/destination ALIGN"):
        result.alignment.wait()
    for child in c.children:
        child.poll_alignment.assert_called()


def test_changed_source_receipt_prevents_global_resume(copy_cluster):
    c = copy_cluster
    result = start_copy(c)
    accept_copies(prepare_copies(c, result))
    receipt = result.alignment.wait()
    child = c.children[0]
    original = child.poll_alignment(receipt.request, Event())
    child.poll_alignment.return_value = replace(original, reservation_id="another-source-hold")
    with pytest.raises(RuntimeError, match="reservation changed"):
        c.controller.resume(receipt).wait()
    for source in c.groups[0]:
        assert source.alignment_state.snapshot()["held_count"] == 1


@pytest.mark.parametrize(
    "changes", [dict(admit_requests=True), dict(source_only=False, release_source=True)]
)
def test_source_control_publication_cannot_enable_generation(changes):
    from tensorrt_llm.execution_alignment import PrefixTransferPlan

    request = AlignmentRequest(
        "session", REVISION, replace(INPUT, transfer=PrefixTransferPlan(PLACEMENTS[0], 3))
    )
    update = VersionUpdate("session", 2, NEW, request, admit_requests=False, source_only=True)
    with pytest.raises(ValueError):
        replace(update, **changes)
