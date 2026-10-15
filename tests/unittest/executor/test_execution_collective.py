# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete participant coverage, per-worker ownership, and partial failures."""

from dataclasses import replace
from itertools import product
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentRequest, WorkerAlignmentState
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_collective import CollectivePublicationTransport, ExecutionParticipant
from tensorrt_llm.execution_fence import FenceReceipt, FenceRequest, RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    PublicationReceipt,
    VersionUpdate,
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", (1 << 63) + 7)
NEW = replace(OLD, epoch=OLD.epoch + 1)
REVISION = Revision("edit", OLD, NEW)
INPUT = AlignmentInput((1, 2, 3, 4, 5), cache_salt="shared")
ALIGN = AlignmentRequest("session", REVISION, INPUT)
FENCE = FenceRequest("session", REVISION)


@pytest.mark.parametrize("parent_limit", [None, 2])
def test_single_worker_keeps_extra_reuse_unless_parent_requires_a_bound(parent_limit):
    participant = ExecutionParticipant(0, 0)
    table = WorkerVersionTable("session")
    hold = Mock(num_tokens=3)
    table.drain_tracker = RequestDrainTracker()
    table.alignment_state = WorkerAlignmentState(lambda _: hold, lambda: None)
    transport = CollectivePublicationTransport(
        (participant,), {participant: LocalPublicationTransport(table)}
    )
    try:
        transport.publish(VersionUpdate("session", 1, OLD), Event())
        transport.publish(VersionUpdate("session", 2, NEW, ALIGN), Event())
        table.alignment_state.process()
        receipt = transport.poll_alignment(ALIGN, Event())
        transport.publish(VersionUpdate("session", 3, NEW, ALIGN, receipt, parent_limit), Event())
        assert (
            table.alignment_state.claim(NEW, 101, INPUT.input_token_ids, INPUT.cache_salt, None)
            == parent_limit
        )
        # Additional matching KV may appear after the original reservation.
        table.alignment_state.confirm_attachment(NEW, 101, 4 if parent_limit is None else 2)
        hold.close.assert_called_once()
    finally:
        transport.close()
        table.close()
        table.alignment_state.process()


def test_source_wake_runs_only_after_every_peer_records_alignment_and_reclaim():
    participants = (ExecutionParticipant(0, 0), ExecutionParticipant(1, 0))
    tables = [WorkerVersionTable("session") for _ in participants]
    holds = [Mock(num_tokens=3) for _ in participants]
    wakes = []

    def wake_all():
        # Model the fastest possible scheduler: process the broadcast at once.
        versions = [
            table.alignment_state._current[(OLD.tenant_id, OLD.scope_id)] for table in tables
        ]
        assert versions[0] == versions[1]
        wakes.append(versions[0])
        for table in tables:
            table.alignment_state.process()

    for rank, (table, hold) in enumerate(zip(tables, holds)):
        table.drain_tracker = RequestDrainTracker()
        table.alignment_state = WorkerAlignmentState(
            lambda revised, hold=hold: hold, wake_all if rank == 0 else lambda: None
        )
    transport = CollectivePublicationTransport(
        participants,
        {p: LocalPublicationTransport(t) for p, t in zip(participants, tables)},
        publication_source=participants[0],
    )
    controller = ExecutionController("session", transport)
    try:
        controller.register(OLD).wait()
        result = controller.advance(REVISION, alignment=INPUT)
        result.publication.wait()
        receipt = result.alignment.wait()
        assert receipt.reusable_prefix_length == 3
        controller.resume(receipt).wait()
        newer = replace(NEW, epoch=NEW.epoch + 1)
        controller.advance(Revision("abandon", NEW, newer)).publication.wait()
        assert wakes[-1] == newer
        for hold in holds:
            hold.close.assert_called_once()
    finally:
        controller.close()


def test_source_is_still_revoked_after_peer_publication_fails():
    participants = (ExecutionParticipant(0, 0), ExecutionParticipant(1, 0))
    calls = []
    update = VersionUpdate("session", 1, NEW)

    def source(value, stop):
        calls.append("source")
        return PublicationReceipt(value, value.version)

    def peer(value, stop):
        calls.append("peer")
        raise TimeoutError("lost peer")

    transport = CollectivePublicationTransport(
        participants,
        {participants[0]: Mock(publish=source), participants[1]: Mock(publish=peer)},
        publication_source=participants[0],
    )
    with pytest.raises(RuntimeError, match="publication failed"):
        transport.publish(update, Event())
    assert calls == ["peer", "source"]


@pytest.fixture(params=[(2, 1, 1, False), (2, 2, 1, False), (2, 1, 2, False), (2, 1, 1, True)])
def cluster(request):
    tp, pp, kv, ipc = request.param
    participants = tuple(ExecutionParticipant(*c) for c in product(range(tp), range(pp), range(kv)))
    tables = [WorkerVersionTable("session") for _ in participants]
    holds = [Mock(num_tokens=3) for _ in participants]
    for table, hold in zip(tables, holds):
        table.drain_tracker = RequestDrainTracker()
        table.alignment_state = WorkerAlignmentState(Mock(return_value=hold), lambda: None)
    servers = [ExecutionMailboxServer(t) for t in tables] if ipc else []
    children = (
        [Mock(wraps=IpcPublicationTransport(s.address)) for s in servers]
        if ipc
        else [Mock(wraps=LocalPublicationTransport(t)) for t in tables]
    )
    transport = CollectivePublicationTransport(participants, dict(zip(participants, children)))
    controller = ExecutionController("session", transport)
    yield SimpleNamespace(
        participants=participants,
        tables=tables,
        holds=holds,
        children=children,
        transport=transport,
        controller=controller,
    )
    controller.close()
    for server in servers:
        server.close()
    for table in tables:
        table.close()
        table.alignment_state.process()


def prepare(cluster):
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    return result, result.alignment.wait()


@pytest.mark.parametrize(
    "coordinates", [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (True, 0, 0), (0, 1.5, 0)]
)
def test_reject_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        ExecutionParticipant(*coordinates)


def test_membership_cannot_be_inferred_from_available_workers():
    first, second = ExecutionParticipant(0, 0), ExecutionParticipant(1, 0)
    one, two = Mock(), Mock()
    for expected, children in [
        ((), {}),
        ((first, first), {first: one}),
        ((first, second), {first: one}),
        ((first,), {first: one, second: two}),
        ((first, second), {first: one, second: one}),
    ]:
        with pytest.raises(ValueError):
            CollectivePublicationTransport(expected, children)
    with pytest.raises(TypeError):
        CollectivePublicationTransport(("rank-0",), {})
    mapping = {first: one, second: two}
    group = CollectivePublicationTransport((second, first), mapping)
    mapping.clear()
    assert group.participants == (first, second)
    group.close()
    one.close.assert_called_once()
    two.close.assert_called_once()


def test_fence_waits_for_every_rank_even_after_timeout(cluster):
    controller = cluster.controller
    controller.register(OLD).wait()
    for index, table in enumerate(cluster.tables):
        assert table.current_version("tenant", "scope") == OLD
        table.drain_tracker.enqueue(OLD, lambda i=index: i)
    result = controller.advance(REVISION)
    result.publication.wait()
    for index, table in enumerate(cluster.tables[:-1]):
        table.drain_tracker.complete(index, OLD)
    with pytest.raises(TimeoutError):
        result.fence.wait(0.02)
    cluster.tables[-1].drain_tracker.complete(len(cluster.tables) - 1, OLD)
    assert result.fence.wait() == FenceReceipt(FENCE)


def test_failed_publication_still_reaches_other_ranks_and_retry_is_idempotent(cluster):
    cluster.children[0].publish.side_effect = TimeoutError("lost rank")
    with pytest.raises(RuntimeError, match="publication failed"):
        cluster.controller.register(OLD).wait()
    for table in cluster.tables[1:]:
        assert table.current_version("tenant", "scope") == OLD
    with pytest.raises(ExecutionAuthorityError, match="not been published"):
        with cluster.controller.admission(OLD):
            pass
    cluster.children[0].publish.side_effect = None
    assert cluster.controller.register(OLD).wait().observed_version == OLD
    with cluster.controller.admission(OLD):
        pass


def test_alignment_waits_for_all_reservations_and_maps_local_ids(cluster):
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    for table in cluster.tables[:-1]:
        table.alignment_state.process()
    with pytest.raises(TimeoutError):
        result.alignment.wait(0.02)
    cluster.tables[-1].alignment_state.process()
    receipt = result.alignment.wait()
    assert receipt.reusable_prefix_length == 3
    local = [t.poll_alignment(ALIGN) for t in cluster.tables]
    assert len({r.reservation_id for r in local}) == len(local)
    assert receipt.reservation_id not in {r.reservation_id for r in local}
    assert cluster.controller.advance(REVISION, alignment=INPUT).alignment.wait() == receipt
    cluster.controller.resume(receipt).wait()
    with cluster.controller.admission(NEW):
        pass
    for table, expected in zip(cluster.tables, local):
        with table.admission(NEW):
            pass
        table.alignment_state.claim(NEW, 101, INPUT.input_token_ids, INPUT.cache_salt, None)
        assert table.poll_alignment(ALIGN) == expected


@pytest.mark.parametrize("common", [0, 2, 3])
def test_common_prefix_is_bound_before_admission_and_holds_survive_until_attachment(
    cluster, common
):
    cluster.holds[-1].num_tokens = common
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    receipt = result.alignment.wait()
    assert receipt.reusable_prefix_length == common
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with cluster.controller.admission(NEW):
            pass
    cluster.controller.resume(receipt).wait()
    with cluster.controller.admission(NEW):
        pass
    for table, hold in zip(cluster.tables, cluster.holds):
        state = table.alignment_state
        assert table.poll_alignment(ALIGN).reusable_prefix_length == hold.num_tokens
        assert state.claim(NEW, 101, INPUT.input_token_ids, INPUT.cache_salt, None) == common
        hold.close.assert_not_called()
        for wrong in {common - 1, common + 1}:
            with pytest.raises(RuntimeError, match="agreed KV prefix"):
                state.confirm_attachment(NEW, 101, wrong)
        hold.close.assert_not_called()
        state.confirm_attachment(NEW, 101, common)
        hold.settle.assert_called_once()
        hold.close.assert_called_once()


def test_common_prefix_cannot_bypass_runtime_minimum(cluster):
    cluster.holds[-1].num_tokens = 2
    required = replace(INPUT, min_reuse_tokens=3)
    cluster.controller.register(OLD).wait()
    result = cluster.controller.advance(REVISION, alignment=required)
    result.publication.wait()
    for table in cluster.tables:
        table.alignment_state.process()
    with pytest.raises(RuntimeError, match="ALIGN"):
        result.alignment.wait()
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with cluster.controller.admission(NEW):
            pass


def test_parent_collective_limit_cannot_be_expanded_by_child(cluster):
    _, receipt = prepare(cluster)
    update = VersionUpdate("session", 3, NEW, ALIGN, receipt, 1)
    assert cluster.transport.publish(update, Event()).update == update
    for table in cluster.tables:
        assert (
            table.alignment_state.claim(NEW, 101, INPUT.input_token_ids, INPUT.cache_salt, None)
            == 1
        )


def test_partial_resume_does_not_open_frontend_and_retry_keeps_holds(cluster):
    cluster.holds[-1].num_tokens = 2
    _, receipt = prepare(cluster)
    cluster.children[-1].publish.side_effect = TimeoutError("lost acknowledgement")
    with pytest.raises(RuntimeError, match="publication failed"):
        cluster.controller.resume(receipt).wait()
    assert cluster.tables[0].alignment_state.snapshot()["records"][0]["resumed"]
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with cluster.controller.admission(NEW):
            pass
    cluster.children[-1].publish.side_effect = None
    cluster.controller.resume(receipt).wait()
    for hold in cluster.holds:
        hold.close.assert_not_called()
    for table in cluster.tables:
        assert table.alignment_state.snapshot()["records"][0]["resume_prefix_length"] == 2
    with cluster.controller.admission(NEW):
        pass


def test_retry_after_worker_applied_resume_but_lost_ack(cluster):
    _, receipt = prepare(cluster)
    child = cluster.children[-1]
    publish = child.publish._mock_wraps

    def lose_ack(update, stop):
        publish(update, stop)
        raise TimeoutError("reply lost after application")

    child.publish.side_effect = lose_ack
    with pytest.raises(RuntimeError, match="publication failed"):
        cluster.controller.resume(receipt).wait()
    assert all(t.alignment_state.snapshot()["records"][0]["resumed"] for t in cluster.tables)
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with cluster.controller.admission(NEW):
            pass
    child.publish.side_effect = None
    cluster.controller.resume(receipt).wait()
    assert all(t.alignment_state.snapshot()["held_count"] == 1 for t in cluster.tables)


def test_resume_revalidates_worker_reservation_identity_before_any_send(cluster):
    _, receipt = prepare(cluster)
    local = cluster.tables[-1].poll_alignment(ALIGN)
    cluster.children[-1].poll_alignment.return_value = replace(local, reservation_id="other")
    for child in cluster.children:
        child.publish.reset_mock()
    with pytest.raises(RuntimeError, match="reservation changed"):
        cluster.controller.resume(receipt).wait()
    for child in cluster.children:
        child.publish.assert_not_called()
    assert not any(t.alignment_state.snapshot()["records"][0]["resumed"] for t in cluster.tables)


def test_supersession_invalidates_every_hold_and_old_resume(cluster):
    _, receipt = prepare(cluster)
    newer = replace(NEW, epoch=NEW.epoch + 1)
    update = cluster.controller.advance(Revision("replace-again", NEW, newer))
    update.publication.wait()
    update.fence.wait()
    with pytest.raises(ValueError, match="obsolete"):
        cluster.controller.resume(receipt)
    for table, hold in zip(cluster.tables, cluster.holds):
        table.alignment_state.process()
        hold.close.assert_called_once()
        assert table.alignment_state.snapshot()["held_count"] == 0


@pytest.mark.parametrize("operation", ["poll_fence", "poll_alignment"])
def test_not_ready_worker_does_not_hide_another_workers_failure(cluster, operation):
    method = getattr(cluster.children[0], operation)
    method.return_value = None
    getattr(cluster.children[-1], operation).side_effect = RuntimeError("failed peer")
    # These direct calls run before the controller has sent anything, so all
    # transport operations still occur serially on this test thread.
    request = FENCE if operation == "poll_fence" else ALIGN
    with pytest.raises(RuntimeError, match="failed for participants") as caught:
        getattr(cluster.transport, operation)(request, Event())
    assert "failed peer" in str(caught.value.__cause__)
    for child in cluster.children:
        getattr(child, operation).assert_called_once()


@pytest.mark.parametrize("operation", ["publish", "poll_fence", "poll_alignment"])
def test_wrong_scope_acknowledgement_cannot_cover_a_participant(operation):
    participants = (ExecutionParticipant(0, 0), ExecutionParticipant(1, 0))
    children = [Mock(), Mock()]
    foreign_old = replace(OLD, scope_id="foreign")
    foreign_new = replace(NEW, scope_id="foreign")
    foreign_revision = Revision("foreign", foreign_old, foreign_new)
    if operation == "publish":
        query = VersionUpdate("session", 1, OLD)
        children[0].publish.return_value = PublicationReceipt(query, OLD)
        wrong_query = replace(query, version=foreign_old)
        children[1].publish.return_value = PublicationReceipt(wrong_query, foreign_old)
    elif operation == "poll_fence":
        query = FENCE
        children[0].poll_fence.return_value = FenceReceipt(FENCE)
        children[1].poll_fence.return_value = FenceReceipt(
            FenceRequest("session", foreign_revision)
        )
    else:
        from tensorrt_llm.execution_alignment import AlignmentReceipt

        query = ALIGN
        children[0].poll_alignment.return_value = AlignmentReceipt(ALIGN, "first", 3)
        children[1].poll_alignment.return_value = AlignmentReceipt(
            AlignmentRequest("session", foreign_revision, INPUT), "second", 3
        )
    group = CollectivePublicationTransport(participants, dict(zip(participants, children)))
    with pytest.raises(RuntimeError, match="failed for participants") as caught:
        getattr(group, operation)(query, Event())
    assert isinstance(caught.value.__cause__, ValueError)
    group.close()


def test_publication_reports_the_oldest_observation_across_participants():
    participants = (ExecutionParticipant(0, 0), ExecutionParticipant(1, 0))
    update = VersionUpdate("session", 1, OLD)
    children = [Mock(), Mock()]
    children[0].publish.return_value = PublicationReceipt(update, NEW)
    children[1].publish.return_value = PublicationReceipt(update, OLD)
    group = CollectivePublicationTransport(participants, dict(zip(participants, children)))
    assert group.publish(update, Event()).observed_version == OLD
    group.close()


def test_shutdown_attempts_every_child_even_if_one_close_fails():
    participants = (ExecutionParticipant(0, 0), ExecutionParticipant(1, 0))
    children = [Mock(), Mock()]
    children[0].close.side_effect = OSError("socket failure")
    group = CollectivePublicationTransport(participants, dict(zip(participants, children)))
    with pytest.raises(RuntimeError, match="close failed"):
        group.close()
    for child in children:
        child.close.assert_called_once()
    group.close()
    with pytest.raises(RuntimeError, match="closed"):
        group.publish(VersionUpdate("session", 1, OLD), Event())


def test_stop_during_last_ack_never_returns_completion():
    participants = (ExecutionParticipant(0, 0), ExecutionParticipant(1, 0))
    children = [Mock(), Mock()]
    update = VersionUpdate("session", 1, OLD)
    stop = Event()
    children[0].publish.return_value = PublicationReceipt(update, OLD)

    def last_ack(*args):
        stop.set()
        return PublicationReceipt(update, OLD)

    children[-1].publish.side_effect = last_ack
    group = CollectivePublicationTransport(participants, dict(zip(participants, children)))
    with pytest.raises(RuntimeError, match="stopped before acknowledgement"):
        group.publish(update, stop)
    group.close()
