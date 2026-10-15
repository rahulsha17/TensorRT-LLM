# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Revocation orders host transfer reservations without blocking physical drain."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as WaitTimeout
from dataclasses import replace
from functools import partial
from threading import Event, Lock, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.native import transfer
from tensorrt_llm._torch.disaggregation.native.execution_content import TransferPrefixIdentity
from tensorrt_llm._torch.disaggregation.native.execution_wire import (
    ExecutionTransferWire,
    TransferIdentityError,
)
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentRequest
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)


def _authority(direction):
    table = WorkerVersionTable("authority")
    table.apply(VersionUpdate("authority", 1, VERSION))
    wire = ExecutionTransferWire()
    wire.bind(table)
    wire.register_local(
        99, direction, VERSION, "cache", TransferPrefixIdentity.from_tokens("model", (1, 2))
    )
    return wire, table


def _revoke(table, state="revoked"):
    if state == "closed":
        table.close()
    elif state == "alignment":
        revision = Revision("edit", replace(VERSION, epoch=6), VERSION)
        alignment = AlignmentRequest("authority", revision, AlignmentInput((1, 2, 3)))
        table.apply(VersionUpdate("authority", 2, VERSION, alignment=alignment))
    else:
        table.apply(VersionUpdate("authority", 2, replace(VERSION, epoch=8)))


def _sender():
    wire, table = _authority("send")
    watchdog = RetirementWatchdog(Mock())
    retirement = watchdog.create_owner(41, "send", 60.0)
    task = transfer.SendTaskBase(DisaggregatedParams(disagg_request_id=99))
    task.bind_logical_outcomes(transfer._LogicalOutcomes(retirement))
    sender = object.__new__(transfer.Sender)
    sender._execution_wire = wire
    sender._enforce_physical_ownership = True
    sender._ownership_poisoned = None
    sender._ownership_poison_lock = watchdog.lock
    sender._sessions_lock = Lock()
    sender._sessions = {99: SimpleNamespace(lock=Lock(), _closed=False, has_failed=lambda: False)}
    sender._agent = Mock()
    sender._agent.submit_transfer_requests.return_value.wait.return_value = True
    sender._shutdown = True
    return sender, task, table, retirement


def _receiver(*, auxiliary=False):
    wire, table = _authority("receive")
    watchdog = RetirementWatchdog(Mock())
    receiver = SimpleNamespace(
        _execution_wire=wire,
        _enforce_physical_ownership=True,
        _ownership_poisoned=None,
        _get_ownership_admission_lock=lambda: watchdog.lock,
        _bounce=Mock(),
        setup_session=Mock(),
        clear_session=Mock(),
        clear_session_req_info=Mock(),
        send_cancel_to_senders=lambda *args, **kwargs: set(),
    )
    params = DisaggregatedParams(
        disagg_request_id=99,
        schedule_style=(
            DisaggScheduleStyle.GENERATION_FIRST if auxiliary else DisaggScheduleStyle.CONTEXT_FIRST
        ),
    )
    session = transfer.RxSession(41, params, receiver, timeout_s=60.0, retirement_watchdog=watchdog)
    task = session.prepare_receive(SimpleNamespace(is_last=False))
    task.expected_transfers = 2
    return session, task, table


@pytest.mark.parametrize("state", ["revoked", "closed", "alignment"])
def test_wire_operation_requires_current_resumed_authority(state):
    wire, table = _authority("send")
    _revoke(table, state)
    with pytest.raises(ExecutionAuthorityError):
        with wire.admission(99, "send"):
            pytest.fail("unauthorized operation admitted")
    assert not table._gate._executions


def test_peer_binding_without_local_session_cannot_admit_operation():
    peer, _ = _authority("receive")
    wire = ExecutionTransferWire()
    wire.bind(WorkerVersionTable("authority"))
    wire.decode(peer.encode([transfer.MessageType.CANCEL_SESSION, b"99"]))
    with pytest.raises(TransferIdentityError, match="locally admitted"):
        with wire.admission(99, "send"):
            pytest.fail("peer metadata granted local admission")


def test_receive_session_cannot_grant_sender_admission():
    wire, _ = _authority("receive")
    with pytest.raises(TransferIdentityError, match="locally admitted"):
        with wire.admission(99, "send"):
            pytest.fail("receive admission granted send authority")


@pytest.mark.parametrize("bound", [False, True])
def test_untagged_operations_keep_legacy_admission(bound):
    wire = ExecutionTransferWire()
    if bound:
        wire.bind(WorkerVersionTable("authority"))
    wire.register_local(99, "send", None, None)
    with wire.admission(99, "send"):
        pass


@pytest.mark.parametrize("raises", [False, True])
def test_operation_context_does_not_leave_host_gate_notifications(raises):
    wire, table = _authority("send")
    if raises:
        with pytest.raises(ValueError, match="reservation failed"):
            with wire.admission(99, "send"):
                raise ValueError("reservation failed")
    else:
        with wire.admission(99, "send"):
            assert table._gate._executions
    assert not table._gate._executions


@pytest.mark.parametrize("state", ["revoked", "closed", "alignment"])
def test_sender_rejects_new_task_before_backend_or_source_access(state):
    sender, task, table, retirement = _sender()
    _revoke(table, state)
    assert sender._begin_task_operation(task, 0) is False
    assert task.resources_drained and retirement.can_retire()
    sender._agent.submit_transfer_requests.assert_not_called()
    assert sender._ownership_poisoned is None
    assert sender._begin_task_operation(task, 0) is None


def test_duplicate_dispatch_after_revocation_cannot_retire_prior_operation():
    sender, task, table, _ = _sender()
    assert sender._begin_task_operation(task, 0) is True
    _revoke(table)
    assert sender._begin_task_operation(task, 0) is None
    assert not task.resources_drained
    task.retire_unsubmitted_physical_operation(0)
    assert task.resources_drained


@pytest.mark.parametrize("state", ["revoked", "closed", "alignment"])
def test_sender_rechecks_after_task_admission_before_backend_submission(state):
    sender, task, table, retirement = _sender()
    assert sender._begin_task_operation(task, 0) is True
    _revoke(table, state)
    with pytest.raises(transfer._TransferNotSubmittedError, match="authority"):
        sender._submit_transfer(task, 0, object())
    sender._agent.submit_transfer_requests.assert_not_called()
    assert task.resources_drained and retirement.can_retire()
    assert sender._ownership_poisoned is None


@pytest.mark.parametrize("boundary", ["submit", "wait"])
def test_revocation_never_waits_for_backend_and_cannot_release_its_accessors(boundary):
    sender, task, table, retirement = _sender()
    assert sender._begin_task_operation(task, 0) is True
    entered, release = Event(), Event()
    request = object()
    status = Mock()
    status.wait.return_value = True

    def pause():
        entered.set()
        assert release.wait(5)

    if boundary == "submit":

        def submit(_request):
            pause()
            return status

        sender._agent.submit_transfer_requests.side_effect = submit
    else:
        sender._agent.submit_transfer_requests.return_value = status

        def wait():
            pause()
            return True

        status.wait.side_effect = wait

    with ThreadPoolExecutor(2) as pool:
        sending = pool.submit(sender._submit_transfer, task, 0, request)
        try:
            assert entered.wait(5)
            pool.submit(_revoke, table).result(5)
            assert not task.resources_drained and not retirement.can_retire()
            assert task._physical_operations[0].request is request
        finally:
            release.set()
        assert sending.result(5) == (True, None)
    assert task.resources_drained and retirement.can_retire()
    assert task._physical_operations[0].request is None


def test_backend_reservation_is_atomic_with_authority_publication(monkeypatch):
    sender, task, table, _ = _sender()
    assert sender._begin_task_operation(task, 0) is True
    entered, release, publishing = Event(), Event(), Event()
    begin = task.begin_backend_submission

    def reserve(*args):
        entered.set()
        assert release.wait(5)
        begin(*args)

    def publish():
        publishing.set()
        _revoke(table)

    monkeypatch.setattr(task, "begin_backend_submission", reserve)
    with ThreadPoolExecutor(2) as pool:
        sending = pool.submit(sender._submit_transfer, task, 0, object())
        try:
            assert entered.wait(5)
            publication = pool.submit(publish)
            assert publishing.wait(5)
            with pytest.raises(WaitTimeout):
                publication.result(0.05)
        finally:
            release.set()
        assert sending.result(5) == (True, None)
        publication.result(5)
    assert task.resources_drained


def test_uncertain_backend_after_revocation_keeps_evidence_until_done():
    sender, task, table, retirement = _sender()
    assert sender._begin_task_operation(task, 0) is True
    status = sender._agent.submit_transfer_requests.return_value

    def fail_wait():
        _revoke(table)
        raise RuntimeError("backend query failed")

    status.wait.side_effect = fail_wait
    request = object()
    assert sender._submit_transfer(task, 0, request) == (False, "backend query failed")
    assert not task.resources_drained and not retirement.can_retire()
    assert task._physical_operations[0].request is request
    assert sender._begin_task_operation(task, 1) is False
    assert task._physical_operations[0].request is request
    assert not task.resources_drained
    status.is_completed.return_value = False
    assert not task.poll_in_doubt_physical_operation(0)
    table.close()
    status.is_completed.return_value = True
    assert task.poll_in_doubt_physical_operation(0)
    assert task.resources_drained and retirement.can_retire()


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("state", ["revoked", "closed", "alignment"])
def test_receive_rejects_before_exposing_destination_or_publishing(state, auxiliary):
    session, task, table = _receiver(auxiliary=auxiliary)
    _revoke(table, state)
    publish = Mock()
    session._receiver.dispatch_task = lambda task: session.try_begin_transfer(
        task.slice_id, {"peer"}, {0, 1}, publish=publish, published_writers=set()
    )
    with pytest.raises(ExecutionAuthorityError):
        session.dispatch_prepared_receive(task)
    publish.assert_not_called()
    assert not session._retirement._claims
    assert not session._publication_may_have_escaped
    assert session.has_failed() and task.is_done
    assert task.status == transfer.TaskStatus.ERROR
    assert task.cancel_unpublished()
    if session._aux_physical_owner is not None:
        assert session._aux_physical_owner.cancel_unpublished()
    assert task.resources_drained


@pytest.mark.parametrize("auxiliary", [False, True])
def test_rejected_later_slice_cannot_release_already_published_kv_or_aux(auxiliary):
    session, first, table = _receiver(auxiliary=auxiliary)
    published = set()
    assert session.try_begin_transfer(
        0, {"peer"}, {0, 1}, publish=lambda: published.update({0, 1}), published_writers=published
    )
    # A published earlier slice and its AUX owner survive rejection of the next.
    previous_claims = dict(session._retirement._claims)
    second = session.prepare_receive(SimpleNamespace(is_last=False))
    second.expected_transfers = 2
    _revoke(table)
    session._receiver.dispatch_task = lambda task: session.try_begin_transfer(
        task.slice_id, {"peer"}, {0, 1}
    )
    with pytest.raises(ExecutionAuthorityError):
        session.dispatch_prepared_receive(second)
    assert session.has_failed() and second.is_done
    assert session._retirement._claims == previous_claims
    assert not first.resources_drained
    table.close()
    for rank in (0, 1):
        first.record_writer_result(rank, False, wait_for_local_completion=False)
    if auxiliary:
        assert not session._retirement.can_retire()
        for rank in (0, 1):
            session._aux_physical_owner.record_writer_result(
                rank, False, wait_for_local_completion=False
            )
    assert first.resources_drained and second.resources_drained
    assert session._retirement.can_retire()


@pytest.mark.parametrize("auxiliary", [False, True])
def test_fence_waits_for_session_cleanup_after_revoked_receive_rejection(auxiliary):
    session, task, table = _receiver(auxiliary=auxiliary)
    drain = table.drain_tracker = RequestDrainTracker()
    drain.enqueue(VERSION, lambda: 41)
    claim = table.admit_resource(41, VERSION, "kv-receive-session")
    session._cleanup.bind_claim(drain, claim)
    drain.complete(41, VERSION)
    _revoke(table)
    session._receiver.dispatch_task = lambda task: session.try_begin_transfer(
        task.slice_id, {"peer"}, {0, 1}
    )
    with pytest.raises(ExecutionAuthorityError):
        session.dispatch_prepared_receive(task)
    assert session.wait_complete() == transfer.WaitResult.FAILED
    assert task.resources_drained
    assert not drain.drained(VERSION)
    assert session.close()
    assert drain.drained(VERSION)


@pytest.mark.parametrize("partial_failure", [False, True])
def test_revocation_during_publication_retains_only_proven_outstanding_writers(partial_failure):
    session, task, table = _receiver()
    entered, release = Event(), Event()
    published = set()

    def publish():
        published.add(0)
        entered.set()
        assert release.wait(5)
        if partial_failure:
            raise RuntimeError("second send failed")
        published.add(1)

    with ThreadPoolExecutor(2) as pool:
        sending = pool.submit(
            session.try_begin_transfer,
            0,
            {"peer"},
            {0, 1},
            publish=publish,
            published_writers=published,
        )
        try:
            assert entered.wait(5)
            pool.submit(_revoke, table).result(5)
            assert not task.resources_drained and not session._retirement.can_retire()
        finally:
            release.set()
        if partial_failure:
            with pytest.raises(RuntimeError, match="second send failed"):
                sending.result(5)
            session._receiver._bounce.abort_publication.assert_called_once_with((99, 0), {0})
        else:
            assert sending.result(5)
    table.close()
    for rank in sorted(published):
        assert not task.resources_drained
        task.record_writer_result(rank, False, wait_for_local_completion=False)
    assert task.resources_drained and session._retirement.can_retire()


def test_receive_reservation_is_atomic_but_later_network_publication_is_not_locked(monkeypatch):
    session, task, table = _receiver()
    entered, release, publishing = Event(), Event(), Event()
    owner = task._get_physical_owner()
    seal = owner.seal_writer_cohort

    def reserve(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        seal(*args, **kwargs)

    def advance():
        publishing.set()
        _revoke(table)

    monkeypatch.setattr(owner, "seal_writer_cohort", reserve)
    with ThreadPoolExecutor(2) as pool:
        receiving = pool.submit(session.try_begin_transfer, 0, {"peer"}, {0, 1})
        try:
            assert entered.wait(5)
            publication = pool.submit(advance)
            assert publishing.wait(5)
            with pytest.raises(WaitTimeout):
                publication.result(0.05)
        finally:
            release.set()
        assert receiving.result(5)
        publication.result(5)
    assert not task.resources_drained
    for rank in (0, 1):
        task.record_writer_result(rank, False, wait_for_local_completion=False)
    assert task.resources_drained


@pytest.mark.parametrize("boundary", ["task", "backend", "receiver"])
def test_waiting_for_session_or_retirement_lock_does_not_stall_revocation(boundary):
    class ContendedLock:
        def __init__(self):
            self.lock = RLock()
            self.entered = Event()

        def __enter__(self):
            self.entered.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    lock = ContendedLock()
    if boundary == "receiver":
        session, task, table = _receiver()
        session.lock = lock
        operation = partial(session.try_begin_transfer, 0, {"peer"}, {0, 1})
    else:
        sender, task, table, _ = _sender()
        if boundary == "task":
            sender._sessions[99].lock = lock
            operation = partial(sender._begin_task_operation, task, 0)
        else:
            assert sender._begin_task_operation(task, 0) is True
            sender._ownership_poison_lock = lock
            operation = partial(sender._submit_transfer, task, 0, object())
    with ThreadPoolExecutor(2) as pool:
        lock.lock.acquire()
        try:
            pending = pool.submit(operation)
            assert lock.entered.wait(5)
            pool.submit(_revoke, table).result(5)
        finally:
            lock.lock.release()
        if boundary in ("backend", "receiver"):
            error = (
                transfer._TransferNotSubmittedError
                if boundary == "backend"
                else ExecutionAuthorityError
            )
            with pytest.raises(error):
                pending.result(5)
        else:
            assert pending.result(5) is False
    if boundary == "receiver":
        assert task.cancel_unpublished()
    assert task.resources_drained
