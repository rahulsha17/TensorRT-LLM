# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An ALIGN copy cannot grant generation authority or evade physical drain."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as WaitTimeout
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_execution_session_drain import VERSION, _bind, _request, _worker
from test_execution_transfer_admission import _sender

from tensorrt_llm._torch.disaggregation.base import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.native.execution_wire import TransferIdentityError
from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentRequest, WorkerAlignmentState
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_fence import (
    FenceReceipt,
    FenceRequest,
    RequestDrainTracker,
    ScopeResourceClaim,
)
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import VersionUpdate

pytestmark = pytest.mark.cpu_only
NEW = replace(VERSION, epoch=1)
REVISION = Revision("edit", VERSION, NEW)
ALIGN = AlignmentRequest("authority", REVISION, AlignmentInput(tuple(range(40))))
FENCE = FenceReceipt(FenceRequest("authority", REVISION))


def _pending(*, old_active=False):
    worker, drain = _worker(), RequestDrainTracker()
    table = _bind(worker, drain)
    reservation = SimpleNamespace(num_tokens=32, settle=Mock(), close=Mock())
    table.alignment_state = WorkerAlignmentState(lambda _: reservation, Mock())
    if old_active:
        drain.enqueue(VERSION, lambda: 7)
    table.apply(VersionUpdate("authority", 2, NEW, alignment=ALIGN, admit_requests=False))
    table.alignment_state.process()
    return worker, table, drain


def _permit(table, direction="send", **changes):
    args = dict(request_id=41, transfer_id=99, direction=direction, prefix_length=32)
    args.update(changes)
    return table.admit_alignment_transfer(ALIGN, FENCE, **args)


def _copy(worker, direction="send", request=None, **changes):
    create = worker.create_tx_session if direction == "send" else worker.create_rx_session
    args = dict(alignment=ALIGN, fence=FENCE)
    args.update(changes)
    return create(_request(NEW) if request is None else request, **args)


def _resume(table, receipt):
    return table.apply(VersionUpdate("authority", 3, NEW, alignment=ALIGN, resume=receipt))


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_copy_before_resume_does_not_admit_a_generation_request(direction):
    worker, table, drain = _pending()
    receipt = table.poll_alignment(ALIGN)
    session = _copy(worker, direction)
    permit = session._cleanup.alignment_permit
    assert isinstance(permit.claim, ScopeResourceClaim)
    assert not drain._requests and not table._gate._executions
    assert drain.has_resource(permit.claim)
    assert table.poll_alignment(ALIGN) is None
    with worker._execution_wire.admission(99, direction):
        pass
    with pytest.raises(ExecutionAuthorityError):
        with table.admission(NEW):
            pytest.fail("copy permit admitted generation")
    with pytest.raises(RuntimeError, match="live ALIGN"):
        _resume(table, receipt)
    assert session.close()
    # Physical cleanup, including cancellation, is not prefix acceptance.
    assert table.poll_alignment(ALIGN) is None
    with pytest.raises(ExecutionAuthorityError, match="closed"):
        with worker._execution_wire.admission(99, direction):
            pass
    table.accept_alignment_transfer(permit)
    assert table.poll_alignment(ALIGN) == receipt
    _resume(table, receipt)
    with table.admission(NEW) as execution:
        execution.finish()
    with pytest.raises(ExecutionAuthorityError, match="RESUME"):
        _permit(table, transfer_id=100)


def test_old_request_must_fence_before_any_alignment_claim_is_created():
    worker, table, drain = _pending(old_active=True)
    with pytest.raises(ExecutionAuthorityError, match="FENCE"):
        _copy(worker)
    assert not drain._resources
    worker._sender.setup_session.assert_not_called()
    drain.complete(7, VERSION)
    assert _copy(worker).close()


@pytest.mark.parametrize(
    "changes",
    [
        dict(request_id=True),
        dict(transfer_id=True),
        dict(transfer_id=-1),
        dict(transfer_id=1 << 63),
        dict(direction="both"),
        dict(prefix_length=True),
        dict(prefix_length=0),
        dict(prefix_length=40),
        dict(prefix_length=41),
    ],
)
def test_invalid_copy_permit_does_not_create_an_owner(changes):
    _, table, drain = _pending()
    with pytest.raises(ValueError):
        _permit(table, **changes)
    assert not drain._resources and not table._alignment_transfers


@pytest.mark.parametrize(
    "state", ["wrong_fence", "wrong_input", "no_alignment", "closed", "revoked"]
)
def test_copy_requires_current_published_alignment_and_fence(state):
    _, table, drain = _pending()
    request, fence = ALIGN, FENCE
    if state == "wrong_fence":
        fence = FenceReceipt(replace(FENCE.request, session_id="other"))
    elif state == "wrong_input":
        request = replace(ALIGN, revised_input=AlignmentInput(tuple(range(41))))
    elif state == "no_alignment":
        table.apply(VersionUpdate("authority", 3, NEW, admit_requests=False))
    elif state == "closed":
        table.close()
    else:
        table.apply(VersionUpdate("authority", 3, replace(NEW, epoch=2)))
    with pytest.raises((ValueError, RuntimeError)):
        table.admit_alignment_transfer(
            request, fence, request_id=41, transfer_id=99, direction="send", prefix_length=32
        )
    assert not drain._resources


@pytest.mark.parametrize("field", ["tokens", "positions", "salt", "lora"])
def test_native_copy_cannot_substitute_the_published_prefix(field):
    worker, table, drain = _pending()
    request = _request(NEW)
    if field == "tokens":
        request.get_tokens = lambda _: [99] * 32
    elif field == "positions":
        request.py_position_ids = list(range(1, 33))
    elif field == "salt":
        request.cache_salt = "other"
    else:
        request.lora_task_id = 7
    with pytest.raises(TransferIdentityError, match="permitted prefix"):
        _copy(worker, request=request)
    assert not drain._resources
    worker._sender.setup_session.assert_not_called()
    with pytest.raises(RuntimeError, match="copy failed"):
        table.poll_alignment(ALIGN)


@pytest.mark.parametrize(
    "state", ["revoke", "close", "tracker_change", "alignment_change", "failed"]
)
def test_revocation_closes_copy_admission_but_does_not_release_resources(state):
    worker, table, drain = _pending()
    session = _copy(worker)
    permit = session._cleanup.alignment_permit
    if state == "revoke":
        table.apply(VersionUpdate("authority", 3, replace(NEW, epoch=2)))
    elif state == "close":
        table.close()
    elif state == "tracker_change":
        table.drain_tracker = RequestDrainTracker()
    elif state == "alignment_change":
        table.alignment_state = WorkerAlignmentState(Mock(), Mock())
    else:
        table.fail_alignment_transfer(permit, RuntimeError("failed"))
    with pytest.raises(RuntimeError):
        with worker._execution_wire.admission(99, "send"):
            pass
    assert drain.has_resource(permit.claim) and not drain.drained(NEW)
    assert session.close()
    assert drain.drained(NEW)


def test_next_revision_fence_waits_for_the_previous_alignment_copy():
    worker, table, drain = _pending()
    session = _copy(worker)
    second = Revision("again", NEW, replace(NEW, epoch=2))
    table.apply(VersionUpdate("authority", 3, second.successor_version))
    fence = FenceRequest("authority", second)
    assert table.poll_fence(fence) is None
    assert session.close()
    assert table.poll_fence(fence) == FenceReceipt(fence)
    with pytest.raises(ExecutionAuthorityError):
        table.accept_alignment_transfer(session._cleanup.alignment_permit)


def test_copy_receipts_and_claims_cannot_be_forged_or_reused():
    _, table, drain = _pending()
    permit = _permit(table)
    with pytest.raises(ExecutionAuthorityError, match="unknown"):
        with table.alignment_transfer_admission(replace(permit)):
            pass
    with pytest.raises(RuntimeError, match="physical resources"):
        table.accept_alignment_transfer(permit)
    drain.release_resource(permit.claim)
    table.accept_alignment_transfer(permit)
    table.accept_alignment_transfer(permit)
    with pytest.raises(ValueError, match="reused"):
        _permit(table)
    table.fail_alignment_transfer(permit, RuntimeError("contradictory late evidence"))
    with pytest.raises(RuntimeError, match="closed"):
        table.current_version("tenant", "scope")


def test_copy_operation_and_advance_share_the_publication_lock():
    worker, table, drain = _pending()
    session = _copy(worker)
    started = Event()

    def advance():
        started.set()
        table.apply(VersionUpdate("authority", 3, replace(NEW, epoch=2)))

    with ThreadPoolExecutor() as pool:
        with worker._execution_wire.admission(99, "send"):
            future = pool.submit(advance)
            assert started.wait(1)
            with pytest.raises(WaitTimeout):
                future.result(timeout=0.03)
        future.result(timeout=1)
    assert not drain.drained(NEW)
    assert session.close() and drain.drained(NEW)


def test_native_backend_rechecks_copy_permission_after_task_admission():
    worker, table, drain = _pending()
    session = _copy(worker)
    sender, task, _, _ = _sender()
    sender._execution_wire = worker._execution_wire
    assert sender._begin_task_operation(task, 7)
    table.apply(VersionUpdate("authority", 3, replace(NEW, epoch=2)))
    with pytest.raises(RuntimeError):
        sender._submit_transfer(task, 7, object())
    sender._agent.submit_transfer_requests.assert_not_called()
    assert drain.has_resource(session._cleanup.alignment_permit.claim)
    assert session.close()


@pytest.mark.parametrize(
    "outcome", ["success", "validation_error", "revision_during_validation", "not_finished"]
)
def test_native_copy_acceptance_requires_success_and_engine_validation(outcome):
    worker, table, drain = _pending()
    receipt = table.poll_alignment(ALIGN)
    session = _copy(worker)
    session.send(Chunk([], [], TokenRange(0, 32), True))
    if outcome != "not_finished":
        session.kv_tasks[0].complete()
    validation = Mock()
    if outcome == "validation_error":
        validation.side_effect = ValueError("destination prefix differs")
    elif outcome == "revision_during_validation":
        validation.side_effect = lambda: table.apply(
            VersionUpdate("authority", 3, replace(NEW, epoch=2))
        )
    if outcome == "success":
        worker.finish_alignment_transfer(session, validation)
        validation.assert_called_once()
        assert table.poll_alignment(ALIGN) == receipt
        assert not drain._resources
    else:
        with pytest.raises((ValueError, RuntimeError)):
            worker.finish_alignment_transfer(session, validation)
        if outcome == "not_finished":
            validation.assert_not_called()
            assert drain._resources
            session.kv_tasks[0].complete()
            assert session.close()
        elif outcome == "validation_error":
            with pytest.raises(RuntimeError, match="copy failed"):
                table.poll_alignment(ALIGN)
        else:
            assert drain.drained(NEW)


@pytest.mark.parametrize("action", ["advance", "shutdown"])
def test_alignment_reservation_survives_until_copy_cleanup(action):
    worker, table, drain = _pending()
    state = table.alignment_state
    reservation = state._records[ALIGN].reservation
    session = _copy(worker)
    if action == "advance":
        table.apply(VersionUpdate("authority", 3, replace(NEW, epoch=2)))
        state.process()
    else:
        table.close()
        with pytest.raises(RuntimeError, match="teardown must wait"):
            state.process()
    reservation.close.assert_not_called()
    reservation.settle.assert_not_called()
    assert state.snapshot()["held_count"] == 1
    state._wake.reset_mock()
    assert session.close()
    state._wake.assert_called_once()
    state.process()
    reservation.settle.assert_called_once()
    reservation.close.assert_called_once()
    assert state.snapshot()["held_count"] == 0
    assert drain.drained(NEW)


def test_unknown_copy_completion_cannot_release_alignment_reservation():
    worker, table, drain = _pending()
    state = table.alignment_state
    reservation = state._records[ALIGN].reservation
    _copy(worker)
    drain.fail(RuntimeError("native close is uncertain"))
    table.close()
    with pytest.raises(RuntimeError, match="completion is unknown"):
        state.process()
    reservation.close.assert_not_called()
    assert state.snapshot()["unsafe"]


def test_two_copies_both_need_acceptance_and_physical_cleanup():
    worker, table, drain = _pending()
    receipt = table.poll_alignment(ALIGN)
    first = _copy(worker)
    second = _copy(worker, request=_request(NEW, wire_id=100))
    assert first.close()
    table.accept_alignment_transfer(first._cleanup.alignment_permit)
    assert table.poll_alignment(ALIGN) is None
    assert second.close()
    assert table.poll_alignment(ALIGN) is None
    table.accept_alignment_transfer(second._cleanup.alignment_permit)
    assert table.poll_alignment(ALIGN) == receipt
    assert drain.drained(NEW)
