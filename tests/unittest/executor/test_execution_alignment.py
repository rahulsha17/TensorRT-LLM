# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ALIGN ownership, revision binding, and IPC status without native dependencies."""

from dataclasses import replace
from threading import Event, get_ident
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_alignment import (
    AlignmentInput,
    AlignmentRequest,
    AlignmentSafetyError,
    WorkerAlignmentState,
)
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    VersionUpdate,
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
REVISION = Revision("edit", OLD, NEW)
INPUT = AlignmentInput((1, 2, 3, 4, 5), cache_salt="shared")


@pytest.mark.parametrize("prefix", [-1, 4, True, 1.5])
def test_resume_limit_rejects_invalid_or_unreserved_prefix(prefix):
    request = AlignmentRequest("session", REVISION, INPUT)
    state = WorkerAlignmentState(lambda _: Mock(num_tokens=3), lambda: None)
    state.observe(NEW, request)
    state.process()
    receipt = state.poll(request)
    with pytest.raises(ValueError, match="prefix"):
        VersionUpdate("session", 2, NEW, request, receipt, prefix)
    with pytest.raises(ValueError, match="prefix"):
        state.authorize_resume(receipt, prefix)
    assert not state.snapshot()["records"][0]["resumed"]
    state.close()
    state.process()


def test_resume_limit_requires_receipt_and_cannot_change_on_retry():
    with pytest.raises(ValueError, match="prefix"):
        VersionUpdate("session", 2, NEW, resume_prefix_length=0)
    request = AlignmentRequest("session", REVISION, INPUT)
    state = WorkerAlignmentState(lambda _: Mock(num_tokens=3), lambda: None)
    state.observe(NEW, request)
    state.process()
    receipt = state.poll(request)
    state.authorize_resume(receipt, 2)
    state.authorize_resume(receipt, 2)
    for changed in (None, 0, 1, 3):
        with pytest.raises(RuntimeError, match="cannot change"):
            state.authorize_resume(receipt, changed)
    assert state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None) == 2
    assert state.pending_prefix_length(NEW, 10) == 2
    assert state.pending_prefix_length(NEW, 11) == 0
    state.confirm_attachment(NEW, 10, 2)
    assert state.pending_prefix_length(NEW, 10) == 0
    state.close()
    state.process()


@pytest.mark.parametrize("prefix", [0, 3])
def test_pending_recovery_prefix_requires_a_current_claim(prefix):
    request = AlignmentRequest("session", REVISION, INPUT)
    state = WorkerAlignmentState(lambda _: Mock(num_tokens=prefix), lambda: None)
    state.observe(NEW, request)
    state.process()
    receipt = state.poll(request)
    state.authorize_resume(receipt)
    assert state.pending_prefix_length(NEW, 10) == 0
    assert state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None) is None
    assert state.pending_prefix_length(NEW, 10) == prefix
    state.observe(replace(NEW, epoch=NEW.epoch + 1), None)
    assert state.pending_prefix_length(NEW, 10) == 0
    state.close()
    state.process()


@pytest.mark.parametrize("agreed_prefix", [None, 0, 3])
def test_zero_prefix_recovery_distinguishes_a_pending_successor(agreed_prefix):
    request = AlignmentRequest("session", REVISION, INPUT)
    held_prefix = 0 if agreed_prefix is None else agreed_prefix
    state = WorkerAlignmentState(lambda _: Mock(num_tokens=held_prefix), lambda: None)
    state.observe(OLD, None)
    assert state.pending_recovery_prefix(OLD, 10) is None
    state.observe(NEW, request)
    state.process()
    receipt = state.poll(request)
    assert state.pending_recovery_prefix(NEW, 10) is None
    state.authorize_resume(receipt, agreed_prefix)
    assert state.pending_recovery_prefix(NEW, 10) is None
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None)
    assert state.pending_recovery_prefix(NEW, 10) == held_prefix
    assert state.pending_recovery_prefix(NEW, 11) is None
    assert state.pending_prefix_length(NEW, 10) == held_prefix
    state.confirm_attachment(NEW, 10, held_prefix)
    assert state.pending_recovery_prefix(NEW, 10) is None
    assert state.pending_prefix_length(NEW, 10) == 0
    state.observe(replace(NEW, epoch=NEW.epoch + 1), None)
    assert state.pending_recovery_prefix(NEW, 10) is None
    state.close()
    state.process()
    assert state.pending_recovery_prefix(NEW, 10) is None


@pytest.fixture(params=["local", "ipc"])
def setup(request):
    table = WorkerVersionTable("session")
    tracker = table.drain_tracker = RequestDrainTracker()
    reserve = Mock(side_effect=lambda revised: Mock(num_tokens=3))
    wake = Event()
    state = table.alignment_state = WorkerAlignmentState(reserve, wake.set)
    server = ExecutionMailboxServer(table) if request.param == "ipc" else None
    transport = (
        IpcPublicationTransport(server.address) if server else LocalPublicationTransport(table)
    )
    controller = ExecutionController("session", transport)
    controller.register(OLD).wait()
    yield controller, table, tracker, state, reserve, wake
    controller.close()
    if server:
        server.close()
    else:
        table.close()
    # Unknown native completion intentionally retains ownership until restart.
    if state._unsafe_error is None:
        state.process()


def test_resident_owner_hint_requires_a_live_claim_and_does_not_reserve_again():
    request = AlignmentRequest("session", REVISION, INPUT)
    hold = Mock(num_tokens=3, has_resident_prefix=True)
    reserve = Mock(return_value=hold)
    state = WorkerAlignmentState(reserve, lambda: None)
    state.observe(OLD, None)
    assert not state.has_resident_recovery(OLD, 10)
    state.observe(NEW, request)
    assert not state.has_resident_recovery(NEW, 10)
    state.process()
    assert not state.has_resident_recovery(NEW, 10)
    state.authorize_resume(state.poll(request))
    assert not state.has_resident_recovery(NEW, 10)
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None)
    assert state.has_resident_recovery(NEW, 10)
    assert not state.has_resident_recovery(NEW, 11)
    assert not state.has_resident_recovery(OLD, 10)
    hold.has_resident_prefix = False
    assert not state.has_resident_recovery(NEW, 10)
    assert state.pending_recovery_prefix(NEW, 10) == 3
    hold.has_resident_prefix = True
    assert state.has_resident_recovery(NEW, 10)
    reserve.assert_called_once_with(INPUT)
    state.close()
    state.process()


@pytest.mark.parametrize("invalidation", ["attachment", "revision", "terminal", "close"])
def test_resident_owner_hint_ends_when_its_claim_is_invalidated(invalidation):
    request = AlignmentRequest("session", REVISION, INPUT)
    hold = Mock(num_tokens=3, has_resident_prefix=True)
    state = WorkerAlignmentState(lambda _: hold, lambda: None)
    state.observe(NEW, request)
    state.process()
    state.authorize_resume(state.poll(request))
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None)
    assert state.has_resident_recovery(NEW, 10)
    if invalidation == "attachment":
        state.confirm_attachment(NEW, 10, 3)
    elif invalidation == "revision":
        state.observe(replace(NEW, epoch=9), None)
    elif invalidation == "terminal":
        state.terminal(NEW, 10)
    else:
        state.close()
    assert not state.has_resident_recovery(NEW, 10)
    state.close()
    state.process()


def test_resident_owner_hint_is_optional_for_a_valid_prefix_reservation():
    request = AlignmentRequest("session", REVISION, INPUT)
    hold = Mock(spec=["num_tokens", "settle", "close"], num_tokens=3)
    state = WorkerAlignmentState(lambda _: hold, lambda: None)
    state.observe(NEW, request)
    state.process()
    state.authorize_resume(state.poll(request))
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None)
    assert state.pending_recovery_prefix(NEW, 10) == 3
    assert not state.has_resident_recovery(NEW, 10)
    state.confirm_attachment(NEW, 10, 3)
    state.close()
    state.process()


def test_reserve_before_old_release_and_acknowledge_only_after_fence(setup):
    controller, _, tracker, state, reserve, wake = setup
    tracker.enqueue(OLD, lambda: 1)
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    assert wake.wait(1)
    reserve.assert_not_called()  # Mailbox threads do not call native KV APIs.
    with state.releasing_owner():
        reserve.assert_called_once_with(INPUT)
    with pytest.raises(TimeoutError):
        result.alignment.wait(0.02)
    tracker.complete(1, OLD)
    receipt = result.alignment.wait()
    assert receipt.reusable_prefix_length == 3
    assert receipt.request.revised_input == INPUT
    result.fence.wait()
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with controller.admission(NEW):
            pass


def test_idle_alignment_and_retry_are_idempotent(setup):
    controller, _, _, state, reserve, _ = setup
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    state.process()
    first = result.alignment.wait()
    retry = controller.advance(REVISION, alignment=INPUT)
    assert retry.alignment is result.alignment
    state.process()
    assert retry.alignment.wait() == first
    reserve.assert_called_once()


@pytest.mark.parametrize(
    "changed",
    [
        replace(INPUT, input_token_ids=(1, 2, 9)),
        replace(INPUT, cache_salt="other"),
        replace(INPUT, lora_task_id=2),
        replace(INPUT, min_reuse_tokens=2),
        None,
    ],
)
def test_retry_cannot_change_revised_input(setup, changed):
    controller, _, _, _, _, _ = setup
    controller.advance(REVISION, alignment=INPUT)
    with pytest.raises(ValueError, match="revised input"):
        controller.advance(REVISION, alignment=changed)
    assert controller.snapshot("tenant", "scope").current_version == NEW


def test_advance_retry_cannot_add_alignment(setup):
    controller, _, _, _, _, _ = setup
    controller.advance(REVISION)
    with pytest.raises(ValueError, match="revised input"):
        controller.advance(REVISION, alignment=INPUT)


@pytest.mark.parametrize("remaining", [0, 2, 3])
def test_separate_align_reserves_only_what_remains_after_fence(setup, remaining):
    controller, table, tracker, state, reserve, _ = setup
    tracker.enqueue(OLD, lambda: 1)
    advance = controller.advance(REVISION)
    advance.publication.wait()
    with state.releasing_owner():
        reserve.assert_not_called()
    tracker.complete(1, OLD)
    advance.fence.wait()
    with state.resetting_cache():
        reserve.side_effect = lambda _: Mock(num_tokens=remaining)
    assert state.snapshot()["held_count"] == 0
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with controller.admission(NEW):
            pass
    with pytest.raises(ExecutionAuthorityError):
        with controller.admission(OLD):
            pass

    aligned = controller.align(REVISION, INPUT)
    aligned.publication.wait()
    assert aligned.advance_receipt == advance.advance_receipt
    assert aligned.fence is advance.fence
    assert aligned.publication.update.sequence > advance.publication.update.sequence
    reserve.assert_not_called()
    state.process()
    receipt = aligned.alignment.wait()
    assert receipt.reusable_prefix_length == remaining
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with table.admission(NEW):
            pass
    controller.resume(receipt).wait()
    with controller.admission(NEW):
        pass
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, None)
    state.confirm_attachment(NEW, 10, remaining)
    assert state.snapshot()["held_count"] == 0
    retry = controller.align(REVISION, INPUT)
    assert retry.alignment is aligned.alignment
    assert retry.publication is controller.resume(receipt)
    state.process()
    reserve.assert_called_once_with(INPUT)
    # Historical ADVANCE retries retain their original arguments and never
    # erase the separately published ALIGN plan.
    controller.advance(REVISION).publication.wait()
    assert controller.align(REVISION, INPUT).alignment.wait() == receipt
    with pytest.raises(ValueError, match="revised input"):
        controller.advance(REVISION, alignment=INPUT)


def test_separate_align_still_waits_for_old_gpu_users(setup):
    controller, _, tracker, state, _, _ = setup
    tracker.enqueue(OLD, lambda: 1)
    controller.advance(REVISION).publication.wait()
    aligned = controller.align(REVISION, INPUT)
    aligned.publication.wait()
    state.process()
    with pytest.raises(TimeoutError):
        aligned.alignment.wait(0.02)
    with pytest.raises(ValueError, match="completed ALIGN"):
        controller.resume(state.poll(aligned.alignment.request))
    tracker.complete(1, OLD)
    controller.resume(aligned.alignment.wait()).wait()
    with controller.admission(NEW):
        pass


def test_separate_align_requires_accepted_current_revision(setup):
    controller, _, _, _, reserve, _ = setup
    with pytest.raises(ValueError, match="accepted"):
        controller.align(REVISION, INPUT)
    controller.advance(REVISION).publication.wait()
    with pytest.raises(ValueError, match="accepted"):
        controller.align(replace(REVISION, revision_id="unknown"), INPUT)
    controller.advance(Revision("next", NEW, replace(NEW, epoch=9))).publication.wait()
    with pytest.raises(ValueError, match="superseded"):
        controller.align(REVISION, INPUT)
    reserve.assert_not_called()


@pytest.mark.parametrize("immediate", [False, True])
@pytest.mark.parametrize(
    "changed",
    [
        replace(INPUT, input_token_ids=(1, 2, 9)),
        replace(INPUT, cache_salt="other"),
        replace(INPUT, lora_task_id=2),
        replace(INPUT, min_reuse_tokens=2),
    ],
)
def test_separate_align_cannot_replace_bound_input(setup, immediate, changed):
    controller, _, _, state, reserve, _ = setup
    controller.advance(REVISION, alignment=INPUT if immediate else None).publication.wait()
    aligned = controller.align(REVISION, INPUT)
    aligned.publication.wait()
    with pytest.raises(ValueError, match="revised input"):
        controller.align(REVISION, changed)
    state.process()
    assert aligned.alignment.wait().request.revised_input == INPUT
    reserve.assert_called_once_with(INPUT)


def test_separate_align_recovers_lost_ack_without_duplicate_hold(setup):
    controller, _, _, state, reserve, _ = setup
    advance = controller.advance(REVISION)
    advance.publication.wait()
    advance.fence.wait()
    publish = controller._transport.publish
    failed = False

    def lose_ack(update, stop):
        nonlocal failed
        receipt = publish(update, stop)
        if update.alignment is not None and not failed:
            failed = True
            raise TimeoutError("lost ALIGN ack")
        return receipt

    controller._transport.publish = lose_ack
    first = controller.align(REVISION, INPUT)
    with pytest.raises(TimeoutError, match="lost ALIGN ack"):
        first.publication.wait()
    state.process()
    receipt = first.alignment.wait()
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with controller.admission(NEW):
            pass
    retry = controller.align(REVISION, INPUT)
    assert retry.publication.update == first.publication.update
    retry.publication.wait()
    assert retry.alignment.wait() == receipt
    controller.resume(receipt).wait()
    reserve.assert_called_once_with(INPUT)


def test_timeout_keeps_hold_and_supersession_releases_it_on_engine(setup):
    controller, _, tracker, state, reserve, _ = setup
    hold = Mock(num_tokens=3)
    reserve.side_effect = lambda revised: hold
    tracker.enqueue(OLD, lambda: 1)
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    state.process()
    with pytest.raises(TimeoutError):
        result.alignment.wait(0.02)
    hold.close.assert_not_called()
    controller.advance(Revision("second", NEW, replace(NEW, epoch=9))).publication.wait()
    with pytest.raises(RuntimeError, match="superseded"):
        result.alignment.wait()
    hold.close.assert_not_called()
    state.process()
    hold.close.assert_called_once()


def test_successive_alignment_acquires_before_releasing_previous_hold(setup):
    controller, _, _, state, reserve, _ = setup
    first_hold = Mock(num_tokens=3)
    order = []
    first_hold.close.side_effect = lambda: order.append("close old")
    reserve.side_effect = lambda revised: first_hold
    first = controller.advance(REVISION, alignment=INPUT)
    first.publication.wait()
    state.process()
    first.alignment.wait()

    def acquire(revised):
        order.append("acquire new")
        return Mock(num_tokens=2)

    reserve.side_effect = acquire
    second = controller.advance(Revision("second", NEW, replace(NEW, epoch=9)), alignment=INPUT)
    second.publication.wait()
    state.process()
    second.alignment.wait()
    assert order == ["acquire new", "close old"]


def test_required_prefix_failure_closes_partial_hold(setup):
    controller, _, _, state, reserve, _ = setup
    hold = Mock(num_tokens=1)
    reserve.side_effect = lambda revised: hold
    result = controller.advance(REVISION, alignment=replace(INPUT, min_reuse_tokens=3))
    result.publication.wait()
    state.process()
    with pytest.raises(RuntimeError, match="prefix"):
        result.alignment.wait()
    hold.close.assert_called_once()
    result.fence.wait()


def test_cache_miss_can_align_for_recompute(setup):
    controller, _, _, state, reserve, _ = setup
    reserve.side_effect = lambda revised: Mock(num_tokens=0)
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    state.process()
    assert result.alignment.wait().reusable_prefix_length == 0


def test_shutdown_never_releases_native_hold_on_mailbox_thread(setup):
    controller, table, _, state, reserve, _ = setup
    hold = Mock(num_tokens=3)
    releases = []
    hold.close.side_effect = lambda: releases.append(get_ident())
    reserve.side_effect = lambda revised: hold
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    state.process()
    result.alignment.wait()
    table.close()
    hold.close.assert_not_called()
    state.process()
    assert releases == [get_ident()]


def test_unknown_native_cleanup_blocks_resource_release():
    reserve = Mock(side_effect=AlignmentSafetyError("unknown copy completion"))
    state = WorkerAlignmentState(reserve, lambda: None)
    state.observe(NEW, AlignmentRequest("session", REVISION, INPUT))
    for _ in range(2):
        with pytest.raises(AlignmentSafetyError):
            with state.releasing_owner():
                pytest.fail("cannot release old memory without copy completion")
    reserve.assert_called_once()


def test_capacity_allows_replacing_same_scope_but_rejects_another():
    state = WorkerAlignmentState(
        lambda revised: Mock(num_tokens=2), lambda: None, max_reservations=1
    )
    first = AlignmentRequest("session", REVISION, INPUT)
    state.observe(NEW, first)
    state.process()
    other_old = replace(OLD, scope_id="other")
    other_revision = Revision("other", other_old, replace(other_old, epoch=8))
    other = AlignmentRequest("session", other_revision, INPUT)
    state.observe(other_revision.successor_version, other)
    with pytest.raises(RuntimeError, match="capacity"):
        state.poll(other)
    second_revision = Revision("again", NEW, replace(NEW, epoch=9))
    second = AlignmentRequest("session", second_revision, INPUT)
    state.observe(second_revision.successor_version, second)
    state.process()
    assert state.poll(second).reusable_prefix_length == 2
    state.close()
    state.process()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_token_ids": []},
        {"input_token_ids": ()},
        {"input_token_ids": (True,)},
        {"input_token_ids": (-1,)},
        {"input_token_ids": (2**31,)},
        {"input_token_ids": (1,), "min_reuse_tokens": 1},
        {"input_token_ids": (1,), "cache_salt": 2},
        {"input_token_ids": (1,), "lora_task_id": True},
    ],
)
def test_invalid_input_rejected_before_revision(kwargs):
    with pytest.raises((TypeError, ValueError)):
        AlignmentInput(**kwargs)


def test_cache_reset_rejected_while_pending_or_reserved(setup):
    controller, _, _, state, _, _ = setup
    with state.resetting_cache():
        pass
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    with pytest.raises(RuntimeError, match="reset"):
        with state.resetting_cache():
            pass
    state.process()
    result.alignment.wait()
    with pytest.raises(RuntimeError, match="reset"):
        with state.resetting_cache():
            pass
    state.close()
    state.process()
    with state.resetting_cache():
        pass


def test_invalidated_alignment_does_not_prevent_further_revocation(setup):
    controller, table, _, state, _, _ = setup
    state.close()
    state.process()
    result = controller.advance(REVISION, alignment=INPUT)
    result.publication.wait()
    assert table.current_version("tenant", "scope") == NEW
    result.fence.wait()
    with pytest.raises(RuntimeError, match="closed"):
        result.alignment.wait()


def _aligned(setup):
    controller, _, _, state, _, _ = setup
    publication = controller.advance(REVISION, alignment=INPUT)
    publication.publication.wait()
    state.process()
    return publication.alignment.wait()


def test_resume_opens_admission_only_after_worker_accepts(setup):
    controller, table, _, state, _, _ = setup
    aligned = _aligned(setup)
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with table.admission(NEW):
            pass
    with pytest.raises(RuntimeError, match="pending RESUME"):
        state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    resumed = controller.resume(aligned)
    resumed.wait()
    assert controller.snapshot("tenant", "scope").pending_revision is None
    with controller.admission(NEW):
        pass
    with table.admission(NEW):
        pass
    assert controller.resume(aligned) is resumed
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    state.confirm_attachment(NEW, 10, 3)
    # Recompute of an already attached request may subsequently have a cache miss.
    state.confirm_attachment(NEW, 10, 0)


@pytest.mark.parametrize(
    "field,value",
    [("input_token_ids", (1, 2, 3, 4, 9)), ("cache_salt", "other"), ("lora_task_id", 2)],
)
def test_resume_rejects_changed_input_without_consuming_claim(setup, field, value):
    controller, _, _, state, _, _ = setup
    controller.resume(_aligned(setup)).wait()
    changed = replace(INPUT, **{field: value})
    with pytest.raises(ValueError, match="input does not match"):
        state.claim(NEW, 10, changed.input_token_ids, changed.cache_salt, changed.lora_task_id)
    state.claim(NEW, 11, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    with pytest.raises(RuntimeError, match="another request"):
        state.claim(NEW, 12, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)


def test_resume_requires_native_prefix_attachment_before_reclaim(setup):
    controller, _, _, state, reserve, _ = setup
    controller.resume(_aligned(setup)).wait()
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    with pytest.raises(RuntimeError, match="aligned KV prefix"):
        state.confirm_attachment(NEW, 10, 2)
    with pytest.raises(RuntimeError, match="does not own"):
        state.confirm_attachment(NEW, 11, 3)
    record = next(iter(state._records.values()))
    hold = record.reservation
    state.confirm_attachment(NEW, 10, 3)
    assert record.attached and record.reservation is None
    assert [call[0] for call in hold.mock_calls] == ["settle", "close"]
    state.confirm_attachment(NEW, 10, 3)
    hold.close.assert_called_once()
    reserve.assert_called_once()


def test_resume_rejects_forged_or_superseded_receipt(setup):
    controller, _, _, state, _, _ = setup
    aligned = _aligned(setup)
    with pytest.raises(ValueError, match="completed ALIGN"):
        controller.resume(replace(aligned, reservation_id="wrong"))
    newer = Revision("next", NEW, replace(NEW, epoch=9))
    controller.advance(newer).publication.wait()
    with pytest.raises(ValueError, match="obsolete"):
        controller.resume(aligned)
    with pytest.raises(RuntimeError, match="no longer current"):
        state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)


def test_resume_worker_rechecks_reservation_after_control_invalidation(setup):
    controller, _, _, state, _, _ = setup
    aligned = _aligned(setup)
    state.close()
    state.process()
    with pytest.raises(RuntimeError, match="closed"):
        controller.resume(aligned).wait()
    assert controller.snapshot("tenant", "scope").pending_revision == REVISION
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with controller.admission(NEW):
            pass


def test_resume_retry_after_lost_ack_does_not_duplicate_ownership(setup):
    controller, table, _, state, reserve, _ = setup
    aligned = _aligned(setup)
    publish = controller._transport.publish
    failed = False

    def lose_ack(update, stop):
        nonlocal failed
        receipt = publish(update, stop)
        if update.resume is not None and not failed:
            failed = True
            raise TimeoutError("lost RESUME ack")
        return receipt

    controller._transport.publish = lose_ack
    with pytest.raises(TimeoutError, match="lost RESUME"):
        controller.resume(aligned).wait()
    assert controller.snapshot("tenant", "scope").pending_revision == REVISION
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with controller.admission(NEW):
            pass
    controller.resume(aligned).wait()
    assert controller.snapshot("tenant", "scope").pending_revision is None
    reserve.assert_called_once()
    assert next(iter(state._records.values())).resumed
    with table.admission(NEW):
        pass


def test_new_revision_during_resume_ack_keeps_new_authority_pending(setup):
    controller, _, _, _, _, _ = setup
    aligned = _aligned(setup)
    accepted, release = Event(), Event()
    publish = controller._transport.publish

    def delay_ack(update, stop):
        receipt = publish(update, stop)
        if update.resume is not None:
            accepted.set()
            assert release.wait(3)
        return receipt

    controller._transport.publish = delay_ack
    resumed = controller.resume(aligned)
    try:
        assert accepted.wait(2)
        newer = Revision("next", NEW, replace(NEW, epoch=9))
        advanced = controller.advance(newer)
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="superseded"):
        resumed.wait()
    advanced.publication.wait()
    assert controller.snapshot("tenant", "scope").pending_revision == newer
    with pytest.raises(ExecutionAuthorityError):
        with controller.admission(NEW):
            pass


def test_zero_prefix_can_resume_for_ordinary_recomputation(setup):
    controller, _, _, state, reserve, _ = setup
    reserve.side_effect = lambda _: Mock(num_tokens=0)
    controller.resume(_aligned(setup)).wait()
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    state.confirm_attachment(NEW, 10, 0)


@pytest.mark.parametrize("failure", ["settle", "close"])
def test_reclaim_failure_retains_ownership_and_blocks_further_cleanup(setup, failure):
    controller, _, _, state, _, _ = setup
    controller.resume(_aligned(setup)).wait()
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    record = next(iter(state._records.values()))
    hold = record.reservation
    getattr(hold, failure).side_effect = RuntimeError("device failure")
    with pytest.raises(AlignmentSafetyError, match="RECLAIM"):
        state.confirm_attachment(NEW, 10, 3)
    assert record.reservation is hold
    if failure == "settle":
        hold.close.assert_not_called()
    with pytest.raises(AlignmentSafetyError):
        with state.releasing_owner():
            pytest.fail("must not release more resources after unknown completion")


def test_terminal_claim_reclaimed_by_engine_and_cannot_be_replayed(setup):
    controller, _, _, state, _, _ = setup
    aligned = _aligned(setup)
    controller.resume(aligned).wait()
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    record = next(iter(state._records.values()))
    hold = record.reservation
    state.terminal(NEW, 99)
    assert record.error is None
    state.terminal(NEW, 10)
    hold.close.assert_not_called()
    state.process()
    hold.close.assert_called_once()
    assert record.reservation is None
    with pytest.raises(RuntimeError):
        state.authorize_resume(aligned)
    with pytest.raises(RuntimeError):
        state.claim(NEW, 11, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    state.process()
    hold.close.assert_called_once()


def test_reclaimed_reservation_frees_capacity_without_reacquiring_on_retry(setup):
    controller, _, _, state, reserve, _ = setup
    state._max_reservations = 1
    aligned = _aligned(setup)
    controller.resume(aligned).wait()
    state.claim(NEW, 10, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)
    state.confirm_attachment(NEW, 10, 3)
    state.process()
    controller.advance(REVISION, alignment=INPUT).publication.wait()
    state.process()
    reserve.assert_called_once()
    with state.resetting_cache():
        pass
    other = replace(OLD, scope_id="other")
    controller.register(other).wait()
    revision = Revision("other-edit", other, replace(other, epoch=8))
    pending = controller.advance(revision, alignment=INPUT)
    pending.publication.wait()
    state.process()
    assert pending.alignment.wait().reusable_prefix_length == 3
    assert reserve.call_count == 2


def test_cancel_before_activation_reclaims_unclaimed_resumed_hold(setup):
    controller, _, _, state, _, _ = setup
    controller.resume(_aligned(setup)).wait()
    record = next(iter(state._records.values()))
    hold = record.reservation
    assert record.request_id is None
    state.terminal(NEW, 10)
    hold.close.assert_not_called()
    state.process()
    hold.settle.assert_called_once()
    hold.close.assert_called_once()
    assert record.reservation is None
    with pytest.raises(RuntimeError):
        state.claim(NEW, 11, INPUT.input_token_ids, INPUT.cache_salt, INPUT.lora_task_id)


def test_snapshot_tracks_ownership_until_engine_reclaims_abandoned_hold(setup):
    controller, _, _, state, _, _ = setup
    assert state.snapshot()["held_count"] == 0
    aligned = _aligned(setup)
    held = state.snapshot()
    assert held["held_count"] == 1 and held["promised_tokens_held"] == 3
    assert held["unknown_prefix_count"] == 0
    assert held["records"][0]["epoch"] == NEW.epoch
    assert held["records"][0]["current"]
    controller.advance(Revision("abandon", NEW, replace(NEW, epoch=9))).publication.wait()
    obsolete = state.snapshot()
    assert obsolete["held_count"] == 1
    assert not obsolete["records"][0]["current"]
    # Neither observing nor revoking authority proves native cleanup.
    state.process()
    assert state.snapshot()["held_count"] == 0
    assert held["records"][0]["current"]  # Earlier snapshots are detached copies.
    with pytest.raises(ValueError, match="obsolete"):
        controller.resume(aligned)


@pytest.mark.parametrize("failure", ["settle", "close"])
def test_snapshot_keeps_failed_native_cleanup_visible(setup, failure):
    controller, _, _, state, _, _ = setup
    controller.resume(_aligned(setup)).wait()
    record = next(iter(state._records.values()))
    getattr(record.reservation, failure).side_effect = RuntimeError("native cleanup failed")
    state.terminal(NEW, 10)
    with pytest.raises(AlignmentSafetyError):
        state.process()
    snapshot = state.snapshot()
    assert snapshot["unsafe"] and snapshot["held_count"] == 1
    assert snapshot["records"][0]["failed"]
    assert snapshot["promised_tokens_held"] == 3


def test_snapshot_reports_unknown_partial_hold_without_native_access():
    class Hold:
        reads = 0

        @property
        def num_tokens(self):
            self.reads += 1
            raise AlignmentSafetyError("prefix length unavailable")

    hold = Hold()
    state = WorkerAlignmentState(lambda _: hold, lambda: None)
    state.observe(NEW, AlignmentRequest("session", REVISION, INPUT))
    with pytest.raises(AlignmentSafetyError):
        state.process()
    snapshot = state.snapshot()
    assert hold.reads == 1
    assert snapshot["unsafe"] and snapshot["held_count"] == 1
    assert snapshot["unknown_prefix_count"] == 1
    assert snapshot["promised_tokens_held"] == 0
    assert snapshot["records"][0]["promised_tokens"] is None
