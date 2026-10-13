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
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
REVISION = Revision("edit", OLD, NEW)
INPUT = AlignmentInput((1, 2, 3, 4, 5), cache_salt="shared")


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
    # Native ownership belongs to the engine thread, even after mailbox closure.
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


def test_alignment_cannot_be_added_after_advance(setup):
    controller, _, _, _, _, _ = setup
    controller.advance(REVISION)
    with pytest.raises(ValueError, match="revised input"):
        controller.advance(REVISION, alignment=INPUT)


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
