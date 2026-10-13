# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FENCE must prove a scope drain independently of publication and output closure."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_fence import FenceRequest, RequestDrainTracker
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


@pytest.fixture(params=["local", "ipc"])
def control(request):
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    server = ExecutionMailboxServer(table) if request.param == "ipc" else None
    transport = (
        IpcPublicationTransport(server.address)
        if server is not None
        else LocalPublicationTransport(table)
    )
    controller = ExecutionController("session", transport)
    controller.register(OLD).wait()
    yield controller, table, table.drain_tracker
    controller.close()
    if server is not None:
        server.close()
    else:
        table.close()


def test_publication_is_not_fence_and_timeout_does_not_cancel(control):
    controller, _, tracker = control
    tracker.enqueue(OLD, lambda: 1)
    tracker.enqueue(OLD, lambda: 2)
    result = controller.advance(REVISION)
    result.publication.wait()
    with pytest.raises(TimeoutError):
        result.fence.wait(0.03)
    tracker.complete(1, OLD)
    with pytest.raises(TimeoutError):
        result.fence.wait(0.03)
    tracker.complete(2, OLD)
    assert result.fence.wait().request.revision == REVISION
    with pytest.raises(ExecutionAuthorityError, match="pending"):
        with controller.admission(NEW):
            pass


def test_empty_scope_and_retry_return_same_historical_fence(control):
    controller, _, _ = control
    first = controller.advance(REVISION)
    receipt = first.fence.wait()
    controller.advance(Revision("again", NEW, replace(NEW, epoch=9))).fence.wait()
    retry = controller.advance(REVISION)
    assert retry.fence is first.fence
    assert retry.fence.wait() == receipt
    assert retry.publication.wait().observed_version.epoch == 9


def test_chained_revision_waits_for_earlier_epochs(control):
    controller, _, tracker = control
    tracker.enqueue(OLD, lambda: 1)
    first = controller.advance(REVISION)
    second = controller.advance(Revision("again", NEW, replace(NEW, epoch=9)))
    second.publication.wait()
    with pytest.raises(TimeoutError):
        second.fence.wait(0.03)
    tracker.complete(1, OLD)
    first.fence.wait()
    second.fence.wait()


def test_other_scopes_and_tenants_do_not_block_fence(control):
    controller, _, tracker = control
    tracker.enqueue(replace(OLD, scope_id="other"), lambda: 1)
    tracker.enqueue(replace(OLD, tenant_id="other"), lambda: 2)
    controller.advance(REVISION).fence.wait()


def test_close_fails_pending_fence(control):
    controller, _, tracker = control
    tracker.enqueue(OLD, lambda: 1)
    result = controller.advance(REVISION)
    result.publication.wait()
    controller.close()
    with pytest.raises(RuntimeError, match="closed"):
        result.fence.wait()


def test_worker_close_never_certifies_drain(control):
    controller, table, tracker = control
    tracker.enqueue(OLD, lambda: 1)
    result = controller.advance(REVISION)
    result.publication.wait()
    table.close()
    with pytest.raises(RuntimeError, match="closed"):
        result.fence.wait()


def test_failed_enqueue_fails_fence_closed(control):
    controller, _, tracker = control

    def failing_enqueue():
        raise RuntimeError("enqueue completion unknown")

    with pytest.raises(RuntimeError):
        tracker.enqueue(OLD, failing_enqueue)
    result = controller.advance(REVISION)
    result.publication.wait()
    with pytest.raises(RuntimeError, match="unknown"):
        result.fence.wait()


def test_fence_rejects_wrong_session_and_requires_observation():
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    query = FenceRequest("session", REVISION)
    assert table.poll_fence(query) is None
    table.apply(VersionUpdate("session", 1, OLD))
    assert table.poll_fence(query) is None
    table.apply(VersionUpdate("session", 2, NEW))
    with pytest.raises(ValueError, match="session"):
        table.poll_fence(FenceRequest("different", REVISION))
    assert table.poll_fence(query).request == query


def test_missing_engine_tracker_cannot_certify_empty_scope():
    table = WorkerVersionTable("session")
    table.apply(VersionUpdate("session", 1, NEW))
    with pytest.raises(RuntimeError, match="tracking"):
        table.poll_fence(FenceRequest("session", REVISION))


def test_completion_cannot_overtake_enqueue_registration():
    tracker = RequestDrainTracker()
    entered, release, completing = Event(), Event(), Event()

    def submit():
        entered.set()
        assert release.wait(5)
        return 1

    def finish():
        completing.set()
        tracker.complete(1, OLD)

    with ThreadPoolExecutor(2) as pool:
        enqueue = pool.submit(tracker.enqueue, OLD, submit)
        assert entered.wait(5)
        completion = pool.submit(finish)
        assert completing.wait(5)
        assert not completion.done()
        release.set()
        assert enqueue.result(5) == 1
        completion.result(5)
    assert tracker.drained(OLD)


def test_changed_identity_and_id_reuse_fail_closed():
    tracker = RequestDrainTracker()
    tracker.enqueue(OLD, lambda: 1)
    with pytest.raises(RuntimeError, match="identity"):
        tracker.complete(1, NEW)
    with pytest.raises(RuntimeError, match="unknown"):
        tracker.drained(OLD)
    tracker = RequestDrainTracker()
    tracker.enqueue(OLD, lambda: 1)
    with pytest.raises(RuntimeError, match="reused"):
        tracker.enqueue(NEW, lambda: 1)
    with pytest.raises(RuntimeError, match="unknown"):
        tracker.drained(NEW)


def test_child_sequences_outlive_parent_and_block_fence(control):
    controller, _, tracker = control
    tracker.enqueue(OLD, lambda: 1)
    tracker.add_child(1, 2, OLD)
    tracker.add_child(1, 3, OLD)
    tracker.complete(1, OLD)
    result = controller.advance(REVISION)
    result.publication.wait()
    with pytest.raises(TimeoutError):
        result.fence.wait(0.03)
    tracker.complete(2, OLD)
    with pytest.raises(TimeoutError):
        result.fence.wait(0.03)
    tracker.complete(3, OLD)
    result.fence.wait()


def test_child_without_pending_parent_fails_closed():
    tracker = RequestDrainTracker()
    with pytest.raises(RuntimeError, match="parent"):
        tracker.add_child(1, 2, OLD)
    with pytest.raises(RuntimeError, match="unknown"):
        tracker.drained(OLD)


def test_busy_scope_does_not_block_another_fence(control):
    controller, _, tracker = control
    tracker.enqueue(OLD, lambda: 1)
    busy = controller.advance(REVISION)
    other = replace(OLD, scope_id="other")
    controller.register(other).wait()
    controller.advance(Revision("other", other, replace(other, epoch=8))).fence.wait()
    with pytest.raises(TimeoutError):
        busy.fence.wait(0.01)


def test_failed_status_exchange_can_retry_without_another_advance():
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()

    class Transport(LocalPublicationTransport):
        fail = True

        def poll_fence(self, request, stop):
            if self.fail:
                raise RuntimeError("status exchange failed")
            return super().poll_fence(request, stop)

    transport = Transport(table)
    controller = ExecutionController("session", transport)
    try:
        controller.register(OLD).wait()
        failed = controller.advance(REVISION)
        failed.publication.wait()
        with pytest.raises(RuntimeError, match="exchange"):
            failed.fence.wait()
        transport.fail = False
        retry = controller.advance(REVISION)
        assert retry.advance_receipt == failed.advance_receipt
        assert retry.fence.wait().request == failed.fence.request
    finally:
        controller.close()
        table.close()


def test_fence_capacity_failure_does_not_undo_revision():
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    table.drain_tracker.enqueue(OLD, lambda: 1)
    controller = ExecutionController("session", LocalPublicationTransport(table), max_pending=1)
    try:
        controller.register(OLD).wait()
        first = controller.advance(REVISION)
        first.publication.wait()
        second_revision = Revision("again", NEW, replace(NEW, epoch=9))
        second = controller.advance(second_revision)
        second.publication.wait()
        with pytest.raises(RuntimeError, match="backlog"):
            second.fence.wait()
        assert controller.snapshot("tenant", "scope").current_version.epoch == 9
        table.drain_tracker.complete(1, OLD)
        first.fence.wait()
        controller.advance(second_revision).fence.wait()
    finally:
        controller.close()
        table.close()
