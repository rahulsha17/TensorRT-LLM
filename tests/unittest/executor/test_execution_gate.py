# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host effects must be ordered with ADVANCE, including lost worker publications."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest

from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    VersionUpdate,
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_control import ExecutionControlMixin
from tensorrt_llm.executor.execution_mailbox import ExecutionMailboxServer, IpcPublicationTransport

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
NEXT = replace(VERSION, epoch=8)
REVISION = Revision("edit", VERSION, NEXT)


@pytest.fixture
def authority():
    table = WorkerVersionTable("session")
    controller = ExecutionController("session", LocalPublicationTransport(table))
    controller.register(VERSION).wait()
    try:
        yield controller, table
    finally:
        controller.close()
        table.close()


def admit(owner, version=VERSION):
    queue = Queue()
    with owner.admission(version) as execution:
        execution._wake = queue.put
    return execution, queue


def test_advance_wakes_all_requests_in_scope_and_rejects_output(authority) -> None:
    controller, _ = authority
    requests = [admit(controller) for _ in range(3)]
    controller.advance(REVISION).publication.wait()
    emitted = []
    for execution, queue in requests:
        assert isinstance(queue.get(timeout=1), ExecutionAuthorityError)
        with pytest.raises(ExecutionAuthorityError, match="superseded"):
            execution.commit(lambda: emitted.append("stale token"))
    assert emitted == []


@pytest.mark.parametrize("owner_index", [0, 1])
def test_enqueued_work_is_cancelled_once_when_authority_changes(authority, owner_index) -> None:
    controller, table = authority
    owner = (controller, table)[owner_index]
    cancelled = []
    with owner.admission(VERSION) as execution:
        execution._cancel = lambda: cancelled.append(41)
    controller.advance(REVISION).publication.wait()
    controller.advance(REVISION).publication.wait()
    assert cancelled == [41]
    with pytest.raises(ExecutionAuthorityError):
        execution.check()


@pytest.mark.parametrize(
    "version", [replace(VERSION, epoch=6), NEXT, replace(VERSION, scope_id="?")]
)
@pytest.mark.parametrize("owner_index", [0, 1])
def test_stale_future_and_unknown_submissions_are_rejected(authority, version, owner_index) -> None:
    entered = []
    with pytest.raises(ExecutionAuthorityError):
        with authority[owner_index].admission(version):
            entered.append(True)
    assert entered == []


def test_pending_successor_cannot_start_even_after_publication(authority) -> None:
    controller, _ = authority
    controller.advance(REVISION).publication.wait()
    with pytest.raises(ExecutionAuthorityError, match="ALIGN and RESUME"):
        with controller.admission(NEXT):
            pytest.fail("pending successor was admitted")


@pytest.mark.parametrize(
    "other", [replace(VERSION, scope_id="other"), replace(VERSION, tenant_id="other")]
)
def test_revision_does_not_revoke_other_scopes_or_tenants(authority, other) -> None:
    controller, _ = authority
    controller.register(other).wait()
    execution, queue = admit(controller, other)
    controller.advance(REVISION).publication.wait()
    execution.commit(lambda: queue.put("kept"))
    assert queue.get(timeout=1) == "kept"


def test_final_response_detaches_cancel_but_buffered_read_still_fails(authority) -> None:
    controller, _ = authority
    cancelled = []
    with controller.admission(VERSION) as execution:
        execution._cancel = lambda: cancelled.append(True)
    execution.finish()
    controller.advance(REVISION).publication.wait()
    assert cancelled == []
    with pytest.raises(ExecutionAuthorityError):
        execution.check()


def test_old_retry_cannot_restore_output_permission(authority) -> None:
    controller, _ = authority
    execution, _ = admit(controller)
    original = controller.advance(REVISION)
    original.publication.wait()
    controller.advance(Revision("edit-again", NEXT, replace(NEXT, epoch=9))).publication.wait()
    assert controller.advance(REVISION).advance_receipt == original.advance_receipt
    with pytest.raises(ExecutionAuthorityError):
        execution.check()


@pytest.mark.parametrize("owner_index", [0, 1])
def test_shutdown_unblocks_consumers_and_rejects_admission(authority, owner_index) -> None:
    owner = authority[owner_index]
    execution, queue = admit(owner)
    owner.close()
    assert isinstance(queue.get(timeout=1), ExecutionAuthorityError)
    with pytest.raises(ExecutionAuthorityError, match="closed"):
        execution.check()
    with pytest.raises(ExecutionAuthorityError, match="closed"):
        with owner.admission(VERSION):
            pytest.fail("closed owner admitted work")


def test_advance_waits_for_in_progress_host_commit(authority) -> None:
    controller, _ = authority
    execution, _ = admit(controller)
    in_commit, release, advancing = Event(), Event(), Event()
    order = []

    def commit() -> None:
        def action() -> None:
            in_commit.set()
            assert release.wait(5)
            order.append("published")

        execution.commit(action)

    def advance() -> None:
        advancing.set()
        controller.advance(REVISION)
        order.append("advanced")

    with ThreadPoolExecutor(2) as pool:
        committed = pool.submit(commit)
        try:
            assert in_commit.wait(5)
            advanced = pool.submit(advance)
            assert advancing.wait(5)
            assert not advanced.done()
        finally:
            release.set()
        committed.result(5)
        advanced.result(5)
    assert order == ["published", "advanced"]
    with pytest.raises(ExecutionAuthorityError):
        execution.commit(lambda: order.append("late"))


def test_worker_update_waits_for_native_enqueue_then_cancels(authority) -> None:
    _, table = authority
    entered, release, updating = Event(), Event(), Event()
    order = []

    def enqueue() -> None:
        with table.admission(VERSION) as execution:
            entered.set()
            assert release.wait(5)
            order.append("enqueued")
            execution._cancel = lambda: order.append("cancelled")
        return execution

    def update() -> None:
        updating.set()
        table.apply(VersionUpdate("session", 2, NEXT))
        order.append("observed")

    with ThreadPoolExecutor(2) as pool:
        enqueued = pool.submit(enqueue)
        try:
            assert entered.wait(5)
            observed = pool.submit(update)
            assert updating.wait(5)
            assert not observed.done()
        finally:
            release.set()
        execution = enqueued.result(5)
        observed.result(5)
    assert order == ["enqueued", "cancelled", "observed"]
    with pytest.raises(ExecutionAuthorityError):
        execution.check()


def test_lost_publication_does_not_allow_frontend_output(authority) -> None:
    controller, table = authority
    execution, queue = admit(controller)
    publication_started, release = Event(), Event()

    class LostPublication:
        def publish(self, update, stop):
            publication_started.set()
            assert release.wait(5)
            raise TimeoutError("lost publication")

        def close(self):
            pass

    controller._transport = LostPublication()
    try:
        result = controller.advance(REVISION)
        assert publication_started.wait(5)
        assert table.current_version("tenant", "scope") == VERSION
        assert isinstance(queue.get(timeout=1), ExecutionAuthorityError)
        with pytest.raises(ExecutionAuthorityError):
            execution.check()
    finally:
        release.set()
    with pytest.raises(TimeoutError):
        result.publication.wait()


def test_initial_work_requires_successful_publication() -> None:
    table = WorkerVersionTable("session")
    entered, release = Event(), Event()

    class SlowTransport(LocalPublicationTransport):
        def publish(self, update, stop):
            entered.set()
            assert release.wait(5)
            return super().publish(update, stop)

    controller = ExecutionController("session", SlowTransport(table))
    try:
        publication = controller.register(VERSION)
        assert entered.wait(5)
        with pytest.raises(ExecutionAuthorityError, match="not been published"):
            with controller.admission(VERSION):
                pytest.fail("unpublished authority admitted work")
        release.set()
        publication.wait()
        execution, _ = admit(controller)
        execution.check()
    finally:
        release.set()
        controller.close()
        table.close()


def test_cancellation_failure_does_not_rollback_or_skip_other_requests(authority) -> None:
    controller, _ = authority

    def fail_cancel() -> None:
        raise RuntimeError("engine shutting down")

    with controller.admission(VERSION) as bad:
        bad._cancel = fail_cancel
    good, queue = admit(controller)
    controller.advance(REVISION).publication.wait()
    assert controller.snapshot("tenant", "scope").current_version == NEXT
    assert isinstance(queue.get(timeout=1), ExecutionAuthorityError)
    for execution in (bad, good):
        with pytest.raises(ExecutionAuthorityError):
            execution.check()


def test_untagged_submission_remains_unmanaged_and_tagged_requires_setup() -> None:
    executor = ExecutionControlMixin()
    executor._init_execution_control()
    executor.doing_shutdown = False
    with executor._execution_admission(None) as execution:
        assert execution is None
    with pytest.raises(ExecutionAuthorityError, match="register"):
        executor._execution_admission(VERSION)
    assert executor._execution_controller is None
    executor._shutdown_execution_control()


def test_result_binding_wakes_the_same_queue_used_by_a_result_copy(authority) -> None:
    controller, _ = authority
    original = SimpleNamespace(queue=Queue())
    with controller.admission(VERSION) as execution:
        ExecutionControlMixin._bind_execution_result(original, execution)
    copied = SimpleNamespace(**original.__dict__)
    controller.advance(REVISION).publication.wait()
    assert copied._execution is original._execution
    assert isinstance(copied.queue.get(timeout=1), ExecutionAuthorityError)


def test_real_ipc_revision_revokes_frontend_and_cancels_worker() -> None:
    table = WorkerVersionTable("session")
    server = ExecutionMailboxServer(table)
    controller = ExecutionController("session", IpcPublicationTransport(server.address))
    cancelled = Queue()
    try:
        controller.register(VERSION).wait()
        frontend_execution, frontend_queue = admit(controller)
        with table.admission(VERSION) as worker_execution:
            worker_execution._cancel = lambda: cancelled.put(99)
        controller.advance(REVISION).publication.wait()
        assert isinstance(frontend_queue.get(timeout=1), ExecutionAuthorityError)
        assert cancelled.get(timeout=1) == 99
        for execution in (frontend_execution, worker_execution):
            with pytest.raises(ExecutionAuthorityError):
                execution.check()
    finally:
        controller.close()
        server.close()
