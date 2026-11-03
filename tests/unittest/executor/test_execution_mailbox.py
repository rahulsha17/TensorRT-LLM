# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as WaitTimeout
from dataclasses import replace
from queue import Queue
from threading import Barrier, Event, Lock
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_mailbox import (
    ExecutionController,
    LocalPublicationTransport,
    PublicationReceipt,
    VersionPublication,
    VersionUpdate,
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only

VERSION = ExecutionVersion("tenant", "scope", 7)


@pytest.fixture
def local_controller():
    table = WorkerVersionTable("session")
    controller = ExecutionController("session", LocalPublicationTransport(table))
    yield controller, table
    controller.close()
    table.close()


def test_register_and_advance_publish_full_version(local_controller) -> None:
    controller, table = local_controller
    version = replace(VERSION, epoch=(1 << 64) - 2)
    initial = controller.register(version)
    assert initial.wait().observed_version == version
    assert table.current_version("tenant", "scope") == version

    successor = replace(version, epoch=version.epoch + 1)
    result = controller.advance(Revision("edit-1", version, successor))
    assert result.advance_receipt.revision.successor_version == successor
    assert result.publication.wait().observed_version == successor
    assert table.current_version("tenant", "scope") == successor
    assert controller.snapshot("tenant", "scope").current_version == successor


def test_late_retry_publishes_current_version_and_keeps_original_acceptance(
    local_controller,
) -> None:
    controller, table = local_controller
    controller.register(VERSION).wait()
    revision = Revision("edit-1", VERSION, replace(VERSION, epoch=8))
    first = controller.advance(revision)
    first.publication.wait()
    second = controller.advance(
        Revision("edit-2", replace(VERSION, epoch=8), replace(VERSION, epoch=9))
    )
    second.publication.wait()

    retried = controller.advance(revision)
    assert retried.advance_receipt == first.advance_receipt
    assert retried.publication is second.publication
    assert retried.publication.wait().observed_version.epoch == 9
    assert table.current_version("tenant", "scope").epoch == 9


@pytest.mark.parametrize(
    "other", [replace(VERSION, tenant_id="another"), replace(VERSION, scope_id="another")]
)
def test_scope_and_tenant_namespaces_are_independent(
    local_controller, other: ExecutionVersion
) -> None:
    controller, table = local_controller
    controller.register(VERSION).wait()
    controller.register(other).wait()
    newer = replace(VERSION, epoch=8)
    controller.advance(Revision("edit", VERSION, newer)).publication.wait()
    assert table.current_version("tenant", "scope") == newer
    assert table.current_version(other.tenant_id, other.scope_id) == other
    assert table.current_version("unknown", "scope") is None


def test_worker_handles_delayed_and_duplicate_updates_without_rollback() -> None:
    table = WorkerVersionTable("session")
    older = VersionUpdate("session", 1, VERSION)
    newer = VersionUpdate("session", 2, replace(VERSION, epoch=8))
    table.apply(newer)
    assert table.apply(older) == PublicationReceipt(older, newer.version)
    assert table.apply(newer) == PublicationReceipt(newer, newer.version)
    assert table.current_version("tenant", "scope") == newer.version


@pytest.mark.parametrize(
    "invalid",
    [
        VersionUpdate("foreign", 3, replace(VERSION, epoch=9)),
        VersionUpdate("session", 2, replace(VERSION, epoch=9)),
        VersionUpdate("session", 1, replace(VERSION, epoch=9)),
        VersionUpdate("session", 3, VERSION),
    ],
)
def test_worker_rejects_foreign_or_inconsistent_updates(invalid: VersionUpdate) -> None:
    table = WorkerVersionTable("session")
    current = replace(VERSION, epoch=8)
    table.apply(VersionUpdate("session", 2, current))
    with pytest.raises(ValueError):
        table.apply(invalid)
    assert table.current_version("tenant", "scope") == current


def test_closed_worker_table_cannot_acknowledge_or_report_authority() -> None:
    table = WorkerVersionTable("session")
    table.apply(VersionUpdate("session", 1, VERSION))
    table.close()
    with pytest.raises(RuntimeError, match="closed"):
        table.apply(VersionUpdate("session", 2, replace(VERSION, epoch=8)))
    with pytest.raises(RuntimeError, match="closed"):
        table.current_version("tenant", "scope")


class _LostAckTransport(LocalPublicationTransport):
    def __init__(self, table: WorkerVersionTable) -> None:
        super().__init__(table)
        self.lose_ack = False
        self.sent = []

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        self.sent.append(update)
        receipt = super().publish(update, stop)
        if self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("acknowledgement lost")
        return receipt


def test_lost_ack_preserves_authority_and_retry_reuses_publication_identity() -> None:
    table = WorkerVersionTable("session")
    transport = _LostAckTransport(table)
    controller = ExecutionController("session", transport)
    try:
        controller.register(VERSION).wait()
        transport.lose_ack = True
        revision = Revision("edit", VERSION, replace(VERSION, epoch=8))
        attempt = controller.advance(revision)
        with pytest.raises(TimeoutError, match="lost"):
            attempt.publication.wait()
        assert controller.snapshot("tenant", "scope").current_version.epoch == 8
        assert table.current_version("tenant", "scope").epoch == 8

        retry = controller.advance(revision)
        assert retry.advance_receipt == attempt.advance_receipt
        assert retry.publication.update == attempt.publication.update
        assert retry.publication.wait().observed_version.epoch == 8
        assert transport.sent[-1] == transport.sent[-2]
    finally:
        controller.close()


def test_old_retry_recovers_failed_latest_publication() -> None:
    table = WorkerVersionTable("session")
    transport = _LostAckTransport(table)
    controller = ExecutionController("session", transport)
    try:
        controller.register(VERSION).wait()
        first = Revision("first", VERSION, replace(VERSION, epoch=8))
        controller.advance(first).publication.wait()
        transport.lose_ack = True
        latest = controller.advance(
            Revision("second", replace(VERSION, epoch=8), replace(VERSION, epoch=9))
        )
        with pytest.raises(TimeoutError):
            latest.publication.wait()
        retry = controller.advance(first)
        assert retry.advance_receipt.revision == first
        assert retry.publication.update == latest.publication.update
        assert retry.publication.wait().observed_version.epoch == 9
    finally:
        controller.close()


class _BlockedTransport(LocalPublicationTransport):
    def __init__(self, table: WorkerVersionTable) -> None:
        super().__init__(table)
        self.entered = Event()
        self.release = Event()

    def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
        self.entered.set()
        while not self.release.wait(0.01):
            if stop.is_set():
                raise RuntimeError("closed")
        return super().publish(update, stop)


@pytest.mark.parametrize("local,enabled", [(True, True), (True, False), (False, True)])
def test_queued_local_advances_coalesce_without_losing_historical_lifetimes(
    monkeypatch, local, enabled
):
    from tensorrt_llm.execution_alignment import AlignmentInput, WorkerAlignmentState
    from tensorrt_llm.execution_fence import RequestDrainTracker

    monkeypatch.setenv("TRTLLM_RETIRE_COALESCE_ADVANCE", "1" if enabled else "0")
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    table.drain_tracker.enqueue(VERSION, lambda: 42)
    reserve = Mock(return_value=Mock(num_tokens=2))
    table.alignment_state = WorkerAlignmentState(reserve, lambda: None)
    transport = _BlockedTransport(table)
    applied = []
    original = table.apply

    def apply(update):
        applied.append(update)
        return original(update)

    monkeypatch.setattr(table, "apply", apply)

    class RemoteTransport:
        publish = transport.publish
        poll_fence = transport.poll_fence
        poll_alignment = transport.poll_alignment
        close = transport.close

    controller = ExecutionController("session", transport if local else RemoteTransport())
    try:
        initial = controller.register(VERSION)
        assert transport.entered.wait(5)
        revisions = []
        for epoch in (8, 9, 10):
            revisions.append(
                controller.advance(
                    Revision(
                        str(epoch), replace(VERSION, epoch=epoch - 1), replace(VERSION, epoch=epoch)
                    ),
                    alignment=AlignmentInput((1, 2, 3)),
                )
            )
        transport.release.set()
        initial.wait()
        for revision in revisions:
            receipt = revision.publication.wait()
            assert receipt.update == revision.publication.update
            assert receipt.observed_version.epoch == (
                10 if local and enabled else receipt.update.version.epoch
            )
            assert not revision.fence._future.done()
        assert [update.version.epoch for update in applied] == (
            [7, 10] if local and enabled else [7, 8, 9, 10]
        )
        assert table.observe_allowed((replace(VERSION, epoch=10),)) == (False,)
        table.alignment_state.process()
        reserve.assert_called_once()
        for revision in revisions[:-1]:
            with pytest.raises(RuntimeError, match="superseded"):
                revision.alignment.wait()
        table.drain_tracker.complete(42, VERSION)
        for revision in revisions:
            assert revision.fence.wait().request == revision.fence.request
        assert revisions[-1].alignment.wait().reusable_prefix_length == 2
    finally:
        transport.release.set()
        controller.close()
        table.close()
        table.alignment_state.process()


def test_coalesced_publication_failure_completes_every_waiter_and_retry_recovers(monkeypatch):
    from tensorrt_llm.execution_fence import RequestDrainTracker

    monkeypatch.setenv("TRTLLM_RETIRE_COALESCE_ADVANCE", "1")
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    table.drain_tracker.enqueue(VERSION, lambda: 42)
    transport = _BlockedTransport(table)
    original = transport.publish
    failed = []

    def publish(update, stop):
        if update.version.epoch == 9 and not failed:
            failed.append(update)
            raise RuntimeError("injected publish failure")
        return original(update, stop)

    monkeypatch.setattr(transport, "publish", publish)
    controller = ExecutionController("session", transport)
    try:
        initial = controller.register(VERSION)
        assert transport.entered.wait(5)
        first_revision = Revision("first", VERSION, replace(VERSION, epoch=8))
        first = controller.advance(first_revision)
        second = controller.advance(
            Revision("second", replace(VERSION, epoch=8), replace(VERSION, epoch=9))
        )
        transport.release.set()
        initial.wait()
        for publication in (first.publication, second.publication):
            with pytest.raises(RuntimeError, match="injected publish failure"):
                publication.wait()
        assert not first.fence._future.done()
        assert not second.fence._future.done()
        retry = controller.advance(first_revision)
        assert retry.publication.update == second.publication.update
        assert retry.publication.wait().observed_version.epoch == 9
        assert controller.snapshot("tenant", "scope").current_version.epoch == 9
        assert retry.fence is first.fence
        assert not second.fence._future.done()
        table.drain_tracker.complete(42, VERSION)
        assert first.fence.wait().request == first.fence.request
        assert second.fence.wait().request == second.fence.request
    finally:
        transport.release.set()
        controller.close()


@pytest.mark.parametrize("kind", ["admission", "resume", "source_release", "transfer"])
def test_advance_coalescing_cannot_overtake_same_scope_barrier(kind):
    from tensorrt_llm.execution_alignment import (
        AlignmentInput,
        AlignmentReceipt,
        AlignmentRequest,
        ExecutionPlacement,
        PrefixTransferPlan,
    )

    old, new = replace(VERSION, epoch=8), replace(VERSION, epoch=9)
    revision = Revision("edit", VERSION, old)
    revised = AlignmentInput((1, 2, 3))
    if kind in ("source_release", "transfer"):
        revised = replace(
            revised,
            destination=ExecutionPlacement("session", "destination"),
            transfer=PrefixTransferPlan(ExecutionPlacement("session", "source"), 2),
        )
    alignment = AlignmentRequest("session", revision, revised)
    barrier = VersionUpdate(
        "session",
        3,
        old,
        alignment=alignment,
        resume=AlignmentReceipt(alignment, "hold", 2) if kind == "resume" else None,
        admit_requests=kind in ("admission", "resume"),
        source_only=kind == "source_release",
        release_source=kind == "source_release",
    )
    first = VersionPublication(VersionUpdate("session", 2, old, admit_requests=False), Future())
    middle = VersionPublication(barrier, Future())
    latest = VersionPublication(VersionUpdate("session", 4, new, admit_requests=False), Future())
    controller = object.__new__(ExecutionController)
    controller._lock = Lock()
    controller._coalesce_advances = True
    controller._latest = {("tenant", "scope"): latest}
    controller._pending = {first, middle, latest}
    assert controller._select_publication(first) == (first, (first,))
    controller._pending.remove(middle)
    selected, covered = controller._select_publication(first)
    assert selected is latest
    assert set(covered) == {first, latest}


@pytest.mark.parametrize("kind", ["fence", "alignment", "failure", "worker_close"])
def test_local_readiness_wakes_waiters_without_timer_polling(monkeypatch, kind):
    from tensorrt_llm import execution_mailbox as module
    from tensorrt_llm.execution_alignment import AlignmentInput, WorkerAlignmentState
    from tensorrt_llm.execution_fence import RequestDrainTracker

    monkeypatch.setenv("TRTLLM_RETIRE_READY_NOTIFY", "1")
    waiting = Event()

    class NoTimerQueue(Queue):
        def get(self, block=True, timeout=None):
            if timeout is not None and timeout > 0 and self.empty():
                waiting.set()
                timeout = None
            return super().get(block=block, timeout=timeout)

    monkeypatch.setattr(module, "Queue", NoTimerQueue)
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    if kind != "alignment":
        table.drain_tracker.enqueue(VERSION, lambda: 42)
    table.alignment_state = WorkerAlignmentState(
        lambda _: Mock(num_tokens=2), lambda: None, notify_ready=table.notify_readiness
    )
    controller = ExecutionController("session", LocalPublicationTransport(table))
    try:
        controller.register(VERSION).wait()
        revised = controller.advance(
            Revision("edit", VERSION, replace(VERSION, epoch=8)),
            alignment=AlignmentInput((1, 2, 3)) if kind == "alignment" else None,
        )
        revised.publication.wait()
        assert waiting.wait(5)
        if kind == "alignment":
            table.alignment_state.process()
            assert revised.alignment.wait().reusable_prefix_length == 2
        elif kind == "fence":
            table.drain_tracker.complete(42, VERSION)
            table.notify_readiness()
            assert revised.fence.wait().request == revised.fence.request
        elif kind == "failure":
            table.drain_tracker.fail(RuntimeError("lost owner"))
            table.notify_readiness()
            with pytest.raises(RuntimeError, match="unknown"):
                revised.fence.wait()
        else:
            table.close()
            with pytest.raises(RuntimeError, match="closed"):
                revised.fence.wait()
    finally:
        controller.close()
        assert not table._readiness_listeners
        table.notify_readiness()
        table.close()
        table.alignment_state.process()


def test_readiness_storm_is_coalesced_and_shutdown_detaches_listener(monkeypatch):
    from tensorrt_llm.execution_mailbox import _ReadinessWake

    monkeypatch.setenv("TRTLLM_RETIRE_READY_NOTIFY", "1")
    table = WorkerVersionTable("session")
    transport = _BlockedTransport(table)
    controller = ExecutionController("session", transport)
    initial = controller.register(VERSION)
    assert transport.entered.wait(5)
    try:
        # An engine notification must never acquire the controller or worker
        # authority lock. These locks are deliberately held during the storm.
        with controller._lock, table._lock:
            for _ in range(1000):
                table.notify_readiness()
        with controller._queue.mutex:
            assert sum(isinstance(item, _ReadinessWake) for item in controller._queue.queue) == 1
        controller.close()
        with pytest.raises(RuntimeError, match="closed"):
            initial.wait()
        assert not table._readiness_listeners
        assert not controller._thread.is_alive()
        before = controller._queue.qsize()
        table.notify_readiness()
        assert controller._queue.qsize() == before
    finally:
        transport.release.set()
        controller.close()


@pytest.mark.parametrize("flag", ["TRTLLM_RETIRE_READY_NOTIFY", "TRTLLM_RETIRE_COALESCE_ADVANCE"])
def test_private_control_flags_reject_ambiguous_values(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    table = WorkerVersionTable("session")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ExecutionController("session", LocalPublicationTransport(table))
    assert not table._readiness_listeners


def test_early_advance_coalescing_keeps_existing_promises_until_engine_reclaim(monkeypatch):
    from tensorrt_llm.execution_alignment import AlignmentInput, WorkerAlignmentState
    from tensorrt_llm.execution_fence import RequestDrainTracker

    monkeypatch.setenv("TRTLLM_RETIRE_COALESCE_ADVANCE", "1")
    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    first_hold, latest_hold = Mock(num_tokens=2), Mock(num_tokens=2)
    holds = iter((first_hold, latest_hold))
    state = table.alignment_state = WorkerAlignmentState(lambda _: next(holds), lambda: None)
    transport = _BlockedTransport(table)
    transport.release.set()
    controller = ExecutionController("session", transport)
    try:
        controller.register(VERSION).wait()
        first = controller.advance(
            Revision("first", VERSION, replace(VERSION, epoch=8)),
            alignment=AlignmentInput((1, 2, 3)),
        )
        first.publication.wait()
        state.process()
        first_receipt = first.alignment.wait()
        transport.entered.clear()
        transport.release.clear()
        unrelated = controller.register(replace(VERSION, scope_id="other"))
        assert transport.entered.wait(5)
        second = controller.advance(
            Revision("second", replace(VERSION, epoch=8), replace(VERSION, epoch=9)),
            alignment=AlignmentInput((1, 2, 3)),
        )
        third = controller.advance(
            Revision("third", replace(VERSION, epoch=9), replace(VERSION, epoch=10)),
            alignment=AlignmentInput((1, 2, 3)),
        )
        transport.release.set()
        unrelated.wait()
        assert second.publication.wait().observed_version.epoch == 10
        third.publication.wait()
        first_hold.close.assert_not_called()
        with pytest.raises(RuntimeError, match="superseded"):
            state.authorize_resume(first_receipt)
        state.process(isolated_only=True)
        first_hold.close.assert_not_called()
        state.process()
        first_hold.close.assert_called_once()
        assert third.alignment.wait().reusable_prefix_length == 2
    finally:
        transport.release.set()
        controller.close()
        table.close()
        state.process()


def test_acceptance_can_advance_while_publication_is_pending() -> None:
    table = WorkerVersionTable("session")
    transport = _BlockedTransport(table)
    controller = ExecutionController("session", transport)
    try:
        initial = controller.register(VERSION)
        assert transport.entered.wait(5)
        revised = controller.advance(Revision("edit", VERSION, replace(VERSION, epoch=8)))
        assert controller.snapshot("tenant", "scope").current_version.epoch == 8
        assert table.current_version("tenant", "scope") is None
        with pytest.raises(WaitTimeout):
            initial.wait(timeout=0.01)
        assert controller.register(replace(VERSION, epoch=8)) is revised.publication
        transport.release.set()
        assert revised.publication.wait().observed_version.epoch == 8
    finally:
        controller.close()


def test_shutdown_fails_all_outstanding_waiters_and_stops_publisher() -> None:
    table = WorkerVersionTable("session")
    transport = _BlockedTransport(table)
    controller = ExecutionController("session", transport)
    first = controller.register(VERSION)
    assert transport.entered.wait(5)
    second = controller.advance(Revision("edit", VERSION, replace(VERSION, epoch=8)))
    controller.close()
    for publication in (first, second.publication):
        with pytest.raises(RuntimeError, match="closed"):
            publication.wait()
    assert not controller._thread.is_alive()
    assert controller.snapshot("tenant", "scope").current_version.epoch == 8
    with pytest.raises(RuntimeError, match="closed"):
        controller.register(VERSION)
    controller.close()


def test_publication_backlog_is_bounded_and_latest_version_can_be_retried() -> None:
    table = WorkerVersionTable("session")
    transport = _BlockedTransport(table)
    controller = ExecutionController("session", transport, max_pending=1)
    try:
        initial = controller.register(VERSION)
        assert transport.entered.wait(5)
        revision = Revision("edit", VERSION, replace(VERSION, epoch=8))
        accepted = controller.advance(revision)
        with pytest.raises(RuntimeError, match="backlog is full"):
            accepted.publication.wait()
        assert controller.snapshot("tenant", "scope").current_version.epoch == 8
        transport.release.set()
        initial.wait()
        retry = controller.advance(revision)
        assert retry.publication.update == accepted.publication.update
        assert retry.publication.wait().observed_version.epoch == 8
    finally:
        controller.close()


def test_concurrent_exact_retries_share_one_publication(local_controller) -> None:
    controller, _ = local_controller
    controller.register(VERSION).wait()
    revision = Revision("edit", VERSION, replace(VERSION, epoch=8))
    start = Barrier(8)

    def retry():
        start.wait(timeout=5)
        return controller.advance(revision)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result(timeout=5) for future in [pool.submit(retry) for _ in range(8)]]
    assert all(result.publication is results[0].publication for result in results)
    assert results[0].publication.wait().observed_version.epoch == 8


def test_mismatched_acknowledgement_cannot_complete_publication() -> None:
    class WrongAck(LocalPublicationTransport):
        def publish(self, update: VersionUpdate, stop: Event) -> PublicationReceipt:
            return PublicationReceipt(replace(update, session_id="old-worker"), update.version)

    controller = ExecutionController("session", WrongAck(WorkerVersionTable("session")))
    try:
        with pytest.raises(ValueError, match="does not match"):
            controller.register(VERSION).wait()
    finally:
        controller.close()


@pytest.mark.parametrize(
    "observed", [replace(VERSION, epoch=6), replace(VERSION, tenant_id="other")]
)
def test_receipt_requires_observation_of_the_requested_scope_and_epoch(
    observed: ExecutionVersion,
) -> None:
    with pytest.raises(ValueError):
        PublicationReceipt(VersionUpdate("session", 1, VERSION), observed)


def test_execution_poll_observes_revision_while_device_publication_is_blocked():
    from threading import Thread

    table = WorkerVersionTable("session")
    entered, release = Event(), Event()

    class Publisher:
        def publish(self, version):
            if version.epoch == 8:
                entered.set()
                assert release.wait(5)

    table.bind_device_publisher(Publisher())
    table.apply(VersionUpdate("session", 1, VERSION))
    next_version = replace(VERSION, epoch=8)
    publisher = Thread(target=lambda: table.apply(VersionUpdate("session", 2, next_version)))
    publisher.start()
    try:
        assert entered.wait(5)
        # The publisher still owns the main authority lock here. A layer hook
        # must observe revocation without waiting for that publisher to finish.
        assert table.observe_allowed((VERSION, next_version, None)) == (False, True, True)
    finally:
        release.set()
        publisher.join(5)
    assert not publisher.is_alive()
    table.apply(VersionUpdate("session", 1, VERSION))
    assert table.observe_allowed((VERSION, next_version)) == (False, True)
    table.close()
    assert table.observe_allowed((next_version, None)) == (False, True)


def test_failed_publication_closes_advisory_execution_observation():
    class Publisher:
        def publish(self, version):
            raise RuntimeError("device unavailable")

    table = WorkerVersionTable("session")
    table.bind_device_publisher(Publisher())
    with pytest.raises(RuntimeError, match="device unavailable"):
        table.apply(VersionUpdate("session", 1, VERSION))
    assert table.observe_allowed((VERSION, None)) == (False, True)


def test_resource_admission_requires_resume_and_retains_request_identity():
    from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentRequest
    from tensorrt_llm.execution_fence import RequestDrainTracker
    from tensorrt_llm.execution_gate import ExecutionAuthorityError

    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    successor = replace(VERSION, epoch=8)
    revision = Revision("edit", VERSION, successor)
    alignment = AlignmentRequest("session", revision, AlignmentInput((1, 2, 3)))
    table.apply(VersionUpdate("session", 1, successor, alignment=alignment))
    table.drain_tracker.enqueue(successor, lambda: 41)
    with pytest.raises(ExecutionAuthorityError, match="ALIGN and RESUME"):
        table.admit_resource(41, successor, "kv-send-session")
    assert not table.drain_tracker._resources
    assert not table._gate._executions


def test_resource_admission_releases_host_gate_handle_without_completing_the_resource():
    from tensorrt_llm.execution_fence import RequestDrainTracker

    table = WorkerVersionTable("session")
    table.drain_tracker = RequestDrainTracker()
    table.apply(VersionUpdate("session", 1, VERSION))
    table.drain_tracker.enqueue(VERSION, lambda: 41)
    claim = table.admit_resource(41, VERSION, "kv-receive-session")
    assert claim.request_id == 41 and claim.version == VERSION
    assert not table._gate._executions
    table.drain_tracker.complete(41, VERSION)
    assert not table.drain_tracker.drained(VERSION)
    table.drain_tracker.release_resource(claim)
    assert table.drain_tracker.drained(VERSION)


class _VirtualPollingQueue:
    """Account for requested idle time without depending on wall-clock timing."""

    def __init__(self):
        from queue import Queue

        self.queue = Queue()
        self.elapsed = 0.0
        self.ready = False
        self.idle_calls = 0
        self.inject = None

    def put(self, item):
        self.queue.put(item)

    def get(self, timeout=None):
        from queue import Empty

        if not self.queue.empty():
            return self.queue.get_nowait()
        self.idle_calls += 1
        assert self.idle_calls < 2000, "Lifecycle polling made no bounded progress"
        if self.inject is not None and self.ready:
            inject, self.inject = self.inject, None
            inject()
            return self.queue.get_nowait()
        if timeout is None:
            return None
        self.elapsed += timeout
        self.ready = True
        raise Empty


def _run_lifecycle_backlog(
    monkeypatch, *, aligned=False, delayed=0.0, fail_one=False, inject=False
):
    import tensorrt_llm.execution_mailbox as mailbox
    from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentReceipt
    from tensorrt_llm.execution_fence import FenceReceipt

    queue = _VirtualPollingQueue()

    class Transport:
        closed = False
        injection_remaining = None

        def publish(self, update, stop):
            if update.version.scope_id == "urgent":
                self.injection_remaining = len(controller._pending_fences)
            return PublicationReceipt(update, update.version)

        def poll_fence(self, request, stop):
            if not queue.ready or queue.elapsed < delayed:
                return None
            if fail_one and request.revision.expected_version.scope_id == "scope-7":
                raise ValueError("synthetic fence failure")
            return FenceReceipt(request)

        def poll_alignment(self, request, stop):
            if not queue.ready or queue.elapsed < delayed:
                return None
            return AlignmentReceipt(request, "held-" + request.revision.revision_id, 0)

        def close(self):
            self.closed = True

    transport = Transport()
    with monkeypatch.context() as patch:
        patch.setattr(mailbox.Thread, "start", lambda self: None)
        patch.setattr(mailbox, "Queue", lambda: queue)
        controller = ExecutionController("session", transport)
    results = []
    for index in range(32):
        old = replace(VERSION, scope_id=f"scope-{index}")
        controller.register(old)
        results.append(
            controller.advance(
                Revision(f"edit-{index}", old, replace(old, epoch=old.epoch + 1)),
                alignment=AlignmentInput((1, 2, 3)) if aligned else None,
            )
        )
    if inject:
        queue.inject = lambda: controller.register(replace(VERSION, scope_id="urgent"))
    controller._run()
    assert transport.closed
    assert not controller._pending_fences
    assert not controller._pending_alignments
    for index, result in enumerate(results):
        if fail_one and index == 7:
            with pytest.raises(ValueError, match="synthetic fence failure"):
                result.fence.wait(0)
        else:
            assert result.fence.wait(0).request.revision == result.advance_receipt.revision
        if aligned:
            assert result.alignment.wait(0).request.revision == result.advance_receipt.revision
    return queue, transport


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("fail_one", [False, True])
def test_ready_lifecycle_backlog_does_not_sleep_per_receipt(monkeypatch, aligned, fail_one):
    queue, _ = _run_lifecycle_backlog(monkeypatch, aligned=aligned, fail_one=fail_one)
    assert queue.elapsed <= 0.02


def test_pending_lifecycle_backlog_sleeps_and_checks_all_scopes(monkeypatch):
    queue, _ = _run_lifecycle_backlog(monkeypatch, aligned=True, delayed=0.05)
    assert 0.05 <= queue.elapsed <= 0.07
    assert queue.idle_calls < 200


def test_ready_lifecycle_backlog_does_not_starve_new_publication(monkeypatch):
    _, transport = _run_lifecycle_backlog(monkeypatch, aligned=True, inject=True)
    assert transport.injection_remaining >= 30
