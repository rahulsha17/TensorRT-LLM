# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A late mirrored admission must never overtake distributed FENCE."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.execution_guard import ExecutionDeviceGuard
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_collective import CollectivePublicationTransport, ExecutionParticipant
from tensorrt_llm.execution_consensus import ExecutionStepConsensus
from tensorrt_llm.execution_fence import FenceRequest, RequestDrainTracker
from tensorrt_llm.execution_mailbox import (
    LocalPublicationTransport,
    VersionUpdate,
    WorkerVersionTable,
)
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
FENCE = FenceRequest("session", Revision("edit", OLD, NEW))


def item(version=OLD, identity=42):
    return SimpleNamespace(
        id=identity, is_normal_request=True, request=SimpleNamespace(py_execution_version=version)
    )


def group(size):
    votes = {}
    lock = Lock()
    barrier = Barrier(size, timeout=3)
    source_entered = Event()

    def gather(vote):
        with lock:
            votes[vote.rank] = vote
            if vote.rank == 0:
                source_entered.set()
        barrier.wait()
        return [votes[rank] for rank in range(size)]

    guards = []
    for rank in range(size):
        guard = object.__new__(ExecutionDeviceGuard)
        guard.table = WorkerVersionTable("session")
        guard.table.drain_tracker = RequestDrainTracker()
        guard.table.apply(VersionUpdate("session", 1, OLD))
        guard.table.apply(VersionUpdate("session", 2, NEW))
        guard._admission_source = rank == 0
        guard._rank_consensus = ExecutionStepConsensus("session", rank, tuple(range(size)), gather)
        guards.append(guard)
    return guards, source_entered


@pytest.mark.parametrize("size", [2, 4])
def test_late_peer_registration_blocks_source_completion_and_global_fence(size):
    guards, source_entered = group(size)
    guards[0].table.drain_tracker.enqueue(OLD, lambda: 42)
    participants = tuple(ExecutionParticipant(rank, 0) for rank in range(size))
    transport = CollectivePublicationTransport(
        participants,
        {
            participant: LocalPublicationTransport(guard.table)
            for participant, guard in zip(participants, guards)
        },
    )
    release = Event()

    def arrive(rank):
        if rank == size - 1:
            assert release.wait(3)
        guards[rank].register_broadcast_requests([item(), item(None, 43)])

    with ThreadPoolExecutor(max_workers=size) as pool:
        futures = [pool.submit(arrive, rank) for rank in range(size)]
        try:
            assert source_entered.wait(3)
            assert not futures[0].done()
            assert guards[-1].table.drain_tracker.drained(OLD)
            assert transport.poll_fence(FENCE, Event()) is None
        finally:
            release.set()
        for future in futures:
            future.result(3)
    # Publication already revoked OLD. Recording an in-flight envelope does not
    # grant authority to that version or allow FENCE before its local cleanup.
    for guard in guards:
        assert guard.table.current_version("tenant", "scope") == NEW
        assert not guard.table.drain_tracker.drained(OLD)
    guards[0].terminal(42, OLD)
    assert transport.poll_fence(FENCE, Event()) is None
    for guard in guards[1:]:
        guard.terminal(42, OLD)
    assert transport.poll_fence(FENCE, Event()).request == FENCE
    transport.close()


@pytest.mark.parametrize(
    "fault", ["missing_source", "changed_version", "lost_tag", "peer_collision"]
)
def test_failed_registration_or_mismatched_envelope_invalidates_every_drain(fault):
    guards, _ = group(2)
    if fault != "missing_source":
        guards[0].table.drain_tracker.enqueue(OLD, lambda: 42)
    if fault == "peer_collision":
        guards[1].table.drain_tracker.enqueue(OLD, lambda: 42)

    def arrive(rank):
        version = OLD
        if fault == "changed_version" and rank == 0:
            version = NEW
        if fault == "lost_tag" and rank == 1:
            version = None
        with pytest.raises(RuntimeError):
            guards[rank].register_broadcast_requests([item(version)])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(arrive, (0, 1)))
    for guard in guards:
        guard.terminal(42, OLD)
        with pytest.raises(RuntimeError, match="unknown"):
            guard.table.poll_fence(FENCE)


@pytest.mark.parametrize("untagged", [False, True])
def test_empty_and_untagged_envelopes_participate_without_tagged_holds(untagged):
    guards, _ = group(2)
    items = [item(None)] if untagged else []
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda guard: guard.register_broadcast_requests(items), guards))
    assert all(guard.table.drain_tracker.drained(OLD) for guard in guards)


@pytest.mark.parametrize("failure", [False, True])
def test_executor_restores_identity_then_acknowledges_before_waiting_queue(failure):
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = None
    executor.control_requests = []
    executor.is_shutdown = False
    executor._disable_mpi = False
    executor._has_pending_connector_transfers = Mock(return_value=False)
    executor.dist = SimpleNamespace(rank=1, tp_size=2, cp_size=1, has_pp=False)
    received = item(None)
    del received.request.py_execution_version
    executor.request_broadcaster = SimpleNamespace(
        broadcast=Mock(return_value=([received], (("py_execution_version", {42: OLD}),)))
    )
    executor._handle_special_queue_items = lambda items: items
    waiting_queue = MagicMock()
    waiting_queue.__len__.return_value = 0

    def acknowledge(items):
        assert items == [received] and received.request.py_execution_version == OLD
        waiting_queue.add_requests.assert_not_called()
        if failure:
            raise RuntimeError("peer did not acknowledge")

    executor._execution_guard = SimpleNamespace(register_broadcast_requests=acknowledge)
    if failure:
        with pytest.raises(RuntimeError, match="acknowledge"):
            executor._fetch_and_enqueue_requests(waiting_queue, 0)
        waiting_queue.add_requests.assert_not_called()
    else:
        executor._fetch_and_enqueue_requests(waiting_queue, 0)
        waiting_queue.add_requests.assert_called_once_with([received])


def test_single_rank_has_no_mirrored_admission_side_effects():
    guard = object.__new__(ExecutionDeviceGuard)
    guard._rank_consensus = None
    guard.table = Mock()
    guard.register_broadcast_requests([item()])
    assert guard.table.mock_calls == []


def test_local_snapshot_failure_reaches_peers_and_invalidates_drain():
    guards, _ = group(2)

    def vote(rank):
        request = SimpleNamespace(
            py_request_id=42,
            py_execution_version=OLD,
            get_num_tokens=lambda _: 32,
            get_token=lambda beam, position: 7,
            is_context_init_state=False,
            context_current_position=0,
            context_chunk_size=32,
            state=None if rank == 0 else SimpleNamespace(name="context"),
        )
        with pytest.raises(RuntimeError, match="invalid local input"):
            guards[rank]._coordinate("context_commit", [request], [True])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(vote, (0, 1)))
    for guard in guards:
        with pytest.raises(RuntimeError, match="unknown"):
            guard.table.poll_fence(FENCE)
