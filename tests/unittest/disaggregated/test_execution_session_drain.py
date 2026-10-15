# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FENCE accounts for native session access independently of request completion."""

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.native.execution_wire import (
    ExecutionTransferWire,
    TransferIdentityError,
)
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 0)


def _worker():
    worker = object.__new__(TransferWorker)
    worker._execution_session_lock = threading.Lock()
    worker._execution_drain = None
    worker._execution_authority = None
    worker._execution_wire = ExecutionTransferWire()
    worker._sessions_started = False
    worker._config = TransferWorkerConfig(
        kv_cache_manager=Mock(),
        device_id=0,
        instance_name="test",
        enforce_physical_ownership=True,
        tx_timeout_s=5.0,
        tx_overall_timeout_s=5.0,
        rx_timeout_s=5.0,
    )
    worker._retirement_watchdog = RetirementWatchdog(Mock())
    worker._aux_buffer = Mock()
    worker._aux_buffer.alloc_slot.return_value = SimpleNamespace(id=7)
    for name in ("_sender", "_receiver"):
        peer = Mock()
        peer._enforce_physical_ownership = True
        peer._sessions_lock = threading.Lock()
        peer._sessions = {}
        peer._resolve_session_entry.side_effect = lambda rid, entry: entry
        peer._get_req_info.return_value = None
        setattr(worker, name, peer)
    return worker


def _request(version=VERSION, wire_id=99):
    return SimpleNamespace(
        py_request_id=41,
        py_execution_version=version,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=wire_id),
        prompt_len=32,
    )


def _bind(worker, drain):
    table = WorkerVersionTable("authority")
    table.drain_tracker = drain
    table.apply(VersionUpdate("authority", 1, VERSION))
    worker.bind_execution_authority(table)
    return table


def _create(worker, direction, request):
    return (worker.create_tx_session if direction == "send" else worker.create_rx_session)(request)


def _peer(worker, direction):
    return worker._sender if direction == "send" else worker._receiver


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_claim_precedes_session_publication(direction):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)

    def published(session):
        assert session.request_id == 41
        drain.complete(41, VERSION)
        assert not drain.drained(VERSION)

    _peer(worker, direction).setup_session.side_effect = published
    session = _create(worker, direction, _request())
    assert not drain.drained(VERSION)
    assert session.close()
    assert drain.drained(VERSION)
    assert session.close()
    assert drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_before_claim_attachment_is_not_lost(direction, cleanup_fails):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)

    def published(session):
        if cleanup_fails:
            worker._aux_buffer.free_slot.side_effect = RuntimeError("early cleanup failed")
            with pytest.raises(RuntimeError, match="early cleanup failed"):
                session.close()
        else:
            assert session.close()
        drain.complete(41, VERSION)
        assert not drain.drained(VERSION)

    _peer(worker, direction).setup_session.side_effect = published
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup is unproven"):
            _create(worker, direction, _request())
        with pytest.raises(RuntimeError, match="drain is unknown"):
            drain.drained(VERSION)
    else:
        session = _create(worker, direction, _request())
        assert drain.drained(VERSION)
        assert session.close()


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_cancelled_session_keeps_fence_pending_until_physical_drain_and_cleanup(direction):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    session = _create(worker, direction, _request())
    accessor = object()
    assert session._retirement.expose(accessor)
    drain.complete(41, VERSION)
    session.cancel_local()
    assert session.has_failed()
    assert session.close() is False
    assert not drain.drained(VERSION)
    assert session._retirement.settle(accessor)

    def unregister(_rid):
        assert not drain.drained(VERSION)

    _peer(worker, direction).clear_session.side_effect = unregister
    assert session.close()
    assert drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("stage", ["construct", "release"])
def test_uncertain_setup_or_cleanup_never_proves_fence(direction, stage):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    peer = _peer(worker, direction)
    if stage == "construct":
        peer.setup_session.side_effect = RuntimeError("publication uncertain")
        with pytest.raises(RuntimeError, match="publication uncertain"):
            _create(worker, direction, _request())
    else:
        session = _create(worker, direction, _request())
        worker._aux_buffer.free_slot.side_effect = RuntimeError("release uncertain")
        with pytest.raises(RuntimeError, match="release uncertain"):
            session.close()
        with pytest.raises(RuntimeError, match="cleanup is unproven"):
            session.close()
        worker._aux_buffer.free_slot.assert_called_once_with(7)
    drain.complete(41, VERSION)
    with pytest.raises(RuntimeError, match="drain is unknown"):
        drain.drained(VERSION)
    assert len(drain._resources) == 1


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_reused_request_id_does_not_release_successor_session(direction):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    old = _create(worker, direction, _request())
    drain.complete(41, VERSION)
    successor = replace(VERSION, epoch=1)
    drain.enqueue(successor, lambda: 41)
    worker._execution_authority.apply(VersionUpdate("authority", 2, successor))
    new = _create(worker, direction, _request(successor, wire_id=100))
    drain.complete(41, successor)
    assert old.close()
    assert drain.drained(VERSION)
    assert not drain.drained(successor)
    assert old.close()
    assert not drain.drained(successor)
    assert new.close()
    assert drain.drained(successor)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_claim_release_error_after_side_effect_keeps_fence_unknown(direction, monkeypatch):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    session = _create(worker, direction, _request())
    drain.complete(41, VERSION)
    release = drain.release_resource

    def fail_after_release(claim):
        release(claim)
        raise RuntimeError("completion uncertain")

    callback = Mock(side_effect=fail_after_release)
    monkeypatch.setattr(drain, "release_resource", callback)
    with pytest.raises(RuntimeError, match="completion uncertain"):
        session.close()
    with pytest.raises(RuntimeError, match="cleanup is unproven"):
        session.close()
    callback.assert_called_once()
    with pytest.raises(RuntimeError, match="drain is unknown"):
        drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_tagged_session_without_tracking_never_reaches_backend(direction):
    worker = _worker()
    with pytest.raises(RuntimeError, match="requires execution drain"):
        _create(worker, direction, _request())
    _peer(worker, direction).setup_session.assert_not_called()


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("live", [False, True])
def test_session_requires_matching_admitted_request(direction, live):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    if live:
        drain.enqueue(replace(VERSION, epoch=1), lambda: 41)
    with pytest.raises(RuntimeError, match="no matching live owner"):
        _create(worker, direction, _request())
    _peer(worker, direction).setup_session.assert_not_called()


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_changed_version_during_construction_fails_closed(direction):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    request = _request()

    def mutate(_session):
        request.py_execution_version = replace(VERSION, epoch=1)

    _peer(worker, direction).setup_session.side_effect = mutate
    with pytest.raises(RuntimeError, match="ownership changed"):
        _create(worker, direction, request)
    with pytest.raises(RuntimeError, match="drain is unknown"):
        drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_untagged_session_still_works_but_prevents_late_binding(direction):
    worker = _worker()
    session = _create(worker, direction, _request(None))
    assert session.close()
    with pytest.raises(RuntimeError, match="after session admission"):
        worker.bind_execution_drain(RequestDrainTracker())


@pytest.mark.parametrize("missing", ["ownership", "retirement"])
def test_binding_requires_physical_retirement(missing):
    worker = _worker()
    if missing == "ownership":
        worker._config.enforce_physical_ownership = False
    else:
        worker._retirement_watchdog = None
    with pytest.raises(RuntimeError, match="requires physical ownership and retirement"):
        worker.bind_execution_drain(RequestDrainTracker())


def test_binding_cannot_replace_the_tracker():
    worker, drain = _worker(), RequestDrainTracker()
    worker.bind_execution_drain(drain)
    worker.bind_execution_drain(drain)
    with pytest.raises(RuntimeError, match="cannot change"):
        worker.bind_execution_drain(RequestDrainTracker())


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("unavailable", ["revoked", "unknown", "closed"])
def test_unavailable_authority_rejects_before_claim_or_native_setup(direction, unavailable):
    worker, drain = _worker(), RequestDrainTracker()
    table = _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    if unavailable == "revoked":
        table.apply(VersionUpdate("authority", 2, replace(VERSION, epoch=1)))
    elif unavailable == "unknown":
        table = WorkerVersionTable("empty")
        table.drain_tracker = drain
        worker = _worker()
        worker.bind_execution_authority(table)
    else:
        table.close()
    with pytest.raises(ExecutionAuthorityError):
        _create(worker, direction, _request())
    assert not drain._resources
    worker._aux_buffer.alloc_slot.assert_not_called()
    _peer(worker, direction).setup_session.assert_not_called()
    drain.complete(41, VERSION)
    assert drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_drain_tracking_alone_does_not_grant_session_authority(direction):
    worker, drain = _worker(), RequestDrainTracker()
    worker.bind_execution_drain(drain)
    drain.enqueue(VERSION, lambda: 41)
    with pytest.raises(RuntimeError, match="requires execution authority"):
        _create(worker, direction, _request())
    assert not drain._resources
    _peer(worker, direction).setup_session.assert_not_called()


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_admitted_session_keeps_fence_pending_without_blocking_publication(direction, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    worker, drain = _worker(), RequestDrainTracker()
    table = _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    registered = threading.Event()
    release_claim = threading.Event()
    constructing = threading.Event()
    release_construction = threading.Event()
    publication_attempt = threading.Event()
    acquire = drain.acquire_resource

    def acquire_and_pause(*args, **kwargs):
        claim = acquire(*args, **kwargs)
        registered.set()
        assert release_claim.wait(5)
        return claim

    def construct(_session):
        constructing.set()
        assert release_construction.wait(5)

    def publish():
        publication_attempt.set()
        return table.apply(VersionUpdate("authority", 2, replace(VERSION, epoch=1)))

    monkeypatch.setattr(drain, "acquire_resource", acquire_and_pause)
    _peer(worker, direction).setup_session.side_effect = construct
    with ThreadPoolExecutor(max_workers=2) as pool:
        creation = pool.submit(_create, worker, direction, _request())
        try:
            assert registered.wait(5)
            publication = pool.submit(publish)
            assert publication_attempt.wait(5)
            assert not publication.done()
            release_claim.set()
            assert constructing.wait(5)
            assert publication.result(5).observed_version.epoch == 1
            drain.complete(41, VERSION)
            assert not drain.drained(VERSION)
        finally:
            release_claim.set()
            release_construction.set()
        session = creation.result(5)
    assert session.close()
    assert drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_replacing_bound_tracker_cannot_move_a_claim_to_another_drain(direction):
    worker, drain = _worker(), RequestDrainTracker()
    table = _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    table.drain_tracker = RequestDrainTracker()
    with pytest.raises(RuntimeError, match="drain tracker changed"):
        _create(worker, direction, _request())
    assert not drain._resources and not table.drain_tracker._resources
    _peer(worker, direction).setup_session.assert_not_called()


@pytest.mark.parametrize("missing", ["ownership", "retirement", "tracker", "open"])
def test_authority_binding_requires_complete_live_ownership(missing):
    worker = _worker()
    table = WorkerVersionTable("authority")
    table.drain_tracker = RequestDrainTracker()
    if missing == "ownership":
        worker._config.enforce_physical_ownership = False
    elif missing == "retirement":
        worker._retirement_watchdog = None
    elif missing == "tracker":
        table.drain_tracker = None
    else:
        table.close()
    with pytest.raises(RuntimeError):
        worker.bind_execution_authority(table)
    assert worker._execution_authority is None and worker._execution_drain is None


def test_authority_binding_is_fixed_before_session_admission():
    worker, drain = _worker(), RequestDrainTracker()
    table = _bind(worker, drain)
    worker.bind_execution_authority(table)
    other = WorkerVersionTable("other")
    other.drain_tracker = drain
    with pytest.raises(RuntimeError, match="cannot change"):
        worker.bind_execution_authority(other)
    fresh = _worker()
    fresh.bind_execution_drain(RequestDrainTracker())
    with pytest.raises(RuntimeError, match="different drain"):
        fresh.bind_execution_authority(table)
    untagged = _worker()
    assert _create(untagged, "send", _request(None)).close()
    with pytest.raises(RuntimeError, match="after session admission"):
        untagged.bind_execution_authority(table)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_reused_exchange_releases_only_unpublished_claim(direction):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    original = _create(worker, direction, _request())
    original_claims = set(drain._resources)
    with pytest.raises(TransferIdentityError, match="reused"):
        _create(worker, direction, _request())
    assert set(drain._resources) == original_claims
    _peer(worker, direction).setup_session.assert_called_once()
    drain.complete(41, VERSION)
    assert not drain.drained(VERSION)
    assert original.close()
    assert drain.drained(VERSION)
    # Cleanup does not permit a delayed message to target a recycled ID.
    drain.enqueue(VERSION, lambda: 42)
    successor = _request()
    successor.py_request_id = 42
    with pytest.raises(TransferIdentityError, match="reused"):
        _create(worker, direction, successor)
    drain.complete(42, VERSION)
    assert drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("field", ["cache_salt", "disagg_request_id"])
def test_native_construction_cannot_change_bound_wire_identity(direction, field):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    request = _request()
    request.cache_salt = "original"

    def mutate(_session):
        if field == "cache_salt":
            request.cache_salt = "changed"
        else:
            request.py_disaggregated_params.disagg_request_id = 100

    _peer(worker, direction).setup_session.side_effect = mutate
    with pytest.raises(RuntimeError, match="ownership changed"):
        _create(worker, direction, request)
    drain.complete(41, VERSION)
    with pytest.raises(RuntimeError, match="drain is unknown"):
        drain.drained(VERSION)


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("context_id", [None, 98])
def test_wire_uses_session_id_fallback_before_publication(direction, context_id):
    worker, drain = _worker(), RequestDrainTracker()
    _bind(worker, drain)
    drain.enqueue(VERSION, lambda: 41)
    request = _request(wire_id=None)
    request.py_disaggregated_params.ctx_request_id = context_id
    session = _create(worker, direction, request)
    assert session.disagg_request_id == (41 if context_id is None else context_id)
    assert session.close()
    drain.complete(41, VERSION)
    assert drain.drained(VERSION)
