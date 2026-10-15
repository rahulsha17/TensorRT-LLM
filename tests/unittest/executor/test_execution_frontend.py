# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote frontend host authority, independent of worker publication and drain."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_alignment import AlignmentInput, AlignmentReceipt, AlignmentRequest
from tensorrt_llm.execution_authority import Revision
from tensorrt_llm.execution_fence import RequestDrainTracker
from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import VersionUpdate
from tensorrt_llm.execution_version import ExecutionVersion
from tensorrt_llm.executor.execution_frontend import FrontendAuthorityGate

pytestmark = pytest.mark.cpu_only
OLD = ExecutionVersion("tenant", "scope", 7)
NEW = replace(OLD, epoch=8)
REVISION = Revision("edit", OLD, NEW)
INITIAL = VersionUpdate("session", 1, OLD)


def resumed(revision=REVISION, sequence=3):
    request = AlignmentRequest("session", revision, AlignmentInput((1, 2, 3)))
    return VersionUpdate(
        "session",
        sequence,
        revision.successor_version,
        alignment=request,
        resume=AlignmentReceipt(request, f"hold-{sequence}", 2),
    )


@pytest.fixture
def gate():
    value = FrontendAuthorityGate("session")
    value.bind("owner", 60)
    value.observe("owner", INITIAL)
    value.activate("owner", INITIAL)
    yield value
    value.close()


def rejects(gate, version):
    with pytest.raises(ExecutionAuthorityError):
        with gate.admission(version):
            pytest.fail("unauthorized work was admitted")


def test_initial_observation_requires_explicit_activation():
    gate = FrontendAuthorityGate("session")
    rejects(gate, OLD)
    gate.bind("owner", 60)
    rejects(gate, OLD)
    assert gate.observe("owner", INITIAL).observed_version == OLD
    rejects(gate, OLD)
    assert not gate.activate("owner", replace(INITIAL, sequence=2))
    assert gate.activate("owner", INITIAL)
    with gate.admission(OLD) as execution:
        execution.check()
    gate.close()


def test_seal_revokes_enqueued_work_and_buffered_output_without_waiting_for_workers(gate):
    with gate.admission(OLD) as execution:
        execution._wake, execution._cancel = Mock(), Mock()
        wake, cancel = execution._wake, execution._cancel
    effect = Mock()
    assert gate.seal("owner", REVISION) == REVISION
    rejects(gate, OLD)
    rejects(gate, NEW)
    with pytest.raises(ExecutionAuthorityError):
        execution.commit(effect)
    effect.assert_not_called()
    wake.assert_called_once()
    cancel.assert_called_once()
    assert isinstance(wake.call_args.args[0], ExecutionAuthorityError)
    # Neither publication nor a FENCE receipt was necessary for host revocation.
    gate.seal("owner", REVISION)
    gate.close()
    wake.assert_called_once()
    cancel.assert_called_once()


def test_successor_requires_exact_resume_activation(gate):
    gate.seal("owner", REVISION)
    update = VersionUpdate("session", 2, NEW)
    gate.observe("owner", update)
    with pytest.raises(ExecutionAuthorityError, match="RESUME"):
        gate.activate("owner", update)
    rejects(gate, NEW)
    resume = resumed()
    assert not gate.activate("owner", resume)
    gate.observe("owner", resume)
    rejects(gate, NEW)
    assert gate.activate("owner", resume)
    with gate.admission(NEW) as execution:
        execution.check()
    rejects(gate, OLD)


def test_historical_retry_and_delayed_activation_do_not_reopen_old_epoch(gate):
    gate.seal("owner", REVISION)
    update = resumed()
    gate.observe("owner", update)
    gate.activate("owner", update)
    next_revision = Revision("second-edit", NEW, replace(NEW, epoch=9))
    gate.seal("owner", next_revision)
    assert not gate.activate("owner", INITIAL)
    assert not gate.activate("owner", update)
    latest = resumed(next_revision, 5)
    gate.observe("owner", latest)
    gate.activate("owner", latest)
    assert gate.seal("owner", REVISION) == REVISION
    assert gate.observe("owner", update).observed_version == latest.version
    with gate.admission(latest.version) as execution:
        execution.check()
    rejects(gate, NEW)


def test_seal_chain_can_advance_before_worker_publications_catch_up(gate):
    gate.seal("owner", REVISION)
    next_revision = Revision("second-edit", NEW, replace(NEW, epoch=9))
    gate.seal("owner", next_revision)
    update = resumed()
    gate.observe("owner", update)
    assert not gate.activate("owner", update)
    latest = resumed(next_revision, 5)
    gate.observe("owner", latest)
    assert gate.activate("owner", latest)


def test_seal_only_affects_the_revised_tenant_scope(gate):
    unrelated = [replace(OLD, tenant_id="other"), replace(OLD, scope_id="other")]
    active = []
    for sequence, version in enumerate(unrelated, 2):
        update = VersionUpdate("session", sequence, version)
        gate.observe("owner", update)
        gate.activate("owner", update)
        with gate.admission(version) as execution:
            execution._cancel = Mock()
            active.append(execution)
    gate.seal("owner", REVISION)
    for execution in active:
        execution.check()
        execution._cancel.assert_not_called()


@pytest.mark.parametrize("fault", ["conflict", "unknown", "stale"])
def test_invalid_seal_cannot_revoke_current_work(gate, fault):
    if fault == "conflict":
        gate.seal("owner", REVISION)
        update = resumed()
        gate.observe("owner", update)
        gate.activate("owner", update)
        invalid = Revision("edit", NEW, replace(NEW, epoch=9))
        current = NEW
    elif fault == "unknown":
        expected = replace(OLD, scope_id="unknown")
        invalid = Revision("edit", expected, replace(expected, epoch=8))
        current = OLD
    else:
        expected = replace(OLD, epoch=6)
        invalid = Revision("older", expected, OLD)
        current = OLD
    with pytest.raises(ValueError):
        gate.seal("owner", invalid)
    with gate.admission(current) as execution:
        execution.check()


@pytest.mark.parametrize(
    "fault", ["sequence-conflict", "unsealed", "rewind", "old-sequence", "floor"]
)
def test_conflicting_publication_closes_authority(gate, fault):
    if fault == "sequence-conflict":
        invalid = replace(INITIAL, version=NEW)
    elif fault == "unsealed":
        invalid = VersionUpdate("session", 2, NEW)
    elif fault == "rewind":
        invalid = VersionUpdate("session", 2, replace(OLD, epoch=6))
    elif fault == "old-sequence":
        gate.observe("owner", replace(INITIAL, sequence=5))
        invalid = VersionUpdate("session", 2, NEW)
    else:
        gate.seal("owner", REVISION)
        invalid = VersionUpdate("session", 2, replace(NEW, epoch=9))
    with pytest.raises(ExecutionAuthorityError):
        gate.observe("owner", invalid)
    assert gate.closed
    rejects(gate, OLD)
    rejects(gate, NEW)


@pytest.mark.parametrize("operation", ["bind", "renew", "seal", "observe", "activate", "release"])
def test_foreign_owner_cannot_change_the_bound_frontend(gate, operation):
    argument = {
        "bind": 60,
        "renew": 1,
        "seal": REVISION,
        "observe": INITIAL,
        "activate": INITIAL,
    }
    with pytest.raises(ValueError):
        if operation == "release":
            gate.release("foreign")
        else:
            getattr(gate, operation)("foreign", argument[operation])
    with gate.admission(OLD) as execution:
        execution.check()


@pytest.mark.parametrize("renewal", ["bind-retry", "duplicate-heartbeat", "older-heartbeat"])
def test_replays_cannot_extend_the_owner_lease(renewal):
    now = [100.0]
    gate = FrontendAuthorityGate("session", clock=lambda: now[0])
    gate.bind("owner", 10)
    gate.observe("owner", INITIAL)
    gate.activate("owner", INITIAL)
    gate.renew("owner", 2)
    with gate.admission(OLD) as execution:
        wake = execution._wake = Mock()
    now[0] = 109
    if renewal == "bind-retry":
        gate.bind("owner", 10)
    else:
        assert gate.renew("owner", 2 if renewal == "duplicate-heartbeat" else 1) == 2
    now[0] = 110
    assert gate.expire()
    wake.assert_called_once()
    with pytest.raises(ExecutionAuthorityError):
        gate.renew("owner", 3)
    with pytest.raises(ExecutionAuthorityError):
        gate.bind("owner", 10)
    with pytest.raises(ExecutionAuthorityError):
        gate.activate("owner", INITIAL)
    rejects(gate, OLD)


def test_fresh_heartbeat_extends_lease_but_commit_also_checks_expiry():
    now = [100.0]
    gate = FrontendAuthorityGate("session", clock=lambda: now[0])
    gate.bind("owner", 10)
    gate.observe("owner", INITIAL)
    gate.activate("owner", INITIAL)
    with gate.admission(OLD) as execution:
        wake = execution._wake = Mock()
    now[0] = 109
    assert gate.renew("owner", 1) == 1
    now[0] = 110
    execution.check()
    now[0] = 119
    effect = Mock()
    with pytest.raises(ExecutionAuthorityError):
        execution.commit(effect)
    effect.assert_not_called()
    wake.assert_called_once()


@pytest.mark.parametrize("duration", [True, 0, -1, float("inf"), float("nan"), None, "10"])
def test_invalid_lease_never_binds(duration):
    gate = FrontendAuthorityGate("session")
    with pytest.raises(ValueError):
        gate.bind("owner", duration)
    gate.bind("different-owner", 10)
    gate.close()


@pytest.mark.parametrize("operation", ["enqueue", "output"])
def test_seal_is_atomic_with_host_effects(gate, operation):
    entered, finish, sealing = Event(), Event(), Event()
    effects = []
    with gate.admission(OLD) as execution:
        pass

    def effect():
        entered.set()
        assert finish.wait(2)
        effects.append("effect")

    def perform():
        if operation == "output":
            execution.commit(effect)
        else:
            with gate.admission(OLD):
                effect()

    def seal():
        sealing.set()
        gate.seal("owner", REVISION)
        effects.append("sealed")

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(perform)
        assert entered.wait(2)
        second = pool.submit(seal)
        assert sealing.wait(2)
        assert not second.done()
        finish.set()
        first.result(2)
        second.result(2)
    assert effects == ["effect", "sealed"]
    with pytest.raises(ExecutionAuthorityError):
        execution.commit(lambda: effects.append("late"))
    rejects(gate, OLD)


def test_same_publication_retry_keeps_active_execution_and_release_is_terminal(gate):
    with gate.admission(OLD) as execution:
        wake = execution._wake = Mock()
    gate.observe("owner", INITIAL)
    execution.check()
    gate.release("owner")
    gate.release("owner")
    wake.assert_called_once()
    rejects(gate, OLD)
    with pytest.raises(ExecutionAuthorityError):
        gate.bind("new-owner", 60)


def test_host_seal_and_close_do_not_prove_resource_drain(gate):
    drain = RequestDrainTracker()
    drain.enqueue(OLD, lambda: 42)
    claim = drain.acquire_resource(42, OLD, "in-flight-transfer")
    with gate.admission(OLD) as execution:
        execution._cancel = lambda: drain.complete(42, OLD)
    gate.seal("owner", REVISION)
    assert not drain.drained(OLD)
    gate.release("owner")
    assert not drain.drained(OLD)
    drain.release_resource(claim)
    assert drain.drained(OLD)
