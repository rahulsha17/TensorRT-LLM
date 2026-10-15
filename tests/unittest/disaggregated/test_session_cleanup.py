# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A successful close must include cleanup, even with duplicate callers."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import RxSession, TxSession
from tensorrt_llm.disaggregated_params import DisaggregatedParams

pytestmark = pytest.mark.cpu_only


def _session(direction, owned):
    peer = Mock()
    peer._enforce_physical_ownership = owned
    aux = Mock()
    aux.alloc_slot.return_value = SimpleNamespace(id=7)
    kwargs = dict(
        request_id=41,
        params=DisaggregatedParams(disagg_request_id=41),
        aux_buffer=aux,
        prompt_len=32,
    )
    if direction == "send":
        session = TxSession(sender=peer, **kwargs)
    else:
        session = RxSession(receiver=peer, **kwargs)
    return session, peer, aux


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("owned", [False, True])
def test_duplicate_close_waits_for_cleanup(direction, owned):
    session, peer, aux = _session(direction, owned)
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    duplicate_started = threading.Event()
    duplicate_returned = threading.Event()

    def cleanup(_slot):
        cleanup_started.set()
        assert release_cleanup.wait(5)

    def duplicate_close():
        duplicate_started.set()
        try:
            return session.close()
        finally:
            duplicate_returned.set()

    aux.free_slot.side_effect = cleanup
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(session.close)
        try:
            assert cleanup_started.wait(5)
            assert session._closed
            second = pool.submit(duplicate_close)
            assert duplicate_started.wait(5)
            assert not duplicate_returned.wait(0.05)
            peer.clear_session.assert_not_called()
        finally:
            release_cleanup.set()
        assert first.result(timeout=5) is True
        assert second.result(timeout=5) is True
    aux.free_slot.assert_called_once_with(7)
    peer.clear_session.assert_called_once_with(41)


@pytest.mark.parametrize(
    "direction,stage",
    [
        ("send", "aux"),
        ("send", "unregister"),
        ("receive", "aux"),
        ("receive", "unregister"),
        ("receive", "bounce"),
    ],
)
@pytest.mark.parametrize("owned", [False, True])
def test_cleanup_failure_never_becomes_success_on_retry(direction, stage, owned):
    session, peer, aux = _session(direction, owned)
    failure = RuntimeError("cleanup failed after side effect")
    if stage == "aux":
        operation = aux.free_slot
    elif stage == "unregister":
        operation = peer.clear_session
    else:
        session._kv_tasks = [SimpleNamespace(slice_id=0, resources_drained=True)]
        operation = peer._bounce.orphan_reservation
    operation.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        session.close()
    assert caught.value is failure
    calls = (
        aux.free_slot.call_count,
        peer.clear_session.call_count,
        peer._bounce.orphan_reservation.call_count,
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cleanup is unproven") as caught:
            session.close()
        assert caught.value.__cause__ is failure
        assert calls == (
            aux.free_slot.call_count,
            peer.clear_session.call_count,
            peer._bounce.orphan_reservation.call_count,
        )


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_physical_drain_refusal_can_be_retried(direction):
    session, peer, aux = _session(direction, True)
    session.resources_drained = Mock(return_value=False)
    assert session.close() is False
    assert not session._closed
    aux.free_slot.assert_not_called()
    peer.clear_session.assert_not_called()
    session.resources_drained.return_value = True
    assert session.close() is True
    assert session.close() is True
    aux.free_slot.assert_called_once_with(7)
    peer.clear_session.assert_called_once_with(41)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_retirement_refusal_can_be_retried(direction):
    session, peer, aux = _session(direction, True)
    session.resources_drained = Mock(return_value=True)
    session._retirement = Mock()
    session._retirement.controller.admission_closed = False
    session._retirement.close.return_value = False
    assert session.close() is False
    aux.free_slot.assert_not_called()
    peer.clear_session.assert_not_called()
    session._retirement.close.return_value = True
    assert session.close() is True
    assert session.close() is True
    assert session._retirement.close.call_count == 2
    aux.free_slot.assert_called_once_with(7)
    peer.clear_session.assert_called_once_with(41)
