# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authority polling deadlines, health, backoff and publication lock regressions."""

from collections.abc import Callable
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import execution_device as device

pytestmark = pytest.mark.cpu_only


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert 0 < seconds <= 0.001
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class Completion:
    def __init__(self, clock: Clock, ready: float) -> None:
        self.clock = clock
        self.ready = ready
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.clock.now >= self.ready


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(device, "time", value)
    return value


@pytest.fixture
def authority() -> device.DeviceExecutionAuthority:
    value = object.__new__(device.DeviceExecutionAuthority)
    value._failure = None
    value._lock = RLock()
    value._pending = []
    return value


@pytest.mark.parametrize("timeout", [-1.0, 0.0, 5.0])
def test_ready_query_precedes_deadline_and_never_sleeps(
    authority: device.DeviceExecutionAuthority, clock: Clock, timeout: float
) -> None:
    event = Completion(clock, 0)
    authority._wait(device._PendingUse([], event), timeout)
    assert event.queries == 1 and clock.sleeps == [] and authority._failure is None


def test_short_completion_observed_before_fixed_poll_interval(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    event = Completion(clock, 0.00005)
    authority.wait_event(event)
    assert clock.now == pytest.approx(0.00007)
    assert clock.sleeps == [0.00001, 0.00002, 0.00004]
    assert event.queries == 4


def test_backoff_cap_and_restart_for_each_wait(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    authority.wait_event(Completion(clock, 0.004))
    assert clock.sleeps == [
        0.00001,
        0.00002,
        0.00004,
        0.00008,
        0.00016,
        0.00032,
        0.00064,
        0.001,
        0.001,
        0.001,
    ]
    clock.sleeps.clear()
    authority.wait_event(Completion(clock, clock.now + 0.000005))
    assert clock.sleeps == [0.00001]


def test_fixed_interval_boundary_can_return_earlier(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    authority.wait_event(Completion(clock, 0.001))
    # A fixed 1 ms policy observes this exact boundary at its first retry.
    assert clock.now == pytest.approx(0.00127)
    assert clock.now > 0.001


def test_absolute_default_deadline_and_sticky_timeout(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    event = Completion(clock, 99)
    with pytest.raises(TimeoutError, match="did not complete") as error:
        authority._wait(device._PendingUse([], event))
    assert clock.now == pytest.approx(5.0)
    assert authority._failure is error.value
    with pytest.raises(RuntimeError, match="restart required") as later:
        authority._healthy()
    assert later.value.__cause__ is error.value


def test_last_sleep_is_clamped_to_absolute_deadline(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    with pytest.raises(TimeoutError):
        authority._wait(device._PendingUse([], Completion(clock, 99)), timeout=0.000055)
    assert clock.now == pytest.approx(0.000055)
    assert clock.sleeps == pytest.approx([0.00001, 0.00002, 0.000025])


def test_query_error_identity_and_pending_ownership_preserved(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    failure = RuntimeError("query failed")

    def query() -> bool:
        raise failure

    marker = object()
    use = device._PendingUse([marker], SimpleNamespace(query=query))
    authority._pending.append(use)
    with pytest.raises(RuntimeError) as error:
        authority._wait(use)
    assert error.value is authority._failure is failure
    assert authority._pending == [use] and use.tensors[0] is marker
    assert clock.sleeps == []


def test_health_rechecked_after_successful_query(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    failure = RuntimeError("concurrent failure")

    def query() -> bool:
        authority._failure = failure
        return True

    with pytest.raises(RuntimeError, match="restart required") as error:
        authority.wait_event(SimpleNamespace(query=query))
    assert error.value.__cause__ is failure and clock.sleeps == []


def test_health_rechecked_after_sleep(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    failure = RuntimeError("concurrent failure")

    def poison() -> None:
        authority._failure = failure

    clock.on_sleep = poison
    with pytest.raises(RuntimeError, match="restart required") as error:
        authority.wait_event(Completion(clock, 99))
    assert error.value.__cause__ is failure and len(clock.sleeps) == 1


def test_success_query_after_clock_deadline_keeps_original_order(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    def query() -> bool:
        clock.now = 6.0
        return True

    authority.wait_event(SimpleNamespace(query=query))
    assert clock.sleeps == [] and authority._failure is None


def test_sleep_error_propagates_and_poisons(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    failure = RuntimeError("sleep failed")

    def fail() -> None:
        raise failure

    clock.on_sleep = fail
    with pytest.raises(RuntimeError) as error:
        authority.wait_event(Completion(clock, 99))
    assert error.value is authority._failure is failure


def test_missing_event_remains_fail_closed(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    with pytest.raises(AssertionError) as error:
        authority._wait(device._PendingUse([]))
    assert authority._failure is error.value and clock.sleeps == []


def test_wait_event_preserves_caller_exception(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    failure = KeyboardInterrupt("caller cancellation")

    def query() -> bool:
        raise failure

    with pytest.raises(KeyboardInterrupt) as error:
        authority.wait_event(SimpleNamespace(query=query))
    assert error.value is authority._failure is failure


def test_publisher_lock_is_available_during_wait(
    authority: device.DeviceExecutionAuthority, clock: Clock
) -> None:
    acquired = Event()
    threads: list[Thread] = []

    def publish() -> None:
        with authority._lock:
            acquired.set()

    def after_sleep() -> None:
        thread = Thread(target=publish, daemon=True)
        threads.append(thread)
        thread.start()
        assert acquired.wait(1), "wait held the publisher lock"

    clock.on_sleep = after_sleep
    authority.wait_event(SimpleNamespace(query=acquired.is_set))
    for thread in threads:
        thread.join(1)
        assert not thread.is_alive()
    assert len(clock.sleeps) == 1 and authority._failure is None
