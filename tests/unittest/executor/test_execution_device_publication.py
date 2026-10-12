# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host publication ordering and failures, without CUDA or native bindings."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import pytest

from tensorrt_llm.execution_gate import ExecutionAuthorityError
from tensorrt_llm.execution_mailbox import VersionUpdate, WorkerVersionTable
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only
VERSION = ExecutionVersion("tenant", "scope", 7)
NEXT = replace(VERSION, epoch=8)


def test_receipt_requires_device_enqueue_but_not_device_completion() -> None:
    table = WorkerVersionTable("session")
    entered, release = Event(), Event()

    def publish(version):
        assert version == VERSION
        entered.set()
        assert release.wait(3)
        # Returning means the write was queued, not that CUDA completed it.

    publisher = Mock(publish=publish)
    table.bind_device_publisher(publisher)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(table.apply, VersionUpdate("session", 1, VERSION))
        try:
            assert entered.wait(3)
            assert not future.done()
        finally:
            release.set()
        assert future.result(3).observed_version == VERSION


def test_device_enqueue_failure_never_rewinds_or_acknowledges_a_retry() -> None:
    table = WorkerVersionTable("session")
    publisher = Mock()
    table.bind_device_publisher(publisher)
    table.apply(VersionUpdate("session", 1, VERSION))
    with table.admission(VERSION) as execution:
        pass
    publisher.publish.side_effect = RuntimeError("CUDA publication failed")
    with pytest.raises(RuntimeError, match="CUDA publication failed"):
        table.apply(VersionUpdate("session", 2, NEXT))
    assert table._versions[("tenant", "scope")].version == NEXT
    with pytest.raises(ExecutionAuthorityError):
        execution.check()
    for sequence, version in [(1, VERSION), (2, NEXT), (3, NEXT)]:
        with pytest.raises(RuntimeError, match="restart required"):
            table.apply(VersionUpdate("session", sequence, version))
    with pytest.raises(RuntimeError, match="restart required"):
        with table.guard_versions([None, VERSION]):
            pytest.fail("failed authority admitted a host commit")


def test_device_publisher_must_bind_once_before_publication() -> None:
    table = WorkerVersionTable("session")
    table.bind_device_publisher(Mock())
    with pytest.raises(RuntimeError):
        table.bind_device_publisher(Mock())
    initialized = WorkerVersionTable("session")
    initialized.apply(VersionUpdate("session", 1, VERSION))
    with pytest.raises(RuntimeError):
        initialized.bind_device_publisher(Mock())


def test_host_state_commit_orders_with_publication_and_preserves_other_scopes() -> None:
    table = WorkerVersionTable("session")
    table.apply(VersionUpdate("session", 1, VERSION))
    other = replace(VERSION, scope_id="other")
    table.apply(VersionUpdate("session", 2, other))
    started = Event()

    def publish():
        started.set()
        return table.apply(VersionUpdate("session", 3, NEXT))

    with ThreadPoolExecutor() as pool:
        with table.guard_versions([VERSION, other, None, NEXT]) as allowed:
            assert allowed == [True, True, True, False]
            future = pool.submit(publish)
            assert started.wait(3)
            assert not future.done()
        assert future.result(3).observed_version == NEXT
    with table.guard_versions([VERSION, other, None, NEXT]) as allowed:
        assert allowed == [False, True, True, True]
