# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier

import pytest

from tensorrt_llm.execution_authority import AdvanceReceipt, Revision, ScopeRegistry
from tensorrt_llm.execution_version import ExecutionVersion

pytestmark = pytest.mark.cpu_only


def test_advance_selects_pending_successor_and_preserves_old_snapshot() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 7)
    initial = registry.register(original)
    successor = replace(original, epoch=8)
    revision = Revision("edit-1", original, successor)

    receipt = registry.advance(revision)
    snapshot = registry.snapshot("tenant", "scope")

    assert receipt.revision == revision
    assert snapshot.current_version == successor
    assert snapshot.pending_revision == revision
    assert registry.is_current(successor)
    assert not registry.is_current(original)
    assert initial.current_version == original
    assert initial.pending_revision is None


def test_registration_cannot_reset_or_bypass_authority() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    initial = registry.register(original)
    assert registry.register(original) == initial

    successor = replace(original, epoch=1)
    registry.advance(Revision("edit-1", original, successor))
    pending = registry.snapshot("tenant", "scope")
    assert registry.register(successor) == pending

    for version in (original, replace(original, epoch=2)):
        with pytest.raises(ValueError, match="registration epoch"):
            registry.register(version)
        assert registry.snapshot("tenant", "scope") == pending


def test_unknown_scope_fails_closed_without_registering_it() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    assert not registry.is_current(original)
    with pytest.raises(KeyError, match="unknown scope"):
        registry.advance(Revision("edit-1", original, replace(original, epoch=1)))
    with pytest.raises(KeyError, match="unknown scope"):
        registry.snapshot("tenant", "scope")

    initial = registry.register(original)
    assert initial.current_version == original
    assert initial.pending_revision is None


def test_exact_retry_returns_historical_receipt_without_changing_latest_pending_revision() -> None:
    registry = ScopeRegistry()
    version_0 = ExecutionVersion("tenant", "scope", 0)
    version_1 = replace(version_0, epoch=1)
    version_2 = replace(version_0, epoch=2)
    registry.register(version_0)

    revision_1 = Revision("edit-1", version_0, version_1)
    receipt_1 = registry.advance(revision_1)
    assert registry.advance(pickle.loads(pickle.dumps(revision_1))) == receipt_1
    revision_2 = Revision("edit-2", version_1, version_2)
    receipt_2 = registry.advance(revision_2)
    latest = registry.snapshot("tenant", "scope")

    assert registry.advance(revision_1) == receipt_1
    assert registry.advance(revision_2) == receipt_2
    assert registry.snapshot("tenant", "scope") == latest
    assert latest.current_version == version_2
    assert latest.pending_revision == revision_2
    assert not registry.is_current(receipt_1.revision.successor_version)


def test_reusing_revision_id_for_another_transition_is_rejected() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    successor = replace(original, epoch=1)
    registry.register(original)
    registry.advance(Revision("edit-1", original, successor))
    before = registry.snapshot("tenant", "scope")

    conflict = Revision("edit-1", successor, replace(original, epoch=2))
    with pytest.raises(ValueError, match="revision_id conflicts"):
        registry.advance(conflict)
    assert registry.snapshot("tenant", "scope") == before


def test_different_revision_id_is_not_an_exact_retry() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    registry.register(original)
    revision = Revision("edit-1", original, replace(original, epoch=1))
    registry.advance(revision)
    before = registry.snapshot("tenant", "scope")

    with pytest.raises(ValueError, match="expected epoch 0, but current epoch is 1"):
        registry.advance(replace(revision, revision_id="competing-edit"))
    assert registry.snapshot("tenant", "scope") == before


def test_future_revision_is_rejected_without_consuming_its_retry_id() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    successor = replace(original, epoch=1)
    initial = registry.register(original)
    future = Revision("edit-2", successor, replace(original, epoch=2))

    with pytest.raises(ValueError, match="expected epoch 1, but current epoch is 0"):
        registry.advance(future)
    assert registry.snapshot("tenant", "scope") == initial

    registry.advance(Revision("edit-1", original, successor))
    assert registry.advance(future).revision == future


@pytest.mark.parametrize(
    "other",
    [ExecutionVersion("other-tenant", "scope", 0), ExecutionVersion("tenant", "other-scope", 0)],
)
def test_authority_and_retry_ids_are_isolated_by_tenant_and_scope(other: ExecutionVersion) -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    registry.register(original)
    other_initial = registry.register(other)

    first = Revision("same-retry-id", original, replace(original, epoch=1))
    registry.advance(first)
    assert registry.is_current(other)
    assert registry.snapshot(other.tenant_id, other.scope_id) == other_initial

    second = Revision("same-retry-id", other, replace(other, epoch=1))
    assert registry.advance(second).revision == second
    assert registry.snapshot("tenant", "scope").pending_revision == first


def test_concurrent_competing_revisions_have_exactly_one_winner() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    registry.register(original)
    successor = replace(original, epoch=1)
    contenders = 8
    start = Barrier(contenders)

    def attempt(index: int) -> AdvanceReceipt | ValueError:
        revision = Revision(f"edit-{index}", original, successor)
        start.wait(timeout=10)
        try:
            return registry.advance(revision)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=contenders) as pool:
        results = list(pool.map(attempt, range(contenders)))

    receipts = [result for result in results if isinstance(result, AdvanceReceipt)]
    rejections = [result for result in results if isinstance(result, ValueError)]
    assert len(receipts) == 1
    assert len(rejections) == contenders - 1
    assert all("expected epoch 0, but current epoch is 1" in str(error) for error in rejections)
    snapshot = registry.snapshot("tenant", "scope")
    assert snapshot.current_version == successor
    assert snapshot.pending_revision == receipts[0].revision


def test_concurrent_exact_retries_advance_only_once() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    registry.register(original)
    revision = Revision("edit-1", original, replace(original, epoch=1))
    retries = 8
    start = Barrier(retries)

    def retry() -> AdvanceReceipt:
        copy = pickle.loads(pickle.dumps(revision))
        start.wait(timeout=10)
        return registry.advance(copy)

    with ThreadPoolExecutor(max_workers=retries) as pool:
        futures = [pool.submit(retry) for _ in range(retries)]
        receipts = [future.result(timeout=10) for future in futures]

    assert all(receipt == receipts[0] for receipt in receipts)
    assert registry.snapshot("tenant", "scope").current_version.epoch == 1


def test_registration_racing_with_advance_cannot_restore_old_version() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    registry.register(original)
    revision = Revision("edit-1", original, replace(original, epoch=1))
    start = Barrier(2)

    def register_again() -> None:
        start.wait(timeout=10)
        try:
            registry.register(original)
        except ValueError as error:
            assert "registration epoch" in str(error)

    def advance() -> AdvanceReceipt:
        start.wait(timeout=10)
        return registry.advance(revision)

    with ThreadPoolExecutor(max_workers=2) as pool:
        registration = pool.submit(register_again)
        advancement = pool.submit(advance)
        registration.result(timeout=10)
        assert advancement.result(timeout=10).revision == revision

    assert registry.snapshot("tenant", "scope").current_version == revision.successor_version


def test_accepted_metadata_is_immutable_and_serializable() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", 0)
    registry.register(original)
    revision = Revision("edit-1", original, replace(original, epoch=1))
    receipt = registry.advance(revision)
    snapshot = registry.snapshot("tenant", "scope")

    for value in (revision, receipt, snapshot):
        assert pickle.loads(pickle.dumps(value)) == value
    with pytest.raises(FrozenInstanceError):
        revision.revision_id = "another-edit"
    with pytest.raises(FrozenInstanceError):
        receipt.revision = None
    with pytest.raises(FrozenInstanceError):
        snapshot.current_version = original
    with pytest.raises(FrozenInstanceError):
        snapshot.pending_revision = None
    assert registry.snapshot("tenant", "scope") == snapshot


@pytest.mark.parametrize(
    "successor",
    [
        ExecutionVersion("other-tenant", "scope", 8),
        ExecutionVersion("tenant", "other-scope", 8),
        ExecutionVersion("tenant", "scope", 6),
        ExecutionVersion("tenant", "scope", 7),
        ExecutionVersion("tenant", "scope", 9),
    ],
)
def test_revision_cannot_cross_namespaces_skip_or_rewind_epochs(
    successor: ExecutionVersion,
) -> None:
    with pytest.raises(ValueError):
        Revision("edit-1", ExecutionVersion("tenant", "scope", 7), successor)


def test_epoch_exhaustion_never_wraps_to_an_old_version() -> None:
    registry = ScopeRegistry()
    original = ExecutionVersion("tenant", "scope", (1 << 64) - 2)
    registry.register(original)
    last = replace(original, epoch=(1 << 64) - 1)
    revision = Revision("last-edit", original, last)
    registry.advance(revision)

    with pytest.raises(ValueError, match="advance exactly one epoch"):
        Revision("wraparound", last, replace(last, epoch=0))
    with pytest.raises(ValueError, match="epoch"):
        replace(last, epoch=1 << 64)
    assert registry.is_current(last)
    assert registry.advance(revision).revision == revision


@pytest.mark.parametrize(
    "revision_id, error", [("", ValueError), (None, TypeError), (1, TypeError)]
)
def test_revision_requires_a_nonempty_retry_id(revision_id, error: type[Exception]) -> None:
    original = ExecutionVersion("tenant", "scope", 0)
    with pytest.raises(error, match="revision_id"):
        Revision(revision_id, original, replace(original, epoch=1))


@pytest.mark.parametrize("field", ["expected_version", "successor_version"])
def test_revision_requires_validated_versions(field: str) -> None:
    original = ExecutionVersion("tenant", "scope", 0)
    revision = Revision("edit-1", original, replace(original, epoch=1))
    with pytest.raises(TypeError, match=field):
        replace(revision, **{field: ("tenant", "scope", 0)})


def test_registry_rejects_untyped_descriptors() -> None:
    registry = ScopeRegistry()
    with pytest.raises(TypeError, match="version"):
        registry.register(("tenant", "scope", 0))
    with pytest.raises(TypeError, match="version"):
        registry.is_current(("tenant", "scope", 0))
    with pytest.raises(TypeError, match="revision"):
        registry.advance({"revision_id": "edit-1"})
