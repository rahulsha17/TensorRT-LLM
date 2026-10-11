# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side authority for runtime-issued execution revisions."""

from dataclasses import dataclass, field
from threading import Lock

from .execution_version import ExecutionVersion

__all__ = ["AdvanceReceipt", "Revision", "ScopeRegistry", "ScopeSnapshot"]


@dataclass(frozen=True)
class Revision:
    """Describe a runtime's proposed change of execution version.

    Args:
        revision_id: Runtime-issued retry ID, unique within the tenant and scope.
            It identifies one revision, independently of physical request IDs.
        expected_version: Version the runtime intends to replace.
        successor_version: Next version in the same tenant and scope. Its epoch
            must be exactly one greater than the expected epoch.

    This descriptor carries version metadata only. It neither supplies revised
    input nor grants permission to launch a successor.
    """

    revision_id: str
    expected_version: ExecutionVersion
    successor_version: ExecutionVersion

    def __post_init__(self) -> None:
        if not isinstance(self.revision_id, str):
            raise TypeError("revision_id must be a str")
        if not self.revision_id:
            raise ValueError("revision_id must not be empty")
        for name, version in (
            ("expected_version", self.expected_version),
            ("successor_version", self.successor_version),
        ):
            if not isinstance(version, ExecutionVersion):
                raise TypeError(f"{name} must be an ExecutionVersion")
        expected = self.expected_version
        successor = self.successor_version
        if (expected.tenant_id, expected.scope_id) != (successor.tenant_id, successor.scope_id):
            raise ValueError("a revision must stay within the same tenant and scope")
        if successor.epoch != expected.epoch + 1:
            raise ValueError("a revision must advance exactly one epoch")


@dataclass(frozen=True)
class AdvanceReceipt:
    """Record an accepted revision, without certifying a fence or safe KV reuse.

    Args:
        revision: The accepted descriptor. A receipt remains a historical record
            if a later revision advances the scope again.
    """

    revision: Revision


@dataclass(frozen=True)
class ScopeSnapshot:
    """Read a scope's authority and pending successor together.

    Args:
        current_version: Version selected by the most recent accepted revision,
            or by initial registration.
        pending_revision: Most recent accepted revision, or None before ADVANCE.
            Pending means that this registry has not launched the successor.

    A snapshot can become stale as soon as another revision is accepted. It is
    not a permit to perform work or publish output.
    """

    current_version: ExecutionVersion
    pending_revision: Revision | None


@dataclass
class _ScopeRecord:
    snapshot: ScopeSnapshot
    receipts: dict[str, AdvanceReceipt] = field(default_factory=dict)


class ScopeRegistry:
    """Serialize registration and ADVANCE for each tenant/scope pair.

    The runtime must route a scope's revisions through one shared registry
    instance. This is an in-memory, process-local authority, not a distributed
    service. It does not gate request admission, outputs, GPU work, or transfers.

    Records and retry receipts live for the lifetime of the registry. Scopes
    cannot be removed or reset here, so a late retry cannot recreate an old
    version. Recreating the registry is not a recovery protocol.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._scopes: dict[tuple[str, str], _ScopeRecord] = {}

    def register(self, version: ExecutionVersion) -> ScopeSnapshot:
        """Register an initial version, or confirm that it is still current.

        Args:
            version: Initial version chosen by the runtime for a new scope.

        Raises:
            ValueError: The scope already exists at a different version.
        """
        if not isinstance(version, ExecutionVersion):
            raise TypeError("version must be an ExecutionVersion")
        key = (version.tenant_id, version.scope_id)
        with self._lock:
            record = self._scopes.get(key)
            if record is None:
                record = _ScopeRecord(ScopeSnapshot(version, pending_revision=None))
                self._scopes[key] = record
            elif record.snapshot.current_version != version:
                raise ValueError(
                    f"scope {key!r} is already at epoch {record.snapshot.current_version.epoch}, "
                    f"not registration epoch {version.epoch}"
                )
            return record.snapshot

    def advance(self, revision: Revision) -> AdvanceReceipt:
        """Accept a revision only if its expected version is current.

        Args:
            revision: Runtime-issued transition metadata. An exact retry returns
                the original receipt, even after subsequent revisions. Reusing
                its ID with different metadata is an error.

        Returns:
            Receipt for the accepted transition. It acknowledges the host-side
            authority change only; the successor remains pending.

        Raises:
            KeyError: The tenant/scope pair has not been registered.
            ValueError: The revision ID conflicts with an accepted revision, or
                its expected version is no longer current.
        """
        if not isinstance(revision, Revision):
            raise TypeError("revision must be a Revision")
        expected = revision.expected_version
        with self._lock:
            record = self._get_record(expected.tenant_id, expected.scope_id)
            receipt = record.receipts.get(revision.revision_id)
            if receipt is not None:
                if receipt.revision != revision:
                    raise ValueError("revision_id conflicts with an accepted revision")
                return receipt
            if record.snapshot.current_version != expected:
                raise ValueError(
                    f"expected epoch {expected.epoch}, "
                    f"but current epoch is {record.snapshot.current_version.epoch}"
                )
            receipt = AdvanceReceipt(revision)
            snapshot = ScopeSnapshot(revision.successor_version, pending_revision=revision)
            record.receipts[revision.revision_id] = receipt
            # The authority and pending successor change together under the lock.
            record.snapshot = snapshot
            return receipt

    def snapshot(self, tenant_id: str, scope_id: str) -> ScopeSnapshot:
        """Return an immutable view of a registered scope.

        Args:
            tenant_id: Runtime's tenant namespace.
            scope_id: Scope within that tenant.

        Raises:
            KeyError: The tenant/scope pair has not been registered.
        """
        with self._lock:
            return self._get_record(tenant_id, scope_id).snapshot

    def is_current(self, version: ExecutionVersion) -> bool:
        """Check authority at this instant, without reserving it for later work.

        Args:
            version: Version to compare with the registered authority. Unknown
                scopes return False.
        """
        if not isinstance(version, ExecutionVersion):
            raise TypeError("version must be an ExecutionVersion")
        with self._lock:
            record = self._scopes.get((version.tenant_id, version.scope_id))
            return record is not None and record.snapshot.current_version == version

    def _get_record(self, tenant_id: str, scope_id: str) -> _ScopeRecord:
        # Callers hold self._lock.
        try:
            return self._scopes[(tenant_id, scope_id)]
        except KeyError:
            raise KeyError(f"unknown scope {(tenant_id, scope_id)!r}") from None
