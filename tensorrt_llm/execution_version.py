# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-supplied identities for revisable execution."""

from dataclasses import dataclass

__all__ = ["ExecutionVersion", "StateTag"]


def _validate_counter(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if not 0 <= value < (1 << 64):
        raise ValueError(f"{name} must be in [0, 2**64)")


@dataclass(frozen=True)
class ExecutionVersion:
    """Identify one execution version within a runtime-owned scope.

    The runtime chooses the scope and advances its epoch when work is revised.
    Several requests may belong to the same version. This tag carries identity;
    it does not establish authority or validate a transition between epochs.

    Args:
        tenant_id: Namespace within which the runtime assigns unique scope IDs.
        scope_id: Work that the runtime revises together.
        epoch: Nonnegative 64-bit version number within that scope.
    """

    tenant_id: str
    scope_id: str
    epoch: int

    def __post_init__(self) -> None:
        for name, value in (("tenant_id", self.tenant_id), ("scope_id", self.scope_id)):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str")
            if not value:
                raise ValueError(f"{name} must not be empty")
        _validate_counter("epoch", self.epoch)


@dataclass(frozen=True)
class StateTag:
    """Identify state associated with a version and an allocation generation.

    Args:
        version: Execution version associated with the state.
        generation: Nonnegative 64-bit allocation generation, assigned by the
            resource owner. It is independent of the execution epoch.

    A tag alone does not retain memory or certify that KV can be reused.
    """

    version: ExecutionVersion
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.version, ExecutionVersion):
            raise TypeError("version must be an ExecutionVersion")
        _validate_counter("generation", self.generation)
