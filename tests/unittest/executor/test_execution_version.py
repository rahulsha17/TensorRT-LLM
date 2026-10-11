# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from dataclasses import FrozenInstanceError, replace

import pytest

from tensorrt_llm.execution_version import ExecutionVersion, StateTag

pytestmark = pytest.mark.cpu_only


def test_identity_distinguishes_tenant_scope_and_epoch() -> None:
    version = ExecutionVersion("tenant", "scope", 7)
    identities = {
        version,
        replace(version, tenant_id="another-tenant"),
        replace(version, scope_id="another-scope"),
        replace(version, epoch=8),
    }
    assert len(identities) == 4
    assert ExecutionVersion("tenant", "scope", 7) in identities


@pytest.mark.parametrize("epoch", [0, (1 << 64) - 1])
def test_identity_and_state_survive_pickle(epoch: int) -> None:
    version = ExecutionVersion("tenant/α", "scope:β", epoch)
    state = StateTag(version, generation=19)
    assert pickle.loads(pickle.dumps(version)) == version
    assert pickle.loads(pickle.dumps(state)) == state


def test_identity_and_generation_are_immutable_and_independent() -> None:
    version = ExecutionVersion("tenant", "scope", 7)
    state = StateTag(version, generation=19)
    assert replace(state, generation=20).version == version
    assert replace(state, version=replace(version, epoch=8)).generation == 19
    with pytest.raises(FrozenInstanceError):
        version.epoch = 8
    with pytest.raises(FrozenInstanceError):
        state.generation = 20


@pytest.mark.parametrize("name", ["tenant_id", "scope_id"])
@pytest.mark.parametrize("value, error", [("", ValueError), (None, TypeError), (1, TypeError)])
def test_invalid_namespace_is_rejected(name: str, value, error: type[Exception]) -> None:
    fields = {"tenant_id": "tenant", "scope_id": "scope", "epoch": 0}
    fields[name] = value
    with pytest.raises(error, match=name):
        ExecutionVersion(**fields)


@pytest.mark.parametrize(
    "value, error",
    [
        (-1, ValueError),
        (1 << 64, ValueError),
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
    ],
)
def test_invalid_counters_are_rejected(value, error: type[Exception]) -> None:
    with pytest.raises(error, match="epoch"):
        ExecutionVersion("tenant", "scope", value)
    with pytest.raises(error, match="generation"):
        StateTag(ExecutionVersion("tenant", "scope", 0), value)


def test_state_requires_a_validated_execution_version() -> None:
    with pytest.raises(TypeError, match="version"):
        StateTag(("tenant", "scope", 0), 1)
