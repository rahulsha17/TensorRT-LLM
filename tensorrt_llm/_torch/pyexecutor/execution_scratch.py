# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded storage reuse for internal scratch after its last device user.

The caller serializes access. Returned output buffers, graph inputs and cached
metadata must never enter this pool: completion of one operation does not end
their consumers' ownership. A failed submission keeps its lease unavailable.
"""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Protocol


class Completion(Protocol):
    def query(self) -> bool: ...


@dataclass
class ScratchLease:
    key: Hashable
    storage: Any
    size: int
    reserved: bool = True
    event: Completion | None = None


class ExecutionScratchPool:
    def __init__(self, max_entries: int = 64, max_bytes: int = 4 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: list[ScratchLease] = []
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def acquire(
        self, key: Hashable, size: int, allocate: Callable[[], Any]
    ) -> tuple[Any, ScratchLease | None]:
        for entry in self.entries:
            if entry.key == key and not entry.reserved:
                assert entry.event is not None
                if entry.event.query():
                    entry.reserved = True
                    entry.event = None
                    self.hits += 1
                    return entry.storage, entry
        # Exhaustion never waits for another stream. The caller retains an
        # unpooled allocation through the ordinary pending-use lifetime.
        storage = allocate()
        self.misses += 1
        if len(self.entries) >= self.max_entries or size > self.max_bytes - self.bytes:
            return storage, None
        entry = ScratchLease(key, storage, size)
        self.entries.append(entry)
        self.bytes += size
        return storage, entry

    @staticmethod
    def submitted(leases: list[ScratchLease], event: Completion) -> None:
        # Only called after successful event recording. Never re-record an
        # event: a ticket or another entry can still refer to that observation.
        for entry in leases:
            assert entry.reserved and entry.event is None
            entry.event = event
            entry.reserved = False
