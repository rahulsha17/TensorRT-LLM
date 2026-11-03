# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound revision-sensitive prefill work without changing request ownership."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....execution_version import ExecutionVersion
    from ..llm_request import LlmRequest


@dataclass
class _Progress:
    request: "LlmRequest"
    version: "ExecutionVersion"
    computed: int = 0
    start: int = 0
    end: int = 0


class RevisionSchedulingPolicy:
    """Limit young versions, then restore normal chunks after observed progress.

    Only scheduled chunks whose cursor subsequently advanced earn stability.
    Reused prefixes, a new version, and an allocation rewind earn no credit.
    All state is bounded by the executor's current request population.
    """

    def __init__(self, chunk_tokens: int, stable_tokens: int) -> None:
        if chunk_tokens <= 0 or stable_tokens < chunk_tokens:
            raise ValueError(
                "revision chunk tokens must be positive and no larger than stability tokens"
            )
        self.chunk_tokens = chunk_tokens
        self.stable_tokens = stable_tokens
        self._progress: dict[int, _Progress] = {}
        self._stable_next = False
        self._last_group: tuple[str, str, int] | None = None

    def refresh(self, requests: list["LlmRequest"]) -> None:
        current = {id(req): req for req in requests}
        self._progress = {
            key: value
            for key, value in self._progress.items()
            if current.get(key) is value.request
            and value.request.py_execution_version == value.version
        }

    def sensitive(self, request: "LlmRequest") -> bool:
        version = request.py_execution_version
        if version is None:
            return False
        progress = self._progress.get(id(request))
        if progress is None or progress.request is not request or progress.version != version:
            return True
        position = request.context_current_position
        if position < progress.start:
            progress.computed = 0
            progress.start = progress.end = position
        elif progress.end > progress.start and position >= progress.end:
            progress.computed += progress.end - progress.start
            progress.start = progress.end = position
        return progress.computed < self.stable_tokens

    def scheduled(self, request: "LlmRequest") -> None:
        version = request.py_execution_version
        if version is None:
            return
        self.sensitive(request)
        progress = self._progress.get(id(request))
        if progress is None or progress.request is not request or progress.version != version:
            progress = _Progress(request, version)
            self._progress[id(request)] = progress
        progress.start = request.context_current_position
        progress.end = progress.start + request.context_chunk_size

    def cohort(
        self, contexts: list["LlmRequest"], stable: list["LlmRequest"]
    ) -> tuple[set[int] | None, int | None]:
        """Alternate stable work with one rotating authority cohort.

        A sensitive pass has a total token cap, so an unrelated live request
        gets a scheduling turn after at most one small cohort. Resource
        admission can still defer it; this is not an admission-time bound.
        The caller retries normal scheduling if the chosen cohort cannot run.
        """
        groups: dict[tuple[str, str, int], list[LlmRequest]] = {}
        for request in contexts:
            if not self.sensitive(request):
                stable.append(request)
                continue
            version = request.py_execution_version
            assert version is not None
            key = (version.tenant_id, version.scope_id, version.epoch)
            groups.setdefault(key, []).append(request)
        if not groups:
            self._stable_next = False
            return None, None
        if stable and self._stable_next:
            self._stable_next = False
            return {id(req) for req in stable}, None
        keys = list(groups)
        index = (keys.index(self._last_group) + 1) % len(keys) if self._last_group in groups else 0
        self._last_group = keys[index]
        self._stable_next = bool(stable)
        return {id(req) for req in groups[keys[index]]}, self.chunk_tokens
