# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable input identity for a versioned KV exchange.

This describes the KV that the exchange is expected to contain. It is not a
checksum of device memory, a completion receipt, or permission to resume work.
"""

import hashlib
import struct
from dataclasses import dataclass


def _sequence_digest(values: tuple[int, ...]) -> bytes:
    digest = hashlib.sha256(b"TRTLLM_KV_INPUT_V1\0")
    digest.update(struct.pack("<Q", len(values)))
    for value in values:
        if type(value) is not int or not 0 <= value < (1 << 31):
            raise ValueError("KV input values must be nonnegative int32 integers")
        digest.update(struct.pack("<I", value))
    return digest.digest()


@dataclass(frozen=True)
class TransferPrefixIdentity:
    """Expected prompt prefix in one immutable model/cache context.

    cache_context_id is supplied by the runtime, shared by compatible workers,
    and must change when model weights, positional semantics, adapter meanings,
    or KV numerical format change. It must not be a mutable model name or a
    worker address. Physical TP/PP layouts remain subject to peer validation;
    different physical layouts do not imply different logical cache contexts.
    """

    cache_context_id: str
    num_tokens: int
    token_digest: bytes
    position_digest: bytes
    lora_task_id: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.cache_context_id, str) or not self.cache_context_id:
            raise ValueError("tagged KV transfer requires an immutable cache context ID")
        if type(self.num_tokens) is not int or not 0 < self.num_tokens < (1 << 31):
            raise ValueError("KV prefix length must be a positive int32 integer")
        for digest in (self.token_digest, self.position_digest):
            if type(digest) is not bytes or len(digest) != 32:
                raise ValueError("KV prefix digests must contain exactly 32 immutable bytes")
        if self.lora_task_id is not None and (
            type(self.lora_task_id) is not int or not 0 <= self.lora_task_id < (1 << 64)
        ):
            raise ValueError("KV LoRA identity must be a uint64 integer or None")

    @classmethod
    def from_tokens(
        cls,
        cache_context_id: str,
        tokens: tuple[int, ...],
        *,
        position_ids: tuple[int, ...] | None = None,
        lora_task_id: int | None = None,
    ) -> "TransferPrefixIdentity":
        if not isinstance(tokens, tuple) or not tokens:
            raise ValueError("KV prefix tokens must be a nonempty tuple")
        if position_ids is None:
            position_ids = tuple(range(len(tokens)))
        if not isinstance(position_ids, tuple) or len(position_ids) != len(tokens):
            raise ValueError("KV prefix positions must match the token tuple")
        return cls(
            cache_context_id,
            len(tokens),
            _sequence_digest(tokens),
            _sequence_digest(position_ids),
            lora_task_id,
        )

    def to_values(self) -> list:
        return [
            self.cache_context_id,
            self.num_tokens,
            self.token_digest,
            self.position_digest,
            self.lora_task_id,
        ]

    @classmethod
    def from_values(cls, values: list) -> "TransferPrefixIdentity":
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError("invalid KV prefix identity schema")
        return cls(*values)
