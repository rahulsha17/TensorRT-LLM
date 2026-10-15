# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculation contracts whose target KV can participate in RETIRE."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...llmapi.llm_args import DecodingBaseConfig


def is_private_ngram(config: "DecodingBaseConfig | None") -> bool:
    """N-gram proposals have no draft-model KV and remain local to a request."""
    return config is not None and config.decoding_type == "NGram" and config.is_public_pool is False
