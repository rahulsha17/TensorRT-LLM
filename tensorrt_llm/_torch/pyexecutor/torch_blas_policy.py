# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own an explicitly requested Torch BLAS preference for a worker process."""

import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ...llmapi.llm_args import TorchLlmArgs

__all__ = [
    "initialize_torch_blas_policy",
    "mark_torch_blas_policy_failed",
    "validate_torch_blas_policy_scope",
    "verify_torch_blas_policy",
]

_Preference = Literal["cublaslt"] | None
_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_registered = False
_preference: _Preference = None
_failed = False
_protected: tuple | None = None


@dataclass(frozen=True)
class _Snapshot:
    pid: int
    thread_id: int
    phase: str
    preference: str
    bf16_pair: tuple[bool, bool]
    protected: tuple


def _after_fork() -> None:
    global _lock, _failed
    _lock = threading.Lock()
    # A fork inherits the parent's Torch state and potentially initialized
    # library handles. An explicitly configured parent requires a fresh spawn.
    _failed = _failed or (_registered and _preference is not None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _read(phase: str) -> _Snapshot:
    import torch

    try:
        cuda = torch.backends.cuda
        public = cuda.preferred_blas_library()
        native = torch._C._get_blas_preferred_backend()
        enum = torch._C._BlasBackend
        pair = torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
        public_pair = (
            cuda.matmul.allow_bf16_reduced_precision_reduction,
            cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        )
        protected = (
            cuda.matmul.allow_tf32,
            cuda.matmul.allow_fp16_reduced_precision_reduction,
            cuda.matmul.allow_fp16_reduced_precision_reduction_split_k,
            cuda.matmul.allow_fp16_accumulation,
            cuda.matmul.fp32_precision,
            torch.get_float32_matmul_precision(),
            torch.are_deterministic_algorithms_enabled(),
            cuda.cublas_workspace_size(),
            cuda.cublaslt_workspace_size(),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Installed Torch lacks the required BLAS policy readbacks") from error
    if type(public) is not enum or type(native) is not enum or public != native:
        raise RuntimeError("Torch public/native BLAS preferences disagree")
    if (
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(value) is not bool for value in pair + public_pair)
        or pair != public_pair
    ):
        raise RuntimeError("Torch public/native BF16 reduction pairs disagree")
    if pair != (True, True):
        raise RuntimeError(
            "Explicit BLAS preference currently requires the unchanged BF16 (True, True) pair"
        )
    return _Snapshot(
        os.getpid(),
        threading.get_ident(),
        phase,
        "cublaslt" if public == enum.Cublaslt else str(public),
        pair,
        protected,
    )


def validate_torch_blas_policy_scope(llm_args: "TorchLlmArgs") -> None:
    """Refuse unqualified worker topologies before assigning any Torch state."""
    if llm_args.torch_blas_preference is None:
        return
    if llm_args.torch_blas_preference != "cublaslt":
        raise ValueError("Only the explicit cublaslt Torch BLAS preference is supported")
    if (
        llm_args.parallel_config.world_size != 1
        or llm_args.orchestrator_type is not None
        or llm_args.torch_compile_config is not None
        or llm_args.speculative_config is not None
        or llm_args.mm_encoder_only
        or llm_args.kv_connector_config is not None
    ):
        raise ValueError(
            "Explicit Torch BLAS preference requires a local TP1 decoder without compilation, "
            "speculation, encoder-only execution or a KV connector"
        )


def initialize_torch_blas_policy(preference: _Preference) -> _Snapshot | None:
    """Assign once before model creation; None never imports or assigns Torch."""
    global _registered, _preference, _failed, _protected
    if preference not in (None, "cublaslt"):
        raise ValueError("Unsupported Torch BLAS preference")
    with _lock:
        if _failed:
            raise RuntimeError(
                "Torch BLAS policy initialization failed; start a fresh worker process"
            )
        if _registered and preference != _preference:
            raise RuntimeError("Conflicting explicit/unset Torch BLAS policies in one process")
        if preference is None:
            _registered = True
            return None
        if _registered:
            return _verify_locked(preference, "repeated_initialization")
        before = _read("before_assignment")
        _registered = True
        _preference = preference
        _protected = before.protected
        # Any exception after assignment begins leaves the process unusable
        # for another initialization. Never restore potentially live handles.
        _failed = True
        import torch

        torch.backends.cuda.preferred_blas_library(preference)
        after = _read("initialization")
        if after.preference != preference or after.protected != _protected:
            raise RuntimeError("Torch BLAS preference assignment changed unexpected state")
        _failed = False
        _logger.info("Torch BLAS policy: %s", after)
        return after


def _verify_locked(preference: _Preference, phase: str) -> _Snapshot:
    global _failed
    # Failed readbacks also retain a failed process record, even when their
    # caller catches the original error and attempts another initialization.
    _failed = True
    snapshot = _read(phase)
    if snapshot.preference != preference or snapshot.protected != _protected:
        raise RuntimeError("Torch BLAS policy changed after worker initialization")
    _failed = False
    _logger.info("Torch BLAS policy: %s", snapshot)
    return snapshot


def verify_torch_blas_policy(preference: _Preference, phase: str) -> _Snapshot | None:
    """Read policy on the actual warmup/execution thread without assigning it."""
    global _registered
    with _lock:
        if _failed:
            raise RuntimeError("Torch BLAS policy is failed; start a fresh worker process")
        if _registered and preference != _preference:
            raise RuntimeError("Conflicting explicit/unset Torch BLAS policies in one process")
        if preference is None:
            _registered = True
            return None
        if not _registered:
            raise RuntimeError("Torch BLAS policy must be initialized before model construction")
        return _verify_locked(preference, phase)


def mark_torch_blas_policy_failed(preference: _Preference) -> None:
    """Preserve explicit process ownership after a model-construction failure."""
    global _failed
    if preference is None:
        return
    with _lock:
        if _registered and preference == _preference:
            _failed = True
