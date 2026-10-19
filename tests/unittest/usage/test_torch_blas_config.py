# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-package configuration checks; require the unapplied BLAS option candidate."""

import json
import pickle

import pytest
import yaml
from pydantic import ValidationError

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.usage.llmapi_config import collect_llm_api_config_payloads, golden_manifest

pytestmark = pytest.mark.cpu_only

_BASE = {"model": "/unused/blas-config-test", "skip_tokenizer_init": True, "load_format": "dummy"}
_SERIALIZED_FIELDS = {*_BASE, "torch_blas_preference"}


@pytest.mark.parametrize("kwargs", [{}, {"torch_blas_preference": None}])
def test_omitted_and_explicit_none_preserve_the_default(kwargs: dict) -> None:
    args = TorchLlmArgs(**_BASE, **kwargs)
    assert args.torch_blas_preference is None
    assert args.model_dump()["torch_blas_preference"] is None


@pytest.mark.parametrize("invalid", ["default", "cublas", "hipblaslt", True, 1])
def test_invalid_preference_is_rejected_by_the_real_configuration(invalid) -> None:
    with pytest.raises(ValidationError, match="torch_blas_preference"):
        TorchLlmArgs(**_BASE, torch_blas_preference=invalid)


@pytest.mark.parametrize("preference", [None, "cublaslt"])
@pytest.mark.parametrize("format_name", ["json", "yaml"])
def test_preference_survives_user_configuration_serialization(preference, format_name: str) -> None:
    args = TorchLlmArgs(**_BASE, torch_blas_preference=preference)
    # The supported JSON representation supplies the YAML-safe scalar types.
    encoded = args.model_dump_json(include=_SERIALIZED_FIELDS)
    payload = json.loads(encoded)
    if format_name == "json":
        restored = TorchLlmArgs.model_validate_json(encoded)
    else:
        restored = TorchLlmArgs.model_validate(yaml.safe_load(yaml.safe_dump(payload)))
    assert restored.torch_blas_preference == preference
    assert restored.model_dump(mode="json", include=_SERIALIZED_FIELDS) == payload


@pytest.mark.parametrize("preference", [None, "cublaslt"])
def test_preference_survives_worker_argument_pickle_transport(preference) -> None:
    args = TorchLlmArgs(**_BASE, torch_blas_preference=preference)
    restored = pickle.loads(pickle.dumps(args))  # noqa: S301
    assert type(restored) is TorchLlmArgs
    assert restored.torch_blas_preference == preference
    assert restored.model_dump(mode="json", include=_SERIALIZED_FIELDS) == args.model_dump(
        mode="json", include=_SERIALIZED_FIELDS
    )


@pytest.mark.parametrize("preference", [None, "cublaslt"])
def test_real_telemetry_captures_only_the_declared_preference(preference) -> None:
    args = TorchLlmArgs(**_BASE, torch_blas_preference=preference)
    config_json, meta_json = collect_llm_api_config_payloads(args)
    config, meta = json.loads(config_json), json.loads(meta_json)
    assert config["torch_blas_preference"] == preference
    assert meta["capture_succeeded"] is True
    assert _BASE["model"] not in config_json


def test_complete_manifest_contains_the_exact_optional_literal_policy() -> None:
    entries = [
        row for row in golden_manifest()["TorchLlmArgs"] if row["path"] == "torch_blas_preference"
    ]
    assert entries == [
        {
            "allowed_values": ["cublaslt"],
            "capture_policy": "literal|none",
            "kind": "categorical",
            "path": "torch_blas_preference",
        }
    ]
