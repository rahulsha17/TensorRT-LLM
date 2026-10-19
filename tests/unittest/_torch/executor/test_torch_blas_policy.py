# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks with a synthetic Torch API; no CUDA or proxy-worker qualification."""

import ast
import contextlib
import copy
import importlib.util
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[4]
EXECUTOR = ROOT / "tensorrt_llm/_torch/pyexecutor"


class Backend(Enum):
    Cublas = 0
    Cublaslt = 1


class FakeTorch:
    def __init__(self):
        self.backend = Backend.Cublas
        self.pair = (True, True)
        self.assignments = []
        self.on_set = None
        matmul = SimpleNamespace(
            allow_bf16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction_split_k=True,
            allow_tf32=True,
            allow_fp16_reduced_precision_reduction=True,
            allow_fp16_reduced_precision_reduction_split_k=True,
            allow_fp16_accumulation=False,
            fp32_precision="tf32",
        )
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(
                preferred_blas_library=self.preferred,
                matmul=matmul,
                cublas_workspace_size=lambda: 33554432,
                cublaslt_workspace_size=lambda: 1048576,
            )
        )
        self._C = SimpleNamespace(
            _BlasBackend=Backend,
            _get_blas_preferred_backend=lambda: self.backend,
            _get_cublas_allow_bf16_reduced_precision_reduction=lambda: self.pair,
        )

    def preferred(self, value=None):
        if value is not None:
            assert value == "cublaslt"
            self.assignments.append(value)
            self.backend = Backend.Cublaslt
            if self.on_set:
                self.on_set()
        return self.backend

    def get_float32_matmul_precision(self):
        return "high"

    def are_deterministic_algorithms_enabled(self):
        return False


@pytest.fixture
def policy(monkeypatch):
    name = "candidate_torch_blas_policy"
    spec = importlib.util.spec_from_file_location(name, EXECUTOR / "torch_blas_policy.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def torch_api(monkeypatch):
    value = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", value)
    return value


def args(preference="cublaslt", **changes):
    value = SimpleNamespace(
        torch_blas_preference=preference,
        parallel_config=SimpleNamespace(world_size=1),
        orchestrator_type=None,
        torch_compile_config=None,
        speculative_config=None,
        mm_encoder_only=False,
        kv_connector_config=None,
    )
    for name, item in changes.items():
        setattr(value, name, item)
    return value


def test_unset_never_imports_or_reads_torch(policy, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert policy.initialize_torch_blas_policy(None) is None
    assert policy.verify_torch_blas_policy(None, "warmup") is None
    policy.mark_torch_blas_policy_failed(None)
    assert policy.initialize_torch_blas_policy(None) is None


def test_assignment_once_preserves_pair_and_protected_preferences(policy, torch_api):
    before = copy.deepcopy(vars(torch_api.backends.cuda.matmul))
    snapshot = policy.initialize_torch_blas_policy("cublaslt")
    assert (snapshot.pid, snapshot.thread_id) == (os.getpid(), threading.get_ident())
    assert snapshot.preference == "cublaslt" and snapshot.bf16_pair == (True, True)
    for phase in ("resolved_config", "warmup", "execution_thread"):
        assert policy.verify_torch_blas_policy("cublaslt", phase).phase == phase
    policy.initialize_torch_blas_policy("cublaslt")
    assert torch_api.assignments == ["cublaslt"]
    assert vars(torch_api.backends.cuda.matmul) == before


@pytest.mark.parametrize("first,second", [(None, "cublaslt"), ("cublaslt", None)])
def test_mixed_process_intents_are_rejected(policy, torch_api, first, second):
    policy.initialize_torch_blas_policy(first)
    count = len(torch_api.assignments)
    with pytest.raises(RuntimeError, match="Conflicting"):
        policy.initialize_torch_blas_policy(second)
    assert len(torch_api.assignments) == count


@pytest.mark.parametrize("value", ["default", "cublas", "hipblaslt", True, 1, []])
def test_unknown_direct_preferences_are_rejected_before_torch(policy, monkeypatch, value):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="Unsupported"):
        policy.initialize_torch_blas_policy(value)


@pytest.mark.parametrize(
    "attack", ["missing", "native_disagreement", "pair_bool", "pair_integer", "pair_other"]
)
def test_unsupported_api_or_pair_fails_before_assignment(policy, torch_api, attack):
    if attack == "missing":
        del torch_api._C._get_blas_preferred_backend
    elif attack == "native_disagreement":
        torch_api._C._get_blas_preferred_backend = lambda: Backend.Cublaslt
    elif attack == "pair_bool":
        torch_api.pair = True
    elif attack == "pair_integer":
        torch_api.pair = (1, True)
    else:
        torch_api.pair = (False, True)
        torch_api.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    with pytest.raises(RuntimeError):
        policy.initialize_torch_blas_policy("cublaslt")
    assert torch_api.assignments == []


@pytest.mark.parametrize("attack", ["setter", "backend", "pair", "protected"])
def test_failed_assignment_poisoned_without_rollback_or_retry(policy, torch_api, attack):
    def mutate():
        if attack == "setter":
            raise RuntimeError("synthetic setter failure")
        if attack == "backend":
            torch_api.backend = Backend.Cublas
        elif attack == "pair":
            torch_api.pair = (False, True)
        else:
            torch_api.backends.cuda.matmul.allow_tf32 = False

    torch_api.on_set = mutate
    with pytest.raises(RuntimeError):
        policy.initialize_torch_blas_policy("cublaslt")
    with pytest.raises(RuntimeError, match="fresh worker"):
        policy.initialize_torch_blas_policy("cublaslt")
    assert torch_api.assignments == ["cublaslt"]


def test_constructor_failure_keeps_process_ownership(policy, torch_api):
    policy.initialize_torch_blas_policy("cublaslt")
    policy.mark_torch_blas_policy_failed("cublaslt")
    with pytest.raises(RuntimeError, match="failed"):
        policy.verify_torch_blas_policy("cublaslt", "warmup")
    assert torch_api.backend is Backend.Cublaslt and torch_api.assignments == ["cublaslt"]


def test_external_drift_detected_on_actual_worker_thread(policy, torch_api):
    policy.initialize_torch_blas_policy("cublaslt")
    with ThreadPoolExecutor(max_workers=1) as workers:
        snapshot = workers.submit(
            policy.verify_torch_blas_policy, "cublaslt", "execution_thread"
        ).result()
        assert snapshot.thread_id != threading.get_ident() and snapshot.pid == os.getpid()
        torch_api.backend = Backend.Cublas
        with pytest.raises(RuntimeError, match="changed"):
            workers.submit(policy.verify_torch_blas_policy, "cublaslt", "execution_thread").result()
    with pytest.raises(RuntimeError, match="fresh worker"):
        policy.initialize_torch_blas_policy("cublaslt")
    assert torch_api.assignments == ["cublaslt"]


def test_concurrent_initializers_assign_once(policy, torch_api):
    barrier = threading.Barrier(4)

    def initialize():
        barrier.wait()
        return policy.initialize_torch_blas_policy("cublaslt")

    with ThreadPoolExecutor(max_workers=4) as workers:
        snapshots = list(workers.map(lambda _: initialize(), range(4)))
    assert len({s.thread_id for s in snapshots}) == 4
    assert torch_api.assignments == ["cublaslt"]


def test_uninitialized_explicit_readback_refused(policy, torch_api):
    with pytest.raises(RuntimeError, match="before model construction"):
        policy.verify_torch_blas_policy("cublaslt", "warmup")
    assert torch_api.assignments == []


@pytest.mark.parametrize("preference", [None, "cublaslt"])
def test_fork_callback_preserves_explicit_failure(policy, torch_api, preference):
    policy.initialize_torch_blas_policy(preference)
    policy._after_fork()
    if preference is None:
        assert policy.initialize_torch_blas_policy(None) is None
    else:
        with pytest.raises(RuntimeError, match="fresh worker"):
            policy.initialize_torch_blas_policy(preference)


@pytest.mark.parametrize(
    "change",
    [
        dict(parallel_config=SimpleNamespace(world_size=2)),
        dict(orchestrator_type="ray"),
        dict(orchestrator_type="rpc"),
        dict(torch_compile_config=object()),
        dict(speculative_config=object()),
        dict(mm_encoder_only=True),
        dict(kv_connector_config=object()),
    ],
)
def test_scope_rejection_has_no_assignment(policy, torch_api, change):
    with pytest.raises(ValueError, match="local TP1"):
        policy.validate_torch_blas_policy_scope(args(**change))
    policy.validate_torch_blas_policy_scope(args(None, **change))
    assert torch_api.assignments == []


def function(path, name, namespace, owner=None):
    parsed = ast.parse(path.read_text())
    body = (
        next(n.body for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == owner)
        if owner
        else parsed.body
    )
    node = copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    if node.args.kwarg:
        node.args.kwarg.annotation = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("fail", [False, True])
def test_actual_creator_wrapper_orders_policy_before_engine_and_retains_failure(
    policy, torch_api, fail
):
    events = []

    def implementation(**kwargs):
        events.append("engine")
        assert torch_api.assignments == ["cublaslt"]
        if fail:
            raise RuntimeError("synthetic model constructor failure")
        return "executor"

    ns = dict(
        get_global_dwdp_manager=lambda: None,
        validate_torch_blas_policy_scope=policy.validate_torch_blas_policy_scope,
        initialize_torch_blas_policy=policy.initialize_torch_blas_policy,
        mark_torch_blas_policy_failed=policy.mark_torch_blas_policy_failed,
        monitor_executor_initialization=contextlib.nullcontext,
        _StartupTimer=lambda *a: contextlib.nullcontext(),
        _create_py_executor_impl=implementation,
    )
    create = function(EXECUTOR / "py_executor_creator.py", "create_py_executor", ns)
    if fail:
        with pytest.raises(RuntimeError, match="constructor failure"):
            create(args())
        with pytest.raises(RuntimeError, match="fresh worker"):
            create(args())
    else:
        assert create(args()) == "executor"
    assert events == ["engine"]


def config_class():
    tree = ast.parse((ROOT / "tensorrt_llm/llmapi/llm_args.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TorchLlmArgs")
    field = copy.deepcopy(
        next(
            n
            for n in owner.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "torch_blas_preference"
        )
    )

    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")

    node = ast.ClassDef(
        name="CandidateConfig",
        bases=[ast.Name(id="Strict", ctx=ast.Load())],
        keywords=[],
        body=[field],
        decorator_list=[],
    )
    actual_field = function(
        ROOT / "tensorrt_llm/llmapi/llm_args.py", "Field", dict(PydanticField=Field)
    )
    ns = dict(Strict=Strict, Optional=Optional, Literal=Literal, Field=actual_field)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "candidate-field",
            "exec",
        ),
        ns,
    )
    return ns["CandidateConfig"]


@pytest.mark.parametrize("resolved", [None, "torch_compile_config", "speculative_config"])
def test_actual_resolved_config_rechecks_scope_before_engine(policy, torch_api, resolved):
    policy.initialize_torch_blas_policy("cublaslt")
    tree = ast.parse((EXECUTOR / "py_executor_creator.py").read_text())
    create = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_create_py_executor"
    )
    end = next(
        i
        for i, n in enumerate(create.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "verify_torch_blas_policy"
    )
    block = copy.deepcopy(create.body[: end + 1])
    events = []

    def load(value, checkpoint_dir):
        events.append("config")
        if resolved is not None:
            setattr(value, resolved, object())
        return value, None

    def verify(preference, phase):
        events.append(phase)
        return policy.verify_torch_blas_policy(preference, phase)

    ns = dict(
        llm_args=args(),
        checkpoint_dir=None,
        creation_metrics={},
        os=os,
        timing_metric=lambda *a: contextlib.nullcontext(),
        _load_config_and_create_checkpoint_loader=load,
        validate_torch_blas_policy_scope=policy.validate_torch_blas_policy_scope,
        verify_torch_blas_policy=verify,
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=block, type_ignores=[])),
        "actual-resolved-config-prefix",
        "exec",
    )
    if resolved is None:
        exec(code, ns)
        assert events == ["config", "resolved_config"]
    else:
        with pytest.raises(ValueError, match="local TP1"):
            exec(code, ns)
        assert events == ["config"]
    assert torch_api.assignments == ["cublaslt"]


def test_typed_config_field_validation_and_serialization():
    cls = config_class()
    assert cls.model_fields["torch_blas_preference"].json_schema_extra == {"status": "prototype"}
    assert cls().model_dump() == {"torch_blas_preference": None}
    assert (
        cls.model_validate_json(
            cls(torch_blas_preference="cublaslt").model_dump_json()
        ).torch_blas_preference
        == "cublaslt"
    )
    for value in (True, 1, "default", "cublas", ["cublaslt"]):
        with pytest.raises(ValidationError):
            cls(torch_blas_preference=value)


def test_actual_warmup_block_reads_policy_before_model_warmup(policy, torch_api):
    policy.initialize_torch_blas_policy("cublaslt")
    events = []
    tree = ast.parse((EXECUTOR / "py_executor.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PyExecutor")
    init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    index = next(i for i, n in enumerate(init.body) if ast.unparse(n) == "self.is_warmup = True")
    block = copy.deepcopy(init.body[index : index + 4])
    stream = SimpleNamespace(wait_stream=lambda _: events.append("stream"))
    torch_api.cuda = SimpleNamespace(
        current_stream=lambda: None, stream=lambda _: contextlib.nullcontext()
    )

    def verify(preference, phase):
        events.append(phase)
        return policy.verify_torch_blas_policy(preference, phase)

    obj = SimpleNamespace(
        llm_args=args(),
        execution_stream=stream,
        dist=SimpleNamespace(mapping=None),
        resource_manager=None,
        draft_model_engine=None,
        model_engine=SimpleNamespace(warmup=lambda _: events.append("model")),
    )
    ns = dict(
        self=obj,
        torch=torch_api,
        verify_torch_blas_policy=verify,
        _distributed_warmup_guard=lambda *a: contextlib.nullcontext(),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=block, type_ignores=[])),
            "actual-warmup-block",
            "exec",
        ),
        ns,
    )
    assert events == ["warmup", "stream", "model"]
    assert torch_api.assignments == ["cublaslt"]


@pytest.mark.parametrize("drift", [False, True])
def test_actual_event_wrapper_checks_policy_on_its_real_thread(policy, torch_api, drift):
    policy.initialize_torch_blas_policy("cublaslt")
    events = []
    if drift:
        torch_api.backend = Backend.Cublas
    obj = SimpleNamespace(
        llm_args=args(),
        dist=SimpleNamespace(world_size=1),
        _event_loop_error=None,
        _event_loop_error_delivered=threading.Event(),
        garbage_collection_gen0_threshold=1,
        is_warmup=False,
        _executor_loop_cleanup=lambda: events.append("cleanup"),
    )

    def verify(preference, phase):
        events.append((phase, threading.get_ident()))
        return policy.verify_torch_blas_policy(preference, phase)

    def event_loop():
        events.append("forward")
        obj._event_loop_completed = True

    obj.event_loop = event_loop
    ns = dict(
        verify_torch_blas_policy=verify,
        os=os,
        traceback=traceback,
        host_profiler_context=lambda **k: contextlib.nullcontext(),
        customized_gc_thresholds=lambda *a: contextlib.nullcontext(),
        logger=SimpleNamespace(error=lambda *a: None),
        start_rank_crash_kill_watchdog=lambda *a, **k: None,
        hard_kill_on_rank_crash=lambda *a, **k: events.append("error_cleanup"),
    )
    run = function(EXECUTOR / "py_executor.py", "_event_loop_wrapper", ns, "PyExecutor")
    with ThreadPoolExecutor(max_workers=1) as workers:
        result = workers.submit(run, obj)
        if drift:
            with pytest.raises(RuntimeError, match="changed"):
                result.result()
        else:
            result.result()
    assert events[0][0] == "execution_thread" and events[0][1] != threading.get_ident()
    assert ("forward" in events) is not drift and "cleanup" in events
    assert torch_api.assignments == ["cublaslt"]


def test_actual_frontend_refuses_policy_before_attachment(monkeypatch):
    monkeypatch.setenv("TLLM_EXECUTOR_ATTACH_INFO", "not-even-parsed")
    ns = dict(os=os)
    create = function(
        ROOT / "tensorrt_llm/executor/executor.py", "create", ns, "GenerationExecutor"
    )
    with pytest.raises(ValueError, match="cannot attach"):
        create(
            engine=None,
            world_size=1,
            llm_args=args(),
            postproc_worker_config=SimpleNamespace(enabled=False),
        )


def test_actual_encoder_entry_rejects_before_import_or_model_creation():
    ns = {}
    create = function(EXECUTOR / "py_executor_creator.py", "create_encoder_executor", ns)
    with pytest.raises(ValueError, match="encoder-only"):
        create(args())
