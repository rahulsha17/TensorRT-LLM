# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions for out-of-tree wheel source staging."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def build_wheel() -> ModuleType:
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("existing_staging", [False, True])
def test_stage_python_package(
    build_wheel: ModuleType, tmp_path: Path, existing_staging: bool
) -> None:
    project = tmp_path / "checkout"
    staging = tmp_path / "build" / "python"
    trees = ("tensorrt_llm", "triton_kernels", "examples")
    for tree in trees:
        package = project / tree
        (package / "__pycache__").mkdir(parents=True)
        (package / "__init__.py").write_text(f"# {tree}\n")
        (package / "__pycache__" / "cached.pyc").write_bytes(b"cache")
        (package / "cached.pyc").write_bytes(b"cache")
    assets = (
        "setup.py",
        "pyproject.toml",
        "constraints.txt",
        "LICENSE",
        "README.md",
        "ATTRIBUTIONS-CPP-test.md",
    )
    for name in assets:
        (project / name).write_text(f"{name}\n")

    if existing_staging:
        build_wheel.stage_python_package(project, staging)

    requirements = (
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-grpc-smg.txt",
        "requirements-windows.txt",
        "requirements-dev-windows.txt",
        "requirements-future.txt",
    )
    for name in requirements:
        (project / name).write_text(f"# {name}\n")
    for name in ("unrelated.txt", "requirements.md", "requirements.txt.bak"):
        (project / name).write_text("not a packaging input\n")
    (project / "unrelated").mkdir()
    (project / "unrelated" / "requirements-nested.txt").write_text("nested\n")
    # A root-level directory whose name matches requirements*.txt must be
    # skipped, not handed to copy() (which would raise IsADirectoryError).
    (project / "requirements-local.txt").mkdir()

    build_wheel.stage_python_package(project, staging)

    assert {path.name for path in staging.iterdir()} == set(trees + assets + requirements)
    for name in assets + requirements:
        assert (staging / name).read_bytes() == (project / name).read_bytes()
    for tree in trees:
        assert {path.name for path in (staging / tree).iterdir()} == {"__init__.py"}
        assert (staging / tree / "__init__.py").read_bytes() == (
            project / tree / "__init__.py"
        ).read_bytes()


def test_stage_python_package_without_optional_files(
    build_wheel: ModuleType, tmp_path: Path
) -> None:
    project = tmp_path / "checkout"
    staging = tmp_path / "staging"
    trees = ("tensorrt_llm", "triton_kernels", "examples")
    for tree in trees:
        (project / tree).mkdir(parents=True)

    build_wheel.stage_python_package(project, staging)

    assert {path.name for path in staging.iterdir()} == set(trees)


@pytest.mark.parametrize("out_of_tree", [False, True])
def test_fmha_generation_resolves_shared_headers(
    build_wheel: ModuleType, tmp_path: Path, out_of_tree: bool
) -> None:
    project = tmp_path / "checkout with spaces"
    generator = project / "cpp/kernels/fmha_v2"
    generator.mkdir(parents=True)
    header = project / "cpp/tensorrt_llm/kernels/retireAuthority.h"
    header.parent.mkdir(parents=True)
    header.write_text("// canonical authority header\n")
    (generator / "setup.py").write_text(
        """import os
from pathlib import Path
root = Path(os.environ['TRTLLM_FMHA_CPP_ROOT'])
assert root.is_absolute()
header = root / 'tensorrt_llm/kernels/retireAuthority.h'
assert header.read_text() == '// canonical authority header\\n'
out = Path('generated')
out.mkdir()
for name in ('fmha_cubin.h', 'fmha_cubin.cpp', 'test_sm90.cu'):
    (out / name).write_text('// generated\\n')
"""
    )
    gen_root = tmp_path / "build" / "fmha-gen" if out_of_tree else None
    build_wheel.generate_fmha_cu(project, None, gen_root)
    kernels, cubins = build_wheel.get_fmha_gen_dirs(project, gen_root)
    assert (kernels / "test_sm90.cu").read_text() == "// generated\n"
    assert (cubins / "fmha_cubin.h").is_file()
    assert build_wheel._fmha_generation_stamp(kernels).is_file()
    if out_of_tree:
        assert not (generator / "generated").exists()
