# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual workflow runtime identity public tests
"""Qualify actual source snapshots, installed metadata and bounded runtime admission."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio.workflow_contracts import WorkflowDefinition, parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json


def definition() -> WorkflowDefinition:
    """Read the shared bounded original compiler graph.

    Returns
    -------
    WorkflowDefinition
        Exact two-stage, six-coordinate compiler graph.

    """
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    return parse_workflow(corpus["workflow"])


@pytest.fixture(scope="module")
def original_runtime_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy the actual package to an owned runtime allocation with every source byte verified.

    Parameters
    ----------
    tmp_path_factory
        Existing Samsung-only allocation owner for the subprocess fixture.

    Returns
    -------
    Path
        Actual complete package source tree; canonical source files stay untouched.

    """
    original = Path(__file__).parents[1] / "src/scpn_quantum_control"
    target = tmp_path_factory.mktemp("workflow-runtime") / "workflow-runtime-source"
    package = target / "scpn_quantum_control"
    shutil.copytree(original, package, ignore=shutil.ignore_patterns("__pycache__"))
    for source in original.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            assert (package / source.relative_to(original)).read_bytes() == source.read_bytes()
    return target


def test_actual_source_snapshot_refuses_an_oversized_package_before_hashing(
    original_runtime_copy: Path, tmp_path: Path
) -> None:
    """Refuse an actual sparse oversized source file without reading its expanded bytes.

    Parameters
    ----------
    original_runtime_copy
        Complete byte-verified package in its own mutable runtime allocation.
    tmp_path
        Owned current graph input for the real public snapshot subprocess.

    """
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(definition().to_dict()))
    script = """
import sys
from pathlib import Path
from scpn_quantum_control.studio import workflow_execution
from scpn_quantum_control.studio.executive_cli import build_default_registry
from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json

execution_path = Path(workflow_execution.__file__).resolve()
assert execution_path == Path(sys.argv[2]) / "scpn_quantum_control/studio/workflow_execution.py"
target = execution_path.parents[1] / "__init__.py"
original_bytes = target.read_bytes()
original_size = target.stat().st_size
try:
    with target.open("r+b") as stream:
        stream.truncate(128 * 1024 * 1024 + 1)
    assert target.stat().st_size > 128 * 1024 * 1024
    try:
        workflow_execution.workflow_runtime_identity(
            build_default_registry(), parse_workflow(read_json(Path(sys.argv[1]).read_text()))
        )
    except ValueError as refusal:
        assert str(refusal) == "runtime source inventory exceeds byte bound"
    else:
        raise AssertionError("oversized actual runtime source was admitted")
finally:
    with target.open("r+b") as stream:
        stream.truncate(original_size)
    assert target.read_bytes() == original_bytes
"""
    repo = Path(__file__).parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=f"{original_runtime_copy}:{repo / 'oscillatools/src'}:{repo}",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(original_runtime_copy)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_actual_source_change_while_reading_refuses_runtime_identity(
    original_runtime_copy: Path, tmp_path: Path
) -> None:
    """Refuse a real filesystem source change between the snapshot's before/after reads.

    Parameters
    ----------
    original_runtime_copy
        Complete verified package used solely by the adverse subprocess.
    tmp_path
        Owned graph input and actual subprocess output allocation.

    """
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(definition().to_dict()))
    script = """
import os, sys, threading
from pathlib import Path
from scpn_quantum_control.studio import workflow_execution
from scpn_quantum_control.studio.executive_cli import build_default_registry
from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json

execution_path = Path(workflow_execution.__file__).resolve()
assert execution_path == Path(sys.argv[2]) / "scpn_quantum_control/studio/workflow_execution.py"
registry = build_default_registry()
definition = parse_workflow(read_json(Path(sys.argv[1]).read_text()))
target = execution_path.parents[1] / "__init__.py"
original_bytes = target.read_bytes()
retained = target.with_suffix(".retained-original")
target.rename(retained)
os.mkfifo(target)
errors = []

def change_during_read():
    try:
        with target.open("wb") as output:
            output.write(original_bytes)
            output.flush()
            current = target.stat()
            os.utime(target, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
    except BaseException as error:
        errors.append(error)

writer = threading.Thread(target=change_during_read, daemon=True)
writer.start()
try:
    try:
        workflow_execution.workflow_runtime_identity(registry, definition)
    except ValueError as refusal:
        assert str(refusal) == "original runtime source changed while hashing"
    else:
        raise AssertionError("changed original source was admitted")
    writer.join(timeout=5)
    assert not writer.is_alive() and errors == []
finally:
    target.rename(target.with_suffix(".retained-adverse-fifo"))
    retained.rename(target)
    assert target.read_bytes() == original_bytes
"""
    repo = Path(__file__).parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=f"{original_runtime_copy}:{repo / 'oscillatools/src'}:{repo}",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(original_runtime_copy)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_actual_runtime_metadata_discovery_change_refuses_saved_completion(tmp_path: Path) -> None:
    """Refuse reuse when a real interpreter loses installed distribution discovery.

    Parameters
    ----------
    tmp_path
        Owned graph and subprocess output allocation; the parent environment stays unchanged.

    """
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(definition().to_dict()))
    script = """
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from scpn_quantum_control.studio.workflow_execution import run_workflow, workflow_runtime_identity
from scpn_quantum_control.studio.executive_cli import build_default_registry
from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json

definition = parse_workflow(read_json(Path(sys.argv[1]).read_text()))
registry = build_default_registry()
original = run_workflow(definition, registry=registry)
assert original.state == "complete" and original.evaluations == 12
before = write_json(original.to_dict())
sys.path[:] = [path for path in sys.path if "site-packages" not in Path(path).parts and not (Path(path or ".") / "oscillatools").exists()]
sys.modules.pop("oscillatools", None)
assert find_spec("oscillatools") is None
missing = []
for name in ("numpy", "qiskit", "qiskit-aer", "scipy", "scpn-studio-platform"):
    try:
        version(name)
    except PackageNotFoundError:
        missing.append(name)
assert len(missing) == 5
assert workflow_runtime_identity(registry, definition) != original.runtime_fingerprint
saved = []
try:
    run_workflow(definition, registry=registry, journal=original, checkpoint=saved.append)
except ValueError as refusal:
    assert str(refusal) == "checkpoint source or runtime differs; original history retained"
else:
    raise AssertionError("saved completion survived a real metadata-discovery change")
assert saved == [] and write_json(original.to_dict()) == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("bound", ["package-count", "dependency-count", "dependency-bytes"])
def test_actual_snapshot_inventory_bounds_before_source_admission(
    original_runtime_copy: Path, tmp_path: Path, bound: str
) -> None:
    """Refuse actual oversized source inventories without executing a workflow stage.

    Parameters
    ----------
    original_runtime_copy
        Verified complete original package copied into the owned allocation.
    tmp_path
        Owned original dependency copy, graph and retained adverse inventory.
    bound
        Package file count, dependency file count or dependency logical byte bound.

    """
    repo = Path(__file__).parents[1]
    dependency = tmp_path / "oscillatools"
    original_dependency = repo / "oscillatools/src/oscillatools"
    shutil.copytree(original_dependency, dependency, ignore=shutil.ignore_patterns("__pycache__"))
    for original in original_dependency.rglob("*"):
        if original.is_file() and "__pycache__" not in original.parts:
            assert (
                dependency / original.relative_to(original_dependency)
            ).read_bytes() == original.read_bytes()
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(definition().to_dict()))
    script = """
import sys
from pathlib import Path
from importlib.util import find_spec
from scpn_quantum_control.studio import workflow_execution
from scpn_quantum_control.studio.executive_cli import build_default_registry
from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json

definition = parse_workflow(read_json(Path(sys.argv[1]).read_text()))
registry = build_default_registry()
execution_path = Path(workflow_execution.__file__).resolve()
assert execution_path == Path(sys.argv[2]) / "scpn_quantum_control/studio/workflow_execution.py"
spec = find_spec("oscillatools")
assert spec is not None and Path(spec.origin).resolve() == Path(sys.argv[3]) / "oscillatools/__init__.py"
bound = sys.argv[4]
target = execution_path.parents[1] if bound == "package-count" else Path(spec.origin).parent
original_source = target / "__init__.py"
original_bytes = original_source.read_bytes()
aliases = target / "adverse-inventory"
original_size = original_source.stat().st_size
try:
    if bound == "dependency-bytes":
        with original_source.open("r+b") as stream:
            stream.truncate(128 * 1024 * 1024 + 1)
        expected = "original native dependency source exceeds byte bound"
    else:
        count = len(list(target.rglob("*.py"))) + len(list(target.rglob("*.so")))
        aliases.mkdir()
        for index in range(4097 - count):
            (aliases / f"source_{index}.py").symlink_to(original_source)
        assert len(list(target.rglob("*.py"))) + len(list(target.rglob("*.so"))) == 4097
        expected = "runtime source inventory exceeds bound" if bound == "package-count" else "original native dependency source inventory exceeds bound"
    try:
        workflow_execution.workflow_runtime_identity(registry, definition)
    except ValueError as refusal:
        assert str(refusal) == expected
    else:
        raise AssertionError("oversized actual inventory was admitted")
finally:
    if aliases.exists():
        aliases.rename(Path(sys.argv[3]) / ("retained-" + bound))
    if bound == "dependency-bytes":
        with original_source.open("r+b") as stream:
            stream.truncate(original_size)
    assert original_source.read_bytes() == original_bytes
"""
    environment = dict(
        os.environ,
        PYTHONPATH=f"{tmp_path}:{original_runtime_copy}:{repo / 'oscillatools/src'}:{repo}",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(source),
            str(original_runtime_copy),
            str(tmp_path),
            bound,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_actual_source_change_after_read_refuses_complete_snapshot(
    original_runtime_copy: Path, tmp_path: Path
) -> None:
    """Refuse an earlier source change observed after a later actual snapshot read.

    Parameters
    ----------
    original_runtime_copy
        Complete original package isolated from canonical sources and other consumers.
    tmp_path
        Owned graph and original subprocess observation allocation.

    """
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(definition().to_dict()))
    script = """
import ctypes, os, select, sys, threading
from pathlib import Path
from scpn_quantum_control.studio import workflow_execution
from scpn_quantum_control.studio.executive_cli import build_default_registry
from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import read_json

execution = Path(workflow_execution.__file__).resolve()
assert execution == Path(sys.argv[2]) / "scpn_quantum_control/studio/workflow_execution.py"
registry = build_default_registry()
definition = parse_workflow(read_json(Path(sys.argv[1]).read_text()))
first = execution.parents[1] / "__init__.py"
watched = execution.parent / "workflow_sweep.py"
original_bytes = first.read_bytes()
original_stat = first.stat()
libc = ctypes.CDLL(None, use_errno=True)
libc.inotify_init1.argtypes = [ctypes.c_int]
libc.inotify_init1.restype = ctypes.c_int
libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
libc.inotify_add_watch.restype = ctypes.c_int
descriptor = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
assert descriptor >= 0
assert libc.inotify_add_watch(descriptor, os.fsencode(watched), 0x00000010) >= 0
errors = []
changed = threading.Event()

def observe_original_read():
    try:
        assert select.select([descriptor], [], [], 20)[0]
        assert os.read(descriptor, 4096)
        os.utime(first, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1))
        changed.set()
    except BaseException as error:
        errors.append(error)

observer = threading.Thread(target=observe_original_read, daemon=True)
observer.start()
try:
    try:
        workflow_execution.workflow_runtime_identity(registry, definition)
    except ValueError as refusal:
        assert str(refusal) == "original runtime source changed during snapshot admission"
    else:
        raise AssertionError("incoherent actual source snapshot was admitted")
    observer.join(timeout=5)
    assert changed.is_set() and not observer.is_alive() and errors == []
finally:
    os.close(descriptor)
    os.utime(first, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert first.read_bytes() == original_bytes
"""
    repo = Path(__file__).parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=f"{original_runtime_copy}:{repo / 'oscillatools/src'}:{repo}",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(original_runtime_copy)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
