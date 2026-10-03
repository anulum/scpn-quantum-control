# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared contract ownership and conformance
"""Bind real shared corpus consumers to mandatory executable CI ownership."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tools import studio_contract_quality_gates as quality
from tools.ci_workflow_inventory import REPOSITORY_ROOT, load_ci_workflow_policy


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _yaml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.load(path.read_text(), Loader=yaml.BaseLoader))


def _isolated_consumers(root: Path) -> tuple[list[str], list[str]]:
    """Invoke original test owners against copied shared fixtures and real production code."""
    root.mkdir()
    shutil.copytree(
        REPOSITORY_ROOT / "tests/data/studio_workspace", root / "tests/data/studio_workspace"
    )
    for name in ["canonical", "contracts", "json_transport"]:
        path = Path(f"tests/test_studio_workspace_{name}.py")
        shutil.copyfile(REPOSITORY_ROOT / path, root / path)
    shutil.copytree(REPOSITORY_ROOT / "studio-web/src/shared", root / "studio-web/src/shared")
    for name in ["vite.config.ts", "module-federation.config.ts", "package.json"]:
        shutil.copyfile(REPOSITORY_ROOT / "studio-web" / name, root / "studio-web" / name)
    (root / "studio-web/node_modules").symlink_to(
        REPOSITORY_ROOT / "studio-web/node_modules", target_is_directory=True
    )
    python = [
        sys.executable,
        "-m",
        "pytest",
        "--noconftest",
        "--no-cov",
        "-q",
        "tests/test_studio_workspace_canonical.py",
        "tests/test_studio_workspace_contracts.py",
        "tests/test_studio_workspace_json_transport.py",
    ]
    node = [
        "node",
        str(REPOSITORY_ROOT / "studio-web/node_modules/vitest/vitest.mjs"),
        "run",
        "src/shared/contracts/canonical.test.ts",
        "src/shared/contracts/workspace.test.ts",
        "src/shared/contracts/jsonTransport.test.ts",
    ]
    return python, node


def _run(root: Path, command: list[str], name: str) -> subprocess.CompletedProcess[str]:
    """Save actual consumer exit and complete diagnostics from the isolated invocation."""
    env = dict(
        os.environ,
        PYTHONPATH=str(REPOSITORY_ROOT / "src") + ":" + str(REPOSITORY_ROOT / "oscillatools/src"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    result = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True, timeout=120
    )
    (root / (name + ".json")).write_text(
        json.dumps(
            {
                "argv": command,
                "cwd": str(root),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
    )
    return result


def test_real_shared_corpus_and_deliberate_digest_drift(tmp_path: Path) -> None:
    """Real Python and Node pass exact independent oracles and fail a changed digest."""
    root = tmp_path / "real"
    python, node = _isolated_consumers(root)
    originals = {
        p: p.read_bytes() for p in (REPOSITORY_ROOT / "tests/data/studio_workspace").glob("*.json")
    }
    native, browser = (
        _run(root, python, "positive-python"),
        _run(root / "studio-web", node, "positive-node"),
    )
    assert native.returncode == browser.returncode == 0, (
        native.stdout + browser.stdout + browser.stderr
    )
    path = root / "tests/data/studio_workspace/canonical.json"
    corpus = _json(path)
    next(row for row in corpus["cases"] if "expected_sha256" in row)["expected_sha256"] = "0" * 64
    path.write_text(json.dumps(corpus))
    native, browser = (
        _run(root, python, "drift-python"),
        _run(root / "studio-web", node, "drift-node"),
    )
    assert native.returncode != 0 and browser.returncode != 0
    assert "AssertionError" in native.stdout and "expected" in browser.stdout + browser.stderr
    assert all(p.read_bytes() == value for p, value in originals.items())
    policy = load_ci_workflow_policy()
    run = _yaml(REPOSITORY_ROOT / policy["coordinator"])["jobs"][policy["required_gate"]]["steps"][
        0
    ]["run"]
    program = "\n".join(run.splitlines()[1:-1])
    env = dict(
        os.environ,
        CATEGORY_RESULTS=json.dumps(
            {
                "studio": {
                    "result": "failure" if native.returncode or browser.returncode else "success"
                }
            }
        ),
    )
    aggregate = subprocess.run(
        [sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=10
    )
    assert aggregate.returncode != 0 and "CI category gate failed" in aggregate.stderr


def test_wire_version_refused_by_real_consumers(tmp_path: Path) -> None:
    """Both public parsers refuse a future major schema without changing raw input."""
    from scpn_quantum_control.studio_workspace import parse_document

    root = tmp_path / "version"
    python, node = _isolated_consumers(root)
    path = root / "tests/data/studio_workspace/documents.json"
    corpus = _json(path)
    payload = corpus["fixtures"]["workspace"]
    payload["schema"] = "quantum_workspace.v999"
    before = deepcopy(payload)
    with pytest.raises(ValueError):
        parse_document(payload)
    assert payload == before
    corpus["cases"] = [{"id": "future_major", "fixture": "workspace", "expectation": "reject"}]
    path.write_text(json.dumps(corpus))
    python = python[:6] + ["tests/test_studio_workspace_contracts.py", "-k", "document_corpus"]
    node = node[:3] + ["src/shared/contracts/workspace.test.ts", "-t", "structural boundary"]
    native, browser = (
        _run(root, python, "future-python"),
        _run(root / "studio-web", node, "future-node"),
    )
    assert native.returncode == browser.returncode == 0, (
        native.stdout + browser.stdout + browser.stderr
    )


def test_run_cli_executes_every_actual_runtime() -> None:
    """Run original Python, Node, native Rust and real compiled WASM corpus owners."""
    assert quality.main(["--run"]) == 0
