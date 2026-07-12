# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Enzyme Audit Tests
"""Architecture tests for Enzyme/MLIR maturity and toolchain probing."""

from __future__ import annotations

import ast
import inspect
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

import scpn_quantum_control.compiler.mlir as facade
import scpn_quantum_control.compiler.mlir_enzyme_audit as leaf

PRIVATE_NAMES = (
    "_default_enzyme_mlir_audit_circuit",
    "_enzyme_mlir_toolchain_status",
    "_probe_toolchain_version",
    "_resolve_toolchain_executable",
)


def test_enzyme_audit_has_no_facade_back_edge() -> None:
    """Keep Enzyme audit imports one-way from the compiler facade."""
    tree = ast.parse(inspect.getsource(leaf))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "mlir" not in relative_imports


def test_enzyme_audit_facade_exports_are_exact_leaf_aliases() -> None:
    """Preserve the Enzyme audit and probe helper facade identities."""
    assert facade.run_enzyme_mlir_maturity_audit is leaf.run_enzyme_mlir_maturity_audit
    for name in PRIVATE_NAMES:
        assert getattr(facade, name) is getattr(leaf, name)


def test_enzyme_audit_public_export_remains_declared() -> None:
    """Retain the maturity audit in the facade export contract."""
    assert "run_enzyme_mlir_maturity_audit" in facade.__all__


def test_enzyme_audit_records_missing_runtime_correctness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record a hard gap when the injected Phase-QNode runtime fails verification."""
    executable = SimpleNamespace(
        verification={"value_close": False, "gradient_close": True},
        runtime_backend="broken_runtime",
    )
    monkeypatch.setattr(
        leaf,
        "compile_phase_qnode_circuit_to_mlir_runtime",
        lambda *args, **kwargs: executable,
    )

    result = leaf.run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: None,
    )

    assert "MLIR/LLVM correctness check missing" in result.hard_gaps


def test_resolve_toolchain_executable_covers_missing_invalid_and_resolve_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject missing, non-file, and unresolvable toolchain paths."""
    monkeypatch.setattr(shutil, "which", lambda command: None)
    assert leaf._resolve_toolchain_executable("enzyme") is None

    monkeypatch.setattr(shutil, "which", lambda command: str(tmp_path))
    assert leaf._resolve_toolchain_executable("enzyme") is None

    class _FailingPath:
        def __init__(self, value: str) -> None:
            del value

        def resolve(self, *, strict: bool = False) -> NoReturn:
            del strict
            raise OSError("blocked")

    monkeypatch.setattr(cast(Any, leaf), "Path", _FailingPath)
    assert leaf._resolve_toolchain_executable("enzyme") is None


def test_probe_toolchain_version_covers_resolve_and_nonfile_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject unresolvable and non-file absolute version-probe targets."""
    assert leaf._probe_toolchain_version(str(tmp_path)) is None

    class _FailingPath:
        def __init__(self, value: str) -> None:
            del value

        def is_absolute(self) -> bool:
            return True

        def resolve(self, *, strict: bool = False) -> NoReturn:
            del strict
            raise OSError("blocked")

    monkeypatch.setattr(cast(Any, leaf), "Path", _FailingPath)
    tool = tmp_path / "tool"
    assert leaf._probe_toolchain_version(str(tool)) is None


def test_probe_toolchain_version_handles_subprocess_failures_and_empty_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return no version after admitted subprocess failures or empty output."""
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o700)

    def fail_run(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise OSError("blocked")

    monkeypatch.setattr(subprocess, "run", fail_run)
    assert leaf._probe_toolchain_version(str(tool)) is None

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=""),
    )
    assert leaf._probe_toolchain_version(str(tool)) is None
