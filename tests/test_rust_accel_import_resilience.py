# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — optional Rust accelerator import-resilience tests
"""Tests for the resilient optional Rust accelerator accessor and integrity tool.

These lock two related import-resilience guarantees:

* :func:`scpn_quantum_control._rust_accel.optional_rust_engine` defers the
  :mod:`oscillatools` accelerator import to call time and soft-fails to ``None``
  when :mod:`oscillatools` is absent, so importing any consumer module never
  hard-requires the accelerator stack; and
* ``scripts/verify_hardware_result_packs.py`` runs the stdlib-only pack verifier
  without importing the heavy ``scpn_quantum_control`` package root, so the
  integrity tool works in a bare or partial environment.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scpn_quantum_control._rust_accel import optional_rust_engine

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERIFIER = _REPO_ROOT / "scripts" / "verify_hardware_result_packs.py"


def test_optional_rust_engine_returns_engine_or_none_when_available() -> None:
    """With oscillatools installed the accessor resolves without raising."""
    result = optional_rust_engine()
    assert result is None or isinstance(result, ModuleType)


def test_optional_rust_engine_soft_fails_without_oscillatools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing oscillatools degrades to ``None`` instead of raising at call time."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "oscillatools" or name.startswith("oscillatools."):
            raise ModuleNotFoundError(f"No module named '{name}'", name="oscillatools")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert optional_rust_engine() is None


def test_verifier_script_runs_without_the_heavy_package_root() -> None:
    """The integrity tool verifies the committed manifest with stdlib-only deps.

    Running with only ``src`` on the path (no ``oscillatools`` and no accelerator
    stack) would fail if the script imported the package root, whose ``__init__``
    eagerly pulls in Qiskit and oscillatools. A clean exit proves the standalone
    loader bypasses the package init.
    """
    result = subprocess.run(
        [sys.executable, str(_VERIFIER)],
        cwd=_REPO_ROOT,
        env={"PYTHONPATH": str(_REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "oscillatools" not in result.stderr


def test_verifier_loader_returns_the_stdlib_verifier_main() -> None:
    """The standalone loader yields the verifier ``main`` loaded from its own file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_verify_hardware_result_packs_script", _VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verifier_main = module.load_verifier_main()
    assert callable(verifier_main)
    assert verifier_main.__module__ == "scpn_quantum_control._hardware_result_packs_standalone"
