# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Rust .pyi export checker
"""Tests for the Rust extension typing contract checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_check_rust_pyi_exports = _load_tool_module(
    "check_rust_pyi_exports_for_tests",
    "check_rust_pyi_exports.py",
)


def _write_contracts(tmp_path: Path, *, rust_text: str, pyi_text: str) -> tuple[Path, Path]:
    rust_path = tmp_path / "lib.rs"
    pyi_path = tmp_path / "scpn_quantum_engine.pyi"
    rust_path.write_text(rust_text, encoding="utf-8")
    pyi_path.write_text(pyi_text, encoding="utf-8")
    return rust_path, pyi_path


def test_rust_and_pyi_exports_ignore_namespaces_and_private_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_path, pyi_path = _write_contracts(
        tmp_path,
        rust_text="\n".join(
            [
                "m.add_function(wrap_pyfunction!(build_knm, m)?)?;",
                "m.add_function(wrap_pyfunction!(module::score_layout, m)?)?;",
            ]
        ),
        pyi_text="\n".join(
            [
                "def build_knm(n: int) -> list[float]: ...",
                "def score_layout() -> float: ...",
                "def _private_helper() -> None: ...",
            ]
        ),
    )
    monkeypatch.setattr(_check_rust_pyi_exports, "RUST_MODULE", rust_path)
    monkeypatch.setattr(_check_rust_pyi_exports, "PYI_MODULE", pyi_path)

    assert _check_rust_pyi_exports.rust_exports() == {"build_knm", "score_layout"}
    assert _check_rust_pyi_exports.pyi_exports() == {"build_knm", "score_layout"}


def test_rust_pyi_checker_returns_zero_when_contracts_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rust_path, pyi_path = _write_contracts(
        tmp_path,
        rust_text="m.add_function(wrap_pyfunction!(build_knm, m)?)?;",
        pyi_text="def build_knm(n: int) -> list[float]: ...",
    )
    monkeypatch.setattr(_check_rust_pyi_exports, "RUST_MODULE", rust_path)
    monkeypatch.setattr(_check_rust_pyi_exports, "PYI_MODULE", pyi_path)

    assert _check_rust_pyi_exports.main() == 0
    assert "Rust extension typing contract OK (1 exports)" in capsys.readouterr().out


def test_rust_pyi_checker_reports_missing_and_stale_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rust_path, pyi_path = _write_contracts(
        tmp_path,
        rust_text="\n".join(
            [
                "m.add_function(wrap_pyfunction!(build_knm, m)?)?;",
                "m.add_function(wrap_pyfunction!(new_kernel, m)?)?;",
            ]
        ),
        pyi_text="\n".join(
            [
                "def build_knm(n: int) -> list[float]: ...",
                "def stale_kernel() -> None: ...",
            ]
        ),
    )
    monkeypatch.setattr(_check_rust_pyi_exports, "RUST_MODULE", rust_path)
    monkeypatch.setattr(_check_rust_pyi_exports, "PYI_MODULE", pyi_path)

    assert _check_rust_pyi_exports.main() == 1
    output = capsys.readouterr().out
    assert "missing from src/scpn_quantum_engine.pyi" in output
    assert "- new_kernel" in output
    assert "declared in src/scpn_quantum_engine.pyi but not exported by Rust" in output
    assert "- stale_kernel" in output
