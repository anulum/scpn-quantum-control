# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for version consistency checker
"""Tests for the version consistency pre-commit hook."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_check_version_consistency = _load_script_module(
    "check_version_consistency_for_tests",
    "check_version_consistency.py",
)


def _write_version_files(tmp_path: Path, versions: dict[str, str]) -> dict[str, Path]:
    files = {
        "pyproject": tmp_path / "pyproject.toml",
        "citation": tmp_path / "CITATION.cff",
        "zenodo": tmp_path / ".zenodo.json",
    }
    files["pyproject"].write_text(f'version = "{versions["pyproject"]}"\n', encoding="utf-8")
    files["citation"].write_text(f'version: "{versions["citation"]}"\n', encoding="utf-8")
    files["zenodo"].write_text(f'{{"version": "{versions["zenodo"]}"}}\n', encoding="utf-8")
    return files


def _patch_contract_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, Path]
) -> None:
    monkeypatch.setattr(_check_version_consistency, "ROOT", tmp_path)
    monkeypatch.setattr(
        _check_version_consistency,
        "PATTERNS",
        {
            files["pyproject"]: re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE),
            files["citation"]: re.compile(r'^version:\s*"([^"]+)"', re.MULTILINE),
            files["zenodo"]: re.compile(r'"version":\s*"([^"]+)"'),
        },
    )


def test_version_consistency_returns_zero_when_all_carriers_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _write_version_files(
        tmp_path,
        {
            "pyproject": "0.9.6",
            "citation": "0.9.6",
            "zenodo": "0.9.6",
        },
    )
    _patch_contract_paths(monkeypatch, tmp_path, files)

    assert _check_version_consistency.main() == 0


def test_version_consistency_reports_mismatched_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _write_version_files(
        tmp_path,
        {
            "pyproject": "0.9.6",
            "citation": "0.9.7",
            "zenodo": "0.9.6",
        },
    )
    _patch_contract_paths(monkeypatch, tmp_path, files)

    assert _check_version_consistency.main() == 1
    output = capsys.readouterr().out.replace("\\", "/")
    assert "Version mismatch (canonical: 0.9.6)" in output
    assert "CITATION.cff: 0.9.7 (expected 0.9.6)" in output


def test_version_consistency_reports_missing_and_unmatched_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _write_version_files(
        tmp_path,
        {
            "pyproject": "0.9.6",
            "citation": "0.9.6",
            "zenodo": "0.9.6",
        },
    )
    files["citation"].unlink()
    files["zenodo"].write_text('{"metadata": {}}\n', encoding="utf-8")
    _patch_contract_paths(monkeypatch, tmp_path, files)

    assert _check_version_consistency.main() == 1
    output = capsys.readouterr().out
    assert "CITATION.cff: file not found" in output
    assert ".zenodo.json: version pattern not found" in output


def test_hardware_status_ledger_is_a_registered_version_carrier() -> None:
    """The ledger Package-line version is enforced against the canonical source.

    This is the live drift gate for Q-HW-03: the hardware status ledger's
    Package-line version must not lag the canonical release version, even between
    dated snapshots. It is deliberately hermetic — it compares only the ledger and
    pyproject, both always present — rather than invoking ``main()`` over every
    carrier, because the reproduction Docker image omits metadata-only carriers
    (``CITATION.cff``, ``.zenodo.json``). The full ``main()`` sweep runs in the
    pre-commit/CI gate against a complete checkout.
    """
    ledger = _check_version_consistency.HARDWARE_STATUS_LEDGER
    assert ledger in _check_version_consistency.PATTERNS
    ledger_match = _check_version_consistency.PATTERNS[ledger].search(
        ledger.read_text(encoding="utf-8")
    )
    assert ledger_match is not None
    pyproject = _check_version_consistency.PYPROJECT
    pyproject_match = _check_version_consistency.PATTERNS[pyproject].search(
        pyproject.read_text(encoding="utf-8")
    )
    assert pyproject_match is not None
    assert ledger_match.group(1) == pyproject_match.group(1)


def test_version_consistency_flags_a_drifted_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ledger Package-line version that lags pyproject is reported as a mismatch."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('version = "0.11.0"\n', encoding="utf-8")
    ledger = tmp_path / "hardware_status_ledger.md"
    ledger.write_text(
        "| Package line | Version `0.10.0`, Python `>=3.11`. | `pyproject.toml` |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_check_version_consistency, "ROOT", tmp_path)
    monkeypatch.setattr(
        _check_version_consistency,
        "PATTERNS",
        {
            pyproject: re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE),
            ledger: re.compile(r"Package line \| Version `([^`]+)`"),
        },
    )

    assert _check_version_consistency.main() == 1
    output = capsys.readouterr().out
    assert "0.10.0 (expected 0.11.0)" in output
