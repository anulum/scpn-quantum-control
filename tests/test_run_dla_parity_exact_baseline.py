# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Exact DLA-parity artifact runner tests
"""Exercise the exact DLA-parity artifact runner through its public entry points."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from scripts import run_dla_parity_exact_baseline as runner

_SCRIPT = Path("scripts/run_dla_parity_exact_baseline.py")


def _write_summary(path: Path) -> None:
    """Write a minimal promoted-summary contract fixture."""
    path.write_text(
        json.dumps(
            {
                "depth_summaries": [
                    {
                        "depth": 2,
                        "leakage_even": 0.08,
                        "leakage_odd": 0.09,
                        "asymmetry_relative": 0.125,
                    },
                    {
                        "depth": 4,
                        "leakage_even": 0.12,
                        "leakage_odd": 0.15,
                        "asymmetry_relative": 0.25,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _zero_leakage(_n: int, _initial: str, _depth: int, _t_step: float) -> float:
    """Return the exact conserved-parity reference."""
    return 0.0


def test_build_comparison_preserves_hardware_and_exact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combine every hardware depth with the exact zero-leakage baseline."""
    summary = tmp_path / "summary.json"
    _write_summary(summary)
    monkeypatch.setattr(runner, "_REPO", tmp_path)
    monkeypatch.setattr(runner, "_SUMMARY", summary)
    monkeypatch.setattr(runner, "exact_parity_leakage", _zero_leakage)

    comparison = runner.build_comparison()

    assert comparison["source_summary"] == "summary.json"
    assert comparison["exact_leakage_all_zero"] is True
    assert comparison["hardware_leakage_max"] == pytest.approx(0.15)
    assert [row["depth"] for row in comparison["per_depth"]] == [2, 4]
    assert all(row["exact_leakage_even"] == 0.0 for row in comparison["per_depth"])
    assert "15.0%" in comparison["conclusion"]


def test_build_comparison_reports_nonzero_exact_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail the exact-zero flag when an opposite-parity reference is nonzero."""
    summary = tmp_path / "summary.json"
    _write_summary(summary)

    def _odd_leakage(_n: int, initial: str, _depth: int, _t_step: float) -> float:
        return 0.01 if initial == runner._ODD_INIT else 0.0

    monkeypatch.setattr(runner, "_REPO", tmp_path)
    monkeypatch.setattr(runner, "_SUMMARY", summary)
    monkeypatch.setattr(runner, "exact_parity_leakage", _odd_leakage)
    assert runner.build_comparison()["exact_leakage_all_zero"] is False


def test_main_writes_the_comparison_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write and report the deterministic artifact through the CLI entry point."""
    summary = tmp_path / "summary.json"
    output = tmp_path / "artifact"
    _write_summary(summary)
    monkeypatch.setattr(runner, "_REPO", tmp_path)
    monkeypatch.setattr(runner, "_SUMMARY", summary)
    monkeypatch.setattr(runner, "_OUT", output)
    monkeypatch.setattr(runner, "exact_parity_leakage", _zero_leakage)

    assert runner.main() == 0

    artifact = output / "dla_parity_exact_baseline.json"
    payload = cast(dict[str, object], json.loads(artifact.read_text(encoding="utf-8")))
    stdout = capsys.readouterr().out
    assert payload["exact_leakage_all_zero"] is True
    assert "exact_leakage_all_zero: True" in stdout
    assert "artifact: artifact/dla_parity_exact_baseline.json" in stdout


def test_script_bootstraps_the_source_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add the repository source tree when the script is loaded in isolation."""
    source = str(_SCRIPT.resolve().parents[1] / "src")
    isolated_path = [entry for entry in sys.path if entry != source]
    spec = importlib.util.spec_from_file_location("_isolated_dla_exact_runner", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    with monkeypatch.context() as context:
        context.setattr(sys, "path", isolated_path)
        spec.loader.exec_module(module)
        assert sys.path[0] == source
