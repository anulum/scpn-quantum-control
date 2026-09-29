# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KYMA v3 probe runner script
"""Run the KYMA v3 runner script end to end with a one-epoch, one-seed budget."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_kyma_v3_probe.py"


def test_runner_writes_the_artefact_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that runner writes the artefact and exits cleanly."""
    out = tmp_path / "result" / "kyma_v3.json"
    argv = [str(SCRIPT), "--nominal-power-w", "12.5", "--commit", "abc123", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--seeds", "0", "--epochs", "1"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stopped.value.code == 0
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["probe"] == "kyma_v3_symbolic_composition"
    assert artefact["pre_registration"].endswith(
        "kyma_v3_symbolic_composition_prereg_2026-09-29.md"
    )
    assert artefact["source_commit"] == "abc123"
    assert artefact["result"]["energy_proxy"]["nominal_power_w"] == 12.5
    assert artefact["result"]["seeds"] == [0]
    printed = capsys.readouterr().out
    assert "pass=" in printed and "artefact ->" in printed


def test_runner_import_defines_main_without_running() -> None:
    """Check that loading the script as a module defines main and runs nothing."""
    namespace = runpy.run_path(str(SCRIPT), run_name="kyma_v3_runner_module")
    assert callable(namespace["main"])
