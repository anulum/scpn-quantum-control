# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QFI/FSS Differentiable Example Tests
"""Smoke tests for the QFI/FSS differentiable example workflow."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_qfi_fss_differentiable_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The QFI/FSS example runs locally and prints bounded evidence fields."""
    runpy.run_path(
        str(ROOT / "examples" / "31_qfi_fss_differentiable_report.py"),
        run_name="__main__",
    )

    output = capsys.readouterr().out

    assert "qfi/fss differentiable evidence" in output
    assert "operation: qfi_fss_report" in output
    assert "bkt model: bkt_log_correction" in output
    assert "power model: power_law_nu_1" in output
    assert "hardware: blocked" in output
