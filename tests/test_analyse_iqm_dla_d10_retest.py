# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM DLA depth-10 retest analysis tests
"""Exercise the real cross-window report and its fail-closed boundaries."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from scripts import analyse_iqm_dla_d10_retest as analysis

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "analyse_iqm_dla_d10_retest.py"
_DATA = _REPO / "data" / "iqm_paper_replication"
_COUNTS = _DATA / "iqm_dla_d10_retest_hw_counts_2026-07-22.json"
_PRIOR_COUNTS = tuple(
    _DATA / f"iqm_dla_powered_hw_counts_rep{repetition}_2026-07-21.json"
    for repetition in analysis.PRIOR_REPETITIONS
)
_COMMITTED = _DATA / "iqm_dla_d10_retest_primary_analysis_2026-07-22.json"


def test_command_surface_reproduces_committed_d10_retest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--counts",
            str(_COUNTS),
            "--prior-counts",
            *(str(path) for path in _PRIOR_COUNTS),
            "--out",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPT), run_name="__main__")

    assert exc_info.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        _COMMITTED.read_text(encoding="utf-8")
    )
    stdout = capsys.readouterr().out
    assert "negative_sign_replicates=False" in stdout
    assert "S2 cross-window: now +0.0133 vs prior -0.0237" in stdout
    assert "S3 total leakage: now 0.4544 vs prior 0.4694" in stdout


def test_incomplete_matrix_stays_nonpromotional(tmp_path: Path) -> None:
    payload = json.loads(_COUNTS.read_text(encoding="utf-8"))
    payload["counts"].pop("main_d10_even_rep1")
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "incomplete.json"

    assert (
        analysis.main(
            [
                "--counts",
                str(partial),
                "--prior-counts",
                *(str(path) for path in _PRIOR_COUNTS),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["matrix_complete"] is False
    assert report["missing_labels"] == ["main_d10_even_rep1"]
    assert report["primary_sign_replication"]["negative_sign_replicates"] is False
    assert len(report["per_repetition_drift"]) == 7


def test_invalid_layout_and_scalar_boundaries_fail_closed(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"counts": {}, "layout": [1]}), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree on layout"):
        analysis.main(
            [
                "--counts",
                str(_COUNTS),
                "--prior-counts",
                str(prior),
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
    with pytest.raises(ValueError, match="empty count block"):
        analysis._leak({}, "0011")
    with pytest.raises(ValueError, match="empty sample"):
        analysis._wilson(0, 0)
    with pytest.raises(ValueError, match="nonzero odd-sector leakage"):
        analysis._relative_asymmetry(0.1, 0.0)


def test_missing_current_or_prior_sector_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"counts": {}, "layout": [2, 7, 12, 13]}), encoding="utf-8")

    with pytest.raises(ValueError, match="retest window lacks a nonempty sector"):
        analysis.main(
            [
                "--counts",
                str(empty),
                "--prior-counts",
                str(empty),
                "--out",
                str(tmp_path / "missing-current.json"),
            ]
        )
    with pytest.raises(ValueError, match="prior window lacks a nonempty sector"):
        analysis.main(
            [
                "--counts",
                str(_COUNTS),
                "--prior-counts",
                str(empty),
                "--out",
                str(tmp_path / "missing-prior.json"),
            ]
        )


def _deterministic_counts(*, even_leaks: bool, odd_leaks: bool) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for sector, initial in analysis.SECTORS.items():
        leaks = even_leaks if sector == "even" else odd_leaks
        outcome = "0001" if initial == "0011" else "0011"
        if not leaks:
            outcome = initial
        for repetition in analysis.REPETITIONS:
            counts[f"main_d10_{sector}_rep{repetition}"] = {outcome: 10}
    return counts


def test_degenerate_primary_and_cross_window_variances_fail_closed(tmp_path: Path) -> None:
    layout = [2, 7, 12, 13]
    no_leaks = tmp_path / "no-leaks.json"
    no_leaks.write_text(
        json.dumps(
            {"counts": _deterministic_counts(even_leaks=False, odd_leaks=False), "layout": layout}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="degenerate pooled proportion"):
        analysis.main(
            [
                "--counts",
                str(no_leaks),
                "--prior-counts",
                str(no_leaks),
                "--out",
                str(tmp_path / "degenerate-primary.json"),
            ]
        )

    deterministic = tmp_path / "deterministic.json"
    deterministic.write_text(
        json.dumps(
            {"counts": _deterministic_counts(even_leaks=False, odd_leaks=True), "layout": layout}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="degenerate cross-window variance"):
        analysis.main(
            [
                "--counts",
                str(deterministic),
                "--prior-counts",
                str(deterministic),
                "--out",
                str(tmp_path / "degenerate-cross-window.json"),
            ]
        )
