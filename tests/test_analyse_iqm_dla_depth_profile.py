# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM DLA depth-profile analysis tests
"""Exercise the real two-window report and its fail-closed boundaries."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from scripts import analyse_iqm_dla_depth_profile as analysis

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "analyse_iqm_dla_depth_profile.py"
_DATA = _REPO / "data" / "iqm_paper_replication"
_DEPTH_COUNTS = tuple(
    _DATA / f"iqm_dla_depth_profile_hw_counts_rep{repetition}_2026-07-22.json"
    for repetition in analysis.REPETITIONS
)
_POWERED_COUNTS = tuple(
    _DATA / f"iqm_dla_powered_hw_counts_rep{repetition}_2026-07-21.json"
    for repetition in analysis.REPETITIONS
)
_COMMITTED = _DATA / "iqm_dla_depth_profile_primary_analysis_2026-07-22.json"


def test_command_surface_reproduces_committed_depth_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--counts",
            *(str(path) for path in _DEPTH_COUNTS),
            "--powered-counts",
            *(str(path) for path in _POWERED_COUNTS),
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
    assert "PRIMARY delta_8 +0.0369 vs delta_12 +0.0132" in stdout
    assert "d10 [prior window]" in stdout


def test_incomplete_matrix_and_missing_prior_depth_stay_nonpromotional(
    tmp_path: Path,
) -> None:
    depth_payload = json.loads(_DEPTH_COUNTS[0].read_text(encoding="utf-8"))
    depth_payload["counts"].pop("main_d8_even_rep1")
    partial_depth = tmp_path / "partial-depth.json"
    partial_depth.write_text(json.dumps(depth_payload), encoding="utf-8")

    prior_payload = json.loads(_POWERED_COUNTS[0].read_text(encoding="utf-8"))
    prior_payload["counts"] = {
        label: counts
        for path in _POWERED_COUNTS
        for label, counts in json.loads(path.read_text(encoding="utf-8"))["counts"].items()
        if not label.startswith("main_d4_")
    }
    missing_prior = tmp_path / "missing-prior.json"
    missing_prior.write_text(json.dumps(prior_payload), encoding="utf-8")
    output = tmp_path / "incomplete.json"

    assert (
        analysis.main(
            [
                "--counts",
                str(partial_depth),
                *(str(path) for path in _DEPTH_COUNTS[1:]),
                "--powered-counts",
                str(missing_prior),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["matrix_complete"] is False
    assert report["missing_labels"] == ["main_d8_even_rep1"]
    assert report["primary_decay_ordering"]["rejects_null"] is False
    assert [
        row["depth"] for row in report["joint_profile_with_cross_window_caveat"]["profile"]
    ] == [
        6,
        8,
        10,
        12,
    ]
    assert len(report["per_repetition_drift"]) == 7


def test_invalid_layout_empty_inputs_and_degenerate_statistics_fail_closed(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"counts": {}, "layout": [1]}), encoding="utf-8")
    second.write_text(json.dumps({"counts": {}, "layout": [2]}), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree on layout"):
        analysis.main(
            [
                "--counts",
                str(first),
                "--powered-counts",
                str(second),
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


def test_missing_sector_and_degenerate_decay_variance_fail_closed(tmp_path: Path) -> None:
    missing_sector = tmp_path / "missing-sector.json"
    missing_sector.write_text(
        json.dumps({"counts": {}, "layout": [2, 7, 12, 13]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="depth 8 lacks a nonempty sector"):
        analysis.main(
            [
                "--counts",
                str(missing_sector),
                "--powered-counts",
                str(missing_sector),
                "--out",
                str(tmp_path / "missing.json"),
            ]
        )

    counts: dict[str, dict[str, int]] = {}
    for depth in analysis.NEW_DEPTHS:
        for sector, initial in analysis.SECTORS.items():
            for repetition in analysis.REPETITIONS:
                counts[f"main_d{depth}_{sector}_rep{repetition}"] = {initial: 10}
    degenerate = tmp_path / "degenerate.json"
    degenerate.write_text(
        json.dumps({"counts": counts, "layout": [2, 7, 12, 13]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="degenerate decay-ordering variance"):
        analysis.main(
            [
                "--counts",
                str(degenerate),
                "--powered-counts",
                str(missing_sector),
                "--out",
                str(tmp_path / "degenerate-out.json"),
            ]
        )
