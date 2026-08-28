# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — powered IQM DLA primary-analysis tests
"""Exercise the real count-backed report and its fail-closed boundaries."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from scripts import analyse_iqm_dla_powered_primary as analysis

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "analyse_iqm_dla_powered_primary.py"
_DATA = _REPO / "data" / "iqm_paper_replication"
_COUNTS = tuple(
    _DATA / f"iqm_dla_powered_hw_counts_rep{repetition}_2026-07-21.json"
    for repetition in analysis.REPETITIONS
)
_COMMITTED = _DATA / "iqm_dla_powered_primary_analysis_2026-07-21.json"


def test_command_surface_reproduces_committed_primary_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT), "--counts", *(str(path) for path in _COUNTS), "--out", str(output)],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPT), run_name="__main__")

    assert exc_info.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        _COMMITTED.read_text(encoding="utf-8")
    )
    stdout = capsys.readouterr().out
    assert "PRIMARY REJECTS H0 (backend-universal direction): True" in stdout
    assert "d10:" in stdout


def test_incomplete_matrix_stays_non_promotional(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    partial_counts = tmp_path / "partial-counts.json"
    output = tmp_path / "incomplete.json"
    payload = json.loads(_COUNTS[0].read_text(encoding="utf-8"))
    payload["counts"] = {
        label: counts
        for label, counts in payload["counts"].items()
        if not label.startswith("main_d10_")
    }
    partial_counts.write_text(json.dumps(payload), encoding="utf-8")

    assert analysis.main(["--counts", str(partial_counts), "--out", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["matrix_complete"] is False
    assert report["missing_labels"]
    assert "10" not in report["per_depth"]
    assert report["primary_pooled"]["rejects_h0"] is False
    assert "matrix incomplete" in capsys.readouterr().out


def test_layout_mismatch_and_degenerate_statistics_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"counts": {}, "layout": [1]}), encoding="utf-8")
    second.write_text(json.dumps({"counts": {}, "layout": [2]}), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree on layout"):
        analysis.main(["--counts", str(first), str(second), "--out", str(tmp_path / "out.json")])
    with pytest.raises(ValueError, match="empty sample"):
        analysis._wilson(0, 0)
    with pytest.raises(ValueError, match="degenerate pooled proportion"):
        analysis._one_sided_two_proportion(0, 10, 0, 10)
