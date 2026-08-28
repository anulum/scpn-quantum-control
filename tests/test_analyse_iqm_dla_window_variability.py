# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — window-variability analysis tests
"""Tests for the preregistered IQM DLA window-variability analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyse_iqm_dla_window_variability.py"
DATA_ROOT = REPO_ROOT / "data" / "iqm_paper_replication"
WINDOW_COUNTS = (
    DATA_ROOT / "iqm_dla_window_variability_hw_counts_w1_2026-07-22.json",
    DATA_ROOT / "iqm_dla_window_variability_hw_counts_w2_2026-07-26.json",
    DATA_ROOT / "iqm_dla_window_variability_hw_counts_w3_2026-07-26.json",
    DATA_ROOT / "iqm_dla_window_variability_hw_counts_w4_2026-07-27.json",
    DATA_ROOT / "iqm_dla_window_variability_hw_counts_w5_2026-07-28.json",
    DATA_ROOT / "iqm_dla_window_variability_hw_counts_w6_2026-07-28.json",
)
CALIBRATIONS = (
    DATA_ROOT / "iqm_dla_window_variability_calibration_w1_2026-07-22.json",
    DATA_ROOT / "iqm_dla_window_variability_calibration_w2_2026-07-26.json",
    DATA_ROOT / "iqm_dla_window_variability_calibration_w3_2026-07-26.json",
    DATA_ROOT / "iqm_dla_window_variability_calibration_w4_2026-07-27.json",
    DATA_ROOT / "iqm_dla_window_variability_calibration_w5_2026-07-28.json",
    DATA_ROOT / "iqm_dla_window_variability_calibration_w6_2026-07-28.json",
)
INTERIM_REPORTS = (
    DATA_ROOT / "iqm_dla_window_variability_interim_analysis_w2_2026-07-26.json",
    DATA_ROOT / "iqm_dla_window_variability_interim_analysis_w3_2026-07-26.json",
    DATA_ROOT / "iqm_dla_window_variability_interim_analysis_w4_2026-07-27.json",
    DATA_ROOT / "iqm_dla_window_variability_interim_analysis_w5_2026-07-28.json",
    DATA_ROOT / "iqm_dla_window_variability_interim_analysis_w6_2026-07-28.json",
)


def _load_analysis_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_analyse_iqm_dla_window_variability",
        ANALYSIS_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load_analysis_script()


def _calibration_covariates() -> dict[str, dict[str, Any]]:
    covariates: dict[str, dict[str, Any]] = {}
    for index, (counts_path, calibration_path) in enumerate(
        zip(WINDOW_COUNTS, CALIBRATIONS, strict=True), start=1
    ):
        layout = json.loads(counts_path.read_text(encoding="utf-8"))["layout"]
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration = payload["calibration"]
        edge_keys = [f"{left}-{right}" for left, right in zip(layout, layout[1:])]
        cz_fidelity = {key: calibration["edge_fidelity"][key] for key in edge_keys}
        readout_error = {str(qubit): calibration["readout_error"][str(qubit)] for qubit in layout}
        covariates[str(index)] = {
            "calibration_set_id": payload["calibration_set_id"],
            "cz_fidelity_by_edge": cz_fidelity,
            "date": payload["date"],
            "layout": layout,
            "mean_cz_fidelity": sum(cz_fidelity.values()) / len(cz_fidelity),
            "mean_readout_error": sum(readout_error.values()) / len(readout_error),
            "readout_error_by_qubit": readout_error,
        }
    return covariates


def test_real_windows_reproduce_every_committed_interim_report(tmp_path: Path) -> None:
    """Each cumulative real window set reproduces its exact committed report."""
    covariates_path = tmp_path / "covariates.json"
    covariates_path.write_text(json.dumps(_calibration_covariates()), encoding="utf-8")

    for achieved, committed in enumerate(INTERIM_REPORTS, start=2):
        output = tmp_path / f"window-{achieved}.json"
        argv = [
            "--window-counts",
            *(str(path) for path in WINDOW_COUNTS[:achieved]),
            "--out",
            str(output),
        ]
        if achieved == analysis.MINIMUM_WINDOWS:
            argv.extend(("--covariates", str(covariates_path)))

        assert analysis.main(argv) == 0
        assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
            committed.read_text(encoding="utf-8")
        )


def _counts(window: int) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for depth in analysis.DEPTHS:
        even_leaks = 1 + ((window + depth) % 3)
        odd_leaks = 2 + ((2 * window + depth) % 3)
        for repetition in analysis.REPETITIONS:
            counts[f"main_d{depth}_even_rep{repetition}"] = {
                "0011": 10 - even_leaks,
                "0001": even_leaks,
            }
            counts[f"main_d{depth}_odd_rep{repetition}"] = {
                "0001": 10 - odd_leaks,
                "0011": odd_leaks,
            }
    counts["readout_0011"] = {"00 11": 9, "0001": 1}
    return counts


def _write_window(path: Path, window: int) -> None:
    payload = {
        "counts": _counts(window),
        "job_ids": [f"job-{window}"],
        "date": f"2026-07-{20 + window:02d}",
        "layout": [0, 1, 2, 3],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_full_campaign_analysis_writes_preregistered_statistics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Six synthetic windows exercise the complete analysable campaign path."""
    window_paths = []
    for window in range(1, 7):
        path = tmp_path / f"window-{window}.json"
        _write_window(path, window)
        window_paths.append(path)
    covariates_path = tmp_path / "covariates.json"
    covariates_path.write_text(json.dumps({"1": {"temperature_mk": 15.0}}), encoding="utf-8")
    output_path = tmp_path / "analysis.json"

    result = analysis.main(
        [
            "--window-counts",
            *(str(path) for path in window_paths),
            "--covariates",
            str(covariates_path),
            "--out",
            str(output_path),
        ]
    )

    assert result == 0
    report: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["campaign"] == "iqm_dla_window_variability_prereg_2026-07-22"
    assert report["analysable"] is True
    assert report["achieved_windows"] == 6
    assert report["primary_d10_heterogeneity"]["minimum_windows"] == 6
    assert set(report["s1_per_depth_heterogeneity_holm"]) == {"4", "8", "12"}
    assert report["s3_d4_sign_stability"]["windows"] == 6
    assert report["s4_calibration_covariates"]["covariates"]["1"]["temperature_mk"] == 15.0
    stdout = capsys.readouterr().out
    assert "PRIMARY d10" in stdout
    assert "S1 d4" in stdout
    assert "S3 d4 sign stability" in stdout


def test_sparse_campaign_stays_unanalysable_and_preserves_absent_signals(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A window without main/readout labels cannot fabricate statistical evidence."""
    window_path = tmp_path / "sparse.json"
    window_path.write_text(json.dumps({"counts": {}}), encoding="utf-8")
    output_path = tmp_path / "analysis.json"

    assert analysis.main(["--window-counts", str(window_path), "--out", str(output_path)]) == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["analysable"] is False
    assert report["primary_d10_heterogeneity"]["drift_exceeds_shot_noise"] is False
    assert report["s1_per_depth_heterogeneity_holm"] == {}
    assert report["s2_tau_profile"] == {}
    assert report["s3_d4_sign_stability"]["fraction_positive"] is None
    assert report["s3_d4_sign_stability"]["clopper_pearson_95"] is None
    assert report["s4_calibration_covariates"]["covariates"] is None
    assert capsys.readouterr().out == f"analysis: {output_path}\n"


def test_statistical_helpers_cover_degenerate_and_boundary_inputs() -> None:
    """Helper boundaries remain explicit for zero totals and exact intervals."""
    assert analysis._parity("00 11") == 0
    assert analysis._leak({"0011": 3, "0001": 1}, "0011") == (1, 4)
    assert analysis._delta_and_variance({"even": (0, 0), "odd": (0, 0)}) is None

    single = analysis._cochran_q([0.25], [0.04])
    assert single["windows"] == 1
    assert single["tau_dl"] == 0.0
    assert analysis._holm({4: 0.04, 8: 0.01, 12: 0.03}) == {
        8: pytest.approx(0.03),
        12: pytest.approx(0.06),
        4: pytest.approx(0.06),
    }
    assert analysis._clopper_pearson(0, 6)[0] == 0.0
    assert analysis._clopper_pearson(6, 6)[1] == 1.0
    middle = analysis._clopper_pearson(3, 6)
    assert 0.0 < middle[0] < middle[1] < 1.0
