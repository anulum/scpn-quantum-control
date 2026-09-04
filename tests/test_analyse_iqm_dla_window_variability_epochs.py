# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — calibration-epoch sensitivity tests
"""Tests for the prospective calibration-epoch sensitivity analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyse_iqm_dla_window_variability_epochs.py"
DATA = REPO_ROOT / "data" / "iqm_paper_replication"
COUNTS = tuple(
    sorted(
        DATA.glob("iqm_dla_window_variability_hardware_counts_observation_window_0[1-7]_*.json")
    )
)
CALIBRATIONS = tuple(
    sorted(DATA.glob("iqm_dla_window_variability_calibration_observation_window_0[1-7]_*.json"))
)
FROZEN = DATA / "iqm_dla_window_variability_analysis_through_observation_window_07_2026-09-04.json"
COMMITTED = (
    DATA
    / "iqm_dla_window_variability_calibration_epoch_sensitivity_through_observation_window_07_2026-09-04.json"
)


def _load_script() -> ModuleType:
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("_iqm_epoch_sensitivity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load_script()


def _argv(output: Path) -> list[str]:
    return [
        "--window-counts",
        *(str(path) for path in COUNTS),
        "--calibrations",
        *(str(path) for path in CALIBRATIONS),
        "--frozen-analysis",
        str(FROZEN),
        "--out",
        str(output),
    ]


def test_first_seven_observation_windows_reproduce_epoch_sensitivity(
    tmp_path: Path,
) -> None:
    """Real custody evidence maps seven windows to six deterministic epochs."""
    output = tmp_path / "epochs.json"
    assert analysis.main(_argv(output)) == 0
    observed = json.loads(output.read_text(encoding="utf-8"))
    expected = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert observed == expected
    epoch = observed["epoch_level_sensitivity"]
    assert epoch["nominal_windows"] == 7
    assert epoch["calibration_epochs"] == 6
    assert epoch["primary_d10_heterogeneity"]["degrees_of_freedom"] == 5
    assert epoch["technical_replicate_groups"] == [
        {
            "calibration_set_id": "c2097be4-1e23-49bc-adaa-8e8c01df6223",
            "nominal_windows": [2, 3],
        }
    ]


def test_status_advances_after_first_post_amendment_window() -> None:
    """Post-amendment reports must not retain the prospective status label."""
    records = analysis._records(list(COUNTS), list(CALIBRATIONS))
    assert analysis._report_status(records) == "prospective_calibration_epoch_sensitivity"
    records.append({**records[-1], "window": 8})
    assert analysis._report_status(records) == "post_amendment_calibration_epoch_sensitivity"


def test_same_epoch_pools_raw_shots_without_new_degree_of_freedom() -> None:
    """Technical replicates double arm precision but remain one epoch."""
    records = analysis._records(list(COUNTS[:3]), list(CALIBRATIONS[:3]))
    report = analysis._epoch_report(records)
    assert report["nominal_windows"] == 3
    assert report["calibration_epochs"] == 2
    assert report["analysable"] is False
    repeated = report["per_epoch"][1]
    assert repeated["nominal_windows"] == [2, 3]
    assert repeated["depths"]["10"]["shots_even"] == 8192
    assert repeated["depths"]["10"]["shots_odd"] == 8192
    assert report["primary_d10_heterogeneity"]["degrees_of_freedom"] == 1


def test_input_pairing_fails_closed(tmp_path: Path) -> None:
    """Mismatched evidence cardinality and dates cannot invent an epoch map."""
    with pytest.raises(ValueError, match="path counts differ"):
        analysis._records(list(COUNTS[:2]), list(CALIBRATIONS[:1]))

    calibration = json.loads(CALIBRATIONS[0].read_text(encoding="utf-8"))
    calibration["date"] = "2099-01-01"
    bad = tmp_path / "bad-calibration.json"
    bad.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="dates differ"):
        analysis._records([COUNTS[0]], [bad])
