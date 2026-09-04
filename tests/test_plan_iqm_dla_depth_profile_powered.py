# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — powered depth-ordering design tests
"""Tests for the powered depth-ordering calibration-epoch design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import plan_iqm_dla_depth_profile_powered as planner

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "iqm_paper_replication"
SOURCE = (
    DATA
    / "iqm_dla_window_variability_calibration_epoch_sensitivity_through_observation_window_10_2026-09-04.json"
)
COMMITTED = DATA / "iqm_dla_depth_profile_powered_design_2026-09-04.json"


def test_window_variability_evidence_reproduces_frozen_design(tmp_path: Path) -> None:
    """The committed future-study power and budget artefact is deterministic."""
    output = tmp_path / "design.json"
    assert planner.main(["--design-evidence", str(SOURCE), "--out", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        COMMITTED.read_text(encoding="utf-8")
    )
    design = json.loads(output.read_text(encoding="utf-8"))
    assert design["frozen_design"]["projected_power"] >= 0.90
    assert design["frozen_design"]["distinct_calibration_epochs"] == 12
    assert design["matrix_and_budget"]["shots_per_epoch"] == 57_344
    assert design["matrix_and_budget"]["estimated_total_jobs"] == 24


def test_design_input_fails_closed_on_repeated_calibration(tmp_path: Path) -> None:
    """Repeated IDs cannot be mistaken for independent design epochs."""
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["epoch_level_sensitivity"]["per_epoch"]
    rows[1]["calibration_set_id"] = rows[0]["calibration_set_id"]
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="repeats a calibration_set_id"):
        planner.build_design(source)


def test_obsolete_design_evidence_schema_is_rejected(tmp_path: Path) -> None:
    """The coded predecessor contract is not retained as an input alias."""
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["schema"] = "scpn.iqm-window-variability-epoch-sensitivity.v1"
    source = tmp_path / "obsolete.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=planner.DESIGN_EVIDENCE_SCHEMA):
        planner.build_design(source)


def test_design_evidence_shape_validation_is_complete(tmp_path: Path) -> None:
    """Every required evidence container and calibration identity fails closed."""
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        planner._load(scalar)

    valid = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert isinstance(valid, dict)

    def rejected(payload: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            planner._design_rows(payload)

    rejected({}, "no epoch_level_sensitivity")
    rejected({"epoch_level_sensitivity": {"per_epoch": []}}, "fewer than two")

    epoch_level = valid["epoch_level_sensitivity"]
    rows = epoch_level["per_epoch"]
    rejected({"epoch_level_sensitivity": {"per_epoch": [None, rows[1]]}}, "not an object")
    rejected(
        {"epoch_level_sensitivity": {"per_epoch": [{}, rows[1]]}},
        "no depths object",
    )
    rejected(
        {
            "epoch_level_sensitivity": {
                "per_epoch": [{"depths": {}, "calibration_set_id": "one"}, rows[1]]
            }
        },
        "lacks depth 8 or depth 12",
    )
    missing_identity = json.loads(json.dumps(rows[0]))
    missing_identity.pop("calibration_set_id")
    rejected(
        {"epoch_level_sensitivity": {"per_epoch": [missing_identity, rows[1]]}},
        "lacks calibration_set_id",
    )


def test_design_helpers_reject_invalid_meta_analysis_and_preserve_external_path(
    tmp_path: Path,
) -> None:
    """Invalid random-effects vectors fail and non-repository provenance stays explicit."""
    with pytest.raises(ValueError, match="cardinality"):
        planner._dersimonian_laird([0.1], [0.01])
    external = tmp_path / "evidence.json"
    assert planner._display_path(external) == external.as_posix()
