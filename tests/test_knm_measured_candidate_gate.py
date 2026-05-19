# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- K_nm measured-candidate gate tests
"""Tests for the K_nm measured-candidate promotion-boundary gate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_knm_measured_candidate_gate import (
    EXPECTED_CANDIDATES,
    validate_candidate_payload,
    validate_committed_candidate_artifacts,
)


def _candidate_payload(
    *,
    unit: str,
    classification: str,
    unit_promotable: bool,
    matched_edges: int,
    physical_validation_closed: bool = False,
    ready: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "decision": {"physical_validation_closed": physical_validation_closed},
        "measured_system_promotion_readiness": {
            "ready": ready,
            "decision": "blocked_measured_system_promotion_gate",
            "matched_edges": matched_edges,
            "coupling_unit": unit,
            "coupling_unit_promotable": unit_promotable,
            "unit_promotion_status": {"classification": classification},
        },
    }


def test_validate_candidate_payload_accepts_blocked_eeg_plv_candidate() -> None:
    expected = EXPECTED_CANDIDATES["eeg_alpha_plv"]
    payload = _candidate_payload(
        unit="phase_locking_value",
        classification="association_observable_not_coupling_magnitude",
        unit_promotable=False,
        matched_edges=28,
    )

    assert validate_candidate_payload(expected, payload) == ()


def test_validate_candidate_payload_rejects_promoted_candidate() -> None:
    expected = EXPECTED_CANDIDATES["eeg_alpha_plv"]
    payload = _candidate_payload(
        unit="phase_locking_value",
        classification="association_observable_not_coupling_magnitude",
        unit_promotable=False,
        matched_edges=28,
        physical_validation_closed=True,
        ready=True,
    )

    blockers = validate_candidate_payload(expected, payload)

    assert any("physical validation must remain open" in blocker for blocker in blockers)
    assert any("promotion readiness must remain false" in blocker for blocker in blockers)


def test_validate_candidate_payload_rejects_unit_class_drift() -> None:
    expected = EXPECTED_CANDIDATES["eeg_alpha_plv"]
    payload = _candidate_payload(
        unit="phase_locking_value",
        classification="calibrated_or_model_derived_coupling_magnitude",
        unit_promotable=True,
        matched_edges=28,
    )

    blockers = validate_candidate_payload(expected, payload)

    assert any("unit-class promotability drift" in blocker for blocker in blockers)
    assert any("unit classification drift" in blocker for blocker in blockers)


def test_validate_committed_candidate_artifacts_checks_all_expected_files(
    tmp_path: Path,
) -> None:
    for expected in EXPECTED_CANDIDATES.values():
        path = tmp_path / expected.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _candidate_payload(
                    unit=expected.unit,
                    classification=expected.unit_classification,
                    unit_promotable=expected.unit_promotable,
                    matched_edges=expected.matched_edges,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    result = validate_committed_candidate_artifacts(repo_root=tmp_path)

    assert result["valid"] is True
    assert result["blockers"] == ()
    assert {item["candidate_id"] for item in result["candidates"]} == set(EXPECTED_CANDIDATES)
