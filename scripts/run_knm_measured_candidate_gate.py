#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- K_nm measured-candidate promotion gate
"""Validate committed K_nm measured-candidate audit artefacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExpectedCandidate:
    """Expected non-promotional state for one measured K_nm candidate."""

    candidate_id: str
    path: str
    unit: str
    unit_promotable: bool
    unit_classification: str
    matched_edges: int


EXPECTED_CANDIDATES: dict[str, ExpectedCandidate] = {
    "eeg_alpha_plv": ExpectedCandidate(
        candidate_id="eeg_alpha_plv",
        path="data/knm_physical_validation/eeg_alpha_plv_knm_comparison.json",
        unit="phase_locking_value",
        unit_promotable=False,
        unit_classification="association_observable_not_coupling_magnitude",
        matched_edges=28,
    ),
    "power_grid_ieee5bus": ExpectedCandidate(
        candidate_id="power_grid_ieee5bus",
        path="data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json",
        unit="dimensionless_swing_equation_coupling",
        unit_promotable=True,
        unit_classification="calibrated_or_model_derived_coupling_magnitude",
        matched_edges=10,
    ),
    "power_grid_ieee14bus": ExpectedCandidate(
        candidate_id="power_grid_ieee14bus",
        path="data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json",
        unit="per_unit_voltage_weighted_susceptance",
        unit_promotable=True,
        unit_classification="calibrated_or_model_derived_coupling_magnitude",
        matched_edges=91,
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_candidate_payload(
    expected: ExpectedCandidate,
    payload: dict[str, Any],
) -> tuple[str, ...]:
    """Return blockers for one measured-candidate audit payload."""

    blockers: list[str] = []
    readiness = payload.get("measured_system_promotion_readiness")
    decision = payload.get("decision")
    if not isinstance(readiness, dict):
        return (f"{expected.candidate_id}: missing measured-system readiness block",)
    if not isinstance(decision, dict):
        return (f"{expected.candidate_id}: missing decision block",)

    if payload.get("schema_version") != 3:
        blockers.append(f"{expected.candidate_id}: schema_version must be 3")
    if decision.get("physical_validation_closed") is not False:
        blockers.append(f"{expected.candidate_id}: physical validation must remain open")
    if readiness.get("ready") is not False:
        blockers.append(f"{expected.candidate_id}: promotion readiness must remain false")
    if readiness.get("decision") != "blocked_measured_system_promotion_gate":
        blockers.append(f"{expected.candidate_id}: readiness decision must stay blocked")
    if readiness.get("matched_edges") != expected.matched_edges:
        blockers.append(
            f"{expected.candidate_id}: matched edge count drift "
            f"{readiness.get('matched_edges')!r} != {expected.matched_edges}"
        )
    if readiness.get("coupling_unit") != expected.unit:
        blockers.append(
            f"{expected.candidate_id}: coupling unit drift "
            f"{readiness.get('coupling_unit')!r} != {expected.unit!r}"
        )
    if readiness.get("coupling_unit_promotable") is not expected.unit_promotable:
        blockers.append(f"{expected.candidate_id}: unit-class promotability drift")

    unit_status = readiness.get("unit_promotion_status", {})
    if not isinstance(unit_status, dict):
        blockers.append(f"{expected.candidate_id}: missing unit promotion-status block")
    elif unit_status.get("classification") != expected.unit_classification:
        blockers.append(
            f"{expected.candidate_id}: unit classification drift "
            f"{unit_status.get('classification')!r} != {expected.unit_classification!r}"
        )
    return tuple(blockers)


def validate_committed_candidate_artifacts(
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate every committed measured-candidate audit artefact."""

    blockers: list[str] = []
    candidates: list[dict[str, Any]] = []
    for expected in EXPECTED_CANDIDATES.values():
        path = repo_root / expected.path
        if not path.exists():
            blockers.append(f"{expected.candidate_id}: missing artifact {expected.path}")
            continue
        payload = _load_json(path)
        candidate_blockers = validate_candidate_payload(expected, payload)
        blockers.extend(candidate_blockers)
        readiness = payload.get("measured_system_promotion_readiness", {})
        candidates.append(
            {
                "candidate_id": expected.candidate_id,
                "path": expected.path,
                "unit": readiness.get("coupling_unit"),
                "unit_classification": readiness.get("unit_promotion_status", {}).get(
                    "classification"
                ),
                "matched_edges": readiness.get("matched_edges"),
                "ready": readiness.get("ready"),
                "blockers": candidate_blockers,
            }
        )

    return {
        "valid": not blockers,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "blockers": tuple(blockers),
    }


def main() -> int:
    """Run the measured-candidate gate."""

    result = validate_committed_candidate_artifacts()
    for candidate in result["candidates"]:
        print(
            "[knm-measured-candidate-gate] "
            f"{candidate['candidate_id']}: unit={candidate['unit']} "
            f"class={candidate['unit_classification']} ready={candidate['ready']}"
        )
    if result["valid"]:
        print("[knm-measured-candidate-gate] all committed candidates remain blocked")
        return 0
    print("[knm-measured-candidate-gate] blockers:")
    for blocker in result["blockers"]:
        print(f"  - {blocker}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
