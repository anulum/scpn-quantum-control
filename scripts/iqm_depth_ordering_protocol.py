# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM powered depth-ordering protocol contract
"""Validate the frozen design, calibration epochs, and retrieved count custody."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPO_ROOT
    / "data"
    / "iqm_paper_replication"
    / "iqm_dla_depth_profile_powered_design_2026-09-04.json"
)
CAMPAIGN_ID = "iqm_dla_depth_profile_powered_epoch_prereg_2026-09-04"
DESIGN_SCHEMA = "scpn.iqm-dla-depth-profile-powered-design.v1"
DESIGN_EVIDENCE_SCHEMA = "scpn.iqm-window-variability-epoch-sensitivity.v2"
RETRIEVED_COUNTS_SCHEMA = "scpn.iqm-retrieved-counts.v1"
ANALYSIS_SCHEMA = "scpn.iqm-dla-depth-profile-powered-analysis.v1"
FROZEN_DESIGN_SHA256 = "df7f9df1b914dd7cf432be35e94a57b6c19c8c48065f9d673f0525bb1e8d25f5"
PRIMARY_LAYOUT = (2, 7, 12, 13)
DEPTHS = (8, 12)
SECTORS = {"even": "0011", "odd": "0001"}
REPETITIONS = tuple(range(1, 13))
READOUT_STATES = ("0011", "0001", "0000", "1111")
EPOCHS = 12
MAIN_SHOTS = 1024
READOUT_SHOTS = 2048
ALPHA = 0.05


@dataclass(frozen=True)
class FrozenDesign:
    """Validated identity and excluded calibration set of the frozen design."""

    sha256: str
    excluded_calibration_set_ids: frozenset[str]


@dataclass(frozen=True)
class EpochAdmission:
    """Validated current epoch identity to persist before provider contact."""

    calibration_set_id: str
    design_sha256: str


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object, rejecting arrays and scalar roots."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def canonical_uuid(value: object, *, field: str, path: Path) -> str:
    """Return a canonical UUID string from one required payload field."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} has no {field}")
    try:
        canonical = str(UUID(value))
    except ValueError as exc:
        raise ValueError(f"{path} has a non-UUID {field}") from exc
    if value != canonical:
        raise ValueError(f"{path} has a non-canonical {field}")
    return canonical


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one custody file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_design(path: Path = DESIGN_PATH) -> FrozenDesign:
    """Require the exact preregistered design bytes and semantic contract."""
    digest = _sha256(path)
    if digest != FROZEN_DESIGN_SHA256:
        raise ValueError(f"{path} differs from the frozen powered depth-ordering design digest")
    payload = load_json_object(path)
    if payload.get("schema") != DESIGN_SCHEMA:
        raise ValueError(f"{path} is not a {DESIGN_SCHEMA} design")
    if payload.get("campaign") != CAMPAIGN_ID:
        raise ValueError(f"{path} has the wrong powered depth-ordering campaign")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("schema") != DESIGN_EVIDENCE_SCHEMA:
        raise ValueError(f"{path} has the wrong calibration-epoch design evidence schema")
    if source.get("role") != "design_only_excluded_from_confirmatory_endpoint":
        raise ValueError(f"{path} does not exclude its design evidence from confirmation")
    raw_excluded = source.get("excluded_calibration_set_ids")
    if not isinstance(raw_excluded, list) or len(raw_excluded) != 6:
        raise ValueError(f"{path} must exclude the six design-evidence calibrations")
    excluded = frozenset(
        canonical_uuid(value, field="excluded calibration_set_id", path=path)
        for value in raw_excluded
    )
    if len(excluded) != len(raw_excluded):
        raise ValueError(f"{path} repeats an excluded calibration_set_id")

    frozen = payload.get("frozen_design")
    required_frozen = {
        "primary_contrast": "(delta_8 - delta_12) > 0",
        "alpha": ALPHA,
        "sidedness": "one-sided",
        "decision_method": "safeguarded_HKSJ_random_effects_t",
        "distinct_calibration_epochs": EPOCHS,
        "repetitions_per_state_depth_epoch": len(REPETITIONS),
        "shots_per_repetition": MAIN_SHOTS,
    }
    if not isinstance(frozen, dict) or any(
        frozen.get(field) != expected for field, expected in required_frozen.items()
    ):
        raise ValueError(f"{path} differs from the frozen primary analysis contract")
    budget = payload.get("matrix_and_budget")
    required_budget = {
        "main_circuits_per_epoch": 48,
        "readout_circuits_per_epoch": 4,
        "circuits_per_epoch": 52,
        "main_shots_per_epoch": 49_152,
        "readout_shots_per_epoch": 8_192,
        "shots_per_epoch": 57_344,
        "total_shots": 688_128,
        "jobs_per_epoch": 2,
        "estimated_total_jobs": 24,
    }
    if not isinstance(budget, dict) or any(
        budget.get(field) != expected for field, expected in required_budget.items()
    ):
        raise ValueError(f"{path} differs from the frozen matrix and budget")
    return FrozenDesign(digest, excluded)


def validate_calibration_snapshot(
    payload: dict[str, Any], path: Path, *, expected_date: str
) -> str:
    """Validate one fresh Garnet snapshot and return its calibration UUID."""
    calibration_id = canonical_uuid(
        payload.get("calibration_set_id"), field="calibration_set_id", path=path
    )
    if payload.get("date") != expected_date:
        raise ValueError(f"{path} calibration and submission dates differ")
    if payload.get("source") != "IQM Resonance garnet":
        raise ValueError(f"{path} is not an IQM Resonance Garnet calibration snapshot")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"{path} has no calibration object")
    raw_edges = calibration.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError(f"{path} has no calibration edge list")
    edges: set[tuple[int, int]] = set()
    for edge in raw_edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in edge)
            or edge[0] == edge[1]
        ):
            raise ValueError(f"{path} has a malformed calibration edge")
        edges.add(tuple(sorted(edge)))
    required_edges = {tuple(sorted(edge)) for edge in pairwise(PRIMARY_LAYOUT)}
    missing = sorted(required_edges - edges)
    if missing:
        raise ValueError(f"{path} lacks primary-layout edges {missing}")
    return calibration_id


def expected_count_labels() -> set[str]:
    """Return the exact 52-label matrix for one confirmatory epoch."""
    main = {
        f"main_d{depth}_{sector}_rep{repetition}"
        for repetition in REPETITIONS
        for depth in DEPTHS
        for sector in SECTORS
    }
    return main | {f"readout_{state}" for state in READOUT_STATES}


def _validate_count_block(block: object, *, label: str, expected_shots: int) -> None:
    """Require a non-negative integer count mapping with an exact shot total."""
    if not isinstance(block, dict) or not block:
        raise ValueError(f"missing or empty count block {label}")
    if not all(
        isinstance(state, str)
        and state
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for state, value in block.items()
    ):
        raise ValueError(f"count block {label} has a malformed state or count")
    total = sum(block.values())
    if total != expected_shots:
        raise ValueError(f"{label} has {total} shots, expected {expected_shots}")


def validate_retrieved_counts(
    payload: dict[str, Any],
    path: Path,
    *,
    expected_epoch: int,
    calibration_set_id: str,
    calibration_date: str,
    design_sha256: str,
) -> None:
    """Validate one complete retrieved epoch against design and calibration custody."""
    expected_fields = {
        "schema": RETRIEVED_COUNTS_SCHEMA,
        "campaign": CAMPAIGN_ID,
        "backend": "garnet",
        "date": calibration_date,
        "repetition": 1,
        "window": 0,
        "epoch": expected_epoch,
        "calibration_set_id": calibration_set_id,
        "design_sha256": design_sha256,
        "layout": list(PRIMARY_LAYOUT),
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise ValueError(f"{path} has the wrong {field}")
    raw_jobs = payload.get("job_ids")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 2:
        raise ValueError(f"{path} must contain exactly two provider job IDs")
    jobs = [canonical_uuid(value, field="job_id", path=path) for value in raw_jobs]
    if len(set(jobs)) != len(jobs):
        raise ValueError(f"{path} repeats a provider job ID")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or set(counts) != expected_count_labels():
        raise ValueError(f"{path} count labels differ from the frozen 52-circuit matrix")
    for label, block in counts.items():
        expected_shots = READOUT_SHOTS if label.startswith("readout_") else MAIN_SHOTS
        _validate_count_block(block, label=label, expected_shots=expected_shots)


def validate_epoch_admission(
    *,
    epoch: int,
    layout: tuple[int, int, int, int],
    layout_choice: str,
    submission_date: str,
    calibration_path: Path,
    prior_calibration_paths: list[Path],
    prior_count_paths: list[Path],
    design_path: Path = DESIGN_PATH,
) -> EpochAdmission:
    """Admit one new calibration epoch only after complete prior-epoch custody."""
    if layout_choice != "primary" or layout != PRIMARY_LAYOUT:
        raise ValueError("powered depth ordering requires the frozen primary layout")
    if not 1 <= epoch <= EPOCHS:
        raise ValueError(f"powered depth-ordering epoch must be in the frozen range 1..{EPOCHS}")
    if len(prior_calibration_paths) != epoch - 1 or len(prior_count_paths) != epoch - 1:
        raise ValueError(
            f"powered depth-ordering epoch {epoch} requires exactly {epoch - 1} "
            "prior calibration and retrieved-count pairs"
        )
    design = validate_frozen_design(design_path)
    current = load_json_object(calibration_path)
    current_id = validate_calibration_snapshot(
        current, calibration_path, expected_date=submission_date
    )
    if current_id in design.excluded_calibration_set_ids:
        raise ValueError("powered depth-ordering calibration was used as design evidence")

    prior_ids: list[str] = []
    for expected_epoch, (prior_calibration_path, prior_count_path) in enumerate(
        zip(prior_calibration_paths, prior_count_paths, strict=True), start=1
    ):
        calibration = load_json_object(prior_calibration_path)
        prior_date = calibration.get("date")
        if not isinstance(prior_date, str) or not prior_date:
            raise ValueError(f"{prior_calibration_path} has no calibration date")
        prior_id = validate_calibration_snapshot(
            calibration, prior_calibration_path, expected_date=prior_date
        )
        if prior_id in design.excluded_calibration_set_ids:
            raise ValueError("a prior confirmatory calibration was used as design evidence")
        validate_retrieved_counts(
            load_json_object(prior_count_path),
            prior_count_path,
            expected_epoch=expected_epoch,
            calibration_set_id=prior_id,
            calibration_date=prior_date,
            design_sha256=design.sha256,
        )
        prior_ids.append(prior_id)
    if len(prior_ids) != len(set(prior_ids)):
        raise ValueError("prior confirmatory calibration list contains a duplicate epoch")
    if current_id in prior_ids:
        raise ValueError("powered depth-ordering calibration was used by an earlier epoch")
    return EpochAdmission(current_id, design.sha256)


__all__ = [
    "ALPHA",
    "ANALYSIS_SCHEMA",
    "CAMPAIGN_ID",
    "DEPTHS",
    "DESIGN_EVIDENCE_SCHEMA",
    "DESIGN_PATH",
    "DESIGN_SCHEMA",
    "EPOCHS",
    "EpochAdmission",
    "FROZEN_DESIGN_SHA256",
    "FrozenDesign",
    "MAIN_SHOTS",
    "PRIMARY_LAYOUT",
    "READOUT_SHOTS",
    "READOUT_STATES",
    "REPETITIONS",
    "RETRIEVED_COUNTS_SCHEMA",
    "SECTORS",
    "canonical_uuid",
    "expected_count_labels",
    "load_json_object",
    "validate_calibration_snapshot",
    "validate_epoch_admission",
    "validate_frozen_design",
    "validate_retrieved_counts",
]
