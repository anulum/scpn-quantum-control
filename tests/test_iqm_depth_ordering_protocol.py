# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — powered depth-ordering protocol tests
"""Exercise fail-closed design, calibration, count, and epoch custody."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import pytest

from scripts import iqm_depth_ordering_protocol as protocol


def _write(path: Path, payload: object) -> None:
    """Write one JSON test payload."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _design() -> dict[str, Any]:
    """Return a detached copy of the frozen design object."""
    payload = json.loads(protocol.DESIGN_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _calibration(calibration_id: UUID, *, date: str = "2026-09-05") -> dict[str, Any]:
    """Return one valid primary-layout calibration snapshot."""
    return {
        "source": "IQM Resonance garnet",
        "date": date,
        "calibration_set_id": str(calibration_id),
        "calibration": {
            "num_qubits": 20,
            "edges": [[2, 7], [7, 12], [12, 13]],
            "edge_fidelity": {},
            "readout_error": {},
        },
    }


def _retrieved_counts(calibration_id: UUID, *, epoch: int = 1) -> dict[str, Any]:
    """Return one complete frozen-matrix retrieved-count payload."""
    counts = {
        label: {
            "0000": (
                protocol.READOUT_SHOTS if label.startswith("readout_") else protocol.MAIN_SHOTS
            )
        }
        for label in protocol.expected_count_labels()
    }
    return {
        "schema": protocol.RETRIEVED_COUNTS_SCHEMA,
        "campaign": protocol.CAMPAIGN_ID,
        "backend": "garnet",
        "date": "2026-09-05",
        "repetition": 1,
        "window": 0,
        "epoch": epoch,
        "calibration_set_id": str(calibration_id),
        "design_sha256": protocol.FROZEN_DESIGN_SHA256,
        "layout": list(protocol.PRIMARY_LAYOUT),
        "job_ids": [str(UUID(int=epoch * 2)), str(UUID(int=epoch * 2 + 1))],
        "counts": counts,
    }


def test_json_object_and_uuid_contracts_reject_ambiguous_identity(tmp_path: Path) -> None:
    """Scalar JSON and missing, malformed, or noncanonical UUIDs fail closed."""
    scalar = tmp_path / "scalar.json"
    _write(scalar, [])
    with pytest.raises(ValueError, match="JSON object"):
        protocol.load_json_object(scalar)

    with pytest.raises(ValueError, match="has no identity"):
        protocol.canonical_uuid(None, field="identity", path=scalar)
    with pytest.raises(ValueError, match="non-UUID identity"):
        protocol.canonical_uuid("invalid", field="identity", path=scalar)
    with pytest.raises(ValueError, match="non-canonical identity"):
        protocol.canonical_uuid(
            "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA", field="identity", path=scalar
        )


def test_frozen_design_rejects_every_semantic_contract_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A matching digest cannot conceal changed identity, evidence, analysis, or budget."""
    monkeypatch.setattr(protocol, "_sha256", lambda _path: protocol.FROZEN_DESIGN_SHA256)
    target = tmp_path / "design.json"

    def rejected(mutate: Callable[[dict[str, Any]], None], message: str) -> None:
        payload = _design()
        mutate(payload)
        _write(target, payload)
        with pytest.raises(ValueError, match=message):
            protocol.validate_frozen_design(target)

    rejected(lambda value: value.update(schema="wrong"), "is not")
    rejected(lambda value: value.update(campaign="wrong"), "wrong powered")
    rejected(lambda value: value.update(source=None), "design evidence schema")
    rejected(lambda value: value["source"].update(schema="wrong"), "design evidence schema")
    rejected(lambda value: value["source"].update(role="wrong"), "does not exclude")
    rejected(
        lambda value: value["source"].update(excluded_calibration_set_ids=[]),
        "six design-evidence",
    )
    rejected(
        lambda value: value["source"].update(excluded_calibration_set_ids=[str(UUID(int=1))] * 6),
        "repeats an excluded",
    )
    rejected(lambda value: value.update(frozen_design=None), "primary analysis contract")
    rejected(
        lambda value: value["frozen_design"].update(alpha=0.1),
        "primary analysis contract",
    )
    rejected(lambda value: value.update(matrix_and_budget=None), "matrix and budget")
    rejected(
        lambda value: value["matrix_and_budget"].update(total_shots=1),
        "matrix and budget",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(source="other"), "not an IQM Resonance"),
        (lambda value: value.update(calibration=None), "no calibration object"),
        (lambda value: value["calibration"].update(edges=None), "no calibration edge list"),
        (lambda value: value["calibration"].update(edges=[[2, 2]]), "malformed"),
        (lambda value: value["calibration"].update(edges=[[2, True]]), "malformed"),
    ],
)
def test_calibration_snapshot_rejects_malformed_provider_evidence(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
    tmp_path: Path,
) -> None:
    """Provider identity, structure, edge types, and topology are mandatory."""
    payload = _calibration(UUID(int=100))
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        protocol.validate_calibration_snapshot(
            payload,
            tmp_path / "calibration.json",
            expected_date="2026-09-05",
        )


def test_retrieved_counts_reject_identity_job_and_matrix_drift(tmp_path: Path) -> None:
    """Every retrieved identity field, provider job, label, state, and shot is frozen."""
    calibration_id = UUID(int=100)
    path = tmp_path / "counts.json"

    def rejected(mutate: Callable[[dict[str, Any]], None], message: str) -> None:
        payload = _retrieved_counts(calibration_id)
        mutate(payload)
        with pytest.raises(ValueError, match=message):
            protocol.validate_retrieved_counts(
                payload,
                path,
                expected_epoch=1,
                calibration_set_id=str(calibration_id),
                calibration_date="2026-09-05",
                design_sha256=protocol.FROZEN_DESIGN_SHA256,
            )

    rejected(lambda value: value.update(backend="other"), "wrong backend")
    rejected(lambda value: value.update(job_ids=[]), "exactly two provider job IDs")
    rejected(
        lambda value: value.update(job_ids=[str(UUID(int=2)), str(UUID(int=2))]),
        "repeats a provider job ID",
    )
    rejected(lambda value: value.update(counts={}), "count labels differ")

    def empty_block(value: dict[str, Any]) -> None:
        value["counts"][next(iter(value["counts"]))] = {}

    def malformed_block(value: dict[str, Any]) -> None:
        value["counts"][next(iter(value["counts"]))] = {"0000": True}

    def wrong_shots(value: dict[str, Any]) -> None:
        value["counts"]["main_d8_even_rep1"] = {"0000": 1}

    rejected(empty_block, "missing or empty count block")
    rejected(malformed_block, "malformed state or count")
    rejected(wrong_shots, "shots, expected")


def test_epoch_admission_rejects_layout_range_and_incomplete_custody(tmp_path: Path) -> None:
    """Admission refuses noncanonical layout, epoch, and predecessor cardinality."""
    calibration = tmp_path / "calibration.json"
    _write(calibration, _calibration(UUID(int=100)))
    common: dict[str, Any] = {
        "layout": protocol.PRIMARY_LAYOUT,
        "layout_choice": "primary",
        "submission_date": "2026-09-05",
        "calibration_path": calibration,
        "prior_calibration_paths": [],
        "prior_count_paths": [],
    }
    with pytest.raises(ValueError, match="frozen primary layout"):
        protocol.validate_epoch_admission(epoch=1, **{**common, "layout_choice": "fallback"})
    with pytest.raises(ValueError, match="frozen range"):
        protocol.validate_epoch_admission(epoch=0, **common)
    with pytest.raises(ValueError, match="exactly 1 prior"):
        protocol.validate_epoch_admission(epoch=2, **common)


def test_epoch_admission_rejects_contaminated_and_repeated_predecessors(tmp_path: Path) -> None:
    """Design evidence, duplicate predecessors, and current-ID reuse remain inadmissible."""
    design = _design()
    excluded = UUID(design["source"]["excluded_calibration_set_ids"][0])
    current = tmp_path / "current.json"
    prior_one = tmp_path / "prior-one.json"
    prior_two = tmp_path / "prior-two.json"
    counts_one = tmp_path / "counts-one.json"
    counts_two = tmp_path / "counts-two.json"

    _write(current, _calibration(UUID(int=102)))
    _write(prior_one, _calibration(excluded))
    _write(counts_one, _retrieved_counts(excluded))
    with pytest.raises(ValueError, match="prior confirmatory calibration was used"):
        protocol.validate_epoch_admission(
            epoch=2,
            layout=protocol.PRIMARY_LAYOUT,
            layout_choice="primary",
            submission_date="2026-09-05",
            calibration_path=current,
            prior_calibration_paths=[prior_one],
            prior_count_paths=[counts_one],
        )

    repeated = UUID(int=101)
    _write(prior_one, _calibration(repeated))
    _write(prior_two, _calibration(repeated))
    _write(counts_one, _retrieved_counts(repeated, epoch=1))
    _write(counts_two, _retrieved_counts(repeated, epoch=2))
    with pytest.raises(ValueError, match="duplicate epoch"):
        protocol.validate_epoch_admission(
            epoch=3,
            layout=protocol.PRIMARY_LAYOUT,
            layout_choice="primary",
            submission_date="2026-09-05",
            calibration_path=current,
            prior_calibration_paths=[prior_one, prior_two],
            prior_count_paths=[counts_one, counts_two],
        )

    _write(current, _calibration(repeated))
    with pytest.raises(ValueError, match="used by an earlier epoch"):
        protocol.validate_epoch_admission(
            epoch=2,
            layout=protocol.PRIMARY_LAYOUT,
            layout_choice="primary",
            submission_date="2026-09-05",
            calibration_path=current,
            prior_calibration_paths=[prior_one],
            prior_count_paths=[counts_one],
        )


def test_epoch_admission_requires_prior_date(tmp_path: Path) -> None:
    """A predecessor without its calibration date cannot establish custody."""
    current = tmp_path / "current.json"
    prior = tmp_path / "prior.json"
    counts = tmp_path / "counts.json"
    _write(current, _calibration(UUID(int=102)))
    prior_payload = _calibration(UUID(int=101))
    prior_payload.pop("date")
    _write(prior, prior_payload)
    _write(counts, _retrieved_counts(UUID(int=101)))
    with pytest.raises(ValueError, match="has no calibration date"):
        protocol.validate_epoch_admission(
            epoch=2,
            layout=protocol.PRIMARY_LAYOUT,
            layout_choice="primary",
            submission_date="2026-09-05",
            calibration_path=current,
            prior_calibration_paths=[prior],
            prior_count_paths=[counts],
        )
