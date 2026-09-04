# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — powered depth-ordering epoch analysis tests
"""Exercise the complete frozen analysis and its custody refusals."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from scripts import analyse_iqm_dla_depth_profile_powered_epochs as analysis
from scripts import iqm_depth_ordering_protocol as protocol


def _write_calibration(path: Path, calibration_id: UUID) -> None:
    """Write one structurally valid Garnet calibration snapshot."""
    path.write_text(
        json.dumps(
            {
                "source": "IQM Resonance garnet",
                "date": "2026-09-05",
                "calibration_set_id": str(calibration_id),
                "calibration": {
                    "num_qubits": 20,
                    "edges": [[2, 7], [7, 12], [12, 13]],
                    "edge_fidelity": {},
                    "readout_error": {},
                },
            }
        ),
        encoding="utf-8",
    )


def _main_block(*, initial: str, leaked: int) -> dict[str, int]:
    """Return exact-shot counts with a selected parity-leak count."""
    if initial == "0011":
        return {"0011": protocol.MAIN_SHOTS - leaked, "0001": leaked}
    return {"0001": protocol.MAIN_SHOTS - leaked, "0011": leaked}


def _write_counts(path: Path, calibration_id: UUID, epoch: int) -> None:
    """Write one complete synthetic epoch with a positive depth contrast."""
    counts: dict[str, dict[str, int]] = {}
    for repetition in protocol.REPETITIONS:
        counts[f"main_d8_even_rep{repetition}"] = _main_block(initial="0011", leaked=110 + epoch)
        counts[f"main_d8_odd_rep{repetition}"] = _main_block(initial="0001", leaked=75 + epoch)
        counts[f"main_d12_even_rep{repetition}"] = _main_block(initial="0011", leaked=95 + epoch)
        counts[f"main_d12_odd_rep{repetition}"] = _main_block(initial="0001", leaked=85 + epoch)
    for state in protocol.READOUT_STATES:
        counts[f"readout_{state}"] = {state: protocol.READOUT_SHOTS}
    path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )


def _complete_custody(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """Create twelve distinct calibration/count pairs in epoch order."""
    count_paths: list[Path] = []
    calibration_paths: list[Path] = []
    for epoch in range(1, protocol.EPOCHS + 1):
        calibration_id = UUID(int=10_000 + epoch)
        calibration = tmp_path / f"calibration-epoch-{epoch:02d}.json"
        counts = tmp_path / f"counts-epoch-{epoch:02d}.json"
        _write_calibration(calibration, calibration_id)
        _write_counts(counts, calibration_id, epoch)
        calibration_paths.append(calibration)
        count_paths.append(counts)
    return count_paths, calibration_paths


def test_complete_custody_writes_the_frozen_primary_analysis(
    tmp_path: Path,
) -> None:
    """Twelve distinct epochs exercise the real CLI and HKSJ endpoint."""
    counts, calibrations = _complete_custody(tmp_path)
    output = tmp_path / "analysis.json"

    assert (
        analysis.main(
            [
                "--epoch-counts",
                *(str(path) for path in counts),
                "--calibrations",
                *(str(path) for path in calibrations),
                "--design",
                str(protocol.DESIGN_PATH),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    primary = report["primary_depth_ordering"]
    assert report["schema"] == protocol.ANALYSIS_SCHEMA
    assert report["design"]["sha256"] == protocol.FROZEN_DESIGN_SHA256
    assert len(report["per_epoch"]) == protocol.EPOCHS
    assert primary["degrees_of_freedom"] == 11
    assert primary["mean"] > 0.0
    assert primary["rejects_null"] is True
    assert set(report["secondary_per_depth_random_effects"]) == {"8", "12"}


def test_partial_repeated_and_design_contaminated_custody_fail_closed(
    tmp_path: Path,
) -> None:
    """No partial, repeated, or design-contaminated epoch set is analysable."""
    counts, calibrations = _complete_custody(tmp_path)
    with pytest.raises(ValueError, match="exactly 12"):
        analysis.analyse(counts[:-1], calibrations[:-1], protocol.DESIGN_PATH)

    with pytest.raises(ValueError, match="is repeated"):
        analysis.analyse(counts, [calibrations[0], *calibrations[0:11]], protocol.DESIGN_PATH)

    design = json.loads(protocol.DESIGN_PATH.read_text(encoding="utf-8"))
    excluded = UUID(design["source"]["excluded_calibration_set_ids"][0])
    _write_calibration(calibrations[0], excluded)
    _write_counts(counts[0], excluded, 1)
    with pytest.raises(ValueError, match="reuses design-evidence calibration"):
        analysis.analyse(counts, calibrations, protocol.DESIGN_PATH)


def test_matrix_and_design_drift_fail_before_statistical_output(tmp_path: Path) -> None:
    """Changed shots and changed design bytes cannot reach the endpoint."""
    counts, calibrations = _complete_custody(tmp_path)
    payload = json.loads(counts[0].read_text(encoding="utf-8"))
    payload["counts"]["main_d8_even_rep1"] = {"0011": protocol.MAIN_SHOTS - 2}
    counts[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="shots, expected"):
        analysis.analyse(counts, calibrations, protocol.DESIGN_PATH)

    changed_design = tmp_path / "changed-design.json"
    design = json.loads(protocol.DESIGN_PATH.read_text(encoding="utf-8"))
    design["frozen_design"]["alpha"] = 0.10
    changed_design.write_text(json.dumps(design), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen powered depth-ordering design digest"):
        analysis.analyse(counts, calibrations, changed_design)


def test_hksj_scale_is_safeguarded_and_degenerate_inputs_are_rejected() -> None:
    """The modified HKSJ variance cannot shrink below its conventional scale."""
    result = analysis._random_effects([0.1, 0.1, 0.1], [0.01, 0.01, 0.01])
    assert result["hksj_scale"] == pytest.approx(0.0)
    assert result["safeguarded_scale"] == 1.0
    with pytest.raises(ValueError, match="cardinality"):
        analysis._random_effects([0.1], [0.01])
    with pytest.raises(ValueError, match="positive"):
        analysis._random_effects([0.1, 0.2], [0.01, 0.0])


def test_low_level_count_and_json_validation_fail_closed(tmp_path: Path) -> None:
    """Malformed JSON roots, counts, missing arms, and changed shots are refused."""
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        analysis._object(scalar)
    with pytest.raises(ValueError, match="malformed state or count"):
        analysis._leak({"0011": True}, "0011")
    with pytest.raises(ValueError, match="empty count block"):
        analysis._leak({}, "0011")
    with pytest.raises(ValueError, match="missing count block"):
        analysis._delta({}, 8)

    counts = {
        f"main_d8_{sector}_rep{repetition}": {initial: protocol.MAIN_SHOTS}
        for sector, initial in protocol.SECTORS.items()
        for repetition in protocol.REPETITIONS
    }
    counts["main_d8_even_rep1"] = {"0011": protocol.MAIN_SHOTS - 1}
    with pytest.raises(ValueError, match="shots, expected"):
        analysis._delta(counts, 8)


def test_analysis_rejects_invalid_epoch_sequence_and_calibration_date(tmp_path: Path) -> None:
    """Epoch IDs must be typed and unique, and calibrations must retain their dates."""
    counts, calibrations = _complete_custody(tmp_path)
    payload = json.loads(counts[0].read_text(encoding="utf-8"))
    payload["epoch"] = True
    counts[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid epoch"):
        analysis.analyse(counts, calibrations, protocol.DESIGN_PATH)

    _write_counts(counts[0], UUID(int=10_001), 1)
    payload = json.loads(counts[1].read_text(encoding="utf-8"))
    payload["epoch"] = 1
    counts[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate powered"):
        analysis.analyse(counts, calibrations, protocol.DESIGN_PATH)

    _write_counts(counts[1], UUID(int=10_002), 2)
    calibration = json.loads(calibrations[0].read_text(encoding="utf-8"))
    calibration.pop("date")
    calibrations[0].write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="has no calibration date"):
        analysis.analyse(counts, calibrations, protocol.DESIGN_PATH)
