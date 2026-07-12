# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — retrieve phase3 large system IBM tests
# scpn-quantum-control -- Phase 3 larger-system retrieval tests
"""Contract tests for Phase 3 larger-system IBM retrieval and reduction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module {name}")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _counts_for_expectation(expectation: float) -> dict[str, int]:
    plus = int(round((expectation + 1.0) * 50))
    return {"00": plus, "10": 100 - plus}


def _synthetic_submission() -> dict[str, object]:
    metadata_rows = []
    index = 0
    for scale in (1, 3, 5):
        metadata_rows.append(
            {
                "circuit_index": index,
                "block": "main",
                "family": "dla_parity",
                "label": "dla_odd_signal",
                "initial": "01",
                "depth": 1,
                "lambda_fim": None,
                "basis_setting": "XX",
                "rep": 0,
                "zne_noise_scale": scale,
                "calibration_state": None,
            }
        )
        index += 1
    for state in ("00", "01", "10", "11"):
        metadata_rows.append(
            {
                "circuit_index": index,
                "block": "readout_calibration",
                "family": None,
                "label": None,
                "initial": None,
                "depth": None,
                "lambda_fim": None,
                "basis_setting": None,
                "rep": 0,
                "zne_noise_scale": 1,
                "calibration_state": state,
            }
        )
        index += 1
    return {
        "schema": "scpn_phase3_large_system_submission_v1",
        "experiment_id": "synthetic_phase3_large",
        "backend": "fake_heron",
        "n_qubits": 2,
        "job_ids": ["job-large"],
        "physical_qubits": [0, 1],
        "shots": 100,
        "metadata_rows": metadata_rows,
        "claim_boundary": "synthetic test boundary",
    }


def _synthetic_result_rows(submission: dict[str, object]) -> list[dict[str, object]]:
    expectations = {1: 0.8, 3: 0.6, 5: 0.4}
    rows = []
    for meta in submission["metadata_rows"]:
        metadata = dict(meta)
        if metadata["block"] == "main":
            counts = _counts_for_expectation(expectations[int(metadata["zne_noise_scale"])])
        else:
            state = str(metadata["calibration_state"])
            counts = {state[::-1]: 100}
        rows.append({"metadata": metadata, "counts": counts, "job_id": "job-large"})
    return rows


def test_large_system_reducer_uses_submission_width_and_full_readout_calibration() -> None:
    module = _load_script_module("retrieve_phase3_large_system_ibm")
    submission = _synthetic_submission()
    raw_payload = module.raw_payload_from_rows(
        submission=submission,
        result_rows=_synthetic_result_rows(submission),
        submission_json=Path("submission.json"),
        submission_sha256="a" * 64,
        timestamp_utc="20260520T200000Z",
    )

    reference_rows = module.reference_rows_for_submission(submission)
    scale_rows, channel_rows, summary = module.analyse_large_system_payload(
        raw_payload,
        reference_rows,
    )

    assert raw_payload["n_qubits"] == 2
    assert len(reference_rows) == 1
    assert [row["zne_noise_scale"] for row in scale_rows] == [1, 3, 5]
    assert [row["mean_expectation"] for row in scale_rows] == pytest.approx([0.8, 0.6, 0.4])
    assert len(channel_rows) == 1
    assert channel_rows[0]["linear_zne_expectation"] == pytest.approx(0.9)
    assert summary["n_qubits"] == 2
    assert summary["readout_mitigation"]["method"] == "full_correlated_readout_inverse"


def test_write_outputs_preserves_raw_and_reference_hashes(tmp_path: Path) -> None:
    module = _load_script_module("retrieve_phase3_large_system_ibm")
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps({"schema": "test"}) + "\n", encoding="utf-8")
    reference_rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "initial": "01",
            "depth": 1,
            "lambda_fim": "",
            "observable": "XX_q0q1",
            "pauli_label": "XX",
            "basis_setting": "XX",
            "exact_expectation": 1.0,
        }
    ]
    scale_rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "basis_setting": "XX",
            "zne_noise_scale": 1,
            "mean_expectation": 0.8,
        }
    ]
    channel_rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "basis_setting": "XX",
            "linear_zne_expectation": 0.9,
        }
    ]
    summary = {
        "schema": "scpn_phase3_large_system_analysis_v1",
        "experiment_id": "synthetic_phase3_large",
        "backend": "fake_heron",
        "n_qubits": 2,
        "job_ids": ["job-large"],
        "n_scale_rows": 1,
        "n_channels": 1,
        "noise_scales": [1],
        "readout_mitigation": {
            "method": "full_correlated_readout_inverse",
            "n_calibration_circuits": 4,
            "claim_boundary": "test boundary",
        },
        "claim_boundary": "test claim boundary",
    }

    outputs = module.write_analysis_outputs(
        raw_path=raw_path,
        reference_rows=reference_rows,
        scale_rows=scale_rows,
        channel_rows=channel_rows,
        summary=summary,
        output_dir=tmp_path,
        docs_dir=tmp_path,
        result_tag="20260520T200000Z_fake_heron_n2",
    )

    assert outputs.analysis_json.name == (
        "phase3_large_system_analysis_20260520T200000Z_fake_heron_n2.json"
    )
    payload = json.loads(outputs.analysis_json.read_text(encoding="utf-8"))
    assert payload["raw_counts_json"].endswith("raw.json")
    assert "raw_counts_sha256" in payload
    assert "reference_rows_sha256" in payload


def test_large_system_reducer_falls_back_to_pseudoinverse_for_singular_readout() -> None:
    module = _load_script_module("retrieve_phase3_large_system_ibm")
    submission = _synthetic_submission()
    rows = _synthetic_result_rows(submission)
    for row in rows:
        metadata = row["metadata"]
        if metadata["block"] == "readout_calibration":  # type: ignore[index]
            row["counts"] = {"00": 100}
    raw_payload = module.raw_payload_from_rows(
        submission=submission,
        result_rows=rows,
        submission_json=Path("submission.json"),
        submission_sha256="a" * 64,
        timestamp_utc="20260520T200000Z",
    )

    reference_rows = module.reference_rows_for_submission(submission)
    scale_rows, channel_rows, summary = module.analyse_large_system_payload(
        raw_payload,
        reference_rows,
    )

    assert scale_rows
    assert channel_rows
    assert summary["readout_mitigation"]["method"] == "full_correlated_readout_pseudoinverse"
    assert summary["readout_mitigation"]["rank"] < 4
