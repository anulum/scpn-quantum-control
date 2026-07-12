# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase3 entanglement ZNE analysis tests
# scpn-quantum-control -- Phase 3 ZNE analysis tests
"""Tests for the Phase 3 reduced-Pauli ZNE reducer."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyse_phase3_entanglement_zne.py"
    spec = importlib.util.spec_from_file_location("analyse_phase3_entanglement_zne", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 3 ZNE analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _counts_for_expectation(expectation: float) -> dict[str, int]:
    plus = int(round((expectation + 1.0) * 50))
    minus = 100 - plus
    return {"0000": plus, "1000": minus}


def _write_reference(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "label",
                "initial",
                "depth",
                "lambda_fim",
                "observable",
                "pauli_label",
                "basis_setting",
                "exact_expectation",
                "half_chain_purity",
                "parity_survival_ideal",
                "shots_per_setting",
                "repetitions",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "family": "dla_parity",
                "label": "dla_odd_signal",
                "initial": "0001",
                "depth": "10",
                "lambda_fim": "",
                "observable": "XX_q0q1",
                "pauli_label": "XXII",
                "basis_setting": "XXII",
                "exact_expectation": "1.0",
                "half_chain_purity": "0.5",
                "parity_survival_ideal": "1.0",
                "shots_per_setting": "2048",
                "repetitions": "3",
            }
        )


def _write_counts(path: Path) -> None:
    circuits: list[dict[str, object]] = []
    for scale, expectation in [(1, 0.8), (3, 0.6), (5, 0.4)]:
        circuits.append(
            {
                "job_id": "main-job",
                "meta": {
                    "block": "main",
                    "family": "dla_parity",
                    "label": "dla_odd_signal",
                    "initial": "0001",
                    "depth": 10,
                    "lambda_fim": None,
                    "basis_setting": "XXII",
                    "rep": 0,
                    "zne_noise_scale": scale,
                },
                "counts": _counts_for_expectation(expectation),
            }
        )
    for state in [format(index, "04b") for index in range(16)]:
        circuits.append(
            {
                "job_id": "readout-job",
                "meta": {"block": "readout", "initial": state},
                "counts": {state[::-1]: 100},
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "scpn_phase3_entanglement_tomography_live_v1",
                "backend": "fake_heron",
                "status": "completed",
                "job_ids": ["main-job", "readout-job"],
                "circuits": circuits,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_zne_reducer_groups_by_noise_scale_and_extrapolates(tmp_path: Path) -> None:
    module = _load_module()
    counts = tmp_path / "counts.json"
    refs = tmp_path / "refs.csv"
    _write_counts(counts)
    _write_reference(refs)

    scale_rows, channel_rows, summary = module.analyse_zne_counts_artifact(counts, refs)

    assert [row["zne_noise_scale"] for row in scale_rows] == [1, 3, 5]
    assert [row["mean_expectation"] for row in scale_rows] == pytest.approx([0.8, 0.6, 0.4])
    assert len(channel_rows) == 1
    row = channel_rows[0]
    assert row["label"] == "dla_odd_signal"
    assert row["basis_setting"] == "XXII"
    assert row["linear_zne_expectation"] == pytest.approx(0.9)
    assert row["linear_zne_absolute_deviation"] == pytest.approx(0.1)
    assert row["quadratic_zne_expectation"] == pytest.approx(0.9)
    assert summary["readout_mitigation"]["method"] == "full_correlated_readout_inverse"
    assert summary["n_channels"] == 1
    assert summary["scale1_mean_absolute_deviation"] == pytest.approx(0.2)
    assert summary["quadratic_zne_mean_absolute_deviation"] == pytest.approx(0.1)


def test_write_outputs_uses_result_tag_and_hashes_rows(tmp_path: Path) -> None:
    module = _load_module()
    scale_rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "basis_setting": "XXII",
            "zne_noise_scale": 1,
            "mean_expectation": 0.8,
        }
    ]
    channel_rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "basis_setting": "XXII",
            "linear_zne_expectation": 0.9,
        }
    ]
    summary = {
        "schema": "scpn_phase3_entanglement_zne_analysis_v1",
        "counts_artifact": "counts.json",
        "reference_csv": "refs.csv",
        "backend": "fake_heron",
        "job_ids": ["job-main", "job-readout"],
        "n_scale_rows": 1,
        "n_channels": 1,
        "readout_mitigation": {
            "method": "full_correlated_readout_inverse",
            "n_calibration_circuits": 16,
            "claim_boundary": "test boundary",
        },
        "claim_boundary": "test claim boundary",
    }

    json_path, scale_path, channel_path, md_path = module.write_outputs(
        scale_rows,
        channel_rows,
        summary,
        output_dir=tmp_path,
        docs_dir=tmp_path,
        result_tag="2026-05-20_ibm_fez_zne",
    )

    assert json_path.name == "entanglement_zne_summary_2026-05-20_ibm_fez_zne.json"
    assert scale_path.name == "entanglement_zne_scale_rows_2026-05-20_ibm_fez_zne.csv"
    assert channel_path.name == "entanglement_zne_channel_summary_2026-05-20_ibm_fez_zne.csv"
    assert md_path.name == "phase3_entanglement_zne_manifest_2026-05-20_ibm_fez_zne.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "scale_rows_sha256" in payload
    assert "channel_summary_sha256" in payload
