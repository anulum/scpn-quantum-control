# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase3 entanglement tomography analysis tests
# scpn-quantum-control -- Phase 3 entanglement/tomography analysis tests
"""Tests for Phase 3 entanglement/tomography raw-count analysis."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyse_phase3_entanglement_tomography.py"
    )
    spec = importlib.util.spec_from_file_location("analyse_phase3_entanglement_tomography", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entanglement/tomography analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pauli_expectation_from_counts_uses_non_identity_parity() -> None:
    module = _load_module()

    assert module.pauli_expectation_from_counts({"0000": 10}, "ZZII") == 1.0
    assert module.pauli_expectation_from_counts({"1000": 10}, "ZZII") == -1.0
    assert module.pauli_expectation_from_counts({"0000": 5, "1000": 5}, "ZZII") == 0.0


def test_single_qubit_readout_matrices_are_estimated_from_calibration_marginals() -> None:
    module = _load_module()

    matrices = module.estimate_single_qubit_readout_matrices(
        [
            {
                "meta": {"block": "readout", "initial": "00"},
                "counts": {"00": 81, "01": 9, "10": 9, "11": 1},
            },
            {
                "meta": {"block": "readout", "initial": "11"},
                "counts": {"11": 64, "10": 16, "01": 16, "00": 4},
            },
        ],
        width=2,
    )

    assert matrices[0]["prepared_0_observed_0"] == pytest.approx(0.90)
    assert matrices[0]["prepared_0_observed_1"] == pytest.approx(0.10)
    assert matrices[0]["prepared_1_observed_0"] == pytest.approx(0.20)
    assert matrices[0]["prepared_1_observed_1"] == pytest.approx(0.80)
    assert matrices[1] == matrices[0]


def test_readout_calibration_reverses_prepared_metadata_into_count_key_order() -> None:
    module = _load_module()

    matrices = module.estimate_single_qubit_readout_matrices(
        [
            {
                "meta": {"block": "readout", "initial": "01"},
                "counts": {"10": 100},
            },
            {
                "meta": {"block": "readout", "initial": "10"},
                "counts": {"01": 100},
            },
        ],
        width=2,
    )

    assert matrices[0]["prepared_1_observed_1"] == 1.0
    assert matrices[1]["prepared_0_observed_0"] == 1.0


def test_readout_mitigated_pauli_expectation_inverts_independent_assignment_errors() -> None:
    module = _load_module()
    matrices = [
        {
            "prepared_0_observed_0": 0.9,
            "prepared_0_observed_1": 0.1,
            "prepared_1_observed_0": 0.2,
            "prepared_1_observed_1": 0.8,
        },
        {
            "prepared_0_observed_0": 0.9,
            "prepared_0_observed_1": 0.1,
            "prepared_1_observed_0": 0.2,
            "prepared_1_observed_1": 0.8,
        },
    ]

    mitigated = module.readout_mitigated_pauli_expectation({"00": 90, "10": 10}, "ZI", matrices)

    assert mitigated == pytest.approx(1.0)


def test_correlated_readout_mitigation_uses_full_calibration_matrix() -> None:
    module = _load_module()
    readout = [
        {
            "meta": {"block": "readout", "initial": prepared[::-1]},
            "counts": {prepared: 100},
        }
        for prepared in ["00", "01", "10", "11"]
    ]

    model = module.build_readout_mitigation_model(readout, width=2)
    mitigated = module.mitigated_pauli_expectation({"11": 100}, "ZZ", model)

    assert model["method"] == "full_correlated_readout_inverse"
    assert mitigated == pytest.approx(1.0)


def test_analyse_counts_artifact_groups_repetitions_against_reference(tmp_path: Path) -> None:
    module = _load_module()
    counts_path = tmp_path / "counts.json"
    refs_path = tmp_path / "refs.csv"
    counts_path.write_text(
        json.dumps(
            {
                "schema": "scpn_phase3_entanglement_tomography_live_v1",
                "backend": "fake_heron",
                "status": "completed",
                "job_ids": ["job-main", "job-readout"],
                "circuits": [
                    {
                        "job_id": "job-main",
                        "meta": {
                            "block": "main",
                            "family": "dla_parity",
                            "label": "dla_even_signal",
                            "initial": "0011",
                            "depth": 10,
                            "lambda_fim": None,
                            "basis_setting": "ZZII",
                            "rep": 0,
                        },
                        "counts": {"0000": 8, "1000": 2},
                    },
                    {
                        "job_id": "job-main",
                        "meta": {
                            "block": "main",
                            "family": "dla_parity",
                            "label": "dla_even_signal",
                            "initial": "0011",
                            "depth": 10,
                            "lambda_fim": None,
                            "basis_setting": "ZZII",
                            "rep": 1,
                        },
                        "counts": {"0000": 10},
                    },
                    {
                        "job_id": "job-readout",
                        "meta": {"block": "readout", "initial": "0000"},
                        "counts": {"0000": 10},
                    },
                    {
                        "job_id": "job-readout",
                        "meta": {"block": "readout", "initial": "1111"},
                        "counts": {"1111": 10},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with refs_path.open("w", encoding="utf-8", newline="") as handle:
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
                "label": "dla_even_signal",
                "initial": "0011",
                "depth": "10",
                "lambda_fim": "",
                "observable": "ZZ_q0q1",
                "pauli_label": "ZZII",
                "basis_setting": "ZZII",
                "exact_expectation": "0.9",
                "half_chain_purity": "0.55",
                "parity_survival_ideal": "1.0",
                "shots_per_setting": "2048",
                "repetitions": "3",
            }
        )

    rows, summary = module.analyse_counts_artifact(counts_path, refs_path)

    assert summary["backend"] == "fake_heron"
    assert summary["job_ids"] == ["job-main", "job-readout"]
    assert summary["analysis_job_ids"] == ["job-main"]
    assert summary["n_observable_rows"] == 1
    assert summary["readout_mitigation"]["method"] == "tensor_product_single_qubit_inverse"
    assert rows[0]["mean_expectation"] == 0.8
    assert rows[0]["readout_mitigated_mean_expectation"] == pytest.approx(0.8)
    assert rows[0]["exact_expectation"] == 0.9
    assert rows[0]["absolute_deviation"] == pytest.approx(0.1)


def test_write_outputs_accepts_backend_specific_result_tag(tmp_path: Path) -> None:
    module = _load_module()

    json_path, csv_path, md_path = module.write_outputs(
        [
            {
                "family": "dla_parity",
                "label": "replication",
                "absolute_deviation": 0.1,
            }
        ],
        {
            "schema": "scpn_phase3_entanglement_tomography_analysis_v1",
            "counts_artifact": "counts.json",
            "reference_csv": "refs.csv",
            "backend": "ibm_fez",
            "job_ids": ["job-main"],
            "n_observable_rows": 1,
            "mean_absolute_deviation": 0.1,
            "max_absolute_deviation": 0.1,
            "readout_mitigated_mean_absolute_deviation": 0.09,
            "readout_mitigated_max_absolute_deviation": 0.09,
            "readout_mitigation": {
                "method": "tensor_product_single_qubit_inverse",
                "n_calibration_circuits": 4,
                "claim_boundary": "test boundary",
            },
            "claim_boundary": "test claim boundary",
        },
        output_dir=tmp_path,
        docs_dir=tmp_path,
        result_tag="2026-05-20_ibm_fez",
    )

    assert json_path.name == "entanglement_tomography_summary_2026-05-20_ibm_fez.json"
    assert csv_path.name == "entanglement_tomography_rows_2026-05-20_ibm_fez.csv"
    assert md_path.name == "phase3_entanglement_tomography_manifest_2026-05-20_ibm_fez.md"
