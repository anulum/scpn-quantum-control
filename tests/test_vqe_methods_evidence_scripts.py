# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- VQE methods evidence script tests
"""Contract tests for methods-paper VQE evidence helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script_module(name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module {name}")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vqe_methods = _load_script_module("benchmark_vqe_convergence_methods")
ibm_validation = _load_script_module("submit_methods_ansatz_ibm_validation")


def test_timing_summary_reports_uncertainty_metrics() -> None:
    stats = vqe_methods.timing_summary([1.0, 2.0, 3.0])

    assert stats["mean_ms"] == pytest.approx(2.0)
    assert stats["median_ms"] == pytest.approx(2.0)
    assert stats["stdev_ms"] > 0.0
    assert stats["stderr_ms"] > 0.0
    assert stats["repeats"] == 3.0


def test_vqe_error_summary_aggregates_seed_spread() -> None:
    rows = [
        {
            "ansatz": "knm_informed",
            "n_qubits": 4,
            "reps": 1,
            "relative_error_pct": 0.1,
            "absolute_error": 0.01,
            "elapsed_ms": 10.0,
        },
        {
            "ansatz": "knm_informed",
            "n_qubits": 4,
            "reps": 1,
            "relative_error_pct": 0.3,
            "absolute_error": 0.03,
            "elapsed_ms": 14.0,
        },
    ]

    [summary] = vqe_methods.summarise_final_errors(rows)

    assert summary["mean_relative_error_pct"] == pytest.approx(0.2)
    assert summary["stdev_relative_error_pct"] > 0.0
    assert summary["stderr_elapsed_ms"] > 0.0


def test_pauli_expectation_uses_qiskit_qubit_order() -> None:
    counts = {"00": 5, "01": 5}

    assert ibm_validation.expectation_from_counts(counts, "IZ") == pytest.approx(0.0)
    assert ibm_validation.expectation_from_counts(counts, "ZI") == pytest.approx(1.0)


def test_pauli_terms_group_into_global_xyz_bases() -> None:
    groups = ibm_validation.pauli_terms_by_basis(4)

    assert sorted(groups) == ["X", "Y", "Z"]
    assert len(groups["Z"]) == 4
    assert len(groups["X"]) == 6
    assert len(groups["Y"]) == 6


def test_energy_reducer_combines_grouped_basis_counts() -> None:
    groups = {
        "Z": [{"label": "IIIZ", "coefficient": -1.0}],
        "X": [{"label": "IIXX", "coefficient": -0.5}],
        "Y": [{"label": "IIYY", "coefficient": -0.25}],
    }
    counts = {
        "Z": {"00": 10},
        "X": {"00": 10},
        "Y": {"00": 10},
    }

    assert ibm_validation.energy_from_basis_counts(counts, groups) == pytest.approx(-1.75)


def test_full_readout_identity_mitigation_preserves_distribution() -> None:
    calibration_rows = [
        {"calibration_state": "00", "counts": {"00": 10}},
        {"calibration_state": "01", "counts": {"01": 10}},
        {"calibration_state": "10", "counts": {"10": 10}},
        {"calibration_state": "11", "counts": {"11": 10}},
    ]

    assignment = ibm_validation.full_readout_assignment_matrix(calibration_rows, n_qubits=2)
    mitigated = ibm_validation.mitigate_counts_with_assignment(
        {"00": 5, "11": 5},
        assignment_matrix=assignment,
        n_qubits=2,
    )

    assert mitigated.tolist() == pytest.approx([0.5, 0.0, 0.0, 0.5])
    assert ibm_validation.energy_from_basis_probabilities(
        {"Z": mitigated},
        {"Z": [{"label": "ZZ", "coefficient": 1.0}]},
    ) == pytest.approx(1.0)


def test_methods_readiness_uses_configured_budget_and_dynamic_claim_boundary() -> None:
    args = ibm_validation._parse_args(
        [
            "--backend",
            "fake_backend",
            "--n-qubits",
            "8",
            "--physical-qubits",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "--max-qpu-seconds",
            "300",
        ]
    )

    circuit = QuantumCircuit(8, 8)
    circuit.measure(range(8), range(8))
    entries = [
        ibm_validation.ValidationCircuit(
            role="ansatz_basis",
            ansatz="knm_informed",
            basis="Z",
            repetition=0,
            calibration_state=None,
            circuit=circuit,
        )
        for _ in range(265)
    ]
    payload = ibm_validation._readiness_payload(
        args=args,
        backend=object(),
        entries=entries,
        isa_circuits=[circuit] * len(entries),
        local_summaries=[],
        grouped_terms={"X": [], "Y": [], "Z": []},
    )

    assert payload["status"] == "ready_for_submission"
    assert payload["qpu_seconds_ceiling"] == 300.0
    assert "n=8 hardware validation" in payload["claim_boundary"]
