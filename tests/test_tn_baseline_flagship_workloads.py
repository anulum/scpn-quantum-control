# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the flagship TN baseline script
"""Tests for scripts/tn_baseline_flagship_workloads.py.

The workload registry is pinned to the March campaign parameters, the
exact statevector reference is cross-checked against independent Pauli
arithmetic, the bounded-χ Aer MPS run must match the exact reference at a
saturating bond dimension (and visibly degrade at χ = 1), and the report,
convergence rule, and CLI (happy path, single workload, bad χ) are all
exercised. Pure classical computation throughout.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scripts import tn_baseline_flagship_workloads as script

SMALL = script.FlagshipWorkload(
    name="small3",
    n_qubits=3,
    evolution_time=0.05,
    trotter_reps=1,
    parent_artifact="synthetic",
)


class TestRegistry:
    def test_flagship_workloads_pin_march_parameters(self) -> None:
        upde = script.WORKLOADS["upde16"]
        assert (upde.n_qubits, upde.evolution_time, upde.trotter_reps) == (16, 0.05, 1)
        kuramoto = script.WORKLOADS["kuramoto8"]
        assert (kuramoto.n_qubits, kuramoto.evolution_time, kuramoto.trotter_reps) == (8, 0.1, 2)
        for workload in script.WORKLOADS.values():
            assert workload.parent_artifact.startswith("results/ibm_hardware_2026-03-28/")

    def test_body_width_matches_workload(self) -> None:
        assert SMALL.body().num_qubits == 3


class TestOrderParameter:
    def test_matches_hand_arithmetic(self) -> None:
        x, y = [0.6, 0.2], [0.0, 0.3]
        assert script.order_parameter(x, y) == pytest.approx(math.hypot(0.8, 0.3) / 2)

    def test_rejects_mismatched_or_empty_input(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            script.order_parameter([0.1], [0.1, 0.2])
        with pytest.raises(ValueError, match="equal-length"):
            script.order_parameter([], [])


class TestExactReference:
    def test_matches_independent_statevector_arithmetic(self) -> None:
        reference = script.exact_reference(SMALL)
        state = Statevector(SMALL.body().decompose())
        for qubit in range(3):
            expected = float(np.real(state.expectation_value(script.single_pauli(3, qubit, "X"))))
            assert reference["x_expectations"][qubit] == pytest.approx(expected)
        assert reference["order_parameter_r"] == pytest.approx(
            script.order_parameter(reference["x_expectations"], reference["y_expectations"])
        )

    def test_single_pauli_targets_little_endian_qubit(self) -> None:
        assert script.single_pauli(3, 0, "X").paulis.to_labels() == ["IIX"]
        assert script.single_pauli(3, 2, "Y").paulis.to_labels() == ["YII"]


class TestMpsAgainstExact:
    def test_saturating_bond_dimension_reproduces_exact(self) -> None:
        reference = script.exact_reference(SMALL)
        x_values, y_values, wall_ms = script.mps_expectations(SMALL, 8)
        assert wall_ms > 0.0
        assert np.allclose(x_values, reference["x_expectations"], atol=1e-9)
        assert np.allclose(y_values, reference["y_expectations"], atol=1e-9)

    def test_chi_one_visibly_degrades_kuramoto8(self) -> None:
        workload = script.WORKLOADS["kuramoto8"]
        reference = script.exact_reference(workload)
        rows = script.chi_sweep(workload, reference, [1, 8])
        assert rows[0]["r_abs_error"] > 0.01
        assert rows[1]["r_abs_error"] < 1e-4

    def test_rejects_non_positive_bond_dimension(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            script.mps_expectations(SMALL, 0)


class TestConvergenceAndReport:
    def rows(self) -> list[dict[str, Any]]:
        return [
            {"bond_dimension": 1, "r_abs_error": 0.2},
            {"bond_dimension": 4, "r_abs_error": 5e-4},
            {"bond_dimension": 8, "r_abs_error": 1e-6},
        ]

    def test_convergence_picks_smallest_within_tolerance(self) -> None:
        assert script.convergence_bond_dimension(self.rows(), 1e-3) == 4

    def test_convergence_none_when_nothing_qualifies(self) -> None:
        assert script.convergence_bond_dimension(self.rows(), 1e-9) is None

    def test_report_shape(self) -> None:
        reference = script.exact_reference(SMALL)
        rows = script.chi_sweep(SMALL, reference, [2, 8])
        report = script.build_report(SMALL, reference, rows, 1e-3)
        assert report["workload"] == "small3"
        assert report["parent_artifact"] == "synthetic"
        assert len(report["chi_sweep"]) == 2
        assert report["converged_bond_dimension"] in (2, 8)


class TestMain:
    def test_single_workload_writes_hash_bound_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = script.main(
            [
                "--workload",
                "kuramoto8",
                "--bond-dimensions",
                "2",
                "8",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert code == 0
        output = next(iter(tmp_path.glob("tn_baseline_flagship_*.json")))
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["schema"] == script.SCHEMA
        assert [w["workload"] for w in payload["workloads"]] == ["kuramoto8"]
        assert "never claimed quantum advantage" in payload["claim_boundary"]
        out = capsys.readouterr().out
        assert "converged at chi" in out
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        assert digest in out

    def test_rejects_non_positive_chi(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert script.main(["--bond-dimensions", "0"]) == 2
        assert "positive" in capsys.readouterr().err

    def test_unknown_workload_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            script.main(["--workload", "unknown"])
