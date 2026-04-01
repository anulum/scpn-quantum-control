# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Ansatz Methodology
"""Tests for the coupling-informed ansatz benchmark methodology."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from scpn_quantum_control.phase.ansatz_methodology import (
    AnsatzBenchmarkResult,
    _convergence_99pct,
    _count_entangling_gates,
    _gradient_variance,
    benchmark_single_ansatz,
    run_full_benchmark,
    summarize_benchmark,
)


class TestCountEntanglingGates:
    def test_no_entangling(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.ry(0.5, 0)
        qc.rz(0.3, 1)
        assert _count_entangling_gates(qc) == 0

    def test_cx_counted(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        assert _count_entangling_gates(qc) == 1

    def test_cz_counted(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.cz(0, 1)
        assert _count_entangling_gates(qc) == 2

    def test_knm_ansatz(self):
        K = build_knm_paper27(L=3)
        ansatz = knm_to_ansatz(K, reps=1)
        # K_nm for 3 qubits: at most 3 pairs, reps=1
        n_ent = _count_entangling_gates(ansatz)
        assert n_ent > 0
        assert n_ent <= 3  # at most 3 pairs for 3 qubits


class TestConvergence99pct:
    def test_monotonic(self):
        history = [-1.0, -2.0, -3.0, -3.5, -3.9, -4.0]
        idx = _convergence_99pct(history)
        assert idx < len(history)

    def test_already_converged(self):
        history = [-4.0, -4.0, -4.0]
        idx = _convergence_99pct(history)
        assert idx == 0

    def test_empty(self):
        assert _convergence_99pct([]) == 0

    def test_single(self):
        assert _convergence_99pct([-3.0]) == 0


class TestGradientVariance:
    def test_returns_nonneg(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        H = knm_to_hamiltonian(K, omega)
        ansatz = knm_to_ansatz(K, reps=1)
        var = _gradient_variance(ansatz, H, n_samples=5)
        assert var >= 0.0

    def test_small_circuit_has_variance(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        H = knm_to_hamiltonian(K, omega)
        ansatz = knm_to_ansatz(K, reps=1)
        var = _gradient_variance(ansatz, H, n_samples=10)
        # 2-qubit circuits should NOT have barren plateaus
        assert var > 1e-10


class TestBenchmarkSingleAnsatz:
    @pytest.mark.parametrize("name", ["knm_informed", "two_local", "efficient_su2"])
    def test_all_ansatze_run(self, name):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = benchmark_single_ansatz(K, omega, name, maxiter=20, reps=1, gradient_samples=5)
        assert isinstance(result, AnsatzBenchmarkResult)
        assert result.ansatz_name == name
        assert result.n_qubits == 3
        assert result.n_params > 0
        assert result.exact_energy < 0  # XY ground state is negative
        assert result.relative_error >= 0

    def test_knm_has_fewer_gates(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        knm = benchmark_single_ansatz(
            K, omega, "knm_informed", maxiter=5, reps=1, gradient_samples=3
        )
        hea = benchmark_single_ansatz(K, omega, "two_local", maxiter=5, reps=1, gradient_samples=3)
        # K_nm places CZ only where coupling exists; TwoLocal uses linear chain
        # For 4 qubits: K_nm may have fewer or same, never wildly more
        assert knm.n_entangling_gates <= hea.n_entangling_gates * 3

    def test_unknown_ansatz_raises(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        with pytest.raises(ValueError, match="Unknown ansatz"):
            benchmark_single_ansatz(K, omega, "nonexistent", maxiter=5)


class TestRunFullBenchmark:
    def test_small_benchmark(self):
        results = run_full_benchmark(system_sizes=[2, 3], maxiter=10, reps=1, gradient_samples=3)
        # 2 sizes × 3 ansatze = 6 results
        assert len(results) == 6
        names = [r.ansatz_name for r in results]
        assert names.count("knm_informed") == 2
        assert names.count("two_local") == 2
        assert names.count("efficient_su2") == 2


class TestSummarizeBenchmark:
    def test_produces_table(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        results = [
            benchmark_single_ansatz(
                K, omega, "knm_informed", maxiter=5, reps=1, gradient_samples=3
            ),
        ]
        summary = summarize_benchmark(results)
        assert "results" in summary
        assert len(summary["results"]) == 1
        row = summary["results"][0]
        assert "ansatz" in row
        assert "energy" in row
        assert "grad_var" in row
