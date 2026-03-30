# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for New Modules Batch2
"""Tests for batch 2 modules: backend_selector, circuit_export, xy_compiler,
param_shift, ancilla_lindblad, tensor_jump."""

from __future__ import annotations

import numpy as np


def _small_system():
    n = 4
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


# =====================================================================
# Backend Selector
# =====================================================================
class TestBackendSelector:
    def test_small_system_selects_ed(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(4)
        assert rec["backend"] == "exact_diag"
        assert rec["feasible"]

    def test_medium_system_selects_sector(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(16, ram_gb=32.0)
        assert rec["backend"] in ("u1_sector_ed", "sector_ed", "statevector")

    def test_large_system_selects_mps(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(32, has_quimb=True)
        assert rec["backend"] == "mps_dmrg"

    def test_open_system_selects_lindblad(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(4, want_open_system=True)
        assert rec["backend"] == "lindblad_scipy"

    def test_huge_system_selects_hardware(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(100, ram_gb=32.0, has_quimb=False)
        assert rec["backend"] == "hardware"

    def test_auto_solve_runs(self):
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _small_system()
        result = auto_solve(K, omega)
        assert "backend_used" in result
        assert "result" in result
        assert result["result"]["ground_energy"] < 0


# =====================================================================
# Circuit Export
# =====================================================================
class TestCircuitExport:
    def test_qasm3_returns_string(self):
        from scpn_quantum_control.hardware.circuit_export import to_qasm3

        _, K, omega = _small_system()
        qasm = to_qasm3(K, omega, t=0.1, reps=2)
        assert isinstance(qasm, str)
        assert "OPENQASM" in qasm or "qreg" in qasm or "measure" in qasm

    def test_quil_returns_string(self):
        from scpn_quantum_control.hardware.circuit_export import to_quil

        _, K, omega = _small_system()
        quil = to_quil(K, omega, t=0.1, reps=2)
        assert isinstance(quil, str)
        assert "DECLARE" in quil
        assert "MEASURE" in quil

    def test_export_all_keys(self):
        from scpn_quantum_control.hardware.circuit_export import export_all

        _, K, omega = _small_system()
        result = export_all(K, omega, t=0.1, reps=2)
        assert "qiskit" in result
        assert "qasm3" in result
        assert "quil" in result
        assert result["n_qubits"] == 4
        assert result["depth"] > 0

    def test_build_trotter_circuit_measurable(self):
        from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit

        _, K, omega = _small_system()
        qc = build_trotter_circuit(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == 4
        assert any(i.operation.name == "measure" for i in qc.data)


# =====================================================================
# XY Compiler
# =====================================================================
class TestXYCompiler:
    def test_compile_returns_circuit(self):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _small_system()
        qc = compile_xy_trotter(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == 4

    def test_order2_deeper_than_order1(self):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _small_system()
        qc1 = compile_xy_trotter(K, omega, t=0.1, reps=3, order=1)
        qc2 = compile_xy_trotter(K, omega, t=0.1, reps=3, order=2)
        assert qc2.depth() >= qc1.depth()

    def test_depth_comparison(self):
        from scpn_quantum_control.phase.xy_compiler import depth_comparison

        _, K, omega = _small_system()
        result = depth_comparison(K, omega, t=0.1, reps=3)
        assert "generic_depth" in result
        assert "optimised_depth" in result
        assert result["optimised_depth"] > 0

    def test_xy_gate_unitary(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        from scpn_quantum_control.phase.xy_compiler import xy_gate

        qc = QuantumCircuit(2)
        xy_gate(qc, 0, 1, 0.5)
        op = Operator(qc)
        assert np.allclose(op.data @ op.data.conj().T, np.eye(4), atol=1e-10)


# =====================================================================
# Parameter Shift
# =====================================================================
class TestParamShift:
    def test_gradient_of_quadratic(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def quadratic(x):
            return float(x[0] ** 2 + x[1] ** 2)

        grad = parameter_shift_gradient(quadratic, np.array([1.0, 2.0]), shift=0.01)
        np.testing.assert_allclose(grad, [2.0, 4.0], atol=0.1)

    def test_vqe_converges(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        def cost(params):
            return float((params[0] - 1.0) ** 2 + (params[1] + 0.5) ** 2)

        result = vqe_with_param_shift(
            cost, n_params=2, learning_rate=0.05, n_iterations=50, seed=42
        )
        assert result["energy"] < 0.5

    def test_vqe_output_keys(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        result = vqe_with_param_shift(
            lambda x: float(sum(x**2)), n_params=3, n_iterations=5, seed=42
        )
        assert set(result.keys()) == {"optimal_params", "energy", "energy_history", "grad_norms"}

    def test_gradient_zero_at_minimum(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def cost(x):
            return float(x[0] ** 2)

        grad = parameter_shift_gradient(cost, np.array([0.0]), shift=0.01)
        assert abs(grad[0]) < 0.01


# =====================================================================
# Ancilla Lindblad
# =====================================================================
class TestAncillaLindblad:
    def test_circuit_has_ancilla(self):
        from scpn_quantum_control.phase.ancilla_lindblad import build_ancilla_lindblad_circuit

        _, K, omega = _small_system()
        qc = build_ancilla_lindblad_circuit(K, omega, t=0.1, trotter_reps=2, n_dissipation_steps=2)
        assert qc.num_qubits == 5  # 4 system + 1 ancilla

    def test_circuit_has_resets(self):
        from scpn_quantum_control.phase.ancilla_lindblad import build_ancilla_lindblad_circuit

        _, K, omega = _small_system()
        qc = build_ancilla_lindblad_circuit(K, omega, t=0.1, n_dissipation_steps=3)
        reset_count = sum(1 for i in qc.data if i.operation.name == "reset")
        assert reset_count == 4 * 3  # n_qubits * n_dissipation_steps

    def test_circuit_stats(self):
        from scpn_quantum_control.phase.ancilla_lindblad import ancilla_circuit_stats

        _, K, omega = _small_system()
        stats = ancilla_circuit_stats(K, omega)
        assert stats["n_qubits"] == 5
        assert stats["n_system"] == 4
        assert stats["n_ancilla"] == 1
        assert stats["n_resets"] > 0

    def test_circuit_has_measurements(self):
        from scpn_quantum_control.phase.ancilla_lindblad import build_ancilla_lindblad_circuit

        _, K, omega = _small_system()
        qc = build_ancilla_lindblad_circuit(K, omega)
        assert any(i.operation.name == "measure" for i in qc.data)


# =====================================================================
# Tensor Jump (MCWF)
# =====================================================================
class TestTensorJump:
    def test_single_trajectory_runs(self):
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])
        result = mcwf_trajectory(K, omega, gamma_amp=0.1, t_max=0.5, dt=0.1, seed=42)
        assert "R" in result
        assert len(result["R"]) == 6
        assert result["n_jumps"] >= 0

    def test_r_bounded(self):
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_trajectory(K, omega, gamma_amp=0.05, t_max=1.0, dt=0.1, seed=42)
        assert all(0 <= r <= 1.01 for r in result["R"])

    def test_ensemble_averages(self):
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_ensemble(
            K, omega, gamma_amp=0.1, t_max=0.3, dt=0.1, n_trajectories=10, seed=42
        )
        assert result["R_mean"].shape == result["R_std"].shape
        assert result["n_trajectories"] == 10
        assert result["R_trajectories"].shape[0] == 10

    def test_no_damping_preserves_norm(self):
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_trajectory(K, omega, gamma_amp=0.0, t_max=0.5, dt=0.1, seed=42)
        norm = float(np.linalg.norm(result["psi_final"]))
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)
        assert result["n_jumps"] == 0

    def test_ensemble_output_keys(self):
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_ensemble(K, omega, t_max=0.1, dt=0.05, n_trajectories=5, seed=42)
        expected = {"times", "R_mean", "R_std", "R_trajectories", "total_jumps", "n_trajectories"}
        assert set(result.keys()) == expected
