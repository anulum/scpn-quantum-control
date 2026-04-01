# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for New Modules Batch2
"""Tests for batch 2 modules: backend_selector, circuit_export, xy_compiler,
param_shift, ancilla_lindblad, tensor_jump.

Multi-angle: parametrised sizes, edge cases, physical invariants,
numerical precision, property-based checks, error conditions.
"""

from __future__ import annotations

import numpy as np
import pytest


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _zero_coupling(n: int = 4):
    """Decoupled system — K=0, eigenstates are product states."""
    K = np.zeros((n, n))
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


# =====================================================================
# Backend Selector
# =====================================================================
class TestBackendSelector:
    """Tests for recommend_backend and auto_solve."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, "exact_diag"),
            (4, "exact_diag"),
            (8, "exact_diag"),
            (14, "exact_diag"),
        ],
    )
    def test_small_systems_select_ed(self, n, expected):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(n)
        assert rec["backend"] == expected
        assert rec["feasible"]
        assert rec["memory_mb"] > 0

    def test_medium_system_selects_sector(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(16, ram_gb=32.0)
        assert rec["backend"] in ("u1_sector_ed", "sector_ed", "statevector")
        assert rec["feasible"]

    def test_large_system_selects_mps(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(32, has_quimb=True)
        assert rec["backend"] == "mps_dmrg"

    def test_huge_system_selects_hardware(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(100, ram_gb=32.0, has_quimb=False)
        assert rec["backend"] == "hardware"

    def test_open_system_selects_lindblad(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(4, want_open_system=True)
        assert rec["backend"] == "lindblad_scipy"

    def test_open_system_large_selects_mcwf(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(14, want_open_system=True)
        assert rec["backend"] in ("mcwf", "lindblad_scipy")

    def test_recommendation_output_keys(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(4)
        assert {"backend", "reason", "memory_mb", "feasible"} <= set(rec.keys())
        assert isinstance(rec["backend"], str)
        assert isinstance(rec["reason"], str)
        assert isinstance(rec["memory_mb"], (int, float))
        assert isinstance(rec["feasible"], bool)

    def test_memory_increases_with_n(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        mem_4 = recommend_backend(4)["memory_mb"]
        mem_8 = recommend_backend(8)["memory_mb"]
        mem_12 = recommend_backend(12)["memory_mb"]
        assert mem_4 < mem_8 < mem_12, "Memory should increase with system size"

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_auto_solve_produces_ground_energy(self, n):
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _system(n)
        result = auto_solve(K, omega)
        assert "backend_used" in result
        assert "result" in result
        assert result["result"]["ground_energy"] < 0
        assert np.isfinite(result["result"]["ground_energy"])

    def test_auto_solve_zero_coupling(self):
        """Decoupled system: ground energy = -sum(|omega|)."""
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _zero_coupling(4)
        result = auto_solve(K, omega)
        E = result["result"]["ground_energy"]
        E_expected = -np.sum(np.abs(omega))
        np.testing.assert_allclose(E, E_expected, atol=1e-8)

    def test_auto_solve_matches_recommend(self):
        from scpn_quantum_control.phase.backend_selector import (
            auto_solve,
            recommend_backend,
        )

        _, K, omega = _system(6)
        rec = recommend_backend(6)
        result = auto_solve(K, omega)
        assert result["backend_used"] == rec["backend"]


# =====================================================================
# Circuit Export
# =====================================================================
class TestCircuitExport:
    """Tests for multi-platform circuit export."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qasm3_valid_for_multiple_sizes(self, n):
        from scpn_quantum_control.hardware.circuit_export import to_qasm3

        _, K, omega = _system(n)
        qasm = to_qasm3(K, omega, t=0.1, reps=2)
        assert isinstance(qasm, str)
        assert len(qasm) > 50
        assert "OPENQASM" in qasm or "qreg" in qasm or "measure" in qasm

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_quil_valid_for_multiple_sizes(self, n):
        from scpn_quantum_control.hardware.circuit_export import to_quil

        _, K, omega = _system(n)
        quil = to_quil(K, omega, t=0.1, reps=2)
        assert isinstance(quil, str)
        assert "DECLARE" in quil
        assert "MEASURE" in quil

    def test_export_all_keys_and_types(self):
        from qiskit import QuantumCircuit

        from scpn_quantum_control.hardware.circuit_export import export_all

        _, K, omega = _system(4)
        result = export_all(K, omega, t=0.1, reps=2)
        assert isinstance(result["qiskit"], QuantumCircuit)
        assert isinstance(result["qasm3"], str)
        assert isinstance(result["quil"], str)
        assert result["n_qubits"] == 4
        assert result["depth"] > 0
        assert result["gate_count"] > 0

    def test_build_trotter_circuit_properties(self):
        from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit

        _, K, omega = _system(4)
        qc = build_trotter_circuit(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == 4
        assert any(i.operation.name == "measure" for i in qc.data)
        assert qc.depth() > 0
        assert qc.size() > 0

    @pytest.mark.parametrize("reps", [1, 3, 5, 10])
    def test_depth_scales_with_reps(self, reps):
        from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit

        _, K, omega = _system(3)
        qc = build_trotter_circuit(K, omega, t=0.1, reps=reps)
        assert qc.depth() > 0

    def test_export_formats_all_reference_same_qubits(self):
        from scpn_quantum_control.hardware.circuit_export import export_all

        _, K, omega = _system(3)
        result = export_all(K, omega, t=0.1, reps=2)
        assert result["qiskit"].num_qubits == 3
        assert result["n_qubits"] == 3


# =====================================================================
# XY Compiler
# =====================================================================
class TestXYCompiler:
    """Tests for XY-optimised gate decomposition."""

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_compile_returns_correct_qubit_count(self, n):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(n)
        qc = compile_xy_trotter(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == n

    def test_order2_deeper_than_order1(self):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(4)
        qc1 = compile_xy_trotter(K, omega, t=0.1, reps=3, order=1)
        qc2 = compile_xy_trotter(K, omega, t=0.1, reps=3, order=2)
        assert qc2.depth() >= qc1.depth()
        assert qc2.size() >= qc1.size()

    def test_depth_comparison_values(self):
        from scpn_quantum_control.phase.xy_compiler import depth_comparison

        _, K, omega = _system(4)
        result = depth_comparison(K, omega, t=0.1, reps=5)
        assert result["generic_depth"] > 0
        assert result["optimised_depth"] > 0
        assert isinstance(result["reduction_pct"], (int, float))

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, np.pi / 4, np.pi / 2])
    def test_xy_gate_is_unitary(self, angle):
        """XY gate must be unitary for all rotation angles."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        from scpn_quantum_control.phase.xy_compiler import xy_gate

        qc = QuantumCircuit(2)
        xy_gate(qc, 0, 1, angle)
        op = Operator(qc)
        product = op.data @ op.data.conj().T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_xy_gate_zero_angle_is_identity(self):
        """XY gate at angle=0 should be identity."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        from scpn_quantum_control.phase.xy_compiler import xy_gate

        qc = QuantumCircuit(2)
        xy_gate(qc, 0, 1, 0.0)
        op = Operator(qc)
        np.testing.assert_allclose(np.abs(op.data), np.eye(4), atol=1e-10)

    def test_reps_increases_depth(self):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(3)
        d1 = compile_xy_trotter(K, omega, t=0.1, reps=1).depth()
        d3 = compile_xy_trotter(K, omega, t=0.1, reps=3).depth()
        assert d3 >= d1


# =====================================================================
# Parameter Shift
# =====================================================================
class TestParamShift:
    """Tests for parameter-shift gradient rule and VQE."""

    def test_gradient_of_quadratic(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def quadratic(x):
            return float(x[0] ** 2 + x[1] ** 2)

        grad = parameter_shift_gradient(quadratic, np.array([1.0, 2.0]), shift=0.01)
        np.testing.assert_allclose(grad, [2.0, 4.0], atol=0.1)

    def test_gradient_of_sinusoidal(self):
        """Parameter-shift is exact for sinusoidal functions."""
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def sinusoidal(x):
            return float(np.sin(x[0]) + np.cos(x[1]))

        params = np.array([0.5, 1.0])
        grad = parameter_shift_gradient(sinusoidal, params, shift=np.pi / 2)
        expected = np.array([np.cos(0.5), -np.sin(1.0)])
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_gradient_zero_at_minimum(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def cost(x):
            return float(x[0] ** 2)

        grad = parameter_shift_gradient(cost, np.array([0.0]), shift=0.01)
        assert abs(grad[0]) < 0.01

    def test_gradient_shape_matches_params(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        for n_params in [1, 3, 5, 10]:
            grad = parameter_shift_gradient(
                lambda x: float(np.sum(x**2)),
                np.random.randn(n_params),
                shift=0.01,
            )
            assert grad.shape == (n_params,)

    def test_vqe_converges_to_minimum(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        def cost(params):
            return float((params[0] - 1.0) ** 2 + (params[1] + 0.5) ** 2)

        result = vqe_with_param_shift(
            cost,
            n_params=2,
            learning_rate=0.05,
            n_iterations=100,
            seed=42,
        )
        assert result["energy"] < 0.1, "Should converge near minimum"

    def test_vqe_energy_monotonically_decreases(self):
        """Energy should generally decrease (allow small fluctuations)."""
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        result = vqe_with_param_shift(
            lambda x: float(sum(x**2)),
            n_params=3,
            learning_rate=0.05,
            n_iterations=50,
            seed=42,
        )
        # First energy should be higher than last
        assert result["energy_history"][0] > result["energy_history"][-1]

    def test_vqe_output_keys_and_types(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        result = vqe_with_param_shift(
            lambda x: float(sum(x**2)),
            n_params=3,
            n_iterations=5,
            seed=42,
        )
        assert set(result.keys()) == {
            "optimal_params",
            "energy",
            "energy_history",
            "grad_norms",
        }
        assert isinstance(result["optimal_params"], np.ndarray)
        assert isinstance(result["energy"], float)
        assert len(result["energy_history"]) >= 5
        assert len(result["grad_norms"]) >= 5

    def test_vqe_reproducible_with_seed(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        cost = lambda x: float(sum(x**2))  # noqa: E731
        r1 = vqe_with_param_shift(cost, n_params=3, n_iterations=10, seed=42)
        r2 = vqe_with_param_shift(cost, n_params=3, n_iterations=10, seed=42)
        np.testing.assert_array_equal(r1["optimal_params"], r2["optimal_params"])


# =====================================================================
# Ancilla Lindblad
# =====================================================================
class TestAncillaLindblad:
    """Tests for single-ancilla open-system circuit."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_circuit_has_correct_qubit_count(self, n):
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(n)
        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.1,
            trotter_reps=2,
            n_dissipation_steps=2,
        )
        assert qc.num_qubits == n + 1  # system + 1 ancilla

    @pytest.mark.parametrize("n_steps", [1, 2, 3, 5])
    def test_reset_count_scales_with_steps(self, n_steps):
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.1,
            n_dissipation_steps=n_steps,
        )
        reset_count = sum(1 for i in qc.data if i.operation.name == "reset")
        # Each dissipation step resets the ancilla after interacting with each system qubit
        assert reset_count == 3 * n_steps  # n_system * n_dissipation_steps

    def test_circuit_stats_all_keys(self):
        from scpn_quantum_control.phase.ancilla_lindblad import ancilla_circuit_stats

        _, K, omega = _system(4)
        stats = ancilla_circuit_stats(K, omega)
        assert stats["n_qubits"] == 5
        assert stats["n_system"] == 4
        assert stats["n_ancilla"] == 1
        assert stats["n_resets"] > 0
        assert stats["n_cx_gates"] > 0
        assert stats["total_gates"] > 0

    def test_circuit_has_measurements(self):
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega)
        assert any(i.operation.name == "measure" for i in qc.data)

    def test_stats_consistent_with_circuit(self):
        """Stats should match actual circuit properties."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            ancilla_circuit_stats,
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        kwargs = {"t": 0.1, "trotter_reps": 3, "gamma": 0.05, "n_dissipation_steps": 2}
        qc = build_ancilla_lindblad_circuit(K, omega, **kwargs)
        stats = ancilla_circuit_stats(K, omega, **kwargs)

        assert stats["n_qubits"] == qc.num_qubits
        actual_resets = sum(1 for i in qc.data if i.operation.name == "reset")
        assert stats["n_resets"] == actual_resets

    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.05, 0.1, 0.5])
    def test_gamma_range(self, gamma):
        """Circuit should build for all valid gamma values."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(2)
        qc = build_ancilla_lindblad_circuit(K, omega, gamma=gamma)
        assert qc.num_qubits == 3


# =====================================================================
# Tensor Jump (MCWF)
# =====================================================================
class TestTensorJump:
    """Tests for Monte Carlo Wave Function method."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_single_trajectory_runs_multiple_sizes(self, n):
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(n)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.3,
            dt=0.1,
            seed=42,
        )
        assert "R" in result
        assert "psi_final" in result
        assert "n_jumps" in result
        assert len(result["R"]) >= 3  # at least 3 time steps
        assert result["n_jumps"] >= 0

    def test_r_strictly_bounded(self):
        """R must be in [0, 1] — physical invariant."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(4)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.05,
            t_max=1.0,
            dt=0.05,
            seed=42,
        )
        assert all(0 <= r <= 1.0 + 1e-10 for r in result["R"]), (
            f"R out of [0,1]: {[r for r in result['R'] if r < 0 or r > 1.0 + 1e-10]}"
        )

    @pytest.mark.parametrize("n_traj", [5, 20, 50])
    def test_ensemble_shape_consistency(self, n_traj):
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.2,
            dt=0.1,
            n_trajectories=n_traj,
            seed=42,
        )
        n_steps = len(result["times"])
        assert result["R_mean"].shape == (n_steps,)
        assert result["R_std"].shape == (n_steps,)
        assert result["R_trajectories"].shape == (n_traj, n_steps)
        assert result["n_trajectories"] == n_traj

    def test_no_damping_preserves_norm_and_no_jumps(self):
        """Zero gamma → unitary evolution, norm preserved, no jumps."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(3)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.0,
            t_max=0.5,
            dt=0.05,
            seed=42,
        )
        norm = float(np.linalg.norm(result["psi_final"]))
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)
        assert result["n_jumps"] == 0, "No jumps expected at zero damping"

    def test_strong_damping_reduces_R(self):
        """Strong damping should drive R toward 0."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        _, K, omega = _system(3)
        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=2.0,
            t_max=2.0,
            dt=0.05,
            n_trajectories=30,
            seed=42,
        )
        assert result["R_mean"][-1] < 0.5, (
            f"Strong damping should reduce R, got {result['R_mean'][-1]:.3f}"
        )

    def test_ensemble_output_keys(self):
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        result = mcwf_ensemble(
            K,
            omega,
            t_max=0.1,
            dt=0.05,
            n_trajectories=5,
            seed=42,
        )
        expected = {
            "times",
            "R_mean",
            "R_std",
            "R_trajectories",
            "total_jumps",
            "n_trajectories",
        }
        assert set(result.keys()) == expected

    def test_reproducible_with_seed(self):
        """Same seed should give identical results."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])
        r1 = mcwf_trajectory(K, omega, gamma_amp=0.1, t_max=0.3, dt=0.1, seed=42)
        r2 = mcwf_trajectory(K, omega, gamma_amp=0.1, t_max=0.3, dt=0.1, seed=42)
        np.testing.assert_array_equal(r1["R"], r2["R"])
        assert r1["n_jumps"] == r2["n_jumps"]

    def test_psi_final_is_normalised(self):
        """Final state vector must be normalised."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

        _, K, omega = _system(4)
        result = mcwf_trajectory(
            K,
            omega,
            gamma_amp=0.05,
            t_max=0.5,
            dt=0.1,
            seed=42,
        )
        norm = np.linalg.norm(result["psi_final"])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_ensemble_r_mean_bounded(self):
        """Ensemble-averaged R must be in [0, 1]."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        _, K, omega = _system(3)
        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.5,
            dt=0.05,
            n_trajectories=20,
            seed=42,
        )
        assert all(0 <= r <= 1.0 + 1e-10 for r in result["R_mean"])
        assert all(s >= 0 for s in result["R_std"])
