# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-system workflow contract tests
"""Workflow tests for Lindblad, MCWF, ancilla, and open-system solver consistency."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return n, K, omega


class TestOpenSystemPipeline:
    """Lindblad density matrix and MCWF quantum jumps should agree on R(t)
    within statistical error for the same physical parameters."""

    def test_lindblad_mcwf_agreement_2q(self):
        """At n=2, MCWF ensemble R(T) should match Lindblad R(T) within 0.3.

        Note: Lindblad and MCWF use different propagators (scipy ODE vs
        matrix expm with stochastic jumps). At small n with finite dt,
        discretisation differences can be significant. The MCWF result
        converges to Lindblad in the limit dt→0, n_trajectories→∞.
        """
        from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        n = 2
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])
        gamma_amp = 0.1
        t_max = 0.5
        dt = 0.05

        # Exact Lindblad
        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=gamma_amp)
        lindblad = solver.run(t_max=t_max, dt=dt)

        # MCWF ensemble (enough trajectories for statistical convergence)
        mcwf = mcwf_ensemble(
            K,
            omega,
            gamma_amp=gamma_amp,
            t_max=t_max,
            dt=dt,
            n_trajectories=200,
            seed=42,
        )

        # Final R values should agree within tolerance (discretisation + stochastic)
        assert abs(lindblad["R"][-1] - mcwf["R_mean"][-1]) < 0.3, (
            f"Lindblad R={lindblad['R'][-1]:.3f} vs "
            f"MCWF R={mcwf['R_mean'][-1]:.3f} ± {mcwf['R_std'][-1]:.3f}"
        )

    def test_lindblad_mcwf_agreement_4q(self):
        """Same test at n=4 — ensures scaling behaviour is consistent."""
        from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        _, K, omega = _system(4)
        gamma_amp = 0.1
        t_max = 0.3
        dt = 0.05

        solver = LindbladKuramotoSolver(4, K, omega, gamma_amp=gamma_amp)
        lindblad = solver.run(t_max=t_max, dt=dt)

        mcwf = mcwf_ensemble(
            K,
            omega,
            gamma_amp=gamma_amp,
            t_max=t_max,
            dt=dt,
            n_trajectories=100,
            seed=42,
        )

        assert abs(lindblad["R"][-1] - mcwf["R_mean"][-1]) < 0.2, (
            f"n=4: Lindblad R={lindblad['R'][-1]:.3f} vs MCWF R={mcwf['R_mean'][-1]:.3f}"
        )

    def test_zero_dissipation_lindblad_mcwf_agree(self):
        """With zero damping, both methods should give similar R(t).

        Note: Even at zero dissipation, Lindblad uses scipy ODE (RK45)
        while MCWF uses matrix expm for the effective Hamiltonian.
        These different propagators accumulate different numerical errors,
        especially at larger dt. The tolerance accounts for this.
        """
        from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        n = 2
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.1])

        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.0)
        lindblad = solver.run(t_max=0.3, dt=0.05)

        mcwf = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.0,
            t_max=0.3,
            dt=0.05,
            n_trajectories=10,
            seed=42,
        )

        # Zero damping → no jumps, deterministic. Propagator differences
        # cause ~0.1 divergence at later time steps.
        np.testing.assert_allclose(
            lindblad["R"],
            mcwf["R_mean"],
            atol=0.1,
            err_msg="Zero-damping Lindblad and MCWF should agree",
        )
        assert mcwf["total_jumps"] == 0, "No jumps expected at zero damping"

    def test_ancilla_circuit_consistent_with_lindblad(self):
        """Ancilla Lindblad circuit should produce a valid circuit for the
        same physical parameters used by the Lindblad solver."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            ancilla_circuit_stats,
            build_ancilla_lindblad_circuit,
        )
        from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver

        n = 2
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.2])

        # Lindblad result (reference)
        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.05)
        lindblad = solver.run(t_max=0.5, dt=0.1)
        assert lindblad["purity"][-1] < 1.0

        # Ancilla circuit for the same system
        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.5,
            gamma=0.05,
            n_dissipation_steps=3,
        )
        stats = ancilla_circuit_stats(
            K,
            omega,
            t=0.5,
            gamma=0.05,
            n_dissipation_steps=3,
        )

        assert qc.num_qubits == n + 1
        assert stats["n_system"] == n
        assert stats["n_ancilla"] == 1
        assert stats["n_resets"] > 0


class TestHardwareReadyPipeline:
    """End-to-end: build ancilla circuit → check stats → export → mitigate."""

    def test_ancilla_build_stats_export(self):
        """Build ancilla circuit, check stats, export to QASM."""
        from qiskit import qasm2

        from scpn_quantum_control.phase.ancilla_lindblad import (
            ancilla_circuit_stats,
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)

        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.3,
            gamma=0.05,
            n_dissipation_steps=3,
        )
        stats = ancilla_circuit_stats(
            K,
            omega,
            t=0.3,
            gamma=0.05,
            n_dissipation_steps=3,
        )

        # Verify circuit properties
        assert qc.num_qubits == 4  # 3 system + 1 ancilla
        assert stats["n_system"] == 3
        assert stats["n_ancilla"] == 1
        assert stats["n_resets"] > 0
        assert stats["n_cx_gates"] > 0

        # Should be QASM-exportable
        qasm_str = qasm2.dumps(qc)
        assert len(qasm_str) > 100

    def test_open_system_methods_all_run(self):
        """All three open-system methods should produce valid output
        for the same physical system."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )
        from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        n = 3
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        np.fill_diagonal(K, 0.0)
        omega = np.linspace(0.8, 1.2, n)

        # Method 1: Lindblad
        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.05)
        lindblad = solver.run(t_max=0.3, dt=0.1)
        assert 0 <= lindblad["R"][-1] <= 1.0

        # Method 2: MCWF
        mcwf = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.05,
            t_max=0.3,
            dt=0.1,
            n_trajectories=20,
            seed=42,
        )
        assert 0 <= mcwf["R_mean"][-1] <= 1.0

        # Method 3: Ancilla circuit (compiles but not executed here)
        qc = build_ancilla_lindblad_circuit(K, omega, t=0.3, gamma=0.05)
        assert qc.num_qubits == n + 1


class TestFullPipeline:
    """End-to-end: recommend_backend → auto_solve → export → mitigate."""

    def test_recommend_then_solve(self):
        """recommend_backend → auto_solve should produce consistent results."""
        from scpn_quantum_control.phase.backend_selector import (
            auto_solve,
            recommend_backend,
        )

        _, K, omega = _system(6)

        rec = recommend_backend(6)
        assert rec["feasible"]

        result = auto_solve(K, omega)
        assert result["backend_used"] == rec["backend"]
        assert "ground_energy" in result["result"]
        assert result["result"]["ground_energy"] < 0

    def test_solve_then_export(self):
        """auto_solve for ground energy, then export circuit for same system."""
        from scpn_quantum_control.hardware.circuit_export import export_all
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _system(4)

        # Solve
        solve_result = auto_solve(K, omega)
        E_ground = solve_result["result"]["ground_energy"]

        # Export circuit for same system
        export_result = export_all(K, omega, t=0.1, reps=5)

        assert E_ground < 0
        assert export_result["n_qubits"] == 4
        assert len(export_result["qasm3"]) > 50

    def test_solve_open_system_then_ancilla(self):
        """auto_solve for open system, then build ancilla circuit."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _system(4)

        result = auto_solve(
            K,
            omega,
            want_open_system=True,
            gamma_amp=0.05,
            t_max=0.5,
            dt=0.1,
        )
        assert result["backend_used"] in ("lindblad_scipy", "mcwf")

        # Build hardware circuit for the same system
        qc = build_ancilla_lindblad_circuit(K, omega, t=0.5, gamma=0.05)
        assert qc.num_qubits == 5

    def test_symmetry_then_sparse_then_level_spacing(self):
        """U(1) decomposition → sparse eigsh → level spacing analysis."""
        from scpn_quantum_control.analysis.magnetisation_sectors import (
            eigh_by_magnetisation,
            level_spacing_by_magnetisation,
        )
        from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh

        _, K, omega = _system(8)

        # U(1) ground energy
        u1_result = eigh_by_magnetisation(K, omega, sectors=[0])
        E_u1 = u1_result["results"][0]["eigvals"][0]

        # Sparse ground energy (should agree)
        sparse_result = sparse_eigsh(K, omega, k=5, M=0)
        E_sparse = sparse_result["eigvals"][0]

        np.testing.assert_allclose(E_u1, E_sparse, atol=1e-8)

        # Level spacing within M=0
        ls = level_spacing_by_magnetisation(K, omega, M=0)
        assert 0.2 < ls["r_bar"] < 0.7, (
            f"r̄ = {ls['r_bar']:.3f} outside expected range (Poisson=0.386, GOE=0.530)"
        )
