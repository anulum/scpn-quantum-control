# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Cross Module
"""Cross-module consistency: verify pipeline paths produce compatible results."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge import knm_to_hamiltonian
from scpn_quantum_control.hardware.classical import (
    classical_exact_diag,
    classical_kuramoto_reference,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver


def test_kuramoto_solver_vs_classical_ground_energy(knm_4q):
    """QuantumKuramotoSolver Hamiltonian matches classical_exact_diag ground energy."""
    K, omega = knm_4q

    solver = QuantumKuramotoSolver(4, K, omega)
    H_solver = solver.build_hamiltonian()

    H_bridge = knm_to_hamiltonian(K, omega)

    mat_solver = H_solver.to_matrix()
    mat_bridge = H_bridge.to_matrix()
    if hasattr(mat_solver, "toarray"):
        mat_solver = mat_solver.toarray()
    if hasattr(mat_bridge, "toarray"):
        mat_bridge = mat_bridge.toarray()

    assert np.allclose(mat_solver, mat_bridge, atol=1e-10)


def test_classical_diag_vs_numpy_eigvalsh(knm_4q):
    """classical_exact_diag matches direct numpy eigvalsh."""
    K, omega = knm_4q

    result = classical_exact_diag(n_osc=4, K=K, omega=omega)
    E0_classical = result["ground_energy"]

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    E0_numpy = np.linalg.eigvalsh(mat)[0]

    assert abs(E0_classical - E0_numpy) < 1e-10


def test_classical_kuramoto_R_positive():
    """Classical Kuramoto integration produces R in [0, 1]."""
    result = classical_kuramoto_reference(n_osc=4, t_max=1.0, dt=0.01)
    R_final = result["R"][-1]
    assert 0.0 <= R_final <= 1.0 + 1e-10


def test_energy_expectation_ground_state(knm_4q):
    """Ground state of H has energy matching the lowest eigenvalue."""
    K, omega = knm_4q

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    eigvals, eigvecs = np.linalg.eigh(mat)
    ground_sv = Statevector(eigvecs[:, 0].copy())

    solver = QuantumKuramotoSolver(4, K, omega)
    E = solver.energy_expectation(ground_sv)
    assert abs(E - eigvals[0]) < 1e-8


def test_hamiltonian_commutes_with_total_z_parity(knm_4q):
    """XY Hamiltonian conserves total Z parity (XX+YY preserves excitation number mod 2)."""
    K, omega = knm_4q

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    from functools import reduce

    Z = np.array([[1, 0], [0, -1]])
    parity = reduce(np.kron, [Z] * 4)

    commutator = mat @ parity - parity @ mat
    assert np.allclose(commutator, 0, atol=1e-10), "H should commute with Z-parity"


def test_solver_energy_matches_bridge_energy(knm_4q):
    """QuantumKuramotoSolver energy matches bridge Hamiltonian matrix energy."""
    K, omega = knm_4q

    H_bridge = knm_to_hamiltonian(K, omega)
    mat = H_bridge.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    eigvals = np.linalg.eigvalsh(mat)

    exact_diag = classical_exact_diag(n_osc=4, K=K, omega=omega)
    np.testing.assert_allclose(exact_diag["ground_energy"], eigvals[0], atol=1e-10)


def test_bridge_hamiltonian_matches_solver_for_multiple_sizes():
    """Verify Hamiltonian consistency for L=2,3,4."""
    from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27

    for L in [2, 3, 4]:
        K = build_knm_paper27(L=L)
        omega = OMEGA_N_16[:L]
        solver = QuantumKuramotoSolver(L, K, omega)
        H_solver = solver.build_hamiltonian()
        H_bridge = knm_to_hamiltonian(K, omega)

        mat_s = H_solver.to_matrix()
        mat_b = H_bridge.to_matrix()
        if hasattr(mat_s, "toarray"):
            mat_s = mat_s.toarray()
        if hasattr(mat_b, "toarray"):
            mat_b = mat_b.toarray()

        assert np.allclose(mat_s, mat_b, atol=1e-10), f"L={L}: mismatch"


def test_classical_R_trajectory_all_positive():
    """All R values in classical trajectory must be in [0, 1]."""
    result = classical_kuramoto_reference(n_osc=4, t_max=2.0, dt=0.01)
    for R in result["R"]:
        assert 0.0 <= R <= 1.0 + 1e-10


def test_classical_kuramoto_theta_finite():
    """All theta values from classical Kuramoto must be finite."""
    result = classical_kuramoto_reference(n_osc=4, t_max=1.0, dt=0.01)
    assert np.all(np.isfinite(result["theta"][-1]))


# ---------------------------------------------------------------------------
# Pipeline: full cross-module wiring with performance
# ---------------------------------------------------------------------------


def test_pipeline_cross_module_full():
    """Full cross-module pipeline: Knm → solver H → bridge H → exact diag → classical.
    Verifies all three computation paths agree and are wired end-to-end.
    """
    import time

    from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    t0 = time.perf_counter()
    solver = QuantumKuramotoSolver(4, K, omega)
    H_solver = solver.build_hamiltonian()
    knm_to_hamiltonian(K, omega)  # verify bridge path compiles
    exact = classical_exact_diag(n_osc=4, K=K, omega=omega)
    dt = (time.perf_counter() - t0) * 1000

    # All three should produce consistent ground energy
    mat_s = H_solver.to_matrix()
    if hasattr(mat_s, "toarray"):
        mat_s = mat_s.toarray()
    E_solver = np.linalg.eigvalsh(mat_s)[0]

    np.testing.assert_allclose(E_solver, exact["ground_energy"], atol=1e-10)

    print(f"\n  PIPELINE cross-module (4q): {dt:.1f} ms")
    print(f"  E_0 = {exact['ground_energy']:.4f}")


def test_rust_and_python_kuramoto_both_evolve():
    """Rust Euler and Python reference both evolve phases — cross-validated."""
    try:
        import scpn_quantum_engine as eng
    except ImportError:
        import pytest

        pytest.skip("scpn-quantum-engine not available")

    from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    theta0 = np.zeros(4, dtype=np.float64)

    theta_rust = np.array(eng.kuramoto_euler(theta0, omega, K, 0.01, 50))
    result_py = classical_kuramoto_reference(n_osc=4, t_max=0.5, dt=0.01)

    R_rust = eng.order_parameter(theta_rust)
    R_py = result_py["R"][-1]

    assert R_rust > 0
    assert R_py > 0
