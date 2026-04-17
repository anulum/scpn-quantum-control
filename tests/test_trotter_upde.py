# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Trotter Upde
"""Tests for phase/trotter_upde.py — multi-angle coverage."""

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.trotter_upde import QuantumUPDESolver


def test_default_16_layers():
    solver = QuantumUPDESolver()
    assert solver.n_layers == 16


def test_custom_small_system():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumUPDESolver(K=K, omega=omega)
    assert solver.n_layers == 4


def test_hamiltonian_exists():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumUPDESolver(K=K, omega=omega)
    H = solver.hamiltonian()
    assert H is not None
    assert H.num_qubits == 4


def test_run_returns_R_trajectory():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=5, dt=0.05)
    assert "R" in result
    assert len(result["R"]) == 6  # n_steps + 1
    for r in result["R"]:
        assert 0.0 <= r <= 1.5  # allow some numerical margin


def test_step_and_reset():
    """step() should accumulate state; reset() should reinitialise."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)

    r1 = solver.step(dt=0.1)
    r2 = solver.step(dt=0.1)
    assert r1["R_global"] != r2["R_global"]  # state evolved

    solver.reset()
    r3 = solver.step(dt=0.1)
    assert abs(r3["R_global"] - r1["R_global"]) < 1e-10  # same first step


def test_second_order_trotter_passthrough():
    """QuantumUPDESolver(trotter_order=2) should pass through to solver."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega, trotter_order=2)
    assert solver._solver.trotter_order == 2
    result = solver.run(n_steps=3, dt=0.05)
    assert len(result["R"]) == 4


@pytest.mark.parametrize("L", [2, 3, 4])
def test_various_sizes(L):
    K = build_knm_paper27(L=L)
    omega = OMEGA_N_16[:L]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=3, dt=0.05)
    assert len(result["R"]) == 4
    for r in result["R"]:
        assert np.isfinite(r)


def test_step_returns_R_global():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.step(dt=0.1)
    assert "R_global" in result
    assert 0.0 <= result["R_global"] <= 1.5


def test_hamiltonian_hermitian():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumUPDESolver(K=K, omega=omega)
    H = solver.hamiltonian()
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_R_trajectory_finite():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=5, dt=0.05)
    for r in result["R"]:
        assert np.isfinite(r)


# ---------------------------------------------------------------------------
# UPDE physics: R trajectory and step consistency
# ---------------------------------------------------------------------------


def test_R_bounded_throughout():
    """R must stay in [0, 1] for all time steps."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=10, dt=0.05)
    for r in result["R"]:
        assert 0.0 <= r <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Pipeline: Knm → UPDE → R trajectory → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_upde():
    """Full pipeline: build_knm → UPDE solver → run → R trajectory.
    Verifies UPDE is wired end-to-end, not decorative.
    """
    import time

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    t0 = time.perf_counter()
    solver = QuantumUPDESolver(K=K, omega=omega)
    result = solver.run(n_steps=5, dt=0.05)
    dt = (time.perf_counter() - t0) * 1000

    assert len(result["R"]) == 6
    assert all(np.isfinite(r) for r in result["R"])

    print(f"\n  PIPELINE Knm→UPDE (4q, 5 steps): {dt:.1f} ms")
    print(f"  R: {[f'{r:.4f}' for r in result['R']]}")
