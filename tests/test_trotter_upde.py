"""Tests for phase/trotter_upde.py."""

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
