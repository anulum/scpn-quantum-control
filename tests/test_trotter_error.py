"""Tests for Trotter error analysis."""

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.trotter_error import trotter_error_norm, trotter_error_sweep


@pytest.fixture
def small_system():
    n = 3
    return build_knm_paper27(L=n), OMEGA_N_16[:n]


def test_error_at_t_zero(small_system):
    K, omega = small_system
    err = trotter_error_norm(K, omega, t=0.0, reps=1)
    assert err < 1e-10


def test_error_decreases_with_reps(small_system):
    K, omega = small_system
    err_1 = trotter_error_norm(K, omega, t=1.0, reps=1)
    err_4 = trotter_error_norm(K, omega, t=1.0, reps=4)
    assert err_4 < err_1


def test_error_increases_with_time(small_system):
    K, omega = small_system
    err_short = trotter_error_norm(K, omega, t=0.05, reps=2)
    err_long = trotter_error_norm(K, omega, t=0.5, reps=2)
    assert err_long > err_short


def test_raises_for_large_n():
    K = build_knm_paper27(L=11)
    omega = OMEGA_N_16[:11]
    with pytest.raises(ValueError, match="too large"):
        trotter_error_norm(K, omega, t=0.1, reps=1)


def test_sweep_returns_2d(small_system):
    K, omega = small_system
    result = trotter_error_sweep(K, omega, t_values=[0.05, 0.1], reps_values=[1, 2])
    assert len(result["errors"]) == 2
    assert len(result["errors"][0]) == 2
    assert all(e >= 0 for row in result["errors"] for e in row)
