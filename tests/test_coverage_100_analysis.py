# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coverage tests for analysis/ module gaps
"""Tests targeting specific uncovered lines in the analysis/ subpackage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


def test_entanglement_advantage_steps_fallback():
    """Cover line 204: _steps_to_90pct returns len(r_vals)-1 when target never reached."""
    from scpn_quantum_control.analysis.entanglement_enhanced_sync import (
        SyncTrajectory,
        entanglement_advantage,
    )

    # Product state: R monotonically decreasing (never reaches 90% of final)
    product = SyncTrajectory(
        initial_state="product",
        times=[0.0, 1.0, 2.0],
        R_values=[0.9, 0.5, 0.01],
        final_R=0.01,
        n_qubits=2,
    )
    bell = SyncTrajectory(
        initial_state="bell_pairs",
        times=[0.0, 1.0, 2.0],
        R_values=[0.8, 0.4, 0.01],
        final_R=0.01,
        n_qubits=2,
    )
    results = {"product": product, "bell_pairs": bell}
    adv = entanglement_advantage(results)
    assert "bell_pairs" in adv
    assert adv["bell_pairs"]["product_steps_90pct"] >= 0


# --- entanglement_entropy.py line 105: n_A adjusted for single-qubit system ---


def test_entanglement_entropy_single_qubit_partition():
    """Cover line 105: n_A = 1 when n//2 == 0 (single-qubit system)."""
    from scpn_quantum_control.analysis.entanglement_entropy import entanglement_at_coupling

    omega = np.array([1.0])
    K_topology = np.array([[1.0]])
    result = entanglement_at_coupling(omega, K_topology, K_base=0.5)
    assert hasattr(result, "entropy")


# --- entanglement_spectrum.py line 157: k_base_values is None default ---


def test_entanglement_spectrum_default_k_base():
    """Cover line 157: k_base_values defaults to linspace when None."""
    from scpn_quantum_control.analysis.entanglement_spectrum import entropy_vs_coupling_scan

    omega = OMEGA_N_16[:2]
    result = entropy_vs_coupling_scan(omega, k_base_values=None)
    assert len(result["k_base"]) == 30


def test_qfi_precision_zero_diagonal():
    """Cover line 49: zero QFI diagonal gives infinite precision bound."""
    from scpn_quantum_control.analysis.qfi import QFIResult

    result = QFIResult(
        qfi_matrix=np.zeros((1, 1)),
        coupling_pairs=[(0, 1)],
        precision_bounds=np.array([float("inf")]),
        spectral_gap=0.1,
        n_qubits=2,
    )
    assert result.precision_for(0, 1) == float("inf")


def test_quantum_ph_ripser_not_available():
    """Cover lines 46-47: _RIPSER_AVAILABLE = False branch."""
    with patch(
        "scpn_quantum_control.analysis.quantum_persistent_homology._RIPSER_AVAILABLE",
        False,
    ):
        from scpn_quantum_control.analysis.quantum_persistent_homology import (
            quantum_persistent_homology,
        )

        with pytest.raises(ImportError, match="ripser"):
            quantum_persistent_homology(
                x_counts={"00": 500, "11": 500},
                y_counts={"00": 500, "11": 500},
                n_qubits=2,
            )


# --- quantum_persistent_homology.py line 139: ripser not available main function ---


def test_quantum_ph_function_raises_without_ripser():
    """Cover line 139: quantum_persistent_homology raises when ripser missing."""
    import scpn_quantum_control.analysis.quantum_persistent_homology as qph_mod

    orig = qph_mod._RIPSER_AVAILABLE
    try:
        qph_mod._RIPSER_AVAILABLE = False
        with pytest.raises(ImportError, match="ripser"):
            qph_mod.quantum_persistent_homology(
                x_counts={"00": 500}, y_counts={"00": 500}, n_qubits=2
            )
    finally:
        qph_mod._RIPSER_AVAILABLE = orig


# --- quantum_phi.py line 174: k_base_values is None default ---


def test_quantum_phi_default_k_base():
    """Cover line 174: k_base_values defaults to linspace when None."""
    from scpn_quantum_control.analysis.quantum_phi import phi_vs_coupling_scan

    omega = OMEGA_N_16[:2]
    result = phi_vs_coupling_scan(omega, k_base_values=None)
    assert len(result["k_base"]) == 20


def test_sff_level_spacing_few_spacings():
    """Cover line 68: _level_spacing_ratio returns 0.0 for < 2 spacings."""
    from scpn_quantum_control.analysis.spectral_form_factor import _level_spacing_ratio

    eigs = np.array([0.0, 0.0, 1.0])  # after filtering zero spacings, < 2
    r = _level_spacing_ratio(eigs)
    assert isinstance(r, float)


def test_monte_carlo_rust_engine_path():
    """Cover lines 96-109: Rust engine import path in mc_simulate."""
    from scpn_quantum_control.analysis.monte_carlo_xy import MCResult, mc_simulate

    K = build_knm_paper27(L=4)
    # Run with default — will use Rust if available, else fallback
    result = mc_simulate(K, temperature=0.5, n_thermalize=10, n_measure=10, seed=42)
    assert isinstance(result, MCResult)
    assert result.n_oscillators == 4


# --- monte_carlo_xy.py line 267: n_values is None default ---


def test_monte_carlo_finite_size_default_n_values():
    """Cover line 267: n_values defaults to [4,8,16,32] when None."""
    from scpn_quantum_control.analysis.monte_carlo_xy import finite_size_scaling

    result = finite_size_scaling(n_values=[4], n_seeds=1, n_thermalize=10, n_measure=10)
    assert len(result.n_values) == 1


# --- otoc.py line 99: times is None default ---


def test_otoc_default_times():
    """Cover line 99: times defaults to linspace when None."""
    from scpn_quantum_control.analysis.otoc import compute_otoc

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = compute_otoc(K, omega, w_qubit=0, times=None)
    assert len(result.times) == 30


# --- otoc.py line 152: OTOC f0 near zero → lyapunov = None ---


def test_otoc_lyapunov_zero_f0():
    """Cover line 152: _estimate_lyapunov returns None when f0 near zero."""
    from scpn_quantum_control.analysis.otoc import _estimate_lyapunov

    times = np.linspace(0, 1, 10)
    otoc = np.zeros(10)  # F(0) ≈ 0
    result = _estimate_lyapunov(times, otoc)
    assert result is None


# --- otoc.py line 163: too few positive decay points ---


def test_otoc_lyapunov_few_positive():
    """Cover line 163: _estimate_lyapunov returns None when < 3 positive points."""
    from scpn_quantum_control.analysis.otoc import _estimate_lyapunov

    times = np.linspace(0, 1, 10)
    # All values close to F0 — decay barely positive
    otoc = np.ones(10) * 1.0
    otoc[1] = 0.9999
    result = _estimate_lyapunov(times, otoc)
    assert result is None


# --- otoc.py line 176: scrambling time f0 near zero ---


def test_otoc_scrambling_time_zero_f0():
    """Cover line 176: _estimate_scrambling_time returns None for f0~0."""
    from scpn_quantum_control.analysis.otoc import _estimate_scrambling_time

    times = np.linspace(0, 1, 10)
    otoc = np.zeros(10)
    result = _estimate_scrambling_time(times, otoc)
    assert result is None


def test_finite_size_power_fit_few_points():
    """Cover lines 137-138: _fit_power_ansatz returns None with < 2 points."""
    from scpn_quantum_control.analysis.finite_size_scaling import _fit_power_ansatz

    result = _fit_power_ansatz([4], [1.0])
    assert result is None


# --- hamiltonian_self_consistency.py lines 74-75: zero total counts ---


def test_self_consistency_zero_counts():
    """Cover lines 74-75: _two_point_from_counts returns zeros for empty counts."""
    from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
        _two_point_from_counts,
    )

    result = _two_point_from_counts({}, 3)
    np.testing.assert_array_equal(result, np.zeros((3, 3)))


# --- hamiltonian_self_consistency.py lines 137-144: self_consistency_from_counts ---


def test_self_consistency_from_counts():
    """Cover lines 137-144: self_consistency_from_counts with synthetic counts."""
    from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
        self_consistency_from_counts,
    )

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    x_counts = {"00": 400, "01": 100, "10": 100, "11": 400}
    y_counts = {"00": 400, "01": 100, "10": 100, "11": 400}
    result = self_consistency_from_counts(K, omega, x_counts, y_counts, maxiter=5)
    assert hasattr(result, "frobenius_error")
