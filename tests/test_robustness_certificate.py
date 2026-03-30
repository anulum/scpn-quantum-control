# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Robustness Certificate
"""Tests for adiabatic robustness certificate."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.identity.robustness import (
    RobustnessCertificate,
    compute_robustness_certificate,
    gap_vs_perturbation_scan,
    perturbation_fidelity,
)


def _small_system():
    """4-oscillator system for fast tests."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    return K, omega


class TestRobustnessCertificate:
    def test_returns_certificate(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert isinstance(cert, RobustnessCertificate)

    def test_gap_positive(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert cert.energy_gap > 0

    def test_max_safe_is_half_gap(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert cert.max_safe_perturbation == pytest.approx(cert.energy_gap / 2.0)

    def test_min_t2_inversely_proportional_to_gap(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        expected = 2.0 / cert.energy_gap
        assert cert.min_t2_for_stability == pytest.approx(expected)

    def test_small_noise_low_transition(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega, noise_strength=0.001)
        assert cert.transition_probability < 0.1

    def test_large_noise_high_transition(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega, noise_strength=100.0)
        assert cert.transition_probability == pytest.approx(1.0)

    def test_transition_scales_quadratically(self):
        K, omega = _small_system()
        c1 = compute_robustness_certificate(K, omega, noise_strength=0.01)
        c2 = compute_robustness_certificate(K, omega, noise_strength=0.02)
        ratio = c2.transition_probability / max(c1.transition_probability, 1e-30)
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_eigenvalues_ordered(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        for i in range(len(cert.eigenvalues) - 1):
            assert cert.eigenvalues[i] <= cert.eigenvalues[i + 1] + 1e-10

    def test_n_oscillators(self):
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert cert.n_oscillators == 4


class TestPerturbationFidelity:
    def test_zero_perturbation_unity_fidelity(self):
        K, omega = _small_system()
        delta = np.zeros_like(K)
        fid = perturbation_fidelity(K, omega, delta)
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_small_perturbation_high_fidelity(self):
        K, omega = _small_system()
        rng = np.random.default_rng(42)
        delta = rng.normal(0, 0.001, size=K.shape)
        delta = (delta + delta.T) / 2.0
        np.fill_diagonal(delta, 0.0)
        fid = perturbation_fidelity(K, omega, delta)
        assert fid > 0.99

    def test_large_perturbation_low_fidelity(self):
        K, omega = _small_system()
        rng = np.random.default_rng(42)
        delta = rng.normal(0, 10.0, size=K.shape)
        delta = (delta + delta.T) / 2.0
        np.fill_diagonal(delta, 0.0)
        fid = perturbation_fidelity(K, omega, delta)
        assert fid < 0.9

    def test_fidelity_bounded(self):
        K, omega = _small_system()
        rng = np.random.default_rng(42)
        delta = rng.normal(0, 0.1, size=K.shape)
        delta = (delta + delta.T) / 2.0
        fid = perturbation_fidelity(K, omega, delta)
        assert 0 <= fid <= 1.0


class TestGapVsPerturbationScan:
    def test_scan_returns_all_keys(self):
        K, omega = _small_system()
        results = gap_vs_perturbation_scan(K, omega, n_samples=5)
        assert "noise_strength" in results
        assert "p_transition_theory" in results
        assert "fidelity_numerical" in results
        assert len(results["noise_strength"]) == 5

    def test_fidelity_decreases_with_noise(self):
        K, omega = _small_system()
        wide_range = np.linspace(0.01, 5.0, 10)
        results = gap_vs_perturbation_scan(K, omega, noise_range=wide_range)
        fids = results["fidelity_numerical"]
        # Overall trend should decrease (first should be higher than last)
        assert fids[0] >= fids[-1]

    def test_theory_increases_with_noise(self):
        K, omega = _small_system()
        results = gap_vs_perturbation_scan(K, omega, n_samples=10)
        p_theory = results["p_transition_theory"]
        for i in range(1, len(p_theory)):
            assert p_theory[i] >= p_theory[i - 1]

    def test_identity_finding(self):
        """Record the actual robustness certificate for SCPN 4-oscillator."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        print(f"\n  Energy gap Δ = {cert.energy_gap:.6f}")
        print(f"  Max safe perturbation = {cert.max_safe_perturbation:.6f}")
        print(f"  Min T2 for stability = {cert.min_t2_for_stability:.2f} μs")
        print(f"  Eigenvalues: {cert.eigenvalues[:4]}")
        assert isinstance(cert.energy_gap, float)
