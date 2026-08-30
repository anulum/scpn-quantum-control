# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Robustness Certificate
"""Tests for adiabatic robustness certificate."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.identity import robustness as robustness_module
from scpn_quantum_control.identity.robustness import (
    RobustnessCertificate,
    compute_robustness_certificate,
    gap_vs_perturbation_scan,
    perturbation_fidelity,
)


def _small_system() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """4-oscillator system for fast tests."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    return K, omega


class TestRobustnessCertificate:
    """Exercise exact-gap certificate construction and bounds."""

    def test_returns_certificate(self) -> None:
        """Return the public certificate type."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert isinstance(cert, RobustnessCertificate)

    def test_gap_positive(self) -> None:
        """Report a positive exact gap for the reference system."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert cert.energy_gap > 0

    def test_max_safe_is_half_gap(self) -> None:
        """Set the safe perturbation threshold to half the exact gap."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert cert.max_safe_perturbation == pytest.approx(cert.energy_gap / 2.0)

    def test_min_t2_inversely_proportional_to_gap(self) -> None:
        """Derive the dephasing-time bound from the exact gap."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        expected = 2.0 / cert.energy_gap
        assert cert.min_t2_for_stability == pytest.approx(expected)

    def test_small_noise_low_transition(self) -> None:
        """Keep the perturbative transition bound low for weak noise."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega, noise_strength=0.001)
        assert cert.transition_probability < 0.1

    def test_large_noise_high_transition(self) -> None:
        """Cap the transition bound at unity for strong noise."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega, noise_strength=100.0)
        assert cert.transition_probability == pytest.approx(1.0)

    def test_transition_scales_quadratically(self) -> None:
        """Follow quadratic perturbative scaling below saturation."""
        K, omega = _small_system()
        c1 = compute_robustness_certificate(K, omega, noise_strength=0.01)
        c2 = compute_robustness_certificate(K, omega, noise_strength=0.02)
        ratio = c2.transition_probability / max(c1.transition_probability, 1e-30)
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_eigenvalues_ordered(self) -> None:
        """Preserve the exact diagonalizer's eigenvalue ordering."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        for i in range(len(cert.eigenvalues) - 1):
            assert cert.eigenvalues[i] <= cert.eigenvalues[i + 1] + 1e-10

    def test_n_oscillators(self) -> None:
        """Record the coupling-matrix width on the certificate."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        assert cert.n_oscillators == 4

    def test_zero_gap_uses_fail_closed_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return saturated bounds when no resolvable energy gap exists."""

        def fake_exact_diag(_n: int, **_kwargs: object) -> dict[str, NDArray[np.float64]]:
            return {
                "eigenvalues": np.array([0.0], dtype=np.float64),
                "ground_state": np.array([1.0], dtype=np.float64),
            }

        monkeypatch.setattr(robustness_module, "classical_exact_diag", fake_exact_diag)
        K, omega = _small_system()

        cert = compute_robustness_certificate(K, omega)

        assert cert.energy_gap == 0.0
        assert cert.max_safe_perturbation == 0.0
        assert cert.min_t2_for_stability == pytest.approx(2.0e15)
        assert cert.transition_probability == 1.0
        assert cert.adiabatic_bound == 1.0
        assert cert.eigenvalues == [0.0]


class TestPerturbationFidelity:
    """Exercise numerical ground-state overlap calculations."""

    def test_zero_perturbation_unity_fidelity(self) -> None:
        """Return unit overlap for an unchanged coupling matrix."""
        K, omega = _small_system()
        delta = np.zeros_like(K)
        fid = perturbation_fidelity(K, omega, delta)
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_small_perturbation_high_fidelity(self) -> None:
        """Retain high overlap under a weak symmetric perturbation."""
        K, omega = _small_system()
        rng = np.random.default_rng(42)
        delta = rng.normal(0, 0.001, size=K.shape)
        delta = (delta + delta.T) / 2.0
        np.fill_diagonal(delta, 0.0)
        fid = perturbation_fidelity(K, omega, delta)
        assert fid > 0.99

    def test_large_perturbation_low_fidelity(self) -> None:
        """Expose overlap loss under a strong symmetric perturbation."""
        K, omega = _small_system()
        rng = np.random.default_rng(42)
        delta = rng.normal(0, 10.0, size=K.shape)
        delta = (delta + delta.T) / 2.0
        np.fill_diagonal(delta, 0.0)
        fid = perturbation_fidelity(K, omega, delta)
        assert fid < 0.9

    def test_fidelity_bounded(self) -> None:
        """Keep numerical overlap within probability bounds."""
        K, omega = _small_system()
        rng = np.random.default_rng(42)
        delta = rng.normal(0, 0.1, size=K.shape)
        delta = (delta + delta.T) / 2.0
        fid = perturbation_fidelity(K, omega, delta)
        assert 0 <= fid <= 1.0


class TestGapVsPerturbationScan:
    """Exercise deterministic perturbation-strength scans."""

    def test_scan_returns_all_keys(self) -> None:
        """Return aligned theoretical and numerical result columns."""
        K, omega = _small_system()
        results = gap_vs_perturbation_scan(K, omega, n_samples=5)
        assert "noise_strength" in results
        assert "p_transition_theory" in results
        assert "fidelity_numerical" in results
        assert len(results["noise_strength"]) == 5

    def test_fidelity_decreases_with_noise(self) -> None:
        """Show the overall overlap trend across a wide noise range."""
        K, omega = _small_system()
        wide_range = np.linspace(0.01, 5.0, 10, dtype=np.float64)
        results = gap_vs_perturbation_scan(K, omega, noise_range=wide_range)
        fids = results["fidelity_numerical"]
        # Overall trend should decrease (first should be higher than last)
        assert fids[0] >= fids[-1]

    def test_theory_increases_with_noise(self) -> None:
        """Increase the theoretical transition bound with noise strength."""
        K, omega = _small_system()
        results = gap_vs_perturbation_scan(K, omega, n_samples=10)
        p_theory = results["p_transition_theory"]
        for i in range(1, len(p_theory)):
            assert p_theory[i] >= p_theory[i - 1]

    def test_zero_gap_scan_saturates_theory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Saturate every theoretical scan point when the gap is zero."""

        def fake_certificate(
            _K: NDArray[np.float64],
            _omega: NDArray[np.float64],
            noise_strength: float = 0.01,
            sweep_rate: float = 0.1,
        ) -> RobustnessCertificate:
            del noise_strength, sweep_rate
            return RobustnessCertificate(0.0, 0.0, 2.0e15, 1.0, 1.0, 2, [0.0])

        def fake_fidelity(
            _K: NDArray[np.float64],
            _omega: NDArray[np.float64],
            _delta_K: NDArray[np.float64],
        ) -> float:
            return 1.0

        monkeypatch.setattr(robustness_module, "compute_robustness_certificate", fake_certificate)
        monkeypatch.setattr(robustness_module, "perturbation_fidelity", fake_fidelity)
        K, omega = _small_system()

        results = gap_vs_perturbation_scan(
            K,
            omega,
            noise_range=np.array([0.1, 0.2], dtype=np.float64),
        )

        assert results["p_transition_theory"] == [1.0, 1.0]
        assert results["fidelity_numerical"] == [1.0, 1.0]

    def test_identity_finding(self) -> None:
        """Record the actual robustness certificate for SCPN 4-oscillator."""
        K, omega = _small_system()
        cert = compute_robustness_certificate(K, omega)
        print(f"\n  Energy gap Δ = {cert.energy_gap:.6f}")
        print(f"  Max safe perturbation = {cert.max_safe_perturbation:.6f}")
        print(f"  Min T2 for stability = {cert.min_t2_for_stability:.2f} μs")
        print(f"  Eigenvalues: {cert.eigenvalues[:4]}")
        assert isinstance(cert.energy_gap, float)
