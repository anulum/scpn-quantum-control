# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Enaqt
"""Tests for ENAQT noise optimisation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.analysis.enaqt as enaqt_module
from scpn_quantum_control.analysis.enaqt import (
    ENAQTResult,
    enaqt_scan,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError


def _array(values: object) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return result


def _complex_array(values: object) -> NDArray[np.complex128]:
    result: NDArray[np.complex128] = np.asarray(values, dtype=np.complex128)
    return result


def _density_from_state(psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    result: NDArray[np.complex128] = np.asarray(np.outer(psi, psi.conj()), dtype=np.complex128)
    return result


def _linspace(start: float, stop: float, num: int) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.linspace(start, stop, num, dtype=np.float64)
    return result


def _logspace(start: float, stop: float, num: int) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.logspace(start, stop, num, dtype=np.float64)
    return result


class TestENAQT:
    def test_rejects_dense_budget_before_hamiltonian_allocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        K = build_knm_paper27(L=10)
        omega = OMEGA_N_16[:10]

        def fail_if_dense_hamiltonian_is_requested(
            _coupling: NDArray[np.float64],
            _omega: NDArray[np.float64],
            *,
            max_dense_gib: float | None = None,
        ) -> NDArray[np.complex128]:
            del max_dense_gib
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            enaqt_module,
            "knm_to_dense_matrix",
            fail_if_dense_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="ENAQT dense density"):
            enaqt_scan(
                K,
                omega,
                gamma_range=_array([0.1]),
                n_steps=1,
                max_dense_gib=1e-12,
            )

    def test_passes_dense_budget_to_bridge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        seen_budgets: list[float | None] = []

        def fake_dense_matrix(
            _coupling: NDArray[np.float64],
            _omega: NDArray[np.float64],
            *,
            max_dense_gib: float | None = None,
        ) -> NDArray[np.complex128]:
            seen_budgets.append(max_dense_gib)
            return np.zeros((4, 4), dtype=np.complex128)

        monkeypatch.setattr(enaqt_module, "knm_to_dense_matrix", fake_dense_matrix)

        enaqt_scan(
            K,
            omega,
            gamma_range=_array([0.1]),
            n_steps=1,
            max_dense_gib=0.25,
        )

        assert seen_budgets == [0.25]

    def test_returns_result(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=_array([0.01, 0.1, 1.0]), n_steps=10)
        assert isinstance(result, ENAQTResult)

    def test_optimal_gamma_positive(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=_array([0.01, 0.1, 1.0]), n_steps=10)
        assert result.optimal_gamma > 0

    def test_r_bounded(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=_array([0.01, 0.1, 1.0]), n_steps=10)
        assert 0 <= result.optimal_r <= 1.0
        for r in result.r_values:
            assert 0 <= r <= 1.0 + 1e-6

    def test_enhancement_type(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=_array([0.01, 0.5, 5.0]), n_steps=10)
        assert isinstance(result.enhancement, float)
        assert result.enhancement > 0

    def test_gamma_values_match(self) -> None:
        gammas = _array([0.01, 0.1, 1.0])
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=gammas, n_steps=10)
        assert len(result.r_values) == 3

    def test_3_oscillators(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = enaqt_scan(K, omega, gamma_range=_array([0.1, 1.0]), n_steps=5)
        assert isinstance(result, ENAQTResult)

    def test_scpn_enaqt(self) -> None:
        """Record ENAQT optimum for SCPN defaults."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = enaqt_scan(K, omega, gamma_range=_logspace(-2, 1, 8), n_steps=20)
        print("\n  ENAQT (3 osc):")
        print(f"  Optimal γ = {result.optimal_gamma:.4f}")
        print(f"  R at optimum = {result.optimal_r:.4f}")
        print(f"  Coherent R = {result.coherent_r:.4f}")
        print(f"  Enhancement = {result.enhancement:.2f}x")
        assert isinstance(result.optimal_gamma, float)


def test_enaqt_coherent_r_positive() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=_array([0.0, 0.5]), n_steps=5)
    assert result.coherent_r >= 0


def test_enaqt_r_values_bounded() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=_linspace(0.01, 2.0, 5), n_steps=5)
    for r in result.r_values:
        assert 0 <= r <= 1.0 + 1e-10


def test_enaqt_optimal_gamma_positive() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=_logspace(-2, 1, 5), n_steps=5)
    assert result.optimal_gamma >= 0


def test_enaqt_enhancement_finite() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=_logspace(-2, 1, 5), n_steps=5)
    assert np.isfinite(result.enhancement)


# ---------------------------------------------------------------------------
# Coverage: internal Lindblad and density matrix helpers
# ---------------------------------------------------------------------------


class TestLindbladEvolve:
    def test_preserves_trace(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        dim = 4
        psi = np.ones(dim, dtype=np.complex128) / 2.0
        rho = _density_from_state(psi)
        H = np.diag(_array([0.0, 1.0, 2.0, 3.0])).astype(np.complex128)
        rho_new = _lindblad_evolve(rho, H, gamma=0.5, dt=0.01, n_qubits=2)
        np.testing.assert_allclose(np.trace(rho_new).real, 1.0, atol=1e-10)

    def test_hermitian_output(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        psi = _complex_array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
        rho = _density_from_state(psi)
        H = np.diag(_array([0.0, 1.0, 1.0, 2.0])).astype(np.complex128)
        rho_new = _lindblad_evolve(rho, H, gamma=0.1, dt=0.01, n_qubits=2)
        np.testing.assert_allclose(rho_new, rho_new.conj().T, atol=1e-10)

    def test_zero_gamma_unitary(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        psi = _complex_array([1.0, 0.0, 0.0, 0.0])
        rho = _density_from_state(psi)
        H = np.diag(_array([0.0, 1.0, 2.0, 3.0])).astype(np.complex128)
        rho_new = _lindblad_evolve(rho, H, gamma=0.0, dt=0.01, n_qubits=2)
        # Pure state remains pure (purity close to 1)
        purity = float(np.trace(rho_new @ rho_new).real)
        assert purity > 0.99

    def test_zero_gamma_matches_exact_unitary_phase(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        psi = _complex_array([1.0, 1.0]) / np.sqrt(2.0)
        rho = _density_from_state(psi)
        H = np.diag(_array([0.0, 1.0])).astype(np.complex128)
        dt = 0.3

        rho_new = _lindblad_evolve(rho, H, gamma=0.0, dt=dt, n_qubits=1)

        expected = _complex_array(
            [
                [0.5, 0.5 * np.exp(1j * dt)],
                [0.5 * np.exp(-1j * dt), 0.5],
            ]
        )
        np.testing.assert_allclose(rho_new, expected, atol=1e-12)

    def test_dephasing_matches_exact_exponential_decay(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        psi = _complex_array([1.0, 1.0]) / np.sqrt(2.0)
        rho = _density_from_state(psi)
        H = np.zeros((2, 2), dtype=np.complex128)
        gamma = 0.7
        dt = 0.4

        rho_new = _lindblad_evolve(rho, H, gamma=gamma, dt=dt, n_qubits=1)

        expected_coherence = 0.5 * np.exp(-2.0 * gamma * dt)
        expected = _complex_array(
            [
                [0.5, expected_coherence],
                [expected_coherence, 0.5],
            ]
        )
        np.testing.assert_allclose(rho_new, expected, atol=1e-12)


class TestRFromDensityMatrix:
    def test_pure_state(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _r_from_density_matrix

        psi = _complex_array([1.0, 0.0, 0.0, 0.0])
        rho = _density_from_state(psi)
        r = _r_from_density_matrix(rho, 2)
        assert 0 <= r <= 1.0

    def test_maximally_mixed_phases_degenerate(self) -> None:
        from scpn_quantum_control.analysis.enaqt import _r_from_density_matrix

        rho = np.asarray(np.eye(4, dtype=np.complex128) / 4.0, dtype=np.complex128)
        r = _r_from_density_matrix(rho, 2)
        # Maximally mixed: ⟨X⟩=⟨Y⟩=0 → arctan2(0,0)=0 → all phases=0 → R=1
        # This is an artefact of the phase extraction, not physical coherence
        assert 0 <= r <= 1.0


class TestEnaqtDefaults:
    def test_default_gamma_range(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, n_steps=5)
        assert len(result.gamma_values) == 20  # default logspace -3 to 1, 20 pts
