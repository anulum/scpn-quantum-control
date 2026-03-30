# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Loschmidt Echo
"""Tests for Loschmidt echo / DQPT."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.loschmidt_echo import (
    LoschmidtResult,
    loschmidt_quench,
    quench_scan,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestLoschmidtQuench:
    def test_returns_result(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=0.5, K_final=3.0)
        assert isinstance(result, LoschmidtResult)
        assert len(result.times) == 200

    def test_amplitude_starts_at_one(self):
        """G(0) = ⟨ψ_i|ψ_i⟩ = 1."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=1.0, K_final=3.0)
        assert abs(result.loschmidt_amplitude[0] - 1.0) < 1e-6

    def test_no_quench_stays_one(self):
        """K_i = K_f → |G(t)| = 1 for all t (no dynamics)."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=2.0, K_final=2.0, n_times=50)
        assert np.all(result.loschmidt_amplitude > 0.99)

    def test_amplitude_bounded(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=0.5, K_final=4.0)
        assert np.all(result.loschmidt_amplitude >= 0)
        assert np.all(result.loschmidt_amplitude <= 1.0 + 1e-6)

    def test_rate_function_nonnegative(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=0.5, K_final=3.0)
        assert np.all(result.rate_function >= -1e-6)

    def test_large_quench_has_oscillations(self):
        """Large quench on 4 qubits should produce amplitude oscillations.

        3-qubit weak-coupling ground state is |000⟩ (Sz=+3/2 sector,
        1D under XY) — trivially invariant. Use 4 qubits + moderate K_i
        so ground state is in a multi-dimensional Sz sector.
        """
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = loschmidt_quench(omega, T, K_initial=2.0, K_final=5.0, t_max=10.0)
        assert np.std(result.loschmidt_amplitude) > 0.001

    def test_4qubit(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = loschmidt_quench(omega, T, K_initial=1.0, K_final=3.0, n_times=50)
        assert len(result.loschmidt_amplitude) == 50


class TestQuenchScan:
    def test_returns_dict(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = quench_scan(
            omega, T, K_initial=0.5, K_final_range=np.array([1.0, 3.0]), n_times=50
        )
        assert "K_final" in result
        assert "n_cusps" in result
        assert len(result["K_final"]) == 2

    def test_all_finite(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = quench_scan(
            omega, T, K_initial=0.5, K_final_range=np.array([1.0, 2.0, 4.0]), n_times=50
        )
        for key in result:
            assert all(np.isfinite(v) for v in result[key])
