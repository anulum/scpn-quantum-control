# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Loschmidt Echo
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


# ---------------------------------------------------------------------------
# DQPT physics: rate function cusps and echo properties
# ---------------------------------------------------------------------------


class TestLoschmidtPhysics:
    def test_rate_function_zero_at_t_zero(self):
        """r(0) = -ln|G(0)|²/N = -ln(1)/N = 0."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=1.0, K_final=3.0)
        assert abs(result.rate_function[0]) < 1e-6

    def test_times_monotonic(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=0.5, K_final=3.0)
        assert np.all(np.diff(result.times) > 0)

    def test_has_n_cusps_attribute(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=0.5, K_final=3.0)
        assert isinstance(result.n_cusps, int)
        assert result.n_cusps >= 0


# ---------------------------------------------------------------------------
# Pipeline: Knm → Loschmidt quench → DQPT → wired
# ---------------------------------------------------------------------------


class TestLoschmidtPipeline:
    def test_pipeline_knm_to_dqpt(self):
        """Full pipeline: build_knm → quench → Loschmidt echo → rate function.
        Verifies DQPT module is wired and produces topological time data.
        """
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        result = loschmidt_quench(omega, K, K_initial=0.5, K_final=3.0, n_times=50)
        dt = (time.perf_counter() - t0) * 1000

        assert abs(result.loschmidt_amplitude[0] - 1.0) < 1e-6
        assert np.all(result.rate_function >= -1e-6)

        print(f"\n  PIPELINE Knm→Loschmidt (3q, 50 times): {dt:.1f} ms")
        print(f"  n_cusps = {result.n_cusps}")
        print(f"  min |G|² = {np.min(result.loschmidt_amplitude):.6f}")


# ---------------------------------------------------------------------------
# Coverage: default parameters, quench scan defaults, cusp detection
# ---------------------------------------------------------------------------


class TestQuenchScanDefaults:
    def test_default_k_final_range(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = quench_scan(omega, T, K_initial=0.5, n_times=20)
        assert len(result["K_final"]) == 10  # default linspace 0.5-5.0, 10 pts

    def test_max_rate_positive(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = quench_scan(omega, T, K_initial=0.5, K_final_range=np.array([3.0]), n_times=50)
        assert result["max_rate"][0] >= 0


class TestLoschmidtCuspDetection:
    def test_cusp_times_sorted(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = loschmidt_quench(omega, T, K_initial=1.0, K_final=5.0, t_max=10.0, n_times=200)
        if result.n_cusps > 1:
            for i in range(len(result.cusp_times) - 1):
                assert result.cusp_times[i] < result.cusp_times[i + 1]

    def test_stores_ki_kf(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=1.5, K_final=3.5)
        assert result.K_initial == 1.5
        assert result.K_final == 3.5


class TestLoschmidtSmallNtimes:
    def test_n_times_4(self):
        """Very few time steps should still work (edge for cusp detection)."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = loschmidt_quench(omega, T, K_initial=1.0, K_final=3.0, n_times=4)
        assert len(result.times) == 4
        assert result.n_cusps >= 0


class TestRateFunctionCap:
    def test_rate_cap_for_near_zero_amplitude(self):
        """Rate function caps at 30/N when amplitude is near zero.

        Construct a quench where overlaps with final eigenstates sum
        to produce near-cancellation at specific times.  The rate function
        should cap rather than diverge.
        """
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        # Large quench across transition — some time points will have very small |G|
        result = loschmidt_quench(omega, T, K_initial=0.1, K_final=10.0, t_max=20.0, n_times=500)
        # Rate should never exceed cap
        assert np.all(result.rate_function <= 30.0 / 4 + 1e-6)
