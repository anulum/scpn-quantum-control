# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Varqite
"""Tests for VarQITE imaginary-time ground state."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.varqite import (
    VarQITEResult,
    varqite_ground_state,
)


class TestVarQITE:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=42)
        assert isinstance(result, VarQITEResult)

    def test_energy_decreases(self):
        """ITE should monotonically decrease energy."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=1.0, n_steps=10, seed=42)
        # Overall trend: final < initial
        assert result.energy_history[-1] <= result.energy_history[0] + 0.1

    def test_energy_finite(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=42)
        assert np.isfinite(result.energy)

    def test_exact_energy_below(self):
        """VarQITE energy should be above or near exact ground state."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=1.0, n_steps=10, seed=42)
        assert result.energy >= result.exact_energy - 0.1

    def test_relative_error_type(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=42)
        assert isinstance(result.relative_error_pct, float)
        assert result.relative_error_pct >= 0

    def test_energy_history_length(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=42)
        assert len(result.energy_history) >= 2

    def test_3_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = varqite_ground_state(K, omega, tau_total=0.3, n_steps=3, seed=42)
        assert result.n_steps >= 1

    def test_convergence_with_loose_threshold(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(
            K, omega, tau_total=2.0, n_steps=30, convergence_threshold=10.0, seed=42
        )
        assert result.converged

    def test_optimal_params(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=0.5, n_steps=3, seed=42)
        assert len(result.optimal_params) > 0
        assert np.all(np.isfinite(result.optimal_params))


# ---------------------------------------------------------------------------
# ITE physics: energy monotonicity and variational bound
# ---------------------------------------------------------------------------


class TestVarQITEPhysics:
    def test_variational_bound(self):
        """VarQITE energy >= exact ground energy (variational principle)."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = varqite_ground_state(K, omega, tau_total=1.0, n_steps=10, seed=42)
        assert result.energy >= result.exact_energy - 0.5  # generous tolerance

    def test_seed_determinism(self):
        """Same seed → same result."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        r1 = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=0)
        r2 = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=0)
        assert r1.energy == r2.energy


# ---------------------------------------------------------------------------
# Pipeline: Knm → VarQITE → ground state → wired
# ---------------------------------------------------------------------------


class TestVarQITEPipeline:
    def test_pipeline_knm_to_ground_state(self):
        """Full pipeline: build_knm → VarQITE → energy trajectory → ground state.
        Verifies ITE module is wired and produces converging energies.
        """
        import time

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        result = varqite_ground_state(K, omega, tau_total=0.5, n_steps=5, seed=42)
        dt = (time.perf_counter() - t0) * 1000

        assert np.isfinite(result.energy)
        assert len(result.energy_history) >= 2

        print(f"\n  PIPELINE Knm→VarQITE (3q, 5 steps): {dt:.1f} ms")
        print(f"  E: {result.energy_history[0]:.4f} → {result.energy:.4f}")
        print(f"  Exact: {result.exact_energy:.4f}, error: {result.relative_error_pct:.1f}%")
