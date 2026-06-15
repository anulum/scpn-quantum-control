# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Commutator Bounds
"""Tests for analytical commutator Trotter error bounds."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import kuramoto_variants as kuramoto_variant_mod
from scpn_quantum_control.phase.kuramoto_variants import (
    HigherOrderKuramotoSpec,
    KuramotoVariant,
    KuramotoVariantResult,
    MonitoredKuramotoSpec,
    PTSymmetricKuramotoSpec,
    build_triadic_ring_terms,
    simulate_higher_order_kuramoto,
    simulate_monitored_kuramoto,
    simulate_pt_symmetric_kuramoto,
)
from scpn_quantum_control.phase.trotter_error import (
    commutator_norm_bound,
    frequency_heterogeneity,
    nested_commutator_norm_bound,
    optimal_dt,
    trotter_error_bound,
    trotter_error_norm,
)


def _variant_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_nm = np.array(
        [
            [0.0, 0.4, 0.0, 0.2],
            [0.4, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.5],
            [0.2, 0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.1, 0.4, 0.8, 1.1], dtype=np.float64)
    theta0 = np.array([0.0, 0.5, 1.7, 2.9], dtype=np.float64)
    return K_nm, omega, theta0


class TestCommutatorBounds:
    def test_commutator_norm_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        gamma = commutator_norm_bound(K, omega)
        assert gamma > 0

    def test_commutator_norm_zero_for_equal_frequencies(self):
        """Key insight: Trotter error vanishes when all frequencies are equal."""
        K = build_knm_paper27(L=4)
        omega_equal = np.ones(4) * 1.0
        gamma = commutator_norm_bound(K, omega_equal)
        assert gamma == pytest.approx(0.0, abs=1e-15)

    def test_bound_exceeds_empirical(self):
        """First-order analytical bound rigorously dominates the empirical spectral error."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        t = 0.5
        reps = 3
        bound = trotter_error_bound(K, omega, t, reps, order=1)
        empirical = trotter_error_norm(K, omega, t, reps, order=1)
        assert bound >= empirical - 1e-12

    def test_second_order_bound_exceeds_empirical(self):
        """Second-order analytical bound rigorously dominates the empirical spectral error."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        t = 0.6
        reps = 2
        bound = trotter_error_bound(K, omega, t, reps, order=2)
        empirical = trotter_error_norm(K, omega, t, reps, order=2)
        assert bound >= empirical - 1e-12

    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("seed", [1, 7, 19, 23])
    def test_bound_dominates_empirical_across_random_systems(self, order, seed):
        """The bound must upper-bound the same-splitting empirical error, every time.

        This is the contract that the empirical measurement and the analytical
        bound describe the *same* two-group product formula in the *same*
        spectral norm; a regression that desynchronises them (wrong norm, wrong
        splitting, or a loosened constant pushed below the true error) fails here.
        """
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 5))
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = rng.uniform(0.05, 0.7)
        omega = rng.uniform(-1.0, 1.2, size=n)
        t = float(rng.uniform(0.2, 1.0))
        reps = int(rng.integers(1, 4))

        bound = trotter_error_bound(K, omega, t, reps, order=order)
        empirical = trotter_error_norm(K, omega, t, reps, order=order)

        assert bound >= empirical - 1e-12

    def test_bound_decreases_with_reps(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        b1 = trotter_error_bound(K, omega, 1.0, 1, order=1)
        b5 = trotter_error_bound(K, omega, 1.0, 5, order=1)
        b10 = trotter_error_bound(K, omega, 1.0, 10, order=1)
        assert b1 > b5 > b10

    def test_second_order_better_than_first(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        b1 = trotter_error_bound(K, omega, 0.5, 5, order=1)
        b2 = trotter_error_bound(K, omega, 0.5, 5, order=2)
        assert b2 < b1

    def test_nested_commutator_bound_matches_exact_small_system(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        h_xy = knm_to_dense_matrix(K, np.zeros_like(omega))
        h_z = knm_to_dense_matrix(np.zeros_like(K), omega)
        comm = h_xy @ h_z - h_z @ h_xy
        nested_xy = h_xy @ comm - comm @ h_xy
        nested_z = h_z @ comm - comm @ h_z
        expected = np.linalg.norm(nested_xy, 2) + np.linalg.norm(nested_z, 2)

        actual = nested_commutator_norm_bound(K, omega, exact_qubit_limit=3)

        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_nested_commutator_bound_has_large_system_upper_bound(self):
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16

        bound = nested_commutator_norm_bound(K, omega, exact_qubit_limit=3)

        assert np.isfinite(bound)
        assert bound > 0.0

    def test_optimal_dt_respects_epsilon(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=1)
        assert result["error_bound"] <= 0.01 * 1.5  # small slack
        assert result["n_steps"] >= 1
        assert result["dt"] > 0

    def test_optimal_dt_order2(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        r1 = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=1)
        r2 = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=2)
        # Second order should need fewer steps
        assert r2["n_steps"] <= r1["n_steps"]

    def test_frequency_heterogeneity(self):
        omega_uniform = np.ones(4) * 1.0
        assert frequency_heterogeneity(omega_uniform) == pytest.approx(0.0)

        omega_spread = np.array([0.5, 1.0, 1.5, 2.0])
        assert frequency_heterogeneity(omega_spread) > 0

    def test_scpn_commutator_norm(self):
        """For 16-layer SCPN, compute and record the commutator norm."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        gamma = commutator_norm_bound(K, omega)
        # From Round 2 derivation: estimated ~18-20 for 16 layers
        assert gamma > 5  # lower bound sanity
        assert gamma < 200  # upper bound sanity

    def test_scpn_optimal_dt_value(self):
        """For 16-layer SCPN at epsilon=0.01, what dt do we need?"""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = optimal_dt(K, omega, epsilon=0.01, t_total=5.0, order=1)
        # Should require many steps for t=5.0 at epsilon=0.01
        assert result["n_steps"] > 100
        assert result["dt"] < 0.1


# ---------------------------------------------------------------------------
# Commutator physics: scaling and universality
# ---------------------------------------------------------------------------


class TestCommutatorPhysics:
    def test_heterogeneity_drives_error(self):
        """More frequency spread → larger commutator norm → larger Trotter error."""
        K = build_knm_paper27(L=4)
        omega_narrow = np.array([1.0, 1.1, 1.2, 1.3])
        omega_wide = np.array([0.5, 1.0, 2.0, 3.0])
        gamma_narrow = commutator_norm_bound(K, omega_narrow)
        gamma_wide = commutator_norm_bound(K, omega_wide)
        assert gamma_wide > gamma_narrow

    def test_error_bound_nonnegative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        b = trotter_error_bound(K, omega, 1.0, 5, order=1)
        assert b >= 0

    def test_invalid_trotter_bound_contracts_are_rejected(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        with pytest.raises(ValueError, match="exact_qubit_limit"):
            nested_commutator_norm_bound(K, omega, exact_qubit_limit=-1)
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            trotter_error_bound(K, omega, t=0.1, reps=1, order=3)
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=4)

    @pytest.mark.parametrize(
        ("bad_K", "bad_omega", "message"),
        [
            (np.ones(3), np.ones(3), "square 2-D"),
            (np.eye(3), np.ones(2), "omega must be 1-D"),
            (np.array([[0.0, np.inf], [np.inf, 0.0]]), np.ones(2), "K contains"),
            (np.eye(2), np.array([0.0, np.nan]), "omega contains"),
        ],
    )
    def test_nested_bound_validates_problem_shape_and_finiteness(self, bad_K, bad_omega, message):
        with pytest.raises(ValueError, match=message):
            nested_commutator_norm_bound(bad_K, bad_omega)

    def test_frequency_heterogeneity_single_frequency_is_zero(self):
        assert frequency_heterogeneity(np.array([1.25])) == pytest.approx(0.0)


class TestKuramotoVariantContracts:
    def test_variant_result_rejects_invalid_trajectory_and_diagnostics(self):
        times = np.array([0.0, 0.1], dtype=np.float64)
        with pytest.raises(ValueError, match="one-dimensional"):
            KuramotoVariantResult(KuramotoVariant.MONITORED, times[:, None], times, "numpy")
        with pytest.raises(ValueError, match="same shape"):
            KuramotoVariantResult(
                KuramotoVariant.MONITORED,
                times,
                np.array([0.2], dtype=np.float64),
                "numpy",
            )
        with pytest.raises(ValueError, match="inside \\[0, 1\\]"):
            KuramotoVariantResult(
                KuramotoVariant.MONITORED,
                times,
                np.array([0.2, 1.2], dtype=np.float64),
                "numpy",
            )
        with pytest.raises(ValueError, match="diagnostic 'readout'"):
            KuramotoVariantResult(
                KuramotoVariant.MONITORED,
                times,
                np.array([0.2, 0.3], dtype=np.float64),
                "numpy",
                diagnostics={"readout": np.array([0.2], dtype=np.float64)},
            )
        with pytest.raises(TypeError, match="unsupported diagnostic"):
            KuramotoVariantResult(
                KuramotoVariant.MONITORED,
                times,
                np.array([0.2, 0.3], dtype=np.float64),
                "numpy",
                diagnostics={"raw": {"not": "serialisable"}},
            )

    def test_variant_specs_reject_metadata_and_shape_errors(self):
        K_nm, omega, theta0 = _variant_problem()
        hyperedges, weights = build_triadic_ring_terms(4, weight=0.15)

        with pytest.raises(ValueError, match="hyper_weights"):
            HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights[:-1], theta0=theta0)
        with pytest.raises(ValueError, match="theta0"):
            HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights, theta0=theta0[:-1])
        with pytest.raises(TypeError, match="metadata keys"):
            MonitoredKuramotoSpec(K_nm, omega, theta0=theta0, metadata={1: "bad"})
        with pytest.raises(TypeError, match="JSON-serialisable"):
            MonitoredKuramotoSpec(K_nm, omega, theta0=theta0, metadata={"bad": object()})
        with pytest.raises(ValueError, match="K_nm must be a square"):
            MonitoredKuramotoSpec(np.ones(4), omega, theta0=theta0)
        with pytest.raises(ValueError, match="omega must have shape"):
            MonitoredKuramotoSpec(K_nm, omega[:-1], theta0=theta0)
        with pytest.raises(ValueError, match="symmetric"):
            MonitoredKuramotoSpec(K_nm + np.triu(np.ones_like(K_nm), 1), omega, theta0=theta0)
        with pytest.raises(ValueError, match="monitor_gain"):
            MonitoredKuramotoSpec(K_nm, omega, monitor_gain=-0.1, theta0=theta0)
        with pytest.raises(ValueError, match="measurement_strength"):
            MonitoredKuramotoSpec(K_nm, omega, measurement_strength=np.inf, theta0=theta0)
        with pytest.raises(ValueError, match="finite values"):
            PTSymmetricKuramotoSpec(
                K_nm,
                omega,
                gain_loss=np.array([0.1, -0.1, np.nan, 0.0], dtype=np.float64),
                theta0=theta0,
            )

    def test_default_theta_initialisation_follows_frequency_phases(self):
        K_nm, omega, _theta0 = _variant_problem()
        hyperedges, weights = build_triadic_ring_terms(4, weight=0.15)

        spec = HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights)

        np.testing.assert_allclose(spec.theta0, np.mod(omega, 2.0 * np.pi))
        assert spec.theta0.flags.writeable is False

    def test_missing_initial_state_guard_is_explicit(self):
        with pytest.raises(ValueError, match="theta0 was not initialised"):
            kuramoto_variant_mod._required_theta0(None)

    def test_variant_time_grid_and_ring_weight_validation(self):
        K_nm, omega, theta0 = _variant_problem()
        hyperedges, weights = build_triadic_ring_terms(4, weight=0.15)
        spec = HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights, theta0=theta0)

        with pytest.raises(ValueError, match="weight must be finite"):
            build_triadic_ring_terms(4, weight=np.inf)
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            simulate_higher_order_kuramoto(spec, dt=0.0, n_steps=2, prefer_rust=False)
        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            simulate_higher_order_kuramoto(spec, dt=0.1, n_steps=0, prefer_rust=False)

    def test_hyperedge_bounds_are_checked_before_simulation(self):
        K_nm, omega, theta0 = _variant_problem()

        with pytest.raises(ValueError, match="hyperedge indices"):
            HigherOrderKuramotoSpec(
                K_nm,
                omega,
                np.array([[0, 1, 4]], dtype=np.int64),
                np.array([0.2], dtype=np.float64),
                theta0=theta0,
            )

    def test_preferred_rust_backends_preserve_metadata(self, monkeypatch):
        K_nm, omega, theta0 = _variant_problem()
        hyperedges, weights = build_triadic_ring_terms(4, weight=0.15)
        times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        r_values = np.array([0.2, 0.35, 0.5], dtype=np.float64)

        fake_engine = SimpleNamespace(
            higher_order_kuramoto_trajectory=lambda *args: (times, r_values),
            monitored_kuramoto_trajectory=lambda *args: (
                times,
                r_values,
                np.array([0.3, 0.4, 0.5], dtype=np.float64),
                np.array([0.1, 0.0, -0.1], dtype=np.float64),
            ),
            pt_symmetric_kuramoto_trajectory=lambda *args: (
                times,
                r_values,
                np.ones(3, dtype=np.float64),
                np.array([0.0, 0.01, -0.01], dtype=np.float64),
            ),
        )
        monkeypatch.setitem(sys.modules, "scpn_quantum_engine", fake_engine)

        higher = simulate_higher_order_kuramoto(
            HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights, theta0=theta0),
            dt=0.1,
            n_steps=2,
        )
        monitored = simulate_monitored_kuramoto(
            MonitoredKuramotoSpec(K_nm, omega, theta0=theta0),
            dt=0.1,
            n_steps=2,
        )
        pt_result = simulate_pt_symmetric_kuramoto(
            PTSymmetricKuramotoSpec(
                K_nm,
                omega,
                gain_loss=np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float64),
                theta0=theta0,
            ),
            dt=0.1,
            n_steps=2,
        )

        assert higher.backend == "rust:higher_order_kuramoto_trajectory"
        assert higher.diagnostics["n_hyperedges"] == 4
        assert monitored.backend == "rust:monitored_kuramoto_trajectory"
        assert monitored.diagnostics["target_r"] == pytest.approx(0.75)
        assert pt_result.backend == "rust:pt_symmetric_kuramoto_trajectory"
        np.testing.assert_allclose(pt_result.diagnostics["pt_norm"], 1.0)


# ---------------------------------------------------------------------------
# Pipeline: Knm → commutator → optimal dt → wired
# ---------------------------------------------------------------------------


class TestCommutatorPipeline:
    def test_pipeline_knm_to_optimal_dt(self):
        """Full pipeline: build_knm → commutator norm → optimal dt for ε=0.01.
        Verifies Trotter error module is wired and produces actionable circuit params.
        """
        import time

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        gamma = commutator_norm_bound(K, omega)
        result = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=1)
        dt_ms = (time.perf_counter() - t0) * 1000

        assert gamma > 0
        assert result["n_steps"] >= 1

        print(f"\n  PIPELINE Knm→Commutator→dt* (4q): {dt_ms:.1f} ms")
        print(f"  γ = {gamma:.4f}, dt* = {result['dt']:.4f}, n_steps = {result['n_steps']}")
