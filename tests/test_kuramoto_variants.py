# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto variant trajectory tests
"""Behavioural tests for higher-order, monitored, and PT-symmetric Kuramoto variants."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control import build_kuramoto_problem, simulate_variant_trajectory
from scpn_quantum_control.phase import (
    HigherOrderKuramotoSpec,
    KuramotoVariant,
    MonitoredKuramotoSpec,
    PTSymmetricKuramotoSpec,
    build_triadic_ring_terms,
    simulate_higher_order_kuramoto,
    simulate_monitored_kuramoto,
    simulate_pt_symmetric_kuramoto,
)


def _ring_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_nm = np.array(
        [
            [0.0, 0.45, 0.0, 0.45],
            [0.45, 0.0, 0.45, 0.0],
            [0.0, 0.45, 0.0, 0.45],
            [0.45, 0.0, 0.45, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.0, 0.6, 1.2, 2.4], dtype=np.float64)
    theta0 = np.array([0.0, 0.8, 2.0, 4.2], dtype=np.float64)
    return K_nm, omega, theta0


def test_triadic_ring_terms_are_anchored_and_periodic() -> None:
    hyperedges, weights = build_triadic_ring_terms(5, weight=0.12)

    assert hyperedges.shape == (5, 3)
    assert tuple(hyperedges[0]) == (0, 4, 1)
    assert tuple(hyperedges[-1]) == (4, 3, 0)
    np.testing.assert_allclose(weights, np.full(5, 0.12))
    with pytest.raises(ValueError, match="at least 3"):
        build_triadic_ring_terms(2, weight=0.1)


def test_higher_order_numpy_and_rust_paths_match_when_rust_is_available() -> None:
    K_nm, omega, theta0 = _ring_problem()
    hyperedges, weights = build_triadic_ring_terms(4, weight=0.25)
    spec = HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights, theta0=theta0)

    numpy_result = simulate_higher_order_kuramoto(spec, dt=0.03, n_steps=24, prefer_rust=False)
    preferred_result = simulate_higher_order_kuramoto(spec, dt=0.03, n_steps=24)

    assert preferred_result.variant is KuramotoVariant.HIGHER_ORDER
    assert preferred_result.backend in {
        "rust:higher_order_kuramoto_trajectory",
        "numpy:higher_order_kuramoto_trajectory",
    }
    np.testing.assert_allclose(preferred_result.r_values, numpy_result.r_values, atol=1e-12)
    assert preferred_result.diagnostics["n_hyperedges"] == 4


def test_higher_order_terms_change_the_pairwise_trajectory() -> None:
    K_nm, omega, theta0 = _ring_problem()
    hyperedges, weights = build_triadic_ring_terms(4, weight=0.35)
    with_terms = simulate_higher_order_kuramoto(
        HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights, theta0=theta0),
        dt=0.04,
        n_steps=20,
        prefer_rust=False,
    )
    without_terms = simulate_higher_order_kuramoto(
        HigherOrderKuramotoSpec(K_nm, omega, hyperedges, np.zeros_like(weights), theta0=theta0),
        dt=0.04,
        n_steps=20,
        prefer_rust=False,
    )

    assert abs(with_terms.final_r - without_terms.final_r) > 1e-4
    assert np.all((with_terms.r_values >= 0.0) & (with_terms.r_values <= 1.0 + 1e-12))


def test_monitored_feedback_records_readout_and_feedback_channels() -> None:
    K_nm, omega, theta0 = _ring_problem()
    spec = MonitoredKuramotoSpec(
        K_nm,
        omega,
        target_r=0.85,
        monitor_gain=1.1,
        measurement_strength=0.25,
        theta0=theta0,
    )

    result = simulate_monitored_kuramoto(spec, dt=0.02, n_steps=30)

    readout = result.diagnostics["readout_r"]
    feedback = result.diagnostics["feedback"]
    assert isinstance(readout, np.ndarray)
    assert isinstance(feedback, np.ndarray)
    assert readout.shape == result.times.shape
    assert feedback.shape == result.times.shape
    np.testing.assert_allclose(
        readout,
        0.75 * result.r_values + 0.25 * spec.target_r,
        atol=1e-12,
    )
    assert float(feedback[0]) == pytest.approx(spec.monitor_gain * (spec.target_r - readout[0]))
    assert result.to_metadata()["diagnostics"]["target_r"] == spec.target_r


def test_monitored_feedback_with_zero_gain_matches_uncontrolled_readout() -> None:
    K_nm, omega, theta0 = _ring_problem()
    spec = MonitoredKuramotoSpec(
        K_nm,
        omega,
        target_r=0.9,
        monitor_gain=0.0,
        measurement_strength=0.5,
        theta0=theta0,
    )

    result = simulate_monitored_kuramoto(spec, dt=0.03, n_steps=12, prefer_rust=False)

    feedback = result.diagnostics["feedback"]
    assert isinstance(feedback, np.ndarray)
    np.testing.assert_allclose(feedback, 0.0)
    assert result.backend == "numpy:monitored_kuramoto_trajectory"


def test_pt_symmetric_balanced_gain_loss_tracks_norm_and_imbalance() -> None:
    K_nm, omega, theta0 = _ring_problem()
    spec = PTSymmetricKuramotoSpec(
        K_nm,
        omega,
        gain_loss=np.array([0.08, -0.08, 0.04, -0.04], dtype=np.float64),
        theta0=theta0,
    )

    result = simulate_pt_symmetric_kuramoto(spec, dt=0.02, n_steps=25)

    pt_norm = result.diagnostics["pt_norm"]
    imbalance = result.diagnostics["gain_loss_imbalance"]
    assert isinstance(pt_norm, np.ndarray)
    assert isinstance(imbalance, np.ndarray)
    assert result.variant is KuramotoVariant.PT_SYMMETRIC
    assert result.backend in {
        "rust:pt_symmetric_kuramoto_trajectory",
        "numpy:pt_symmetric_kuramoto_trajectory",
    }
    np.testing.assert_allclose(pt_norm, 1.0, atol=5e-4)
    assert np.max(np.abs(imbalance)) > 0.0


def test_pt_symmetric_rejects_unbalanced_gain_loss() -> None:
    K_nm, omega, theta0 = _ring_problem()

    with pytest.raises(ValueError, match="sum to zero"):
        PTSymmetricKuramotoSpec(
            K_nm,
            omega,
            gain_loss=np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float64),
            theta0=theta0,
        )


def test_stable_facade_dispatches_all_three_variants() -> None:
    K_nm, omega, theta0 = _ring_problem()
    problem = build_kuramoto_problem(K_nm, omega, metadata={"case": "variant-dispatch"})
    hyperedges, weights = build_triadic_ring_terms(4, weight=0.1)

    higher = simulate_variant_trajectory(
        problem,
        "higher_order",
        dt=0.03,
        n_steps=8,
        theta0=theta0,
        hyperedges=hyperedges,
        hyper_weights=weights,
        prefer_rust=False,
    )
    monitored = simulate_variant_trajectory(
        problem,
        "monitored",
        dt=0.03,
        n_steps=8,
        theta0=theta0,
        target_r=0.8,
        prefer_rust=False,
    )
    pt_result = simulate_variant_trajectory(
        problem,
        "pt_symmetric",
        dt=0.03,
        n_steps=8,
        theta0=theta0,
        gain_loss=np.array([0.05, -0.05, 0.02, -0.02], dtype=np.float64),
        prefer_rust=False,
    )

    assert higher.variant is KuramotoVariant.HIGHER_ORDER
    assert monitored.variant is KuramotoVariant.MONITORED
    assert pt_result.variant is KuramotoVariant.PT_SYMMETRIC
    with pytest.raises(ValueError, match="requires hyperedges"):
        simulate_variant_trajectory(problem, "higher_order", dt=0.03, n_steps=8)
    with pytest.raises(ValueError, match="variant must be one of"):
        simulate_variant_trajectory(problem, "unknown", dt=0.03, n_steps=8)


def test_specs_defensively_copy_inputs_and_expose_readonly_arrays() -> None:
    K_nm, omega, theta0 = _ring_problem()
    hyperedges, weights = build_triadic_ring_terms(4, weight=0.2)
    spec = HigherOrderKuramotoSpec(K_nm, omega, hyperedges, weights, theta0=theta0)
    K_nm[0, 1] = 9.0
    theta0[0] = 9.0

    assert spec.K_nm[0, 1] == 0.45
    assert spec.theta0[0] == 0.0
    with pytest.raises(ValueError):
        spec.K_nm[0, 1] = 0.2
    with pytest.raises(ValueError):
        spec.theta0[0] = 0.2


def test_invalid_shapes_are_rejected_before_simulation() -> None:
    K_nm, omega, theta0 = _ring_problem()

    with pytest.raises(ValueError, match="hyperedges must have shape"):
        HigherOrderKuramotoSpec(
            K_nm,
            omega,
            np.array([[0, 1]], dtype=np.int64),
            np.array([0.1], dtype=np.float64),
            theta0=theta0,
        )
    with pytest.raises(ValueError, match="target_r"):
        MonitoredKuramotoSpec(K_nm, omega, target_r=1.5, theta0=theta0)
    with pytest.raises(ValueError, match="gain_loss must have shape"):
        PTSymmetricKuramotoSpec(
            K_nm,
            omega,
            gain_loss=np.array([0.1, -0.1], dtype=np.float64),
            theta0=theta0,
        )
