# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — identifiability, uncertainty and held-out evaluation of network inference
"""The inverse problem reports what the data cannot decide and is scored on data it never saw.

Oracles: synthetic ground truth from the generating stochastic model, an
ordinary least-squares fit of the same linear observation model, repeated
independent noise realisations for interval coverage, and duplicated
observations whose unidentified direction is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_bayesian_inference import (
    evaluate_held_out,
    fit_with_held_out,
    infer_network_bayesian,
)

_DT = 0.01
_NOISE = 0.04
_COUPLING = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.5, 0.0, 0.0]])
_OMEGA = np.array([0.3, -0.2, 0.1])


def _generate(
    n_steps: int,
    seed: int,
    coupling: NDArray[np.float64] = _COUPLING,
    omega: NDArray[np.float64] = _OMEGA,
) -> NDArray[np.float64]:
    """Simulate the generating Euler-Maruyama model ``dφ = (ω + Σ K sin(φ_j - φ_i)) dt + noise``."""
    rng = np.random.default_rng(seed)
    count = omega.size
    phases = np.empty((n_steps + 1, count), dtype=np.float64)
    phases[0] = rng.uniform(0.0, 2.0 * np.pi, size=count)
    scale = np.sqrt(_NOISE * _DT)
    for step in range(n_steps):
        difference = phases[step][None, :] - phases[step][:, None]
        drift = omega + np.sum(coupling * np.sin(difference), axis=1)
        phases[step + 1] = phases[step] + _DT * drift + scale * rng.standard_normal(count)
    return phases


def _truth(node: int) -> NDArray[np.float64]:
    """Return the true ``[ω_i, K_ij (j ≠ i)]`` of one node."""
    others = [other for other in range(_OMEGA.size) if other != node]
    return np.concatenate(([_OMEGA[node]], _COUPLING[node, others]))


def test_inverse_problem_identifiability_01_identifiable_recovery_with_intervals() -> None:
    """Ground truth lies inside the intervals, and the estimate equals a classical least-squares fit."""
    phases = _generate(40000, seed=1)
    posterior = infer_network_bayesian(phases, _DT)
    assert posterior.identifiable and posterior.data_rank.tolist() == [3, 3, 3]
    assert all(basis.shape == (3, 0) for basis in posterior.unidentified_directions)
    for node in range(3):
        others = [other for other in range(3) if other != node]
        estimate = np.concatenate(
            ([posterior.frequencies[node]], posterior.coupling[node, others])
        )
        spread = np.sqrt(np.diag(posterior.parameter_covariances[node]))
        assert np.all(np.abs(estimate - _truth(node)) <= 4.0 * spread)
        design = np.column_stack(
            [np.ones(phases.shape[0] - 1)]
            + [np.sin(phases[:-1, other] - phases[:-1, node]) for other in others]
        )
        increments = np.diff(phases[:, node])
        classical = np.linalg.lstsq(design * _DT, increments, rcond=None)[0]
        np.testing.assert_allclose(estimate, classical, rtol=1e-6, atol=1e-8)


def test_inverse_problem_identifiability_02_duplicate_columns_are_reported() -> None:
    """A duplicated observation leaves one direction undecided; it is reported, not resolved.

    Node 2 is recorded as an exact copy of node 1. For node 0 the design
    columns ``sin(φ_1 - φ_0)`` and ``sin(φ_2 - φ_0)`` coincide, so only
    ``K_01 + K_02`` is learned and ``(0, 1, -1)/√2`` is unidentified.
    """
    phases = _generate(20000, seed=2)
    phases[:, 2] = phases[:, 1]
    posterior = infer_network_bayesian(phases, _DT)
    assert not posterior.identifiable
    assert posterior.data_rank[0] == 2
    direction = posterior.unidentified_directions[0][:, 0]
    expected = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
    assert abs(abs(float(direction @ expected)) - 1.0) < 1e-8
    covariance = posterior.parameter_covariances[0]
    assert float(direction @ covariance @ direction) > 1e5
    total = np.array([0.0, 1.0, 1.0])
    assert float(total @ covariance @ total) < 1e-2
    # Nodes 1 and 2 see sin(φ_2 - φ_1) = 0: their mutual coupling is not observed either.
    assert posterior.data_rank[1] == 2 and posterior.data_rank[2] == 2


def test_inverse_problem_identifiability_03_declared_uncertainty_matches_repeated_noise() -> None:
    """Across independent noise realisations, 95 % intervals cover the truth about 95 % of the time."""
    hits = 0
    checks = 0
    estimates: list[NDArray[np.float64]] = []
    declared: list[NDArray[np.float64]] = []
    for seed in range(40):
        posterior = infer_network_bayesian(_generate(4000, seed=100 + seed), _DT)
        for node in range(3):
            others = [other for other in range(3) if other != node]
            estimate = np.concatenate(
                ([posterior.frequencies[node]], posterior.coupling[node, others])
            )
            spread = np.sqrt(np.diag(posterior.parameter_covariances[node]))
            hits += int(np.sum(np.abs(estimate - _truth(node)) <= 1.959963984540054 * spread))
            checks += estimate.size
            if node == 0:
                estimates.append(estimate)
                declared.append(spread)
    coverage = hits / checks
    assert 0.90 <= coverage <= 0.99, coverage
    ratio = np.std(np.array(estimates), axis=0, ddof=1) / np.mean(np.array(declared), axis=0)
    assert np.all((ratio > 0.7) & (ratio < 1.4)), ratio


def test_inverse_problem_identifiability_04_held_out_segments_are_never_read_by_the_fit() -> None:
    """Changing the calibration and test segments leaves the posterior exactly unchanged."""
    phases = _generate(30000, seed=7)
    split = fit_with_held_out(phases, _DT)
    start, first, second, end = split.boundaries
    assert (start, end) == (0, 30000) and 0 < first < second < end
    tampered = phases.copy()
    tampered[first + 1 :] += np.random.default_rng(0).normal(
        0.0, 1.0, size=tampered[first + 1 :].shape
    )
    other = fit_with_held_out(tampered, _DT)
    np.testing.assert_array_equal(other.posterior.coupling, split.posterior.coupling)
    np.testing.assert_array_equal(other.posterior.noise, split.posterior.noise)
    # On the well-specified held-out segments the predictive intervals are calibrated.
    for evaluation in (split.calibration, split.test):
        assert 0.93 <= evaluation.coverage_95 <= 0.97
        assert 0.95 <= evaluation.standardized_rms <= 1.05
    assert split.test.samples == 3 * (end - second)
    # The tampered segments are scored as badly miscalibrated, not silently absorbed.
    assert other.test.coverage_95 < 0.5 and other.test.standardized_rms > 3.0
    # A changed coupling on the test segment lowers the predictive density.
    drifted = _generate(6000, seed=8, coupling=_COUPLING * 2.5)
    assert (
        evaluate_held_out(split.posterior, drifted, _DT).mean_log_predictive_density
        < split.test.mean_log_predictive_density
    )


def test_inverse_problem_refusals() -> None:
    """A noise-free series, bad fractions and a mismatched posterior are refused."""
    noiseless = np.cumsum(np.full((50, 2), 0.01), axis=0)
    with pytest.raises(ValueError, match="noise is zero"):
        infer_network_bayesian(noiseless, _DT)
    phases = _generate(300, seed=3)
    for fractions, message in (
        ((0.5, 0.5), "three positive"),
        ((0.6, 0.0, 0.4), "three positive"),
        ((0.6, 0.3, 0.3), "sum to one"),
        ((0.998, 0.001, 0.001), "at least one increment"),
    ):
        with pytest.raises(ValueError, match=message):
            fit_with_held_out(phases, _DT, fractions=fractions)
    posterior = infer_network_bayesian(phases, _DT)
    with pytest.raises(ValueError, match="same N nodes"):
        evaluate_held_out(
            posterior, _generate(300, seed=4, coupling=np.zeros((2, 2)), omega=np.zeros(2)), _DT
        )
