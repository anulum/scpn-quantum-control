# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for dynamical Bayesian inference and network reconstruction
"""Module-specific tests for :mod:`kuramoto_bayesian_inference`.

The contracts: from a noisy phase series the inference reconstructs the *directed* coupling matrix
and the frequencies near their true values, with the true values inside a few posterior standard
deviations and the asymmetry (directionality) recovered; the estimator is consistent (the error
shrinks as the record lengthens); Bayesian window propagation tracks a coupling that changes in time;
and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_bayesian_inference import (
    DynamicalBayesianPosterior,
    HeldOutEvaluation,
    InverseProblemSplit,
    TimeVaryingCouplingHistory,
    evaluate_held_out,
    fit_with_held_out,
    infer_network_bayesian,
    track_time_varying_coupling,
)

_DT = 0.01
_NOISE = 0.04
# a directed (asymmetric) ring-like coupling: 1->0, 2->1, 3->2, 0->3
_COUPLING = np.array(
    [[0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 0.9], [0.5, 0.0, 0.0, 0.0]]
)
_OMEGA = np.array([0.3, -0.2, 0.1, -0.1])


def _generate(
    n_steps: int,
    seed: int,
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate the generating Euler-Maruyama network model with the module noise."""
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


def test_reconstructs_directed_coupling_and_frequencies() -> None:
    """Directed coupling and frequencies are recovered within a few posterior deviations."""
    posterior = infer_network_bayesian(_generate(40000, 1, _COUPLING, _OMEGA), _DT)
    assert isinstance(posterior, DynamicalBayesianPosterior)
    assert np.max(np.abs(posterior.coupling - _COUPLING)) < 0.15
    assert np.max(np.abs(posterior.frequencies - _OMEGA)) < 0.1
    # directionality: the 1->0 link is strong, the reverse 0->1 link is ~absent
    assert posterior.coupling[0, 1] > 0.5
    assert abs(posterior.coupling[1, 0]) < 0.2
    # the true coupling lies within a few posterior standard deviations everywhere
    off_diagonal = ~np.eye(4, dtype=bool)
    standardised = np.abs(posterior.coupling - _COUPLING)[off_diagonal] / (
        posterior.coupling_std[off_diagonal] + 1e-12
    )
    assert np.max(standardised) < 4.0
    assert np.all(posterior.coupling_std[off_diagonal] > 0.0)
    assert np.all(posterior.noise > 0.0)


def test_estimator_is_consistent() -> None:
    """The error and the posterior spread shrink as the record lengthens."""
    short = infer_network_bayesian(_generate(40000, 2, _COUPLING, _OMEGA), _DT)
    long = infer_network_bayesian(_generate(160000, 2, _COUPLING, _OMEGA), _DT)
    short_error = np.max(np.abs(short.coupling - _COUPLING))
    long_error = np.max(np.abs(long.coupling - _COUPLING))
    assert long_error < short_error
    # the posterior also contracts with more data
    assert np.mean(long.coupling_std) < np.mean(short.coupling_std)


def test_tracks_time_varying_coupling() -> None:
    """Window propagation follows a coupling that weakens midway."""
    n_steps = 80000
    rng = np.random.default_rng(3)
    count = _OMEGA.size
    phases = np.empty((n_steps + 1, count), dtype=np.float64)
    phases[0] = rng.uniform(0.0, 2.0 * np.pi, size=count)
    scale = np.sqrt(_NOISE * _DT)
    for step in range(n_steps):
        coupling = _COUPLING.copy()
        coupling[0, 1] = 0.8 if step < n_steps // 2 else 0.2  # the 1->0 link weakens midway
        difference = phases[step][None, :] - phases[step][:, None]
        drift = _OMEGA + np.sum(coupling * np.sin(difference), axis=1)
        phases[step + 1] = phases[step] + _DT * drift + scale * rng.standard_normal(count)

    history = track_time_varying_coupling(phases, _DT, 20000, propagation_inflation=50.0)
    assert isinstance(history, TimeVaryingCouplingHistory)
    assert history.coupling.shape == (4, count, count)
    early = history.coupling[0, 0, 1]
    late = history.coupling[-1, 0, 1]
    assert early > 0.55  # the strong link in the first window
    assert late < 0.45  # the weakened link in the last window
    assert history.window_times[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("infer", {"phases": np.zeros((1, 4))}, "phases must be a"),
        ("infer", {"phases": np.zeros((10, 1))}, "phases must be a"),
        ("infer", {"dt": 0.0}, "dt must be positive"),
        ("infer", {"prior_precision": 0.0}, "prior_precision must be positive"),
        ("infer", {"noise_iterations": 0}, "noise_iterations must be positive"),
        ("track", {"window_size": 0}, r"window_size must be in"),
        ("track", {"window_size": 9999}, r"window_size must be in"),
        ("track", {"propagation_inflation": 0.0}, "propagation_inflation must be positive"),
        ("track", {"prior_precision": 0.0}, "prior_precision must be positive"),
        ("track", {"noise_iterations": 0}, "noise_iterations must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    """Each argument outside its documented bound is refused.

    Parameters
    ----------
    call
        ``infer`` or ``track``.
    kwargs
        Arguments replaced in an otherwise valid call.
    message
        Expected error text.

    """
    series = _generate(200, 0, _COUPLING, _OMEGA)
    with pytest.raises(ValueError, match=message):
        if call == "infer":
            args: dict[str, Any] = {
                "phases": series,
                "dt": _DT,
                "prior_precision": 1e-6,
                "noise_iterations": 4,
            }
            args.update(kwargs)
            infer_network_bayesian(
                args["phases"],
                args["dt"],
                prior_precision=args["prior_precision"],
                noise_iterations=args["noise_iterations"],
            )
        else:
            args = {
                "window_size": 50,
                "propagation_inflation": 10.0,
                "prior_precision": 1e-6,
                "noise_iterations": 4,
            }
            args.update(kwargs)
            track_time_varying_coupling(
                series,
                _DT,
                args["window_size"],
                propagation_inflation=args["propagation_inflation"],
                prior_precision=args["prior_precision"],
                noise_iterations=args["noise_iterations"],
            )


def test_identifiability_is_reported_for_a_duplicated_observation() -> None:
    """A node recorded twice gives a rank-deficient design that is reported, not patched."""
    phases = _generate(5000, 4, _COUPLING, _OMEGA)
    full = infer_network_bayesian(phases, _DT)
    assert full.identifiable and full.data_rank.tolist() == [4, 4, 4, 4]
    assert len(full.parameter_covariances) == 4
    phases[:, 3] = phases[:, 2]
    duplicated = infer_network_bayesian(phases, _DT)
    assert not duplicated.identifiable
    assert duplicated.data_rank[0] == 3 and duplicated.unidentified_directions[0].shape == (4, 1)


def test_noise_free_series_is_refused() -> None:
    """A series without noise has a degenerate transition likelihood."""
    with pytest.raises(ValueError, match="noise is zero to rounding"):
        infer_network_bayesian(np.cumsum(np.full((40, 2), 0.02), axis=0), _DT)


def test_held_out_split_and_evaluation() -> None:
    """The split fits on the first segment and scores the later ones."""
    phases = _generate(12000, 5, _COUPLING, _OMEGA)
    split = fit_with_held_out(phases, _DT, fractions=[0.5, 0.25, 0.25])
    assert isinstance(split, InverseProblemSplit)
    assert split.boundaries == (0, 6000, 9000, 12000)
    assert isinstance(split.test, HeldOutEvaluation) and split.test.samples == 4 * 3000
    assert 0.9 <= split.calibration.coverage_95 <= 0.99
    direct = evaluate_held_out(split.posterior, phases[9000:], _DT)
    assert direct == split.test
    with pytest.raises(ValueError, match="sum to one"):
        fit_with_held_out(phases, _DT, fractions=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="three positive"):
        fit_with_held_out(phases, _DT, fractions=(1.0,))
    with pytest.raises(ValueError, match="at least one increment"):
        fit_with_held_out(phases[:40], _DT, fractions=(0.98, 0.01, 0.01))
    with pytest.raises(ValueError, match="same N nodes"):
        evaluate_held_out(split.posterior, phases[:, :3], _DT)
