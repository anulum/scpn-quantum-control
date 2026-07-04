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
    TimeVaryingCouplingHistory,
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
    short = infer_network_bayesian(_generate(40000, 2, _COUPLING, _OMEGA), _DT)
    long = infer_network_bayesian(_generate(160000, 2, _COUPLING, _OMEGA), _DT)
    short_error = np.max(np.abs(short.coupling - _COUPLING))
    long_error = np.max(np.abs(long.coupling - _COUPLING))
    assert long_error < short_error
    # the posterior also contracts with more data
    assert np.mean(long.coupling_std) < np.mean(short.coupling_std)


def test_tracks_time_varying_coupling() -> None:
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
