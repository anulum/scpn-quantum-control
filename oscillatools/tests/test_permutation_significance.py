# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — tests for the surrogate / permutation significance test
r"""Contract tests for the surrogate / permutation significance test.

The load-bearing claims are statistical: a genuinely phase-locked series is flagged significant with the
smallest attainable p-value while its surrogate null collapses to the incoherent floor; a series of
independent oscillators is *not* flagged, because the circular-shift surrogate reproduces its coherence;
the ``less`` and ``two-sided`` tails behave; the exact ``(1 + k) / (n + 1)`` estimator is honoured (never
zero); the test is deterministic under a fixed seed; and every validation contract holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.order_parameter_observables import order_parameter
from oscillatools.accel.permutation_significance import (
    PermutationSignificanceResult,
    permutation_significance_test,
)

_T = 160
_N = 6
_PERMUTATIONS = 99
_TIME = np.arange(_T) * 0.05


def _mean_order_parameter(trajectory: np.ndarray) -> float:
    """The time-averaged Kuramoto order parameter of a ``(T, N)`` phase series."""
    return float(np.mean([order_parameter(trajectory[t]) for t in range(trajectory.shape[0])]))


def _incoherence(trajectory: np.ndarray) -> float:
    """One minus the time-averaged order parameter — low for a phase-locked series."""
    return 1.0 - _mean_order_parameter(trajectory)


@pytest.fixture(scope="module")
def synchronised() -> np.ndarray:
    """A tightly phase-locked series: one common frequency, a narrow phase cluster."""
    rng = np.random.default_rng(1)
    offsets = rng.normal(0.0, 0.2, _N)
    return (_TIME[:, None] + offsets[None, :]) % (2.0 * np.pi)


@pytest.fixture(scope="module")
def independent() -> np.ndarray:
    """A series of unrelated oscillators: spread frequencies and independent phase offsets."""
    rng = np.random.default_rng(2)
    frequencies = rng.uniform(-1.5, 1.5, _N)
    offsets = rng.uniform(0.0, 2.0 * np.pi, _N)
    drift = rng.normal(0.0, 0.3, (_T, _N))
    return (frequencies[None, :] * _TIME[:, None] + offsets[None, :] + drift) % (2.0 * np.pi)


def test_phase_locked_series_is_significant(synchronised: np.ndarray) -> None:
    """A phase-locked series beats every circular-shift surrogate — the smallest p-value."""
    result = permutation_significance_test(
        synchronised, _mean_order_parameter, n_permutations=_PERMUTATIONS, seed=3
    )
    assert result.p_value == pytest.approx(1.0 / (_PERMUTATIONS + 1))
    assert result.observed > result.null_mean
    assert result.z_score > 0.0


def test_independent_series_is_not_significant(independent: np.ndarray) -> None:
    """Unrelated oscillators are not flagged: the surrogate null reproduces their coherence."""
    result = permutation_significance_test(
        independent, _mean_order_parameter, n_permutations=_PERMUTATIONS, seed=3
    )
    assert result.p_value > 0.05
    assert abs(result.z_score) < 2.0


def test_phase_shuffle_surrogate_runs(synchronised: np.ndarray) -> None:
    """The phase-shuffle surrogate is a valid alternative null."""
    result = permutation_significance_test(
        synchronised,
        _mean_order_parameter,
        n_permutations=_PERMUTATIONS,
        surrogate="phase_shuffle",
        seed=3,
    )
    assert result.surrogate == "phase_shuffle"
    assert result.p_value == pytest.approx(1.0 / (_PERMUTATIONS + 1))


def test_less_alternative_flags_a_low_statistic(synchronised: np.ndarray) -> None:
    """With the incoherence statistic the phase-locked series is unusually *low* — the left tail."""
    result = permutation_significance_test(
        synchronised,
        _incoherence,
        n_permutations=_PERMUTATIONS,
        alternative="less",
        seed=3,
    )
    assert result.p_value == pytest.approx(1.0 / (_PERMUTATIONS + 1))
    assert result.observed < result.null_mean


def test_two_sided_alternative_flags_an_extreme_statistic(synchronised: np.ndarray) -> None:
    """The two-sided tail flags the phase-locked series, which sits far from the null centre."""
    result = permutation_significance_test(
        synchronised,
        _mean_order_parameter,
        n_permutations=_PERMUTATIONS,
        alternative="two-sided",
        seed=3,
    )
    assert result.alternative == "two-sided"
    assert result.p_value == pytest.approx(1.0 / (_PERMUTATIONS + 1))


def test_p_value_respects_the_permutation_bounds(independent: np.ndarray) -> None:
    """The estimator stays in ``[1/(n+1), 1]`` and never reports zero."""
    result = permutation_significance_test(
        independent, _mean_order_parameter, n_permutations=_PERMUTATIONS, seed=7
    )
    assert result.p_value >= 1.0 / (_PERMUTATIONS + 1)
    assert result.p_value <= 1.0


def test_constant_statistic_has_zero_spread_and_z(synchronised: np.ndarray) -> None:
    """A statistic that ignores its input yields a degenerate null: no spread, zero z-score."""
    result = permutation_significance_test(
        synchronised, lambda _trajectory: 2.5, n_permutations=_PERMUTATIONS, seed=3
    )
    assert result.null_std == 0.0
    assert result.z_score == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_is_deterministic(synchronised: np.ndarray) -> None:
    """A fixed seed reproduces the p-value and effect size exactly."""
    first = permutation_significance_test(
        synchronised, _mean_order_parameter, n_permutations=_PERMUTATIONS, seed=5
    )
    second = permutation_significance_test(
        synchronised, _mean_order_parameter, n_permutations=_PERMUTATIONS, seed=5
    )
    assert first.p_value == second.p_value
    assert first.z_score == second.z_score
    assert first.null_mean == second.null_mean


def test_result_echoes_the_settings(synchronised: np.ndarray) -> None:
    """The record reports the surrogate, alternative and permutation count used."""
    result = permutation_significance_test(
        synchronised, _mean_order_parameter, n_permutations=50, surrogate="circular_shift", seed=3
    )
    assert isinstance(result, PermutationSignificanceResult)
    assert result.n_permutations == 50
    assert result.surrogate == "circular_shift"
    assert result.alternative == "greater"


@pytest.mark.parametrize(
    "override",
    [
        {"phases": np.zeros(_N)},
        {"phases": np.zeros((1, _N))},
        {"phases": np.zeros((_T, 0))},
        {"phases": np.full((_T, _N), np.nan)},
        {"n_permutations": 0},
        {"surrogate": "bootstrap"},
        {"alternative": "sideways"},
    ],
)
def test_rejects_out_of_bound_arguments(
    synchronised: np.ndarray, override: dict[str, object]
) -> None:
    """Every phase-series, permutation-count, surrogate and alternative bound is enforced."""
    kwargs: dict[str, object] = {
        "phases": synchronised,
        "n_permutations": _PERMUTATIONS,
        "seed": 3,
        **override,
    }
    phases = kwargs.pop("phases")
    with pytest.raises(ValueError):
        permutation_significance_test(phases, _mean_order_parameter, **kwargs)
