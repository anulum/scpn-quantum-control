# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — surrogate / permutation significance test for a synchronisation statistic
r"""A surrogate / permutation significance test for coupled-phase-oscillator statistics.

A measured synchronisation statistic — an order parameter, a phase-locking value, a coherence, an
inferred coupling — is only meaningful against a null: how large would it be if the oscillators were
in fact *unrelated*? This module answers that with a nonparametric surrogate test. Given a phase time
series ``θ(t)`` of shape ``(T, N)`` and any statistic that reduces it to a scalar, it builds a null
distribution by repeatedly destroying the inter-oscillator relationship while preserving each
oscillator's own dynamics, recomputes the statistic on each surrogate, and reports where the observed
value falls in that distribution.

Two surrogate schemes are provided. **Circular time-shift** rolls each oscillator's phase series by an
independent random lag: it preserves every oscillator's marginal phase distribution and its
autocorrelation exactly, and destroys only the cross-oscillator alignment, so it is the appropriate
null for "is the synchronisation real, or could similar-but-independent oscillators have produced it?"
(the standard construction for phase-locking and coupling significance). **Phase shuffle** instead
permutes the time index of each oscillator independently, a stronger null that also destroys the
temporal structure.

The p-value uses the exact permutation estimator ``p = (1 + k) / (n + 1)`` with ``k`` the number of
surrogates at least as extreme as the observation and ``n`` the number of surrogates — the estimator
that stays valid for any number of permutations and never reports ``p = 0``. The test is model-free:
it makes no distributional assumption, reuses the toolkit's own observables as the statistic (each
carrying its own accelerated backends), and adds no compute kernel of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_SURROGATES = ("circular_shift", "phase_shuffle")
_ALTERNATIVES = ("greater", "less", "two-sided")


@dataclass(frozen=True)
class PermutationSignificanceResult:
    """The outcome of a surrogate / permutation significance test.

    Attributes
    ----------
    observed : float
        The statistic evaluated on the measured phase series.
    p_value : float
        The permutation p-value ``(1 + k) / (n + 1)`` for the chosen alternative — the probability, under
        the surrogate null, of a statistic at least as extreme as ``observed``.
    null_mean : float
        The mean of the statistic over the surrogate null distribution.
    null_std : float
        The standard deviation of the statistic over the surrogate null distribution.
    z_score : float
        ``(observed − null_mean) / null_std`` — the standardised effect size (``0`` when the null has no
        spread).
    n_permutations : int
        The number of surrogates drawn.
    surrogate : str
        The surrogate scheme used (``"circular_shift"`` or ``"phase_shuffle"``).
    alternative : str
        The tested alternative (``"greater"``, ``"less"`` or ``"two-sided"``).
    """

    observed: float
    p_value: float
    null_mean: float
    null_std: float
    z_score: float
    n_permutations: int
    surrogate: str
    alternative: str


def _circular_shift(phases: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Roll each oscillator's phase series by an independent random lag (marginals + autocorr kept)."""
    length, count = phases.shape
    lags = rng.integers(0, length, size=count)
    columns = [np.roll(phases[:, index], int(lags[index])) for index in range(count)]
    return np.stack(columns, axis=1)


def _phase_shuffle(phases: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Permute the time index of each oscillator independently (destroys temporal structure too)."""
    length, count = phases.shape
    surrogate = np.empty_like(phases)
    for index in range(count):
        surrogate[:, index] = phases[rng.permutation(length), index]
    return surrogate


_SURROGATE_FUNCTIONS = {"circular_shift": _circular_shift, "phase_shuffle": _phase_shuffle}


def _p_value(observed: float, null: NDArray[np.float64], alternative: str) -> float:
    """The exact permutation p-value ``(1 + k) / (n + 1)`` for the chosen tail."""
    total = null.size
    if alternative == "greater":
        extreme = int(np.count_nonzero(null >= observed))
    elif alternative == "less":
        extreme = int(np.count_nonzero(null <= observed))
    else:
        centre = float(np.mean(null))
        extreme = int(np.count_nonzero(np.abs(null - centre) >= abs(observed - centre)))
    return (1.0 + extreme) / (total + 1.0)


def permutation_significance_test(
    phases: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    n_permutations: int = 999,
    surrogate: str = "circular_shift",
    alternative: str = "greater",
    seed: int = 0,
) -> PermutationSignificanceResult:
    r"""Test whether a synchronisation statistic of a phase series is significant against a surrogate null.

    Evaluates ``statistic`` on the measured phase series, builds a null distribution by drawing
    ``n_permutations`` surrogates that destroy the inter-oscillator relationship while preserving each
    oscillator's own dynamics, recomputes the statistic on each, and returns the permutation p-value and
    the standardised effect size.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase time series of shape ``(T, N)`` — ``T ≥ 2`` samples of ``N ≥ 1`` oscillators.
    statistic : callable
        A reduction ``(T, N) → float`` of the phase series; the toolkit's observables (an order
        parameter, a phase-locking or coherence measure, an inferred coupling strength) plug in directly.
    n_permutations : int, optional
        The number of surrogates (``≥ 1``, default ``999``); the smallest attainable p-value is
        ``1 / (n_permutations + 1)``.
    surrogate : str, optional
        ``"circular_shift"`` (default; independent random circular time-shift per oscillator, preserving
        marginals and autocorrelation) or ``"phase_shuffle"`` (independent time-index permutation per
        oscillator).
    alternative : str, optional
        ``"greater"`` (default; the observed statistic is unusually large), ``"less"`` or
        ``"two-sided"``.
    seed : int, optional
        The seed for the deterministic surrogate draws (default ``0``).

    Returns
    -------
    PermutationSignificanceResult
        The observed statistic, the p-value, the null mean/spread, the z-score, and the test settings.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    series = np.ascontiguousarray(phases, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] < 2 or series.shape[1] < 1:
        raise ValueError(
            f"phases must be a (T, N) array with T >= 2 and N >= 1, got shape {series.shape}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("phases must be finite")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}")
    if surrogate not in _SURROGATES:
        raise ValueError(f"surrogate must be one of {_SURROGATES}, got {surrogate!r}")
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")

    draw = _SURROGATE_FUNCTIONS[surrogate]
    rng = np.random.default_rng(seed)
    observed = float(statistic(series))
    null = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        null[index] = float(statistic(draw(series, rng)))

    null_mean = float(np.mean(null))
    null_std = float(np.std(null))
    z_score = 0.0 if null_std == 0.0 else (observed - null_mean) / null_std
    return PermutationSignificanceResult(
        observed=observed,
        p_value=_p_value(observed, null, alternative),
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        n_permutations=int(n_permutations),
        surrogate=surrogate,
        alternative=alternative,
    )


__all__ = [
    "PermutationSignificanceResult",
    "permutation_significance_test",
]
