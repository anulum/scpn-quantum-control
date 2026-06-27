# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Shot-noise uncertainty for synchronisation metrics
"""Shot-noise uncertainty quantification for synchronisation metrics.

The count-based synchronisation metrics in this package — the Z-basis
magnetisation synchronisation proxy in
:class:`~scpn_quantum_control.analysis.sync_order_parameter.SyncOrderParameter`
and the witnesses in :mod:`scpn_quantum_control.analysis.sync_witness` — return a
point estimate from a finite-shot measurement record. On hardware those records
carry shot noise, so a bare point estimate overstates what the data support.

This module attaches defensible error bars to those metrics two ways:

* :func:`order_parameter_shot_noise` — an analytic shot-noise (delta-method)
  standard error and normal interval for the Z-basis proxy, exact for the
  per-shot magnetisation estimator, and
* :func:`order_parameter_bootstrap` / :func:`metric_bootstrap` — a
  distribution-free bootstrap percentile interval for the Z-basis proxy or any
  count-to-scalar metric (witnesses included). The bootstrap stays valid where the
  delta-method linearisation does not: the proxy is ``|·|`` of a mean and is
  non-smooth at zero.

Both report a stated ``coverage`` level; the bootstrap interval's empirical coverage
is exercised against a known distribution in the tests, which is the certified-
coverage evidence behind the error bars.

The numerics are vectorised NumPy reductions plus a single C-level
:meth:`numpy.random.Generator.multinomial` draw per call — there is no Python
per-shot loop for the order parameter — so the NumPy path is the optimised path and
no separate compiled kernel is warranted, matching the majority of pure-NumPy
analysis modules in this package. (Mitigation-bias bounds for extrapolated
estimators, e.g. ZNE, build on this shot-noise layer and are tracked separately.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

DEFAULT_BOOTSTRAP_RESAMPLES = 2000


@dataclass(frozen=True)
class UncertaintyInterval:
    """A point estimate with a shot-noise error bar and a coverage interval.

    Parameters
    ----------
    point:
        The metric value computed from the observed counts.
    standard_error:
        Estimated standard error of ``point``.
    low, high:
        Lower and upper interval bounds at ``coverage`` (``low <= point <= high``).
    coverage:
        Target coverage probability of ``[low, high]`` (for example ``0.95``).
    method:
        ``"shot-noise-delta"`` or ``"bootstrap-percentile"``.
    n_shots:
        Total measurement shots backing the estimate.
    n_resamples:
        Bootstrap resamples used (``0`` for the analytic method).
    """

    point: float
    standard_error: float
    low: float
    high: float
    coverage: float
    method: str
    n_shots: int
    n_resamples: int

    @property
    def width(self) -> float:
        """Width ``high - low`` of the coverage interval."""
        return self.high - self.low


def _validate_coverage(coverage: float) -> None:
    """Raise :class:`ValueError` unless ``coverage`` lies in the open unit interval."""
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}")


def _validate_resamples(n_resamples: int) -> None:
    """Raise :class:`ValueError` unless ``n_resamples`` is a positive integer."""
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive integer, got {n_resamples}")


def _counts_arrays(counts: Mapping[str, int]) -> tuple[list[str], NDArray[np.int64], int]:
    """Return the bitstrings, their integer shot weights, and the total shots.

    Validates that every weight is a non-negative integer and that at least one
    shot is present; bitstring content is left to the caller (the order-parameter
    path enforces ``0``/``1`` content, a generic metric may not need to).
    """
    if not counts:
        raise ValueError("counts must contain at least one measured bitstring")
    bitstrings = list(counts)
    weights = np.empty(len(bitstrings), dtype=np.int64)
    for i, key in enumerate(bitstrings):
        shots = counts[key]
        if not isinstance(shots, (int, np.integer)) or shots < 0:
            raise ValueError("counts values must be non-negative integers")
        weights[i] = int(shots)
    n_shots = int(weights.sum())
    if n_shots <= 0:
        raise ValueError("counts must contain at least one shot")
    return bitstrings, weights, n_shots


def _magnetisations(bitstrings: list[str]) -> NDArray[np.float64]:
    """Return the per-bitstring magnetisation (``'0'`` -> ``+1``, ``'1'`` -> ``-1``).

    Matches the estimator in
    :class:`~scpn_quantum_control.analysis.sync_order_parameter.SyncOrderParameter`.
    """
    width = len(bitstrings[0])
    if width == 0:
        raise ValueError("bitstrings must contain at least one qubit")
    mags = np.empty(len(bitstrings), dtype=np.float64)
    for i, bits in enumerate(bitstrings):
        if len(bits) != width:
            raise ValueError("all bitstrings must share the same qubit count")
        if any(char not in "01" for char in bits):
            raise ValueError(f"bitstring {bits!r} must contain only '0' and '1'")
        ones = bits.count("1")
        # mean spin = ((width - ones) * (+1) + ones * (-1)) / width
        mags[i] = (width - 2 * ones) / width
    return mags


def order_parameter_estimate(counts: Mapping[str, int]) -> float:
    """Return the Z-basis proxy ``|mean per-shot magnetisation|`` in ``[0, 1]``.

    Identical in value to
    :class:`~scpn_quantum_control.analysis.sync_order_parameter.SyncOrderParameter`
    for the same counts. This is not the X/Y Kuramoto order parameter; it is
    provided so an interval and its point estimate share one code path.
    """
    bitstrings, weights, n_shots = _counts_arrays(counts)
    mags = _magnetisations(bitstrings)
    return abs(float(np.dot(mags, weights) / n_shots))


def _order_parameter_interval_from_mean(mean: float, half_width: float) -> tuple[float, float]:
    """Map a symmetric interval ``mean ± half_width`` through ``|·|``.

    The order parameter is ``|mean|``; the image of ``[mean-h, mean+h]`` under the
    absolute value collapses to ``[0, max|·|]`` when the mean interval straddles
    zero, and is the sorted absolute bounds otherwise.
    """
    lo_mean = mean - half_width
    hi_mean = mean + half_width
    if lo_mean <= 0.0 <= hi_mean:
        return 0.0, max(abs(lo_mean), abs(hi_mean))
    abs_bounds = (abs(lo_mean), abs(hi_mean))
    return min(abs_bounds), max(abs_bounds)


def order_parameter_shot_noise(
    counts: Mapping[str, int], *, coverage: float = 0.95
) -> UncertaintyInterval:
    """Analytic shot-noise interval for the Z-basis synchronisation proxy.

    The per-shot magnetisations are independent and identically distributed; the
    proxy is ``|m_bar|`` with ``m_bar`` their mean. The standard error of ``m_bar``
    is ``s / sqrt(S)`` with ``s`` the Bessel-corrected sample standard deviation and
    ``S`` the shot count, and the reported interval is the image under ``|.|`` of the
    normal interval for ``m_bar``.

    Parameters
    ----------
    counts:
        Measurement counts keyed by ``0``/``1`` bitstring.
    coverage:
        Target coverage in ``(0, 1)``.

    Returns
    -------
    UncertaintyInterval
        Method ``"shot-noise-delta"``.

    Raises
    ------
    ValueError
        If ``coverage`` is outside ``(0, 1)``, the counts are empty/malformed, or
        fewer than two shots are present (a variance needs at least two samples).
    """
    _validate_coverage(coverage)
    bitstrings, weights, n_shots = _counts_arrays(counts)
    if n_shots < 2:
        raise ValueError("analytic shot-noise SE requires at least 2 shots")
    mags = _magnetisations(bitstrings)
    mean = float(np.dot(mags, weights) / n_shots)
    pop_var = float(np.dot(weights, (mags - mean) ** 2) / n_shots)
    sample_var = pop_var * n_shots / (n_shots - 1)
    standard_error = float(np.sqrt(sample_var / n_shots))
    z = NormalDist().inv_cdf(0.5 + coverage / 2.0)
    low, high = _order_parameter_interval_from_mean(mean, z * standard_error)
    return UncertaintyInterval(
        point=abs(mean),
        standard_error=standard_error,
        low=low,
        high=high,
        coverage=coverage,
        method="shot-noise-delta",
        n_shots=n_shots,
        n_resamples=0,
    )


def order_parameter_bootstrap(
    counts: Mapping[str, int],
    *,
    coverage: float = 0.95,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> UncertaintyInterval:
    """Bootstrap percentile interval for the Z-basis synchronisation proxy.

    Resamples the shot record ``n_resamples`` times from the empirical bitstring
    distribution (one vectorised multinomial draw) and reports the percentile
    interval of ``|mean magnetisation|``. Distribution-free and valid near zero,
    where :func:`order_parameter_shot_noise` linearises a non-smooth ``|·|``.

    Parameters
    ----------
    counts:
        Measurement counts keyed by ``0``/``1`` bitstring.
    coverage:
        Target coverage in ``(0, 1)``.
    n_resamples:
        Number of bootstrap resamples (positive).
    seed:
        Seed for the bootstrap generator; fixed for reproducibility.

    Returns
    -------
    UncertaintyInterval
        Method ``"bootstrap-percentile"``.
    """
    _validate_coverage(coverage)
    _validate_resamples(n_resamples)
    bitstrings, weights, n_shots = _counts_arrays(counts)
    mags = _magnetisations(bitstrings)
    probabilities = weights / n_shots
    rng = np.random.default_rng(seed)
    resampled = rng.multinomial(n_shots, probabilities, size=n_resamples).astype(np.float64)
    distribution = np.abs(resampled @ mags / n_shots)
    alpha = (1.0 - coverage) / 2.0
    low = float(np.quantile(distribution, alpha))
    high = float(np.quantile(distribution, 1.0 - alpha))
    standard_error = float(np.std(distribution, ddof=1)) if n_resamples > 1 else 0.0
    point = abs(float(np.dot(mags, weights) / n_shots))
    return UncertaintyInterval(
        point=point,
        standard_error=standard_error,
        low=low,
        high=high,
        coverage=coverage,
        method="bootstrap-percentile",
        n_shots=n_shots,
        n_resamples=int(n_resamples),
    )


def metric_bootstrap(
    counts: Mapping[str, int],
    metric: Callable[[Mapping[str, int]], float],
    *,
    coverage: float = 0.95,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> UncertaintyInterval:
    """Bootstrap percentile interval for any count-to-scalar metric.

    Resamples the shot record and re-evaluates ``metric`` on each resample, giving a
    distribution-free interval for witnesses or any other function of measurement
    counts. ``metric`` is arbitrary Python, so this evaluates it once per resample;
    the order-parameter helpers above are fully vectorised and should be preferred
    for that metric.

    Parameters
    ----------
    counts:
        Measurement counts keyed by bitstring.
    metric:
        Callable mapping a counts mapping to a scalar.
    coverage:
        Target coverage in ``(0, 1)``.
    n_resamples:
        Number of bootstrap resamples (positive).
    seed:
        Seed for the bootstrap generator.

    Returns
    -------
    UncertaintyInterval
        Method ``"bootstrap-percentile"``; ``point`` is ``metric(counts)``.
    """
    _validate_coverage(coverage)
    _validate_resamples(n_resamples)
    bitstrings, weights, n_shots = _counts_arrays(counts)
    probabilities = weights / n_shots
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_shots, probabilities, size=n_resamples)
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        resampled_counts = {
            bits: int(count) for bits, count in zip(bitstrings, draws[i]) if count > 0
        }
        values[i] = float(metric(resampled_counts))
    alpha = (1.0 - coverage) / 2.0
    low = float(np.quantile(values, alpha))
    high = float(np.quantile(values, 1.0 - alpha))
    standard_error = float(np.std(values, ddof=1)) if n_resamples > 1 else 0.0
    return UncertaintyInterval(
        point=float(metric(dict(counts))),
        standard_error=standard_error,
        low=low,
        high=high,
        coverage=coverage,
        method="bootstrap-percentile",
        n_shots=n_shots,
        n_resamples=int(n_resamples),
    )
