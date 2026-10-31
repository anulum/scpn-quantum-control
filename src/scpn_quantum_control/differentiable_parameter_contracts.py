# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable parameter contracts module
# scpn-quantum-control -- differentiable parameter contract records
"""Parameter metadata and validation contracts for differentiable objectives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    """Return a real numeric array without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric scalars")
    array = np.asarray(raw, dtype=np.float64)
    return array


def _as_real_scalar(name: str, value: object) -> float:
    """Return an explicit finite real numeric scalar without implicit coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_index_vector(name: str, values: object) -> NDArray[np.int64]:
    """Return a one-dimensional non-negative integer index vector."""
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer indices")
    array = np.asarray(raw, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative indices")
    return array


def _as_parameter_array(values: ArrayLike) -> NDArray[np.float64]:
    """Return a one-dimensional finite differentiable parameter vector."""
    array = _as_real_numeric_array("parameters", values)
    if array.ndim != 1:
        raise ValueError("parameters must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("parameters must contain only finite values")
    return array


@dataclass(frozen=True)
class Parameter:
    """One differentiable scalar parameter in an SCPN objective."""

    name: str
    trainable: bool = True

    def __post_init__(self) -> None:
        """Validate parameter identity and trainability metadata."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be non-empty")
        if not isinstance(self.trainable, bool):
            raise ValueError("parameter trainable flag must be a boolean")

    def to_semantic_source(self) -> dict[str, object]:
        """Project native derivative-request identity and trainability.

        Returns
        -------
        dict[str, object]
            Name and mask bit without inventing a parameter value or unit.

        """
        return {
            "producer_identity": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "trainable": self.trainable,
        }


@dataclass(frozen=True)
class ParameterBounds:
    """Closed interval constraint for one differentiable scalar parameter."""

    lower: float | None = None
    upper: float | None = None
    periodic: bool = False

    def __post_init__(self) -> None:
        """Validate finite interval and periodic-bound metadata."""
        if not isinstance(self.periodic, bool):
            raise ValueError("periodic flag must be a boolean")
        lower = None if self.lower is None else _as_real_scalar("lower bound", self.lower)
        upper = None if self.upper is None else _as_real_scalar("upper bound", self.upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower bound must be less than or equal to upper bound")
        if self.periodic:
            if lower is None or upper is None:
                raise ValueError("periodic bounds require finite lower and upper bounds")
            if lower == upper:
                raise ValueError("periodic bounds require lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class ParameterShiftRule:
    """Symmetric parameter-shift rule for one- or multi-frequency generators."""

    shift: float = float(np.pi / 2.0)
    coefficient: float = 0.5
    shifts: Sequence[float] | None = None
    coefficients: Sequence[float] | None = None
    frequencies: Sequence[float] | None = None

    def __post_init__(self) -> None:
        """Validate and freeze explicit parameter-shift terms."""
        shifts: tuple[float, ...]
        coefficients: tuple[float, ...]
        if (self.shifts is None) != (self.coefficients is None):
            raise ValueError("shifts and coefficients must be provided together")
        if self.shifts is None:
            shift = _as_real_scalar("shift", self.shift)
            coefficient = _as_real_scalar("coefficient", self.coefficient)
            if shift <= 0.0:
                raise ValueError("shift must be finite and positive")
            shifts = (shift,)
            coefficients = (coefficient,)
        else:
            raw_shifts = _as_parameter_array(self.shifts)
            raw_coefficients = _as_parameter_array(cast(Sequence[float], self.coefficients))
            if raw_shifts.shape != raw_coefficients.shape:
                raise ValueError("shifts and coefficients must have matching shapes")
            if raw_shifts.size == 0:
                raise ValueError("parameter-shift rule must contain at least one term")
            if np.any(raw_shifts <= 0.0):
                raise ValueError("shifts must contain finite positive values")
            shifts = tuple(float(value) for value in raw_shifts)
            coefficients = tuple(float(value) for value in raw_coefficients)
            shift = shifts[0]
            coefficient = coefficients[0]
        frequencies: tuple[float, ...] | None = None
        if self.frequencies is not None:
            raw_frequencies = _as_parameter_array(self.frequencies)
            if raw_frequencies.size == 0:
                raise ValueError("frequencies must contain at least one value")
            if np.any(raw_frequencies <= 0.0):
                raise ValueError("frequencies must contain finite positive values")
            if np.unique(raw_frequencies).size != raw_frequencies.size:
                raise ValueError("frequencies must be unique")
            frequencies = tuple(float(value) for value in raw_frequencies)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "shifts", shifts)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "frequencies", frequencies)

    @property
    def terms(self) -> tuple[tuple[float, float], ...]:
        """``(shift, coefficient)`` terms for symmetric plus/minus probes."""
        shifts = cast(tuple[float, ...], self.shifts)
        coefficients = cast(tuple[float, ...], self.coefficients)
        return tuple(zip(shifts, coefficients, strict=True))

    @property
    def is_single_term(self) -> bool:
        """Whether this rule is the legacy two-evaluation rule."""
        return len(self.terms) == 1

    @property
    def shot_variance_factor(self) -> float:
        """Variance of the estimate per unit single-evaluation variance, ``2 * sum c**2``.

        With ``n`` shots per evaluation and single-shot variance ``s**2`` the
        estimator's variance is ``shot_variance_factor * s**2 / n``, so this is
        the shot cost of the rule relative to one exact evaluation pair.
        """
        coefficients = cast(tuple[float, ...], self.coefficients)
        return 2.0 * float(sum(coefficient * coefficient for coefficient in coefficients))


def multi_frequency_parameter_shift_rule(
    frequencies: ArrayLike,
    *,
    shifts: ArrayLike | None = None,
    max_condition: float = 1.0e10,
) -> ParameterShiftRule:
    """Return an exact multi-frequency parameter-shift rule.

    For trigonometric objectives with positive generator frequency set
    ``frequencies``, the coefficients solve
    ``2 * sin(frequency_i * shift_j) @ coefficient_j = frequency_i``.
    The resulting rule can exactly differentiate any supported linear
    combination of sine/cosine components at those frequencies.
    """
    frequency_values = _as_parameter_array(frequencies)
    if frequency_values.size == 0:
        raise ValueError("frequencies must contain at least one value")
    if np.any(frequency_values <= 0.0):
        raise ValueError("frequencies must contain finite positive values")
    if np.unique(frequency_values).size != frequency_values.size:
        raise ValueError("frequencies must be unique")
    condition_limit = _as_real_scalar("max_condition", max_condition)
    if condition_limit <= 1.0:
        raise ValueError("max_condition must be finite and greater than one")

    if shifts is None:
        frequency_values, shift_values = _default_multi_frequency_shifts(
            frequency_values, max_condition=condition_limit
        )
    else:
        shift_values = _as_parameter_array(shifts)
    if shift_values.shape != frequency_values.shape:
        raise ValueError("shifts must have the same length as frequencies")
    if np.any(shift_values <= 0.0):
        raise ValueError("shifts must contain finite positive values")

    sine_system = 2.0 * np.sin(np.outer(frequency_values, shift_values))
    system_scale = float(np.max(np.abs(sine_system))) if sine_system.size else 0.0
    if system_scale <= np.finfo(np.float64).eps:
        raise ValueError("multi-frequency parameter-shift system is singular or ill-conditioned")
    rank_tolerance = max(sine_system.shape) * np.finfo(np.float64).eps * max(1.0, system_scale)
    singular_values = np.linalg.svd(sine_system, compute_uv=False)
    rank = sum(1 for singular_value in singular_values if singular_value > rank_tolerance)
    if rank != frequency_values.size or float(singular_values[-1]) < _MIN_SINE_SINGULAR_VALUE:
        raise ValueError("multi-frequency parameter-shift system is singular or ill-conditioned")
    condition = float(np.linalg.cond(sine_system))
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError("multi-frequency parameter-shift system is singular or ill-conditioned")
    coefficients = np.linalg.solve(sine_system, frequency_values)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("multi-frequency parameter-shift coefficients must be finite")
    return ParameterShiftRule(
        shift=float(shift_values[0]),
        coefficient=float(coefficients[0]),
        shifts=tuple(float(value) for value in shift_values),
        coefficients=tuple(float(value) for value in coefficients),
        frequencies=tuple(float(value) for value in frequency_values),
    )


_MIN_SINE_SINGULAR_VALUE = 1.0e-6
"""Absolute floor on the sine system's smallest singular value.

The condition number is scale-free: a one-term system ``2*sin(w*s)`` always has
condition one, even when ``w*s`` is a multiple of pi and the coefficient blows
up. Its entries are bounded by two, so this floor keeps every coefficient
within ``|w| * 1e6`` of the exact rule.
"""

MAX_GENERAL_SPECTRUM_FREQUENCIES = 16
"""Most frequencies a non-equidistant spectrum may have; larger ones are refused
before any shift search, so planning time stays bounded."""

MAX_EQUIDISTANT_SPECTRUM_ORDER = 64
"""Most frequencies of an equidistant spectrum; its rule takes two evaluations each."""

_EQUIDISTANT_TOLERANCE = 1.0e-9
"""Relative distance from an integer multiple accepted as an equidistant spectrum."""


def _equidistant_order(frequencies: NDArray[np.float64]) -> int | None:
    """Return ``R`` when the frequencies are ``{1, ..., R}`` times their smallest."""
    ordered = np.sort(frequencies)
    multiples = ordered / ordered[0]
    expected = np.arange(1, ordered.size + 1, dtype=np.float64)
    if np.all(np.abs(multiples - expected) <= _EQUIDISTANT_TOLERANCE * expected):
        return int(ordered.size)
    return None


def _equidistant_shifts(order: int, base: float) -> NDArray[np.float64]:
    """Return the equidistant-rule shifts ``(2m - 1) pi / (2 R W)`` for ``m = 1..R``."""
    return (2.0 * np.arange(1, order + 1, dtype=np.float64) - 1.0) * np.pi / (2.0 * order * base)


def _superset_order(frequencies: NDArray[np.float64]) -> int | None:
    """Return ``M`` when every frequency is an integer multiple ``<= M`` of the smallest."""
    base = float(np.min(frequencies))
    multiples = frequencies / base
    rounded = np.round(multiples)
    if np.all(np.abs(multiples - rounded) <= _EQUIDISTANT_TOLERANCE * rounded):
        return int(np.max(rounded))
    return None


def _default_multi_frequency_shifts(
    frequencies: NDArray[np.float64],
    *,
    max_condition: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the frequencies to solve over and the cheapest admissible shifts.

    An equidistant spectrum ``{W, 2W, ..., RW}`` gets the equidistant rule,
    shifts ``(2m - 1) * pi / (2 R W)`` for ``m = 1..R``, which is exact and
    whose coefficients are ``W (-1)**(m-1) / (4 R sin**2((2m - 1) pi / (4R)))``.
    A single frequency ``w`` is the case ``R = 1``: shift ``pi / (2w)`` and
    coefficient ``w / 2``, however close ``w`` is to an integer.

    Any other spectrum is solved over several candidate grids, and the one
    with the smallest shot cost ``2 * sum c**2`` times its number of terms is
    kept. The candidates are:
    - the equidistant shifts of its own size, solved on the actual
      frequencies, which is exact for a nearly equidistant spectrum;
    - the equidistant rule of the smallest equidistant superset, when every
      frequency is an integer multiple of the smallest; that rule is exact on
      the superset and therefore on the spectrum;
    - grids ``k pi / D`` at unit top frequency;
    - grids ``k pi / D`` at the original scale.
    The original first-admissible grid is among the candidates, so the chosen
    rule never costs more than it did.
    """
    order = _equidistant_order(frequencies)
    if order is not None:
        if order > MAX_EQUIDISTANT_SPECTRUM_ORDER:
            raise ValueError(
                f"an equidistant spectrum may have at most {MAX_EQUIDISTANT_SPECTRUM_ORDER} "
                "frequencies"
            )
        return frequencies, _equidistant_shifts(order, float(np.min(frequencies)))
    count = int(frequencies.size)
    if count > MAX_GENERAL_SPECTRUM_FREQUENCIES:
        raise ValueError(
            f"a non-equidistant spectrum may have at most {MAX_GENERAL_SPECTRUM_FREQUENCIES} "
            "frequencies"
        )
    base = float(np.min(frequencies))
    top = float(np.max(frequencies))
    steps = np.arange(1, count + 1, dtype=np.float64)
    candidates: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = [
        (frequencies, _equidistant_shifts(count, base))
    ]
    superset = _superset_order(frequencies)
    if superset is not None and superset <= MAX_EQUIDISTANT_SPECTRUM_ORDER:
        candidates.append(
            (
                base * np.arange(1, superset + 1, dtype=np.float64),
                _equidistant_shifts(superset, base),
            )
        )
    candidates.extend(
        (frequencies, steps * np.pi / (float(denominator) * top))
        for denominator in range(count + 1, count + 129)
    )
    raw_start = 2 * count + 1
    raw_stop = min(max(raw_start + 64, int(np.ceil(4.0 * top * count)) + 65), raw_start + 512)
    candidates.extend(
        (frequencies, steps * np.pi / float(denominator))
        for denominator in range(raw_start, raw_stop)
    )
    best: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
    for solved, candidate in candidates:
        # An exact rule never needs a phase above pi times its size; larger
        # phases lose the derivative to rounding in the evaluated objective.
        if float(np.max(solved)) * float(np.max(candidate)) > np.pi * solved.size:
            continue
        sine_system = 2.0 * np.sin(np.outer(solved, candidate))
        singular_values = np.linalg.svd(sine_system, compute_uv=False)
        smallest = float(singular_values[-1])
        if smallest < _MIN_SINE_SINGULAR_VALUE or float(singular_values[0]) / smallest > (
            max_condition
        ):
            continue
        coefficients = np.linalg.solve(sine_system, solved)
        cost = 2.0 * float(np.sum(coefficients**2)) * candidate.size
        if best is None or cost < best[0]:
            best = (cost, solved, candidate)
    if best is None:
        raise ValueError("could not construct a well-conditioned multi-frequency shift system")
    return best[1], best[2]


__all__ = [
    "MAX_EQUIDISTANT_SPECTRUM_ORDER",
    "MAX_GENERAL_SPECTRUM_FREQUENCIES",
    "Parameter",
    "ParameterBounds",
    "ParameterShiftRule",
    "multi_frequency_parameter_shift_rule",
]
