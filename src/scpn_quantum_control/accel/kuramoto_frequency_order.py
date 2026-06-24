# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto frequency-locking diagnostics from a phase trajectory
r"""Frequency-locking diagnostics for a Kuramoto phase trajectory.

The phase order parameter :math:`r = |\langle e^{i\theta}\rangle|` measures *phase*
coherence; the frequency analogue measures *frequency* locking — whether the
oscillators share a common long-time rotation rate even if their phases stay spread.
For a trajectory sampled at a fixed step :math:`\Delta t` the observed (effective)
frequency of oscillator :math:`i` is the time average of its instantaneous angular
velocity,

.. math::

    \Omega_i = \frac{1}{T\,\Delta t}\sum_{k=0}^{T-1}
        \operatorname{wrap}\!\big(\theta_i(t_{k+1}) - \theta_i(t_k)\big),

where :math:`\operatorname{wrap}` maps an angle into :math:`(-\pi, \pi]`. Summing the
*wrapped* per-step advance makes the estimate independent of how the trajectory is
stored (wrapped to :math:`(-\pi,\pi]` or freely accumulating), and when no single step
advances by more than :math:`\pi` the sum telescopes to
:math:`(\theta_i(t_{\mathrm{end}}) - \theta_i(t_0))/(T\,\Delta t)`. The estimate aliases
only if the true per-step advance exceeds :math:`\pi` (the rotational Nyquist limit), so
the step must resolve the fastest oscillator.

The frequency-synchronisation index is the population standard deviation of the
effective frequencies: it is zero for a frequency-locked state (all :math:`\Omega_i`
equal) and grows with the frequency spread. The locked fraction reports how many
oscillators sit within a tolerance of the mean effective frequency — the size of the
frequency-locked cluster.

This is a pure-Python analysis layer over the accelerated snapshot observables; it adds
no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _validate_trajectory(phases: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """Return a contiguous ``(T, N)`` trajectory after validating it and ``dt``.

    Raises
    ------
    ValueError
        If ``phases`` is not a two-dimensional array with at least two time samples and
        one oscillator, or if ``dt`` is not strictly positive.
    """
    trajectory = np.ascontiguousarray(phases, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(
            f"phases must be a two-dimensional (T, N) trajectory, got ndim {trajectory.ndim}"
        )
    samples, count = trajectory.shape
    if samples < 2:
        raise ValueError(
            f"phases needs at least two time samples to form a frequency, got {samples}"
        )
    if count < 1:
        raise ValueError("phases must hold at least one oscillator")
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
    return trajectory


def _wrap(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map angles into ``(-π, π]`` via the complex argument."""
    return np.asarray(np.angle(np.exp(1j * values)), dtype=np.float64)


def effective_frequencies(phases: NDArray[np.float64], *, dt: float) -> NDArray[np.float64]:
    r"""Time-averaged observed frequency of each oscillator along a trajectory.

    Returns :math:`\Omega_i`, the mean of the wrapped per-step phase advance divided by
    :math:`\Delta t`. Summing the wrapped advance makes the result independent of whether
    the trajectory is stored wrapped or freely accumulating, provided no single step
    advances by more than :math:`\pi`.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians, sampled at a
        fixed step; ``T`` is the number of time samples and ``N`` the oscillator count.
    dt : float
        The sampling step in time units; must be strictly positive.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of ``N`` effective angular frequencies.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 2`` and ``N ≥ 1``, or if
        ``dt`` is not strictly positive.
    """
    trajectory = _validate_trajectory(phases, dt)
    advance = _wrap(np.diff(trajectory, axis=0))
    span = (trajectory.shape[0] - 1) * dt
    return np.ascontiguousarray(advance.sum(axis=0) / span, dtype=np.float64)


def frequency_synchronisation_index(phases: NDArray[np.float64], *, dt: float) -> float:
    r"""Population standard deviation of the effective frequencies.

    The frequency analogue of the order parameter's dispersion: zero for a
    frequency-locked state (every :math:`\Omega_i` equal) and growing with the spread of
    observed rotation rates. Unlike the bounded phase order parameter this index is an
    unbounded spread, so it scales with the frequency units of the trajectory.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    dt : float
        The sampling step in time units; must be strictly positive.

    Returns
    -------
    float
        The population standard deviation of the effective frequencies; ``0.0`` for a
        single oscillator.

    Raises
    ------
    ValueError
        Propagated from :func:`effective_frequencies`.
    """
    return float(np.std(effective_frequencies(phases, dt=dt)))


def frequency_synchronisation_index_gradient(
    phases: NDArray[np.float64], *, dt: float
) -> NDArray[np.float64]:
    r"""Gradient of the frequency-synchronisation index with respect to the trajectory.

    Returns :math:`\partial\,\sigma_\Omega / \partial\,\theta_{k,i}` for the
    synchronisation index :math:`\sigma_\Omega = \operatorname{std}\{\Omega_i\}`, shaped
    like the input ``(T, N)`` trajectory. Because each effective frequency telescopes to
    its endpoints, :math:`\Omega_i = (\theta_i(t_{\mathrm{end}}) - \theta_i(t_0))/(T\Delta
    t)` below the rotational Nyquist limit, the gradient is non-zero **only in the first
    and last time rows** — the interior samples cancel — with

    .. math::

        \frac{\partial\,\sigma_\Omega}{\partial\,\theta_{T-1,i}}
            = \frac{\Omega_i - \langle\Omega\rangle}{N\,\sigma_\Omega\,T\Delta t}
            = -\,\frac{\partial\,\sigma_\Omega}{\partial\,\theta_{0,i}} .

    It is the first link of the adjoint chain for steering an ensemble towards frequency
    locking. At a frequency-locked state (:math:`\sigma_\Omega \to 0`) the standard
    deviation is non-differentiable, so the zero subgradient is returned, mirroring the
    incoherent-state convention of
    :func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter_gradient`.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    dt : float
        The sampling step in time units; must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(T, N)`` float64 gradient; only rows ``0`` and ``T-1`` are
        non-zero and they are negatives of one another.

    Raises
    ------
    ValueError
        Propagated from :func:`effective_frequencies`.
    """
    trajectory = _validate_trajectory(phases, dt)
    samples, count = trajectory.shape
    span = (samples - 1) * dt
    frequencies = _wrap(np.diff(trajectory, axis=0)).sum(axis=0) / span
    spread = float(np.std(frequencies))
    gradient = np.zeros_like(trajectory)
    scale = max(1.0, float(np.max(np.abs(frequencies))))
    if spread <= 1e-12 * scale:
        # Frequency-locked: the standard deviation has a kink at zero spread; the zero
        # subgradient is the honest derivative of a non-differentiable point.
        return gradient
    endpoint = (frequencies - float(np.mean(frequencies))) / (count * spread * span)
    gradient[-1] = endpoint
    gradient[0] = -endpoint
    return np.ascontiguousarray(gradient, dtype=np.float64)


def frequency_locked_fraction(
    phases: NDArray[np.float64], *, dt: float, tolerance: float = 1e-3
) -> float:
    r"""Fraction of oscillators locked to the mean effective frequency.

    Reports the share of oscillators whose effective frequency lies within ``tolerance``
    of the mean effective frequency :math:`\langle\Omega\rangle` — the relative size of
    the frequency-locked cluster. A fully frequency-locked ensemble returns ``1.0``; when
    the ensemble splits into drifting groups straddling the mean the fraction falls.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    dt : float
        The sampling step in time units; must be strictly positive.
    tolerance : float, optional
        The absolute frequency window about the mean counted as locked; must be
        non-negative. Defaults to ``1e-3``.

    Returns
    -------
    float
        The locked fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``tolerance`` is negative, or propagated from :func:`effective_frequencies`.
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    frequencies = effective_frequencies(phases, dt=dt)
    locked = np.abs(frequencies - float(np.mean(frequencies))) <= tolerance
    return float(np.mean(locked))


@dataclass(frozen=True)
class FrequencyOrder:
    """Bundled frequency-locking diagnostics for a trajectory.

    Attributes
    ----------
    effective_frequencies : numpy.ndarray
        The ``N`` observed angular frequencies :math:`\\Omega_i`.
    synchronisation_index : float
        The population standard deviation of ``effective_frequencies``.
    locked_fraction : float
        The fraction of oscillators within the tolerance of the mean effective frequency.
    """

    effective_frequencies: NDArray[np.float64]
    synchronisation_index: float
    locked_fraction: float


def frequency_order_diagnostics(
    phases: NDArray[np.float64], *, dt: float, tolerance: float = 1e-3
) -> FrequencyOrder:
    """Compute all frequency-locking diagnostics in a single pass.

    Evaluates the effective frequencies once and derives the synchronisation index and
    the locked fraction from them, avoiding the repeated trajectory differencing of the
    individual helpers.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    dt : float
        The sampling step in time units; must be strictly positive.
    tolerance : float, optional
        The absolute frequency window about the mean counted as locked; must be
        non-negative. Defaults to ``1e-3``.

    Returns
    -------
    FrequencyOrder
        The bundled effective frequencies, synchronisation index and locked fraction.

    Raises
    ------
    ValueError
        If ``tolerance`` is negative, or propagated from :func:`effective_frequencies`.
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    frequencies = effective_frequencies(phases, dt=dt)
    mean_frequency = float(np.mean(frequencies))
    locked = np.abs(frequencies - mean_frequency) <= tolerance
    return FrequencyOrder(
        effective_frequencies=frequencies,
        synchronisation_index=float(np.std(frequencies)),
        locked_fraction=float(np.mean(locked)),
    )


__all__ = [
    "FrequencyOrder",
    "effective_frequencies",
    "frequency_locked_fraction",
    "frequency_order_diagnostics",
    "frequency_synchronisation_index",
    "frequency_synchronisation_index_gradient",
]
