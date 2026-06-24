# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Daido m-th Fourier-mode phase and its gradient
"""Daido m-th Fourier-mode phase and its gradient.

The m-th Fourier mode of the phase distribution is ``z_m = r_m e^{iψ_m}`` with magnitude
``r_m`` (the Daido order parameter) and phase ``ψ_m = atan2(⟨sin mθ⟩, ⟨cos mθ⟩)``. This module
is to :mod:`~scpn_quantum_control.accel.daido_observables` what
:mod:`~scpn_quantum_control.accel.mean_phase_observables` is to
:mod:`~scpn_quantum_control.accel.order_parameter_observables`: the phase partner of the
magnitude. For ``m = 1`` it equals the Kuramoto mean phase. Its gradient is
``∂ψ_m/∂θ_j = (m / (N r_m)) cos(ψ_m − m θ_j)``, whose components sum to ``m`` (a global phase
shift advances ψ_m by ``m`` times the shift); the incoherent mode ``r_m = 0`` has an undefined
phase and returns the zero subgradient.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _validate_harmonic(m: int) -> None:
    """Raise :class:`ValueError` if ``m`` is not a positive integer."""
    if m < 1:
        raise ValueError(f"harmonic order m must be a positive integer, got {m}")


def _rust_daido_mode_phase(theta: NDArray[np.float64], m: int) -> float:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_phase = getattr(engine, "daido_mode_phase", None)
    if not callable(rust_phase):
        raise ImportError("scpn_quantum_engine.daido_mode_phase is unavailable")

    return float(rust_phase(np.ascontiguousarray(theta, dtype=np.float64), m))


def _julia_daido_mode_phase(theta: NDArray[np.float64], m: int) -> float:
    _validate_harmonic(m)
    from .julia import daido_mode_phase as julia_phase

    return julia_phase(theta, m)


def _python_daido_mode_phase(theta: NDArray[np.float64], m: int) -> float:
    # Correctness floor — ψ_m = atan2(⟨sin mθ⟩, ⟨cos mθ⟩). The 1/N scaling cancels inside
    # atan2. An empty input and the incoherent mode both report 0.0. For m = 1 this is the
    # Kuramoto mean phase.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    scaled = m * phases
    return float(np.arctan2(float(np.sum(np.sin(scaled))), float(np.sum(np.cos(scaled)))))


_DAIDO_MODE_PHASE_CHAIN: list[tuple[str, Callable[[NDArray[np.float64], int], float]]] = [
    ("rust", _rust_daido_mode_phase),
    ("julia", _julia_daido_mode_phase),
    ("python", _python_daido_mode_phase),
]


def _rust_daido_mode_phase_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_gradient = getattr(engine, "daido_mode_phase_gradient", None)
    if not callable(rust_gradient):
        raise ImportError("scpn_quantum_engine.daido_mode_phase_gradient is unavailable")

    return np.asarray(
        rust_gradient(np.ascontiguousarray(theta, dtype=np.float64), m), dtype=np.float64
    )


def _julia_daido_mode_phase_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    _validate_harmonic(m)
    from .julia import daido_mode_phase_gradient as julia_gradient

    return julia_gradient(theta, m)


def _python_daido_mode_phase_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    # Correctness floor — ∂ψ_m/∂θ_j = (m / (N r_m)) cos(ψ_m − m θ_j)
    #   = (m / (N r_m²)) (C_m cos(m θ_j) + S_m sin(m θ_j)), with C_m = ⟨cos mθ⟩, S_m = ⟨sin mθ⟩
    # and r_m = hypot(C_m, S_m). The components sum to m; the incoherent mode r_m = 0 returns
    # the zero subgradient. For m = 1 this is the Kuramoto mean-phase gradient.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    scaled = m * phases
    cos_mean = float(np.mean(np.cos(scaled)))
    sin_mean = float(np.mean(np.sin(scaled)))
    magnitude_squared = cos_mean * cos_mean + sin_mean * sin_mean
    if magnitude_squared == 0.0:
        return np.zeros(count, dtype=np.float64)
    gradient = (m / (count * magnitude_squared)) * (
        cos_mean * np.cos(scaled) + sin_mean * np.sin(scaled)
    )
    return np.ascontiguousarray(gradient, dtype=np.float64)


# The gradient chain mirrors the value chain (Rust → Julia → Python floor). Their
# micro-benchmark (at m = 2) is recorded in ``docs/benchmarks/daido_mode_phase_tiers.json``;
# rerun ``python scripts/bench_daido_mode_phase_tiers.py`` when these chains are edited.
_DAIDO_MODE_PHASE_GRADIENT_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], int], NDArray[np.float64]]]
] = [
    ("rust", _rust_daido_mode_phase_gradient),
    ("julia", _julia_daido_mode_phase_gradient),
    ("python", _python_daido_mode_phase_gradient),
]


def _rust_daido_mode_phase_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_hessian = getattr(engine, "daido_mode_phase_hessian", None)
    if not callable(rust_hessian):
        raise ImportError("scpn_quantum_engine.daido_mode_phase_hessian is unavailable")

    return np.asarray(
        rust_hessian(np.ascontiguousarray(theta, dtype=np.float64), m), dtype=np.float64
    )


def _julia_daido_mode_phase_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    _validate_harmonic(m)
    from .julia import daido_mode_phase_hessian as julia_hessian

    return julia_hessian(theta, m)


def _python_daido_mode_phase_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    # Correctness floor — H_ij = m² [δ_ij s_j/(N r_m) − (s_i c_j + c_i s_j)/(N² r_m²)], with
    # s_k = sin(ψ_m − m θ_k), c_k = cos(ψ_m − m θ_k). The matrix is symmetric and every row
    # sums to zero (the mode-phase gradient sums to the constant m). The incoherent mode
    # r_m = 0 returns the zero matrix. For m = 1 this reduces to ``mean_phase_hessian``.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    scaled = m * phases
    cos_scaled = np.cos(scaled)
    sin_scaled = np.sin(scaled)
    cos_mean = float(np.mean(cos_scaled))
    sin_mean = float(np.mean(sin_scaled))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros((count, count), dtype=np.float64)
    sin_aligned = (sin_mean * cos_scaled - cos_mean * sin_scaled) / magnitude
    cos_aligned = (cos_mean * cos_scaled + sin_mean * sin_scaled) / magnitude
    hessian = (m * m) * (
        np.diag(sin_aligned / (count * magnitude))
        - (np.outer(sin_aligned, cos_aligned) + np.outer(cos_aligned, sin_aligned))
        / (count * count * magnitude * magnitude)
    )
    return np.ascontiguousarray(hessian, dtype=np.float64)


# The Hessian chain mirrors the value/gradient chains. Its micro-benchmark (at m = 2) is
# recorded in ``docs/benchmarks/daido_mode_phase_tiers.json``; rerun
# ``python scripts/bench_daido_mode_phase_tiers.py`` when this chain is edited.
_DAIDO_MODE_PHASE_HESSIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], int], NDArray[np.float64]]]
] = [
    ("rust", _rust_daido_mode_phase_hessian),
    ("julia", _julia_daido_mode_phase_hessian),
    ("python", _python_daido_mode_phase_hessian),
]


_daido_mode_phase_dispatcher = MultiLangDispatcher(_DAIDO_MODE_PHASE_CHAIN)
_daido_mode_phase_gradient_dispatcher = MultiLangDispatcher(_DAIDO_MODE_PHASE_GRADIENT_CHAIN)
_daido_mode_phase_hessian_dispatcher = MultiLangDispatcher(_DAIDO_MODE_PHASE_HESSIAN_CHAIN)


def daido_mode_phase(theta: NDArray[np.float64], m: int) -> float:
    r"""m-th Fourier-mode phase with multi-language dispatch.

    Returns :math:`\psi_m = \operatorname{atan2}(\langle\sin m\theta\rangle,
    \langle\cos m\theta\rangle)`, the phase of the m-th Fourier mode
    :math:`z_m = r_m e^{i\psi_m}`. For :math:`m = 1` it equals
    :func:`~scpn_quantum_control.accel.mean_phase_observables.mean_phase`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    float
        The mode phase in radians on ``(-π, π]``. An empty input returns ``0.0``.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_mode_phase_tier_used`.
    """
    return float(_daido_mode_phase_dispatcher(theta, m))


def daido_mode_phase_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    r"""Gradient of the m-th Fourier-mode phase with multi-language dispatch.

    Returns :math:`\partial \psi_m / \partial \theta_j = (m / (N r_m)) \cos(\psi_m - m\theta_j)`,
    whose components sum to ``m`` (a global phase shift advances :math:`\psi_m` by m times the
    shift). At the incoherent mode (:math:`r_m = 0`) the zero subgradient is returned. For
    :math:`m = 1` it equals
    :func:`~scpn_quantum_control.accel.mean_phase_observables.mean_phase_gradient`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of the same length as ``theta``. An empty input yields
        an empty array.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_mode_phase_gradient_tier_used`.
    """
    return np.asarray(_daido_mode_phase_gradient_dispatcher(theta, m), dtype=np.float64)


def last_daido_mode_phase_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_mode_phase``."""
    return _daido_mode_phase_dispatcher.last_tier


def last_daido_mode_phase_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_mode_phase_gradient``."""
    return _daido_mode_phase_gradient_dispatcher.last_tier


def daido_mode_phase_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    r"""Hessian of the m-th Fourier-mode phase with multi-language dispatch.

    Returns :math:`\partial^2 \psi_m / \partial \theta_i \partial \theta_j = m^2 [\delta_{ij}
    s_j / (N r_m) - (s_i c_j + c_i s_j) / (N^2 r_m^2)]`, with :math:`s_k = \sin(\psi_m -
    m\theta_k)` and :math:`c_k = \cos(\psi_m - m\theta_k)`. The matrix is symmetric and every
    row sums to zero (the gradient sums to the constant m). At the incoherent mode
    (:math:`r_m = 0`) the zero matrix is returned. For :math:`m = 1` it reduces to
    :func:`~scpn_quantum_control.accel.mean_phase_observables.mean_phase_hessian`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Hessian matrix. An empty input yields a
        ``(0, 0)`` array.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_mode_phase_hessian_tier_used`.
    """
    return np.asarray(_daido_mode_phase_hessian_dispatcher(theta, m), dtype=np.float64)


def last_daido_mode_phase_hessian_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_mode_phase_hessian``."""
    return _daido_mode_phase_hessian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("daido_mode_phase", _daido_mode_phase_dispatcher)
register_dispatcher("daido_mode_phase_gradient", _daido_mode_phase_gradient_dispatcher)
register_dispatcher("daido_mode_phase_hessian", _daido_mode_phase_hessian_dispatcher)


__all__ = [
    "daido_mode_phase",
    "daido_mode_phase_gradient",
    "daido_mode_phase_hessian",
    "last_daido_mode_phase_gradient_tier_used",
    "last_daido_mode_phase_hessian_tier_used",
    "last_daido_mode_phase_tier_used",
]
