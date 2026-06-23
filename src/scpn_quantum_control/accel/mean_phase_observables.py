# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto mean-phase observables
"""Kuramoto mean-phase observables.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _rust_mean_phase(theta: NDArray[np.float64]) -> float:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_mean_phase = getattr(engine, "mean_phase", None)
    if not callable(rust_mean_phase):
        raise ImportError("scpn_quantum_engine.mean_phase is unavailable")

    return float(rust_mean_phase(np.ascontiguousarray(theta, dtype=np.float64)))


def _julia_mean_phase(theta: NDArray[np.float64]) -> float:
    from .julia import mean_phase as julia_mean_phase

    return julia_mean_phase(theta)


def _python_mean_phase(theta: NDArray[np.float64]) -> float:
    # Correctness floor — circular mean phase ψ = atan2(<sin θ>, <cos θ>), no
    # acceleration. The empty input maps to 0.0; the incoherent state is reported as
    # atan2(0, 0) = 0.0 (the mean phase is undefined there but the value stays finite).
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    return float(np.arctan2(float(np.mean(np.sin(phases))), float(np.mean(np.cos(phases)))))


def _rust_mean_phase_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_grad = getattr(engine, "mean_phase_gradient", None)
    if not callable(rust_grad):
        raise ImportError("scpn_quantum_engine.mean_phase_gradient is unavailable")

    return np.asarray(rust_grad(np.ascontiguousarray(theta, dtype=np.float64)), dtype=np.float64)


def _julia_mean_phase_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    from .julia import mean_phase_gradient as julia_grad

    return julia_grad(theta)


def _python_mean_phase_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    # Correctness floor — gradient of the mean phase ψ = atan2(S, C). With C = <cos θ>,
    # S = <sin θ> and r = hypot(C, S):
    #     ∂ψ/∂θ_j = cos(ψ − θ_j) / (N r) = (C cos θ_j + S sin θ_j) / (N r²).
    # The components sum to one (a global phase shift advances ψ identically). At the
    # incoherent state r = 0 the mean phase is undefined and the zero gradient is returned.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros(count, dtype=np.float64)
    gradient = (cos_mean * np.cos(phases) + sin_mean * np.sin(phases)) / (
        count * magnitude * magnitude
    )
    return np.ascontiguousarray(gradient, dtype=np.float64)


# The mean-phase value and gradient chains mirror the order-parameter chains
# (Rust → Julia → Python floor); they share the same per-oscillator trigonometric
# pre-pass. Their micro-benchmarks are recorded in
# ``docs/benchmarks/mean_phase_tiers.json``; rerun
# ``python scripts/bench_mean_phase_tiers.py`` when these chains are edited.
_MEAN_PHASE_CHAIN: list[tuple[str, Callable[[NDArray[np.float64]], float]]] = [
    ("rust", _rust_mean_phase),
    ("julia", _julia_mean_phase),
    ("python", _python_mean_phase),
]
_MEAN_PHASE_GRADIENT_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_mean_phase_gradient),
    ("julia", _julia_mean_phase_gradient),
    ("python", _python_mean_phase_gradient),
]


def _rust_mean_phase_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_hessian = getattr(engine, "mean_phase_hessian", None)
    if not callable(rust_hessian):
        raise ImportError("scpn_quantum_engine.mean_phase_hessian is unavailable")

    return np.asarray(
        rust_hessian(np.ascontiguousarray(theta, dtype=np.float64)), dtype=np.float64
    )


def _julia_mean_phase_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    from .julia import mean_phase_hessian as julia_hessian

    return julia_hessian(theta)


def _python_mean_phase_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    # Correctness floor — analytic Hessian of the mean phase ψ = atan2(S, C), no
    # acceleration. With c_k = cos(ψ − θ_k) = (C cos θ_k + S sin θ_k) / r and
    # s_k = sin(ψ − θ_k) = (S cos θ_k − C sin θ_k) / r:
    #     ∂²ψ/∂θ_i∂θ_j = δ_ij s_j / (N r) − (s_i c_j + c_i s_j) / (N² r²).
    # The matrix is symmetric and each row sums to zero (the second derivative along a
    # global phase shift vanishes). At the incoherent state r = 0 the zero matrix is
    # returned.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros((count, count), dtype=np.float64)
    aligned_cos = (cos_mean * np.cos(phases) + sin_mean * np.sin(phases)) / magnitude
    aligned_sin = (sin_mean * np.cos(phases) - cos_mean * np.sin(phases)) / magnitude
    hessian = -(np.outer(aligned_sin, aligned_cos) + np.outer(aligned_cos, aligned_sin)) / (
        count * count * magnitude * magnitude
    )
    hessian += np.diag(aligned_sin / (count * magnitude))
    return np.ascontiguousarray(hessian, dtype=np.float64)


# The mean-phase Hessian chain mirrors the order-parameter and mean-phase chains
# (Rust → Julia → Python floor). Its micro-benchmark is recorded in
# ``docs/benchmarks/mean_phase_hessian_tiers.json``; rerun
# ``python scripts/bench_mean_phase_hessian_tiers.py`` when this chain is edited.
_MEAN_PHASE_HESSIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_mean_phase_hessian),
    ("julia", _julia_mean_phase_hessian),
    ("python", _python_mean_phase_hessian),
]


_mean_phase_dispatcher = MultiLangDispatcher(_MEAN_PHASE_CHAIN)
_mean_phase_gradient_dispatcher = MultiLangDispatcher(_MEAN_PHASE_GRADIENT_CHAIN)
_mean_phase_hessian_dispatcher = MultiLangDispatcher(_MEAN_PHASE_HESSIAN_CHAIN)


def mean_phase(theta: NDArray[np.float64]) -> float:
    r"""Circular mean phase of a Kuramoto ensemble with multi-language dispatch.

    Returns :math:`\psi = \operatorname{atan2}(\langle \sin\theta \rangle,
    \langle \cos\theta \rangle)`, the collective phase of the complex order
    parameter :math:`z = r e^{i\psi}`. An empty input returns ``0.0``; the
    incoherent state reports ``atan2(0, 0) = 0.0``.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    float
        The mean phase in radians on ``(-π, π]``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier
    is recorded on :func:`last_mean_phase_tier_used`.
    """
    return float(_mean_phase_dispatcher(theta))


def mean_phase_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Gradient of the Kuramoto mean phase with multi-language dispatch.

    Returns :math:`\partial \psi / \partial \theta_j = \cos(\psi - \theta_j) /
    (N r)` for the mean phase :math:`\psi`. The components sum to one, since a
    global phase shift advances :math:`\psi` identically. At the incoherent state
    (:math:`r = 0`) the mean phase is undefined and the zero gradient is returned.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of the same length as ``theta``. An empty
        input yields an empty array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier
    is recorded on :func:`last_mean_phase_gradient_tier_used`.
    """
    return np.asarray(_mean_phase_gradient_dispatcher(theta), dtype=np.float64)


def last_mean_phase_tier_used() -> str | None:
    """Return the tier that served the most recent ``mean_phase``."""
    return _mean_phase_dispatcher.last_tier


def mean_phase_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Hessian of the Kuramoto mean phase with multi-language dispatch.

    Returns the second-derivative matrix :math:`\partial^2 \psi / \partial \theta_i
    \partial \theta_j` for the mean phase :math:`\psi`. With :math:`s_k = \sin(\psi -
    \theta_k)` and :math:`c_k = \cos(\psi - \theta_k)`, the entries are
    :math:`\delta_{ij} s_j / (N r) - (s_i c_j + c_i s_j) / (N^2 r^2)`. The matrix is
    symmetric and every row sums to zero (the second derivative along a global phase
    shift vanishes). At the incoherent state (:math:`r = 0`) the zero matrix is returned.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Hessian matrix. An empty input yields a
        ``(0, 0)`` array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_mean_phase_hessian_tier_used`.
    """
    return np.asarray(_mean_phase_hessian_dispatcher(theta), dtype=np.float64)


def last_mean_phase_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``mean_phase_gradient``."""
    return _mean_phase_gradient_dispatcher.last_tier


def last_mean_phase_hessian_tier_used() -> str | None:
    """Return the tier that served the most recent ``mean_phase_hessian``."""
    return _mean_phase_hessian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("mean_phase", _mean_phase_dispatcher)
register_dispatcher("mean_phase_gradient", _mean_phase_gradient_dispatcher)
register_dispatcher("mean_phase_hessian", _mean_phase_hessian_dispatcher)


__all__ = [
    "last_mean_phase_gradient_tier_used",
    "last_mean_phase_hessian_tier_used",
    "last_mean_phase_tier_used",
    "mean_phase",
    "mean_phase_gradient",
    "mean_phase_hessian",
]
