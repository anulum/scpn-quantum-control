# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto order-parameter observables
"""Kuramoto order-parameter observables.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~oscillatools.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _rust_order_parameter(theta: NDArray[np.float64]) -> float:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_op = getattr(engine, "order_parameter", None)
    if not callable(rust_op):
        raise ImportError("scpn_quantum_engine.order_parameter is unavailable")

    return float(rust_op(np.ascontiguousarray(theta, dtype=np.float64)))


def _julia_order_parameter(theta: NDArray[np.float64]) -> float:
    from .julia import order_parameter as julia_op

    return julia_op(theta)


def _python_order_parameter(theta: NDArray[np.float64]) -> float:
    # Correctness floor — same math, no acceleration.
    z = np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))
    return float(abs(z))


# Ordering for order_parameter — measured 2026-04-17 on the local
# Linux runner (Intel i5-11600K, Python 3.12). See
# ``docs/benchmarks/order_parameter_tiers.json`` for the raw samples
# and ``docs/pipeline_performance.md`` §"Multi-language accel chain"
# for the summary table.
#
#   N       Rust       Julia      Python
#   -----------------------------------------
#      4    1.13 µs   11.19 µs    6.22 µs
#     16    0.90      13.93       5.92
#    256    2.97      16.83      12.82
#   1024   13.12      21.62      26.60
#   4096   38.10      58.29     123.89
#  16384  256.50     275.80     465.78
#
# Rust wins at every measured N. Julia is SLOWER than Python for
# N <= 256 (juliacall FFI overhead dominates) but beats it from
# N >= 1024. The chain order Rust -> Julia -> Python is correct
# where Julia is available, because the dispatcher only falls
# through when Rust is unavailable AND the workload is large
# enough for Julia to help; for small N with Rust missing, Python
# is faster than Julia, and the user should either install the
# Rust wheel or accept a small per-call cost.
#
# Rerun ``python scripts/bench_order_parameter_tiers.py`` when
# this file is edited; the measurements above must stay in sync
# with the committed JSON artefact.
_ORDER_PARAMETER_CHAIN: list[tuple[str, Callable[[NDArray[np.float64]], float]]] = [
    ("rust", _rust_order_parameter),
    ("julia", _julia_order_parameter),
    ("python", _python_order_parameter),
]


def _rust_order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_grad = getattr(engine, "order_parameter_gradient", None)
    if not callable(rust_grad):
        raise ImportError("scpn_quantum_engine.order_parameter_gradient is unavailable")

    return np.asarray(rust_grad(np.ascontiguousarray(theta, dtype=np.float64)), dtype=np.float64)


def _julia_order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    from .julia import order_parameter_gradient as julia_grad

    return julia_grad(theta)


def _python_order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    # Correctness floor — analytic gradient of r = |<exp(i θ)>| with respect to
    # each phase, no acceleration. With C = <cos θ>, S = <sin θ> and r = hypot(C, S):
    #     ∂r/∂θ_j = (S cos θ_j - C sin θ_j) / (N r) = (1/N) sin(ψ - θ_j),
    # where ψ = atan2(S, C) is the mean phase. At the incoherent state r = 0 the mean
    # phase is undefined, so the gradient is the zero subgradient there.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros(count, dtype=np.float64)
    gradient = (sin_mean * np.cos(phases) - cos_mean * np.sin(phases)) / (count * magnitude)
    return np.ascontiguousarray(gradient, dtype=np.float64)


# Ordering for order_parameter_gradient mirrors the order_parameter value chain
# (Rust -> Julia -> Python floor); the gradient touches the same per-oscillator
# trigonometric work, so the measured value ordering carries over. The dedicated
# gradient micro-benchmark is recorded in
# ``docs/benchmarks/order_parameter_gradient_tiers.json`` and summarised in
# ``docs/pipeline_performance.md``. Rerun
# ``python scripts/bench_order_parameter_gradient_tiers.py`` when this chain is edited.
_ORDER_PARAMETER_GRADIENT_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_order_parameter_gradient),
    ("julia", _julia_order_parameter_gradient),
    ("python", _python_order_parameter_gradient),
]


def _rust_order_parameter_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_hessian = getattr(engine, "order_parameter_hessian", None)
    if not callable(rust_hessian):
        raise ImportError("scpn_quantum_engine.order_parameter_hessian is unavailable")

    return np.asarray(
        rust_hessian(np.ascontiguousarray(theta, dtype=np.float64)), dtype=np.float64
    )


def _julia_order_parameter_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    from .julia import order_parameter_hessian as julia_hessian

    return julia_hessian(theta)


def _python_order_parameter_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    # Correctness floor — analytic Hessian of r = |<exp(i θ)>|, no acceleration. With
    # C = <cos θ>, S = <sin θ>, r = hypot(C, S), mean phase ψ = atan2(S, C), and the
    # alignment a_j = cos(ψ − θ_j) = (C cos θ_j + S sin θ_j) / r:
    #     ∂²r/∂θ_i∂θ_j = a_i a_j / (N² r) − δ_ij a_j / N.
    # The matrix is symmetric and each row sums to zero (a global phase shift leaves r
    # invariant). At the incoherent state r = 0 the zero matrix is returned.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros((count, count), dtype=np.float64)
    aligned = (cos_mean * np.cos(phases) + sin_mean * np.sin(phases)) / magnitude
    hessian = np.outer(aligned, aligned) / (count * count * magnitude)
    hessian -= np.diag(aligned / count)
    return np.ascontiguousarray(hessian, dtype=np.float64)


# The Hessian chain mirrors the order_parameter value and gradient chains
# (Rust → Julia → Python floor); it reuses the same per-oscillator trigonometric
# work plus a rank-one outer product. Its micro-benchmark is recorded in
# ``docs/benchmarks/order_parameter_hessian_tiers.json``; rerun
# ``python scripts/bench_order_parameter_hessian_tiers.py`` when this chain is edited.
_ORDER_PARAMETER_HESSIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_order_parameter_hessian),
    ("julia", _julia_order_parameter_hessian),
    ("python", _python_order_parameter_hessian),
]


_order_parameter_dispatcher = MultiLangDispatcher(_ORDER_PARAMETER_CHAIN)
_order_parameter_gradient_dispatcher = MultiLangDispatcher(_ORDER_PARAMETER_GRADIENT_CHAIN)
_order_parameter_hessian_dispatcher = MultiLangDispatcher(_ORDER_PARAMETER_HESSIAN_CHAIN)


def order_parameter(theta: NDArray[np.float64]) -> float:
    """Kuramoto order parameter with multi-language dispatch.

    Chain (measured fastest first): Rust → Julia → Python floor.
    The served tier is recorded on :data:`last_tier_used`.
    """
    return float(_order_parameter_dispatcher(theta))


def order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Gradient of the Kuramoto order parameter with multi-language dispatch.

    Returns :math:`\partial r / \partial \theta_j` for the order parameter
    :math:`r = |\langle e^{i\theta} \rangle|`, where each component is the
    synchronisation force :math:`(1/N)\sin(\psi - \theta_j)` pulling oscillator
    ``j`` towards the mean phase :math:`\psi`. At the incoherent state
    (:math:`r = 0`) the mean phase is undefined and the zero subgradient is
    returned.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of the same length as ``theta`` holding
        the per-phase gradient. An empty input yields an empty array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served
    tier is recorded on :func:`last_gradient_tier_used`.
    """
    return np.asarray(_order_parameter_gradient_dispatcher(theta), dtype=np.float64)


def last_tier_used() -> str | None:
    """Return the tier that served the most recent ``order_parameter``."""
    return _order_parameter_dispatcher.last_tier


def order_parameter_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Hessian of the Kuramoto order parameter with multi-language dispatch.

    Returns the second-derivative matrix :math:`\partial^2 r / \partial \theta_i
    \partial \theta_j` for the order parameter :math:`r = |\langle e^{i\theta}
    \rangle|`. With alignment :math:`a_j = \cos(\psi - \theta_j)` and mean phase
    :math:`\psi`, the entries are :math:`a_i a_j / (N^2 r) - \delta_{ij} a_j / N`.
    The matrix is symmetric and every row sums to zero (a global phase shift leaves
    ``r`` invariant). At the incoherent state (:math:`r = 0`) the zero matrix is
    returned, mirroring the gradient's zero subgradient.

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
    recorded on :func:`last_hessian_tier_used`.
    """
    return np.asarray(_order_parameter_hessian_dispatcher(theta), dtype=np.float64)


def last_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``order_parameter_gradient``."""
    return _order_parameter_gradient_dispatcher.last_tier


def last_hessian_tier_used() -> str | None:
    """Return the tier that served the most recent ``order_parameter_hessian``."""
    return _order_parameter_hessian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("order_parameter", _order_parameter_dispatcher)
register_dispatcher("order_parameter_gradient", _order_parameter_gradient_dispatcher)
register_dispatcher("order_parameter_hessian", _order_parameter_hessian_dispatcher)


__all__ = [
    "last_gradient_tier_used",
    "last_hessian_tier_used",
    "last_tier_used",
    "order_parameter",
    "order_parameter_gradient",
    "order_parameter_hessian",
]
