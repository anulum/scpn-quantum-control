# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Daido higher-order order-parameter observables
"""Daido higher-order order-parameter observables.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~oscillatools.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _validate_harmonic(m: int) -> None:
    """Reject non-positive Daido harmonic orders."""
    if m < 1:
        raise ValueError(f"Daido harmonic order m must be a positive integer, got {m}")


def _rust_daido_order_parameter(theta: NDArray[np.float64], m: int) -> float:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_daido = getattr(engine, "daido_order_parameter", None)
    if not callable(rust_daido):
        raise ImportError("scpn_quantum_engine.daido_order_parameter is unavailable")

    return float(rust_daido(np.ascontiguousarray(theta, dtype=np.float64), m))


def _julia_daido_order_parameter(theta: NDArray[np.float64], m: int) -> float:
    _validate_harmonic(m)
    from .julia import daido_order_parameter as julia_daido

    return julia_daido(theta, m)


def _python_daido_order_parameter(theta: NDArray[np.float64], m: int) -> float:
    # Correctness floor — the m-th Daido order parameter r_m = |<exp(i m θ)>|, which
    # detects m-cluster synchronisation. The empty input maps to 0.0.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    return float(abs(np.mean(np.exp(1j * m * phases))))


def _rust_daido_order_parameter_gradient(
    theta: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_grad = getattr(engine, "daido_order_parameter_gradient", None)
    if not callable(rust_grad):
        raise ImportError("scpn_quantum_engine.daido_order_parameter_gradient is unavailable")

    return np.asarray(
        rust_grad(np.ascontiguousarray(theta, dtype=np.float64), m), dtype=np.float64
    )


def _julia_daido_order_parameter_gradient(
    theta: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    from .julia import daido_order_parameter_gradient as julia_grad

    return julia_grad(theta, m)


def _python_daido_order_parameter_gradient(
    theta: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    # Correctness floor — gradient of the m-th Daido order parameter. With
    # C_m = <cos(m θ)>, S_m = <sin(m θ)> and r_m = hypot(C_m, S_m):
    #     ∂r_m/∂θ_j = (m/N) sin(ψ_m − m θ_j) = (m / (N r_m)) (S_m cos(m θ_j) − C_m sin(m θ_j)),
    # where ψ_m = atan2(S_m, C_m). The components sum to zero (a global phase shift leaves
    # r_m invariant). The incoherent state r_m = 0 returns the zero subgradient. For m = 1
    # this reduces exactly to ``order_parameter_gradient``.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    scaled = m * phases
    cos_mean = float(np.mean(np.cos(scaled)))
    sin_mean = float(np.mean(np.sin(scaled)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros(count, dtype=np.float64)
    gradient = (m / (count * magnitude)) * (sin_mean * np.cos(scaled) - cos_mean * np.sin(scaled))
    return np.ascontiguousarray(gradient, dtype=np.float64)


# The Daido value and gradient chains mirror the order-parameter chains (Rust → Julia →
# Python floor) at the m-th harmonic. Their micro-benchmark (at m = 2) is recorded in
# ``docs/benchmarks/daido_order_parameter_tiers.json``; rerun
# ``python scripts/bench_daido_order_parameter_tiers.py`` when these chains are edited.
_DAIDO_ORDER_PARAMETER_CHAIN: list[tuple[str, Callable[[NDArray[np.float64], int], float]]] = [
    ("rust", _rust_daido_order_parameter),
    ("julia", _julia_daido_order_parameter),
    ("python", _python_daido_order_parameter),
]
_DAIDO_ORDER_PARAMETER_GRADIENT_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], int], NDArray[np.float64]]]
] = [
    ("rust", _rust_daido_order_parameter_gradient),
    ("julia", _julia_daido_order_parameter_gradient),
    ("python", _python_daido_order_parameter_gradient),
]


def _rust_daido_order_parameter_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_hessian = getattr(engine, "daido_order_parameter_hessian", None)
    if not callable(rust_hessian):
        raise ImportError("scpn_quantum_engine.daido_order_parameter_hessian is unavailable")

    return np.asarray(
        rust_hessian(np.ascontiguousarray(theta, dtype=np.float64), m), dtype=np.float64
    )


def _julia_daido_order_parameter_hessian(
    theta: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    from .julia import daido_order_parameter_hessian as julia_hessian

    return julia_hessian(theta, m)


def _python_daido_order_parameter_hessian(
    theta: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    # Correctness floor — Hessian of the m-th Daido order parameter. With C_m = <cos(m θ)>,
    # S_m = <sin(m θ)>, r_m = hypot(C_m, S_m) and a_k = cos(ψ_m − m θ_k) =
    # (C_m cos(m θ_k) + S_m sin(m θ_k)) / r_m:
    #     ∂²r_m/∂θ_i∂θ_j = m² (a_i a_j / (N² r_m) − δ_ij a_j / N).
    # The matrix is symmetric and each row sums to zero. The incoherent state r_m = 0
    # returns the zero matrix. For m = 1 this reduces to ``order_parameter_hessian``.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    scaled = m * phases
    cos_mean = float(np.mean(np.cos(scaled)))
    sin_mean = float(np.mean(np.sin(scaled)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros((count, count), dtype=np.float64)
    aligned = (cos_mean * np.cos(scaled) + sin_mean * np.sin(scaled)) / magnitude
    hessian = (m * m) * (
        np.outer(aligned, aligned) / (count * count * magnitude) - np.diag(aligned / count)
    )
    return np.ascontiguousarray(hessian, dtype=np.float64)


# The Daido Hessian chain mirrors the order-parameter Hessian chain at the m-th harmonic.
# Its micro-benchmark (at m = 2) is recorded in
# ``docs/benchmarks/daido_order_parameter_hessian_tiers.json``; rerun
# ``python scripts/bench_daido_order_parameter_hessian_tiers.py`` when this chain is edited.
_DAIDO_ORDER_PARAMETER_HESSIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], int], NDArray[np.float64]]]
] = [
    ("rust", _rust_daido_order_parameter_hessian),
    ("julia", _julia_daido_order_parameter_hessian),
    ("python", _python_daido_order_parameter_hessian),
]


_daido_order_parameter_dispatcher = MultiLangDispatcher(_DAIDO_ORDER_PARAMETER_CHAIN)
_daido_order_parameter_gradient_dispatcher = MultiLangDispatcher(
    _DAIDO_ORDER_PARAMETER_GRADIENT_CHAIN
)
_daido_order_parameter_hessian_dispatcher = MultiLangDispatcher(
    _DAIDO_ORDER_PARAMETER_HESSIAN_CHAIN
)


def daido_order_parameter(theta: NDArray[np.float64], m: int) -> float:
    r"""m-th Daido order parameter with multi-language dispatch.

    Returns :math:`r_m = |\langle e^{i m \theta} \rangle|`, the magnitude of the m-th
    Fourier mode of the phase distribution. It detects m-cluster synchronisation: a
    state split into ``m`` evenly spaced clusters has :math:`r_1 = 0` but
    :math:`r_m = 1`. For :math:`m = 1` it equals :func:`order_parameter`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    float
        The Daido order parameter in ``[0, 1]``. An empty input returns ``0.0``.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_tier_used`.
    """
    return float(_daido_order_parameter_dispatcher(theta, m))


def daido_order_parameter_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    r"""Gradient of the m-th Daido order parameter with multi-language dispatch.

    Returns :math:`\partial r_m / \partial \theta_j = (m/N) \sin(\psi_m - m \theta_j)`,
    where :math:`\psi_m` is the m-th mode phase. The components sum to zero (a global
    phase shift leaves :math:`r_m` invariant). At the incoherent state
    (:math:`r_m = 0`) the zero subgradient is returned.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of the same length as ``theta``. An empty input
        yields an empty array.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_gradient_tier_used`.
    """
    return np.asarray(_daido_order_parameter_gradient_dispatcher(theta, m), dtype=np.float64)


def last_daido_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_order_parameter``."""
    return _daido_order_parameter_dispatcher.last_tier


def last_daido_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_order_parameter_gradient``."""
    return _daido_order_parameter_gradient_dispatcher.last_tier


def daido_order_parameter_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    r"""Hessian of the m-th Daido order parameter with multi-language dispatch.

    Returns :math:`\partial^2 r_m / \partial \theta_i \partial \theta_j = m^2 (a_i a_j /
    (N^2 r_m) - \delta_{ij} a_j / N)`, with :math:`a_k = \cos(\psi_m - m \theta_k)`. The
    matrix is symmetric and every row sums to zero. At the incoherent state
    (:math:`r_m = 0`) the zero matrix is returned. For :math:`m = 1` it reduces to
    :func:`~oscillatools.accel.order_parameter_observables.order_parameter_hessian`.

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
    recorded on :func:`last_daido_hessian_tier_used`.
    """
    return np.asarray(_daido_order_parameter_hessian_dispatcher(theta, m), dtype=np.float64)


def last_daido_hessian_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_order_parameter_hessian``."""
    return _daido_order_parameter_hessian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("daido_order_parameter", _daido_order_parameter_dispatcher)
register_dispatcher("daido_order_parameter_gradient", _daido_order_parameter_gradient_dispatcher)
register_dispatcher("daido_order_parameter_hessian", _daido_order_parameter_hessian_dispatcher)


__all__ = [
    "daido_order_parameter",
    "daido_order_parameter_gradient",
    "daido_order_parameter_hessian",
    "last_daido_gradient_tier_used",
    "last_daido_hessian_tier_used",
    "last_daido_tier_used",
]
