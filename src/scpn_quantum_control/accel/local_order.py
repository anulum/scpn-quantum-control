# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Network-local Kuramoto order parameter and its Jacobian
"""Network-local Kuramoto order parameter and its Jacobian.

For a non-negative adjacency/coupling matrix ``A`` the local order parameter of oscillator
``j`` is ``r_j = |Σ_k A_jk e^{iθ_k}| / Σ_k A_jk`` — the degree-normalised coherence of node
``j``'s neighbourhood. It measures local synchronisation and is the standard diagnostic for
chimera states (coexisting synchronous and incoherent regions). For the all-to-all uniform
adjacency it reduces to the global Kuramoto order parameter. Its Jacobian is
``∂r_j/∂θ_l = (A_jl / d_j) sin(ψ_j − θ_l)`` with local mean phase ``ψ_j`` and degree
``d_j = Σ_k A_jk``; a node with zero degree or an incoherent neighbourhood
(``|Σ_k A_jk e^{iθ_k}| = 0``) contributes a zero value and a zero subgradient row.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _validate_adjacency(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta, adjacency)`` after square-shape validation.

    Raises
    ------
    ValueError
        If ``adjacency`` is not a square matrix whose order matches ``theta``.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(adjacency, dtype=np.float64)
    count = phases.size
    if matrix.shape != (count, count):
        raise ValueError(
            f"adjacency must be a square matrix of order {count}, got shape {matrix.shape}"
        )
    return phases, matrix


def _rust_local_order_parameter(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_local = getattr(engine, "local_order_parameter", None)
    if not callable(rust_local):
        raise ImportError("scpn_quantum_engine.local_order_parameter is unavailable")

    return np.asarray(rust_local(phases, matrix), dtype=np.float64)


def _julia_local_order_parameter(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    from .julia import local_order_parameter as julia_local

    return julia_local(phases, matrix)


def _python_local_order_parameter(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — r_j = |Σ_k A_jk e^{iθ_k}| / Σ_k A_jk. A zero-degree node has r_j = 0.
    phases, matrix = _validate_adjacency(theta, adjacency)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    cos_sum = matrix @ np.cos(phases)
    sin_sum = matrix @ np.sin(phases)
    degree = matrix.sum(axis=1)
    magnitude = np.hypot(cos_sum, sin_sum)
    out = np.zeros(count, dtype=np.float64)
    nonzero = degree != 0.0
    out[nonzero] = magnitude[nonzero] / degree[nonzero]
    return np.ascontiguousarray(out, dtype=np.float64)


_LOCAL_ORDER_PARAMETER_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_local_order_parameter),
    ("julia", _julia_local_order_parameter),
    ("python", _python_local_order_parameter),
]


def _rust_local_order_parameter_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "local_order_parameter_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.local_order_parameter_jacobian is unavailable")

    return np.asarray(rust_jacobian(phases, matrix), dtype=np.float64)


def _julia_local_order_parameter_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    from .julia import local_order_parameter_jacobian as julia_jacobian

    return julia_jacobian(phases, matrix)


def _python_local_order_parameter_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — ∂r_j/∂θ_l = (A_jl / d_j) sin(ψ_j − θ_l) =
    # (A_jl / (d_j |Z_j|)) (S_j cos θ_l − C_j sin θ_l), with Z_j = C_j + i S_j. A zero-degree
    # node or an incoherent neighbourhood (|Z_j| = 0) yields a zero subgradient row.
    phases, matrix = _validate_adjacency(theta, adjacency)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_sum = matrix @ np.cos(phases)
    sin_sum = matrix @ np.sin(phases)
    degree = matrix.sum(axis=1)
    magnitude = np.hypot(cos_sum, sin_sum)
    denominator = degree * magnitude
    inverse = np.zeros(count, dtype=np.float64)
    nonzero = denominator != 0.0
    inverse[nonzero] = 1.0 / denominator[nonzero]
    aligned = (
        sin_sum[:, None] * np.cos(phases)[None, :] - cos_sum[:, None] * np.sin(phases)[None, :]
    )
    jacobian = matrix * inverse[:, None] * aligned
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the value chain (Rust → Julia → Python floor). Their
# micro-benchmark (on a dense random adjacency matrix) is recorded in
# ``docs/benchmarks/local_order_parameter_tiers.json``; rerun
# ``python scripts/bench_local_order_parameter_tiers.py`` when these chains are edited.
_LOCAL_ORDER_PARAMETER_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_local_order_parameter_jacobian),
    ("julia", _julia_local_order_parameter_jacobian),
    ("python", _python_local_order_parameter_jacobian),
]


_local_order_parameter_dispatcher = MultiLangDispatcher(_LOCAL_ORDER_PARAMETER_CHAIN)
_local_order_parameter_jacobian_dispatcher = MultiLangDispatcher(
    _LOCAL_ORDER_PARAMETER_JACOBIAN_CHAIN
)


def local_order_parameter(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Network-local Kuramoto order parameter with multi-language dispatch.

    Returns :math:`r_j = |\sum_k A_{jk} e^{i\theta_k}| / \sum_k A_{jk}`, the
    degree-normalised local coherence of each oscillator's neighbourhood — the standard
    diagnostic for chimera states. For the all-to-all uniform adjacency it equals the global
    :func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency/coupling matrix ``A``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of ``N`` local order parameters in ``[0, 1]``. A
        zero-degree node yields ``0``; an empty input yields an empty array.

    Raises
    ------
    ValueError
        If ``adjacency`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_local_order_parameter_tier_used`.
    """
    return np.asarray(_local_order_parameter_dispatcher(theta, adjacency), dtype=np.float64)


def local_order_parameter_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Jacobian of the network-local Kuramoto order parameter with multi-language dispatch.

    Returns :math:`\partial r_j / \partial \theta_l = (A_{jl}/d_j)\sin(\psi_j - \theta_l)`,
    with local mean phase :math:`\psi_j` and degree :math:`d_j = \sum_k A_{jk}`. A zero-degree
    node or an incoherent neighbourhood yields a zero subgradient row.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency/coupling matrix ``A``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. An empty input yields a
        ``(0, 0)`` array.

    Raises
    ------
    ValueError
        If ``adjacency`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_local_order_parameter_jacobian_tier_used`.
    """
    return np.asarray(
        _local_order_parameter_jacobian_dispatcher(theta, adjacency), dtype=np.float64
    )


def last_local_order_parameter_tier_used() -> str | None:
    """Return the tier that served the most recent ``local_order_parameter``."""
    return _local_order_parameter_dispatcher.last_tier


def last_local_order_parameter_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``local_order_parameter_jacobian``."""
    return _local_order_parameter_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("local_order_parameter", _local_order_parameter_dispatcher)
register_dispatcher("local_order_parameter_jacobian", _local_order_parameter_jacobian_dispatcher)


__all__ = [
    "last_local_order_parameter_jacobian_tier_used",
    "last_local_order_parameter_tier_used",
    "local_order_parameter",
    "local_order_parameter_jacobian",
]
