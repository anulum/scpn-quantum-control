# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Network-local Kuramoto mean phase and its Jacobian
"""Network-local Kuramoto mean phase and its Jacobian.

For a non-negative adjacency/coupling matrix ``A`` the local mean phase of oscillator ``j`` is
``ψ_j = atan2(Σ_k A_jk sin θ_k, Σ_k A_jk cos θ_k)`` — the argument of the local complex order
``Z_j = Σ_k A_jk e^{iθ_k}``. It is the phase partner of the network-local order parameter
``r_j = |Z_j| / Σ_k A_jk``: together they give the local complex order ``Z_j / d_j =
r_j e^{iψ_j}``. For the all-to-all uniform adjacency it reduces to the global Kuramoto mean
phase. Its Jacobian is ``∂ψ_j/∂θ_l = A_jl cos(ψ_j − θ_l) / |Z_j|`` — a node with zero degree
or an incoherent neighbourhood (``|Z_j| = 0``) contributes a zero phase and a zero subgradient
row.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~oscillatools.accel.dispatcher.MultiLangDispatcher`.
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


def _rust_local_mean_phase(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_local = getattr(engine, "local_mean_phase", None)
    if not callable(rust_local):
        raise ImportError("scpn_quantum_engine.local_mean_phase is unavailable")

    return np.asarray(rust_local(phases, matrix), dtype=np.float64)


def _julia_local_mean_phase(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    from .julia import local_mean_phase as julia_local

    return julia_local(phases, matrix)


def _python_local_mean_phase(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — ψ_j = atan2(Σ_k A_jk sin θ_k, Σ_k A_jk cos θ_k). A zero-degree node or
    # an incoherent neighbourhood (|Z_j| = 0) yields ψ_j = 0.
    phases, matrix = _validate_adjacency(theta, adjacency)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    cos_sum = matrix @ np.cos(phases)
    sin_sum = matrix @ np.sin(phases)
    out = np.zeros(count, dtype=np.float64)
    coherent = np.hypot(cos_sum, sin_sum) != 0.0
    out[coherent] = np.arctan2(sin_sum[coherent], cos_sum[coherent])
    return np.ascontiguousarray(out, dtype=np.float64)


_LOCAL_MEAN_PHASE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_local_mean_phase),
    ("julia", _julia_local_mean_phase),
    ("python", _python_local_mean_phase),
]


def _rust_local_mean_phase_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "local_mean_phase_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.local_mean_phase_jacobian is unavailable")

    return np.asarray(rust_jacobian(phases, matrix), dtype=np.float64)


def _julia_local_mean_phase_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_adjacency(theta, adjacency)
    from .julia import local_mean_phase_jacobian as julia_jacobian

    return julia_jacobian(phases, matrix)


def _python_local_mean_phase_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — ∂ψ_j/∂θ_l = A_jl cos(ψ_j − θ_l) / |Z_j| =
    # (A_jl / |Z_j|²) (C_j cos θ_l + S_j sin θ_l), with Z_j = C_j + i S_j. A zero-degree node or
    # an incoherent neighbourhood (|Z_j| = 0) yields a zero subgradient row.
    phases, matrix = _validate_adjacency(theta, adjacency)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_sum = matrix @ np.cos(phases)
    sin_sum = matrix @ np.sin(phases)
    magnitude_squared = cos_sum * cos_sum + sin_sum * sin_sum
    inverse = np.zeros(count, dtype=np.float64)
    coherent = magnitude_squared != 0.0
    inverse[coherent] = 1.0 / magnitude_squared[coherent]
    aligned = (
        cos_sum[:, None] * np.cos(phases)[None, :] + sin_sum[:, None] * np.sin(phases)[None, :]
    )
    jacobian = matrix * inverse[:, None] * aligned
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the value chain (Rust → Julia → Python floor). Their
# micro-benchmark (on a dense random adjacency matrix) is recorded in
# ``docs/benchmarks/local_mean_phase_tiers.json``; rerun
# ``python scripts/bench_local_mean_phase_tiers.py`` when these chains are edited.
_LOCAL_MEAN_PHASE_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_local_mean_phase_jacobian),
    ("julia", _julia_local_mean_phase_jacobian),
    ("python", _python_local_mean_phase_jacobian),
]


_local_mean_phase_dispatcher = MultiLangDispatcher(_LOCAL_MEAN_PHASE_CHAIN)
_local_mean_phase_jacobian_dispatcher = MultiLangDispatcher(_LOCAL_MEAN_PHASE_JACOBIAN_CHAIN)


def local_mean_phase(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Network-local Kuramoto mean phase with multi-language dispatch.

    Returns :math:`\psi_j = \operatorname{atan2}(\sum_k A_{jk}\sin\theta_k, \sum_k
    A_{jk}\cos\theta_k)`, the argument of the local complex order
    :math:`Z_j = \sum_k A_{jk} e^{i\theta_k}` — the phase partner of
    :func:`~oscillatools.accel.local_order.local_order_parameter`. For the all-to-all
    uniform adjacency it equals the global
    :func:`~oscillatools.accel.mean_phase_observables.mean_phase`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency/coupling matrix ``A``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of ``N`` local mean phases in ``(−π, π]``. A zero-degree
        node or an incoherent neighbourhood yields ``0``; an empty input yields an empty array.

    Raises
    ------
    ValueError
        If ``adjacency`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_local_mean_phase_tier_used`.
    """
    return np.asarray(_local_mean_phase_dispatcher(theta, adjacency), dtype=np.float64)


def local_mean_phase_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Jacobian of the network-local Kuramoto mean phase with multi-language dispatch.

    Returns :math:`\partial \psi_j / \partial \theta_l = A_{jl}\cos(\psi_j - \theta_l)/|Z_j|`,
    with the local complex order :math:`Z_j = \sum_k A_{jk} e^{i\theta_k}`. A zero-degree node
    or an incoherent neighbourhood yields a zero subgradient row.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency/coupling matrix ``A``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix (sparse in ``A``'s pattern). An
        empty input yields a ``(0, 0)`` array.

    Raises
    ------
    ValueError
        If ``adjacency`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_local_mean_phase_jacobian_tier_used`.
    """
    return np.asarray(_local_mean_phase_jacobian_dispatcher(theta, adjacency), dtype=np.float64)


def last_local_mean_phase_tier_used() -> str | None:
    """Return the tier that served the most recent ``local_mean_phase``."""
    return _local_mean_phase_dispatcher.last_tier


def last_local_mean_phase_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``local_mean_phase_jacobian``."""
    return _local_mean_phase_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("local_mean_phase", _local_mean_phase_dispatcher)
register_dispatcher("local_mean_phase_jacobian", _local_mean_phase_jacobian_dispatcher)


__all__ = [
    "last_local_mean_phase_jacobian_tier_used",
    "last_local_mean_phase_tier_used",
    "local_mean_phase",
    "local_mean_phase_jacobian",
]
