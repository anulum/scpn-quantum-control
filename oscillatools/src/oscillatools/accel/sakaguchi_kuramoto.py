# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto–Sakaguchi frustrated force and stability Jacobian
"""Kuramoto–Sakaguchi frustrated coupling force and its stability Jacobian.

The Sakaguchi generalisation adds a phase-frustration angle ``α`` to the coupling:
``F_j = Σ_{k≠j} K_jk sin(θ_k − θ_j − α)``. The frustration breaks the reciprocity of the
interaction — for ``α ≠ 0`` the Jacobian is not symmetric even for symmetric ``K``, so the
dynamics is non-variational (it has no energy/Lyapunov function) and supports chimera states
and travelling waves. Its Jacobian is ``J_jl = K_jl cos(θ_l − θ_j − α)`` for ``l ≠ j`` with
diagonal ``J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j − α)``; every row still sums to zero (the
global-phase Goldstone mode survives), and the self-coupling ``k = j`` term is excluded. For
``α = 0`` the force and Jacobian reduce to the networked-Kuramoto force and Jacobian.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~oscillatools.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _validate_coupling(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta, coupling)`` after square-shape validation.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix whose order matches ``theta``.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = phases.size
    if matrix.shape != (count, count):
        raise ValueError(
            f"coupling must be a square matrix of order {count}, got shape {matrix.shape}"
        )
    return phases, matrix


def _rust_sakaguchi_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_force = getattr(engine, "sakaguchi_force", None)
    if not callable(rust_force):
        raise ImportError("scpn_quantum_engine.sakaguchi_force is unavailable")

    return np.asarray(rust_force(phases, matrix, float(frustration)), dtype=np.float64)


def _julia_sakaguchi_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import sakaguchi_force as julia_force

    return julia_force(phases, matrix, frustration)


def _python_sakaguchi_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    # Correctness floor — F_j = Σ_{k≠j} K_jk sin(θ_k − θ_j − α). The self-coupling term is
    # excluded; the empty input yields an empty array. For α = 0 this is the networked force.
    phases, matrix = _validate_coupling(theta, coupling)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    difference = phases[None, :] - phases[:, None] - frustration
    contributions = matrix * np.sin(difference)
    np.fill_diagonal(contributions, 0.0)
    return np.ascontiguousarray(contributions.sum(axis=1), dtype=np.float64)


_SAKAGUCHI_FORCE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64], float], NDArray[np.float64]]]
] = [
    ("rust", _rust_sakaguchi_force),
    ("julia", _julia_sakaguchi_force),
    ("python", _python_sakaguchi_force),
]


def _rust_sakaguchi_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "sakaguchi_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.sakaguchi_jacobian is unavailable")

    return np.asarray(rust_jacobian(phases, matrix, float(frustration)), dtype=np.float64)


def _julia_sakaguchi_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import sakaguchi_jacobian as julia_jacobian

    return julia_jacobian(phases, matrix, frustration)


def _python_sakaguchi_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    # Correctness floor — J_jl = K_jl cos(θ_l − θ_j − α) for l ≠ j, with diagonal
    # J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j − α). Every row sums to zero; the matrix is
    # asymmetric for α ≠ 0 even when K is symmetric. The empty input yields a ``(0, 0)`` array.
    phases, matrix = _validate_coupling(theta, coupling)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    difference = phases[None, :] - phases[:, None] - frustration
    off_diagonal = matrix * np.cos(difference)
    np.fill_diagonal(off_diagonal, 0.0)
    jacobian = off_diagonal.copy()
    np.fill_diagonal(jacobian, -off_diagonal.sum(axis=1))
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the force chain (Rust → Julia → Python floor). Their
# micro-benchmark (on a dense random coupling matrix at α = 0.3) is recorded in
# ``docs/benchmarks/sakaguchi_kuramoto_tiers.json``; rerun
# ``python scripts/bench_sakaguchi_kuramoto_tiers.py`` when these chains are edited.
_SAKAGUCHI_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64], float], NDArray[np.float64]]]
] = [
    ("rust", _rust_sakaguchi_jacobian),
    ("julia", _julia_sakaguchi_jacobian),
    ("python", _python_sakaguchi_jacobian),
]


_sakaguchi_force_dispatcher = MultiLangDispatcher(_SAKAGUCHI_FORCE_CHAIN)
_sakaguchi_jacobian_dispatcher = MultiLangDispatcher(_SAKAGUCHI_JACOBIAN_CHAIN)


def sakaguchi_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    r"""Kuramoto–Sakaguchi frustrated coupling force with multi-language dispatch.

    Returns :math:`F_j = \sum_{k \neq j} K_{jk} \sin(\theta_k - \theta_j - \alpha)`, the
    phase-frustrated Kuramoto interaction for a coupling matrix ``K`` and frustration angle
    :math:`\alpha`. For :math:`\alpha = 0` it equals
    :func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_force`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    frustration : float
        The phase-frustration angle ``α`` in radians (any real value).

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of length ``N``. An empty input yields an
        empty array.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_sakaguchi_force_tier_used`.
    """
    return np.asarray(_sakaguchi_force_dispatcher(theta, coupling, frustration), dtype=np.float64)


def sakaguchi_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    r"""Kuramoto–Sakaguchi stability Jacobian with multi-language dispatch.

    Returns the linearisation :math:`J_{jl} = K_{jl}\cos(\theta_l - \theta_j - \alpha)` for
    :math:`l \neq j` with diagonal :math:`J_{jj} = -\sum_{k \neq j} K_{jk}\cos(\theta_k -
    \theta_j - \alpha)`. Every row sums to zero (the global-phase Goldstone mode), but for
    :math:`\alpha \neq 0` the matrix is asymmetric even for symmetric ``K`` — the frustration
    makes the dynamics non-variational. For :math:`\alpha = 0` it equals
    :func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_jacobian`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    frustration : float
        The phase-frustration angle ``α`` in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. An empty input yields a
        ``(0, 0)`` array.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_sakaguchi_jacobian_tier_used`.
    """
    return np.asarray(
        _sakaguchi_jacobian_dispatcher(theta, coupling, frustration), dtype=np.float64
    )


def last_sakaguchi_force_tier_used() -> str | None:
    """Return the tier that served the most recent ``sakaguchi_force``."""
    return _sakaguchi_force_dispatcher.last_tier


def last_sakaguchi_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``sakaguchi_jacobian``."""
    return _sakaguchi_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("sakaguchi_force", _sakaguchi_force_dispatcher)
register_dispatcher("sakaguchi_jacobian", _sakaguchi_jacobian_dispatcher)


__all__ = [
    "last_sakaguchi_force_tier_used",
    "last_sakaguchi_jacobian_tier_used",
    "sakaguchi_force",
    "sakaguchi_jacobian",
]
