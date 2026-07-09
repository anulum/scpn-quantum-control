# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Networked Kuramoto coupling force and stability Jacobian
"""Networked Kuramoto coupling force and its synchronisation stability Jacobian.

For a coupling matrix ``K`` (the SCPN ``K_nm`` weights) the phase-coupling force on
oscillator ``j`` is ``F_j = Σ_k K_jk sin(θ_k − θ_j)``, the general (graph) form of the
Kuramoto interaction whose all-to-all special case ``K_jk = c/N`` is the mean field. Its
Jacobian ``J_jl = K_jl cos(θ_l − θ_j)`` for ``l ≠ j`` with diagonal
``J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j)`` linearises the dynamics about a phase
configuration: every row sums to zero (the global-phase Goldstone mode) and the matrix is
symmetric whenever ``K`` is. Because each ``k = j`` term of the force is identically zero
(``sin 0 = 0``), both the force and the Jacobian are independent of the diagonal of ``K``.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~oscillatools.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher
from .tensor_io import as_float64_array, restore_array, tensor_template


def _validate_coupling(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta, coupling)`` after shape validation.

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


def _rust_networked_kuramoto_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_force = getattr(engine, "networked_kuramoto_force", None)
    if not callable(rust_force):
        raise ImportError("scpn_quantum_engine.networked_kuramoto_force is unavailable")

    return np.asarray(rust_force(phases, matrix), dtype=np.float64)


def _julia_networked_kuramoto_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import networked_kuramoto_force as julia_force

    return julia_force(phases, matrix)


def _python_networked_kuramoto_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — F_j = Σ_k K_jk sin(θ_k − θ_j). The k = j term is sin(0) = 0,
    # so the force is independent of the diagonal of K. The empty input yields an empty array.
    phases, matrix = _validate_coupling(theta, coupling)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    difference = phases[None, :] - phases[:, None]
    force = np.einsum("jk,jk->j", matrix, np.sin(difference))
    return np.ascontiguousarray(force, dtype=np.float64)


_NETWORKED_KURAMOTO_FORCE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_networked_kuramoto_force),
    ("julia", _julia_networked_kuramoto_force),
    ("python", _python_networked_kuramoto_force),
]


def _rust_networked_kuramoto_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "networked_kuramoto_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.networked_kuramoto_jacobian is unavailable")

    return np.asarray(rust_jacobian(phases, matrix), dtype=np.float64)


def _julia_networked_kuramoto_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import networked_kuramoto_jacobian as julia_jacobian

    return julia_jacobian(phases, matrix)


def _python_networked_kuramoto_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — J_jl = K_jl cos(θ_l − θ_j) for l ≠ j, with diagonal
    # J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j). The k = j term is excluded (the force is
    # independent of the diagonal of K). Symmetric when K is; every row sums to zero.
    phases, matrix = _validate_coupling(theta, coupling)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    difference = phases[None, :] - phases[:, None]
    off_diagonal = matrix * np.cos(difference)
    np.fill_diagonal(off_diagonal, 0.0)
    jacobian = off_diagonal.copy()
    np.fill_diagonal(jacobian, -off_diagonal.sum(axis=1))
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the force chain (Rust → Julia → Python floor). Their
# micro-benchmark (on a dense random coupling matrix) is recorded in
# ``docs/benchmarks/networked_kuramoto_tiers.json``; rerun
# ``python scripts/bench_networked_kuramoto_tiers.py`` when these chains are edited.
_NETWORKED_KURAMOTO_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_networked_kuramoto_jacobian),
    ("julia", _julia_networked_kuramoto_jacobian),
    ("python", _python_networked_kuramoto_jacobian),
]


_networked_kuramoto_force_dispatcher = MultiLangDispatcher(_NETWORKED_KURAMOTO_FORCE_CHAIN)
_networked_kuramoto_jacobian_dispatcher = MultiLangDispatcher(_NETWORKED_KURAMOTO_JACOBIAN_CHAIN)


def networked_kuramoto_force(theta: object, coupling: object) -> NDArray[np.float64] | Any:
    r"""Networked Kuramoto coupling force with multi-language dispatch.

    Returns :math:`F_j = \sum_k K_{jk} \sin(\theta_k - \theta_j)`, the general (graph) form
    of the Kuramoto interaction for a coupling matrix ``K``. The all-to-all special case
    :math:`K_{jk} = c/N` recovers
    :func:`~oscillatools.accel.kuramoto_mean_field.mean_field_force`. Each
    :math:`k = j` term is :math:`\sin 0 = 0`, so the force is independent of the diagonal
    of ``K``.

    Parameters
    ----------
    theta : array-like
        One-dimensional array or optional Torch/JAX tensor of ``N`` oscillator
        phases in radians.
    coupling : array-like
        Two-dimensional ``(N, N)`` coupling matrix ``K``. Torch/JAX tensors are
        accepted without making either backend a required dependency.

    Returns
    -------
    numpy.ndarray or tensor
        One-dimensional float64 force array of length ``N``. When ``theta`` or
        ``coupling`` is a Torch/JAX tensor, the result is restored to that first
        detected tensor namespace. An empty input yields an empty array.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_networked_kuramoto_force_tier_used`.
    """
    template = tensor_template(theta, coupling)
    result = np.asarray(
        _networked_kuramoto_force_dispatcher(
            as_float64_array(theta),
            as_float64_array(coupling),
        ),
        dtype=np.float64,
    )
    return restore_array(result, template)


def networked_kuramoto_jacobian(theta: object, coupling: object) -> NDArray[np.float64] | Any:
    r"""Networked Kuramoto stability Jacobian with multi-language dispatch.

    Returns the linearisation :math:`J_{jl} = K_{jl}\cos(\theta_l - \theta_j)` for
    :math:`l \neq j` with diagonal :math:`J_{jj} = -\sum_{k \neq j} K_{jk}\cos(\theta_k -
    \theta_j)`. Every row sums to zero (the global-phase Goldstone mode) and the matrix is
    symmetric whenever ``K`` is. Like the force it is independent of the diagonal of ``K``;
    its spectrum classifies the linear stability of synchronisation on the network.

    Parameters
    ----------
    theta : array-like
        One-dimensional array or optional Torch/JAX tensor of ``N`` oscillator
        phases in radians.
    coupling : array-like
        Two-dimensional ``(N, N)`` coupling matrix ``K``. Torch/JAX tensors are
        accepted without making either backend a required dependency.

    Returns
    -------
    numpy.ndarray or tensor
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. When ``theta`` or
        ``coupling`` is a Torch/JAX tensor, the result is restored to that first
        detected tensor namespace. An empty input yields a ``(0, 0)`` array.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_networked_kuramoto_jacobian_tier_used`.
    """
    template = tensor_template(theta, coupling)
    result = np.asarray(
        _networked_kuramoto_jacobian_dispatcher(
            as_float64_array(theta),
            as_float64_array(coupling),
        ),
        dtype=np.float64,
    )
    return restore_array(result, template)


def last_networked_kuramoto_force_tier_used() -> str | None:
    """Return the tier that served the most recent ``networked_kuramoto_force``."""
    return _networked_kuramoto_force_dispatcher.last_tier


def last_networked_kuramoto_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``networked_kuramoto_jacobian``."""
    return _networked_kuramoto_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("networked_kuramoto_force", _networked_kuramoto_force_dispatcher)
register_dispatcher("networked_kuramoto_jacobian", _networked_kuramoto_jacobian_dispatcher)


__all__ = [
    "last_networked_kuramoto_force_tier_used",
    "last_networked_kuramoto_jacobian_tier_used",
    "networked_kuramoto_force",
    "networked_kuramoto_jacobian",
]
