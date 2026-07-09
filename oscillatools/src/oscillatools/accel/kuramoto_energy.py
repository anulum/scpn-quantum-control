# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto interaction energy and its gradient
"""Kuramoto interaction energy (Hamiltonian / Lyapunov function) and its gradient.

For a coupling matrix ``K`` the interaction energy is
``E(θ) = −½ Σ_jk K_jk cos(θ_j − θ_k)`` — the potential whose gradient flow is the Kuramoto
dynamics: for symmetric ``K`` the phase velocity is ``θ̇ = −∇E``, so ``E`` is a Lyapunov
function that decreases monotonically towards synchronisation. Its gradient is
``∂E/∂θ_j = ½ Σ_k (K_jk + K_kj) sin(θ_j − θ_k)``, the symmetrised coupling force whose
components sum to zero (``E`` is invariant under a global phase shift). Because ``cos`` is
even, the energy depends only on the symmetric part of ``K``; for symmetric ``K`` the
gradient equals the negated networked-Kuramoto force.

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


def _rust_kuramoto_interaction_energy(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_energy = getattr(engine, "kuramoto_interaction_energy", None)
    if not callable(rust_energy):
        raise ImportError("scpn_quantum_engine.kuramoto_interaction_energy is unavailable")

    return float(rust_energy(phases, matrix))


def _julia_kuramoto_interaction_energy(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import kuramoto_interaction_energy as julia_energy

    return julia_energy(phases, matrix)


def _python_kuramoto_interaction_energy(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    # Correctness floor — E = −½ Σ_jk K_jk cos(θ_j − θ_k). The empty input has energy 0.
    phases, matrix = _validate_coupling(theta, coupling)
    if phases.size == 0:
        return 0.0
    difference = phases[:, None] - phases[None, :]
    return float(-0.5 * np.sum(matrix * np.cos(difference)))


_KURAMOTO_INTERACTION_ENERGY_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], float]]
] = [
    ("rust", _rust_kuramoto_interaction_energy),
    ("julia", _julia_kuramoto_interaction_energy),
    ("python", _python_kuramoto_interaction_energy),
]


def _rust_kuramoto_interaction_energy_gradient(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_gradient = getattr(engine, "kuramoto_interaction_energy_gradient", None)
    if not callable(rust_gradient):
        raise ImportError(
            "scpn_quantum_engine.kuramoto_interaction_energy_gradient is unavailable"
        )

    return np.asarray(rust_gradient(phases, matrix), dtype=np.float64)


def _julia_kuramoto_interaction_energy_gradient(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import kuramoto_interaction_energy_gradient as julia_gradient

    return julia_gradient(phases, matrix)


def _python_kuramoto_interaction_energy_gradient(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — ∂E/∂θ_j = ½ Σ_k (K_jk + K_kj) sin(θ_j − θ_k). The components sum
    # to zero (E is invariant under a global phase shift). The empty input yields an empty
    # array. For symmetric K this equals the negated networked-Kuramoto force.
    phases, matrix = _validate_coupling(theta, coupling)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    difference = phases[:, None] - phases[None, :]
    gradient = 0.5 * np.sum((matrix + matrix.T) * np.sin(difference), axis=1)
    return np.ascontiguousarray(gradient, dtype=np.float64)


# The gradient chain mirrors the energy chain (Rust → Julia → Python floor). Their
# micro-benchmark (on a dense random coupling matrix) is recorded in
# ``docs/benchmarks/kuramoto_energy_tiers.json``; rerun
# ``python scripts/bench_kuramoto_energy_tiers.py`` when these chains are edited.
_KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_kuramoto_interaction_energy_gradient),
    ("julia", _julia_kuramoto_interaction_energy_gradient),
    ("python", _python_kuramoto_interaction_energy_gradient),
]


def _rust_kuramoto_interaction_energy_hessian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_hessian = getattr(engine, "kuramoto_interaction_energy_hessian", None)
    if not callable(rust_hessian):
        raise ImportError("scpn_quantum_engine.kuramoto_interaction_energy_hessian is unavailable")

    return np.asarray(rust_hessian(phases, matrix), dtype=np.float64)


def _julia_kuramoto_interaction_energy_hessian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    phases, matrix = _validate_coupling(theta, coupling)
    from .julia import kuramoto_interaction_energy_hessian as julia_hessian

    return julia_hessian(phases, matrix)


def _python_kuramoto_interaction_energy_hessian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Correctness floor — H_il = −½(K_il + K_li) cos(θ_i − θ_l) for l ≠ i, with diagonal
    # H_ii = ½ Σ_{k≠i}(K_ik + K_ki) cos(θ_i − θ_k) = −Σ_{l≠i} H_il. The matrix is symmetric
    # and every row sums to zero (E is invariant under a global phase shift). For symmetric K
    # this equals the negated networked-Kuramoto Jacobian.
    phases, matrix = _validate_coupling(theta, coupling)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    symmetrised = matrix + matrix.T
    difference = phases[:, None] - phases[None, :]
    off_diagonal = -0.5 * symmetrised * np.cos(difference)
    np.fill_diagonal(off_diagonal, 0.0)
    hessian = off_diagonal.copy()
    np.fill_diagonal(hessian, -off_diagonal.sum(axis=1))
    return np.ascontiguousarray(hessian, dtype=np.float64)


# The Hessian chain mirrors the energy/gradient chains. Its micro-benchmark (on a dense
# random coupling matrix) is recorded in ``docs/benchmarks/kuramoto_energy_tiers.json``;
# rerun ``python scripts/bench_kuramoto_energy_tiers.py`` when this chain is edited.
_KURAMOTO_INTERACTION_ENERGY_HESSIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_kuramoto_interaction_energy_hessian),
    ("julia", _julia_kuramoto_interaction_energy_hessian),
    ("python", _python_kuramoto_interaction_energy_hessian),
]


_kuramoto_interaction_energy_dispatcher = MultiLangDispatcher(_KURAMOTO_INTERACTION_ENERGY_CHAIN)
_kuramoto_interaction_energy_gradient_dispatcher = MultiLangDispatcher(
    _KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN
)
_kuramoto_interaction_energy_hessian_dispatcher = MultiLangDispatcher(
    _KURAMOTO_INTERACTION_ENERGY_HESSIAN_CHAIN
)


def kuramoto_interaction_energy(theta: object, coupling: object) -> float:
    r"""Kuramoto interaction energy with multi-language dispatch.

    Returns :math:`E(\theta) = -\tfrac12 \sum_{jk} K_{jk} \cos(\theta_j - \theta_k)`, the
    interaction potential of the Kuramoto network. For symmetric ``K`` the dynamics is the
    gradient flow :math:`\dot\theta = -\nabla E`, so ``E`` is a Lyapunov function. Because
    :math:`\cos` is even, ``E`` depends only on the symmetric part of ``K``.

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
    float
        The scalar interaction energy. An empty input has energy ``0.0``.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_interaction_energy_tier_used`.
    """
    return float(
        _kuramoto_interaction_energy_dispatcher(
            as_float64_array(theta),
            as_float64_array(coupling),
        )
    )


def kuramoto_interaction_energy_gradient(
    theta: object, coupling: object
) -> NDArray[np.float64] | Any:
    r"""Gradient of the Kuramoto interaction energy with multi-language dispatch.

    Returns :math:`\partial E / \partial \theta_j = \tfrac12 \sum_k (K_{jk} + K_{kj})
    \sin(\theta_j - \theta_k)`, the symmetrised coupling force. The components sum to zero
    (``E`` is invariant under a global phase shift); for symmetric ``K`` this equals the
    negated networked-Kuramoto force.

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
        One-dimensional float64 gradient array of length ``N``. An empty input yields an
        empty array. When ``theta`` or ``coupling`` is a Torch/JAX tensor, the result is
        restored to that first detected tensor namespace.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_interaction_energy_gradient_tier_used`.
    """
    template = tensor_template(theta, coupling)
    result = np.asarray(
        _kuramoto_interaction_energy_gradient_dispatcher(
            as_float64_array(theta),
            as_float64_array(coupling),
        ),
        dtype=np.float64,
    )
    return restore_array(result, template)


def kuramoto_interaction_energy_hessian(
    theta: object, coupling: object
) -> NDArray[np.float64] | Any:
    r"""Hessian of the Kuramoto interaction energy with multi-language dispatch.

    Returns :math:`\partial^2 E / \partial \theta_i \partial \theta_l = -\tfrac12 (K_{il} +
    K_{li})\cos(\theta_i - \theta_l)` for :math:`l \neq i`, with diagonal
    :math:`H_{ii} = \tfrac12 \sum_{k \neq i}(K_{ik} + K_{ki})\cos(\theta_i - \theta_k)`. The
    matrix is symmetric and every row sums to zero (``E`` is invariant under a global phase
    shift); for symmetric ``K`` it equals the negated networked-Kuramoto Jacobian, the
    curvature of the synchronisation energy landscape.

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
        Two-dimensional ``(N, N)`` float64 Hessian matrix. An empty input yields a
        ``(0, 0)`` array. When ``theta`` or ``coupling`` is a Torch/JAX tensor, the
        result is restored to that first detected tensor namespace.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_interaction_energy_hessian_tier_used`.
    """
    template = tensor_template(theta, coupling)
    result = np.asarray(
        _kuramoto_interaction_energy_hessian_dispatcher(
            as_float64_array(theta),
            as_float64_array(coupling),
        ),
        dtype=np.float64,
    )
    return restore_array(result, template)


def last_kuramoto_interaction_energy_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_interaction_energy``."""
    return _kuramoto_interaction_energy_dispatcher.last_tier


def last_kuramoto_interaction_energy_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_interaction_energy_gradient``."""
    return _kuramoto_interaction_energy_gradient_dispatcher.last_tier


def last_kuramoto_interaction_energy_hessian_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_interaction_energy_hessian``."""
    return _kuramoto_interaction_energy_hessian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("kuramoto_interaction_energy", _kuramoto_interaction_energy_dispatcher)
register_dispatcher(
    "kuramoto_interaction_energy_gradient", _kuramoto_interaction_energy_gradient_dispatcher
)
register_dispatcher(
    "kuramoto_interaction_energy_hessian", _kuramoto_interaction_energy_hessian_dispatcher
)


__all__ = [
    "kuramoto_interaction_energy",
    "kuramoto_interaction_energy_gradient",
    "kuramoto_interaction_energy_hessian",
    "last_kuramoto_interaction_energy_gradient_tier_used",
    "last_kuramoto_interaction_energy_hessian_tier_used",
    "last_kuramoto_interaction_energy_tier_used",
]
