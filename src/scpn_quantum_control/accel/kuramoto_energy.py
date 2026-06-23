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
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
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


_kuramoto_interaction_energy_dispatcher = MultiLangDispatcher(_KURAMOTO_INTERACTION_ENERGY_CHAIN)
_kuramoto_interaction_energy_gradient_dispatcher = MultiLangDispatcher(
    _KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN
)


def kuramoto_interaction_energy(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    r"""Kuramoto interaction energy with multi-language dispatch.

    Returns :math:`E(\theta) = -\tfrac12 \sum_{jk} K_{jk} \cos(\theta_j - \theta_k)`, the
    interaction potential of the Kuramoto network. For symmetric ``K`` the dynamics is the
    gradient flow :math:`\dot\theta = -\nabla E`, so ``E`` is a Lyapunov function. Because
    :math:`\cos` is even, ``E`` depends only on the symmetric part of ``K``.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.

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
    return float(_kuramoto_interaction_energy_dispatcher(theta, coupling))


def kuramoto_interaction_energy_gradient(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Gradient of the Kuramoto interaction energy with multi-language dispatch.

    Returns :math:`\partial E / \partial \theta_j = \tfrac12 \sum_k (K_{jk} + K_{kj})
    \sin(\theta_j - \theta_k)`, the symmetrised coupling force. The components sum to zero
    (``E`` is invariant under a global phase shift); for symmetric ``K`` this equals the
    negated networked-Kuramoto force.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 gradient array of length ``N``. An empty input yields an
        empty array.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N``.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_interaction_energy_gradient_tier_used`.
    """
    return np.asarray(
        _kuramoto_interaction_energy_gradient_dispatcher(theta, coupling), dtype=np.float64
    )


def last_kuramoto_interaction_energy_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_interaction_energy``."""
    return _kuramoto_interaction_energy_dispatcher.last_tier


def last_kuramoto_interaction_energy_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_interaction_energy_gradient``."""
    return _kuramoto_interaction_energy_gradient_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("kuramoto_interaction_energy", _kuramoto_interaction_energy_dispatcher)
register_dispatcher(
    "kuramoto_interaction_energy_gradient", _kuramoto_interaction_energy_gradient_dispatcher
)


__all__ = [
    "kuramoto_interaction_energy",
    "kuramoto_interaction_energy_gradient",
    "last_kuramoto_interaction_energy_gradient_tier_used",
    "last_kuramoto_interaction_energy_tier_used",
]
