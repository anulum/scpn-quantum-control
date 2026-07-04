# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto mean-field force and stability Jacobian
"""Kuramoto mean-field force and its synchronisation stability Jacobian.

The all-to-all Kuramoto mean field couples each oscillator to the ensemble through
the force ``F_j = K (S cos θ_j − C sin θ_j)`` with ``C = ⟨cos θ⟩`` and ``S = ⟨sin θ⟩``
— the phase-coupling term of ``dθ_j/dt = ω_j + F_j``. Its Jacobian
``J_jk = (K/N) cos(θ_j − θ_k) − K δ_jk (C cos θ_j + S sin θ_j)`` is the linearisation of
the dynamics about a phase configuration; its eigenvalues classify the linear stability
of synchronisation, with a guaranteed zero eigenvalue (the global-phase Goldstone mode).

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~oscillatools.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _rust_mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_force = getattr(engine, "mean_field_force", None)
    if not callable(rust_force):
        raise ImportError("scpn_quantum_engine.mean_field_force is unavailable")

    return np.asarray(
        rust_force(np.ascontiguousarray(theta, dtype=np.float64), float(coupling)),
        dtype=np.float64,
    )


def _julia_mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    from .julia import mean_field_force as julia_force

    return julia_force(theta, coupling)


def _python_mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    # Correctness floor — mean-field coupling force F_j = K (S cos θ_j − C sin θ_j),
    # with C = ⟨cos θ⟩, S = ⟨sin θ⟩. The empty input yields an empty array.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    force = coupling * (sin_mean * np.cos(phases) - cos_mean * np.sin(phases))
    return np.ascontiguousarray(force, dtype=np.float64)


_MEAN_FIELD_FORCE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
] = [
    ("rust", _rust_mean_field_force),
    ("julia", _julia_mean_field_force),
    ("python", _python_mean_field_force),
]


def _rust_mean_field_jacobian(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "mean_field_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.mean_field_jacobian is unavailable")

    return np.asarray(
        rust_jacobian(np.ascontiguousarray(theta, dtype=np.float64), float(coupling)),
        dtype=np.float64,
    )


def _julia_mean_field_jacobian(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    from .julia import mean_field_jacobian as julia_jacobian

    return julia_jacobian(theta, coupling)


def _python_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    # Correctness floor — stability Jacobian of the mean-field force. With C = ⟨cos θ⟩,
    # S = ⟨sin θ⟩: J_jk = (K/N) cos(θ_j − θ_k) − K δ_jk (C cos θ_j + S sin θ_j). The matrix
    # is symmetric and every row sums to zero (the global-phase Goldstone mode). The empty
    # input yields a ``(0, 0)`` array.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    jacobian = (coupling / count) * np.cos(phases[:, None] - phases[None, :])
    diagonal = coupling * (cos_mean * np.cos(phases) + sin_mean * np.sin(phases))
    jacobian -= np.diag(diagonal)
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the force chain (Rust → Julia → Python floor). Their
# micro-benchmark (at K = 1) is recorded in
# ``docs/benchmarks/mean_field_tiers.json``; rerun
# ``python scripts/bench_mean_field_tiers.py`` when these chains are edited.
_MEAN_FIELD_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
] = [
    ("rust", _rust_mean_field_jacobian),
    ("julia", _julia_mean_field_jacobian),
    ("python", _python_mean_field_jacobian),
]


_mean_field_force_dispatcher = MultiLangDispatcher(_MEAN_FIELD_FORCE_CHAIN)
_mean_field_jacobian_dispatcher = MultiLangDispatcher(_MEAN_FIELD_JACOBIAN_CHAIN)


def mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    r"""Kuramoto mean-field coupling force with multi-language dispatch.

    Returns :math:`F_j = K (S \cos\theta_j - C \sin\theta_j)`, the phase-coupling term
    of the all-to-all Kuramoto dynamics :math:`\dot\theta_j = \omega_j + F_j`, where
    :math:`C = \langle\cos\theta\rangle` and :math:`S = \langle\sin\theta\rangle`. This
    equals :math:`K r \sin(\psi - \theta_j)` in terms of the order parameter
    :math:`r e^{i\psi}`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K`` (any real value; negative is repulsive).

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of the same length as ``theta``. An empty
        input yields an empty array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_mean_field_force_tier_used`.
    """
    return np.asarray(_mean_field_force_dispatcher(theta, coupling), dtype=np.float64)


def mean_field_jacobian(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    r"""Kuramoto synchronisation stability Jacobian with multi-language dispatch.

    Returns the linearisation :math:`J_{jk} = \partial F_j / \partial \theta_k =
    (K/N)\cos(\theta_j - \theta_k) - K \delta_{jk}(C\cos\theta_j + S\sin\theta_j)` of the
    mean-field force about a phase configuration. The matrix is symmetric and every row
    sums to zero (the global-phase Goldstone mode is a guaranteed zero eigenvalue); its
    spectrum classifies the linear stability of synchronisation.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. An empty input yields a
        ``(0, 0)`` array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_mean_field_jacobian_tier_used`.
    """
    return np.asarray(_mean_field_jacobian_dispatcher(theta, coupling), dtype=np.float64)


def last_mean_field_force_tier_used() -> str | None:
    """Return the tier that served the most recent ``mean_field_force``."""
    return _mean_field_force_dispatcher.last_tier


def last_mean_field_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``mean_field_jacobian``."""
    return _mean_field_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("mean_field_force", _mean_field_force_dispatcher)
register_dispatcher("mean_field_jacobian", _mean_field_jacobian_dispatcher)


__all__ = [
    "last_mean_field_force_tier_used",
    "last_mean_field_jacobian_tier_used",
    "mean_field_force",
    "mean_field_jacobian",
]
