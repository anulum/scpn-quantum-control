# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sakaguchi–Kuramoto mean-field force and stability Jacobian
"""Sakaguchi–Kuramoto mean-field force and its stability Jacobian.

The all-to-all Sakaguchi–Kuramoto mean field adds a phase frustration ``α`` to the Kuramoto
mean field: ``F_j = K r sin(ψ − θ_j − α)``, with order parameter ``r`` and mean phase ``ψ``.
It is the frustrated all-to-all coupling that drives travelling-wave and partially synchronised
states; for ``α = 0`` it reduces to the mean-field force. Its Jacobian
``J_jl = (K/N) cos(θ_j − θ_l + α) − δ_jl K r cos(ψ − θ_j − α)`` is non-symmetric whenever
``α ≠ 0`` (the dynamics are non-variational), yet every row still sums to zero (the
global-phase Goldstone mode survives the frustration); for ``α = 0`` it reduces to the
mean-field Jacobian.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _rust_sakaguchi_mean_field_force(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_force = getattr(engine, "sakaguchi_mean_field_force", None)
    if not callable(rust_force):
        raise ImportError("scpn_quantum_engine.sakaguchi_mean_field_force is unavailable")

    return np.asarray(
        rust_force(
            np.ascontiguousarray(theta, dtype=np.float64), float(coupling), float(frustration)
        ),
        dtype=np.float64,
    )


def _julia_sakaguchi_mean_field_force(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    from .julia import sakaguchi_mean_field_force as julia_force

    return julia_force(theta, coupling, frustration)


def _python_sakaguchi_mean_field_force(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    # Correctness floor — F_j = K r sin(ψ − θ_j − α) =
    # K [(S cos θ_j − C sin θ_j) cos α − (C cos θ_j + S sin θ_j) sin α], with C = ⟨cos θ⟩,
    # S = ⟨sin θ⟩. For α = 0 this is the mean-field force; an empty input yields an empty array.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    cos_a = np.cos(frustration)
    sin_a = np.sin(frustration)
    cos_theta = np.cos(phases)
    sin_theta = np.sin(phases)
    in_phase = sin_mean * cos_theta - cos_mean * sin_theta
    quadrature = cos_mean * cos_theta + sin_mean * sin_theta
    force = coupling * (in_phase * cos_a - quadrature * sin_a)
    return np.ascontiguousarray(force, dtype=np.float64)


_SAKAGUCHI_MEAN_FIELD_FORCE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float, float], NDArray[np.float64]]]
] = [
    ("rust", _rust_sakaguchi_mean_field_force),
    ("julia", _julia_sakaguchi_mean_field_force),
    ("python", _python_sakaguchi_mean_field_force),
]


def _rust_sakaguchi_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "sakaguchi_mean_field_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.sakaguchi_mean_field_jacobian is unavailable")

    return np.asarray(
        rust_jacobian(
            np.ascontiguousarray(theta, dtype=np.float64), float(coupling), float(frustration)
        ),
        dtype=np.float64,
    )


def _julia_sakaguchi_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    from .julia import sakaguchi_mean_field_jacobian as julia_jacobian

    return julia_jacobian(theta, coupling, frustration)


def _python_sakaguchi_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    # Correctness floor — J_jl = (K/N) cos(θ_j − θ_l + α) − δ_jl K r cos(ψ − θ_j − α) =
    # (K/N) cos(θ_j − θ_l + α) − δ_jl K (C cos(θ_j + α) + S sin(θ_j + α)). Non-symmetric for
    # α ≠ 0, every row sums to zero. For α = 0 this is the mean-field Jacobian.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    jacobian = (coupling / count) * np.cos(phases[:, None] - phases[None, :] + frustration)
    diagonal = coupling * (
        cos_mean * np.cos(phases + frustration) + sin_mean * np.sin(phases + frustration)
    )
    jacobian -= np.diag(diagonal)
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the force chain (Rust → Julia → Python floor). Their
# micro-benchmark (at K = 1, α = 0.5) is recorded in
# ``docs/benchmarks/sakaguchi_mean_field_tiers.json``; rerun
# ``python scripts/bench_sakaguchi_mean_field_tiers.py`` when these chains are edited.
_SAKAGUCHI_MEAN_FIELD_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float, float], NDArray[np.float64]]]
] = [
    ("rust", _rust_sakaguchi_mean_field_jacobian),
    ("julia", _julia_sakaguchi_mean_field_jacobian),
    ("python", _python_sakaguchi_mean_field_jacobian),
]


_sakaguchi_mean_field_force_dispatcher = MultiLangDispatcher(_SAKAGUCHI_MEAN_FIELD_FORCE_CHAIN)
_sakaguchi_mean_field_jacobian_dispatcher = MultiLangDispatcher(
    _SAKAGUCHI_MEAN_FIELD_JACOBIAN_CHAIN
)


def sakaguchi_mean_field_force(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    r"""Sakaguchi–Kuramoto mean-field force with multi-language dispatch.

    Returns :math:`F_j = K r \sin(\psi - \theta_j - \alpha)`, the frustrated all-to-all
    coupling term, with order parameter :math:`r` and mean phase :math:`\psi`. For
    :math:`\alpha = 0` it equals
    :func:`~scpn_quantum_control.accel.kuramoto_mean_field.mean_field_force`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K`` (any real value).
    frustration : float
        The phase-frustration angle ``α`` in radians (any real value).

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of the same length as ``theta``. An empty input
        yields an empty array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_sakaguchi_mean_field_force_tier_used`.
    """
    return np.asarray(
        _sakaguchi_mean_field_force_dispatcher(theta, coupling, frustration), dtype=np.float64
    )


def sakaguchi_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    r"""Sakaguchi–Kuramoto mean-field stability Jacobian with multi-language dispatch.

    Returns :math:`J_{jl} = (K/N)\cos(\theta_j - \theta_l + \alpha) - \delta_{jl} K r
    \cos(\psi - \theta_j - \alpha)`, the linearisation of the frustrated mean-field force. The
    matrix is non-symmetric whenever :math:`\alpha \neq 0` (the dynamics are non-variational),
    yet every row sums to zero (the global-phase Goldstone mode). For :math:`\alpha = 0` it
    equals :func:`~scpn_quantum_control.accel.kuramoto_mean_field.mean_field_jacobian`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K``.
    frustration : float
        The phase-frustration angle ``α`` in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. An empty input yields a
        ``(0, 0)`` array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_sakaguchi_mean_field_jacobian_tier_used`.
    """
    return np.asarray(
        _sakaguchi_mean_field_jacobian_dispatcher(theta, coupling, frustration), dtype=np.float64
    )


def last_sakaguchi_mean_field_force_tier_used() -> str | None:
    """Return the tier that served the most recent ``sakaguchi_mean_field_force``."""
    return _sakaguchi_mean_field_force_dispatcher.last_tier


def last_sakaguchi_mean_field_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``sakaguchi_mean_field_jacobian``."""
    return _sakaguchi_mean_field_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("sakaguchi_mean_field_force", _sakaguchi_mean_field_force_dispatcher)
register_dispatcher("sakaguchi_mean_field_jacobian", _sakaguchi_mean_field_jacobian_dispatcher)


__all__ = [
    "last_sakaguchi_mean_field_force_tier_used",
    "last_sakaguchi_mean_field_jacobian_tier_used",
    "sakaguchi_mean_field_force",
    "sakaguchi_mean_field_jacobian",
]
