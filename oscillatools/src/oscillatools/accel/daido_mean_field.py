# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Daido m-th-harmonic mean-field force and stability Jacobian
"""Daido m-th-harmonic mean-field force and its synchronisation stability Jacobian.

The Kuramoto–Daido mean field couples each oscillator to the m-th Fourier mode of the
ensemble through ``F_j = K (S_m cos m θ_j − C_m sin m θ_j) = K r_m sin(ψ_m − m θ_j)``, with
``C_m = ⟨cos m θ⟩`` and ``S_m = ⟨sin m θ⟩`` — the phase-coupling term that drives m-cluster
synchronisation, generalising the all-to-all mean field (``m = 1``). Its Jacobian
``J_jl = K m [(1/N) cos(m(θ_j − θ_l)) − δ_jl r_m cos(ψ_m − m θ_j)]`` is symmetric with a zero
row sum (the global-phase Goldstone mode); for ``m = 1`` both reduce to the mean-field force
and Jacobian.

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
    """Raise :class:`ValueError` if ``m`` is not a positive integer."""
    if m < 1:
        raise ValueError(f"harmonic order m must be a positive integer, got {m}")


def _rust_daido_mean_field_force(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_force = getattr(engine, "daido_mean_field_force", None)
    if not callable(rust_force):
        raise ImportError("scpn_quantum_engine.daido_mean_field_force is unavailable")

    return np.asarray(
        rust_force(np.ascontiguousarray(theta, dtype=np.float64), float(coupling), m),
        dtype=np.float64,
    )


def _julia_daido_mean_field_force(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    from .julia import daido_mean_field_force as julia_force

    return julia_force(theta, coupling, m)


def _python_daido_mean_field_force(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    # Correctness floor — F_j = K (S_m cos m θ_j − C_m sin m θ_j), with C_m = ⟨cos m θ⟩,
    # S_m = ⟨sin m θ⟩. For m = 1 this is the mean-field force; the empty input yields an empty array.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    scaled = m * phases
    cos_mean = float(np.mean(np.cos(scaled)))
    sin_mean = float(np.mean(np.sin(scaled)))
    force = coupling * (sin_mean * np.cos(scaled) - cos_mean * np.sin(scaled))
    return np.ascontiguousarray(force, dtype=np.float64)


_DAIDO_MEAN_FIELD_FORCE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float, int], NDArray[np.float64]]]
] = [
    ("rust", _rust_daido_mean_field_force),
    ("julia", _julia_daido_mean_field_force),
    ("python", _python_daido_mean_field_force),
]


def _rust_daido_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "daido_mean_field_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.daido_mean_field_jacobian is unavailable")

    return np.asarray(
        rust_jacobian(np.ascontiguousarray(theta, dtype=np.float64), float(coupling), m),
        dtype=np.float64,
    )


def _julia_daido_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    _validate_harmonic(m)
    from .julia import daido_mean_field_jacobian as julia_jacobian

    return julia_jacobian(theta, coupling, m)


def _python_daido_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    # Correctness floor — J_jl = K m [(1/N) cos(m(θ_j − θ_l)) − δ_jl r_m cos(ψ_m − m θ_j)]
    #   = K m [(1/N) cos(m(θ_j − θ_l)) − δ_jl (C_m cos m θ_j + S_m sin m θ_j)]. Symmetric with a
    # zero row sum. For m = 1 this is the mean-field Jacobian.
    _validate_harmonic(m)
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    scaled = m * phases
    cos_mean = float(np.mean(np.cos(scaled)))
    sin_mean = float(np.mean(np.sin(scaled)))
    jacobian = (coupling * m / count) * np.cos(scaled[:, None] - scaled[None, :])
    diagonal = coupling * m * (cos_mean * np.cos(scaled) + sin_mean * np.sin(scaled))
    jacobian -= np.diag(diagonal)
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the force chain (Rust → Julia → Python floor). Their
# micro-benchmark (at m = 2, K = 1) is recorded in
# ``docs/benchmarks/daido_mean_field_tiers.json``; rerun
# ``python scripts/bench_daido_mean_field_tiers.py`` when these chains are edited.
_DAIDO_MEAN_FIELD_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float, int], NDArray[np.float64]]]
] = [
    ("rust", _rust_daido_mean_field_jacobian),
    ("julia", _julia_daido_mean_field_jacobian),
    ("python", _python_daido_mean_field_jacobian),
]


_daido_mean_field_force_dispatcher = MultiLangDispatcher(_DAIDO_MEAN_FIELD_FORCE_CHAIN)
_daido_mean_field_jacobian_dispatcher = MultiLangDispatcher(_DAIDO_MEAN_FIELD_JACOBIAN_CHAIN)


def daido_mean_field_force(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    r"""Daido m-th-harmonic mean-field force with multi-language dispatch.

    Returns :math:`F_j = K (S_m \cos m\theta_j - C_m \sin m\theta_j) = K r_m
    \sin(\psi_m - m\theta_j)`, the phase-coupling term driving m-cluster synchronisation. For
    :math:`m = 1` it equals
    :func:`~oscillatools.accel.kuramoto_mean_field.mean_field_force`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K`` (any real value).
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of the same length as ``theta``. An empty input
        yields an empty array.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_mean_field_force_tier_used`.
    """
    return np.asarray(_daido_mean_field_force_dispatcher(theta, coupling, m), dtype=np.float64)


def daido_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    r"""Daido m-th-harmonic mean-field stability Jacobian with multi-language dispatch.

    Returns :math:`J_{jl} = K m [(1/N)\cos(m(\theta_j - \theta_l)) - \delta_{jl} r_m
    \cos(\psi_m - m\theta_j)]`, the linearisation of the m-th-harmonic mean-field force. The
    matrix is symmetric and every row sums to zero (the global-phase Goldstone mode). For
    :math:`m = 1` it equals
    :func:`~oscillatools.accel.kuramoto_mean_field.mean_field_jacobian`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K``.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. An empty input yields a
        ``(0, 0)`` array.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_daido_mean_field_jacobian_tier_used`.
    """
    return np.asarray(_daido_mean_field_jacobian_dispatcher(theta, coupling, m), dtype=np.float64)


def last_daido_mean_field_force_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_mean_field_force``."""
    return _daido_mean_field_force_dispatcher.last_tier


def last_daido_mean_field_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``daido_mean_field_jacobian``."""
    return _daido_mean_field_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("daido_mean_field_force", _daido_mean_field_force_dispatcher)
register_dispatcher("daido_mean_field_jacobian", _daido_mean_field_jacobian_dispatcher)


__all__ = [
    "daido_mean_field_force",
    "daido_mean_field_jacobian",
    "last_daido_mean_field_force_tier_used",
    "last_daido_mean_field_jacobian_tier_used",
]
