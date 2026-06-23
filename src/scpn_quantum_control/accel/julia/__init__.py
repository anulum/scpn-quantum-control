# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia accel tier
"""Julia acceleration tier via ``juliacall``.

Activation is lazy: the Julia runtime boots on first
:func:`is_available` call that returns True, and the ``.jl`` source
files are ``include``'d at that moment. Subsequent calls reuse the
cached Julia ``Main`` module so the JIT warm-up cost is paid exactly
once per Python process.

If ``juliacall`` is not installed or Julia itself is not on PATH,
:func:`is_available` returns False and the module-level accessors
raise :class:`ImportError`. The dispatcher above handles that
gracefully and falls through to the next tier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_JL: Any = None
_INCLUDED: bool = False
_JULIA_DIR = Path(__file__).parent


def _load() -> Any:
    """Return the ``juliacall.Main`` handle, booting Julia on first call."""
    global _JL, _INCLUDED
    if _JL is not None and _INCLUDED:
        return _JL
    try:
        from juliacall import Main as jl
    except Exception as exc:
        raise ImportError("juliacall is not installed") from exc
    _JL = jl
    if not _INCLUDED:
        for jl_file in sorted(_JULIA_DIR.glob("*.jl")):
            jl.include(str(jl_file))
        _INCLUDED = True
    return _JL


def is_available() -> bool:
    """Return True if Julia + juliacall can run in this process.

    First call incurs the Julia boot cost (~20 s). Callers that want a
    cheap probe without warming Julia should check whether
    ``juliacall`` is importable instead.
    """
    try:
        _load()
        return True
    except Exception:
        return False


def order_parameter(theta: NDArray[np.float64]) -> float:
    """Julia-tier implementation of the Kuramoto order parameter."""
    jl = _load()
    # juliacall converts a numpy array to a Julia Vector{Float64}
    # without copying when the dtype is float64 and C-contiguous.
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return float(jl.order_parameter(arr))


def order_parameters_batch(theta_batch: NDArray[np.float64]) -> NDArray[np.float64]:
    """Julia-tier batched variant — T time-slices × N oscillators."""
    jl = _load()
    arr = np.ascontiguousarray(theta_batch, dtype=np.float64)
    return np.asarray(jl.order_parameters_batch(arr), dtype=np.float64)


def order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Julia-tier gradient of the Kuramoto order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of per-phase gradient components.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.order_parameter_gradient(arr), dtype=np.float64)


def order_parameter_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Julia-tier Hessian of the Kuramoto order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Hessian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.order_parameter_hessian(arr), dtype=np.float64)


def mean_phase(theta: NDArray[np.float64]) -> float:
    """Julia-tier circular mean phase of a Kuramoto ensemble.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    float
        The mean phase in radians.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return float(jl.mean_phase(arr))


def mean_phase_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Julia-tier gradient of the Kuramoto mean phase.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of per-phase gradient components.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.mean_phase_gradient(arr), dtype=np.float64)


def mean_phase_hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Julia-tier Hessian of the Kuramoto mean phase.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Hessian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.mean_phase_hessian(arr), dtype=np.float64)


def daido_order_parameter(theta: NDArray[np.float64], m: int) -> float:
    """Julia-tier m-th Daido order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    float
        The Daido order parameter ``r_m``.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return float(jl.daido_order_parameter(arr, m))


def daido_order_parameter_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    """Julia-tier gradient of the m-th Daido order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of per-phase gradient components.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.daido_order_parameter_gradient(arr, m), dtype=np.float64)


def daido_order_parameter_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    """Julia-tier Hessian of the m-th Daido order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Hessian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.daido_order_parameter_hessian(arr, m), dtype=np.float64)


def mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    """Julia-tier Kuramoto mean-field coupling force.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.mean_field_force(arr, float(coupling)), dtype=np.float64)


def mean_field_jacobian(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    """Julia-tier Kuramoto synchronisation stability Jacobian.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.mean_field_jacobian(arr, float(coupling)), dtype=np.float64)


def networked_kuramoto_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier networked Kuramoto coupling force.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of length ``N``.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return np.asarray(jl.networked_kuramoto_force(phases, matrix), dtype=np.float64)


def networked_kuramoto_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier networked Kuramoto stability Jacobian.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return np.asarray(jl.networked_kuramoto_jacobian(phases, matrix), dtype=np.float64)


def kuramoto_interaction_energy(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    """Julia-tier Kuramoto interaction energy.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.

    Returns
    -------
    float
        The scalar interaction energy.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return float(jl.kuramoto_interaction_energy(phases, matrix))


def kuramoto_interaction_energy_gradient(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier gradient of the Kuramoto interaction energy.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 gradient array of length ``N``.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return np.asarray(jl.kuramoto_interaction_energy_gradient(phases, matrix), dtype=np.float64)


def sakaguchi_force(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    """Julia-tier Kuramoto–Sakaguchi frustrated force.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    frustration : float
        The phase-frustration angle ``α`` in radians.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of length ``N``.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return np.asarray(jl.sakaguchi_force(phases, matrix, float(frustration)), dtype=np.float64)


def sakaguchi_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], frustration: float
) -> NDArray[np.float64]:
    """Julia-tier Kuramoto–Sakaguchi stability Jacobian.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    frustration : float
        The phase-frustration angle ``α`` in radians.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return np.asarray(jl.sakaguchi_jacobian(phases, matrix, float(frustration)), dtype=np.float64)


def local_order_parameter(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier network-local Kuramoto order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency matrix.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of ``N`` local order parameters.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(adjacency, dtype=np.float64)
    return np.asarray(jl.local_order_parameter(phases, matrix), dtype=np.float64)


def local_order_parameter_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier Jacobian of the network-local Kuramoto order parameter.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(adjacency, dtype=np.float64)
    return np.asarray(jl.local_order_parameter_jacobian(phases, matrix), dtype=np.float64)


__all__ = [
    "daido_order_parameter",
    "daido_order_parameter_gradient",
    "daido_order_parameter_hessian",
    "is_available",
    "kuramoto_interaction_energy",
    "kuramoto_interaction_energy_gradient",
    "local_order_parameter",
    "local_order_parameter_jacobian",
    "mean_field_force",
    "mean_field_jacobian",
    "networked_kuramoto_force",
    "networked_kuramoto_jacobian",
    "mean_phase",
    "mean_phase_gradient",
    "mean_phase_hessian",
    "order_parameter",
    "order_parameter_gradient",
    "order_parameter_hessian",
    "order_parameters_batch",
    "sakaguchi_force",
    "sakaguchi_jacobian",
]
