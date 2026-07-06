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


def kuramoto_interaction_energy_hessian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier Hessian of the Kuramoto interaction energy.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Hessian matrix.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    return np.asarray(jl.kuramoto_interaction_energy_hessian(phases, matrix), dtype=np.float64)


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


def daido_mode_phase(theta: NDArray[np.float64], m: int) -> float:
    """Julia-tier m-th Fourier-mode phase.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    m : int
        Harmonic order, a positive integer.

    Returns
    -------
    float
        The mode phase ``ψ_m`` in radians.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return float(jl.daido_mode_phase(arr, m))


def daido_mode_phase_gradient(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    """Julia-tier gradient of the m-th Fourier-mode phase.

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
    return np.asarray(jl.daido_mode_phase_gradient(arr, m), dtype=np.float64)


def daido_mode_phase_hessian(theta: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    """Julia-tier Hessian of the m-th Fourier-mode phase.

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
    return np.asarray(jl.daido_mode_phase_hessian(arr, m), dtype=np.float64)


def daido_mean_field_force(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    """Julia-tier Daido m-th-harmonic mean-field force.

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
        One-dimensional float64 force array.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.daido_mean_field_force(arr, float(coupling), m), dtype=np.float64)


def daido_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, m: int
) -> NDArray[np.float64]:
    """Julia-tier Daido m-th-harmonic mean-field stability Jacobian.

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
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.daido_mean_field_jacobian(arr, float(coupling), m), dtype=np.float64)


def local_mean_phase(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier network-local Kuramoto mean phase.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency/coupling matrix.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of ``N`` local mean phases.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(adjacency, dtype=np.float64)
    return np.asarray(jl.local_mean_phase(phases, matrix), dtype=np.float64)


def local_mean_phase_jacobian(
    theta: NDArray[np.float64], adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Julia-tier Jacobian of the network-local Kuramoto mean phase.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.
    adjacency : numpy.ndarray
        Two-dimensional ``(N, N)`` non-negative adjacency/coupling matrix.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    matrix = np.ascontiguousarray(adjacency, dtype=np.float64)
    return np.asarray(jl.local_mean_phase_jacobian(phases, matrix), dtype=np.float64)


def sakaguchi_mean_field_force(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    """Julia-tier Sakaguchi–Kuramoto mean-field force.

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
        One-dimensional float64 force array.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(
        jl.sakaguchi_mean_field_force(arr, float(coupling), float(frustration)), dtype=np.float64
    )


def sakaguchi_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, frustration: float
) -> NDArray[np.float64]:
    """Julia-tier Sakaguchi–Kuramoto mean-field stability Jacobian.

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
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(
        jl.sakaguchi_mean_field_jacobian(arr, float(coupling), float(frustration)),
        dtype=np.float64,
    )


def triadic_mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    """Julia-tier triadic (2-simplex) Kuramoto mean-field force.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The triadic coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.triadic_mean_field_force(arr, float(coupling)), dtype=np.float64)


def triadic_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    """Julia-tier triadic (2-simplex) Kuramoto mean-field stability Jacobian.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The triadic coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix.
    """
    jl = _load()
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return np.asarray(jl.triadic_mean_field_jacobian(arr, float(coupling)), dtype=np.float64)


def kuramoto_euler_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Julia-tier forward networked-Kuramoto Euler trajectory.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    dt : float
        The Euler step size.
    n_steps : int
        The number of integration steps.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` float64 trajectory.
    """
    jl = _load()
    return np.ascontiguousarray(
        jl.kuramoto_euler_trajectory(
            np.ascontiguousarray(theta0, dtype=np.float64),
            np.ascontiguousarray(omega, dtype=np.float64),
            np.ascontiguousarray(coupling, dtype=np.float64),
            float(dt),
            int(n_steps),
        ),
        dtype=np.float64,
    )


def kuramoto_euler_vjp(
    trajectory: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier reverse-mode adjoint of the networked-Kuramoto Euler integrator.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` forward trajectory.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    dt : float
        The Euler step size.
    cotangent : numpy.ndarray
        One-dimensional ``(N,)`` cotangent on the final phase.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_theta0, grad_omega, grad_coupling)``.
    """
    jl = _load()
    grad_theta0, grad_omega, grad_coupling = jl.kuramoto_euler_vjp(
        np.ascontiguousarray(trajectory, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(dt),
        np.ascontiguousarray(cotangent, dtype=np.float64),
    )
    return (
        np.ascontiguousarray(grad_theta0, dtype=np.float64),
        np.ascontiguousarray(grad_omega, dtype=np.float64),
        np.ascontiguousarray(grad_coupling, dtype=np.float64),
    )


def kuramoto_rk4_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Julia-tier forward networked-Kuramoto RK4 trajectory.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    dt : float
        The RK4 step size.
    n_steps : int
        The number of integration steps.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` float64 trajectory.
    """
    jl = _load()
    return np.ascontiguousarray(
        jl.kuramoto_rk4_trajectory(
            np.ascontiguousarray(theta0, dtype=np.float64),
            np.ascontiguousarray(omega, dtype=np.float64),
            np.ascontiguousarray(coupling, dtype=np.float64),
            float(dt),
            int(n_steps),
        ),
        dtype=np.float64,
    )


def kuramoto_dopri_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    t_end: float,
    rtol: float,
    atol: float,
    safety: float,
    min_factor: float,
    max_factor: float,
    max_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier adaptive Dormand–Prince networked-Kuramoto forward trajectory.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    t_end : float
        The integration horizon; integration runs from ``0`` to ``t_end``.
    rtol, atol : float
        The relative and absolute error tolerances of the embedded-error step controller.
    safety, min_factor, max_factor : float
        The step-size controller's safety factor and clamp on the per-step growth factor.
    max_steps : int
        The maximum number of accepted steps before integration stops.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, phases, steps)`` — the accepted times ``(M + 1,)``, the phases at those times
        ``(M + 1, N)`` and the realised step sizes ``(M,)``.
    """
    jl = _load()
    count = int(np.asarray(theta0).size)
    times, phases_flat, steps = jl.kuramoto_dopri_trajectory(
        np.ascontiguousarray(theta0, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(t_end),
        float(rtol),
        float(atol),
        float(safety),
        float(min_factor),
        float(max_factor),
        int(max_steps),
    )
    accepted_times = np.ascontiguousarray(times, dtype=np.float64)
    realised_steps = np.ascontiguousarray(steps, dtype=np.float64)
    phases = np.ascontiguousarray(phases_flat, dtype=np.float64).reshape(
        accepted_times.size, count
    )
    return accepted_times, phases, realised_steps


def kuramoto_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier inertial (second-order) networked-Kuramoto forward trajectory.

    Integrates the swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` on the ``(θ, v)`` phase-space state
    by a fixed-step RK4, mirroring the Python floor and Rust tier (tolerance-parity, ~1e-11).

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    velocities : numpy.ndarray
        One-dimensional array of ``N`` initial velocities ``v = θ̇``.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies / power injections.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float
        The damping ``γ`` (``≥ 0``).
    dt : float
        The fixed RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, phases, velocities)`` — the ``(M + 1,)`` sample times, the ``(M + 1, N)`` phases
        and the ``(M + 1, N)`` velocities, with ``M = n_steps``.
    """
    jl = _load()
    count = int(np.asarray(theta0).size)
    times, phases_flat, velocities_flat = jl.kuramoto_inertial_trajectory(
        np.ascontiguousarray(theta0, dtype=np.float64),
        np.ascontiguousarray(velocities, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(mass),
        float(damping),
        float(dt),
        int(n_steps),
    )
    sample_times = np.ascontiguousarray(times, dtype=np.float64)
    phases = np.ascontiguousarray(phases_flat, dtype=np.float64).reshape(sample_times.size, count)
    velocity_history = np.ascontiguousarray(velocities_flat, dtype=np.float64).reshape(
        sample_times.size, count
    )
    return sample_times, phases, velocity_history


def kuramoto_symplectic_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier symplectic (velocity-Verlet) inertial networked-Kuramoto forward trajectory.

    Integrates the swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` by a damped velocity-Verlet splitting,
    mirroring the Python floor and Rust tier (tolerance-parity, ~1e-11).

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    velocities : numpy.ndarray
        One-dimensional array of ``N`` initial velocities ``v = θ̇``.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies / power injections.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float
        The damping ``γ`` (``≥ 0``); ``0`` is the exactly symplectic Hamiltonian limit.
    dt : float
        The fixed Verlet time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, phases, velocities)`` — the ``(M + 1,)`` sample times, the ``(M + 1, N)`` phases
        and the ``(M + 1, N)`` velocities, with ``M = n_steps``.
    """
    jl = _load()
    count = int(np.asarray(theta0).size)
    times, phases_flat, velocities_flat = jl.kuramoto_symplectic_inertial_trajectory(
        np.ascontiguousarray(theta0, dtype=np.float64),
        np.ascontiguousarray(velocities, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(mass),
        float(damping),
        float(dt),
        int(n_steps),
    )
    sample_times = np.ascontiguousarray(times, dtype=np.float64)
    phases = np.ascontiguousarray(phases_flat, dtype=np.float64).reshape(sample_times.size, count)
    velocity_history = np.ascontiguousarray(velocities_flat, dtype=np.float64).reshape(
        sample_times.size, count
    )
    return sample_times, phases, velocity_history


def kuramoto_delayed_trajectory(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier time-delayed (method-of-steps) networked-Kuramoto forward trajectory.

    Integrates the delay-differential equation ``θ̇(t) = ω + F(θ(t), θ(t-τ))`` with the networked
    delayed coupling force by a delay-aware fixed-step RK4, mirroring the Python floor and Rust tier
    (tolerance-parity, ~1e-11). The delay ``τ = delay_steps·dt`` is inferred from the history block.

    Parameters
    ----------
    initial_history : numpy.ndarray
        Two-dimensional ``(delay_steps + 1, N)`` phase history on ``[-τ, 0]``; the last row is
        ``θ(0)``.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    dt : float
        The fixed RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, phases)`` — the ``(M + 1,)`` sample times and the ``(M + 1, N)`` phases, with
        ``M = n_steps``.
    """
    jl = _load()
    count = int(np.asarray(omega).size)
    times, phases_flat = jl.kuramoto_delayed_trajectory(
        np.ascontiguousarray(initial_history, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(dt),
        int(n_steps),
    )
    sample_times = np.ascontiguousarray(times, dtype=np.float64)
    phases = np.ascontiguousarray(phases_flat, dtype=np.float64).reshape(sample_times.size, count)
    return sample_times, phases


def kuramoto_noisy_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    dt: float,
    noise: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier stochastic (Euler–Maruyama) networked-Kuramoto forward trajectory.

    Advances ``dθ = (ω + F(θ)) dt + √(2D) dW`` with the supplied standard-normal increments,
    mirroring the Python floor and Rust tier (tolerance-parity). The reproducible noise is drawn
    once by the caller and passed in, so every tier consumes the same increments.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    diffusion : float
        The diffusion / noise intensity ``D`` (``≥ 0``).
    dt : float
        The Euler–Maruyama time step (``> 0``).
    noise : numpy.ndarray
        Two-dimensional ``(n_steps, N)`` standard-normal Wiener increments.

    Returns
    -------
    tuple of numpy.ndarray
        ``(order_parameter_series, terminal_phases)`` — the ``(n_steps,)`` order-parameter series
        and the ``(N,)`` terminal phases.
    """
    jl = _load()
    series, terminal = jl.kuramoto_noisy_trajectory(
        np.ascontiguousarray(theta0, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(diffusion),
        float(dt),
        np.ascontiguousarray(noise, dtype=np.float64),
    )
    return (
        np.ascontiguousarray(series, dtype=np.float64),
        np.ascontiguousarray(terminal, dtype=np.float64),
    )


def kuramoto_rk4_vjp(
    trajectory: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Julia-tier reverse-mode adjoint of the networked-Kuramoto RK4 integrator.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` forward trajectory.
    omega : numpy.ndarray
        One-dimensional ``(N,)`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix.
    dt : float
        The RK4 step size.
    cotangent : numpy.ndarray
        One-dimensional ``(N,)`` cotangent on the final phase.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_theta0, grad_omega, grad_coupling)``.
    """
    jl = _load()
    grad_theta0, grad_omega, grad_coupling = jl.kuramoto_rk4_vjp(
        np.ascontiguousarray(trajectory, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(coupling, dtype=np.float64),
        float(dt),
        np.ascontiguousarray(cotangent, dtype=np.float64),
    )
    return (
        np.ascontiguousarray(grad_theta0, dtype=np.float64),
        np.ascontiguousarray(grad_omega, dtype=np.float64),
        np.ascontiguousarray(grad_coupling, dtype=np.float64),
    )


__all__ = [
    "daido_order_parameter",
    "daido_order_parameter_gradient",
    "daido_order_parameter_hessian",
    "daido_mode_phase",
    "daido_mode_phase_gradient",
    "daido_mode_phase_hessian",
    "daido_mean_field_force",
    "daido_mean_field_jacobian",
    "is_available",
    "kuramoto_interaction_energy",
    "kuramoto_interaction_energy_gradient",
    "kuramoto_interaction_energy_hessian",
    "local_order_parameter",
    "local_order_parameter_jacobian",
    "local_mean_phase",
    "local_mean_phase_jacobian",
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
    "sakaguchi_mean_field_force",
    "sakaguchi_mean_field_jacobian",
    "triadic_mean_field_force",
    "triadic_mean_field_jacobian",
    "kuramoto_euler_trajectory",
    "kuramoto_euler_vjp",
    "kuramoto_dopri_trajectory",
    "kuramoto_inertial_trajectory",
    "kuramoto_symplectic_inertial_trajectory",
    "kuramoto_delayed_trajectory",
    "kuramoto_noisy_trajectory",
    "kuramoto_rk4_trajectory",
    "kuramoto_rk4_vjp",
]
