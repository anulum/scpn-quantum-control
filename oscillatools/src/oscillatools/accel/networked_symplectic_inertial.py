# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Polyglot symplectic inertial networked-Kuramoto forward trajectory
r"""Polyglot symplectic (velocity-Verlet) inertial networked-Kuramoto forward trajectory.

Integrates the second-order swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` for the *networked* coupling
force :math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)` by a damped velocity-Verlet (leapfrog)
splitting: a half-step exponential velocity decay, a velocity-Verlet kick–drift–kick of the
Hamiltonian part, and a second half-step decay. At ``γ = 0`` this is exactly symplectic, so unlike
the fixed-grid RK4 (:func:`~oscillatools.accel.networked_inertial.networked_inertial_trajectory`) its
energy error stays *bounded* over long integration times rather than drifting secularly.

This is the accelerated, networked-force special case of the general
:func:`~oscillatools.accel.kuramoto_symplectic_inertial.integrate_symplectic_inertial`: where that
function composes an arbitrary Python force callable (and runs on the pure-Python floor), this
function pins the force to the networked coupling matrix so the whole Verlet hot loop can run in a
compiled tier. The forward integration dispatches across a Rust → Julia → Python floor tier chain
(measured fastest first, recorded on :func:`last_networked_symplectic_inertial_trajectory_tier_used`).
The tiers share the Verlet arithmetic and differ only in the coupling-force summation order — a scalar
loop in the compiled tiers versus NumPy's vectorised reduction in the floor — so they are
*tolerance-parity* (~1e-11), not bit-identical.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher
from .kuramoto_inertial import InertialTrajectory
from .kuramoto_symplectic_inertial import integrate_symplectic_inertial

_Forward = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _force(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Networked coupling force ``F_j = Σ_k K_jk sin(θ_k − θ_j)`` (pure-NumPy floor)."""
    difference = phases[None, :] - phases[:, None]
    return np.asarray((coupling * np.sin(difference)).sum(axis=1), dtype=np.float64)


def _validate_state(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta0, velocities, omega, coupling)`` after shape validation."""
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    speed = np.ascontiguousarray(velocities, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("theta0 must be a non-empty one-dimensional array")
    if speed.shape != phases.shape:
        raise ValueError(f"velocities must match theta0 shape {phases.shape}, got {speed.shape}")
    if frequencies.shape != phases.shape:
        raise ValueError(f"omega must match theta0 shape {phases.shape}, got {frequencies.shape}")
    count = phases.size
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must be a square matrix of order {count}, got {matrix.shape}")
    return phases, speed, frequencies, matrix


def _rust_networked_symplectic_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> _Forward:
    phases, speed, frequencies, matrix = _validate_state(theta0, velocities, omega, coupling)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_forward = getattr(engine, "kuramoto_symplectic_inertial_trajectory", None)
    if not callable(rust_forward):
        raise ImportError(
            "scpn_quantum_engine.kuramoto_symplectic_inertial_trajectory is unavailable"
        )
    times, path, velocity_history = rust_forward(
        phases,
        speed,
        frequencies,
        matrix,
        float(mass),
        float(damping),
        float(dt),
        int(n_steps),
    )
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(path, dtype=np.float64),
        np.ascontiguousarray(velocity_history, dtype=np.float64),
    )


def _julia_networked_symplectic_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> _Forward:
    phases, speed, frequencies, matrix = _validate_state(theta0, velocities, omega, coupling)
    from .julia import kuramoto_symplectic_inertial_trajectory as julia_forward

    times, path, velocity_history = julia_forward(
        phases,
        speed,
        frequencies,
        matrix,
        float(mass),
        float(damping),
        float(dt),
        int(n_steps),
    )
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(path, dtype=np.float64),
        np.ascontiguousarray(velocity_history, dtype=np.float64),
    )


def _python_networked_symplectic_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> _Forward:
    """Pure-Python floor: the general Verlet orchestration bound to the networked force."""
    phases, speed, frequencies, matrix = _validate_state(theta0, velocities, omega, coupling)
    trajectory = integrate_symplectic_inertial(
        phases,
        speed,
        frequencies,
        lambda values: _force(values, matrix),
        mass,
        damping=damping,
        dt=dt,
        n_steps=n_steps,
    )
    return (
        np.ascontiguousarray(trajectory.times, dtype=np.float64),
        np.ascontiguousarray(trajectory.phases, dtype=np.float64),
        np.ascontiguousarray(trajectory.velocities, dtype=np.float64),
    )


_NETWORKED_SYMPLECTIC_INERTIAL_TRAJECTORY_CHAIN: list[tuple[str, Callable[..., _Forward]]] = [
    ("rust", _rust_networked_symplectic_inertial_trajectory),
    ("julia", _julia_networked_symplectic_inertial_trajectory),
    ("python", _python_networked_symplectic_inertial_trajectory),
]
_networked_symplectic_inertial_trajectory_dispatcher = MultiLangDispatcher(
    _NETWORKED_SYMPLECTIC_INERTIAL_TRAJECTORY_CHAIN
)


def networked_symplectic_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    *,
    damping: float = 1.0,
    dt: float,
    n_steps: int,
) -> InertialTrajectory:
    r"""Integrate the inertial networked Kuramoto model by a symplectic velocity-Verlet splitting.

    Advances the swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` with the networked coupling force
    :math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)` by a damped velocity-Verlet step, sampling
    ``θ`` and ``v`` at every step. At ``γ = 0`` the scheme is exactly symplectic, so the energy error
    stays bounded over long times. This is the accelerated, networked-force special case of
    :func:`~oscillatools.accel.kuramoto_symplectic_inertial.integrate_symplectic_inertial`; the forward
    integration dispatches across a Rust → Julia → Python floor tier chain (measured fastest first,
    recorded on :func:`last_networked_symplectic_inertial_trajectory_tier_used`). The tiers are
    tolerance-parity: they share the Verlet arithmetic and differ only in the coupling-force summation
    order, agreeing to ~1e-11.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    velocities : numpy.ndarray
        One-dimensional array of ``N`` initial velocities ``v(0) = θ̇(0)``.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies / power injections ``ω``.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    mass : float
        The inertia ``m``; must be strictly positive.
    damping : float, optional
        The damping ``γ``; must be non-negative. Defaults to ``1``. ``0`` is the symplectic limit.
    dt : float
        The fixed Verlet time step; must be strictly positive.
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    InertialTrajectory
        The sampled phase and velocity trajectory with the ``mass`` and ``damping`` used.

    Raises
    ------
    ValueError
        If the state shapes are inconsistent, or ``mass``/``damping``/``dt``/``n_steps`` are out of
        range.
    """
    _validate_state(theta0, velocities, omega, coupling)
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    times, path, velocity_history = _networked_symplectic_inertial_trajectory_dispatcher(
        theta0, velocities, omega, coupling, mass, damping, dt, n_steps
    )
    return InertialTrajectory(
        times=times,
        phases=path,
        velocities=velocity_history,
        mass=float(mass),
        damping=float(damping),
    )


def last_networked_symplectic_inertial_trajectory_tier_used() -> str | None:
    """The tier (``'rust'``/``'julia'``/``'python'``) that served the last dispatched forward.

    ``None`` before the first dispatched call.
    """
    return _networked_symplectic_inertial_trajectory_dispatcher.last_tier


register_dispatcher(
    "networked_symplectic_inertial_trajectory",
    _networked_symplectic_inertial_trajectory_dispatcher,
)


__all__ = [
    "last_networked_symplectic_inertial_trajectory_tier_used",
    "networked_symplectic_inertial_trajectory",
]
