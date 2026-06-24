# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable networked-Kuramoto Euler integrator and adjoint
"""Differentiable networked-Kuramoto Euler integrator and its reverse-mode adjoint.

Forward integration of the networked Kuramoto dynamics by explicit Euler,
``θ_{n+1} = θ_n + dt (ω + F(θ_n))`` with ``F_j(θ) = Σ_k K_jk sin(θ_k − θ_j)``, recording the
full phase trajectory. The reverse-mode adjoint propagates a cotangent ``λ_N = ∂L/∂θ_N`` on the
final state back through the trajectory,

    λ_n = λ_{n+1} + dt · J(θ_n)ᵀ λ_{n+1},

where ``J = ∂F/∂θ`` is the networked stability Jacobian, accumulating the gradients of a scalar
objective ``L`` with respect to every input: ``∂L/∂θ₀ = λ₀``, ``∂L/∂ω = dt Σ_n λ_{n+1}`` and
``∂L/∂K_pq = dt Σ_n λ_{n+1,p} sin(θ_{n,q} − θ_{n,p})``. Together these turn the Kuramoto
simulation into a differentiable program: gradients of any terminal objective with respect to
the initial phases, the natural frequencies and the coupling matrix flow through the dynamics,
which is what optimal coupling design, pinning control and learning ``K`` from data require.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher

_VjpResult = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _validate_forward(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta0, omega, coupling)`` after shape validation.

    Raises
    ------
    ValueError
        If ``omega`` or ``coupling`` does not match ``theta0``'s order, or ``n_steps`` < 0.
    """
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = phases.size
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    return phases, frequencies, matrix


def _rust_kuramoto_euler_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_forward = getattr(engine, "kuramoto_euler_trajectory", None)
    if not callable(rust_forward):
        raise ImportError("scpn_quantum_engine.kuramoto_euler_trajectory is unavailable")

    return np.asarray(
        rust_forward(phases, frequencies, matrix, float(dt), int(n_steps)), dtype=np.float64
    )


def _julia_kuramoto_euler_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    from .julia import kuramoto_euler_trajectory as julia_forward

    return julia_forward(phases, frequencies, matrix, float(dt), int(n_steps))


def _python_kuramoto_euler_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    # Correctness floor — θ_{n+1} = θ_n + dt (ω + F(θ_n)), F_j = Σ_k K_jk sin(θ_k − θ_j).
    # Returns the (n_steps + 1, N) phase trajectory including the initial state.
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    count = phases.size
    trajectory = np.zeros((n_steps + 1, count), dtype=np.float64)
    trajectory[0] = phases
    current = phases
    for step in range(n_steps):
        difference = current[None, :] - current[:, None]
        force = (matrix * np.sin(difference)).sum(axis=1)
        current = current + dt * (frequencies + force)
        trajectory[step + 1] = current
    return np.ascontiguousarray(trajectory, dtype=np.float64)


_KURAMOTO_EULER_TRAJECTORY_CHAIN: list[
    tuple[
        str,
        Callable[
            [
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
                int,
            ],
            NDArray[np.float64],
        ],
    ]
] = [
    ("rust", _rust_kuramoto_euler_trajectory),
    ("julia", _julia_kuramoto_euler_trajectory),
    ("python", _python_kuramoto_euler_trajectory),
]


def _validate_vjp(
    trajectory: NDArray[np.float64],
    coupling: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(trajectory, coupling, cotangent)`` after shape validation.

    Raises
    ------
    ValueError
        If ``trajectory`` is not two-dimensional, or ``coupling``/``cotangent`` do not match its
        oscillator count.
    """
    path = np.ascontiguousarray(trajectory, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    seed = np.ascontiguousarray(cotangent, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError(f"trajectory must be two-dimensional, got shape {path.shape}")
    count = path.shape[1]
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if seed.shape != (count,):
        raise ValueError(f"cotangent must have shape ({count},), got {seed.shape}")
    return path, matrix, seed


def _rust_kuramoto_euler_vjp(
    trajectory: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    path, matrix, seed = _validate_vjp(trajectory, coupling, cotangent)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_vjp = getattr(engine, "kuramoto_euler_vjp", None)
    if not callable(rust_vjp):
        raise ImportError("scpn_quantum_engine.kuramoto_euler_vjp is unavailable")

    grad_theta0, grad_omega, grad_coupling = rust_vjp(path, matrix, float(dt), seed)
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def _julia_kuramoto_euler_vjp(
    trajectory: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    path, matrix, seed = _validate_vjp(trajectory, coupling, cotangent)
    from .julia import kuramoto_euler_vjp as julia_vjp

    grad_theta0, grad_omega, grad_coupling = julia_vjp(path, matrix, float(dt), seed)
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def _python_kuramoto_euler_vjp(
    trajectory: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    # Correctness floor — reverse-mode adjoint. With λ_N = cotangent (= ∂L/∂θ_N),
    # λ_n = λ_{n+1} + dt J(θ_n)ᵀ λ_{n+1}, J_jl = K_jl cos(θ_l − θ_j) off-diagonal,
    # J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j). Accumulate ∂L/∂ω = dt Σ_n λ_{n+1} and
    # ∂L/∂K_pq = dt Σ_n λ_{n+1,p} sin(θ_{n,q} − θ_{n,p}). Returns (∂L/∂θ₀, ∂L/∂ω, ∂L/∂K).
    path, matrix, seed = _validate_vjp(trajectory, coupling, cotangent)
    n_steps = path.shape[0] - 1
    count = path.shape[1]
    adjoint = seed.copy()
    grad_omega = np.zeros(count, dtype=np.float64)
    grad_coupling = np.zeros((count, count), dtype=np.float64)
    for step in range(n_steps - 1, -1, -1):
        phases = path[step]
        difference = phases[None, :] - phases[:, None]
        grad_omega += dt * adjoint
        grad_coupling += dt * (adjoint[:, None] * np.sin(difference))
        cosine = matrix * np.cos(difference)
        np.fill_diagonal(cosine, 0.0)
        jacobian = cosine - np.diag(cosine.sum(axis=1))
        adjoint = adjoint + dt * (jacobian.T @ adjoint)
    return (
        np.ascontiguousarray(adjoint, dtype=np.float64),
        np.ascontiguousarray(grad_omega, dtype=np.float64),
        np.ascontiguousarray(grad_coupling, dtype=np.float64),
    )


# The VJP chain mirrors the forward chain (Rust → Julia → Python floor). Their micro-benchmark
# (on a dense random coupling matrix over a fixed step budget) is recorded in
# ``docs/benchmarks/diff_kuramoto_euler_tiers.json``; rerun
# ``python scripts/bench_diff_kuramoto_euler_tiers.py`` when these chains are edited.
_KURAMOTO_EULER_VJP_CHAIN: list[
    tuple[
        str,
        Callable[
            [NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]],
            _VjpResult,
        ],
    ]
] = [
    ("rust", _rust_kuramoto_euler_vjp),
    ("julia", _julia_kuramoto_euler_vjp),
    ("python", _python_kuramoto_euler_vjp),
]


_kuramoto_euler_trajectory_dispatcher = MultiLangDispatcher(_KURAMOTO_EULER_TRAJECTORY_CHAIN)
_kuramoto_euler_vjp_dispatcher = MultiLangDispatcher(_KURAMOTO_EULER_VJP_CHAIN)


def kuramoto_euler_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    r"""Forward networked-Kuramoto Euler trajectory with multi-language dispatch.

    Integrates :math:`\theta_{n+1} = \theta_n + dt(\omega + F(\theta_n))` with
    :math:`F_j(\theta) = \sum_k K_{jk}\sin(\theta_k - \theta_j)` for ``n_steps`` explicit-Euler
    steps, returning the full phase trajectory (the initial state plus every step). The
    trajectory is the forward record the reverse-mode adjoint :func:`kuramoto_euler_vjp`
    consumes.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    dt : float
        The Euler step size.
    n_steps : int
        The number of integration steps (non-negative).

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` float64 trajectory; row 0 is ``theta0``.

    Raises
    ------
    ValueError
        If ``omega``/``coupling`` shapes do not match ``theta0`` or ``n_steps`` is negative.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_euler_trajectory_tier_used`.
    """
    return np.asarray(
        _kuramoto_euler_trajectory_dispatcher(theta0, omega, coupling, dt, n_steps),
        dtype=np.float64,
    )


def kuramoto_euler_vjp(
    trajectory: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    r"""Reverse-mode adjoint of the networked-Kuramoto Euler integrator.

    Given the forward ``trajectory`` from :func:`kuramoto_euler_trajectory` and a cotangent
    :math:`\lambda_N = \partial L/\partial \theta_N` on the final state, propagates the adjoint
    :math:`\lambda_n = \lambda_{n+1} + dt\,J(\theta_n)^\mathsf{T}\lambda_{n+1}` back through the
    dynamics and returns the gradients of the scalar objective ``L`` with respect to every
    input: :math:`\partial L/\partial\theta_0 = \lambda_0`,
    :math:`\partial L/\partial\omega = dt\sum_n \lambda_{n+1}` and
    :math:`\partial L/\partial K_{pq} = dt\sum_n \lambda_{n+1,p}\sin(\theta_{n,q}-\theta_{n,p})`.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` forward trajectory.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K`` (the same one used in the forward pass).
    dt : float
        The Euler step size used in the forward pass.
    cotangent : numpy.ndarray
        One-dimensional ``(N,)`` cotangent on the final phase, ``∂L/∂θ_N``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_theta0, grad_omega, grad_coupling)`` with shapes ``(N,)``, ``(N,)`` and
        ``(N, N)``.

    Raises
    ------
    ValueError
        If ``trajectory`` is not two-dimensional or ``coupling``/``cotangent`` shapes are
        inconsistent with it.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_euler_vjp_tier_used`.
    """
    grad_theta0, grad_omega, grad_coupling = _kuramoto_euler_vjp_dispatcher(
        trajectory, coupling, dt, cotangent
    )
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def last_kuramoto_euler_trajectory_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_euler_trajectory``."""
    return _kuramoto_euler_trajectory_dispatcher.last_tier


def last_kuramoto_euler_vjp_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_euler_vjp``."""
    return _kuramoto_euler_vjp_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("kuramoto_euler_trajectory", _kuramoto_euler_trajectory_dispatcher)
register_dispatcher("kuramoto_euler_vjp", _kuramoto_euler_vjp_dispatcher)


__all__ = [
    "kuramoto_euler_trajectory",
    "kuramoto_euler_vjp",
    "last_kuramoto_euler_trajectory_tier_used",
    "last_kuramoto_euler_vjp_tier_used",
]
