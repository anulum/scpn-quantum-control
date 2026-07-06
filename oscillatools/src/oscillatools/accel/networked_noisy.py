# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Polyglot stochastic (Euler–Maruyama) networked-Kuramoto trajectory
r"""Polyglot stochastic (Euler–Maruyama) networked-Kuramoto forward trajectory.

Integrates the Langevin equation ``dθ_j = (ω_j + F_j(θ)) dt + \sqrt{2D}\,dW_j`` — the noisy Kuramoto
model with additive white phase noise of intensity ``D`` — for the *networked* coupling force
:math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)`, by the Euler–Maruyama step
``θ ← θ + (ω + F(θ)) dt + \sqrt{2 D dt}\,ξ``.

This is the accelerated, networked-force special case of the general
:func:`~oscillatools.accel.kuramoto_noisy.integrate_noisy_kuramoto`: where that function composes an
arbitrary Python force callable (and so runs on the pure-Python floor), this function pins the force
to the networked coupling matrix so the whole Euler–Maruyama hot loop can run in a compiled tier. The
forward integration dispatches across a Rust → Julia → Python floor tier chain (measured fastest
first, recorded on :func:`last_networked_noisy_trajectory_tier_used`).

Cross-language RNG reproduction is deliberately avoided. Rather than reproduce NumPy's PCG64 stream in
Rust and Julia — fragile and not worth the risk — the reproducible Wiener increments are drawn *once*
in this wrapper from a seeded :class:`numpy.random.Generator` and passed into whichever tier serves the
call, so every tier consumes the exact same ``(n_steps, N)`` standard-normal array. The tiers then
differ only in the coupling-force summation order (a scalar loop in the compiled tiers versus NumPy's
vectorised reduction), so they are *tolerance-parity*, not bit-identical. It parallels the deterministic
accelerated integrators (:func:`~oscillatools.accel.networked_inertial.networked_inertial_trajectory`
and :func:`~oscillatools.accel.networked_delayed.networked_delayed_trajectory`), completing the polyglot
forward-integrator family with the stochastic model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher
from .kuramoto_noisy import NoisyKuramotoRun, noisy_kuramoto_step
from .order_parameter_observables import order_parameter

_Forward = tuple[NDArray[np.float64], NDArray[np.float64]]


def _force(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Instantaneous networked coupling force ``F_j = Σ_k K_jk sin(θ_k − θ_j)`` (pure-NumPy floor)."""
    difference = phases[None, :] - phases[:, None]
    return np.asarray((coupling * np.sin(difference)).sum(axis=1), dtype=np.float64)


def _validate_noisy_state(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Return contiguous ``(theta0, omega, coupling)`` and ``N`` after shape validation."""
    theta = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("omega must be a non-empty one-dimensional array")
    if theta.shape != frequencies.shape:
        raise ValueError(f"theta0 must match omega shape {frequencies.shape}, got {theta.shape}")
    count = frequencies.size
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must be a square matrix of order {count}, got {matrix.shape}")
    return theta, frequencies, matrix, count


def _rust_networked_noisy_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    dt: float,
    noise: NDArray[np.float64],
) -> _Forward:
    theta, frequencies, matrix, _count = _validate_noisy_state(theta0, omega, coupling)
    increments = np.ascontiguousarray(noise, dtype=np.float64)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_forward = getattr(engine, "kuramoto_noisy_trajectory", None)
    if not callable(rust_forward):
        raise ImportError("scpn_quantum_engine.kuramoto_noisy_trajectory is unavailable")
    series, terminal = rust_forward(
        theta, frequencies, matrix, float(diffusion), float(dt), increments
    )
    return (
        np.ascontiguousarray(series, dtype=np.float64),
        np.ascontiguousarray(terminal, dtype=np.float64),
    )


def _julia_networked_noisy_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    dt: float,
    noise: NDArray[np.float64],
) -> _Forward:
    theta, frequencies, matrix, _count = _validate_noisy_state(theta0, omega, coupling)
    increments = np.ascontiguousarray(noise, dtype=np.float64)
    from .julia import kuramoto_noisy_trajectory as julia_forward

    series, terminal = julia_forward(
        theta, frequencies, matrix, float(diffusion), float(dt), increments
    )
    return (
        np.ascontiguousarray(series, dtype=np.float64),
        np.ascontiguousarray(terminal, dtype=np.float64),
    )


def _python_networked_noisy_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    dt: float,
    noise: NDArray[np.float64],
) -> _Forward:
    """Pure-Python floor: the audited Euler–Maruyama step bound to the networked force."""
    theta, frequencies, matrix, _count = _validate_noisy_state(theta0, omega, coupling)
    increments = np.ascontiguousarray(noise, dtype=np.float64)
    n_steps = increments.shape[0]

    def force(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _force(values, matrix)

    series = np.empty(n_steps, dtype=np.float64)
    current = theta
    for step in range(n_steps):
        current = noisy_kuramoto_step(current, frequencies, force, diffusion, dt, increments[step])
        series[step] = order_parameter(current)
    return series, np.ascontiguousarray(current, dtype=np.float64)


_NETWORKED_NOISY_TRAJECTORY_CHAIN: list[tuple[str, Callable[..., _Forward]]] = [
    ("rust", _rust_networked_noisy_trajectory),
    ("julia", _julia_networked_noisy_trajectory),
    ("python", _python_networked_noisy_trajectory),
]
_networked_noisy_trajectory_dispatcher = MultiLangDispatcher(_NETWORKED_NOISY_TRAJECTORY_CHAIN)


def networked_noisy_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    *,
    dt: float,
    n_steps: int,
    seed: int,
    settle_steps: int | None = None,
) -> NoisyKuramotoRun:
    r"""Integrate the noisy networked Kuramoto model by a seeded Euler–Maruyama scheme.

    Advances the Langevin equation ``dθ = (ω + F(θ)) dt + \sqrt{2D}\,dW`` with the networked coupling
    force :math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)`, recording the order parameter after
    every step and averaging over the trailing settle window for the stationary coherence. This is the
    accelerated, networked-force special case of
    :func:`~oscillatools.accel.kuramoto_noisy.integrate_noisy_kuramoto`; the forward integration
    dispatches across a Rust → Julia → Python floor tier chain (measured fastest first, recorded on
    :func:`last_networked_noisy_trajectory_tier_used`). The Wiener increments are drawn once from a
    :class:`numpy.random.Generator` seeded with ``seed`` and shared across tiers, so the run is
    reproducible and the tiers are tolerance-parity (they share the Euler–Maruyama arithmetic and the
    noise, and differ only in the coupling-force summation order).

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies ``ω``.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    diffusion : float
        The diffusion / noise intensity ``D`` (``≥ 0``).
    dt : float
        The Euler–Maruyama time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the order-parameter series has ``n_steps`` samples.
    seed : int
        The seed of the noise generator; identical seeds reproduce the run.
    settle_steps : int, optional
        The trailing window averaged for the stationary order parameter; defaults to the final half of
        ``n_steps``. Must lie in ``[1, n_steps]``.

    Returns
    -------
    NoisyKuramotoRun
        The order-parameter series, terminal phases and the settle-window coherence statistics.

    Raises
    ------
    ValueError
        If the state shapes are inconsistent, or ``diffusion``/``dt``/``n_steps``/``settle_steps`` are
        out of range.
    """
    theta, frequencies, matrix, count = _validate_noisy_state(theta0, omega, coupling)
    if diffusion < 0.0:
        raise ValueError(f"diffusion must be non-negative, got {diffusion}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    resolved_settle = settle_steps if settle_steps is not None else max(1, n_steps // 2)
    if not 1 <= resolved_settle <= n_steps:
        raise ValueError(f"settle_steps must be in [1, {n_steps}], got {resolved_settle}")

    noise: NDArray[np.float64] = np.random.default_rng(seed).standard_normal((n_steps, count))
    series, terminal = _networked_noisy_trajectory_dispatcher(
        theta, frequencies, matrix, diffusion, dt, noise
    )
    settle = series[n_steps - resolved_settle :]
    return NoisyKuramotoRun(
        order_parameter_series=series,
        terminal_phases=terminal,
        mean_order_parameter=float(settle.mean()),
        order_parameter_std=float(settle.std()),
        diffusion=float(diffusion),
        settle_steps=resolved_settle,
    )


def last_networked_noisy_trajectory_tier_used() -> str | None:
    """The tier (``'rust'``/``'julia'``/``'python'``) that served the last dispatched forward.

    ``None`` before the first dispatched call.
    """
    return _networked_noisy_trajectory_dispatcher.last_tier


register_dispatcher("networked_noisy_trajectory", _networked_noisy_trajectory_dispatcher)


__all__ = [
    "last_networked_noisy_trajectory_tier_used",
    "networked_noisy_trajectory",
]
