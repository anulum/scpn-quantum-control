# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Independent JAX forward-mode witness for the Kuramoto delay sensitivity
r"""Independent JAX forward-mode witness for the time-delayed Kuramoto delay sensitivity.

:mod:`oscillatools.accel.diff_kuramoto_delay_sensitivity` co-integrates a hand-derived
tangent ``σ(t) = ∂θ(t)/∂τ`` alongside the phases to produce the delay sensitivity ``∂θ_N/∂τ``, and
validates it against a central finite difference of the terminal state over ``τ`` — a numerical
witness limited to roughly ``√eps`` precision. This module supplies the missing *second* witness: it
re-implements the identical continuous-delay method-of-steps **forward map** in JAX and lets JAX's
forward-mode autodiff (:func:`jax.jvp` along the scalar ``τ``) differentiate it. Only the
differentiation differs, so agreement between the hand-co-integrated ``σ_N`` and the JAX tangent at
(double-precision) machine tolerance is evidence the tangent recurrence is the exact derivative of the
realised discrete map, not merely close to a coarse estimate.

The forward map reads the lagged phase ``θ(t − τ)`` from a **linear interpolation** of the running
trajectory exactly as the production integrator does. Rather than branch on a negative lag (reading
the constant history) as the production module does, it prepends ``⌈τ/dt⌉ + 2`` constant-history rows
so every stage lag brackets two stored rows: interpolating between two identical history rows returns
the history phase with zero slope and zero tangent, which is numerically identical to the production
negative-lag branch. Because the interpolation index is a floor of the ``τ``-dependent position and
``jax.numpy.floor`` has zero derivative, the JAX tangent of the lagged phase is
``σ_lag − (θ[i+1] − θ[i])/dt`` — precisely the production module's total ``τ``-derivative. The delay
must not be a half-integer multiple of ``dt`` (a grid node where the interpolant has a kink); the
production validator enforces that, and the witness is exercised only at interior delays.

JAX is an optional dependency; importing this module skips any collecting test when JAX is absent,
matching the framework-overlay lane that provisions ``jax[cpu]``. The forward enables 64-bit JAX so
the reference matches the float64 production integrator.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Match the float64 production integrator so the reference tangent is limited by the shared
# arithmetic, not by JAX's default single precision.
jax.config.update("jax_enable_x64", True)

Array = jax.Array


def _history_prefix_length(delay: float, dt: float) -> int:
    r"""Number of constant-history rows to prepend so every stage lag brackets two stored rows.

    The deepest lookback is the first step's opening stage at ``t = 0``, whose lag reaches back
    ``τ`` in time, i.e. ``⌈τ/dt⌉`` grid rows; two extra rows cover the bracketing ``index + 1`` and
    floating-point slack. Computed from the concrete ``delay`` so it is a static array dimension
    under :func:`jax.jvp`.
    """
    return int(math.ceil(delay / dt)) + 2


def _delayed_force(current: Array, lagged: Array, omega: Array, coupling: Array) -> Array:
    r"""Delayed networked Kuramoto field ``θ̇_j = ω_j + Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))``.

    Mirrors :func:`oscillatools.accel.diff_kuramoto_delay_sensitivity._force` with
    ``difference[j, k] = θ_k(t−τ) − θ_j(t)``.
    """
    difference = lagged[jnp.newaxis, :] - current[:, jnp.newaxis]
    return omega + (coupling * jnp.sin(difference)).sum(axis=1)


def delayed_terminal_continuous_delay(
    history: Array,
    omega: Array,
    coupling: Array,
    delay: Array,
    dt: float,
    n_steps: int,
    prefix_length: int,
) -> Array:
    r"""Terminal phase ``θ_N`` of the continuous-delay method-of-steps RK4 on a constant history.

    Integrates ``θ̇_j = ω_j + Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))`` from a constant history
    ``θ(s) = θ_0`` (``s ≤ 0``) with the lagged phase read by linear interpolation of the running
    trajectory. Rows ``0 … prefix_length`` of the buffer hold the constant history; row
    ``prefix_length`` is ``θ(t = 0)`` and the integration writes rows ``prefix_length + 1 …``. The
    ``delay`` is a traced scalar under :func:`jax.jvp`; ``prefix_length`` is static so it can size the
    buffer.
    """
    count = history.shape[0]
    total_rows = prefix_length + n_steps + 1
    phases = jnp.zeros((total_rows, count), dtype=jnp.float64)
    for row in range(prefix_length + 1):
        phases = phases.at[row].set(history)

    def lagged_phase(stage_time: float, buffer: Array) -> Array:
        position = prefix_length + (stage_time - delay) / dt
        index_float = jnp.floor(position + 1e-12)
        fraction = position - index_float
        index = index_float.astype(jnp.int64)
        lower = buffer[index]
        upper = buffer[index + 1]
        return (1.0 - fraction) * lower + fraction * upper

    for step in range(n_steps):
        time = step * dt
        current = phases[prefix_length + step]
        k1 = _delayed_force(current, lagged_phase(time, phases), omega, coupling)
        k2 = _delayed_force(
            current + 0.5 * dt * k1, lagged_phase(time + 0.5 * dt, phases), omega, coupling
        )
        k3 = _delayed_force(
            current + 0.5 * dt * k2, lagged_phase(time + 0.5 * dt, phases), omega, coupling
        )
        k4 = _delayed_force(current + dt * k3, lagged_phase(time + dt, phases), omega, coupling)
        new_row = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        phases = phases.at[prefix_length + step + 1].set(new_row)

    return phases[prefix_length + n_steps]


def delay_sensitivity_via_jvp(
    history: Array,
    omega: Array,
    coupling: Array,
    delay: float,
    dt: float,
    n_steps: int,
) -> tuple[Array, Array]:
    r"""Terminal phases and their JAX forward-mode delay tangent ``(θ_N, ∂θ_N/∂τ)``.

    Differentiates :func:`delayed_terminal_continuous_delay` along the scalar delay ``τ`` with a
    single :func:`jax.jvp` (a seed tangent of ``1.0`` on ``τ``), returning the terminal phases and the
    full ``∂θ_N/∂τ`` vector in one forward pass — the independent witness for the hand-co-integrated
    sensitivity.
    """
    prefix_length = _history_prefix_length(delay, dt)
    history_j = jnp.asarray(history, dtype=jnp.float64)
    omega_j = jnp.asarray(omega, dtype=jnp.float64)
    coupling_j = jnp.asarray(coupling, dtype=jnp.float64)

    def terminal(tau: Array) -> Array:
        return delayed_terminal_continuous_delay(
            history_j, omega_j, coupling_j, tau, dt, n_steps, prefix_length
        )

    theta_final, dtheta_dtau = jax.jvp(
        terminal, (jnp.asarray(delay, dtype=jnp.float64),), (jnp.asarray(1.0, dtype=jnp.float64),)
    )
    return theta_final, dtheta_dtau
