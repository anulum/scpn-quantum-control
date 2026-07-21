# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 Kuramoto dynamics with per-trial gated coupling
"""Differentiable Kuramoto RK4 integrator with a **per-trial** coupling matrix.

v2 gates the coupling by the control code (audit AUD-7 / KYMA v2 fix 1), so each
trial integrates under its own ``K_eff`` rather than a single shared matrix::

    dθ_i/dt = ω_i + Σ_j K_ij(trial) sin(θ_j − θ_i)

The integrator is therefore batched over ``(θ0, ω, K)`` triples; the map is
smooth and reverse-mode differentiable for motif training. ``float32`` keeps the
graph light on a memory-constrained host — the readout quantisation (π/2 bins)
and order-parameter tolerances are far coarser than float32 round-off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def kuramoto_rhs_batched(theta: jax.Array, omega: jax.Array, coupling: jax.Array) -> jax.Array:
    """Right-hand side ``dθ/dt`` with a per-trial coupling matrix.

    Args:
        theta: ``(batch, n)`` phases.
        omega: ``(batch, n)`` natural-frequency drive.
        coupling: ``(batch, n, n)`` per-trial symmetric coupling, zero diagonal.

    Returns:
        ``(batch, n)`` phase velocities.
    """
    # diff[b, i, j] = θ_j − θ_i  → (batch, n_i, n_j)
    diff = theta[:, None, :] - theta[:, :, None]
    interaction = jnp.sum(coupling * jnp.sin(diff), axis=2)
    return omega + interaction


def integrate_kuramoto_batched(
    theta0: jax.Array,
    omega: jax.Array,
    coupling: jax.Array,
    dt: float,
    steps: int,
) -> jax.Array:
    """Integrate the batch with fixed-step RK4 under per-trial coupling.

    Args:
        theta0: ``(batch, n)`` initial phases.
        omega: ``(batch, n)`` natural-frequency drive.
        coupling: ``(batch, n, n)`` per-trial symmetric coupling.
        dt: fixed step size.
        steps: number of RK4 steps (horizon ``T = steps * dt``).

    Returns:
        ``(batch, n)`` final phases wrapped to ``(-π, π]``.
    """

    def rk4_step(theta: jax.Array, _: None) -> tuple[jax.Array, None]:
        k1 = kuramoto_rhs_batched(theta, omega, coupling)
        k2 = kuramoto_rhs_batched(theta + 0.5 * dt * k1, omega, coupling)
        k3 = kuramoto_rhs_batched(theta + 0.5 * dt * k2, omega, coupling)
        k4 = kuramoto_rhs_batched(theta + dt * k3, omega, coupling)
        return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    final, _ = jax.lax.scan(rk4_step, theta0, None, length=steps)
    return jnp.angle(jnp.exp(1j * final))


def order_parameter(theta: jax.Array, members: jax.Array) -> jax.Array:
    """Kuramoto order parameter ``R = |mean exp(iθ)|`` over a fixed member set.

    Args:
        theta: ``(batch, n)`` phases.
        members: ``(m,)`` integer indices of the oscillators in the set.

    Returns:
        ``(batch,)`` order parameter per trial (1 in-phase, ~0 anti-phase).
    """
    selected = theta[:, members]
    return jnp.abs(jnp.mean(jnp.exp(1j * selected), axis=1))


def phase_label(theta: jax.Array, readout_oscillator: int, n_bins: int) -> jax.Array:
    """Quantise a readout oscillator's final phase into ``n_bins`` equal bins.

    ``label = floor((φ mod 2π) / (2π / n_bins)) ∈ {0, …, n_bins−1}`` — the
    data-dependent, non-separable readout of the v2 probe. The phase is taken
    modulo ``2π`` so the lab-frame angle maps to a class in ``[0, n_bins)``.

    Args:
        theta: ``(batch, n)`` final phases.
        readout_oscillator: index of the oscillator whose phase is read.
        n_bins: number of equal angular bins (chance = ``1 / n_bins``).

    Returns:
        ``(batch,)`` integer class labels.
    """
    phi = jnp.mod(theta[:, readout_oscillator], 2.0 * jnp.pi)
    width = (2.0 * jnp.pi) / n_bins
    idx = jnp.floor(phi / width).astype(jnp.int32)
    # Guard the φ == 2π edge (mod can return exactly 2π in float) → clamp to last bin.
    return jnp.clip(idx, 0, n_bins - 1)
