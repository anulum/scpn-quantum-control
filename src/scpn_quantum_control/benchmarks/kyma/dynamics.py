# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA Kuramoto dynamics
"""Differentiable Kuramoto RK4 integrator and order-parameter readout.

Canonical model, batched over trials::

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)

``K`` is a shared symmetric coupling matrix (zero diagonal); ``ω`` is a
per-trial, input-conditioned natural-frequency drive. Integration is
fixed-step RK4 over a fixed horizon, so the whole map ``(θ0, ω, K) → θ_T`` is
smooth and reverse-mode differentiable for motif training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# float32 (the JAX default) keeps the training graph light enough to compile and
# run on a memory-constrained host; the probe's order-parameter tolerances
# (ε = 0.15) are far coarser than float32 round-off, so precision is ample.


def kuramoto_rhs(theta: jax.Array, omega: jax.Array, coupling: jax.Array) -> jax.Array:
    """Right-hand side ``dθ/dt`` for a batch of Kuramoto states.

    Args:
        theta: ``(batch, n)`` phases.
        omega: ``(batch, n)`` natural-frequency drive.
        coupling: ``(n, n)`` symmetric coupling matrix, zero diagonal.

    Returns:
        ``(batch, n)`` phase velocities.
    """
    # pairwise phase differences θ_j − θ_i → (batch, n_i, n_j)
    diff = theta[:, None, :] - theta[:, :, None]
    interaction = jnp.sum(coupling[None, :, :] * jnp.sin(diff), axis=2)
    return omega + interaction


def integrate_kuramoto(
    theta0: jax.Array,
    omega: jax.Array,
    coupling: jax.Array,
    dt: float,
    steps: int,
) -> jax.Array:
    """Integrate the Kuramoto batch with fixed-step RK4.

    Args:
        theta0: ``(batch, n)`` initial phases.
        omega: ``(batch, n)`` natural-frequency drive.
        coupling: ``(n, n)`` symmetric coupling.
        dt: fixed step size.
        steps: number of RK4 steps (horizon ``T = steps * dt``).

    Returns:
        ``(batch, n)`` final phases wrapped to ``(-π, π]``.
    """

    def rk4_step(theta: jax.Array, _: None) -> tuple[jax.Array, None]:
        k1 = kuramoto_rhs(theta, omega, coupling)
        k2 = kuramoto_rhs(theta + 0.5 * dt * k1, omega, coupling)
        k3 = kuramoto_rhs(theta + 0.5 * dt * k2, omega, coupling)
        k4 = kuramoto_rhs(theta + dt * k3, omega, coupling)
        return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    final, _ = jax.lax.scan(rk4_step, theta0, None, length=steps)
    return jnp.angle(jnp.exp(1j * final))


def cluster_order_parameter(theta: jax.Array, members: jax.Array) -> jax.Array:
    """Kuramoto order parameter ``R`` over a fixed set of oscillators.

    ``R = |mean_{i∈S} exp(iθ_i)| ∈ [0, 1]``: 1 when the members are perfectly
    in phase, ~0 when two equal sub-groups sit π apart (anti-phase lock).

    Args:
        theta: ``(batch, n)`` phases.
        members: ``(m,)`` integer indices of the oscillators in the set.

    Returns:
        ``(batch,)`` order parameter per trial.
    """
    selected = theta[:, members]
    z = jnp.mean(jnp.exp(1j * selected), axis=1)
    return jnp.abs(z)
