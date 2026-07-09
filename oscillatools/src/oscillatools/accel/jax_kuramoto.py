# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX autodiff accelerator tier for the networked Kuramoto RK4 integrator
r"""A JAX autodiff tier for the networked-Kuramoto RK4 integrator.

The toolkit's forward integrators dispatch Rust → Julia → NumPy, and each has a *hand-written*
reverse-mode adjoint. This module adds a fourth kind of tier: the same RK4 solve expressed in JAX,
where the gradient is obtained by **automatic differentiation** rather than by a hand-derived scheme,
and the whole solve runs on whatever accelerator JAX has selected (a CUDA GPU when one is present).

The tier exposes both single-solve and batched-ensemble capabilities:

* :func:`jax_kuramoto_rk4_trajectory` — the forward RK4 trajectory, bit-for-bit faithful to the
  production integrator (with ``jax_enable_x64``, the GPU result matches the Rust tier to machine
  precision on the reference cases; a different device or a very large network may differ at the
  reduction-ordering level, so parity is asserted under a tolerance, not as bit-identity).
* :func:`jax_kuramoto_rk4_gradient` — the reverse-mode gradient of a scalar objective
  :math:`L` (with :math:`\partial L/\partial\theta_N` supplied as the ``cotangent``) with respect to
  the initial phases, the natural frequencies and the coupling matrix, obtained from :func:`jax.vjp`
  of the forward solve. It matches the hand-derived :func:`~oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_vjp`
  to machine precision — the autodiff tier both *verifies* the hand-written adjoint and supplies the
  same gradient for objectives where a hand derivation would be laborious.
* :func:`jax_kuramoto_rk4_ensemble` / :func:`jax_kuramoto_rk4_ensemble_gradient` — the ``vmap`` batched
  counterparts that forward-solve and differentiate a whole batch of initial conditions in a single
  accelerator call, a vectorisation of the entire solve the NumPy and Rust tiers cannot express (they
  would loop over the ensemble). Each member is identical to its single-initial-condition counterpart.

This tier is **opt-in**: it is a directly callable accelerated path, not a member of the default
dispatch chain, so the default :func:`~oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_trajectory`
still serves the Rust tier and every existing test is byte-unchanged. It requires JAX
(``oscillatools[jax]``) and raises :class:`ImportError` with an install hint when JAX is
absent — the exception the multi-language dispatcher treats as "fall through", so a later slice may
place this tier in a size-aware accelerated chain without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_rk4 import _validate_forward

_VjpResult = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class _Backend:
    """The cached JAX backend: the imported modules plus the JIT-compiled single and ensemble solves.

    ``trajectory`` / ``gradient`` are the single-initial-condition forward solve and its reverse-mode
    gradient; ``ensemble_trajectory`` / ``ensemble_gradient`` are their ``vmap`` counterparts that map
    over a batch of initial conditions in a single accelerator call.
    """

    jax: Any
    jnp: Any
    trajectory: Any
    gradient: Any
    ensemble_trajectory: Any
    ensemble_gradient: Any


#: Lazily built backend, cached after the first successful load so the JIT-compiled kernels are reused.
_BACKEND: _Backend | None = None


def _load_backend() -> _Backend:
    """Return the cached JAX backend, building it on first use.

    Enables 64-bit precision (a global JAX configuration flag, hence set lazily rather than at import)
    and JIT-compiles the single-trajectory solve, its reverse-mode gradient, and their ``vmap`` ensemble
    counterparts, each with the step count as a static argument so JAX caches one compiled kernel per
    step budget and array shape.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError(
            "the JAX Kuramoto tier requires JAX; install oscillatools[jax]"
        ) from error
    jax.config.update("jax_enable_x64", True)

    def force(theta: Any, coupling: Any) -> Any:
        difference = theta[None, :] - theta[:, None]
        return jnp.sum(coupling * jnp.sin(difference), axis=1)

    def rk4_step(theta: Any, omega: Any, coupling: Any, dt: float) -> Any:
        k1 = omega + force(theta, coupling)
        k2 = omega + force(theta + 0.5 * dt * k1, coupling)
        k3 = omega + force(theta + 0.5 * dt * k2, coupling)
        k4 = omega + force(theta + dt * k3, coupling)
        return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(theta0: Any, omega: Any, coupling: Any, dt: float, n_steps: int) -> Any:
        def body(carry: Any, _: Any) -> tuple[Any, Any]:
            advanced = rk4_step(carry, omega, coupling, dt)
            return advanced, advanced

        if n_steps == 0:
            return theta0[None, :]
        _, stepped = jax.lax.scan(body, theta0, None, length=n_steps)
        return jnp.concatenate([theta0[None, :], stepped], axis=0)

    def gradient(
        theta0: Any, omega: Any, coupling: Any, dt: float, n_steps: int, cotangent: Any
    ) -> Any:
        def final_state(initial: Any, frequency: Any, matrix: Any) -> Any:
            return solve(initial, frequency, matrix, dt, n_steps)[-1]

        _, pullback = jax.vjp(final_state, theta0, omega, coupling)
        return pullback(cotangent)

    _BACKEND = _Backend(
        jax=jax,
        jnp=jnp,
        trajectory=jax.jit(solve, static_argnums=(4,)),
        gradient=jax.jit(gradient, static_argnums=(4,)),
        ensemble_trajectory=jax.jit(
            jax.vmap(solve, in_axes=(0, None, None, None, None)), static_argnums=(4,)
        ),
        ensemble_gradient=jax.jit(
            jax.vmap(gradient, in_axes=(0, None, None, None, None, 0)), static_argnums=(4,)
        ),
    )
    return _BACKEND


def jax_kuramoto_rk4_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    r"""Forward networked-Kuramoto RK4 trajectory evaluated on the JAX accelerator tier.

    Integrates :math:`\dot\theta = \omega + F(\theta)` with
    :math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)` by classical fourth-order Runge–Kutta for
    ``n_steps`` steps, returning the initial state plus every step — the identical contract to
    :func:`~oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_trajectory`, but computed in JAX
    (on the GPU when one is available) at 64-bit precision.

    Parameters
    ----------
    theta0 : numpy.ndarray
        The initial phases ``θ_0`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step size.
    n_steps : int
        The number of RK4 steps (``≥ 0``).

    Returns
    -------
    numpy.ndarray
        The ``(n_steps + 1, N)`` phase trajectory (row 0 is ``theta0``).

    Raises
    ------
    ValueError
        If ``omega`` or ``coupling`` does not match ``theta0``'s order, or ``n_steps`` is negative.
    ImportError
        If JAX is not installed.
    """
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    backend = _load_backend()
    jnp = backend.jnp
    result = backend.trajectory(
        jnp.asarray(phases), jnp.asarray(frequencies), jnp.asarray(matrix), float(dt), int(n_steps)
    )
    return np.asarray(result, dtype=np.float64)


def jax_kuramoto_rk4_gradient(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    r"""Reverse-mode (autodiff) gradient of the RK4 solve on the JAX accelerator tier.

    Given a cotangent :math:`\lambda_N = \partial L/\partial\theta_N` on the final phase, returns the
    gradients of the scalar objective ``L`` with respect to the initial phases, the natural
    frequencies and the coupling matrix — the same contract as the hand-derived
    :func:`~oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_vjp`, here obtained by
    :func:`jax.vjp` of the forward solve rather than a hand-written adjoint. The two agree to machine
    precision, so this tier verifies the hand-written scheme and generalises to objectives whose
    adjoint would be laborious to derive.

    Parameters
    ----------
    theta0 : numpy.ndarray
        The initial phases ``θ_0`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step size.
    n_steps : int
        The number of RK4 steps (``≥ 1``).
    cotangent : numpy.ndarray
        The cotangent ``∂L/∂θ_N`` on the final phase (length ``N``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_theta0, grad_omega, grad_coupling)`` with shapes ``(N,)``, ``(N,)`` and ``(N, N)``.

    Raises
    ------
    ValueError
        If any shape is inconsistent, or ``n_steps`` is not positive.
    ImportError
        If JAX is not installed.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive for a gradient, got {n_steps}")
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    seed = np.ascontiguousarray(cotangent, dtype=np.float64)
    if seed.shape != (phases.size,):
        raise ValueError(f"cotangent must have shape ({phases.size},), got {seed.shape}")
    backend = _load_backend()
    jnp = backend.jnp
    grad_theta0, grad_omega, grad_coupling = backend.gradient(
        jnp.asarray(phases),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        int(n_steps),
        jnp.asarray(seed),
    )
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def _validate_ensemble(
    theta0_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta0_batch, omega, coupling)`` after batch-shape validation."""
    batch = np.ascontiguousarray(theta0_batch, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[0] < 1:
        raise ValueError(
            f"theta0_batch must be a two-dimensional (B, N) array with B >= 1, got shape {batch.shape}"
        )
    _, frequencies, matrix = _validate_forward(batch[0], omega, coupling, n_steps)
    return batch, frequencies, matrix


def jax_kuramoto_rk4_ensemble(
    theta0_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    r"""Batched forward RK4 trajectories for a whole ensemble of initial conditions.

    Solves ``B`` initial conditions that share the same frequencies and coupling in a single
    accelerator call by :func:`jax.vmap` over the batch axis — a vectorisation of the *entire* solve
    the NumPy and Rust tiers cannot express (they would loop over the ensemble). Each member is
    identical to the single-initial-condition :func:`jax_kuramoto_rk4_trajectory`.

    Parameters
    ----------
    theta0_batch : numpy.ndarray
        The ``(B, N)`` batch of initial phases.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``), shared across the batch.
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``, shared across the batch.
    dt : float
        The RK4 step size.
    n_steps : int
        The number of RK4 steps (``≥ 0``).

    Returns
    -------
    numpy.ndarray
        The ``(B, n_steps + 1, N)`` batch of phase trajectories.

    Raises
    ------
    ValueError
        If ``theta0_batch`` is not ``(B, N)`` with ``B ≥ 1``, the frequencies or coupling are
        inconsistent, or ``n_steps`` is negative.
    ImportError
        If JAX is not installed.
    """
    batch, frequencies, matrix = _validate_ensemble(theta0_batch, omega, coupling, n_steps)
    backend = _load_backend()
    jnp = backend.jnp
    result = backend.ensemble_trajectory(
        jnp.asarray(batch), jnp.asarray(frequencies), jnp.asarray(matrix), float(dt), int(n_steps)
    )
    return np.asarray(result, dtype=np.float64)


def jax_kuramoto_rk4_ensemble_gradient(
    theta0_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    cotangent_batch: NDArray[np.float64],
) -> _VjpResult:
    r"""Batched reverse-mode gradients for a whole ensemble of initial conditions.

    For each ensemble member ``i`` with its own cotangent ``∂L_i/∂θ_N``, returns the per-member
    gradients with respect to the initial phases, the frequencies and the coupling, computed in a
    single accelerator call by :func:`jax.vmap` over the batch. Each member is identical to the
    single-initial-condition :func:`jax_kuramoto_rk4_gradient`.

    Parameters
    ----------
    theta0_batch : numpy.ndarray
        The ``(B, N)`` batch of initial phases.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``), shared across the batch.
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``, shared across the batch.
    dt : float
        The RK4 step size.
    n_steps : int
        The number of RK4 steps (``≥ 1``).
    cotangent_batch : numpy.ndarray
        The ``(B, N)`` batch of cotangents on the final phase.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_theta0, grad_omega, grad_coupling)`` with shapes ``(B, N)``, ``(B, N)`` and
        ``(B, N, N)`` — one gradient per ensemble member.

    Raises
    ------
    ValueError
        If a shape is inconsistent, or ``n_steps`` is not positive.
    ImportError
        If JAX is not installed.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive for a gradient, got {n_steps}")
    batch, frequencies, matrix = _validate_ensemble(theta0_batch, omega, coupling, n_steps)
    seeds = np.ascontiguousarray(cotangent_batch, dtype=np.float64)
    if seeds.shape != batch.shape:
        raise ValueError(
            f"cotangent_batch must match theta0_batch shape {batch.shape}, got {seeds.shape}"
        )
    backend = _load_backend()
    jnp = backend.jnp
    grad_theta0, grad_omega, grad_coupling = backend.ensemble_gradient(
        jnp.asarray(batch),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        int(n_steps),
        jnp.asarray(seeds),
    )
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


__all__ = [
    "jax_kuramoto_rk4_ensemble",
    "jax_kuramoto_rk4_ensemble_gradient",
    "jax_kuramoto_rk4_gradient",
    "jax_kuramoto_rk4_trajectory",
]
