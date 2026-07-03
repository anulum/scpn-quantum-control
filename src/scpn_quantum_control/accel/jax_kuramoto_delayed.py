# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX autodiff accelerator tier for the delayed networked Kuramoto integrator
r"""A JAX autodiff tier for the time-delayed networked-Kuramoto method-of-steps integrator.

The delay-differential Kuramoto model :math:`\dot\theta_j(t) = \omega_j + \sum_k K_{jk}\sin(\theta_k(t-\tau) - \theta_j(t))`
is integrated by :func:`~scpn_quantum_control.accel.kuramoto_delayed.integrate_delayed_kuramoto`, a
delay-aware method-of-steps RK4 whose running phase grid doubles as the history buffer, and its
gradient is supplied by the *hand-written* forward-mode sensitivity in
:mod:`~scpn_quantum_control.accel.diff_kuramoto_delayed`. This module adds the JAX counterpart of that
integrator, where the gradient comes from **automatic differentiation** of the discrete method-of-steps
map rather than a hand-derived scheme, and the whole solve runs on whatever accelerator JAX has
selected (a CUDA GPU when one is present).

The delay makes the state a history rather than a point, so the forward solve is expressed as a
:func:`jax.lax.scan` whose carry is the sliding window of the last :math:`\tau/\Delta t + 1` phase
vectors — the exact span the method-of-steps stages read (the current phase, the lagged phase one
delay back, and their half-step mean). Because :math:`\tau` is an integer number of steps the window
is a fixed size, so the scan is a static-shape kernel that JAX can compile and vectorise.

The tier exposes both single-solve and batched-ensemble capabilities:

* :func:`jax_kuramoto_delayed_trajectory` — the forward method-of-steps trajectory, faithful to the
  production integrator (with ``jax_enable_x64`` the GPU result matches it to machine precision on the
  reference cases; a different device or a very large network may differ at the reduction-ordering
  level, so parity is asserted under a tolerance, not as bit-identity).
* :func:`jax_kuramoto_delayed_gradient` — the reverse-mode gradient of a scalar objective :math:`L`
  (with :math:`\partial L/\partial\theta_N` supplied as the ``cotangent``) with respect to the full
  initial history, the natural frequencies and the coupling matrix, obtained from :func:`jax.vjp` of
  the forward solve. It matches the hand-derived
  :func:`~scpn_quantum_control.accel.diff_kuramoto_delayed.delayed_terminal_value_and_grad` to machine
  precision — the autodiff tier both *verifies* the hand-written sensitivity and supplies the same
  gradient for objectives where a hand derivation would be laborious. The delay :math:`\tau` is
  structural (an integer step count) and is not a differentiable parameter here; its dedicated
  sensitivity lives in :mod:`~scpn_quantum_control.accel.diff_kuramoto_delay_sensitivity`.
* :func:`jax_kuramoto_delayed_ensemble` / :func:`jax_kuramoto_delayed_ensemble_gradient` — the ``vmap``
  batched counterparts that forward-solve and differentiate a whole batch of initial histories in a
  single accelerator call, a vectorisation of the entire delayed solve the NumPy and Rust tiers cannot
  express (they would loop over the ensemble). Each member is identical to its single-history counterpart.

This tier is **opt-in**: it is a directly callable accelerated path, not a member of any dispatch chain
(the delayed integrator has no multi-language chain), so every existing test is byte-unchanged. It
requires JAX (``scpn-quantum-control[jax]``) and raises :class:`ImportError` with an install hint when
JAX is absent — the exception the multi-language dispatcher treats as "fall through".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_delayed import _validate as _validate_delayed

#: ``(grad_initial_history, grad_omega, grad_coupling)`` with shapes ``(delay_steps + 1, N)``,
#: ``(N,)`` and ``(N, N)``.
_DelayedVjpResult = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class _DelayedBackend:
    """The cached JAX backend for the delayed tier: the imported modules plus the JIT-compiled solves.

    ``trajectory`` / ``gradient`` are the single-history forward solve and its reverse-mode gradient;
    ``ensemble_trajectory`` / ``ensemble_gradient`` are their ``vmap`` counterparts that map over a
    batch of initial histories in a single accelerator call.
    """

    jax: Any
    jnp: Any
    trajectory: Any
    gradient: Any
    ensemble_trajectory: Any
    ensemble_gradient: Any


#: Lazily built backend, cached after the first successful load so the JIT-compiled kernels are reused.
_BACKEND: _DelayedBackend | None = None


def _load_backend() -> _DelayedBackend:
    """Return the cached JAX delayed backend, building it on first use.

    Enables 64-bit precision (a global JAX configuration flag, hence set lazily rather than at import)
    and JIT-compiles the single-history method-of-steps solve, its reverse-mode gradient, and their
    ``vmap`` ensemble counterparts, each with the step count as a static argument so JAX caches one
    compiled kernel per step budget and window shape.

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
    except ImportError as error:  # pragma: no cover - exercised only without the optional extra
        raise ImportError(
            "the JAX delayed-Kuramoto tier requires JAX; install scpn-quantum-control[jax]"
        ) from error
    jax.config.update("jax_enable_x64", True)

    def force(current: Any, lagged: Any, coupling: Any) -> Any:
        difference = lagged[None, :] - current[:, None]  # Δ_jk = θ_k(t-τ) − θ_j(t)
        return jnp.sum(coupling * jnp.sin(difference), axis=1)

    def solve(history: Any, omega: Any, coupling: Any, dt: float, n_steps: int) -> Any:
        # The scan carry is the sliding window of the last (delay_steps + 1) phase vectors: window[-1]
        # is the current phase θ(t), window[0] the phase one full delay back θ(t-τ), window[1] the phase
        # one step nearer, so the half-step lag is their mean — exactly the samples the method-of-steps
        # RK4 stages read. The initial history is precisely that window at t = 0.
        def step(window: Any, _: Any) -> tuple[Any, Any]:
            current = window[-1]
            lag_full = window[0]
            lag_next = window[1]
            lag_half = 0.5 * (lag_full + lag_next)
            k1 = omega + force(current, lag_full, coupling)
            k2 = omega + force(current + 0.5 * dt * k1, lag_half, coupling)
            k3 = omega + force(current + 0.5 * dt * k2, lag_half, coupling)
            k4 = omega + force(current + dt * k3, lag_next, coupling)
            advanced = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return jnp.concatenate([window[1:], advanced[None, :]], axis=0), advanced

        _, stepped = jax.lax.scan(step, history, None, length=n_steps)
        return jnp.concatenate([history[-1][None, :], stepped], axis=0)

    def gradient(
        history: Any, omega: Any, coupling: Any, dt: float, n_steps: int, cotangent: Any
    ) -> Any:
        def final_state(initial: Any, frequency: Any, matrix: Any) -> Any:
            return solve(initial, frequency, matrix, dt, n_steps)[-1]

        _, pullback = jax.vjp(final_state, history, omega, coupling)
        return pullback(cotangent)

    _BACKEND = _DelayedBackend(
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


def _prepare_single(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int]:
    """Validate and return contiguous ``(history, omega, coupling, count, delay_steps)``."""
    history = np.ascontiguousarray(initial_history, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count, delay_steps = _validate_delayed(
        history, frequencies, matrix, delay, dt, n_steps, delay_tolerance
    )
    return history, frequencies, matrix, count, delay_steps


def jax_kuramoto_delayed_trajectory(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> NDArray[np.float64]:
    r"""Forward delayed networked-Kuramoto method-of-steps trajectory on the JAX accelerator tier.

    Integrates the delay-differential equation :math:`\dot\theta(t) = \omega + F(\theta(t), \theta(t-\tau))`
    with :math:`F_j = \sum_k K_{jk}\sin(\theta_k(t-\tau) - \theta_j(t))` by the delay-aware
    method-of-steps RK4 for ``n_steps`` steps, returning ``θ(0)`` plus every step — the identical
    contract to
    :func:`~scpn_quantum_control.accel.kuramoto_delayed.integrate_delayed_kuramoto` (its ``phases``
    field), but computed in JAX (on the GPU when one is available) at 64-bit precision.

    Parameters
    ----------
    initial_history : numpy.ndarray
        The phase history on ``[-τ, 0]`` as a ``(delay_steps + 1, N)`` array; row ``s`` is time
        ``-(delay_steps - s)·dt`` and the last row is ``θ(0)``.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``); must be a positive integer multiple of ``dt``.
    dt : float
        The integration time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.
    delay_tolerance : float, optional
        The absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    numpy.ndarray
        The ``(n_steps + 1, N)`` phase trajectory (row 0 is ``θ(0)``), left unwrapped.

    Raises
    ------
    ValueError
        If ``delay``/``dt``/``n_steps`` are out of range, ``τ`` is not an integer multiple of ``dt``,
        or ``initial_history``/``omega``/``coupling`` are malformed.
    ImportError
        If JAX is not installed.
    """
    history, frequencies, matrix, _, _ = _prepare_single(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    backend = _load_backend()
    jnp = backend.jnp
    result = backend.trajectory(
        jnp.asarray(history),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        int(n_steps),
    )
    return np.asarray(result, dtype=np.float64)


def jax_kuramoto_delayed_gradient(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    cotangent: NDArray[np.float64],
    delay_tolerance: float = 1e-9,
) -> _DelayedVjpResult:
    r"""Reverse-mode (autodiff) gradient of the delayed solve on the JAX accelerator tier.

    Given a cotangent :math:`\lambda_N = \partial L/\partial\theta_N` on the final phase, returns the
    gradients of the scalar objective ``L`` with respect to the full initial history, the natural
    frequencies and the coupling matrix — the same contract as the hand-derived
    :func:`~scpn_quantum_control.accel.diff_kuramoto_delayed.delayed_terminal_value_and_grad`, here
    obtained by :func:`jax.vjp` of the forward solve rather than a hand-written sensitivity. The two
    agree to machine precision, so this tier verifies the hand-written scheme and generalises to
    objectives whose sensitivity would be laborious to derive.

    Parameters
    ----------
    initial_history : numpy.ndarray
        The phase history on ``[-τ, 0]`` as a ``(delay_steps + 1, N)`` array (the last row is ``θ(0)``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``, an integer multiple of ``dt``).
    dt : float
        The integration time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    cotangent : numpy.ndarray
        The cotangent ``∂L/∂θ_N`` on the final phase (length ``N``).
    delay_tolerance : float, optional
        The absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_initial_history, grad_omega, grad_coupling)`` with shapes ``(delay_steps + 1, N)``,
        ``(N,)`` and ``(N, N)``.

    Raises
    ------
    ValueError
        If any shape is inconsistent, ``τ`` is not an integer multiple of ``dt``, or the cotangent has
        the wrong shape.
    ImportError
        If JAX is not installed.
    """
    history, frequencies, matrix, count, _ = _prepare_single(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    seed = np.ascontiguousarray(cotangent, dtype=np.float64)
    if seed.shape != (count,):
        raise ValueError(f"cotangent must have shape ({count},), got {seed.shape}")
    backend = _load_backend()
    jnp = backend.jnp
    grad_history, grad_omega, grad_coupling = backend.gradient(
        jnp.asarray(history),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        int(n_steps),
        jnp.asarray(seed),
    )
    return (
        np.asarray(grad_history, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def _prepare_ensemble(
    initial_history_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Validate a batch of histories; return ``(batch, omega, coupling, count)``."""
    batch = np.ascontiguousarray(initial_history_batch, dtype=np.float64)
    if batch.ndim != 3 or batch.shape[0] < 1:
        raise ValueError(
            "initial_history_batch must be a three-dimensional (B, delay_steps + 1, N) array with "
            f"B >= 1, got shape {batch.shape}"
        )
    _, frequencies, matrix, count, _ = _prepare_single(
        batch[0], omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    return batch, frequencies, matrix, count


def jax_kuramoto_delayed_ensemble(
    initial_history_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> NDArray[np.float64]:
    r"""Batched forward delayed trajectories for a whole ensemble of initial histories.

    Solves ``B`` initial histories that share the same frequencies, coupling and delay in a single
    accelerator call by :func:`jax.vmap` over the batch axis — a vectorisation of the *entire* delayed
    solve the NumPy and Rust tiers cannot express (they would loop over the ensemble). Each member is
    identical to the single-history :func:`jax_kuramoto_delayed_trajectory`.

    Parameters
    ----------
    initial_history_batch : numpy.ndarray
        The ``(B, delay_steps + 1, N)`` batch of initial histories.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``), shared across the batch.
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``, shared across the batch.
    delay : float
        The coupling delay ``τ`` (``> 0``, an integer multiple of ``dt``).
    dt : float
        The integration time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    delay_tolerance : float, optional
        The absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    numpy.ndarray
        The ``(B, n_steps + 1, N)`` batch of phase trajectories.

    Raises
    ------
    ValueError
        If ``initial_history_batch`` is not ``(B, delay_steps + 1, N)`` with ``B ≥ 1``, or the shared
        parameters are inconsistent.
    ImportError
        If JAX is not installed.
    """
    batch, frequencies, matrix, _ = _prepare_ensemble(
        initial_history_batch, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    backend = _load_backend()
    jnp = backend.jnp
    result = backend.ensemble_trajectory(
        jnp.asarray(batch),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        int(n_steps),
    )
    return np.asarray(result, dtype=np.float64)


def jax_kuramoto_delayed_ensemble_gradient(
    initial_history_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    cotangent_batch: NDArray[np.float64],
    delay_tolerance: float = 1e-9,
) -> _DelayedVjpResult:
    r"""Batched reverse-mode gradients for a whole ensemble of initial histories.

    For each ensemble member ``i`` with its own cotangent ``∂L_i/∂θ_N``, returns the per-member
    gradients with respect to the initial history, the frequencies and the coupling, computed in a
    single accelerator call by :func:`jax.vmap` over the batch. Each member is identical to the
    single-history :func:`jax_kuramoto_delayed_gradient`.

    Parameters
    ----------
    initial_history_batch : numpy.ndarray
        The ``(B, delay_steps + 1, N)`` batch of initial histories.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``), shared across the batch.
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``, shared across the batch.
    delay : float
        The coupling delay ``τ`` (``> 0``, an integer multiple of ``dt``).
    dt : float
        The integration time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    cotangent_batch : numpy.ndarray
        The ``(B, N)`` batch of cotangents on the final phase.
    delay_tolerance : float, optional
        The absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_initial_history, grad_omega, grad_coupling)`` with shapes ``(B, delay_steps + 1, N)``,
        ``(B, N)`` and ``(B, N, N)`` — one gradient per ensemble member.

    Raises
    ------
    ValueError
        If a shape is inconsistent, or the cotangent batch does not match the batch size.
    ImportError
        If JAX is not installed.
    """
    batch, frequencies, matrix, count = _prepare_ensemble(
        initial_history_batch, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    seeds = np.ascontiguousarray(cotangent_batch, dtype=np.float64)
    if seeds.shape != (batch.shape[0], count):
        raise ValueError(
            f"cotangent_batch must have shape ({batch.shape[0]}, {count}), got {seeds.shape}"
        )
    backend = _load_backend()
    jnp = backend.jnp
    grad_history, grad_omega, grad_coupling = backend.ensemble_gradient(
        jnp.asarray(batch),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        int(n_steps),
        jnp.asarray(seeds),
    )
    return (
        np.asarray(grad_history, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


__all__ = [
    "jax_kuramoto_delayed_ensemble",
    "jax_kuramoto_delayed_ensemble_gradient",
    "jax_kuramoto_delayed_gradient",
    "jax_kuramoto_delayed_trajectory",
]
