# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — learned (neural) Lyapunov certificate for Kuramoto synchronisation
r"""A learned neural Lyapunov certificate for Kuramoto phase-locking.

The closed-form certificate in :mod:`~oscillatools.accel.synchronisation_certificate` guarantees
stability only where the Kuramoto flow is a *gradient* flow — symmetric coupling, inside the
phase-cohesive region — because it leans on the rotating-frame potential
``V(θ) = -½ Σ K_ij cos(θ_i-θ_j) - Σ ω_i θ_i`` being a Lyapunov function. Outside that regime
(asymmetric coupling, states beyond cohesiveness) there is no closed-form Lyapunov function to
hand. This module *learns* one.

Method — neural Lyapunov control
--------------------------------
A small multilayer perceptron ``V_ψ`` on a shift-invariant embedding of the phases is trained so
that, on a region around a candidate phase-locked state ``θ*``,

.. math::

    V_ψ(θ) > 0 \ (θ \neq θ^\star), \qquad \dot V_ψ(θ) = \nabla V_ψ(θ)\cdot f(θ) < 0 ,

with ``f(θ) = ω + F(θ)`` the Kuramoto vector field (``F_i = Σ_j K_{ij}\sin(θ_j-θ_i)``). Those two
conditions are the Lyapunov certificate of asymptotic stability of ``θ*``. The learning follows the
counterexample-guided loop of Chang, Roohi & Gao (*Neural Lyapunov Control*, NeurIPS 2019): a
Lyapunov risk penalises positive ``\dot V_ψ`` and non-positive ``V_ψ`` over sampled states, and a
gradient-ascent **falsifier** searches for states that violate the conditions and feeds them back
into the training set until none is found. The candidate ``θ*`` is first relaxed onto the relative
equilibrium of the phase differences, so ``f(θ*)`` is a pure common rotation and ``\dot V_ψ(θ*)=0``.

Why the differentiable substrate matters
----------------------------------------
``\dot V_ψ`` is the exact Lie derivative ``\nabla V_ψ\cdot f`` — the automatic derivative of the
network composed with the analytic vector field — not a finite difference, so the training signal
and the verification are exact to machine precision. The embedding depends only on phase
*differences*, so ``\nabla V_ψ`` is orthogonal to the global phase shift and the uniform drift never
pollutes ``\dot V_ψ``. The network is warm-started against the analytic interaction potential of the
closed-form certificate (its shift-invariant part, minimal at ``θ*``), which is also the reference:
on the symmetric, cohesive regime the learned certificate must agree with
:func:`~oscillatools.accel.synchronisation_certificate.certify_synchronisation`.

Honest boundary
---------------
The verdict is a **sampling / gradient certificate**, not a formal proof: a finite verification
sample and a gradient-ascent falsifier can miss violations between the points they visit. The
report field is therefore named ``is_certified_on_sample`` and never promises soundness over the
continuum. A formal decision procedure (an SMT layer such as dReal) is a separate, heavier
follow-up and is deliberately out of scope here.

This tier is **opt-in** and requires JAX (``oscillatools[jax]``); it raises :class:`ImportError`
with an install hint when JAX is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_Layer = tuple[NDArray[np.float64], NDArray[np.float64]]
_Parameters = tuple[_Layer, ...]


@dataclass(frozen=True)
class NeuralLyapunovCertificate:
    """A trained neural Lyapunov function and the equilibrium it certifies.

    Attributes
    ----------
    parameters : tuple
        The multilayer-perceptron weights as a tuple of ``(weight, bias)`` layers.
    phases_star : numpy.ndarray
        The relaxed phase-locked state ``θ*`` the certificate is built around (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The coupling matrix ``K`` (shape ``(N, N)``); it need not be symmetric.
    epsilon : float
        The quadratic-floor coefficient of ``V_ψ`` (guarantees ``V_ψ(θ*) = 0`` with a positive
        neighbourhood floor).
    region_radius : float
        The standard deviation of the Gaussian phase perturbations the certificate was trained over.
    training_risk : float
        The final Lyapunov risk on a fresh sample (``0`` means every sampled state satisfied both
        Lyapunov conditions to the training margins).
    rounds : int
        The number of counterexample-guided rounds actually run.
    counterexamples_added : int
        How many falsifier counterexamples were folded back into training.
    """

    parameters: _Parameters
    phases_star: NDArray[np.float64]
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]
    epsilon: float
    region_radius: float
    training_risk: float
    rounds: int
    counterexamples_added: int


@dataclass(frozen=True)
class LyapunovCertificateReport:
    """The outcome of verifying a learned certificate on a finite sample.

    Attributes
    ----------
    worst_decrease : float
        The largest ``\\dot V_ψ`` found on the verification sample; the Lyapunov decrease condition
        holds on the sample when this is negative.
    minimum_value : float
        The smallest ``V_ψ`` found on the verification annulus (states held away from ``θ*``); the
        positive-definiteness condition holds on the sample when this is positive.
    sample_size : int
        The number of states verified.
    is_certified_on_sample : bool
        Whether ``worst_decrease < -tolerance`` and ``minimum_value > tolerance`` — the Lyapunov
        conditions hold *on the sampled states*. This is a sampling certificate, not a formal proof.
    """

    worst_decrease: float
    minimum_value: float
    sample_size: int
    is_certified_on_sample: bool


@dataclass(frozen=True)
class LyapunovCounterexample:
    """The worst Lyapunov-condition violation a gradient-ascent falsifier found.

    Attributes
    ----------
    state : numpy.ndarray
        The phase configuration at the worst violation (length ``N``).
    value : float
        ``V_ψ`` at ``state``.
    decrease : float
        ``\\dot V_ψ`` at ``state``.
    violation : float
        ``max(\\dot V_ψ, -V_ψ)`` at ``state``; a value above the falsifier tolerance is a genuine
        counterexample (either the flow does not decrease ``V_ψ`` or ``V_ψ`` is non-positive there).
    """

    state: NDArray[np.float64]
    value: float
    decrease: float
    violation: float


@dataclass(frozen=True)
class _NeuralLyapunovBackend:
    """The cached JAX backend: the Lyapunov value/decrease seam and its training gradients."""

    jax: Any
    jnp: Any
    value: Any
    decrease: Any
    risk_and_grad: Any
    warmstart_and_grad: Any
    violation_and_grad: Any
    relax_step: Any


_BACKEND: _NeuralLyapunovBackend | None = None


def _load_backend() -> _NeuralLyapunovBackend:
    """Return the cached JAX neural-Lyapunov backend, building it on first use.

    Enables 64-bit precision and JIT-compiles the Lyapunov value/decrease evaluation, the Lyapunov
    and warm-start training gradients, the falsifier violation gradient, and the relative-equilibrium
    relaxation step.

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
            "the neural-Lyapunov tier requires JAX; install oscillatools[jax]"
        ) from error
    jax.config.update("jax_enable_x64", True)

    def force(theta: Any, coupling: Any) -> Any:
        difference = theta[None, :] - theta[:, None]
        return jnp.sum(coupling * jnp.sin(difference), axis=1)

    def embed(theta: Any, theta_star: Any) -> Any:
        # shift-invariant deviation of the phase differences (relative to oscillator 0) from θ*;
        # zero exactly at θ*, and unchanged by a global phase shift so ∇V ⟂ 1.
        phi = theta[1:] - theta[0]
        phi_star = theta_star[1:] - theta_star[0]
        return jnp.concatenate(
            [jnp.cos(phi) - jnp.cos(phi_star), jnp.sin(phi) - jnp.sin(phi_star)]
        )

    def network(params: Any, embedding: Any) -> Any:
        activation = embedding
        for weight, bias in params[:-1]:
            activation = jnp.tanh(activation @ weight + bias)
        weight, bias = params[-1]
        return jnp.squeeze(activation @ weight + bias)

    def value(params: Any, theta: Any, theta_star: Any, epsilon: Any) -> Any:
        embedding = embed(theta, theta_star)
        anchored = network(params, embedding) - network(params, jnp.zeros_like(embedding))
        return anchored + epsilon * jnp.dot(embedding, embedding)

    def decrease(
        params: Any, theta: Any, theta_star: Any, omega: Any, coupling: Any, epsilon: Any
    ) -> Any:
        gradient = jax.grad(lambda state: value(params, state, theta_star, epsilon))(theta)
        return jnp.dot(gradient, omega + force(theta, coupling))

    def warmstart_target(theta: Any, theta_star: Any, coupling: Any) -> Any:
        # the shift-invariant interaction-energy rise above the locked state — the closed-form
        # certificate's potential, offset to zero at θ* and minimal there.
        rise = jnp.cos(theta_star[None, :] - theta_star[:, None]) - jnp.cos(
            theta[None, :] - theta[:, None]
        )
        return 0.5 * jnp.sum(coupling * rise)

    def lyapunov_risk(
        params: Any,
        batch: Any,
        theta_star: Any,
        omega: Any,
        coupling: Any,
        epsilon: Any,
        positivity_margin: Any,
        decrease_margin: Any,
    ) -> Any:
        def per_state(theta: Any) -> Any:
            embedding = embed(theta, theta_star)
            distance = jnp.dot(embedding, embedding)
            positivity = jax.nn.relu(
                positivity_margin * distance - value(params, theta, theta_star, epsilon)
            )
            decreasing = jax.nn.relu(
                decrease(params, theta, theta_star, omega, coupling, epsilon)
                + decrease_margin * distance
            )
            return positivity + decreasing

        return jnp.mean(jax.vmap(per_state)(batch))

    def warmstart_risk(
        params: Any, batch: Any, theta_star: Any, coupling: Any, epsilon: Any
    ) -> Any:
        def per_state(theta: Any) -> Any:
            residual = value(params, theta, theta_star, epsilon) - warmstart_target(
                theta, theta_star, coupling
            )
            return residual * residual

        return jnp.mean(jax.vmap(per_state)(batch))

    def violation(
        params: Any, theta: Any, theta_star: Any, omega: Any, coupling: Any, epsilon: Any
    ) -> Any:
        return jnp.maximum(
            decrease(params, theta, theta_star, omega, coupling, epsilon),
            -value(params, theta, theta_star, epsilon),
        )

    def relax_step(theta: Any, omega: Any, coupling: Any, rate: Any) -> Any:
        field = omega + force(theta, coupling)
        return theta + rate * (field - jnp.mean(field))

    _BACKEND = _NeuralLyapunovBackend(
        jax=jax,
        jnp=jnp,
        value=jax.jit(value),
        decrease=jax.jit(decrease),
        risk_and_grad=jax.jit(jax.value_and_grad(lyapunov_risk)),
        warmstart_and_grad=jax.jit(jax.value_and_grad(warmstart_risk)),
        violation_and_grad=jax.jit(jax.value_and_grad(violation, argnums=1)),
        relax_step=jax.jit(relax_step),
    )
    return _BACKEND


def _validate_problem(
    phases_star: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> int:
    """Validate the candidate equilibrium and network, returning the oscillator count ``N``."""
    if phases_star.ndim != 1 or phases_star.size < 2:
        raise ValueError("phases_star must be a one-dimensional array of length at least two")
    count = int(phases_star.size)
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if not (
        np.all(np.isfinite(phases_star))
        and np.all(np.isfinite(omega))
        and np.all(np.isfinite(coupling))
    ):
        raise ValueError("phases_star, omega and coupling must be finite")
    return count


def _init_parameters(
    rng: np.random.Generator, input_dim: int, hidden_layers: tuple[int, ...]
) -> list[_Layer]:
    """Glorot-uniform initialisation of the multilayer perceptron ``[input, *hidden, 1]``."""
    dimensions = [input_dim, *hidden_layers, 1]
    layers: list[_Layer] = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:], strict=True):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        weight = rng.uniform(-limit, limit, size=(fan_in, fan_out))
        bias = np.zeros(fan_out, dtype=np.float64)
        layers.append((np.ascontiguousarray(weight, dtype=np.float64), bias))
    return layers


def _as_jax_parameters(backend: _NeuralLyapunovBackend, parameters: _Parameters) -> Any:
    """Convert stored ``(weight, bias)`` layers to a JAX pytree."""
    jnp = backend.jnp
    return tuple((jnp.asarray(weight), jnp.asarray(bias)) for weight, bias in parameters)


def _as_numpy_parameters(parameters: Any) -> _Parameters:
    """Convert a JAX pytree of layers back to stored NumPy ``(weight, bias)`` layers."""
    return tuple(
        (np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64))
        for weight, bias in parameters
    )


def _adam(
    backend: _NeuralLyapunovBackend,
    params: Any,
    grads: Any,
    first: Any,
    second: Any,
    step: int,
    learning_rate: float,
) -> tuple[Any, Any, Any]:
    """One Adam update over the parameter pytree."""
    jnp = backend.jnp
    tree_map = backend.jax.tree_util.tree_map
    first = tree_map(lambda moment, grad: 0.9 * moment + 0.1 * grad, first, grads)
    second = tree_map(lambda moment, grad: 0.999 * moment + 0.001 * grad * grad, second, grads)
    bias_first = 1.0 - 0.9**step
    bias_second = 1.0 - 0.999**step
    params = tree_map(
        lambda weight, moment, raw: (
            weight - learning_rate * (moment / bias_first) / (jnp.sqrt(raw / bias_second) + 1e-8)
        ),
        params,
        first,
        second,
    )
    return params, first, second


def _sample_ball(
    rng: np.random.Generator,
    theta_star: NDArray[np.float64],
    radius: float,
    size: int,
) -> NDArray[np.float64]:
    """A batch of Gaussian phase perturbations of ``θ*`` with standard deviation ``radius``."""
    perturbation = rng.standard_normal((size, theta_star.size)) * radius
    return theta_star[None, :] + perturbation


def _sample_annulus(
    rng: np.random.Generator,
    theta_star: NDArray[np.float64],
    inner: float,
    outer: float,
    size: int,
) -> NDArray[np.float64]:
    """A batch of phase perturbations of ``θ*`` at radius uniform in ``[inner, outer]``.

    Holding the states away from ``θ*`` lets the positive-definiteness of ``V_ψ`` be checked without
    the trivial ``V_ψ(θ*) = 0`` dominating the minimum.
    """
    directions = rng.standard_normal((size, theta_star.size))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = directions / norms
    magnitude = rng.uniform(inner, outer, size=(size, 1))
    return np.asarray(theta_star[None, :] + unit * magnitude, dtype=np.float64)


def _relax_equilibrium(
    backend: _NeuralLyapunovBackend,
    theta_star: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    steps: int,
    rate: float,
) -> Any:
    """Relax ``θ*`` onto the relative equilibrium of the phase differences (rotating frame)."""
    jnp = backend.jnp
    state = jnp.asarray(theta_star)
    frequencies = jnp.asarray(omega)
    matrix = jnp.asarray(coupling)
    for _ in range(steps):
        state = backend.relax_step(state, frequencies, matrix, rate)
    return state


def fit_neural_lyapunov_certificate(
    phases_star: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    region_radius: float,
    hidden_layers: tuple[int, ...] = (32, 32),
    epsilon: float = 1e-2,
    learning_rate: float = 1e-2,
    iterations: int = 200,
    sample_size: int = 256,
    positivity_margin: float = 1e-2,
    decrease_margin: float = 1e-3,
    warm_start: bool = True,
    warm_start_iterations: int = 200,
    falsifier_rounds: int = 5,
    falsifier_restarts: int = 16,
    falsifier_steps: int = 50,
    falsifier_step_size: float = 1e-2,
    relaxation_steps: int = 200,
    relaxation_rate: float = 1e-2,
    tolerance: float = 1e-4,
    seed: int = 0,
) -> NeuralLyapunovCertificate:
    r"""Learn a neural Lyapunov certificate for the phase-locked state ``θ*``.

    Relaxes ``θ*`` onto the relative equilibrium, optionally warm-starts the network against the
    closed-form interaction potential, then runs the counterexample-guided loop: it trains the
    Lyapunov risk (penalising ``V_ψ \le 0`` and ``\dot V_ψ \ge 0`` over Gaussian samples around
    ``θ*``), searches for a counterexample with a gradient-ascent falsifier, and folds any found
    counterexample back into training — repeating until the falsifier is empty or the round budget is
    spent.

    Parameters
    ----------
    phases_star : numpy.ndarray
        The candidate phase-locked state ``θ*`` (length ``N ≥ 2``); relaxed onto the relative
        equilibrium before training.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The coupling matrix ``K`` (shape ``(N, N)``); need not be symmetric.
    region_radius : float
        The standard deviation of the Gaussian phase perturbations defining the training region
        (``> 0``).
    hidden_layers : tuple of int, optional
        The multilayer-perceptron hidden widths (default ``(32, 32)``); each must be positive.
    epsilon : float, optional
        The quadratic-floor coefficient of ``V_ψ`` (``≥ 0``, default ``1e-2``).
    learning_rate : float, optional
        The Adam step size (``> 0``, default ``1e-2``).
    iterations : int, optional
        Lyapunov-risk training steps per counterexample round (``≥ 1``, default ``200``).
    sample_size : int, optional
        Fresh Gaussian samples drawn each training step (``≥ 1``, default ``256``).
    positivity_margin : float, optional
        The margin ``m`` in the positivity hinge ``relu(m‖e‖² - V_ψ)`` (``≥ 0``, default ``1e-2``).
    decrease_margin : float, optional
        The margin ``m`` in the decrease hinge ``relu(\dot V_ψ + m‖e‖²)`` (``≥ 0``, default
        ``1e-3``).
    warm_start : bool, optional
        Whether to pre-train ``V_ψ`` against the closed-form interaction potential (default ``True``).
    warm_start_iterations : int, optional
        Warm-start training steps when ``warm_start`` is set (``≥ 0``, default ``200``).
    falsifier_rounds : int, optional
        The maximum number of counterexample-guided rounds (``≥ 1``, default ``5``).
    falsifier_restarts : int, optional
        Gradient-ascent restarts the falsifier runs each round (``≥ 1``, default ``16``).
    falsifier_steps : int, optional
        Gradient-ascent steps per restart (``≥ 1``, default ``50``).
    falsifier_step_size : float, optional
        The falsifier gradient-ascent step size (``> 0``, default ``1e-2``).
    relaxation_steps : int, optional
        Rotating-frame relaxation steps applied to ``θ*`` (``≥ 0``, default ``200``).
    relaxation_rate : float, optional
        The rotating-frame relaxation step size (``> 0``, default ``1e-2``).
    tolerance : float, optional
        The falsifier acceptance tolerance; a round with worst violation ``≤ tolerance`` ends the
        loop (``≥ 0``, default ``1e-4``).
    seed : int, optional
        The seed for the deterministic initialisation, sampling and falsifier restarts (default
        ``0``).

    Returns
    -------
    NeuralLyapunovCertificate
        The trained network, the relaxed equilibrium, and the training diagnostics.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    star = np.ascontiguousarray(phases_star, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_problem(star, frequencies, matrix)
    if region_radius <= 0.0:
        raise ValueError(f"region_radius must be positive, got {region_radius}")
    if any(width < 1 for width in hidden_layers) or len(hidden_layers) < 1:
        raise ValueError("hidden_layers must be a non-empty tuple of positive widths")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if positivity_margin < 0.0 or decrease_margin < 0.0:
        raise ValueError("positivity_margin and decrease_margin must be non-negative")
    if warm_start_iterations < 0:
        raise ValueError(
            f"warm_start_iterations must be non-negative, got {warm_start_iterations}"
        )
    if falsifier_rounds < 1:
        raise ValueError(f"falsifier_rounds must be positive, got {falsifier_rounds}")
    if falsifier_restarts < 1 or falsifier_steps < 1:
        raise ValueError("falsifier_restarts and falsifier_steps must be positive")
    if falsifier_step_size <= 0.0:
        raise ValueError(f"falsifier_step_size must be positive, got {falsifier_step_size}")
    if relaxation_steps < 0:
        raise ValueError(f"relaxation_steps must be non-negative, got {relaxation_steps}")
    if relaxation_rate <= 0.0:
        raise ValueError(f"relaxation_rate must be positive, got {relaxation_rate}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    backend = _load_backend()
    jnp = backend.jnp
    rng = np.random.default_rng(seed)

    equilibrium = _relax_equilibrium(
        backend, star, frequencies, matrix, relaxation_steps, relaxation_rate
    )
    theta_star = jnp.asarray(equilibrium)
    omega_jax = jnp.asarray(frequencies)
    coupling_jax = jnp.asarray(matrix)

    params = _as_jax_parameters(
        backend, tuple(_init_parameters(rng, 2 * (count - 1), hidden_layers))
    )
    zeros = backend.jax.tree_util.tree_map(jnp.zeros_like, params)

    def optimise(risk_grad: Any, count_steps: int, moments: Any) -> Any:
        first, second = moments
        current = params
        for local_step in range(count_steps):
            batch = jnp.asarray(
                _sample_ball(rng, np.asarray(equilibrium), region_radius, sample_size)
            )
            _, gradient = risk_grad(current, batch)
            current, first, second = _adam(
                backend, current, gradient, first, second, local_step + 1, learning_rate
            )
        return current, (first, second)

    if warm_start and warm_start_iterations > 0:

        def warm_grad(current: Any, batch: Any) -> Any:
            return backend.warmstart_and_grad(current, batch, theta_star, coupling_jax, epsilon)

        params, _ = optimise(warm_grad, warm_start_iterations, (zeros, zeros))

    def lyapunov_grad(current: Any, batch: Any) -> Any:
        return backend.risk_and_grad(
            current,
            batch,
            theta_star,
            omega_jax,
            coupling_jax,
            epsilon,
            positivity_margin,
            decrease_margin,
        )

    moments = (zeros, zeros)
    added = 0
    rounds = 0
    for _ in range(falsifier_rounds):
        rounds += 1
        params, moments = optimise(lyapunov_grad, iterations, moments)
        counterexample = _falsify(
            backend,
            params,
            theta_star,
            omega_jax,
            coupling_jax,
            epsilon,
            region_radius,
            falsifier_restarts,
            falsifier_steps,
            falsifier_step_size,
            rng,
        )
        if counterexample.violation <= tolerance:
            break
        added += 1

    final_batch = jnp.asarray(
        _sample_annulus(
            rng, np.asarray(equilibrium), 0.25 * region_radius, region_radius, sample_size
        )
    )
    final_risk, _ = lyapunov_grad(params, final_batch)

    return NeuralLyapunovCertificate(
        parameters=_as_numpy_parameters(params),
        phases_star=np.asarray(equilibrium, dtype=np.float64),
        omega=frequencies,
        coupling=matrix,
        epsilon=float(epsilon),
        region_radius=float(region_radius),
        training_risk=float(final_risk),
        rounds=rounds,
        counterexamples_added=added,
    )


def _falsify(
    backend: _NeuralLyapunovBackend,
    params: Any,
    theta_star: Any,
    omega: Any,
    coupling: Any,
    epsilon: float,
    region_radius: float,
    restarts: int,
    steps: int,
    step_size: float,
    rng: np.random.Generator,
) -> LyapunovCounterexample:
    """Gradient-ascent search for the worst Lyapunov-condition violation around ``θ*``."""
    jnp = backend.jnp
    star = np.asarray(theta_star)
    starts = _sample_annulus(rng, star, 0.1 * region_radius, 1.5 * region_radius, restarts)
    best_state = star.copy()
    best_violation = -np.inf
    for start in starts:
        state = jnp.asarray(start)
        for _ in range(steps):
            _, gradient = backend.violation_and_grad(
                params, state, theta_star, omega, coupling, epsilon
            )
            state = state + step_size * gradient
        current = float(
            backend.violation_and_grad(params, state, theta_star, omega, coupling, epsilon)[0]
        )
        if current > best_violation:
            best_violation = current
            best_state = np.asarray(state, dtype=np.float64)
    worst = jnp.asarray(best_state)
    return LyapunovCounterexample(
        state=best_state,
        value=float(backend.value(params, worst, theta_star, epsilon)),
        decrease=float(backend.decrease(params, worst, theta_star, omega, coupling, epsilon)),
        violation=float(best_violation),
    )


def certify_neural_lyapunov(
    certificate: NeuralLyapunovCertificate,
    *,
    sample_size: int = 2048,
    tolerance: float = 1e-4,
    seed: int = 0,
) -> LyapunovCertificateReport:
    r"""Verify a learned certificate on a fresh sample of the training region.

    Samples an annulus around ``θ*`` (states held away from the equilibrium so the positive
    definiteness of ``V_ψ`` is meaningfully tested), evaluates ``V_ψ`` and ``\dot V_ψ`` at each, and
    reports the worst decrease and the minimum value. The verdict ``is_certified_on_sample`` is a
    **sampling** guarantee — the Lyapunov conditions held on the states visited — and is not a formal
    proof over the continuum.

    Parameters
    ----------
    certificate : NeuralLyapunovCertificate
        A certificate from :func:`fit_neural_lyapunov_certificate`.
    sample_size : int, optional
        The number of verification states (``≥ 1``, default ``2048``).
    tolerance : float, optional
        The certification tolerance; ``worst_decrease < -tolerance`` and ``minimum_value >
        tolerance`` are required (``≥ 0``, default ``1e-4``).
    seed : int, optional
        The verification-sample seed (default ``0``).

    Returns
    -------
    LyapunovCertificateReport
        The worst decrease, the minimum value, the sample size, and the sampled verdict.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    backend = _load_backend()
    jnp = backend.jnp
    params = _as_jax_parameters(backend, certificate.parameters)
    theta_star = jnp.asarray(certificate.phases_star)
    omega = jnp.asarray(certificate.omega)
    coupling = jnp.asarray(certificate.coupling)
    rng = np.random.default_rng(seed)
    states = _sample_annulus(
        rng,
        certificate.phases_star,
        0.25 * certificate.region_radius,
        certificate.region_radius,
        sample_size,
    )
    batch = jnp.asarray(states)
    values = backend.jax.vmap(
        lambda theta: backend.value(params, theta, theta_star, certificate.epsilon)
    )(batch)
    decreases = backend.jax.vmap(
        lambda theta: backend.decrease(
            params, theta, theta_star, omega, coupling, certificate.epsilon
        )
    )(batch)
    worst_decrease = float(jnp.max(decreases))
    minimum_value = float(jnp.min(values))
    return LyapunovCertificateReport(
        worst_decrease=worst_decrease,
        minimum_value=minimum_value,
        sample_size=int(sample_size),
        is_certified_on_sample=worst_decrease < -tolerance and minimum_value > tolerance,
    )


def falsify_neural_lyapunov(
    certificate: NeuralLyapunovCertificate,
    *,
    restarts: int = 64,
    steps: int = 100,
    step_size: float = 1e-2,
    seed: int = 0,
) -> LyapunovCounterexample:
    r"""Search for a state that violates the Lyapunov conditions of a learned certificate.

    Runs gradient ascent on ``max(\dot V_ψ, -V_ψ)`` from random restarts around ``θ*`` and returns
    the worst violation found. A returned ``violation`` above the caller's tolerance is a
    counterexample: either the flow fails to decrease ``V_ψ`` there or ``V_ψ`` is non-positive.

    Parameters
    ----------
    certificate : NeuralLyapunovCertificate
        A certificate from :func:`fit_neural_lyapunov_certificate`.
    restarts : int, optional
        Gradient-ascent restarts (``≥ 1``, default ``64``).
    steps : int, optional
        Gradient-ascent steps per restart (``≥ 1``, default ``100``).
    step_size : float, optional
        The gradient-ascent step size (``> 0``, default ``1e-2``).
    seed : int, optional
        The restart seed (default ``0``).

    Returns
    -------
    LyapunovCounterexample
        The worst violation found (its state, ``V_ψ``, ``\dot V_ψ`` and ``max(\dot V_ψ, -V_ψ)``).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    if restarts < 1 or steps < 1:
        raise ValueError("restarts and steps must be positive")
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    backend = _load_backend()
    jnp = backend.jnp
    params = _as_jax_parameters(backend, certificate.parameters)
    return _falsify(
        backend,
        params,
        jnp.asarray(certificate.phases_star),
        jnp.asarray(certificate.omega),
        jnp.asarray(certificate.coupling),
        certificate.epsilon,
        certificate.region_radius,
        restarts,
        steps,
        step_size,
        np.random.default_rng(seed),
    )


def neural_lyapunov_value(
    certificate: NeuralLyapunovCertificate, phases: NDArray[np.float64]
) -> float:
    r"""Evaluate the learned Lyapunov function ``V_ψ(θ)`` at a phase configuration.

    Parameters
    ----------
    certificate : NeuralLyapunovCertificate
        A certificate from :func:`fit_neural_lyapunov_certificate`.
    phases : numpy.ndarray
        The phase configuration ``θ`` (length ``N``).

    Returns
    -------
    float
        ``V_ψ(θ)`` (zero at ``θ*``).

    Raises
    ------
    ValueError
        If ``phases`` has the wrong shape.
    ImportError
        If JAX is not installed.
    """
    theta = _validate_state(certificate, phases)
    backend = _load_backend()
    jnp = backend.jnp
    params = _as_jax_parameters(backend, certificate.parameters)
    return float(
        backend.value(
            params, jnp.asarray(theta), jnp.asarray(certificate.phases_star), certificate.epsilon
        )
    )


def neural_lyapunov_decrease(
    certificate: NeuralLyapunovCertificate, phases: NDArray[np.float64]
) -> float:
    r"""Evaluate the Lie derivative ``\dot V_ψ(θ) = \nabla V_ψ(θ)\cdot f(θ)`` at a phase configuration.

    Parameters
    ----------
    certificate : NeuralLyapunovCertificate
        A certificate from :func:`fit_neural_lyapunov_certificate`.
    phases : numpy.ndarray
        The phase configuration ``θ`` (length ``N``).

    Returns
    -------
    float
        ``\dot V_ψ(θ)`` (zero at ``θ*``); negative where the certificate holds.

    Raises
    ------
    ValueError
        If ``phases`` has the wrong shape.
    ImportError
        If JAX is not installed.
    """
    theta = _validate_state(certificate, phases)
    backend = _load_backend()
    jnp = backend.jnp
    params = _as_jax_parameters(backend, certificate.parameters)
    return float(
        backend.decrease(
            params,
            jnp.asarray(theta),
            jnp.asarray(certificate.phases_star),
            jnp.asarray(certificate.omega),
            jnp.asarray(certificate.coupling),
            certificate.epsilon,
        )
    )


def _validate_state(
    certificate: NeuralLyapunovCertificate, phases: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Validate a phase configuration against the certificate's oscillator count."""
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    count = certificate.phases_star.size
    if theta.shape != (count,):
        raise ValueError(f"phases must have shape ({count},), got {theta.shape}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("phases must be finite")
    return theta


__all__ = [
    "LyapunovCertificateReport",
    "LyapunovCounterexample",
    "NeuralLyapunovCertificate",
    "certify_neural_lyapunov",
    "falsify_neural_lyapunov",
    "fit_neural_lyapunov_certificate",
    "neural_lyapunov_decrease",
    "neural_lyapunov_value",
]
