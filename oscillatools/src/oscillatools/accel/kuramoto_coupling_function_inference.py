# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Data-driven inference of the Kuramoto coupling function
r"""Physics-informed inference of the Kuramoto coupling function from observed dynamics.

The phase-coupled oscillator model ``θ̇_i = ω_i + Σ_j K_{ij} Γ(θ_j - θ_i)`` is usually written with
the sinusoidal interaction ``Γ(x) = sin x``, but the *shape* of the coupling function ``Γ`` carries
the physics (frustration, higher harmonics, multistability) and is what one wants to recover from
data. This module infers a flexible, diverse coupling function — represented in the natural
zero-mean Fourier basis

.. math::

    \Gamma(x) = \sum_{m=1}^{M} a_m \sin(m x) + b_m \cos(m x)

(so ``a_1 = 1`` recovers classic Kuramoto, ``a_1 = \cos\beta, b_1 = -\sin\beta`` recovers
Sakaguchi–Kuramoto, and higher harmonics recover Hansel/Daido couplings) — by two complementary
estimators that both sit on the toolkit's differentiable substrate:

* a **physics-informed collocation** estimator that minimises the ODE residual
  ``θ̇_i - ω_i - Σ_j K_{ij} Γ(θ_j - θ_i)`` at observed states. Because the residual is linear in the
  Fourier coefficients and in the natural frequencies, this is a single linear least-squares solve
  that recovers the coupling function *and* the frequencies exactly from clean snapshot/derivative
  data; and
* a **differentiable trajectory-match** estimator that, when only a noisy sampled trajectory is
  available (no reliable derivatives), integrates the model with the parametrised ``Γ`` and matches
  the simulated to the observed trajectory, taking the gradient of the trajectory-match loss with
  respect to the Fourier coefficients through the discrete adjoint of the RK4 map. This is the
  gradient-based-inference-on-a-differentiable-model path; it matches finite differences to machine
  precision and is robust because it never differentiates the data.

It complements the toolkit's coupling-matrix system identification (which fixes ``Γ = \sin`` and
learns ``K``): here ``K`` is known and the coupling *function* is the unknown. It adds no compute
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CouplingFunctionEstimate:
    """An inferred Kuramoto coupling function and natural frequencies.

    Attributes
    ----------
    sine_coefficients : numpy.ndarray
        The ``(M,)`` sine coefficients ``a_1, …, a_M`` of ``Γ``.
    cosine_coefficients : numpy.ndarray
        The ``(M,)`` cosine coefficients ``b_1, …, b_M`` of ``Γ``.
    frequencies : numpy.ndarray
        The ``(N,)`` natural frequencies ``ω`` (the value passed through for the trajectory-match
        estimator, or the jointly inferred frequencies for the collocation estimator).
    residual : float
        The root-mean-square fit residual (the ODE residual for collocation, the trajectory-match
        loss for the differentiable estimator).
    """

    sine_coefficients: NDArray[np.float64]
    cosine_coefficients: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    residual: float

    def evaluate(self, angle: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the inferred coupling function ``Γ`` at the given phase differences."""
        return coupling_function_value(angle, self.sine_coefficients, self.cosine_coefficients)


@dataclass(frozen=True)
class CouplingFunctionGradients:
    """The trajectory-match loss and its gradients with respect to the Fourier coefficients.

    Attributes
    ----------
    loss : float
        The trajectory-match loss ``Σ_k ‖θ_k − θ^obs_k‖²``.
    sine_gradient : numpy.ndarray
        The ``(M,)`` gradient with respect to the sine coefficients.
    cosine_gradient : numpy.ndarray
        The ``(M,)`` gradient with respect to the cosine coefficients.
    """

    loss: float
    sine_gradient: NDArray[np.float64]
    cosine_gradient: NDArray[np.float64]


def coupling_function_value(
    angle: NDArray[np.float64],
    sine_coefficients: NDArray[np.float64],
    cosine_coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Evaluate the Fourier coupling function ``Γ(x) = Σ_m a_m sin(m x) + b_m cos(m x)``.

    Parameters
    ----------
    angle : numpy.ndarray
        The phase differences ``x`` at which to evaluate ``Γ``.
    sine_coefficients, cosine_coefficients : numpy.ndarray
        The ``(M,)`` sine ``a_m`` and cosine ``b_m`` coefficients (harmonics ``m = 1 … M``).

    Returns
    -------
    numpy.ndarray
        ``Γ(angle)``, broadcast over ``angle``.

    Raises
    ------
    ValueError
        If the coefficient arrays are mismatched or empty.
    """
    sine = np.ascontiguousarray(sine_coefficients, dtype=np.float64)
    cosine = np.ascontiguousarray(cosine_coefficients, dtype=np.float64)
    if sine.ndim != 1 or sine.shape != cosine.shape or sine.size < 1:
        raise ValueError("sine and cosine coefficients must be equal-length non-empty 1-D arrays")
    grid = np.asarray(angle, dtype=np.float64)
    harmonics = np.arange(1, sine.size + 1, dtype=np.float64)
    phases = harmonics * grid[..., None]
    return np.asarray(np.sin(phases) @ sine + np.cos(phases) @ cosine, dtype=np.float64)


def _coupling_derivative(
    angle: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The derivative ``Γ'(x) = Σ_m m(a_m cos(m x) − b_m sin(m x))``."""
    harmonics = np.arange(1, sine.size + 1, dtype=np.float64)
    phases = harmonics * angle[..., None]
    return np.asarray(
        np.cos(phases) @ (harmonics * sine) - np.sin(phases) @ (harmonics * cosine),
        dtype=np.float64,
    )


def _validate_dataset(
    phases: NDArray[np.float64],
    second: NDArray[np.float64],
    coupling: NDArray[np.float64],
    n_harmonics: int,
    second_name: str,
) -> int:
    if phases.ndim != 2 or phases.shape[0] < 1 or phases.shape[1] < 1:
        raise ValueError("phases must be a (n_samples, N) array with positive dimensions")
    count = phases.shape[1]
    if second.shape != phases.shape:
        raise ValueError(f"{second_name} must have the same shape as phases, got {second.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be positive, got {n_harmonics}")
    return int(count)


def infer_coupling_function(
    phases: NDArray[np.float64],
    derivatives: NDArray[np.float64],
    coupling: NDArray[np.float64],
    n_harmonics: int,
) -> CouplingFunctionEstimate:
    r"""Infer the coupling function and frequencies by physics-informed collocation least-squares.

    Minimises the ODE residual ``θ̇_i - ω_i - Σ_j K_{ij} Γ(θ_j - θ_i)`` over the Fourier coefficients
    of ``Γ`` and the natural frequencies ``ω``. The residual is linear in both, so this is a single
    linear least-squares solve, exact for clean snapshot/derivative data.

    Parameters
    ----------
    phases : numpy.ndarray
        The ``(n_samples, N)`` observed phase snapshots.
    derivatives : numpy.ndarray
        The ``(n_samples, N)`` observed phase velocities ``θ̇``.
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    n_harmonics : int
        The number ``M`` of Fourier harmonics to fit (``≥ 1``).

    Returns
    -------
    CouplingFunctionEstimate
        The inferred sine / cosine coefficients, the jointly inferred frequencies, and the RMS
        ODE-residual.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    snapshots = np.ascontiguousarray(phases, dtype=np.float64)
    velocity = np.ascontiguousarray(derivatives, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_dataset(snapshots, velocity, matrix, n_harmonics, "derivatives")
    n_samples = snapshots.shape[0]
    harmonics = np.arange(1, n_harmonics + 1, dtype=np.float64)

    design = np.zeros((n_samples * count, 2 * n_harmonics + count), dtype=np.float64)
    for sample in range(n_samples):
        difference = snapshots[sample][None, :] - snapshots[sample][:, None]
        scaled = harmonics[:, None, None] * difference[None, :, :]
        sine_block = np.einsum("ij,mij->im", matrix, np.sin(scaled))
        cosine_block = np.einsum("ij,mij->im", matrix, np.cos(scaled))
        rows = slice(sample * count, (sample + 1) * count)
        design[rows, :n_harmonics] = sine_block
        design[rows, n_harmonics : 2 * n_harmonics] = cosine_block
        design[rows, 2 * n_harmonics :] = np.eye(count, dtype=np.float64)

    target = velocity.reshape(-1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(np.sqrt(np.mean((design @ solution - target) ** 2)))
    return CouplingFunctionEstimate(
        sine_coefficients=np.ascontiguousarray(solution[:n_harmonics], dtype=np.float64),
        cosine_coefficients=np.ascontiguousarray(
            solution[n_harmonics : 2 * n_harmonics], dtype=np.float64
        ),
        frequencies=np.ascontiguousarray(solution[2 * n_harmonics :], dtype=np.float64),
        residual=residual,
    )


def _field(
    phases: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The coupling-function-parametrised Kuramoto field ``ω + Σ_j K_ij Γ(θ_j − θ_i)``."""
    difference = phases[None, :] - phases[:, None]
    interaction = coupling * coupling_function_value(difference, sine, cosine)
    return np.asarray(omega + interaction.sum(axis=1), dtype=np.float64)


def _state_and_parameter_jacobians(
    phases: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(∂field/∂phases, ∂field/∂[a;b])`` at ``phases``."""
    count = phases.size
    n_harmonics = sine.size
    difference = phases[None, :] - phases[:, None]
    weighted = coupling * _coupling_derivative(difference, sine, cosine)
    state = weighted.copy()
    np.fill_diagonal(state, -weighted.sum(axis=1))

    harmonics = np.arange(1, n_harmonics + 1, dtype=np.float64)
    scaled = harmonics[:, None, None] * difference[None, :, :]
    sine_columns = np.einsum("ij,mij->im", coupling, np.sin(scaled))
    cosine_columns = np.einsum("ij,mij->im", coupling, np.cos(scaled))
    parameter = np.empty((count, 2 * n_harmonics), dtype=np.float64)
    parameter[:, :n_harmonics] = sine_columns
    parameter[:, n_harmonics:] = cosine_columns
    return state, parameter


def _step_with_jacobians(
    phases: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance one RK4 step and return ``(next, ∂next/∂phases, ∂next/∂[a;b])``."""
    count = phases.size
    identity = np.eye(count, dtype=np.float64)

    stage1 = _field(phases, sine, cosine, omega, coupling)
    j1_state, j1_param = _state_and_parameter_jacobians(phases, sine, cosine, coupling)

    point2 = phases + 0.5 * dt * stage1
    stage2 = _field(point2, sine, cosine, omega, coupling)
    j2_state, j2_param = _state_and_parameter_jacobians(point2, sine, cosine, coupling)
    d2_state = j2_state @ (identity + 0.5 * dt * j1_state)
    d2_param = j2_state @ (0.5 * dt * j1_param) + j2_param

    point3 = phases + 0.5 * dt * stage2
    stage3 = _field(point3, sine, cosine, omega, coupling)
    j3_state, j3_param = _state_and_parameter_jacobians(point3, sine, cosine, coupling)
    d3_state = j3_state @ (identity + 0.5 * dt * d2_state)
    d3_param = j3_state @ (0.5 * dt * d2_param) + j3_param

    point4 = phases + dt * stage3
    stage4 = _field(point4, sine, cosine, omega, coupling)
    j4_state, j4_param = _state_and_parameter_jacobians(point4, sine, cosine, coupling)
    d4_state = j4_state @ (identity + dt * d3_state)
    d4_param = j4_state @ (dt * d3_param) + j4_param

    next_phases = phases + (dt / 6.0) * (stage1 + 2.0 * stage2 + 2.0 * stage3 + stage4)
    d_state = identity + (dt / 6.0) * (j1_state + 2.0 * d2_state + 2.0 * d3_state + d4_state)
    d_param = (dt / 6.0) * (j1_param + 2.0 * d2_param + 2.0 * d3_param + d4_param)
    return next_phases, d_state, d_param


def _validate_trajectory(
    initial_phases: NDArray[np.float64],
    observations: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> int:
    count = initial_phases.size
    if initial_phases.ndim != 1 or count < 1:
        raise ValueError("initial_phases must be a non-empty one-dimensional array")
    if observations.ndim != 2 or observations.shape[1] != count or observations.shape[0] < 2:
        raise ValueError(f"observations must have shape (n_samples >= 2, {count})")
    if sine.ndim != 1 or sine.shape != cosine.shape or sine.size < 1:
        raise ValueError("sine and cosine coefficients must be equal-length non-empty 1-D arrays")
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return int(count)


def coupling_function_trajectory_value_and_grad(
    initial_phases: NDArray[np.float64],
    observations: NDArray[np.float64],
    sine_coefficients: NDArray[np.float64],
    cosine_coefficients: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> CouplingFunctionGradients:
    r"""Trajectory-match loss and its exact adjoint gradient w.r.t. the Fourier coefficients.

    Integrates the model with the parametrised ``Γ`` from ``initial_phases`` and scores the
    trajectory-match loss ``Σ_k ‖θ_k − θ^obs_k‖²`` against the observed trajectory; the gradient with
    respect to the Fourier coefficients is the discrete adjoint of the RK4 map with the data-match
    cost injected at every sample, matching finite differences to machine precision.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``); must match ``observations[0]``.
    observations : numpy.ndarray
        The ``(n_samples, N)`` observed trajectory (``n_samples ≥ 2``), one sample per RK4 step.
    sine_coefficients, cosine_coefficients : numpy.ndarray
        The ``(M,)`` Fourier coefficients of the trial coupling function.
    omega : numpy.ndarray
        The known natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step (``> 0``).

    Returns
    -------
    CouplingFunctionGradients
        The trajectory-match loss and the sine / cosine coefficient gradients.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    state = np.ascontiguousarray(initial_phases, dtype=np.float64)
    observed = np.ascontiguousarray(observations, dtype=np.float64)
    sine = np.ascontiguousarray(sine_coefficients, dtype=np.float64)
    cosine = np.ascontiguousarray(cosine_coefficients, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_trajectory(state, observed, sine, cosine, frequencies, matrix, dt)
    n_harmonics = sine.size
    n_steps = observed.shape[0] - 1

    trajectory = np.empty((n_steps + 1, count), dtype=np.float64)
    trajectory[0] = state
    state_jacobians = np.empty((n_steps, count, count), dtype=np.float64)
    parameter_jacobians = np.empty((n_steps, count, 2 * n_harmonics), dtype=np.float64)
    for index in range(n_steps):
        state, d_state, d_param = _step_with_jacobians(
            state, sine, cosine, frequencies, matrix, dt
        )
        trajectory[index + 1] = state
        state_jacobians[index] = d_state
        parameter_jacobians[index] = d_param

    misfit = trajectory - observed
    loss = float(np.sum(misfit**2))

    adjoint = np.zeros(count, dtype=np.float64)
    parameter_gradient = np.zeros(2 * n_harmonics, dtype=np.float64)
    for index in range(n_steps, -1, -1):
        adjoint = adjoint + 2.0 * misfit[index]
        if index > 0:
            parameter_gradient = parameter_gradient + parameter_jacobians[index - 1].T @ adjoint
            adjoint = state_jacobians[index - 1].T @ adjoint

    return CouplingFunctionGradients(
        loss=loss,
        sine_gradient=np.ascontiguousarray(parameter_gradient[:n_harmonics], dtype=np.float64),
        cosine_gradient=np.ascontiguousarray(parameter_gradient[n_harmonics:], dtype=np.float64),
    )


def refine_coupling_function(
    initial_phases: NDArray[np.float64],
    observations: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_harmonics: int,
    *,
    learning_rate: float,
    n_iterations: int,
    initial_sine: NDArray[np.float64] | None = None,
    initial_cosine: NDArray[np.float64] | None = None,
) -> tuple[CouplingFunctionEstimate, NDArray[np.float64]]:
    r"""Recover the coupling function from a sampled trajectory by differentiable trajectory matching.

    Runs ``n_iterations`` gradient-descent updates on the Fourier coefficients using
    :func:`coupling_function_trajectory_value_and_grad`, returning the refined estimate and the loss
    history. Use this when only a noisy sampled trajectory is available (no reliable derivatives);
    it never differentiates the data.

    Parameters
    ----------
    initial_phases, observations, omega, coupling, dt
        As for :func:`coupling_function_trajectory_value_and_grad`.
    n_harmonics : int
        The number ``M`` of Fourier harmonics (``≥ 1``).
    learning_rate : float
        The gradient-descent step size (``> 0``).
    n_iterations : int
        The number of descent iterations (``≥ 1``).
    initial_sine, initial_cosine : numpy.ndarray, optional
        The starting ``(M,)`` coefficients; default to zeros.

    Returns
    -------
    tuple
        The refined :class:`CouplingFunctionEstimate` (its ``residual`` is the final trajectory-match
        loss) and the ``(n_iterations + 1,)`` loss history.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be positive, got {n_harmonics}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    sine: NDArray[np.float64]
    cosine: NDArray[np.float64]
    sine = (
        np.zeros(n_harmonics, dtype=np.float64)
        if initial_sine is None
        else np.ascontiguousarray(initial_sine, dtype=np.float64)
    )
    cosine = (
        np.zeros(n_harmonics, dtype=np.float64)
        if initial_cosine is None
        else np.ascontiguousarray(initial_cosine, dtype=np.float64)
    )
    if sine.shape != (n_harmonics,) or cosine.shape != (n_harmonics,):
        raise ValueError(f"initial coefficients must have shape ({n_harmonics},)")

    history = np.empty(n_iterations + 1, dtype=np.float64)
    gradients = coupling_function_trajectory_value_and_grad(
        initial_phases, observations, sine, cosine, frequencies, coupling, dt
    )
    for iteration in range(n_iterations):
        history[iteration] = gradients.loss
        sine = sine - learning_rate * gradients.sine_gradient
        cosine = cosine - learning_rate * gradients.cosine_gradient
        gradients = coupling_function_trajectory_value_and_grad(
            initial_phases, observations, sine, cosine, frequencies, coupling, dt
        )
    history[n_iterations] = gradients.loss
    estimate = CouplingFunctionEstimate(
        sine_coefficients=sine,
        cosine_coefficients=cosine,
        frequencies=frequencies,
        residual=float(gradients.loss),
    )
    return estimate, history


__all__ = [
    "CouplingFunctionEstimate",
    "CouplingFunctionGradients",
    "coupling_function_trajectory_value_and_grad",
    "coupling_function_value",
    "infer_coupling_function",
    "refine_coupling_function",
]
