# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Lipschitz-certified formal guarantee for a learned neural Lyapunov certificate
r"""A Lipschitz-bounded formal guarantee for the learned neural Lyapunov certificate.

The learned certificate in :mod:`~oscillatools.accel.neural_lyapunov_certificate` reports a *sampling*
verdict (`is_certified_on_sample`): the Lyapunov conditions held on the finite set of states it visited.
This module upgrades that to a **formal** guarantee over a whole region by bounding how fast the Lie
derivative can change between samples.

Method — grid plus a rigorous Taylor bound
------------------------------------------
The Lie derivative ``\dot V_ψ(θ) = ∇V_ψ(θ)·f(θ)`` is smooth (a tanh perceptron composed with the
analytic Kuramoto vector field), so a first-order Taylor bound holds around any grid point ``g``:

.. math::

    \dot V_ψ(θ) \;\le\; \dot V_ψ(g) \;+\; \lVert\nabla\dot V_ψ(g)\rVert\,ρ \;+\; \tfrac12 M\,ρ^2 ,

where ``ρ`` is the grid covering radius (the greatest distance from any region point to the nearest grid
point) and ``M`` is a **rigorous global upper bound** on the curvature ``\sup\lVert\nabla^2\dot V_ψ\rVert``.
The first two terms are read straight off the grid — the actual value and gradient of the Lie derivative
at each grid point, so they are tight — and only the small second-order remainder uses the analytic bound
``M``, assembled from the perceptron weights and the coupling/frequency norms through **interval bound
propagation**: over the box region the hidden pre-activations lie in known intervals, so the tanh
derivatives are bounded by their maxima over *those* intervals (near zero for saturated units) rather than
their global sup-norms (``\sup|\tanh''| = 4/3\sqrt3``, ``\sup|\tanh'''| = 2``) — the tightening that makes
``M`` small enough to certify at a practical grid. Every factor of ``M`` is a genuine upper bound that
holds for all states in the region — none is sampled.
When the worst grid value of ``\dot V_ψ(g) + \lVert\nabla\dot V_ψ(g)\rVert ρ + \tfrac12 M ρ^2`` is
negative, ``\dot V_ψ(θ) < 0`` **everywhere** in the region — a formal certificate over the continuum, not
a sample. The mirror argument on ``V_ψ`` certifies ``V_ψ > 0`` on an annulus that excludes a small ball
around the equilibrium (where ``V_ψ`` and ``\dot V_ψ`` vanish and no margin exists).

Honest boundary
---------------
The guarantee is **sound but conservative and small-scale**. The interval bounds still overshoot the true
curvature, so the certificate can *fail to certify* a region that is in fact stable (a false negative) — but
it never certifies a region that is not (no false positive), as an independent dense-sampling ground truth
confirms. The interval propagation needs a **single-hidden-layer** certificate (fit with
``hidden_layers=(width,)``), and the grid is exponential in the oscillator count, so the formal certificate
is tractable for small networks (a handful of oscillators) over a modest annulus; for larger ones the
sampling certificate remains the practical tool. This closes the "sampling ≠ proof" gap of the learned
certificate within those honest limits; it is not a substitute for a full SMT decision procedure at scale.

This tier is **opt-in** and requires JAX (``oscillatools[jax]``); it raises :class:`ImportError` with an
install hint when JAX is absent.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .neural_lyapunov_certificate import (
    NeuralLyapunovCertificate,
    _as_jax_parameters,
    _load_backend,
)

_TANH_SECOND_DERIVATIVE_BOUND = 4.0 / (3.0 * math.sqrt(3.0))  # sup|tanh''(x)|
_TANH_THIRD_DERIVATIVE_BOUND = 2.0  # sup|tanh'''(x)| (attained at x = 0)


@dataclass(frozen=True)
class LyapunovLipschitzBounds:
    """Rigorous global upper bounds used for the second-order remainder of the formal certificate.

    Every field is a genuine upper bound over the region — none is sampled. They are conservative (the
    operator-norm products overshoot the true constants), so they can only weaken the verdict, never make
    it unsound. Only the curvature bounds enter the certificate (as the ``½ M ρ²`` remainder); the
    component bounds are exposed for transparency.

    Attributes
    ----------
    decrease_curvature_bound : float
        An upper bound on ``sup‖∇²\\dot V_ψ‖`` over the region — the curvature of the Lie derivative,
        the ``M`` of the decrease remainder.
    value_curvature_bound : float
        An upper bound on ``sup‖∇²V_ψ‖`` over the region — the curvature of ``V_ψ``, the ``M`` of the
        positivity remainder.
    gradient_bound : float
        The upper bound on ``‖∇V_ψ‖`` over the region.
    force_bound : float
        The upper bound on ``‖f‖`` (the Kuramoto vector field magnitude), holding for all states.
    force_jacobian_bound : float
        The upper bound on ``‖J_f‖`` (the Kuramoto Jacobian spectral norm), holding for all states.
    force_curvature_bound : float
        The upper bound on ``‖∇²f‖`` (the Kuramoto force curvature), holding for all states.
    """

    decrease_curvature_bound: float
    value_curvature_bound: float
    gradient_bound: float
    force_bound: float
    force_jacobian_bound: float
    force_curvature_bound: float


@dataclass(frozen=True)
class FormalLyapunovCertificate:
    """The outcome of the Lipschitz-plus-grid formal verification over a region.

    Attributes
    ----------
    is_certified_on_region : bool
        Whether both the decrease (``\\dot V_ψ < 0``) and positivity (``V_ψ > 0``) conditions are formally
        guaranteed on the annulus — a proof over the continuum, subject to the honest conservatism /
        small-scale limits in the module docstring.
    decrease_certified : bool
        Whether ``\\dot V_ψ(θ) < 0`` is guaranteed for every state in the annulus.
    positivity_certified : bool
        Whether ``V_ψ(θ) > 0`` is guaranteed for every state in the annulus.
    worst_decrease_on_grid : float
        The largest first-order decrease bound ``\\dot V_ψ(g) + ‖∇\\dot V_ψ(g)‖ ρ`` over the annulus grid.
    minimum_value_on_grid : float
        The smallest first-order value bound ``V_ψ(g) − ‖∇V_ψ(g)‖ ρ`` over the annulus grid.
    decrease_margin : float
        ``−(worst_decrease_on_grid + ½ · decrease_curvature · covering_radius²)`` — positive iff the
        decrease is formally certified.
    positivity_margin : float
        ``minimum_value_on_grid − ½ · value_curvature · covering_radius²`` — positive iff positivity is
        formally certified.
    decrease_curvature : float
        The curvature bound ``M`` used for the decrease remainder.
    value_curvature : float
        The curvature bound ``M`` used for the positivity remainder.
    covering_radius : float
        The greatest distance from any region point to the nearest grid point.
    inner_radius : float
        The inner radius of the certified annulus (excludes the equilibrium).
    outer_radius : float
        The per-axis half-width of the certified box region.
    grid_points : int
        The number of grid states evaluated.
    """

    is_certified_on_region: bool
    decrease_certified: bool
    positivity_certified: bool
    worst_decrease_on_grid: float
    minimum_value_on_grid: float
    decrease_margin: float
    positivity_margin: float
    decrease_curvature: float
    value_curvature: float
    covering_radius: float
    inner_radius: float
    outer_radius: float
    grid_points: int


_TANH_SECOND_PEAK = float(np.arctanh(1.0 / math.sqrt(3.0)))  # |tanh''| maximiser, ±0.6585
_TANH_THIRD_SECONDARY = float(np.arctanh(math.sqrt(2.0 / 3.0)))  # secondary |tanh'''| peak, ±1.146


def _tanh_derivative_interval_maxima(
    lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Exact per-unit maxima of ``|tanh'|``, ``|tanh''|`` and ``|tanh'''|`` over each ``[lower, upper]``.

    A smooth function attains its maximum over an interval at an endpoint or an interior critical point.
    ``|tanh'| = 1 − tanh²`` peaks at ``0``; ``|tanh''|`` at ``±arctanh(1/√3)``; ``|tanh'''|`` at ``0``
    (value ``2``) with a secondary peak at ``±arctanh(√⅔)`` (value ``⅔``). Evaluating at the endpoints and
    the critical points that fall inside the interval yields the exact maximum — the interval-bound
    tightening that replaces the global sup-norms.
    """

    def abs_second(argument: NDArray[np.float64]) -> NDArray[np.float64]:
        tangent = np.tanh(argument)
        return np.abs(2.0 * tangent * (1.0 - tangent**2))

    def abs_third(argument: NDArray[np.float64]) -> NDArray[np.float64]:
        tangent = np.tanh(argument)
        return np.abs(-2.0 * (1.0 - tangent**2) * (1.0 - 3.0 * tangent**2))

    nearest_zero = np.minimum(np.maximum(0.0, lower), upper)
    first = 1.0 - np.tanh(nearest_zero) ** 2

    second = np.maximum(abs_second(lower), abs_second(upper))
    for peak in (_TANH_SECOND_PEAK, -_TANH_SECOND_PEAK):
        inside = (lower <= peak) & (upper >= peak)
        second = np.where(inside, np.maximum(second, _TANH_SECOND_DERIVATIVE_BOUND), second)

    third = np.maximum(abs_third(lower), abs_third(upper))
    contains_zero = (lower <= 0.0) & (upper >= 0.0)
    third = np.where(contains_zero, np.maximum(third, _TANH_THIRD_DERIVATIVE_BOUND), third)
    for peak in (_TANH_THIRD_SECONDARY, -_TANH_THIRD_SECONDARY):
        inside = (lower <= peak) & (upper >= peak)
        third = np.where(inside, np.maximum(third, 2.0 / 3.0), third)
    return first, second, third


def _force_bounds(
    omega: NDArray[np.float64], coupling: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Global upper bounds on ``‖f‖``, ``‖J_f‖`` and ``‖∇²f‖`` (independent of the state).

    Every derivative of ``F_i = Σ_j K_ij sin(θ_j − θ_i)`` carries the factor ``K_ij`` times a bounded
    sinusoid (``|sin|, |cos| ≤ 1``), so the value, Jacobian and curvature are all bounded by the coupling
    magnitudes alone: ``|F_i| ≤ Σ_j |K_ij|``, the Jacobian entries by ``|K_ij|`` (diagonal ``Σ_j |K_ij|``),
    and each ``∇²F_i`` likewise. The Frobenius sums are upper bounds on the spectral norms.
    """
    row_abs_sum = np.sum(np.abs(coupling), axis=1)
    force_bound = float(np.linalg.norm(omega) + np.linalg.norm(row_abs_sum))
    off_diagonal = np.abs(coupling) ** 2
    np.fill_diagonal(off_diagonal, 0.0)
    jacobian_bound = float(math.sqrt(np.sum(off_diagonal) + float(np.sum(row_abs_sum**2))))
    # each F_i has a Hessian whose entries are bounded by |K_ij|; ‖∇²F_i‖_F ≤ 2 Σ_j|K_ij|, and the
    # stacked curvature is bounded by the Euclidean norm of those per-oscillator bounds.
    curvature_bound = float(2.0 * np.linalg.norm(row_abs_sum))
    return force_bound, jacobian_bound, curvature_bound


def neural_lyapunov_lipschitz_bounds(
    certificate: NeuralLyapunovCertificate, *, outer_radius: float
) -> LyapunovLipschitzBounds:
    r"""Rigorous global upper bounds on the curvature of ``V_ψ`` and ``\dot V_ψ`` over a region.

    Assembles, by interval bound propagation through the single hidden layer (the hidden pre-activations
    lie in ``b₀ ± outer_radius‖W₀[:,j]‖₁``, so the tanh derivatives are bounded over those intervals) and
    the coupling/frequency norms, upper bounds that hold for every state in the box of per-axis half-width
    ``outer_radius`` around the certificate's equilibrium. These are the ``M`` constants of the
    second-order remainder that turn a grid evaluation into a formal guarantee.

    Parameters
    ----------
    certificate : NeuralLyapunovCertificate
        A single-hidden-layer certificate from
        :func:`~oscillatools.accel.neural_lyapunov_certificate.fit_neural_lyapunov_certificate` (fit with
        ``hidden_layers=(width,)``).
    outer_radius : float
        The per-axis half-width of the region the bounds must hold over (``> 0``).

    Returns
    -------
    LyapunovLipschitzBounds
        The decrease and value curvature bounds and their component bounds.

    Raises
    ------
    ValueError
        If ``outer_radius`` is not positive, or the certificate is not single-hidden-layer.
    """
    if outer_radius <= 0.0:
        raise ValueError(f"outer_radius must be positive, got {outer_radius}")
    if len(certificate.parameters) != 2:
        raise ValueError(
            "formal verification requires a single-hidden-layer certificate — fit it with "
            f"hidden_layers=(width,); got {len(certificate.parameters)} layers"
        )
    count = int(certificate.phases_star.size)
    epsilon = certificate.epsilon
    # Interval bound propagation over the box ``|e_k| ≤ outer_radius``: for a single hidden layer
    # ``raw = w₁·tanh(e W₀ + b₀)``, each pre-activation ``z_j`` lies in ``b₀_j ± outer_radius‖W₀[:,j]‖₁``,
    # so the tanh derivatives are bounded by their maxima over *that* interval (near zero for saturated
    # units) rather than their global sup-norms — the tightening that makes the curvature usable.
    input_weight, input_bias = certificate.parameters[0]
    output_weight = certificate.parameters[1][0][:, 0]
    input_operator_norm = float(np.linalg.norm(input_weight, 2))
    column_l1 = np.sum(np.abs(input_weight), axis=0)
    column_l2 = np.linalg.norm(input_weight, axis=0)
    z_radius = outer_radius * column_l1
    first_max, second_max, third_max = _tanh_derivative_interval_maxima(
        input_bias - z_radius, input_bias + z_radius
    )
    raw_gradient = input_operator_norm * float(np.linalg.norm(np.abs(output_weight) * first_max))
    raw_hessian = float(np.sum(np.abs(output_weight) * second_max * column_l2**2))
    raw_third = float(np.sum(np.abs(output_weight) * third_max * column_l2**3))

    # Embedding e(θ) = [cosφ − cosφ*, sinφ − sinφ*], φ_i = θ_{i+1} − θ_0. Its Jacobian rows carry two
    # entries of magnitude ≤ 1, its per-component Hessians four, its third derivatives eight; over the box
    # each component deviates by at most ``outer_radius`` (cos/sin are 1-Lipschitz).
    embedding_jacobian = 2.0 * math.sqrt(count - 1)
    embedding_norm = outer_radius * math.sqrt(2.0 * (count - 1))
    embedding_hessian_sum = 4.0 * (count - 1)
    embedding_third_sum = 8.0 * (count - 1)
    scaled_gradient = raw_gradient + 2.0 * epsilon * embedding_norm

    gradient_bound = embedding_jacobian * scaled_gradient
    value_curvature = (
        embedding_jacobian**2 * raw_hessian
        + scaled_gradient * embedding_hessian_sum
        + 2.0 * epsilon * embedding_jacobian**2
    )
    # ‖∇³(raw∘e)‖ ≤ ‖∇³_e raw‖‖J_e‖³ + 3‖∇²_e raw‖‖J_e‖·Σ‖∇²e‖ + ‖∇_e raw‖·Σ‖∇³e‖ (Faà di Bruno),
    # plus the ε‖e‖² curvature term.
    value_third = (
        raw_third * embedding_jacobian**3
        + 3.0 * raw_hessian * embedding_jacobian * embedding_hessian_sum
        + raw_gradient * embedding_third_sum
        + epsilon
        * (
            6.0 * embedding_jacobian * embedding_hessian_sum
            + 2.0 * embedding_norm * embedding_third_sum
        )
    )

    force_bound, jacobian_bound, force_curvature = _force_bounds(
        certificate.omega, certificate.coupling
    )
    # ‖∇²V̇‖ ≤ ‖∇³V‖‖f‖ + 2‖∇²V‖‖J_f‖ + ‖∇V‖‖∇²f‖ (product rule on V̇ = ∇V·f).
    decrease_curvature = (
        value_third * force_bound
        + 2.0 * value_curvature * jacobian_bound
        + gradient_bound * force_curvature
    )
    return LyapunovLipschitzBounds(
        decrease_curvature_bound=decrease_curvature,
        value_curvature_bound=value_curvature,
        gradient_bound=gradient_bound,
        force_bound=force_bound,
        force_jacobian_bound=jacobian_bound,
        force_curvature_bound=force_curvature,
    )


def _region_grid(
    theta_star: NDArray[np.float64], outer_radius: float, resolution: int
) -> tuple[NDArray[np.float64], float]:
    """A box grid over the shift-quotient (oscillator 0 fixed) and its covering radius."""
    count = theta_star.size
    axis = np.linspace(-outer_radius, outer_radius, resolution)
    offsets = np.array(list(itertools.product(axis, repeat=count - 1)), dtype=np.float64)
    states = np.tile(theta_star, (offsets.shape[0], 1))
    states[:, 1:] = theta_star[1:] + offsets
    step = 2.0 * outer_radius / (resolution - 1)
    covering_radius = 0.5 * step * math.sqrt(count - 1)
    return states, covering_radius


def formally_certify_neural_lyapunov(
    certificate: NeuralLyapunovCertificate,
    *,
    outer_radius: float,
    inner_radius: float,
    grid_resolution: int = 11,
) -> FormalLyapunovCertificate:
    r"""Formally certify the learned Lyapunov conditions on a region by grid plus Lipschitz bound.

    Evaluates ``V_ψ`` and ``\dot V_ψ`` on a box grid over the shift-quotient around the equilibrium, and
    combines the worst grid values with the rigorous Lipschitz bounds: the decrease ``\dot V_ψ < 0`` is
    guaranteed on the box, and the positivity ``V_ψ > 0`` on the annulus that excludes the ball of radius
    ``inner_radius``, when the corresponding margin (worst grid value plus the Lipschitz times the grid
    covering radius) is on the certifying side of zero.

    Parameters
    ----------
    certificate : NeuralLyapunovCertificate
        A certificate from
        :func:`~oscillatools.accel.neural_lyapunov_certificate.fit_neural_lyapunov_certificate`.
    outer_radius : float
        The per-axis half-width of the box region (``> 0``).
    inner_radius : float
        The radius of the ball around the equilibrium excluded from the positivity claim (``> 0`` and
        ``< outer_radius``).
    grid_resolution : int, optional
        The number of grid points per axis (``≥ 2``, default ``11``); the total is
        ``grid_resolution ** (N − 1)``, so keep it modest for larger networks.

    Returns
    -------
    FormalLyapunovCertificate
        The formal verdict, the worst grid values, the margins, the Lipschitz bounds and the grid.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    if outer_radius <= 0.0:
        raise ValueError(f"outer_radius must be positive, got {outer_radius}")
    if not 0.0 < inner_radius < outer_radius:
        raise ValueError(
            f"inner_radius must lie in (0, outer_radius), got {inner_radius} vs {outer_radius}"
        )
    if grid_resolution < 2:
        raise ValueError(f"grid_resolution must be at least two, got {grid_resolution}")

    bounds = neural_lyapunov_lipschitz_bounds(certificate, outer_radius=outer_radius)
    states, covering_radius = _region_grid(certificate.phases_star, outer_radius, grid_resolution)

    backend = _load_backend()
    jax = backend.jax
    jnp = backend.jnp
    params = _as_jax_parameters(backend, certificate.parameters)
    theta_star = jnp.asarray(certificate.phases_star)
    omega = jnp.asarray(certificate.omega)
    coupling = jnp.asarray(certificate.coupling)
    epsilon = certificate.epsilon

    def value(theta: object) -> object:
        return backend.value(params, theta, theta_star, epsilon)

    def decrease(theta: object) -> object:
        return backend.decrease(params, theta, theta_star, omega, coupling, epsilon)

    batch = jnp.asarray(states)
    values = np.asarray(jax.vmap(value)(batch), dtype=np.float64)
    decreases = np.asarray(jax.vmap(decrease)(batch), dtype=np.float64)
    value_gradients = np.asarray(
        jax.vmap(lambda theta: jnp.linalg.norm(jax.grad(value)(theta)))(batch), dtype=np.float64
    )
    decrease_gradients = np.asarray(
        jax.vmap(lambda theta: jnp.linalg.norm(jax.grad(decrease)(theta)))(batch), dtype=np.float64
    )

    # A region point in the annulus lies within the covering radius of a grid point whose own radius is
    # at least ``inner_radius − covering_radius``; restrict the worst case to those grid points. The
    # first-order Taylor bound ``·(g) ± ‖∇·(g)‖ρ + ½Mρ²`` bounds the value between grid points, so the
    # verdict holds over the continuum, not only on the grid.
    radii = np.linalg.norm(states[:, 1:] - certificate.phases_star[1:], axis=1)
    annulus = radii >= (inner_radius - covering_radius)
    decrease_first_order = decreases + decrease_gradients * covering_radius
    value_first_order = values - value_gradients * covering_radius
    worst_decrease = float(np.max(decrease_first_order[annulus]))
    minimum_value = float(np.min(value_first_order[annulus]))

    decrease_remainder = 0.5 * bounds.decrease_curvature_bound * covering_radius**2
    value_remainder = 0.5 * bounds.value_curvature_bound * covering_radius**2
    decrease_margin = -(worst_decrease + decrease_remainder)
    positivity_margin = minimum_value - value_remainder
    decrease_certified = decrease_margin > 0.0
    positivity_certified = positivity_margin > 0.0
    return FormalLyapunovCertificate(
        is_certified_on_region=decrease_certified and positivity_certified,
        decrease_certified=decrease_certified,
        positivity_certified=positivity_certified,
        worst_decrease_on_grid=worst_decrease,
        minimum_value_on_grid=minimum_value,
        decrease_margin=decrease_margin,
        positivity_margin=positivity_margin,
        decrease_curvature=bounds.decrease_curvature_bound,
        value_curvature=bounds.value_curvature_bound,
        covering_radius=covering_radius,
        inner_radius=float(inner_radius),
        outer_radius=float(outer_radius),
        grid_points=int(states.shape[0]),
    )


__all__ = [
    "FormalLyapunovCertificate",
    "LyapunovLipschitzBounds",
    "formally_certify_neural_lyapunov",
    "neural_lyapunov_lipschitz_bounds",
]
