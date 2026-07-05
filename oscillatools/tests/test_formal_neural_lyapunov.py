# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — tests for the Lipschitz-certified formal neural Lyapunov guarantee
r"""Contract tests for the interval-bound formal neural Lyapunov certificate.

These exercise real JAX and skip without the optional ``[jax]`` extra. The load-bearing claims are
soundness: every analytic bound is a genuine upper bound on the densely sampled quantity (the interval
curvature bounds, the gradient bound and the force bounds), so the ``½ M ρ²`` remainder can never
understate the true variation; a region the certificate certifies is confirmed stable by an independent
dense-sampling ground truth (no false positive), while a repulsive coupling is refused; and the single
hidden layer requirement, validation and record contracts hold.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from oscillatools.accel import neural_lyapunov_certificate as nlc  # noqa: E402
from oscillatools.accel.formal_neural_lyapunov import (  # noqa: E402
    FormalLyapunovCertificate,
    LyapunovLipschitzBounds,
    formally_certify_neural_lyapunov,
    neural_lyapunov_lipschitz_bounds,
)
from oscillatools.accel.neural_lyapunov_certificate import (  # noqa: E402
    NeuralLyapunovCertificate,
    fit_neural_lyapunov_certificate,
)

_N = 3
_OUTER = 0.35
_INNER = 0.28
_RESOLUTION = 51
# A single-hidden-layer configuration (interval bound propagation supports one hidden layer).
_CONFIG = dict(
    region_radius=0.5,
    hidden_layers=(16,),
    epsilon=5e-2,
    learning_rate=1e-2,
    iterations=250,
    sample_size=128,
    warm_start_iterations=200,
    falsifier_rounds=2,
    falsifier_restarts=8,
    falsifier_steps=25,
    relaxation_steps=100,
)


def _symmetric_coupling(strength: float) -> np.ndarray:
    return (np.ones((_N, _N)) - np.eye(_N)) * strength


@pytest.fixture(scope="module")
def certificate() -> NeuralLyapunovCertificate:
    """A single-hidden-layer certificate for a symmetric, attractive, certifiable regime."""
    return fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(0.9), seed=1, **_CONFIG
    )


def _dense_region(
    phases_star: np.ndarray, outer: float, inner: float, count: int, seed: int
) -> np.ndarray:
    """A dense sample of the box annulus (oscillator 0 fixed) the certificate reasons over."""
    rng = np.random.default_rng(seed)
    offsets = np.zeros((count, phases_star.size))
    offsets[:, 1:] = rng.uniform(-outer, outer, (count, phases_star.size - 1))
    states = phases_star + offsets
    radii = np.linalg.norm(states[:, 1:] - phases_star[1:], axis=1)
    return states[radii >= inner]


def test_curvature_bounds_upper_bound_the_true_curvature(
    certificate: NeuralLyapunovCertificate,
) -> None:
    """The interval curvature bounds exceed the densely sampled Hessian spectral norms."""
    import jax
    import jax.numpy as jnp

    backend = nlc._load_backend()
    params = nlc._as_jax_parameters(backend, certificate.parameters)
    star = jnp.asarray(certificate.phases_star)
    omega = jnp.asarray(certificate.omega)
    coupling = jnp.asarray(certificate.coupling)
    epsilon = certificate.epsilon
    value_hessian = jax.jit(jax.hessian(lambda theta: backend.value(params, theta, star, epsilon)))
    decrease_hessian = jax.jit(
        jax.hessian(lambda theta: backend.decrease(params, theta, star, omega, coupling, epsilon))
    )
    bounds = neural_lyapunov_lipschitz_bounds(certificate, outer_radius=_OUTER)
    states = _dense_region(certificate.phases_star, _OUTER, 0.0, 4000, seed=5)[:600]
    worst_value = max(
        float(jnp.linalg.norm(value_hessian(jnp.asarray(state)), 2)) for state in states
    )
    worst_decrease = max(
        float(jnp.linalg.norm(decrease_hessian(jnp.asarray(state)), 2)) for state in states
    )
    assert bounds.value_curvature_bound >= worst_value
    assert bounds.decrease_curvature_bound >= worst_decrease


def test_component_bounds_upper_bound_the_true_quantities(
    certificate: NeuralLyapunovCertificate,
) -> None:
    """The gradient and force bounds exceed the densely sampled gradient and force norms."""
    import jax
    import jax.numpy as jnp

    backend = nlc._load_backend()
    params = nlc._as_jax_parameters(backend, certificate.parameters)
    star = jnp.asarray(certificate.phases_star)
    coupling = jnp.asarray(certificate.coupling)
    epsilon = certificate.epsilon
    value_gradient = jax.jit(jax.grad(lambda theta: backend.value(params, theta, star, epsilon)))

    def force(theta: object) -> object:
        return jnp.asarray(certificate.omega) + jnp.sum(
            coupling * jnp.sin(theta[None, :] - theta[:, None]), axis=1
        )

    force_jacobian = jax.jit(jax.jacfwd(force))
    bounds = neural_lyapunov_lipschitz_bounds(certificate, outer_radius=_OUTER)
    states = _dense_region(certificate.phases_star, _OUTER, 0.0, 4000, seed=7)[:600]
    worst_gradient = max(
        float(jnp.linalg.norm(value_gradient(jnp.asarray(state)))) for state in states
    )
    worst_force = max(float(jnp.linalg.norm(force(jnp.asarray(state)))) for state in states)
    worst_jacobian = max(
        float(jnp.linalg.norm(force_jacobian(jnp.asarray(state)), 2)) for state in states
    )
    assert bounds.gradient_bound >= worst_gradient
    assert bounds.force_bound >= worst_force
    assert bounds.force_jacobian_bound >= worst_jacobian


def test_stable_region_is_formally_certified(certificate: NeuralLyapunovCertificate) -> None:
    """A comfortably stable annulus is certified with positive margins."""
    formal = formally_certify_neural_lyapunov(
        certificate, outer_radius=_OUTER, inner_radius=_INNER, grid_resolution=_RESOLUTION
    )
    assert formal.is_certified_on_region
    assert formal.decrease_certified
    assert formal.positivity_certified
    assert formal.decrease_margin > 0.0
    assert formal.positivity_margin > 0.0


def test_certified_region_has_no_violation_under_dense_sampling(
    certificate: NeuralLyapunovCertificate,
) -> None:
    """The formal verdict is confirmed: dense sampling of the region finds no Lyapunov violation."""
    import jax
    import jax.numpy as jnp

    formal = formally_certify_neural_lyapunov(
        certificate, outer_radius=_OUTER, inner_radius=_INNER, grid_resolution=_RESOLUTION
    )
    assert formal.is_certified_on_region
    backend = nlc._load_backend()
    params = nlc._as_jax_parameters(backend, certificate.parameters)
    star = jnp.asarray(certificate.phases_star)
    omega = jnp.asarray(certificate.omega)
    coupling = jnp.asarray(certificate.coupling)
    epsilon = certificate.epsilon
    states = _dense_region(certificate.phases_star, _OUTER, _INNER, 400000, seed=123)
    batch = jnp.asarray(states)
    values = np.asarray(jax.vmap(lambda theta: backend.value(params, theta, star, epsilon))(batch))
    decreases = np.asarray(
        jax.vmap(lambda theta: backend.decrease(params, theta, star, omega, coupling, epsilon))(
            batch
        )
    )
    assert float(np.max(decreases)) < 0.0
    assert float(np.min(values)) > 0.0


def test_repulsive_coupling_is_not_certified() -> None:
    """A repulsive regime, which is not stable, is refused by the formal certificate."""
    repulsive = fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(-0.9), seed=1, **_CONFIG
    )
    formal = formally_certify_neural_lyapunov(
        repulsive, outer_radius=_OUTER, inner_radius=_INNER, grid_resolution=_RESOLUTION
    )
    assert not formal.is_certified_on_region


def test_multi_hidden_layer_certificate_is_rejected() -> None:
    """Interval bound propagation supports a single hidden layer; deeper networks are rejected."""
    deep = fit_neural_lyapunov_certificate(
        np.zeros(_N),
        np.zeros(_N),
        _symmetric_coupling(0.9),
        seed=1,
        **{**_CONFIG, "hidden_layers": (8, 8)},
    )
    with pytest.raises(ValueError):
        neural_lyapunov_lipschitz_bounds(deep, outer_radius=_OUTER)
    with pytest.raises(ValueError):
        formally_certify_neural_lyapunov(
            deep, outer_radius=_OUTER, inner_radius=_INNER, grid_resolution=21
        )


def test_verdict_is_the_conjunction_of_the_two_conditions(
    certificate: NeuralLyapunovCertificate,
) -> None:
    """``is_certified_on_region`` holds exactly when both conditions do."""
    formal = formally_certify_neural_lyapunov(
        certificate, outer_radius=_OUTER, inner_radius=_INNER, grid_resolution=_RESOLUTION
    )
    assert formal.is_certified_on_region == (
        formal.decrease_certified and formal.positivity_certified
    )
    assert formal.decrease_certified == (formal.decrease_margin > 0.0)
    assert formal.positivity_certified == (formal.positivity_margin > 0.0)


def test_certificate_reports_its_region_and_grid(
    certificate: NeuralLyapunovCertificate,
) -> None:
    """The record carries the region, the grid size and the curvature bounds."""
    formal = formally_certify_neural_lyapunov(
        certificate, outer_radius=_OUTER, inner_radius=_INNER, grid_resolution=_RESOLUTION
    )
    assert isinstance(formal, FormalLyapunovCertificate)
    assert formal.outer_radius == _OUTER
    assert formal.inner_radius == _INNER
    assert formal.grid_points == _RESOLUTION ** (_N - 1)
    assert formal.covering_radius == pytest.approx(
        0.5 * (2.0 * _OUTER / (_RESOLUTION - 1)) * np.sqrt(_N - 1)
    )
    assert formal.decrease_curvature > 0.0
    assert formal.value_curvature > 0.0


def test_bounds_record_exposes_its_components(certificate: NeuralLyapunovCertificate) -> None:
    """The bounds record carries the curvature and component bounds, all positive."""
    bounds = neural_lyapunov_lipschitz_bounds(certificate, outer_radius=_OUTER)
    assert isinstance(bounds, LyapunovLipschitzBounds)
    assert bounds.decrease_curvature_bound > 0.0
    assert bounds.value_curvature_bound > 0.0
    assert bounds.gradient_bound > 0.0
    assert bounds.force_bound > 0.0
    assert bounds.force_jacobian_bound > 0.0
    assert bounds.force_curvature_bound > 0.0


@pytest.mark.parametrize(
    "override",
    [
        {"outer_radius": 0.0},
        {"inner_radius": 0.0},
        {"inner_radius": _OUTER + 0.1},
        {"grid_resolution": 1},
    ],
)
def test_formal_certify_rejects_out_of_bound_arguments(
    certificate: NeuralLyapunovCertificate, override: dict[str, object]
) -> None:
    """The region and grid bounds are validated."""
    kwargs = {
        "outer_radius": _OUTER,
        "inner_radius": _INNER,
        "grid_resolution": _RESOLUTION,
        **override,
    }
    with pytest.raises(ValueError):
        formally_certify_neural_lyapunov(certificate, **kwargs)


def test_lipschitz_bounds_rejects_non_positive_radius(
    certificate: NeuralLyapunovCertificate,
) -> None:
    """The bounds require a positive region radius."""
    with pytest.raises(ValueError):
        neural_lyapunov_lipschitz_bounds(certificate, outer_radius=0.0)
