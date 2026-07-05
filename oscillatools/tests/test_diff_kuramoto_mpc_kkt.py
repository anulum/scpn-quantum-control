# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the implicit-function-theorem MPC-optimum sensitivity tier
r"""Contract tests for differentiating through the Kuramoto MPC optimum by the implicit function theorem.

These exercise real JAX and skip without the optional ``[jax]`` extra. The load-bearing claim is the
RG1 second witness of the design: at a converged, non-degenerate optimum the level-2 implicit-function
gradient must reproduce the level-1 gradient obtained by differentiating *through* the unrolled solver,
and a central finite difference, to machine-ish tolerance — with the level-1 gradient converging onto
the level-2 gradient as the solver tightens. The remaining tests pin the reverse-mode pullback against a
hand-seeded cotangent, the plan-energy convenience, the forward solve, the reported optimality residual,
determinism, and every validation and residency contract.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from oscillatools.accel import diff_kuramoto_mpc_kkt as kkt  # noqa: E402
from oscillatools.accel.jax_kuramoto_mpc import _load_backend  # noqa: E402

_DT = 0.05
_TARGET = 0.4
_WEIGHT = 5e-2  # a control weight that keeps the objective Hessian well-conditioned
_STEP = 0.3


def _problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(phases, omega, coupling)`` for a small, non-degenerate horizon problem."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.3, size=n)
    coupling = rng.uniform(0.0, 1.0, size=(n, n))
    np.fill_diagonal(coupling, 0.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return phases, omega, coupling


def _level_one_energy_gradient(
    phases: np.ndarray,
    omega: np.ndarray,
    coupling: np.ndarray,
    horizon: int,
    n_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate the plan energy ``½‖u*‖²`` *through* the unrolled solver (the level-1 witness)."""
    backend = _load_backend()
    jnp = backend.jnp
    jax = backend.jax

    def energy(ph: object, om: object, cp: object) -> object:
        control, _ = backend.horizon_solve(
            ph,
            om,
            cp,
            _DT,
            _TARGET,
            _WEIGHT,
            _STEP,
            n_iterations,
            jnp.zeros((horizon, phases.size)),
            False,
        )
        return 0.5 * jnp.sum(control * control)

    grads = jax.grad(energy, argnums=(0, 1, 2))(
        jnp.asarray(phases), jnp.asarray(omega), jnp.asarray(coupling)
    )
    return tuple(np.asarray(g, dtype=np.float64) for g in grads)  # type: ignore[return-value]


def test_level_two_matches_level_one_unroll_at_convergence() -> None:
    """The IFT gradient reproduces the unrolled-solver gradient to machine-ish tolerance at convergence."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=5)
    result = kkt.mpc_plan_energy_gradient(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=9000,
    )
    l1_phases, l1_omega, l1_coupling = _level_one_energy_gradient(
        phases, omega, coupling, horizon, 9000
    )
    assert result.stationarity_residual < 1e-6
    np.testing.assert_allclose(result.phases_cotangent, l1_phases, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(result.omega_cotangent, l1_omega, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(result.coupling_cotangent, l1_coupling, rtol=0.0, atol=1e-4)


def test_level_one_converges_onto_level_two_as_the_solver_tightens() -> None:
    """The unrolled-solver gradient approaches the fixed IFT gradient as iterations increase."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=6)
    reference = kkt.mpc_plan_energy_gradient(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=12000,
    ).omega_cotangent
    gaps = []
    for iterations in (500, 2000, 8000):
        l1 = _level_one_energy_gradient(phases, omega, coupling, horizon, iterations)[1]
        gaps.append(float(np.linalg.norm(l1 - reference)))
    # monotone contraction of the unroll toward the implicit-function limit
    assert gaps[0] > gaps[1] > gaps[2]


def test_ift_gradient_matches_central_finite_difference() -> None:
    """A central finite difference of the plan energy confirms the IFT gradient independently."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=7)
    iterations = 9000
    result = kkt.mpc_plan_energy_gradient(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=iterations,
    )

    def energy(ph: np.ndarray, om: np.ndarray, cp: np.ndarray) -> float:
        control = kkt.jax_mpc_optimum(
            ph,
            om,
            cp,
            _DT,
            horizon,
            target_coherence=_TARGET,
            control_weight=_WEIGHT,
            step_size=_STEP,
            n_iterations=iterations,
        )
        return 0.5 * float(np.sum(control * control))

    eps = 1e-6
    for k in range(n):
        up = omega.copy()
        up[k] += eps
        down = omega.copy()
        down[k] -= eps
        fd = (energy(phases, up, coupling) - energy(phases, down, coupling)) / (2.0 * eps)
        assert abs(result.omega_cotangent[k] - fd) < 5e-4


def test_reverse_mode_pullback_matches_the_unrolled_vjp() -> None:
    """A hand-seeded control cotangent pulls back through the optimum as the unrolled solver's VJP does."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=8)
    cotangent = np.random.default_rng(11).normal(0.0, 1.0, size=(horizon, n))
    iterations = 9000
    got = kkt.mpc_optimum_parameter_sensitivity(
        phases,
        omega,
        coupling,
        _DT,
        cotangent,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=iterations,
    )
    backend = _load_backend()
    jnp = backend.jnp
    jax = backend.jax

    def solve(ph: object, om: object, cp: object) -> object:
        control, _ = backend.horizon_solve(
            ph, om, cp, _DT, _TARGET, _WEIGHT, _STEP, iterations, jnp.zeros((horizon, n)), False
        )
        return control

    _, pullback = jax.vjp(solve, jnp.asarray(phases), jnp.asarray(omega), jnp.asarray(coupling))
    l1_phases, l1_omega, l1_coupling = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(got.phases_cotangent, np.asarray(l1_phases), rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(got.omega_cotangent, np.asarray(l1_omega), rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(
        got.coupling_cotangent, np.asarray(l1_coupling), rtol=0.0, atol=1e-4
    )


def test_plan_energy_gradient_seeds_the_optimum_as_its_own_cotangent() -> None:
    """The plan-energy convenience equals the general pullback seeded with ``u*`` itself."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=9)
    iterations = 6000
    optimum = kkt.jax_mpc_optimum(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=iterations,
    )
    seeded = kkt.mpc_optimum_parameter_sensitivity(
        phases,
        omega,
        coupling,
        _DT,
        optimum,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=iterations,
    )
    convenience = kkt.mpc_plan_energy_gradient(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=iterations,
    )
    np.testing.assert_allclose(
        convenience.omega_cotangent, seeded.omega_cotangent, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        convenience.phases_cotangent, seeded.phases_cotangent, rtol=0.0, atol=1e-10
    )


def test_forward_solve_matches_the_shipped_horizon_control() -> None:
    """``jax_mpc_optimum`` returns exactly the shipped horizon-control optimum, shape ``(horizon, N)``."""
    from oscillatools.accel.jax_kuramoto_mpc import jax_mpc_horizon_control

    n, horizon = 4, 5
    phases, omega, coupling = _problem(n, seed=1)
    optimum = kkt.jax_mpc_optimum(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=800,
    )
    reference, _ = jax_mpc_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=800,
    )
    assert optimum.shape == (horizon, n)
    np.testing.assert_allclose(optimum, reference, rtol=0.0, atol=0.0)


def test_reported_residual_tightens_with_iterations() -> None:
    """The reported stationarity residual shrinks as the solver runs longer."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=2)
    cotangent = np.ones((horizon, n))
    loose = kkt.mpc_optimum_parameter_sensitivity(
        phases,
        omega,
        coupling,
        _DT,
        cotangent,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=100,
    )
    tight = kkt.mpc_optimum_parameter_sensitivity(
        phases,
        omega,
        coupling,
        _DT,
        cotangent,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=_STEP,
        n_iterations=9000,
    )
    assert tight.stationarity_residual < loose.stationarity_residual
    assert tight.stationarity_residual < 1e-6


def test_adam_optimiser_and_warm_start_paths() -> None:
    """The Adam schedule and a warm-start control both drive to a low-residual optimum."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=3)
    warm = 0.01 * np.ones((horizon, n))
    result = kkt.mpc_plan_energy_gradient(
        phases,
        omega,
        coupling,
        _DT,
        horizon,
        target_coherence=_TARGET,
        control_weight=_WEIGHT,
        step_size=0.02,
        n_iterations=4000,
        optimiser="adam",
        initial_control=warm,
    )
    assert result.stationarity_residual < 1e-4
    assert result.optimal_control.shape == (horizon, n)


def test_sensitivity_is_deterministic() -> None:
    """Identical inputs give identical gradients."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=4)
    cotangent = np.random.default_rng(0).normal(size=(horizon, n))
    kwargs = dict(
        target_coherence=_TARGET, control_weight=_WEIGHT, step_size=_STEP, n_iterations=1500
    )
    first = kkt.mpc_optimum_parameter_sensitivity(
        phases, omega, coupling, _DT, cotangent, **kwargs
    )
    second = kkt.mpc_optimum_parameter_sensitivity(
        phases, omega, coupling, _DT, cotangent, **kwargs
    )
    np.testing.assert_array_equal(first.coupling_cotangent, second.coupling_cotangent)
    np.testing.assert_array_equal(first.phases_cotangent, second.phases_cotangent)


_BASE = dict(target_coherence=_TARGET, control_weight=_WEIGHT, step_size=_STEP, n_iterations=100)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p | {"phases": np.zeros((2, 2))}, "one-dimensional"),
        (lambda p: p | {"omega": np.zeros(2)}, "omega must have shape"),
        (lambda p: p | {"coupling": np.zeros((2, 2))}, "coupling must have shape"),
        (lambda p: p | {"dt": 0.0}, "dt must be positive"),
        (lambda p: p | {"target_coherence": 1.5}, r"target_coherence must be in \[0, 1\]"),
        (lambda p: p | {"control_weight": -1.0}, "control_weight must be non-negative"),
        (lambda p: p | {"step_size": 0.0}, "step_size must be positive"),
        (lambda p: p | {"n_iterations": 0}, "n_iterations must be positive"),
        (lambda p: p | {"optimiser": "sgd"}, "optimiser must be one of"),
    ],
)
def test_sensitivity_validation(mutate: object, message: str) -> None:
    """Every documented bound on the sensitivity entry point raises ``ValueError``."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=0)
    params = {
        "phases": phases,
        "omega": omega,
        "coupling": coupling,
        "dt": _DT,
        "control_cotangent": np.ones((horizon, n)),
        **_BASE,
    }
    params = mutate(params)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        kkt.mpc_optimum_parameter_sensitivity(
            params["phases"],
            params["omega"],
            params["coupling"],
            params["dt"],
            params["control_cotangent"],
            target_coherence=params["target_coherence"],
            control_weight=params["control_weight"],
            step_size=params["step_size"],
            n_iterations=params["n_iterations"],
            optimiser=params.get("optimiser", "gd"),
        )


def test_cotangent_shape_is_validated() -> None:
    """A control cotangent whose oscillator axis mismatches the problem is rejected."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=0)
    with pytest.raises(ValueError, match="control_cotangent must have shape"):
        kkt.mpc_optimum_parameter_sensitivity(
            phases, omega, coupling, _DT, np.ones((horizon, n + 1)), **_BASE
        )


def test_initial_control_shape_is_validated() -> None:
    """A warm-start control whose shape mismatches the horizon problem is rejected."""
    n, horizon = 3, 4
    phases, omega, coupling = _problem(n, seed=0)
    with pytest.raises(ValueError, match="initial_control must have shape"):
        kkt.mpc_optimum_parameter_sensitivity(
            phases,
            omega,
            coupling,
            _DT,
            np.ones((horizon, n)),
            initial_control=np.zeros((horizon + 1, n)),
            **_BASE,
        )


def test_energy_gradient_rejects_a_non_positive_horizon() -> None:
    """The plan-energy convenience rejects a non-positive horizon."""
    n = 3
    phases, omega, coupling = _problem(n, seed=0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        kkt.mpc_plan_energy_gradient(phases, omega, coupling, _DT, 0, **_BASE)
