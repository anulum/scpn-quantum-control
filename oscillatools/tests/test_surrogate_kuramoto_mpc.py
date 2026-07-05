# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — tests for the learned-surrogate receding-horizon MPC tier
r"""Contract tests for the learned-surrogate Kuramoto MPC.

These exercise real JAX and skip without the optional ``[jax]`` extra. The load-bearing claims: the
control-conditioned surrogate ``Φ_ψ(θ, u)`` reproduces the true plant's one-step map; its accuracy is set
by the control range it was trained on, so a surrogate trained on a narrow control range mispredicts large
controls (the honest degradation mechanism); the surrogate-planned controller tracks the true-model
controller closely on the same plant; control is genuinely engaged; and every validation and residency
contract holds. The comparison surfaces the surrogate's cost honestly — it is never presented as free.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from oscillatools.accel import surrogate_kuramoto_mpc as smpc  # noqa: E402
from oscillatools.accel.jax_kuramoto_mpc import RecedingHorizonResult  # noqa: E402
from oscillatools.accel.kuramoto_network_control import (  # noqa: E402
    integrate_controlled_network,
)
from oscillatools.accel.surrogate_kuramoto_mpc import (  # noqa: E402
    SurrogateControlComparison,
    SurrogateStepModel,
    compare_surrogate_control,
    fit_surrogate_step_model,
    surrogate_receding_horizon_control,
    surrogate_step,
)

_N = 4
_OMEGA = np.array([-1.5, -0.5, 0.5, 1.5])
_COUPLING = (np.ones((_N, _N)) - np.eye(_N)) * 0.1
_DT = 0.05
_START = np.random.default_rng(3).uniform(0.0, 2.0 * np.pi, _N)
# A fixed fit configuration shared across the JAX-touching tests so the JIT kernels are reused.
_FIT = dict(control_scale=2.5, iterations=500, sample_size=3000)
_MPC = dict(
    horizon=6,
    n_control_steps=20,
    target_coherence=0.9,
    control_weight=1e-4,
    inner_iterations=50,
    inner_step_size=0.4,
)


def _one_step_rms_error(model: SurrogateStepModel, control_scale: float, seed: int) -> float:
    """The RMS wrapped one-step phase error of the surrogate against the true plant."""
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(200):
        theta = rng.uniform(0.0, 2.0 * np.pi, _N)
        control = rng.normal(0.0, control_scale, _N)
        predicted = surrogate_step(model, theta, control)
        truth = integrate_controlled_network(
            theta, control[None, :], _OMEGA, _COUPLING, _DT
        ).terminal_phases
        wrapped = np.angle(np.exp(1j * (predicted - truth)))
        errors.append(np.sqrt(np.mean(wrapped**2)))
    return float(np.mean(errors))


@pytest.fixture(scope="module")
def well_trained() -> SurrogateStepModel:
    """A surrogate trained over a wide control range (faithful over the applied controls)."""
    return fit_surrogate_step_model(_OMEGA, _COUPLING, _DT, seed=1, **_FIT)


@pytest.fixture(scope="module")
def comparison(well_trained: SurrogateStepModel) -> SurrogateControlComparison:
    """The surrogate vs true-model closed-loop comparison for the well-trained surrogate."""
    return compare_surrogate_control(well_trained, _START, **_MPC)


def test_surrogate_learns_the_plant(well_trained: SurrogateStepModel) -> None:
    """The fitted surrogate reaches a small mean-squared one-step increment error."""
    assert well_trained.training_loss < 1e-2


def test_surrogate_step_matches_the_true_plant(well_trained: SurrogateStepModel) -> None:
    """``Φ_ψ(θ, u)`` reproduces the true controlled one-step map to a small wrapped RMS error."""
    assert _one_step_rms_error(well_trained, control_scale=1.0, seed=9) < 5e-2


def test_narrow_training_degrades_out_of_distribution(
    well_trained: SurrogateStepModel,
) -> None:
    """A surrogate trained on a narrow control range mispredicts large controls far more."""
    narrow = fit_surrogate_step_model(
        _OMEGA, _COUPLING, _DT, seed=1, **{**_FIT, "control_scale": 0.3}
    )
    wide_error = _one_step_rms_error(well_trained, control_scale=2.5, seed=13)
    narrow_error = _one_step_rms_error(narrow, control_scale=2.5, seed=13)
    assert narrow_error > wide_error


def test_surrogate_controller_tracks_the_true_model(
    comparison: SurrogateControlComparison,
) -> None:
    """The surrogate-planned closed loop reaches a terminal coherence close to the true-model one."""
    assert abs(comparison.coherence_gap) < 2e-2
    assert 0.0 <= comparison.surrogate_terminal_coherence <= 1.0
    assert 0.0 <= comparison.true_model_terminal_coherence <= 1.0


def test_control_is_engaged(comparison: SurrogateControlComparison) -> None:
    """Both controllers spend non-trivial control energy — the plan is not idle."""
    assert comparison.surrogate_control_energy > 0.0
    assert comparison.true_model_control_energy > 0.0
    assert comparison.n_control_steps == _MPC["n_control_steps"]


def test_surrogate_receding_horizon_result_is_well_formed(
    well_trained: SurrogateStepModel,
) -> None:
    """The closed-loop result has consistent shapes and a bounded coherence series."""
    result = surrogate_receding_horizon_control(well_trained, _START, **_MPC)
    steps = _MPC["n_control_steps"]
    assert isinstance(result, RecedingHorizonResult)
    assert result.phases.shape == (steps + 1, _N)
    assert result.applied_control.shape == (steps, _N)
    assert result.coherence.shape == (steps + 1,)
    assert result.times.shape == (steps + 1,)
    assert np.all(result.coherence >= 0.0) and np.all(result.coherence <= 1.0)
    assert result.terminal_coherence == pytest.approx(float(result.coherence[-1]))


def test_warm_start_disabled_runs(well_trained: SurrogateStepModel) -> None:
    """The controller runs without warm starting each replan."""
    result = surrogate_receding_horizon_control(
        well_trained, _START, **{**_MPC, "warm_start": False}
    )
    assert result.applied_control.shape == (_MPC["n_control_steps"], _N)


def test_surrogate_step_returns_finite_next_state(well_trained: SurrogateStepModel) -> None:
    """A single surrogate step returns a finite next state of the right shape."""
    nxt = surrogate_step(well_trained, _START, np.zeros(_N))
    assert nxt.shape == (_N,)
    assert np.all(np.isfinite(nxt))


def test_fit_is_deterministic() -> None:
    """The same seed reproduces identical surrogate parameters bit for bit."""
    first = fit_surrogate_step_model(_OMEGA, _COUPLING, _DT, seed=2, **_FIT)
    second = fit_surrogate_step_model(_OMEGA, _COUPLING, _DT, seed=2, **_FIT)
    for (weight_a, bias_a), (weight_b, bias_b) in zip(
        first.parameters, second.parameters, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
        assert np.array_equal(bias_a, bias_b)


def test_model_carries_its_plant_and_diagnostics(well_trained: SurrogateStepModel) -> None:
    """The record exposes the plant it was fitted to and its training summary."""
    assert well_trained.omega.shape == (_N,)
    assert well_trained.coupling.shape == (_N, _N)
    assert well_trained.dt == _DT
    assert well_trained.control_scale == _FIT["control_scale"]
    assert well_trained.training_loss >= 0.0


def test_backend_is_cached() -> None:
    """The JAX backend is built once and reused."""
    assert smpc._load_backend() is smpc._load_backend()


@pytest.mark.parametrize(
    "override",
    [
        {"control_scale": 0.0},
        {"hidden_layers": ()},
        {"hidden_layers": (0,)},
        {"learning_rate": 0.0},
        {"iterations": 0},
        {"sample_size": 0},
    ],
)
def test_fit_rejects_out_of_bound_hyperparameters(override: dict[str, object]) -> None:
    """Every fitting hyperparameter bound is enforced before any work begins."""
    with pytest.raises(ValueError):
        fit_surrogate_step_model(_OMEGA, _COUPLING, _DT, seed=1, **{**_FIT, **override})


@pytest.mark.parametrize(
    ("omega", "coupling", "dt"),
    [
        (np.zeros(1), np.zeros((1, 1)), _DT),
        (np.zeros(_N), np.zeros((_N, _N + 1)), _DT),
        (np.array([0.0, np.nan, 0.0, 0.0]), _COUPLING, _DT),
        (_OMEGA, _COUPLING, 0.0),
    ],
)
def test_fit_rejects_malformed_plants(omega: np.ndarray, coupling: np.ndarray, dt: float) -> None:
    """The plant frequencies, coupling shape, finiteness and step are validated."""
    with pytest.raises(ValueError):
        fit_surrogate_step_model(omega, coupling, dt, **_FIT)


@pytest.mark.parametrize(
    ("phases", "control"),
    [
        (np.zeros(_N + 1), np.zeros(_N)),
        (np.zeros(_N), np.zeros(_N + 1)),
    ],
)
def test_surrogate_step_rejects_malformed_arguments(
    well_trained: SurrogateStepModel, phases: np.ndarray, control: np.ndarray
) -> None:
    """The surrogate step validates the phase and control shapes."""
    with pytest.raises(ValueError):
        surrogate_step(well_trained, phases, control)


@pytest.mark.parametrize(
    "override",
    [
        {"horizon": 0},
        {"n_control_steps": 0},
        {"inner_iterations": 0},
        {"inner_step_size": 0.0},
        {"target_coherence": 1.5},
        {"control_weight": -1e-3},
    ],
)
def test_surrogate_control_rejects_out_of_bound_arguments(
    well_trained: SurrogateStepModel, override: dict[str, object]
) -> None:
    """The receding-horizon schedule and objective bounds are enforced."""
    with pytest.raises(ValueError):
        surrogate_receding_horizon_control(well_trained, _START, **{**_MPC, **override})


def test_surrogate_control_rejects_malformed_state(well_trained: SurrogateStepModel) -> None:
    """The initial phases must match the plant oscillator count."""
    with pytest.raises(ValueError):
        surrogate_receding_horizon_control(well_trained, np.zeros(_N + 1), **_MPC)
