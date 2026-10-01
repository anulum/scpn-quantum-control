# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto convention boundary tests
"""Qualify scientific phase bindings through original public solver owners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as qc
from oscillatools.accel.order_parameter_observables import order_parameter


def _binding(
    theta: NDArray[np.float64], omega: NDArray[np.float64], coupling: NDArray[np.float64]
) -> qc.ScientificProblemParameters:
    """Bind the exact physical phase convention without a substitute numerical rule."""
    n = len(theta)
    return qc.ScientificProblemParameters(
        qc.build_kuramoto_problem(coupling, omega),
        qc.ScientificDesign(
            model="phase_kuramoto",
            normalisation="pairwise_sum",
            coordinate_space="logical",
            units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
            topology=tuple(
                (i, j) for i in range(n) for j in range(i + 1, n) if coupling[i, j] != 0
            ),
            initial_state=theta,
            observable="phase_order_parameter",
            observable_weights=np.ones(n),
            objective=qc.DesignObjective("simulate", None, "1"),
        ),
    )


def test_original_phase_factory_runs_without_importing_studio() -> None:
    """The public numerical boundary consumes typed declarations below the UI layer."""
    program = """
import json, sys
import numpy as np
import scpn_quantum_control as qc
problem = qc.build_kuramoto_problem(np.zeros((1, 1)), np.array([0.4]))
design = qc.ScientificDesign(
    model='phase_kuramoto', normalisation='pairwise_sum', coordinate_space='logical',
    units=qc.ScientificUnits('s', 'rad/s', 'rad/s', 'rad', '1'), topology=(),
    initial_state=np.array([0.2]), observable='phase_order_parameter',
    observable_weights=np.ones(1), objective=qc.DesignObjective('simulate', None, '1'))
system = qc.build_scientific_phase_system(problem, design, dt=0.25)
path = system.trajectory(4)
np.testing.assert_allclose(path[:, 0], 0.2 + 0.4 * np.arange(5) / 4)
forbidden = [name for name in sys.modules if name.startswith(
    ('scpn_quantum_control.studio_workspace', 'scpn_quantum_control.scientific_problem_parameters'))]
print(json.dumps({'forbidden': forbidden, 'shape': list(path.shape)}))
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == {"forbidden": [], "shape": [5, 1]}


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
@pytest.mark.parametrize("model", ["finite_networked", "finite_mean_field"])
def test_kuramoto_model_conventions_01(model: str, scheme: str) -> None:
    """Uncoupled original trajectories satisfy theta(t)=theta0+omega*t."""
    theta = np.array([0.1, -0.4, 0.7])
    omega = np.array([0.2, -0.5, 0.8])
    parameters = _binding(theta, omega, np.zeros((3, 3)))
    before = parameters.identity
    system = qc.build_scientific_phase_system(
        parameters.problem, parameters.design, dt=1 / 32, model=model, scheme=scheme
    )
    trajectory = system.trajectory(32)
    times = np.arange(33) / 32
    np.testing.assert_allclose(trajectory, theta + times[:, None] * omega, atol=1e-12, rtol=0)
    assert system.current_time == 1.0
    assert parameters.identity == before


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
def test_kuramoto_model_conventions_02(scheme: str) -> None:
    """Uniform phase shifts preserve relative phases and the original observable."""
    theta = np.array([0.1, -0.4, 0.7])
    omega = np.array([0.2, -0.5, 0.8])
    coupling = np.full((3, 3), 0.3)
    first_parameters = _binding(theta, omega, coupling)
    shifted_parameters = _binding(theta + 0.63, omega, coupling)
    first = qc.build_scientific_phase_system(
        first_parameters.problem, first_parameters.design, dt=1 / 32, scheme=scheme
    )
    shifted = qc.build_scientific_phase_system(
        shifted_parameters.problem, shifted_parameters.design, dt=1 / 32, scheme=scheme
    )
    original, moved = first.trajectory(32), shifted.trajectory(32)
    np.testing.assert_allclose(moved, original + 0.63, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        [order_parameter(row) for row in moved],
        [order_parameter(row) for row in original],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
def test_kuramoto_model_conventions_03(scheme: str) -> None:
    """A common frequency shift adds the expected rotating-frame phase c*t."""
    theta = np.array([0.1, -0.4, 0.7])
    omega = np.array([0.2, -0.5, 0.8])
    coupling = np.full((3, 3), 0.3)
    first_parameters = _binding(theta, omega, coupling)
    shifted_parameters = _binding(theta, omega + 0.4, coupling)
    first = qc.build_scientific_phase_system(
        first_parameters.problem, first_parameters.design, dt=1 / 32, scheme=scheme
    )
    shifted = qc.build_scientific_phase_system(
        shifted_parameters.problem, shifted_parameters.design, dt=1 / 32, scheme=scheme
    )
    original, moved = first.trajectory(32), shifted.trajectory(32)
    np.testing.assert_allclose(
        moved, original + (0.4 * np.arange(33) / 32)[:, None], atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
def test_kuramoto_model_conventions_04(scheme: str) -> None:
    """Equal-frequency symmetric two-oscillator equilibrium has zero relative drift."""
    parameters = _binding(np.array([0.7, 0.7]), np.array([1.5, 1.5]), np.ones((2, 2)))
    system = qc.build_scientific_phase_system(
        parameters.problem, parameters.design, dt=1 / 32, scheme=scheme
    )
    np.testing.assert_allclose(system.rule_value(), [1.5, 1.5], atol=1e-12, rtol=0)
    path = system.trajectory(32)
    np.testing.assert_allclose(path[:, 0] - path[:, 1], 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(path[:, 0], 0.7 + 1.5 * np.arange(33) / 32, atol=1e-12, rtol=0)


@pytest.mark.parametrize("model", ["finite_networked", "finite_mean_field"])
@pytest.mark.parametrize("normalisation", ["pairwise_sum", "population_mean"])
@pytest.mark.parametrize("weight", [-0.6, 0.6])
def test_explicit_normalisation_retains_signed_original_force(
    model: str, normalisation: Literal["pairwise_sum", "population_mean"], weight: float
) -> None:
    """The original force agrees with an independent two-phase equation and K/N."""
    theta = np.array([0.0, 0.4])
    parameters = _binding(theta, np.array([0.2, -0.3]), np.full((2, 2), weight))
    parameters = qc.ScientificProblemParameters(
        parameters.problem, replace(parameters.design, normalisation=normalisation)
    )
    system = qc.build_scientific_phase_system(
        parameters.problem, parameters.design, dt=1 / 32, model=model
    )
    coefficient = weight / (2 if normalisation == "population_mean" else 1)
    expected = np.array([0.2 + coefficient * np.sin(0.4), -0.3 - coefficient * np.sin(0.4)])
    np.testing.assert_allclose(system.rule_value(), expected, atol=1e-12, rtol=0)
    assert parameters.design.normalisation == normalisation


def test_attractive_coupling_matches_two_oscillator_closed_form() -> None:
    """Positive pairwise coupling contracts phase separation with the correct sign."""
    initial_gap, coefficient = 0.7, 0.8
    parameters = _binding(
        np.array([-initial_gap / 2, initial_gap / 2]),
        np.full(2, 0.5),
        np.full((2, 2), coefficient),
    )
    system = qc.build_scientific_phase_system(parameters.problem, parameters.design, dt=1 / 64)
    trajectory = system.trajectory(64)
    times = np.arange(65) / 64
    gap = 2 * np.arctan(np.tan(initial_gap / 2) * np.exp(-2 * coefficient * times))
    np.testing.assert_allclose(trajectory[:, 1] - trajectory[:, 0], gap, atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.mean(trajectory, axis=1), 0.5 * times, atol=1e-12, rtol=0)


@pytest.mark.parametrize("model", ["finite_networked", "finite_mean_field"])
def test_single_oscillator_has_no_self_coupling(model: str) -> None:
    """N=1 remains a supported finite population rather than a spurious topology error."""
    parameters = _binding(np.array([0.2]), np.array([0.4]), np.array([[0.7]]))
    system = qc.build_scientific_phase_system(
        parameters.problem, parameters.design, dt=0.25, model=model
    )
    np.testing.assert_allclose(system.trajectory(4)[:, 0], 0.2 + 0.4 * np.arange(5) / 4)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf, -np.inf, True, "0.1", None])
def test_invalid_step_refuses_and_valid_recovery_retains_inputs(dt: object) -> None:
    """Malformed time steps cannot submit, modify or invalidate an existing binding."""
    parameters = _binding(np.array([0.1, -0.2]), np.array([0.3, 0.4]), np.zeros((2, 2)))
    before = parameters.to_dict()
    with pytest.raises(ValueError, match="finite positive"):
        qc.build_scientific_phase_system(parameters.problem, parameters.design, dt=cast(float, dt))
    assert parameters.to_dict() == before
    recovered = qc.build_scientific_phase_system(parameters.problem, parameters.design, dt=0.1)
    np.testing.assert_allclose(recovered.trajectory(1)[-1], [0.13, -0.16])


@pytest.mark.parametrize(
    ("model", "solver"),
    [
        ("unknown", "rk4"),
        ("finite_networked", "adaptive"),
        ("finite_mean_field", "euler_maruyama"),
        ("continuum_ott_antonsen", "euler"),
        ("quantum_xy", "rk4"),
        (None, "rk4"),
        ("finite_networked", []),
    ],
)
def test_unsupported_model_solver_pairs_refuse_without_fallback(
    model: object, solver: object
) -> None:
    """Unsupported pairs cannot reuse a nearby model or silently switch methods."""
    parameters = _binding(np.array([0.1]), np.array([0.3]), np.zeros((1, 1)))
    before = parameters.identity
    with pytest.raises(ValueError):
        qc.kuramoto_model_convention(cast(str, model), cast(str, solver))
    with pytest.raises(ValueError):
        qc.build_scientific_phase_system(
            parameters.problem,
            parameters.design,
            dt=0.1,
            model=cast(str, model),
            scheme=cast(str, solver),
        )
    assert parameters.identity == before


@pytest.mark.parametrize(
    ("model", "scheme"),
    [
        ("finite_delayed_networked", "rk4"),
        ("finite_noisy_networked", "euler_maruyama"),
        ("continuum_ott_antonsen", "rk4"),
        ("quantum_xy", "suzuki_trotter"),
        ("finite_hypergraph", "force_only"),
    ],
)
def test_distinct_supported_owners_do_not_enter_the_plain_phase_factory(
    model: str, scheme: str
) -> None:
    """Inventory support is separate from admissible inputs of the instantaneous factory."""
    parameters = _binding(np.array([0.1]), np.array([0.3]), np.zeros((1, 1)))
    assert qc.kuramoto_model_convention(model, scheme).model == model
    with pytest.raises(ValueError, match="instantaneous finite"):
        qc.build_scientific_phase_system(
            parameters.problem, parameters.design, dt=0.1, model=model, scheme=scheme
        )


def test_history_binding_refuses_without_discarding_history() -> None:
    """The plain solver refuses an explicit history and preserves its exact companion."""
    parameters = _binding(np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.zeros((2, 2)))
    design = replace(
        parameters.design,
        history_times=np.array([-0.1, 0.0]),
        history_states=np.array([[0.07, 0.16], [0.1, 0.2]]),
    )
    historical = qc.ScientificProblemParameters(parameters.problem, design)
    before = historical.to_dict()
    with pytest.raises(ValueError, match="does not consume history"):
        qc.build_scientific_phase_system(historical.problem, historical.design, dt=0.1)
    assert historical.to_dict() == before
    np.testing.assert_allclose(
        qc.build_scientific_phase_system(parameters.problem, parameters.design, dt=0.1).trajectory(
            1
        )[-1],
        [0.13, 0.24],
    )


def test_quantum_binding_cannot_be_interpreted_as_phase_angles() -> None:
    """Dimensionally explicit spin amplitudes cannot silently become a phase state."""
    phase = _binding(np.array([0.1]), np.array([0.3]), np.zeros((1, 1)))
    quantum = qc.ScientificProblemParameters(
        phase.problem,
        replace(
            phase.design,
            model="quantum_xy",
            units=qc.ScientificUnits("s", "rad/s", "rad/s", "1", "1"),
            initial_state=np.array([1.0, 0.0]),
            observable="spin_z",
        ),
    )
    before = quantum.identity
    with pytest.raises(ValueError, match="phase_kuramoto"):
        qc.build_scientific_phase_system(quantum.problem, quantum.design, dt=0.1)
    assert quantum.identity == before


def test_mean_field_refuses_heterogeneous_coupling_and_scalar_overflow() -> None:
    """The scalar owner never substitutes an average for heterogeneous graph weights."""
    coupling = np.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]])
    parameters = _binding(np.array([0.1, 0.2, 0.3]), np.ones(3), coupling)
    before = parameters.identity
    with pytest.raises(ValueError, match="uniform off-diagonal"):
        qc.build_scientific_phase_system(
            parameters.problem, parameters.design, dt=0.1, model="finite_mean_field"
        )
    assert parameters.identity == before
    assert qc.build_scientific_phase_system(
        parameters.problem, parameters.design, dt=0.1
    ).trajectory(1).shape == (2, 3)
    huge = _binding(np.array([0.1, 0.2, 0.3]), np.zeros(3), np.full((3, 3), 1e308))
    with pytest.raises(ValueError, match="scalar coupling must remain finite"):
        qc.build_scientific_phase_system(
            huge.problem, huge.design, dt=0.1, model="finite_mean_field"
        )


def test_factory_requires_a_real_scientific_binding_and_copies_state() -> None:
    """Refusal and simulation leave both the source companion and input arrays intact."""
    theta = np.array([0.1, 0.2])
    omega = np.array([0.3, 0.4])
    coupling = np.zeros((2, 2))
    parameters = _binding(theta, omega, coupling)
    with pytest.raises(ValueError, match="KuramotoProblem"):
        qc.build_scientific_phase_system(
            cast(qc.KuramotoProblem, object()), parameters.design, dt=0.1
        )
    before = parameters.to_dict()
    system = qc.build_scientific_phase_system(parameters.problem, parameters.design, dt=0.1)
    theta[:] = 5
    omega[:] = 8
    coupling[:] = 9
    system.set_state(np.array([0.8, 0.9]))
    assert parameters.to_dict() == before
    np.testing.assert_allclose(system.trajectory(1)[-1], [0.83, 0.94])


@pytest.mark.parametrize("damage", ["state_count", "topology", "zero_weights", "units", "object"])
def test_factory_validates_direct_design_before_evolution_and_recovers(damage: str) -> None:
    """A direct unbound declaration receives the original full scientific validation."""
    parameters = _binding(np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.ones((2, 2)))
    before = parameters.to_dict()
    problem, design = parameters.problem, parameters.design
    if damage == "state_count":
        invalid = replace(design, initial_state=np.array([0.1]))
    elif damage == "topology":
        invalid = replace(design, topology=())
    elif damage == "zero_weights":
        invalid = replace(design, observable_weights=np.zeros(2))
    elif damage == "units":
        invalid = replace(design, units=replace(design.units, frequency="Hz"))
    else:
        invalid = cast(qc.ScientificDesign, object())
    with pytest.raises(ValueError):
        qc.build_scientific_phase_system(problem, invalid, dt=0.1)
    assert parameters.to_dict() == before
    recovered = qc.build_scientific_phase_system(problem, design, dt=0.1)
    assert recovered.trajectory(1).shape == (2, 2)
    assert parameters.to_dict() == before


def test_source_convention_matrix_has_distinct_immutable_model_identities() -> None:
    """Public inventory never labels continuum, finite reductions or spin states alike."""
    rows = qc.kuramoto_convention_matrix()
    assert len({(row.model, row.solver) for row in rows}) == len(rows)
    for row in rows:
        assert qc.kuramoto_model_convention(row.model, row.solver) is row
        assert isinstance(row.backend_owners, tuple) and isinstance(row.assumptions, tuple)
    finite = qc.kuramoto_model_convention("finite_mean_field", "rk4")
    reduced = qc.kuramoto_model_convention("finite_watanabe_strogatz", "rk4")
    continuum = qc.kuramoto_model_convention("continuum_ott_antonsen", "rk4")
    quantum = qc.kuramoto_model_convention("quantum_xy", "suzuki_trotter")
    assert len({row.interpretation for row in (finite, reduced, continuum, quantum)}) == 4
    with pytest.raises(FrozenInstanceError):
        finite.__setattr__("model", "continuum_ott_antonsen")
    assert "K/N" in finite.normalisation
    assert "positive K and Delta" in " ".join(continuum.assumptions)
    assert "not the classical phase" in " ".join(quantum.assumptions)
    assert (
        qc.kuramoto_model_convention("finite_delayed_mean_field", "rk4").sensitivity_owner is None
    )
    assert (
        qc.kuramoto_model_convention("finite_noisy_mean_field", "euler_maruyama").sensitivity_owner
        is None
    )


def test_delayed_owner_retains_grid_history_and_uncoupled_linear_solution() -> None:
    """The declared delayed owner consumes full history and reports actual evolved times."""
    from oscillatools.accel.kuramoto_delayed import (
        delayed_networked_force,
        integrate_delayed_kuramoto,
    )

    theta, omega, coupling = np.array([0.1, -0.2]), np.array([0.3, 0.4]), np.zeros((2, 2))
    dt, delay = 1 / 32, 1 / 16
    history = theta + np.arange(-2, 1)[:, None] * dt * omega
    before = history.copy()

    def force(current: NDArray[np.float64], lagged: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the original network force with separately supplied delayed neighbours."""
        return delayed_networked_force(current, lagged, coupling)

    row = qc.kuramoto_model_convention("finite_delayed_networked", "rk4")
    assert (
        row.evolution_owner
        == integrate_delayed_kuramoto.__module__ + "." + integrate_delayed_kuramoto.__name__
    )
    result = integrate_delayed_kuramoto(history, omega, force, delay=delay, dt=dt, n_steps=32)
    np.testing.assert_array_equal(result.times, np.arange(33) * dt)
    np.testing.assert_allclose(
        result.phases, theta + result.times[:, None] * omega, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(history, before)
    for bad_history, bad_delay in [(history[:-1], delay), (history, 0.0), (history, 1.5 * dt)]:
        with pytest.raises(ValueError):
            integrate_delayed_kuramoto(
                bad_history, omega, force, delay=bad_delay, dt=dt, n_steps=2
            )
    np.testing.assert_array_equal(history, before)
    recovered = integrate_delayed_kuramoto(history, omega, force, delay=delay, dt=dt, n_steps=1)
    np.testing.assert_allclose(recovered.phases[-1], theta + omega * dt)


def test_additive_noise_owner_matches_supplied_increment_and_seeded_recovery() -> None:
    """Euler–Maruyama convention has the original sqrt(2Ddt) increment and seeded path."""
    from oscillatools.accel.kuramoto_noisy import integrate_noisy_kuramoto, noisy_kuramoto_step
    from oscillatools.accel.networked_kuramoto import networked_kuramoto_force

    theta, omega, coupling = np.array([0.1, -0.2]), np.array([0.3, 0.4]), np.zeros((2, 2))
    increments = np.array([-0.4, 0.7])

    def force(current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reuse the actual force owner; randomness remains with the original stepper."""
        return networked_kuramoto_force(current, coupling)

    step = noisy_kuramoto_step(theta, omega, force, 0.2, 0.1, increments)
    np.testing.assert_allclose(
        step, theta + omega * 0.1 + np.sqrt(2 * 0.2 * 0.1) * increments, atol=1e-12, rtol=0
    )
    before = theta.copy()
    with pytest.raises(ValueError):
        noisy_kuramoto_step(theta, omega, force, -0.1, 0.1, increments)
    np.testing.assert_array_equal(theta, before)
    first = integrate_noisy_kuramoto(
        theta, omega, force, diffusion=0.2, dt=1 / 32, n_steps=32, seed=7
    )
    repeated = integrate_noisy_kuramoto(
        theta, omega, force, diffusion=0.2, dt=1 / 32, n_steps=32, seed=7
    )
    np.testing.assert_array_equal(first.terminal_phases, repeated.terminal_phases)
    np.testing.assert_array_equal(first.order_parameter_series, repeated.order_parameter_series)
    assert first.order_parameter_series.shape == (32,)
    assert "after steps" in " ".join(
        qc.kuramoto_model_convention("finite_noisy_networked", "euler_maruyama").assumptions
    )
    noiseless = integrate_noisy_kuramoto(
        theta, omega, force, diffusion=0.0, dt=1 / 32, n_steps=32, seed=7
    )
    np.testing.assert_allclose(noiseless.terminal_phases, theta + omega, atol=1e-12, rtol=0)


def test_finite_reduction_preserves_identical_frequency_circle_flow() -> None:
    """Original WS reconstruction matches independent uncoupled complex phase evolution."""
    from oscillatools.accel.kuramoto_watanabe_strogatz import integrate_watanabe_strogatz

    theta = np.array([-0.4, 0.2, 0.9])
    result = integrate_watanabe_strogatz(theta, omega=0.7, coupling=0.0, dt=1 / 64, n_steps=64)
    expected = np.exp(1j * (theta + result.times[:, None] * 0.7))
    np.testing.assert_allclose(np.exp(1j * result.phases), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.constants, np.exp(1j * theta), atol=1e-12, rtol=0)
    row = qc.kuramoto_model_convention("finite_watanabe_strogatz", "rk4")
    assert row.interpretation == "exact_finite_reduction"
    assert "identical" in row.topology
    with pytest.raises(ValueError):
        integrate_watanabe_strogatz(theta, omega=0.7, coupling=0.0, dt=0.0, n_steps=2)


def test_continuum_reduction_has_its_own_steady_state_and_domain() -> None:
    """OA flow preserves its analytic Lorentzian equilibrium without finite-model relabelling."""
    from oscillatools.accel.kuramoto_ott_antonsen import (
        ott_antonsen_field,
        ott_antonsen_trajectory,
    )

    coupling, half_width = 3.0, 0.5
    equilibrium = complex(np.sqrt(1 - 2 * half_width / coupling))
    assert abs(ott_antonsen_field(equilibrium, coupling, half_width)) < 1e-12
    trajectory = ott_antonsen_trajectory(equilibrium, coupling, half_width, 1 / 32, 32)
    np.testing.assert_allclose(trajectory, equilibrium, atol=1e-12, rtol=0)
    assert (
        qc.kuramoto_model_convention("continuum_ott_antonsen", "rk4").interpretation
        == "continuum_reduction"
    )
    with pytest.raises(ValueError):
        ott_antonsen_trajectory(equilibrium, 0.0, half_width, 1 / 32, 32)
    assert trajectory.shape == (33,)


def test_higher_order_force_identities_distinguish_simplex_from_harmonic_fields() -> None:
    """Actual source kernels match independent complex-order expressions, with distinct models."""
    from oscillatools.accel.daido_mean_field import daido_mean_field_force
    from oscillatools.accel.kuramoto_simplex_mean_field import simplex_mean_field_force
    from oscillatools.accel.triadic_mean_field import triadic_mean_field_force

    theta, coupling = np.array([-0.4, 0.2, 0.9]), 0.7
    z = np.mean(np.exp(1j * theta))
    z2 = np.mean(np.exp(2j * theta))
    simplex = coupling * np.imag(z**2 * np.exp(-2j * theta))
    harmonic = coupling * np.imag(z2 * np.exp(-2j * theta))
    np.testing.assert_allclose(
        simplex_mean_field_force(theta, coupling, 2), simplex, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        triadic_mean_field_force(theta, coupling), simplex, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        daido_mean_field_force(theta, coupling, 2), harmonic, atol=1e-12, rtol=0
    )
    assert not np.allclose(simplex, harmonic)
    for name in (
        "finite_simplex_mean_field",
        "finite_triadic_mean_field",
        "finite_daido_mean_field",
    ):
        row = qc.kuramoto_model_convention(name, "force_only")
        assert row.evolution_owner is None and row.interpretation == "force_operator"
