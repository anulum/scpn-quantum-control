# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the adaptive (plastic) Kuramoto model
r"""Tests for :mod:`scpn_quantum_control.accel.kuramoto_adaptive`.

The Hebbian plasticity rule is checked against its closed form and its equilibrium
``K^* = cos(θ_j − θ_i)``; the coupled ``(N + N^2)`` Jacobian is checked against a central finite
difference and against the standalone networked Kuramoto Jacobian on its phase–phase block; and
the co-evolving integrator is checked for the self-organised Hebbian structure that is the
roadmap acceptance criterion — starting from a random coupling, the weights learn the cosine of
the phase differences they carry (``+1`` in-phase, ``-1`` anti-phase).
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.accel.kuramoto_adaptive import (
    AdaptiveTrajectory,
    adaptive_vector_field,
    hebbian_adaptive_jacobian,
    hebbian_coupling_equilibrium,
    hebbian_plasticity_rate,
    integrate_adaptive_kuramoto,
)
from scpn_quantum_control.accel.networked_kuramoto import (
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
)


def _networked_force(phases: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    return networked_kuramoto_force(phases, coupling)


def _hebbian(rate: float):
    def rule(phases: np.ndarray, coupling: np.ndarray) -> np.ndarray:
        return hebbian_plasticity_rate(phases, coupling, plasticity_rate=rate)

    return rule


def _joint_flat(phases, coupling, omega, rate):
    velocity = omega + networked_kuramoto_force(phases, coupling)
    rate_matrix = hebbian_plasticity_rate(phases, coupling, plasticity_rate=rate)
    return np.concatenate([velocity, rate_matrix.ravel()])


# --------------------------------------------------------------------------- equilibrium / rule


def test_equilibrium_is_cosine_of_phase_difference() -> None:
    phases = np.array([0.0, 0.5, 1.3, 2.1])
    equilibrium = hebbian_coupling_equilibrium(phases)
    expected = np.cos(phases[None, :] - phases[:, None])
    assert np.allclose(equilibrium, expected)
    # In-phase self-pairs sit at +1; the diagonal is cos(0) = 1.
    assert np.allclose(np.diag(equilibrium), 1.0)


def test_equilibrium_rejects_non_vector_phases() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hebbian_coupling_equilibrium(np.zeros((2, 2)))


def test_equilibrium_rejects_empty_phases() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hebbian_coupling_equilibrium(np.empty(0))


def test_plasticity_rate_matches_formula() -> None:
    phases = np.array([0.1, 0.9, 1.7])
    coupling = np.array([[0.0, 0.3, -0.2], [0.4, 0.0, 0.1], [-0.5, 0.2, 0.0]])
    rate = 0.7
    expected = rate * (np.cos(phases[None, :] - phases[:, None]) - coupling)
    result = hebbian_plasticity_rate(phases, coupling, plasticity_rate=rate)
    assert np.allclose(result, expected)


def test_plasticity_vanishes_at_equilibrium() -> None:
    phases = np.array([0.2, 1.1, 2.4, 0.8])
    coupling = hebbian_coupling_equilibrium(phases)
    rate_matrix = hebbian_plasticity_rate(phases, coupling, plasticity_rate=1.5)
    assert np.allclose(rate_matrix, 0.0)


def test_plasticity_rejects_negative_rate() -> None:
    with pytest.raises(ValueError, match="plasticity_rate must be non-negative"):
        hebbian_plasticity_rate(np.zeros(3), np.zeros((3, 3)), plasticity_rate=-0.1)


def test_plasticity_rejects_wrong_coupling_shape() -> None:
    with pytest.raises(ValueError, match="coupling must have shape"):
        hebbian_plasticity_rate(np.zeros(3), np.zeros((3, 4)), plasticity_rate=0.5)


def test_plasticity_rejects_non_vector_phases() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hebbian_plasticity_rate(np.zeros((2, 2)), np.zeros((4, 4)), plasticity_rate=0.5)


# --------------------------------------------------------------------------- vector field


def test_vector_field_combines_force_and_plasticity() -> None:
    phases = np.array([0.0, 0.7, 1.5])
    coupling = np.array([[0.0, 0.4, -0.1], [0.2, 0.0, 0.3], [0.1, -0.2, 0.0]])
    omega = np.array([0.5, -0.3, 0.2])
    rate = 0.6
    velocity, rate_matrix = adaptive_vector_field(
        phases, coupling, omega, _networked_force, _hebbian(rate)
    )
    assert np.allclose(velocity, omega + networked_kuramoto_force(phases, coupling))
    assert np.allclose(
        rate_matrix, hebbian_plasticity_rate(phases, coupling, plasticity_rate=rate)
    )


def test_vector_field_rejects_mismatched_omega() -> None:
    with pytest.raises(ValueError, match="omega must have shape"):
        adaptive_vector_field(
            np.zeros(3), np.zeros((3, 3)), np.zeros(2), _networked_force, _hebbian(0.5)
        )


def test_vector_field_rejects_wrong_coupling_shape() -> None:
    with pytest.raises(ValueError, match="coupling must have shape"):
        adaptive_vector_field(
            np.zeros(3), np.zeros((3, 2)), np.zeros(3), _networked_force, _hebbian(0.5)
        )


# --------------------------------------------------------------------------- coupled Jacobian


def test_jacobian_matches_central_difference() -> None:
    rng = np.random.default_rng(7)
    count, rate = 5, 0.3
    phases = rng.uniform(0.0, 2.0 * np.pi, count)
    coupling = rng.normal(0.0, 0.5, (count, count))
    omega = rng.normal(0.0, 0.4, count)
    analytic = hebbian_adaptive_jacobian(phases, coupling, plasticity_rate=rate)

    state = np.concatenate([phases, coupling.ravel()])
    dim = state.size
    numeric = np.empty((dim, dim))
    step = 1e-6
    for column in range(dim):
        forward = state.copy()
        forward[column] += step
        backward = state.copy()
        backward[column] -= step
        f_plus = _joint_flat(forward[:count], forward[count:].reshape(count, count), omega, rate)
        f_minus = _joint_flat(
            backward[:count], backward[count:].reshape(count, count), omega, rate
        )
        numeric[:, column] = (f_plus - f_minus) / (2.0 * step)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_jacobian_phase_block_equals_networked_jacobian() -> None:
    rng = np.random.default_rng(11)
    count = 4
    phases = rng.uniform(0.0, 2.0 * np.pi, count)
    coupling = rng.normal(0.0, 0.6, (count, count))
    analytic = hebbian_adaptive_jacobian(phases, coupling, plasticity_rate=0.5)
    assert np.allclose(analytic[:count, :count], networked_kuramoto_jacobian(phases, coupling))


def test_jacobian_coupling_block_is_minus_rate_identity() -> None:
    count, rate = 3, 0.8
    phases = np.array([0.2, 1.0, 2.5])
    coupling = np.zeros((count, count))
    analytic = hebbian_adaptive_jacobian(phases, coupling, plasticity_rate=rate)
    block = analytic[count:, count:]
    assert np.allclose(block, -rate * np.eye(count * count))


def test_jacobian_has_full_joint_dimension() -> None:
    count = 4
    analytic = hebbian_adaptive_jacobian(
        np.zeros(count), np.zeros((count, count)), plasticity_rate=0.5
    )
    assert analytic.shape == (count + count * count, count + count * count)


def test_jacobian_rejects_negative_rate() -> None:
    with pytest.raises(ValueError, match="plasticity_rate must be non-negative"):
        hebbian_adaptive_jacobian(np.zeros(3), np.zeros((3, 3)), plasticity_rate=-1.0)


def test_jacobian_rejects_wrong_coupling_shape() -> None:
    with pytest.raises(ValueError, match="coupling must have shape"):
        hebbian_adaptive_jacobian(np.zeros(3), np.zeros((2, 2)), plasticity_rate=0.5)


# --------------------------------------------------------------------------- trajectory accessors


def test_trajectory_accessors_return_final_state() -> None:
    times = np.arange(3.0)
    phases = np.arange(6.0).reshape(3, 2)
    couplings = np.arange(12.0).reshape(3, 2, 2)
    traj = AdaptiveTrajectory(times, phases, couplings, np.ones(3))
    assert np.array_equal(traj.terminal_phases, phases[-1])
    assert np.array_equal(traj.terminal_coupling, couplings[-1])


def test_trajectory_hebbian_gap_is_zero_at_equilibrium() -> None:
    phases = np.array([[0.0, 0.6, 1.4]])
    coupling = hebbian_coupling_equilibrium(phases[0])[None, ...]
    traj = AdaptiveTrajectory(np.array([0.0]), phases, coupling, np.ones(1))
    assert traj.hebbian_equilibrium_gap() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- integrator


def test_integrator_validation() -> None:
    omega = np.zeros(3)
    coupling = np.zeros((3, 3))
    with pytest.raises(ValueError, match="omega must have shape"):
        integrate_adaptive_kuramoto(
            np.zeros(3), coupling, np.zeros(2), _networked_force, _hebbian(0.5), dt=0.1, n_steps=1
        )
    with pytest.raises(ValueError, match="coupling must have shape"):
        integrate_adaptive_kuramoto(
            np.zeros(3),
            np.zeros((3, 2)),
            omega,
            _networked_force,
            _hebbian(0.5),
            dt=0.1,
            n_steps=1,
        )
    with pytest.raises(ValueError, match="dt must be positive"):
        integrate_adaptive_kuramoto(
            np.zeros(3), coupling, omega, _networked_force, _hebbian(0.5), dt=0.0, n_steps=1
        )
    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrate_adaptive_kuramoto(
            np.zeros(3), coupling, omega, _networked_force, _hebbian(0.5), dt=0.1, n_steps=0
        )


def test_integrator_shapes_and_initial_sample() -> None:
    count, n_steps = 4, 20
    rng = np.random.default_rng(1)
    phases = rng.uniform(0.0, 2.0 * np.pi, count)
    coupling = rng.normal(0.0, 0.3, (count, count))
    omega = rng.normal(0.0, 0.2, count)
    traj = integrate_adaptive_kuramoto(
        phases, coupling, omega, _networked_force, _hebbian(0.4), dt=0.01, n_steps=n_steps
    )
    assert traj.times.shape == (n_steps + 1,)
    assert traj.phases.shape == (n_steps + 1, count)
    assert traj.couplings.shape == (n_steps + 1, count, count)
    assert traj.order_parameter_series.shape == (n_steps + 1,)
    assert np.array_equal(traj.phases[0], phases)
    assert np.array_equal(traj.couplings[0], coupling)


def test_zero_plasticity_freezes_coupling() -> None:
    # With ε = 0 the weights do not adapt: the coupling matrix is constant through the run.
    count = 4
    rng = np.random.default_rng(2)
    coupling = rng.normal(0.0, 0.5, (count, count))
    traj = integrate_adaptive_kuramoto(
        rng.uniform(0.0, 2.0 * np.pi, count),
        coupling,
        rng.normal(0.0, 0.3, count),
        _networked_force,
        _hebbian(0.0),
        dt=0.02,
        n_steps=50,
    )
    for step in range(traj.couplings.shape[0]):
        assert np.allclose(traj.couplings[step], coupling)


def test_network_learns_hebbian_structure() -> None:
    # Roadmap acceptance: from a random coupling the weights self-organise onto the Hebbian
    # equilibrium K_ij → cos(θ_j − θ_i) bounded in [−1, 1].
    count = 5
    rng = np.random.default_rng(0)
    coupling = rng.normal(0.0, 0.3, (count, count))
    np.fill_diagonal(coupling, 0.0)
    traj = integrate_adaptive_kuramoto(
        rng.uniform(0.0, 2.0 * np.pi, count),
        coupling,
        np.full(count, 0.5),
        _networked_force,
        _hebbian(0.2),
        dt=0.01,
        n_steps=40000,
    )
    assert traj.hebbian_equilibrium_gap() < 1e-6
    assert traj.terminal_coupling.min() >= -1.0 - 1e-9
    assert traj.terminal_coupling.max() <= 1.0 + 1e-9


def test_two_oscillator_phase_relations_set_weight_sign() -> None:
    # A frozen-phase Hebbian relaxation: an in-phase pair learns +1, an anti-phase pair learns −1.
    in_phase = np.array([0.3, 0.3])
    anti_phase = np.array([0.3, 0.3 + np.pi])
    zero_coupling = np.zeros((2, 2))
    in_run = integrate_adaptive_kuramoto(
        in_phase,
        zero_coupling,
        np.zeros(2),
        _networked_force,
        _hebbian(1.0),
        dt=0.01,
        n_steps=2000,
    )
    anti_run = integrate_adaptive_kuramoto(
        anti_phase,
        zero_coupling,
        np.zeros(2),
        _networked_force,
        _hebbian(1.0),
        dt=0.01,
        n_steps=2000,
    )
    assert in_run.terminal_coupling[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert anti_run.terminal_coupling[0, 1] == pytest.approx(-1.0, abs=1e-3)


def test_run_is_deterministic() -> None:
    count = 3
    rng = np.random.default_rng(5)
    phases = rng.uniform(0.0, 2.0 * np.pi, count)
    coupling = rng.normal(0.0, 0.4, (count, count))
    omega = rng.normal(0.0, 0.2, count)
    first = integrate_adaptive_kuramoto(
        phases, coupling, omega, _networked_force, _hebbian(0.5), dt=0.01, n_steps=200
    )
    second = integrate_adaptive_kuramoto(
        phases, coupling, omega, _networked_force, _hebbian(0.5), dt=0.01, n_steps=200
    )
    assert np.array_equal(first.phases, second.phases)
    assert np.array_equal(first.couplings, second.couplings)
