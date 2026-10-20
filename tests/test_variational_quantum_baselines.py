# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — variational baseline conformance
"""Qualify real seeded optimisation, frozen evaluation and state diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import scpn_quantum_control.jax_nqs_baseline_product as product

pytestmark = pytest.mark.framework_imports


def test_variational_quantum_baselines_01() -> None:
    """Bound a real diagonal RBM energy by the independent product-state minimum."""
    spec = product.JAXNQSBaselineSpec.from_arrays(
        np.zeros((2, 2)), np.array((0.7, -0.2)), n_iterations=60, seed=7
    )
    qualified = product.qualify_jax_nqs_baseline(spec)
    assert qualified.training.exact_ground_energy == pytest.approx(-0.9, abs=1e-12)
    assert qualified.training.variational_energy >= -0.9 - 1e-5
    assert qualified.errors.representation_bias == pytest.approx(0.0, abs=1e-12)
    assert qualified.errors.optimisation_excess == pytest.approx(
        qualified.training.variational_energy + 0.9, abs=1e-6
    )
    assert qualified.errors.monte_carlo_standard_error == 0.0
    assert qualified.costs.training_gradient_evaluations == 120
    assert (
        qualified.training.variational_energy < qualified.baseline.comparison.initial_energy - 0.1
    )
    early = product.qualify_jax_nqs_baseline(
        product.JAXNQSBaselineSpec.from_arrays(
            np.zeros((2, 2)), np.array((0.7, -0.2)), n_iterations=1, seed=7
        )
    )
    assert qualified.errors.combined_energy_excess < early.errors.combined_energy_excess
    assert (
        qualified.training.reference_subspace_overlap > early.training.reference_subspace_overlap
    )


def test_variational_quantum_baselines_02() -> None:
    """Repeat the public real-JAX qualification with an identical declared seed."""
    spec = product.JAXNQSBaselineSpec.from_arrays(
        np.array(((0.0, 0.4), (0.4, 0.0))), np.zeros(2), n_iterations=8, seed=19
    )
    first = product.qualify_jax_nqs_baseline(spec)
    second = product.qualify_jax_nqs_baseline(spec)
    assert first.baseline.to_dict() == second.baseline.to_dict()
    assert first.parameters_sha256 == second.parameters_sha256
    assert first.replay_max_energy_difference == 0.0
    assert first.training == second.training
    assert dict(first.parameter_shapes) == {"a": (2,), "b": (4,), "W": (4, 2)}


def test_variational_quantum_baselines_03() -> None:
    """Retain a predeclared evaluation Hamiltonian without updating the fitted state."""
    spec = product.JAXNQSBaselineSpec.from_arrays(
        np.zeros((2, 2)), np.array((0.7, -0.2)), n_iterations=12, seed=3
    )
    coupling = np.zeros((2, 2))
    omega = np.array((-0.5, 0.1))
    evaluated = product.qualify_jax_nqs_baseline(
        spec, held_out={"opposite-field": (coupling, omega)}
    )
    training_only = product.qualify_jax_nqs_baseline(spec)
    assert evaluated.parameters_sha256 == training_only.parameters_sha256
    assert evaluated.baseline.to_dict() == training_only.baseline.to_dict()
    assert len(evaluated.held_out) == 1
    observation = evaluated.held_out[0]
    assert observation.label == "opposite-field"
    assert observation.exact_ground_energy == pytest.approx(-0.6, abs=1e-12)
    assert observation.variational_energy >= -0.6 - 1e-5
    assert observation.hamiltonian_sha256 != evaluated.training.hamiltonian_sha256
    assert observation.numpy_energy == pytest.approx(observation.variational_energy, abs=2e-6)
    np.testing.assert_array_equal(coupling, np.zeros((2, 2)))
    np.testing.assert_array_equal(omega, np.array((-0.5, 0.1)))


def test_variational_quantum_baselines_04() -> None:
    """Expose the positive-amplitude ansatz limit despite an improving real loss."""
    spec = product.JAXNQSBaselineSpec.from_arrays(
        np.array(((0.0, -1.0), (-1.0, 0.0))), np.zeros(2), n_iterations=120, seed=7
    )
    qualified = product.qualify_jax_nqs_baseline(spec)
    assert qualified.baseline.comparison.energy_decreased
    assert qualified.training.exact_ground_energy == pytest.approx(-2.0, abs=1e-12)
    assert qualified.errors.ansatz_energy_infimum == 0.0
    assert qualified.errors.representation_bias == pytest.approx(2.0, abs=1e-12)
    assert qualified.training.reference_subspace_overlap <= 0.5 + 2e-6
    assert qualified.training.numpy_overlap == pytest.approx(
        qualified.training.reference_subspace_overlap, abs=2e-6
    )
    assert not qualified.global_optimum_claimed
    assert not qualified.performance_advantage_claimed
    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.z(0)
    circuit.save_statevector()
    result = AerSimulator(method="matrix_product_state").run(circuit, shots=1).result()
    assert result.success
    reference = np.asarray(result.get_statevector(), dtype=np.complex128)
    hamiltonian = np.array(
        ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0), (0.0, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    )
    assert float(np.real(reference.conj() @ hamiltonian @ reference)) == pytest.approx(-2.0)
    assert result.results[0].metadata["method"] == "matrix_product_state"
