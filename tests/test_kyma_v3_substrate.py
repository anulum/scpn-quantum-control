# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KYMA v3 staged oscillator substrate
"""Tests for the KYMA v3 staged gated-coupling oscillator substrate."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp
import numpy as np

from scpn_quantum_control.benchmarks.kyma_v3 import substrate, task


def test_param_count_matches_the_gate_tensors() -> None:
    """Check that param count matches the gate tensors."""
    gates = substrate.init_gates(0)
    assert substrate.substrate_param_count() == 108
    assert sum(int(np.prod(value.shape)) for value in gates.values()) == 108


def test_init_is_seeded() -> None:
    """Check that init is seeded."""
    first, again, other = substrate.init_gates(3), substrate.init_gates(3), substrate.init_gates(4)
    for name in first:
        assert np.array_equal(np.asarray(first[name]), np.asarray(again[name]))
    assert not np.array_equal(np.asarray(first["coupling"]), np.asarray(other["coupling"]))


def test_phase_to_label_rounds_to_the_nearest_lattice_point_with_wrap() -> None:
    """Check that phase to label rounds to the nearest lattice point with wrap."""
    step = substrate.PHASE_STEP
    phases = jnp.asarray([0.0, step - 0.1, 2 * step + 0.7, -0.2, 4 * step - 0.3, -step])
    assert np.asarray(substrate.phase_to_label(phases)).tolist() == [0, 1, 2, 0, 0, 3]


def test_hand_gates_solve_every_training_and_test_item() -> None:
    """Check that hand gates solve every training and test item."""
    dataset = task.build_dataset()
    gates = substrate.hand_gates()
    everything = np.ones(dataset.size, dtype=bool)
    assert np.array_equal(substrate.predict(gates, dataset, everything), dataset.label)


def test_hand_gates_land_close_to_the_lattice() -> None:
    """Check that hand gates land close to the lattice."""
    states = jnp.asarray([[0, 1, 2], [3, 3, 1]])
    phases = substrate.final_phases(
        substrate.hand_gates(), states, jnp.asarray([0, 0]), jnp.asarray([1, 1])
    )
    expected = np.array([[3, 2, 0], [0, 1, 3]]) * substrate.PHASE_STEP
    error = np.angle(np.exp(1j * (np.asarray(phases) - expected)))
    assert np.max(np.abs(error)) < 0.05


def test_single_operation_skips_the_second_stage() -> None:
    """Check that single operation skips the second stage."""
    states = jnp.asarray([[1, 2, 3]])
    single = substrate.final_phases(
        substrate.hand_gates(), states, jnp.asarray([2]), jnp.asarray([task.NO_OPERATION])
    )
    labels = np.asarray(substrate.phase_to_label(single))[0].tolist()
    assert labels == [1, 3, 3]


def test_weak_coupling_is_not_realisable() -> None:
    """Check that weak coupling is not realisable."""
    dataset = task.build_dataset()
    gates = substrate.hand_gates(coupling=0.05)
    everything = np.ones(dataset.size, dtype=bool)
    assert np.mean(substrate.predict(gates, dataset, everything) == dataset.label) < 0.9


def test_training_updates_the_gates_and_predicts_valid_labels() -> None:
    """Check that training updates the gates and predicts valid labels."""
    dataset = task.build_dataset()
    trained = substrate.train(dataset, seed=0, epochs=3, learning_rate=0.05)
    initial = substrate.init_gates(0)
    assert set(trained) == set(initial)
    assert not np.allclose(np.asarray(trained["coupling"]), np.asarray(initial["coupling"]))
    predicted = substrate.predict(trained, dataset, dataset.is_test)
    assert predicted.shape == (64,)
    assert set(np.unique(predicted)) <= {0, 1, 2, 3}
