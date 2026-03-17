# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for bridge/snn_adapter.py."""

import numpy as np

from scpn_quantum_control.bridge.snn_adapter import (
    SNNQuantumBridge,
    quantum_measurement_to_current,
    spike_train_to_rotations,
)


def test_spike_to_rotations_all_firing():
    spikes = np.ones((10, 3))
    angles = spike_train_to_rotations(spikes, window=5)
    np.testing.assert_allclose(angles, np.pi, atol=1e-10)


def test_spike_to_rotations_none_firing():
    spikes = np.zeros((10, 3))
    angles = spike_train_to_rotations(spikes, window=5)
    np.testing.assert_allclose(angles, 0.0, atol=1e-10)


def test_spike_to_rotations_1d():
    spikes = np.array([1, 0, 1, 0, 1])
    angles = spike_train_to_rotations(spikes, window=5)
    assert angles.shape == (5,)


def test_measurement_to_current_scaling():
    probs = np.array([0.0, 0.5, 1.0])
    currents = quantum_measurement_to_current(probs, scale=2.0)
    np.testing.assert_allclose(currents, [0.0, 1.0, 2.0])


def test_bridge_init():
    bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
    assert bridge.n_neurons == 2
    assert bridge.n_inputs == 3


def test_bridge_forward_shape():
    bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
    spikes = np.random.default_rng(0).integers(0, 2, (10, 3))
    result = bridge.forward(spikes)
    assert result.shape == (2,)


def test_bridge_forward_bounded():
    bridge = SNNQuantumBridge(n_neurons=2, n_inputs=2, scale=1.0, seed=0)
    spikes = np.random.default_rng(0).integers(0, 2, (5, 2))
    result = bridge.forward(spikes)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_bridge_deterministic():
    b1 = SNNQuantumBridge(n_neurons=2, n_inputs=2, seed=42)
    b2 = SNNQuantumBridge(n_neurons=2, n_inputs=2, seed=42)
    spikes = np.ones((5, 2))
    np.testing.assert_array_equal(b1.forward(spikes), b2.forward(spikes))
