# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Qlayer
"""Tests for qsnn/qlayer.py."""

import numpy as np

from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer


def test_output_shape():
    layer = QuantumDenseLayer(n_neurons=3, n_inputs=2)
    out = layer.forward(np.array([0.5, 0.5]))
    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({0, 1})


def test_zero_input_no_spikes():
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, spike_threshold=0.5)
    out = layer.forward(np.array([0.0, 0.0]))
    assert np.all(out == 0)


def test_qubit_count():
    layer = QuantumDenseLayer(n_neurons=3, n_inputs=2)
    assert layer.n_qubits == 5


def test_get_weights_shape():
    layer = QuantumDenseLayer(n_neurons=3, n_inputs=4)
    W = layer.get_weights()
    assert W.shape == (3, 4)


def test_custom_weights():
    W = np.array([[0.9, 0.9], [0.1, 0.1]])
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, weights=W)
    np.testing.assert_allclose(layer.get_weights(), W, atol=1e-10)


def test_low_threshold_fires():
    W = np.array([[0.9, 0.9]])
    layer = QuantumDenseLayer(n_neurons=1, n_inputs=2, weights=W, spike_threshold=0.01)
    out = layer.forward(np.array([1.0, 1.0]))
    assert out[0] == 1


def test_random_weight_bounds():
    layer = QuantumDenseLayer(n_neurons=4, n_inputs=3)
    W = layer.get_weights()
    assert W.shape == (4, 3)
    assert np.all((W >= 0.0) & (W <= 1.0))


def test_default_weights_seeded_deterministic():
    layer1 = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
    layer2 = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
    np.testing.assert_array_equal(layer1.get_weights(), layer2.get_weights())


def test_different_seeds_different_weights():
    layer1 = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=1)
    layer2 = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=999)
    assert not np.array_equal(layer1.get_weights(), layer2.get_weights())


def test_forward_binary_output():
    """Forward output must be strictly 0 or 1."""
    layer = QuantumDenseLayer(n_neurons=3, n_inputs=2, seed=42)
    for _ in range(5):
        out = layer.forward(np.array([0.5, 0.5]))
        assert set(np.unique(out)).issubset({0, 1})


def test_n_qubits_formula():
    """n_qubits = n_neurons + n_inputs."""
    for n, m in [(2, 3), (4, 2), (1, 5)]:
        layer = QuantumDenseLayer(n_neurons=n, n_inputs=m)
        assert layer.n_qubits == n + m


def test_high_weight_high_fire_rate():
    """Neurons with weight=1 and input=1 should fire more than weight=0."""
    layer_high = QuantumDenseLayer(
        n_neurons=1, n_inputs=1, weights=np.array([[1.0]]), spike_threshold=0.3
    )
    layer_low = QuantumDenseLayer(
        n_neurons=1, n_inputs=1, weights=np.array([[0.01]]), spike_threshold=0.3
    )
    fires_high = sum(layer_high.forward(np.array([1.0]))[0] for _ in range(20))
    fires_low = sum(layer_low.forward(np.array([1.0]))[0] for _ in range(20))
    assert fires_high >= fires_low
