"""Tests for qsnn/qlayer.py."""
import numpy as np
import pytest

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
