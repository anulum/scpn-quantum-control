"""Tests for control/qpetri.py."""

import numpy as np
import pytest

from scpn_quantum_control.control.qpetri import QuantumPetriNet


@pytest.fixture
def simple_net():
    W_in = np.array([[0.8, 0.0], [0.0, 0.6]])  # 2 transitions, 2 places
    W_out = np.array([[0.0, 0.7], [0.5, 0.0]])  # 2 places, 2 transitions
    thresholds = np.array([0.5, 0.5])
    return QuantumPetriNet(2, 2, W_in, W_out, thresholds)


def test_encode_marking(simple_net):
    qc = simple_net.encode_marking(np.array([0.5, 0.3]))
    assert qc.num_qubits == 2


def test_step_returns_marking(simple_net):
    marking = np.array([0.7, 0.3])
    new_marking = simple_net.step(marking)
    assert new_marking.shape == (2,)
    assert np.all(new_marking >= 0)
    assert np.all(new_marking <= 1.0)


def test_from_matrices():
    W_in = np.array([[0.5, 0.3]])
    W_out = np.array([[0.4], [0.6]])
    thresholds = np.array([0.5])
    net = QuantumPetriNet.from_matrices(W_in, W_out, thresholds)
    assert net.n_places == 2
    assert net.n_transitions == 1


def test_zero_marking():
    W_in = np.array([[0.5]])
    W_out = np.array([[0.5]])
    thresholds = np.array([0.5])
    net = QuantumPetriNet(1, 1, W_in, W_out, thresholds)
    new_m = net.step(np.array([0.0]))
    assert new_m.shape == (1,)


def test_controlled_output_depends_on_input():
    """Output place token density should differ depending on input place state.

    With input place at |0⟩ vs |1⟩, the CRy-controlled output should differ.
    """
    W_in = np.array([[0.3, 0.0]])  # light consumption from place 0
    W_out = np.array([[0.0], [0.6]])  # output to place 1
    thresholds = np.array([1.0])
    net = QuantumPetriNet.from_matrices(W_in, W_out, thresholds)

    out_a = net.step(np.array([0.0, 0.0]))
    out_b = net.step(np.array([0.8, 0.0]))
    # The two should give different place-1 markings (gating effect)
    assert abs(out_a[1] - out_b[1]) > 1e-4


def test_multiple_transitions_preserve_bounds():
    """Token densities stay in [0, 1] after multiple steps."""
    W_in = np.array([[0.5, 0.0], [0.0, 0.5]])
    W_out = np.array([[0.0, 0.4], [0.4, 0.0]])
    thresholds = np.array([0.8, 0.8])
    net = QuantumPetriNet.from_matrices(W_in, W_out, thresholds)
    marking = np.array([0.6, 0.4])
    for _ in range(5):
        marking = net.step(marking)
        assert np.all(marking >= 0) and np.all(marking <= 1.0)
