"""Tests for qsnn/qstdp.py."""

import numpy as np

from scpn_quantum_control.qsnn.qstdp import QuantumSTDP
from scpn_quantum_control.qsnn.qsynapse import QuantumSynapse


def test_no_update_without_pre_spike():
    syn = QuantumSynapse(0.5)
    stdp = QuantumSTDP(learning_rate=0.1)
    original = syn.weight
    stdp.update(syn, pre_measured=0, post_measured=1)
    assert syn.weight == original


def test_update_changes_weight():
    syn = QuantumSynapse(0.5)
    stdp = QuantumSTDP(learning_rate=0.1)
    original = syn.weight
    stdp.update(syn, pre_measured=1, post_measured=1)
    assert syn.weight != original


def test_gradient_is_finite():
    stdp = QuantumSTDP()
    exp = stdp._expectation_z(np.pi / 4)
    assert np.isfinite(exp)


def test_weight_stays_bounded():
    syn = QuantumSynapse(0.5, w_min=0.0, w_max=1.0)
    stdp = QuantumSTDP(learning_rate=0.5)
    for _ in range(20):
        stdp.update(syn, pre_measured=1, post_measured=1)
    assert 0.0 <= syn.weight <= 1.0


def test_stdp_ltp_increases_weight():
    """Hebbian LTP: pre=1 + post=1 should increase synapse weight."""
    syn = QuantumSynapse(0.3, w_min=0.0, w_max=1.0)
    stdp = QuantumSTDP(learning_rate=0.1)
    w_before = syn.weight
    stdp.update(syn, pre_measured=1, post_measured=1)
    assert syn.weight > w_before


def test_stdp_ltd_decreases_weight():
    """Hebbian LTD: pre=1 + post=0 should decrease synapse weight."""
    syn = QuantumSynapse(0.5, w_min=0.0, w_max=1.0)
    stdp = QuantumSTDP(learning_rate=0.1)
    w_before = syn.weight
    stdp.update(syn, pre_measured=1, post_measured=0)
    assert syn.weight < w_before


def test_stdp_gradient_sign_at_half():
    """At theta = pi/2 (weight=0.5), <Z> = 0, gradient magnitude should be nonzero."""
    stdp = QuantumSTDP()
    exp_plus = stdp._expectation_z(np.pi / 2 + np.pi / 2)
    exp_minus = stdp._expectation_z(np.pi / 2 - np.pi / 2)
    gradient = (exp_plus - exp_minus) / (2.0 * np.sin(np.pi / 2))
    assert np.isfinite(gradient)
    assert abs(gradient) > 0.5
