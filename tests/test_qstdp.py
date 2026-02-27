"""Tests for qsnn/qstdp.py."""
import numpy as np
import pytest

from scpn_quantum_control.qsnn.qsynapse import QuantumSynapse
from scpn_quantum_control.qsnn.qstdp import QuantumSTDP


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
