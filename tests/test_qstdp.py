# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qstdp
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


def test_stdp_no_spike_no_change():
    """No pre spike → no weight change."""
    syn = QuantumSynapse(0.5, w_min=0.0, w_max=1.0)
    stdp = QuantumSTDP(learning_rate=0.1)
    w_before = syn.weight
    stdp.update(syn, pre_measured=0, post_measured=0)
    assert syn.weight == w_before


def test_stdp_learning_rate_effect():
    """Higher learning rate → bigger weight change."""
    syn_slow = QuantumSynapse(0.5)
    syn_fast = QuantumSynapse(0.5)
    QuantumSTDP(learning_rate=0.01).update(syn_slow, 1, 1)
    QuantumSTDP(learning_rate=0.5).update(syn_fast, 1, 1)
    assert abs(syn_fast.weight - 0.5) > abs(syn_slow.weight - 0.5)


def test_stdp_weight_stays_bounded():
    """Repeated LTP shouldn't exceed w_max."""
    syn = QuantumSynapse(0.9, w_min=0.0, w_max=1.0)
    stdp = QuantumSTDP(learning_rate=0.5)
    for _ in range(20):
        stdp.update(syn, pre_measured=1, post_measured=1)
    assert syn.weight <= 1.0


def test_stdp_expectation_z_range():
    stdp = QuantumSTDP()
    for theta in np.linspace(0, 2 * np.pi, 20):
        z = stdp._expectation_z(theta)
        assert -1.0 <= z <= 1.0 + 1e-10
