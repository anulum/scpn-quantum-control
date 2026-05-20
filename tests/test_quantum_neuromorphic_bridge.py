# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the QSNN quantum neuromorphic bridge."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control as sqc
from scpn_quantum_control.qsnn.quantum_neuromorphic_bridge import (
    CLAIM_BOUNDARY,
    DynamicCouplingConfig,
    QuantumLIFConfig,
    QuantumNeuromorphicBridge,
    TraceSTDPConfig,
    TraceSTDPState,
)


def test_bridge_step_returns_complete_finite_state():
    bridge = QuantumNeuromorphicBridge(
        n_inputs=2,
        n_neurons=3,
        lif=QuantumLIFConfig(v_threshold=0.4, tau_mem=3.0, dt=1.0),
        input_weights=np.array([[0.7, 0.1], [0.1, 0.8], [0.4, 0.4]]),
        seed=7,
        deterministic=True,
    )

    result = bridge.step(np.array([1.0, 0.2]))

    assert result.spikes.shape == (3,)
    assert result.spike_probabilities.shape == (3,)
    assert result.membrane.shape == (3,)
    assert result.input_weights.shape == (3, 2)
    assert result.recurrent_weights.shape == (3, 3)
    assert result.quantum_circuit.num_qubits == 3
    assert set(np.unique(result.spikes)).issubset({0, 1})
    assert np.all(np.isfinite(result.spike_probabilities))
    assert np.all((result.spike_probabilities >= 0.0) & (result.spike_probabilities <= 1.0))
    assert result.claim_boundary == CLAIM_BOUNDARY


def test_bridge_is_exported_from_public_api():
    bridge = sqc.QuantumNeuromorphicBridge(n_inputs=1, n_neurons=1, seed=1)
    result = bridge.step(np.array([1.0]))

    assert isinstance(result, sqc.NeuromorphicStepResult)
    assert sqc.CLAIM_BOUNDARY == CLAIM_BOUNDARY


def test_trace_stdp_potentiates_causal_pre_before_post_input_weight():
    bridge = QuantumNeuromorphicBridge(
        n_inputs=1,
        n_neurons=1,
        lif=QuantumLIFConfig(v_threshold=0.5, tau_mem=2.0, dt=1.0),
        stdp=TraceSTDPConfig(a_plus=0.08, a_minus=0.02, tau_pre=5.0, tau_post=5.0),
        input_weights=np.array([[0.2]]),
        seed=3,
        deterministic=True,
    )

    bridge.apply_plasticity(pre_spikes=np.array([1.0]), post_spikes=np.array([0.0]))
    before = bridge.input_weights[0, 0]
    bridge.apply_plasticity(pre_spikes=np.array([0.0]), post_spikes=np.array([1.0]))

    assert bridge.input_weights[0, 0] > before


def test_trace_stdp_depresses_anti_causal_post_before_pre_input_weight():
    bridge = QuantumNeuromorphicBridge(
        n_inputs=1,
        n_neurons=1,
        lif=QuantumLIFConfig(v_threshold=0.5, tau_mem=2.0, dt=1.0),
        stdp=TraceSTDPConfig(a_plus=0.02, a_minus=0.08, tau_pre=5.0, tau_post=5.0),
        input_weights=np.array([[0.8]]),
        seed=3,
        deterministic=True,
    )

    bridge.apply_plasticity(pre_spikes=np.array([0.0]), post_spikes=np.array([1.0]))
    before = bridge.input_weights[0, 0]
    bridge.apply_plasticity(pre_spikes=np.array([1.0]), post_spikes=np.array([0.0]))

    assert bridge.input_weights[0, 0] < before


def test_dynamic_coupling_updates_recurrent_weights_without_self_loops():
    bridge = QuantumNeuromorphicBridge(
        n_inputs=2,
        n_neurons=3,
        lif=QuantumLIFConfig(v_threshold=0.4, tau_mem=3.0, dt=1.0),
        coupling=DynamicCouplingConfig(learning_rate=0.2, decay_rate=0.05, coherence_gain=0.5),
        seed=11,
        deterministic=True,
    )

    before = bridge.recurrent_weights.copy()
    result = bridge.step(np.array([1.0, 0.8]))

    assert not np.allclose(result.recurrent_weights, before)
    np.testing.assert_allclose(np.diag(result.recurrent_weights), 0.0)
    assert np.all(result.recurrent_weights >= 0.0)
    assert np.all(result.recurrent_weights <= 1.0)
    assert result.coupling_delta.shape == (3, 3)


def test_seeded_stochastic_bridge_is_reproducible():
    kwargs = dict(
        n_inputs=2,
        n_neurons=2,
        lif=QuantumLIFConfig(v_threshold=0.3, tau_mem=2.0, dt=1.0, n_shots=32),
        seed=123,
        deterministic=False,
    )
    bridge_a = QuantumNeuromorphicBridge(**kwargs)
    bridge_b = QuantumNeuromorphicBridge(**kwargs)

    spikes_a = [bridge_a.step(np.array([0.7, 0.2])).spikes.tolist() for _ in range(6)]
    spikes_b = [bridge_b.step(np.array([0.7, 0.2])).spikes.tolist() for _ in range(6)]

    assert spikes_a == spikes_b


def test_bridge_rejects_invalid_shapes_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="input_weights shape"):
        QuantumNeuromorphicBridge(n_inputs=2, n_neurons=3, input_weights=np.ones((2, 3)))

    bridge = QuantumNeuromorphicBridge(n_inputs=2, n_neurons=3, seed=5)

    with pytest.raises(ValueError, match="external_current shape"):
        bridge.step(np.ones(3))

    with pytest.raises(ValueError, match="finite"):
        bridge.step(np.array([1.0, np.nan]))


def test_trace_state_decay_is_bounded_and_finite():
    state = TraceSTDPState(n_pre=2, n_post=2)
    state.pre_trace[:] = 1.0
    state.post_trace[:] = 1.0

    state.decay(dt=1.0, tau_pre=2.0, tau_post=4.0)

    assert np.all((state.pre_trace > 0.0) & (state.pre_trace < 1.0))
    assert np.all((state.post_trace > 0.0) & (state.post_trace < 1.0))
    assert np.all(np.isfinite(state.pre_trace))
    assert np.all(np.isfinite(state.post_trace))
