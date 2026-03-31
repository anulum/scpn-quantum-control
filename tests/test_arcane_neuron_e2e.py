# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Arcane Neuron E2E
"""End-to-end integration tests for ArcaneNeuronBridge with real sc-neurocore."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scpn_quantum_control.bridge.snn_adapter import ArcaneNeuronBridge

HAS_SC_NEUROCORE = importlib.util.find_spec("sc_neurocore") is not None

_SKIP_NO_SC = pytest.mark.skipif(not HAS_SC_NEUROCORE, reason="sc-neurocore not installed")


@_SKIP_NO_SC
class TestArcaneNeuronE2E:
    def test_10_step_trajectory(self):
        """Run 10 steps and verify output structure at each step."""
        bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
        for _step in range(10):
            result = bridge.step(np.array([1.5, 0.8, 0.2]))
            assert result["spikes"].shape == (3,)
            assert result["output_currents"].shape == (2,)
            assert result["v_deep"].shape == (3,)
            assert result["confidence"].shape == (3,)
            assert all(s in (0.0, 1.0) for s in result["spikes"])

    def test_identity_survives_reset(self):
        """v_deep accumulates over time and survives reset."""
        bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
        for _ in range(50):
            bridge.step(np.array([2.0, 2.0, 2.0]))
        deep_before = bridge.step(np.array([0.0, 0.0, 0.0]))["v_deep"].copy()
        bridge.reset()
        deep_after = np.array([n.get_state()["v_deep"] for n in bridge.neurons])
        np.testing.assert_array_equal(deep_before, deep_after)

    def test_quantum_output_varies_with_input(self):
        """Different input patterns produce different spike histories."""
        bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
        for _ in range(100):
            bridge.step(np.array([5.0, 0.0, 0.0]))
        hist_a = np.array(bridge._spike_history)
        rate_a = hist_a.mean(axis=0)

        bridge.reset()
        bridge._spike_history.clear()
        for _ in range(100):
            bridge.step(np.array([0.0, 0.0, 5.0]))
        hist_b = np.array(bridge._spike_history)
        rate_b = hist_b.mean(axis=0)

        # Neuron 0 should fire more in pattern A, neuron 2 in pattern B
        assert not np.allclose(rate_a, rate_b, atol=0.05)

    def test_spike_history_length(self):
        """Spike history grows with each step."""
        bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
        for i in range(25):
            bridge.step(np.array([1.0, 1.0, 1.0]))
            assert len(bridge._spike_history) == i + 1

    def test_confidence_evolves(self):
        """Confidence changes as neuron accumulates experience."""
        bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
        result_0 = bridge.step(np.array([1.0, 1.0, 1.0]))
        for _ in range(100):
            bridge.step(np.array([1.0, 1.0, 1.0]))
        result_100 = bridge.step(np.array([1.0, 1.0, 1.0]))
        assert not np.allclose(result_0["confidence"], result_100["confidence"], atol=0.01)

    def test_zero_input_no_crash(self):
        """Zero input doesn't crash."""
        bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
        result = bridge.step(np.zeros(3))
        assert result["spikes"].shape == (3,)


# ---------------------------------------------------------------------------
# Tests that work WITHOUT sc-neurocore (pure numpy SNN bridge)
# ---------------------------------------------------------------------------


class TestSNNQuantumBridgeNoDeps:
    """Test SNNQuantumBridge and utility functions without sc-neurocore."""

    def test_spike_train_to_rotations_2d(self):
        from scpn_quantum_control.bridge.snn_adapter import spike_train_to_rotations

        spikes = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        angles = spike_train_to_rotations(spikes, window=3)
        assert angles.shape == (3,)
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi)

    def test_spike_train_to_rotations_1d(self):
        from scpn_quantum_control.bridge.snn_adapter import spike_train_to_rotations

        spikes = np.array([1, 0, 1, 0])
        angles = spike_train_to_rotations(spikes, window=4)
        assert angles.shape == (4,)

    def test_spike_train_to_rotations_all_zeros(self):
        from scpn_quantum_control.bridge.snn_adapter import spike_train_to_rotations

        spikes = np.zeros((10, 3))
        angles = spike_train_to_rotations(spikes, window=5)
        np.testing.assert_allclose(angles, 0.0)

    def test_spike_train_to_rotations_all_ones(self):
        from scpn_quantum_control.bridge.snn_adapter import spike_train_to_rotations

        spikes = np.ones((10, 3))
        angles = spike_train_to_rotations(spikes, window=5)
        np.testing.assert_allclose(angles, np.pi)

    def test_quantum_measurement_to_current(self):
        from scpn_quantum_control.bridge.snn_adapter import quantum_measurement_to_current

        values = np.array([0.5, 0.8, 0.1])
        currents = quantum_measurement_to_current(values, scale=2.0)
        np.testing.assert_allclose(currents, [1.0, 1.6, 0.2])

    def test_quantum_measurement_to_current_default_scale(self):
        from scpn_quantum_control.bridge.snn_adapter import quantum_measurement_to_current

        values = np.array([0.5, 1.0])
        currents = quantum_measurement_to_current(values)
        np.testing.assert_allclose(currents, [0.5, 1.0])

    def test_snn_quantum_bridge_forward(self):
        from scpn_quantum_control.bridge.snn_adapter import SNNQuantumBridge

        bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
        spikes = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        output = bridge.forward(spikes)
        assert output.shape == (2,)
        assert np.all(np.isfinite(output))

    def test_snn_bridge_zero_spikes(self):
        from scpn_quantum_control.bridge.snn_adapter import SNNQuantumBridge

        bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
        spikes = np.zeros((5, 3))
        output = bridge.forward(spikes)
        assert output.shape == (2,)

    def test_arcane_neuron_bridge_import_error(self):
        """Without sc-neurocore, ArcaneNeuronBridge should raise ImportError."""
        if HAS_SC_NEUROCORE:
            pytest.skip("sc-neurocore is installed")
        with pytest.raises(ImportError, match="sc-neurocore"):
            ArcaneNeuronBridge(n_neurons=2, n_inputs=3)
