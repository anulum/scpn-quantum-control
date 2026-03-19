# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for cross-repo integration wiring (SNN, SSGF, orchestrator, fusion-core)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.snn_adapter import ArcaneNeuronBridge
from scpn_quantum_control.bridge.ssgf_adapter import SSGFQuantumLoop
from scpn_quantum_control.control.q_disruption_iter import from_fusion_core_shot
from scpn_quantum_control.identity.binding_spec import (
    ORCHESTRATOR_MAPPING,
    orchestrator_to_quantum_phases,
    quantum_to_orchestrator_phases,
)

# --- ArcaneNeuronBridge tests ---


class TestArcaneNeuronBridge:
    def test_import_error_without_sc_neurocore(self):
        """ArcaneNeuronBridge raises ImportError if sc-neurocore not installed."""
        try:
            bridge = ArcaneNeuronBridge(2, 3)
            # sc-neurocore is installed — test the bridge works
            assert bridge.bridge.n_neurons == 2
            assert len(bridge.neurons) == 3
        except ImportError:
            pass  # expected when sc-neurocore not available

    @pytest.fixture
    def bridge(self):
        try:
            return ArcaneNeuronBridge(2, 3, seed=42)
        except ImportError:
            pytest.skip("sc-neurocore not installed")

    def test_step_neurons(self, bridge):
        currents = np.array([0.5, 1.0, 1.5])
        spikes = bridge.step_neurons(currents)
        assert spikes.shape == (3,)
        assert all(s in (0.0, 1.0) for s in spikes)

    def test_quantum_forward_empty(self, bridge):
        out = bridge.quantum_forward()
        assert out.shape == (2,)
        np.testing.assert_array_equal(out, 0.0)

    def test_full_step(self, bridge):
        result = bridge.step(np.array([1.0, 1.0, 1.0]))
        assert "spikes" in result
        assert "output_currents" in result
        assert "v_deep" in result
        assert "confidence" in result
        assert result["v_deep"].shape == (3,)

    def test_reset_preserves_identity(self, bridge):
        for _ in range(10):
            bridge.step(np.array([1.0, 1.0, 1.0]))
        deep_before = np.array([n.get_state()["v_deep"] for n in bridge.neurons])
        bridge.reset()
        deep_after = np.array([n.get_state()["v_deep"] for n in bridge.neurons])
        np.testing.assert_array_equal(deep_before, deep_after)

    def test_multiple_steps_accumulate_history(self, bridge):
        for _ in range(5):
            bridge.step(np.array([1.0, 0.5, 0.0]))
        assert len(bridge._spike_history) == 5


# --- SSGFQuantumLoop tests ---


class MockNodeSpace:
    """Minimal mock of SSGF NodeSpace for testing."""

    def __init__(self, n: int = 4):
        self.theta = np.random.default_rng(42).uniform(0, 2 * np.pi, n)
        self.W = np.array(
            [
                [0.0, 0.3, 0.1, 0.0],
                [0.3, 0.0, 0.2, 0.1],
                [0.1, 0.2, 0.0, 0.3],
                [0.0, 0.1, 0.3, 0.0],
            ]
        )


class MockSSGFEngine:
    """Minimal mock of SSGFEngine for testing."""

    def __init__(self, n: int = 4):
        self.ns = MockNodeSpace(n)


class TestSSGFQuantumLoop:
    def test_read_engine(self):
        engine = MockSSGFEngine(4)
        loop = SSGFQuantumLoop(engine, dt=0.1)
        W, theta = loop._read_engine()
        assert W.shape == (4, 4)
        assert theta.shape == (4,)

    def test_write_theta(self):
        engine = MockSSGFEngine(4)
        loop = SSGFQuantumLoop(engine, dt=0.1)
        new_theta = np.array([0.0, 1.0, 2.0, 3.0])
        loop._write_theta(new_theta)
        np.testing.assert_array_equal(engine.ns.theta, new_theta)

    def test_quantum_step_returns_state(self):
        engine = MockSSGFEngine(4)
        loop = SSGFQuantumLoop(engine, dt=0.05, trotter_reps=1)
        result = loop.quantum_step()
        assert "theta" in result
        assert "R_global" in result
        assert len(result["theta"]) == 4
        assert 0.0 <= result["R_global"] <= 1.0

    def test_quantum_step_modifies_engine_theta(self):
        engine = MockSSGFEngine(4)
        theta_before = engine.ns.theta.copy()
        loop = SSGFQuantumLoop(engine, dt=0.1, trotter_reps=2)
        loop.quantum_step()
        # theta should change (non-trivial W coupling)
        assert not np.allclose(engine.ns.theta, theta_before, atol=1e-10)


# --- Orchestrator phase mapping tests ---


class TestOrchestratorMapping:
    def test_mapping_covers_all_quantum_ids(self):
        from scpn_quantum_control.identity.binding_spec import ARCANE_SAPIENCE_SPEC

        all_ids = [oid for lay in ARCANE_SAPIENCE_SPEC["layers"] for oid in lay["oscillator_ids"]]
        assert len(all_ids) == 18
        for qid in all_ids:
            assert qid in ORCHESTRATOR_MAPPING

    def test_mapping_covers_all_orchestrator_ids(self):
        total = sum(len(v) for v in ORCHESTRATOR_MAPPING.values())
        assert total == 35  # 5+5+5+5+8+7 from identity_coherence domainpack

    def test_quantum_to_orchestrator_roundtrip(self):
        theta_q = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        orch_phases = quantum_to_orchestrator_phases(theta_q)
        assert len(orch_phases) == 35
        theta_back = orchestrator_to_quantum_phases(orch_phases)
        assert theta_back.shape == (18,)
        # Roundtrip correct modulo 2pi (atan2 returns [-pi, pi])
        diff = np.angle(np.exp(1j * (theta_back - theta_q)))
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)

    def test_orchestrator_to_quantum_circular_mean(self):
        # Two sub-oscillators at 0 and pi/2 -> circular mean at pi/4
        orch_phases = {"ws_action_first": 0.0, "ws_verify_before_claim": np.pi / 2}
        for key in ORCHESTRATOR_MAPPING:
            if key != "ws_0":
                for sub_key in ORCHESTRATOR_MAPPING[key]:
                    orch_phases[sub_key] = 0.0
        theta = orchestrator_to_quantum_phases(orch_phases)
        assert abs(theta[0] - np.pi / 4) < 0.01


# --- Fusion-core shot adapter tests ---


class TestFusionCoreShot:
    def test_from_fusion_core_shot_basic(self):
        shot = {
            "Ip_MA": np.array([14.5, 14.8, 15.0]),
            "q95": np.array([3.1, 3.0, 2.9]),
            "beta_N": np.array([1.5, 1.6, 1.7]),
            "locked_mode_amp": np.array([0.0001, 0.0002, 0.0001]),
            "ne_1e19": np.array([0.8, 0.85, 0.9]),
            "is_disruption": 0,
        }
        features, label, warnings = from_fusion_core_shot(shot)
        assert features.shape == (11,)
        assert label == 0
        assert np.all(features >= 0.0) and np.all(features <= 1.0)
        assert len(warnings) == 6  # 6 of 11 features defaulted

    def test_from_fusion_core_shot_disruption(self):
        shot = {
            "Ip_MA": np.array([10.0]),
            "q95": np.array([2.0]),
            "beta_N": np.array([3.8]),
            "locked_mode_amp": np.array([0.008]),
            "is_disruption": 1,
        }
        features, label, warnings = from_fusion_core_shot(shot)
        assert label == 1
        assert features.shape == (11,)
        assert len(warnings) == 7  # 7 of 11 defaulted (no ne_1e19)

    def test_from_fusion_core_shot_missing_keys(self):
        shot = {"is_disruption": 0}
        features, label, warnings = from_fusion_core_shot(shot)
        assert features.shape == (11,)
        assert label == 0
        assert len(warnings) == 11  # all features defaulted
