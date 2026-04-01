# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Orchestrator Adapter
"""Tests for orchestrator-to-quantum artifact adapter — elite multi-angle coverage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scpn_quantum_control.bridge.orchestrator_adapter import PhaseOrchestratorAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _DummyLock:
    source_layer: int
    target_layer: int
    plv: float
    mean_lag: float


@dataclass
class _DummyLayer:
    R: float
    psi: float
    lock_signatures: dict[str, _DummyLock]


@dataclass
class _DummyState:
    layers: list[_DummyLayer]
    cross_layer_alignment: np.ndarray
    stability_proxy: float
    regime_id: str


# ---------------------------------------------------------------------------
# from_orchestrator_state — dataclass input
# ---------------------------------------------------------------------------


class TestFromDataclassState:
    def test_accepts_dataclass(self):
        state = _DummyState(
            layers=[
                _DummyLayer(R=0.81, psi=0.5, lock_signatures={"0_1": _DummyLock(0, 1, 0.93, 0.1)}),
                _DummyLayer(R=0.42, psi=1.4, lock_signatures={}),
            ],
            cross_layer_alignment=np.array([[1.0, 0.93], [0.93, 1.0]]),
            stability_proxy=-0.21,
            regime_id="NOMINAL",
        )
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(state)
        assert artifact.regime_id == "NOMINAL"
        assert artifact.layers[0].lock_signatures["0_1"].target_layer == 1
        np.testing.assert_allclose(artifact.cross_layer_alignment[0, 1], 0.93)

    def test_stability_proxy_preserved(self):
        state = _DummyState(
            layers=[_DummyLayer(R=0.5, psi=0.0, lock_signatures={})],
            cross_layer_alignment=np.eye(1),
            stability_proxy=-0.75,
            regime_id="TEST",
        )
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(state)
        assert artifact.stability_proxy == -0.75

    def test_metadata_injected(self):
        state = _DummyState(
            layers=[_DummyLayer(R=0.5, psi=0.0, lock_signatures={})],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.0,
            regime_id="X",
        )
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(
            state, metadata={"source": "test"}
        )
        assert artifact.metadata["source"] == "test"
        assert artifact.metadata["adapter"] == "scpn_quantum_control.bridge.orchestrator"


# ---------------------------------------------------------------------------
# from_orchestrator_state — dict input with legacy keys
# ---------------------------------------------------------------------------


class TestFromDictPayload:
    def test_legacy_lock_keys(self):
        payload = {
            "layers": [
                {"R": 0.6, "psi": 1.2, "locks": {"2_3": {"plv": 0.88, "lag": 0.25}}},
                {"R": 0.4, "psi": 2.4, "locks": {}},
            ],
            "cross_alignment": [[1.0, 0.5], [0.5, 1.0]],
            "stability": -0.5,
            "regime": "DEGRADED",
        }
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(payload)
        sig = artifact.layers[0].lock_signatures["2_3"]
        assert sig.source_layer == 2
        assert sig.target_layer == 3
        assert sig.mean_lag == 0.25

    def test_canonical_field_names(self):
        payload = {
            "layers": [{"R": 0.9, "psi": 0.1, "lock_signatures": {}}],
            "cross_layer_alignment": [[1.0]],
            "stability_proxy": 0.5,
            "regime_id": "STABLE",
        }
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(payload)
        assert artifact.regime_id == "STABLE"
        assert artifact.stability_proxy == 0.5


# ---------------------------------------------------------------------------
# to_scpn_control_telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_layout(self):
        payload = {
            "layers": [
                {"R": 0.75, "psi": 0.9, "lock_signatures": {}},
                {"R": 0.5, "psi": 2.0},
            ],
            "cross_layer_alignment": [[1.0, 0.4], [0.4, 1.0]],
            "stability_proxy": -0.33,
            "regime_id": "RECOVERY",
        }
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(payload)
        telemetry = PhaseOrchestratorAdapter.to_scpn_control_telemetry(artifact)
        assert telemetry["regime"] == "RECOVERY"
        assert telemetry["stability"] == -0.33
        assert telemetry["layers"][0]["R"] == 0.75
        assert telemetry["cross_alignment"][0][1] == 0.4

    def test_locks_in_telemetry(self):
        payload = {
            "layers": [
                {"R": 0.8, "psi": 0.0, "locks": {"0_1": {"plv": 0.95, "lag": 0.1}}},
            ],
            "cross_alignment": [[1.0]],
            "stability": 0.0,
            "regime": "X",
        }
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(payload)
        telemetry = PhaseOrchestratorAdapter.to_scpn_control_telemetry(artifact)
        assert telemetry["layers"][0]["locks"]["0_1"]["plv"] == 0.95


# ---------------------------------------------------------------------------
# to_orchestrator_payload
# ---------------------------------------------------------------------------


class TestToOrchestratorPayload:
    def test_roundtrip(self):
        payload = {
            "layers": [{"R": 0.7, "psi": 1.0, "lock_signatures": {}}],
            "cross_layer_alignment": [[1.0]],
            "stability_proxy": -0.1,
            "regime_id": "NOMINAL",
        }
        artifact = PhaseOrchestratorAdapter.from_orchestrator_state(payload)
        out = PhaseOrchestratorAdapter.to_orchestrator_payload(artifact)
        assert out["regime_id"] == "NOMINAL"
        assert len(out["layers"]) == 1


# ---------------------------------------------------------------------------
# build_knm_from_binding_spec
# ---------------------------------------------------------------------------


class TestBuildKnmFromBindingSpec:
    def _make_spec(self, n_osc_per_layer=None):
        if n_osc_per_layer is None:
            n_osc_per_layer = [2, 1]
        layers = []
        for i, n in enumerate(n_osc_per_layer):
            layer = {
                "name": f"layer_{i}",
                "index": i,
                "oscillator_ids": [f"o{i}_{j}" for j in range(n)],
                "natural_frequency": 1.0 + i * 0.5,
            }
            layers.append(layer)
        return {
            "layers": layers,
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
        }

    def test_shape(self):
        spec = self._make_spec([2, 1])
        knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)
        assert knm.shape == (3, 3)

    def test_diagonal_equals_base_strength(self):
        spec = self._make_spec([2, 1])
        knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)
        np.testing.assert_allclose(np.diag(knm), 0.45)

    def test_zero_diagonal_option(self):
        spec = self._make_spec([2, 1])
        knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec, zero_diagonal=True)
        np.testing.assert_allclose(np.diag(knm), 0.0)

    def test_symmetric(self):
        spec = self._make_spec([3, 2])
        knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)
        np.testing.assert_allclose(knm, knm.T)

    def test_empty_raises(self):
        spec = {"layers": [], "coupling": {"base_strength": 0.1, "decay_alpha": 0.1}}
        with pytest.raises(ValueError, match="at least one oscillator"):
            PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)

    def test_coupling_decays_with_distance(self):
        spec = self._make_spec([4])
        knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)
        # K(0,1) > K(0,3) because closer oscillators are more strongly coupled
        assert knm[0, 1] > knm[0, 3]


# ---------------------------------------------------------------------------
# build_omega_from_binding_spec
# ---------------------------------------------------------------------------


class TestBuildOmegaFromBindingSpec:
    def test_shape_and_values(self):
        spec = {
            "layers": [
                {
                    "name": "a",
                    "index": 0,
                    "oscillator_ids": ["m0", "m1"],
                    "natural_frequency": 1.4,
                },
                {"name": "b", "index": 1, "oscillator_ids": ["e0"]},
            ],
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
        }
        omega = PhaseOrchestratorAdapter.build_omega_from_binding_spec(spec, default_omega=1.0)
        np.testing.assert_allclose(omega, [1.4, 1.4, 1.0])

    def test_default_omega_applied(self):
        spec = {
            "layers": [{"name": "x", "index": 0, "oscillator_ids": ["o0"]}],
            "coupling": {"base_strength": 0.1, "decay_alpha": 0.1, "templates": {}},
        }
        omega = PhaseOrchestratorAdapter.build_omega_from_binding_spec(spec, default_omega=2.5)
        np.testing.assert_allclose(omega, [2.5])

    def test_empty_raises(self):
        spec = {"layers": [], "coupling": {"base_strength": 0.1, "decay_alpha": 0.1}}
        with pytest.raises(ValueError, match="at least one oscillator"):
            PhaseOrchestratorAdapter.build_omega_from_binding_spec(spec)
