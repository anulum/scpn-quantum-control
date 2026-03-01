"""Tests for orchestrator-to-quantum artifact adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_quantum_control.bridge.orchestrator_adapter import PhaseOrchestratorAdapter


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


def test_adapter_accepts_dataclass_state() -> None:
    state = _DummyState(
        layers=[
            _DummyLayer(
                R=0.81,
                psi=0.5,
                lock_signatures={"0_1": _DummyLock(0, 1, 0.93, 0.1)},
            ),
            _DummyLayer(R=0.42, psi=1.4, lock_signatures={}),
        ],
        cross_layer_alignment=np.array([[1.0, 0.93], [0.93, 1.0]], dtype=np.float64),
        stability_proxy=-0.21,
        regime_id="NOMINAL",
    )

    artifact = PhaseOrchestratorAdapter.from_orchestrator_state(state)
    assert artifact.regime_id == "NOMINAL"
    assert artifact.layers[0].lock_signatures["0_1"].target_layer == 1
    np.testing.assert_allclose(artifact.cross_layer_alignment[0, 1], 0.93)


def test_adapter_accepts_dict_payload_with_legacy_lock_keys() -> None:
    payload = {
        "layers": [
            {
                "R": 0.6,
                "psi": 1.2,
                "locks": {"2_3": {"plv": 0.88, "lag": 0.25}},
            },
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


def test_adapter_emits_control_telemetry_layout() -> None:
    payload = {
        "layers": [{"R": 0.75, "psi": 0.9, "lock_signatures": {}}, {"R": 0.5, "psi": 2.0}],
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


def test_binding_spec_to_knm_and_omega() -> None:
    binding_spec = {
        "layers": [
            {
                "name": "macro",
                "index": 0,
                "oscillator_ids": ["m0", "m1"],
                "natural_frequency": 1.4,
            },
            {
                "name": "edge",
                "index": 1,
                "oscillator_ids": ["e0"],
            },
        ],
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
    }
    knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(binding_spec)
    omega = PhaseOrchestratorAdapter.build_omega_from_binding_spec(binding_spec, default_omega=1.0)

    assert knm.shape == (3, 3)
    np.testing.assert_allclose(np.diag(knm), np.array([0.45, 0.45, 0.45]))
    np.testing.assert_allclose(omega, np.array([1.4, 1.4, 1.0]))
