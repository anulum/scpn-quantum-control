# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Orchestrator Adapter Helpers
"""Cover uncovered lines in orchestrator_adapter.py: 17-22, 29, 86, 134."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.orchestrator_adapter import PhaseOrchestratorAdapter


def _make_state(n_layers=2, regime="stable"):
    """Minimal orchestrator state dict."""
    return {
        "layers": [
            {
                "R": 0.9,
                "psi": 1.2,
                "lock_signatures": {"0_1": {"plv": 0.8, "mean_lag": 0.1}},
            }
            for _ in range(n_layers)
        ],
        "cross_layer_alignment": np.eye(n_layers).tolist(),
        "stability_proxy": 0.95,
        "regime_id": regime,
    }


def test_read_field_raises_on_missing_required():
    from scpn_quantum_control.bridge.orchestrator_adapter import _read_field

    with pytest.raises(KeyError, match="Missing required field"):
        _read_field({"a": 1}, "x", "y")


def test_read_field_returns_default():
    from scpn_quantum_control.bridge.orchestrator_adapter import _read_field

    assert _read_field({"a": 1}, "x", default=42) == 42


def test_read_field_getattr_path():
    from scpn_quantum_control.bridge.orchestrator_adapter import _read_field

    class Obj:
        foo = 99

    assert _read_field(Obj(), "foo") == 99


def test_infer_layer_pair_non_digit_key():
    from scpn_quantum_control.bridge.orchestrator_adapter import _infer_layer_pair

    assert _infer_layer_pair("lock_a", 5) == (5, 5)


def test_to_orchestrator_payload():
    state = _make_state()
    artifact = PhaseOrchestratorAdapter.from_orchestrator_state(state)
    payload = PhaseOrchestratorAdapter.to_orchestrator_payload(artifact)
    assert "layers" in payload
    assert "regime_id" in payload


def test_build_knm_zero_diagonal():
    binding = {
        "layers": [
            {"oscillator_ids": [0, 1], "natural_frequency": 1.0},
            {"oscillator_ids": [2, 3], "natural_frequency": 2.0},
        ],
        "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
    }
    knm = PhaseOrchestratorAdapter.build_knm_from_binding_spec(binding, zero_diagonal=True)
    assert np.all(np.diag(knm) == 0.0)
    assert knm.shape == (4, 4)


def test_from_orchestrator_state_roundtrip():
    state = _make_state(3, "NOMINAL")
    artifact = PhaseOrchestratorAdapter.from_orchestrator_state(state)
    assert artifact.regime_id == "NOMINAL"
    assert len(artifact.layers) == 3


def test_read_field_first_match():
    from scpn_quantum_control.bridge.orchestrator_adapter import _read_field

    d = {"a": 1, "b": 2}
    assert _read_field(d, "a", "b") == 1


def test_infer_layer_pair_digit_key():
    from scpn_quantum_control.bridge.orchestrator_adapter import _infer_layer_pair

    assert _infer_layer_pair("3_7", 0) == (3, 7)


def test_telemetry_output():
    state = _make_state(2, "OK")
    artifact = PhaseOrchestratorAdapter.from_orchestrator_state(state)
    telemetry = PhaseOrchestratorAdapter.to_scpn_control_telemetry(artifact)
    assert telemetry["regime"] == "OK"
    assert len(telemetry["layers"]) == 2
