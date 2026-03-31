# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Orchestrator Adapter Errors
"""Multi-angle tests for PhaseOrchestratorAdapter: error paths, edge cases,
field resolution, payload conversion.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.orchestrator_adapter import PhaseOrchestratorAdapter


class TestBuildKnmErrors:
    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="at least one oscillator"):
            PhaseOrchestratorAdapter.build_knm_from_binding_spec({"layers": []})

    def test_none_layers_raises(self):
        with pytest.raises((ValueError, TypeError, KeyError)):
            PhaseOrchestratorAdapter.build_knm_from_binding_spec({"layers": None})

    def test_missing_layers_raises(self):
        with pytest.raises((ValueError, TypeError, KeyError, AttributeError)):
            PhaseOrchestratorAdapter.build_knm_from_binding_spec({})


class TestBuildOmegaErrors:
    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="at least one oscillator"):
            PhaseOrchestratorAdapter.build_omega_from_binding_spec({"layers": []})

    def test_none_layers_raises(self):
        with pytest.raises((ValueError, TypeError, KeyError)):
            PhaseOrchestratorAdapter.build_omega_from_binding_spec({"layers": None})


class TestFieldResolution:
    """Test _read_field helper via public API behaviour."""

    def test_layers_as_list(self):
        """Layers field accessed correctly when it's a list."""
        # Empty oscillator_ids means zero oscillators
        spec = {"layers": [{"oscillator_ids": []}]}
        with pytest.raises(ValueError, match="at least one oscillator"):
            PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)

    def test_layers_no_oscillator_ids_raises(self):
        """Layer without oscillator_ids → 0 oscillators → error."""
        spec = {"layers": [{"name": "L0"}]}
        with pytest.raises(ValueError, match="at least one oscillator"):
            PhaseOrchestratorAdapter.build_knm_from_binding_spec(spec)


class TestStaticMethods:
    def test_adapter_methods_are_static(self):
        assert isinstance(
            PhaseOrchestratorAdapter.__dict__["build_knm_from_binding_spec"],
            staticmethod,
        )
        assert isinstance(
            PhaseOrchestratorAdapter.__dict__["build_omega_from_binding_spec"],
            staticmethod,
        )

    def test_adapter_has_from_orchestrator_state(self):
        assert hasattr(PhaseOrchestratorAdapter, "from_orchestrator_state")
        assert callable(PhaseOrchestratorAdapter.from_orchestrator_state)

    def test_adapter_has_to_orchestrator_payload(self):
        assert hasattr(PhaseOrchestratorAdapter, "to_orchestrator_payload")
        assert callable(PhaseOrchestratorAdapter.to_orchestrator_payload)
