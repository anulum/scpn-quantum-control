# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Error-path tests for PhaseOrchestratorAdapter binding spec builders."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.orchestrator_adapter import PhaseOrchestratorAdapter


def test_build_knm_zero_oscillators():
    with pytest.raises(ValueError, match="at least one oscillator"):
        PhaseOrchestratorAdapter.build_knm_from_binding_spec({"layers": []})


def test_build_omega_zero_oscillators():
    with pytest.raises(ValueError, match="at least one oscillator"):
        PhaseOrchestratorAdapter.build_omega_from_binding_spec({"layers": []})
