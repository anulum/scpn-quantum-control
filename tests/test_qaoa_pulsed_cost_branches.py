# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the FRC pulsed QAOA cost
"""Branch and guard tests for the FRC pulsed-shot QAOA cost model.

Covers the short-field MRTI shortcut, the field-decode and capacitor-energy
guards and the native MRTI-growth kernel fallback.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.control.qaoa_pulsed_cost import (
    FRCPlasmaSurrogate,
    FRCQAOAObjective,
    _mrti_growth,
    decode_schedule_to_field,
    frc_pulsed_shot_cost,
)


def _objective() -> FRCQAOAObjective:
    return FRCQAOAObjective(
        target_s_parameter=2.5,
        bank_energy_budget_J=5.0e5,
        mrti_amplitude_max_m=1.0e-2,
        tilt_margin_required=0.3,
    )


def test_mrti_amplitude_short_field_returns_initial() -> None:
    """A field profile shorter than two samples returns the initial perturbation."""
    surrogate = FRCPlasmaSurrogate()
    amplitude = surrogate.mrti_amplitude(np.array([1.0], dtype=np.float64), dt_s=1.0e-6)
    assert amplitude == surrogate.initial_perturbation_m


def test_decode_schedule_rejects_non_positive_delta_field() -> None:
    """A non-positive field increment is rejected."""
    with pytest.raises(ValueError, match="delta_field_T must be positive"):
        decode_schedule_to_field(np.ones(4, dtype=np.float64), delta_field_T=0.0)


def test_cost_rejects_non_positive_capacitor_energy() -> None:
    """A non-positive available capacitor energy is rejected."""
    schedule = np.ones(8, dtype=np.float64)
    target = np.linspace(0.5, 4.0, 8)
    with pytest.raises(ValueError, match="available_capacitor_energy_J must be positive"):
        frc_pulsed_shot_cost(schedule, target, 0.0, _objective())


def test_mrti_growth_falls_back_on_engine_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising native MRTI-growth kernel falls back to the NumPy integrator."""

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("engine refused the MRTI growth")

    stub = types.ModuleType("scpn_quantum_engine")
    stub.frc_mrti_growth = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    field = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    growth = _mrti_growth(field, 1.0e-6, 0.5, 0.3, 1.0e-3, 1.0e-7)
    assert np.isfinite(growth)
    assert growth >= 0.0
