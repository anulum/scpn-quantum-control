# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for pulse-schedule feasibility helpers
"""Guard and coercion tests for the pulse-schedule feasibility helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from scpn_quantum_control.hardware.pulse_feasibility import (
    _optional_float,
    _optional_int,
    summarise_pulse_schedule,
)
from scpn_quantum_control.phase.pulse_shaping import PulseSchedule


def test_summarise_rejects_non_positive_qubits() -> None:
    """A schedule with a non-positive qubit count is rejected."""
    schedule = cast(PulseSchedule, SimpleNamespace(n_qubits=0, pulses=()))
    with pytest.raises(ValueError, match="schedule n_qubits must be positive"):
        summarise_pulse_schedule(schedule)


def test_optional_float_coerces_positive_value() -> None:
    """A positive optional float is coerced to float."""
    assert _optional_float(2.0, "min_time_step") == 2.0


def test_optional_int_passes_through_positive_value() -> None:
    """A positive optional integer is returned unchanged."""
    assert _optional_int(3, "max_pulses") == 3
