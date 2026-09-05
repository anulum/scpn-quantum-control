# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Analog Execution Unit Admission
"""Unit vocabulary for design-plan admission, not provider SDK conversion."""

from collections.abc import Mapping
from typing import Literal

UNIT_CONTRACT = "analog_execution_units.v1"
UnitStatus = Literal["canonical_design_rates", "uncalibrated_design_units", "unverified"]


def _execution_unit_status(calibration: Mapping[str, object]) -> UnitStatus:
    """Classify declared design-plan units without rescaling payload values."""
    units = tuple(calibration[key] for key in ("duration_unit", "coupling_unit", "detuning_unit"))
    if units == ("us", "rad/us", "rad/us"):
        return "canonical_design_rates"
    if units in (
        ("design_time", "dimensionless_native_coupling", "dimensionless_detuning"),
        ("dt", "arb", "arb"),
    ):
        return "uncalibrated_design_units"
    raise ValueError(
        "unsupported analog execution units: expected us, rad/us, rad/us; "
        "unit conversion and provider-native calibration require a separate adapter"
    )
