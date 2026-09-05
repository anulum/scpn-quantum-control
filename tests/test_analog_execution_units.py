# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Analog Execution Unit Tests
"""Exercise unit admission through real compile/export/execution-plan consumers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.hardware.analog_kuramoto import (
    AnalogKuramotoPlatform,
    ProviderAnalogPayload,
    compile_analog_kuramoto,
    export_provider_payload,
    prepare_provider_execution_plan,
)


@pytest.fixture(params=["pulser", "bloqade", "ibm_pulse"])
def analog_export(request: pytest.FixtureRequest) -> ProviderAnalogPayload:
    """Compile a signed two-oscillator problem and export each supported design target."""
    provider = str(request.param)
    platform = (
        AnalogKuramotoPlatform.CIRCUIT_QED
        if provider == "ibm_pulse"
        else AnalogKuramotoPlatform.NEUTRAL_ATOMS
    )
    program = compile_analog_kuramoto(
        np.array([[0.0, -0.3], [-0.3, 0.0]]),
        np.array([0.2, -0.1]),
        platform=platform,
        duration=1.5,
    )
    return export_provider_payload(program, provider)


def test_canonical_units_preserve_payload_and_legacy_calibration(
    analog_export: ProviderAnalogPayload,
) -> None:
    """Keep numeric exports and caller calibration unchanged while recording unit status."""
    calibration = {
        "calibration_id": "local-design-v1",
        "duration_unit": "us",
        "coupling_unit": "rad/us",
        "detuning_unit": "rad/us",
        "operator_note": "local export only",
    }
    raw_payload = json.loads(json.dumps(analog_export.payload))
    plan = prepare_provider_execution_plan(analog_export, calibration=calibration, approved=True)
    record = json.loads(json.dumps(plan.to_dict()))
    assert record["payload"] == raw_payload
    assert record["calibration"] == calibration
    assert record["unit_contract"] == "analog_execution_units.v1"
    assert record["unit_status"] == "canonical_design_rates"
    assert plan.can_execute is analog_export.sdk_available
    assert plan.can_construct_sdk_object is analog_export.sdk_available
    assert not analog_export.can_submit
    calibration["duration_unit"] = "s"
    assert plan.calibration["duration_unit"] == "us"


@pytest.mark.parametrize(
    "units",
    [
        ("design_time", "dimensionless_native_coupling", "dimensionless_detuning"),
        ("dt", "arb", "arb"),
    ],
)
def test_design_units_cannot_be_promoted_by_approval(
    analog_export: ProviderAnalogPayload, units: tuple[str, str, str]
) -> None:
    """Retain old design records but block their promotion before SDK readiness checks."""
    calibration = {
        "calibration_id": "uncalibrated-design-v1",
        "duration_unit": units[0],
        "coupling_unit": units[1],
        "detuning_unit": units[2],
    }
    plan = prepare_provider_execution_plan(analog_export, calibration=calibration, approved=True)
    assert plan.unit_status == "uncalibrated_design_units"
    assert not plan.can_execute
    assert not plan.can_construct_sdk_object
    assert plan.reason == "blocked_until_calibrated_unit_conversion"
    assert plan.to_dict()["calibration"] == calibration
    assert "design_units_require_explicit_calibrated_conversion" in plan.limitations
    blocked = prepare_provider_execution_plan(analog_export, calibration=calibration)
    assert blocked.reason == "blocked_until_explicit_execution_approval"
    assert blocked.unit_status == "uncalibrated_design_units"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_unit", "bananas"),
        ("duration_unit", "s"),
        ("duration_unit", "ns"),
        ("duration_unit", " us "),
        ("coupling_unit", "Hz"),
        ("coupling_unit", "rad/s"),
        ("detuning_unit", "MHz"),
        ("detuning_unit", "dimensionless_detuning"),
    ],
)
def test_unknown_or_mixed_units_refuse_before_plan_creation(
    analog_export: ProviderAnalogPayload, field: str, value: str
) -> None:
    """Reject unknown labels and unsupported scale conversions instead of relabelling values."""
    calibration = {
        "calibration_id": "invalid-units-v1",
        "duration_unit": "us",
        "coupling_unit": "rad/us",
        "detuning_unit": "rad/us",
    }
    calibration[field] = value
    with pytest.raises(ValueError, match="unsupported analog execution units"):
        prepare_provider_execution_plan(analog_export, calibration=calibration, approved=True)
