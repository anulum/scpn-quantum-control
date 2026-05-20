# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S10 analog-native readiness tests
"""Tests for the S10 analog-native Kuramoto readiness gate."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.analog_kuramoto import AnalogProviderTarget
from scpn_quantum_control.hardware.analog_native_readiness import (
    ANALOG_NATIVE_SCHEMA,
    AnalogNativeReadinessConfig,
    analog_native_markdown,
    analog_native_payload,
    compare_native_to_digital_primitives,
    provider_readiness_rows,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    K_nm = np.array(
        [
            [0.0, 0.50, -0.25, 0.125],
            [0.50, 0.0, 0.375, 0.0],
            [-0.25, 0.375, 0.0, -0.125],
            [0.125, 0.0, -0.125, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.05, -0.10, 0.20, -0.15], dtype=np.float64)
    return K_nm, omega


def test_native_primitive_comparison_blocks_analog_advantage_claims() -> None:
    K_nm, omega = _inputs()
    comparison = compare_native_to_digital_primitives(
        K_nm,
        omega,
        config=AnalogNativeReadinessConfig(duration=1.5, trotter_steps=8),
    )

    assert comparison.schema == "s10_analog_native_primitive_comparison_v1"
    assert comparison.digital_two_qubit_gate_count == 80
    assert comparison.native_coupler_count == 5
    assert comparison.native_to_digital_ratio < 1.0
    assert comparison.fixed_tolerance == pytest.approx(0.02)
    assert comparison.hardware_submission_allowed is False
    assert comparison.analog_advantage_claim_allowed is False
    assert "digital Trotter" in comparison.falsifier


def test_provider_rows_cover_targets_without_submission() -> None:
    K_nm, omega = _inputs()
    rows = provider_readiness_rows(K_nm, omega)
    by_provider = {row.provider: row for row in rows}

    assert set(by_provider) == {
        AnalogProviderTarget.PULSER,
        AnalogProviderTarget.BLOQADE,
        AnalogProviderTarget.IBM_PULSE,
    }
    assert by_provider[AnalogProviderTarget.PULSER].program_platform == "neutral_atoms"
    assert by_provider[AnalogProviderTarget.IBM_PULSE].program_platform == "circuit_qed"
    assert all(row.can_submit is False for row in rows)
    assert all(row.can_execute is False for row in rows)
    assert all("execution_plan_only_no_provider_contact" in row.limitations for row in rows)


def test_analog_native_payload_records_boundary_and_falsifier() -> None:
    payload = analog_native_payload()

    assert payload["schema"] == ANALOG_NATIVE_SCHEMA
    assert payload["hardware_submission_allowed"] is False
    assert payload["analog_advantage_claim_allowed"] is False
    assert payload["primitive_comparison"]["native_to_digital_ratio"] < 1.0
    assert len(payload["provider_readiness"]) == 3
    assert "same declared tolerance" in payload["falsifier"]


def test_analog_native_markdown_records_gate_and_provider_table() -> None:
    markdown = analog_native_markdown(analog_native_payload())

    assert "scpn-bench s10-analog-native-readiness" in markdown
    assert "| provider | platform | sdk available | can execute |" in markdown
    assert "analog advantage claim allowed: `False`" in markdown


def test_readiness_config_and_inputs_fail_closed() -> None:
    K_nm, omega = _inputs()
    with pytest.raises(ValueError, match="trotter_steps"):
        AnalogNativeReadinessConfig(trotter_steps=0)
    with pytest.raises(ValueError, match="duration"):
        AnalogNativeReadinessConfig(duration=0.0)
    with pytest.raises(ValueError, match="square"):
        compare_native_to_digital_primitives(K_nm[:2], omega)
    with pytest.raises(ValueError, match="omega length"):
        compare_native_to_digital_primitives(K_nm, omega[:2])
