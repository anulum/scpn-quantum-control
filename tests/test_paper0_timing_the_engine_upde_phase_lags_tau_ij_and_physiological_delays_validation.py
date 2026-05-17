# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays validation tests
"""Tests for Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig,
    classify_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_component,
    timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_labels,
    validate_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture,
)


def test_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture()
    assert result.source_ledger_span == ("P0R02223", "P0R02236")
    assert result.source_record_count == 14
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02237"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02223"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02236"


def test_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",):
        assert (
            classify_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_labels()
    assert (
        labels["section"]
        == "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays"
    )
    assert labels["next_boundary"] == "P0R02237"


def test_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig(
            expected_source_record_count=13
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02237"):
        TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig(
            next_source_boundary="P0R02236"
        )
    with pytest.raises(
        ValueError,
        match="unknown timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays component",
    ):
        classify_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_component(
            "empirical_validation_claim"
        )
