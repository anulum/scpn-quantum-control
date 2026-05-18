# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Intrinsic Dynamics (iL): validation tests
"""Tests for Paper 0 Intrinsic Dynamics (iL): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.intrinsic_dynamics_il_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IntrinsicDynamicsIlConfig,
    classify_intrinsic_dynamics_il_component,
    intrinsic_dynamics_il_labels,
    validate_intrinsic_dynamics_il_fixture,
)


def test_intrinsic_dynamics_il_fixture_preserves_source_boundary() -> None:
    result = validate_intrinsic_dynamics_il_fixture()
    assert result.source_ledger_span == ("P0R02624", "P0R02631")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02632"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_intrinsic_dynamics_il_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02624"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02631"


def test_intrinsic_dynamics_il_classification_and_labels_are_explicit() -> None:
    for component in (
        "intrinsic_dynamics_il",
        "intra_layer_coupling_kijl",
        "inter_layer_coupling_cinterlayer",
    ):
        assert (
            classify_intrinsic_dynamics_il_component(component) == f"{component}_source_boundary"
        )
    labels = intrinsic_dynamics_il_labels()
    assert labels["section"] == "Intrinsic Dynamics (iL):"
    assert labels["next_boundary"] == "P0R02632"


def test_intrinsic_dynamics_il_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IntrinsicDynamicsIlConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IntrinsicDynamicsIlConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02632"):
        IntrinsicDynamicsIlConfig(next_source_boundary="P0R02631")
    with pytest.raises(ValueError, match="unknown intrinsic_dynamics_il component"):
        classify_intrinsic_dynamics_il_component("empirical_validation_claim")
