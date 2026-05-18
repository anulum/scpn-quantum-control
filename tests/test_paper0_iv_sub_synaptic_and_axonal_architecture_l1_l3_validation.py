# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) validation tests
"""Tests for Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iv_sub_synaptic_and_axonal_architecture_l1_l3_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IvSubSynapticAndAxonalArchitectureL1L3Config,
    classify_iv_sub_synaptic_and_axonal_architecture_l1_l3_component,
    iv_sub_synaptic_and_axonal_architecture_l1_l3_labels,
    validate_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture,
)


def test_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture_preserves_source_boundary() -> None:
    result = validate_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture()
    assert result.source_ledger_span == ("P0R04786", "P0R04793")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04794"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iv_sub_synaptic_and_axonal_architecture_l1_l3_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04786"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04793"


def test_iv_sub_synaptic_and_axonal_architecture_l1_l3_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iv_sub_synaptic_and_axonal_architecture_l1_l3",
        "1_the_post_synaptic_density_psd",
        "2_axonal_structure_and_transport",
        "v_the_deep_quantum_milieu_l1",
    ):
        assert (
            classify_iv_sub_synaptic_and_axonal_architecture_l1_l3_component(component)
            == f"{component}_source_boundary"
        )
    labels = iv_sub_synaptic_and_axonal_architecture_l1_l3_labels()
    assert labels["section"] == "IV. Sub-Synaptic and Axonal Architecture (L1-L3)"
    assert labels["next_boundary"] == "P0R04794"


def test_iv_sub_synaptic_and_axonal_architecture_l1_l3_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IvSubSynapticAndAxonalArchitectureL1L3Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        IvSubSynapticAndAxonalArchitectureL1L3Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04794"):
        IvSubSynapticAndAxonalArchitectureL1L3Config(next_source_boundary="P0R04793")
    with pytest.raises(
        ValueError, match="unknown iv_sub_synaptic_and_axonal_architecture_l1_l3 component"
    ):
        classify_iv_sub_synaptic_and_axonal_architecture_l1_l3_component(
            "empirical_validation_claim"
        )
