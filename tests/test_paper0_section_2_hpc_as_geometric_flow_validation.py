# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. HPC as Geometric Flow: validation tests
"""Tests for Paper 0 2. HPC as Geometric Flow: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_hpc_as_geometric_flow_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2HpcAsGeometricFlowConfig,
    classify_section_2_hpc_as_geometric_flow_component,
    section_2_hpc_as_geometric_flow_labels,
    validate_section_2_hpc_as_geometric_flow_fixture,
)


def test_section_2_hpc_as_geometric_flow_fixture_preserves_source_boundary() -> None:
    result = validate_section_2_hpc_as_geometric_flow_fixture()
    assert result.source_ledger_span == ("P0R04849", "P0R04857")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04858"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_hpc_as_geometric_flow_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04849"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04857"


def test_section_2_hpc_as_geometric_flow_classification_and_labels_are_explicit() -> None:
    for component in (
        "2_hpc_as_geometric_flow",
        "3_the_geometry_of_the_self_the_strange_loop_and_the_dmn",
        "vi_synthesis_the_brain_as_a_geometric_transducer",
        "the_integrative_physiology_of_the_scpn_the_embodied_brain",
    ):
        assert (
            classify_section_2_hpc_as_geometric_flow_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_hpc_as_geometric_flow_labels()
    assert labels["section"] == "2. HPC as Geometric Flow:"
    assert labels["next_boundary"] == "P0R04858"


def test_section_2_hpc_as_geometric_flow_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section2HpcAsGeometricFlowConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section2HpcAsGeometricFlowConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04858"):
        Section2HpcAsGeometricFlowConfig(next_source_boundary="P0R04857")
    with pytest.raises(ValueError, match="unknown section_2_hpc_as_geometric_flow component"):
        classify_section_2_hpc_as_geometric_flow_component("empirical_validation_claim")
