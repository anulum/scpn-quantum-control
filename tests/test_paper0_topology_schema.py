# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 topology schema tests
"""Tests for the source-anchored Paper 0 topology schema."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.topology_schema import (
    Paper0SyntheticControl,
    Paper0TopologyValidationError,
    build_paper0_topology_schema,
    schema_to_s19_source_boundary,
    validate_paper0_topology_schema,
)


def test_default_schema_is_source_anchored_without_numeric_matrix_claim() -> None:
    schema = build_paper0_topology_schema()

    assert schema.provenance_status == "paper0_source_anchored_no_numeric_matrix"
    assert len(schema.layers) == 16
    assert schema.layers[0].layer_id == 1
    assert schema.layers[-1].layer_id == 16
    assert schema.layers[-1].is_meta_layer
    assert schema.layers[-1].source_ledger_ids == ("P0R00049", "P0R00050", "P0R00574")
    assert schema.numeric_coupling_matrix is None
    assert schema.provider_ready is False


def test_schema_keeps_coupling_ports_and_controls_separate() -> None:
    schema = build_paper0_topology_schema()

    channels = {edge.channel for edge in schema.inter_layer_edges}
    assert channels == {"downward_generation", "upward_inference", "recursive_closure"}
    assert all(
        edge.source_equation_ids or edge.source_ledger_ids for edge in schema.inter_layer_edges
    )

    field = schema.field_ports["global_psi_field"]
    assert field.source_equation_ids == ("EQ0034", "EQ0041", "EQ0043")
    assert field.couples_to_layers == tuple(range(1, 16))

    adaptive = schema.adaptive_parameters["quasicritical_controller"]
    assert adaptive.source_equation_ids == ("EQ0045",)
    assert set(adaptive.variables) >= {"gamma_L", "lambda_L", "alpha_L", "sigma_L"}

    controls = {control.key: control for control in schema.synthetic_controls}
    assert set(controls) == {"control.chain", "control.ring", "control.complete"}
    assert all(control.is_control for control in controls.values())
    assert all(
        control.provenance_status == "synthetic_control_not_paper0_topology"
        for control in controls.values()
    )


def test_schema_validation_rejects_unanchored_edges_and_mislabelled_controls() -> None:
    schema = build_paper0_topology_schema()
    bad_edge = schema.inter_layer_edges[0].__class__(
        source_layer=1,
        target_layer=2,
        channel="downward_generation",
        source_equation_ids=(),
        source_ledger_ids=(),
        description="unanchored edge",
    )
    invalid = schema.__class__(
        layers=schema.layers,
        intra_layer_topologies=schema.intra_layer_topologies,
        inter_layer_edges=(bad_edge, *schema.inter_layer_edges[1:]),
        field_ports=schema.field_ports,
        adaptive_parameters=schema.adaptive_parameters,
        synthetic_controls=schema.synthetic_controls,
        numeric_coupling_matrix=None,
        provenance_status=schema.provenance_status,
        provider_ready=False,
    )

    with pytest.raises(Paper0TopologyValidationError, match="source anchor"):
        validate_paper0_topology_schema(invalid)

    mislabelled_control = Paper0SyntheticControl(
        key="control.bad",
        topology="chain",
        is_control=False,
        provenance_status="paper0_source_anchored",
        purpose="bad control",
    )
    invalid_control_schema = schema.__class__(
        layers=schema.layers,
        intra_layer_topologies=schema.intra_layer_topologies,
        inter_layer_edges=schema.inter_layer_edges,
        field_ports=schema.field_ports,
        adaptive_parameters=schema.adaptive_parameters,
        synthetic_controls=(mislabelled_control,),
        numeric_coupling_matrix=None,
        provenance_status=schema.provenance_status,
        provider_ready=False,
    )
    with pytest.raises(Paper0TopologyValidationError, match="synthetic control"):
        validate_paper0_topology_schema(invalid_control_schema)


def test_s19_source_boundary_exports_provenance_not_provider_payload() -> None:
    boundary = schema_to_s19_source_boundary(build_paper0_topology_schema())

    assert boundary["schema_key"] == "paper0.topology.source_boundary.v1"
    assert boundary["provider_ready"] is False
    assert boundary["numeric_coupling_matrix"] is None
    assert boundary["hardware_status"] == "source_boundary_only_no_provider_submission"
    assert boundary["layer_count"] == 16
    assert set(boundary["synthetic_controls"]) == {
        "control.chain",
        "control.ring",
        "control.complete",
    }
    assert "P0R02559" in boundary["source_ledger_ids"]
    assert "EQ0045" in boundary["source_equation_ids"]
