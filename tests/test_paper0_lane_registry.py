# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Paper 0 lane registry
"""Focused tests for the Paper 0 lane registry surface."""

from __future__ import annotations

import json

from scpn_quantum_control.paper0.lane_registry import (
    HARDWARE_BOUNDARY_LABEL,
    PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY,
    PAPER0_LANE_REGISTRY_SCHEMA,
    Paper0LaneRecord,
    paper0_lane_registry_json,
    paper0_lane_registry_lanes,
    paper0_lane_registry_markdown,
    paper0_lane_registry_payload,
)
from scpn_quantum_control.paper0.spec_loader import DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE


def test_registry_lanes_are_deterministic_and_ordered() -> None:
    lanes_first = paper0_lane_registry_lanes()
    lanes_second = paper0_lane_registry_lanes()

    lane_ids_first = [lane.lane_id for lane in lanes_first]
    lane_ids_second = [lane.lane_id for lane in lanes_second]

    assert lane_ids_first == lane_ids_second
    assert lane_ids_first == sorted(lane_ids_first)
    assert all(isinstance(lane, Paper0LaneRecord) for lane in lanes_first)


def test_registry_payload_is_json_serialisable_and_complete() -> None:
    payload = paper0_lane_registry_payload()
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["schema"] == PAPER0_LANE_REGISTRY_SCHEMA
    assert decoded["claim_boundary"] == PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY
    assert decoded["lane_count"] == len(decoded["lanes"])
    assert isinstance(decoded["lanes"], list)
    for entry in decoded["lanes"]:
        assert isinstance(entry["lane_id"], str)
        assert isinstance(entry["blockers"], list)
        assert isinstance(entry["source_ledger_span"], list)
        assert len(entry["source_ledger_span"]) == 2


def test_registry_enforces_no_qpu_no_hardware_boundaries() -> None:
    lanes = paper0_lane_registry_lanes()

    assert all(lane.no_qpu_submission for lane in lanes)
    assert all(lane.no_hardware_submission for lane in lanes)
    assert all(lane.hardware_boundary == HARDWARE_BOUNDARY_LABEL for lane in lanes)
    claim_boundary = PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY.lower()
    assert "no qpu" in claim_boundary
    assert "no hardware" in claim_boundary


def test_front_matter_lane_is_linked_to_spec_loader_constants() -> None:
    lanes = paper0_lane_registry_lanes()

    front_matter_lanes = [lane for lane in lanes if lane.lane_id == "P0L01_front_matter_context"]
    assert front_matter_lanes

    lane = front_matter_lanes[0]
    assert lane.related_spec_bundle_path == DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE
    assert lane.related_spec_key == "front_matter_context.collection_identity"


def test_registry_text_renderers_are_deterministic() -> None:
    payload = paper0_lane_registry_payload()

    assert paper0_lane_registry_json(payload) == paper0_lane_registry_json(payload)
    markdown = paper0_lane_registry_markdown(payload)

    assert "# Paper 0 Lane Registry" in markdown
    assert "scpn-bench paper0-lane-registry-gate" in markdown
    assert PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY in markdown
