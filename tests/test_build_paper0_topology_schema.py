# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 topology schema generator tests
"""Tests for the Paper 0 topology-schema artefact builder."""

from __future__ import annotations

import json

from scripts.build_paper0_topology_schema import write_topology_schema_outputs


def test_write_topology_schema_outputs_records_source_boundary(tmp_path) -> None:
    outputs = write_topology_schema_outputs(output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["schema_key"] == "paper0.topology.source_boundary.v1"
    assert payload["provider_ready"] is False
    assert payload["numeric_coupling_matrix"] is None
    assert payload["hardware_status"] == "source_boundary_only_no_provider_submission"
    assert payload["layer_count"] == 16
    assert "control.ring" in payload["synthetic_controls"]
    assert "Paper 0 Topology Source Boundary" in report
    assert "No numeric coupling matrix is exported" in report
