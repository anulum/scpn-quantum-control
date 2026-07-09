# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Witness Evidence Tests
"""Tests for BL-18 synchronisation-witness artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import (
    SYNC_WITNESS_EVIDENCE_SCHEMA,
    render_sync_witness_evidence_markdown,
    sync_witness_evidence_payload,
    write_sync_witness_evidence_artifact,
)
from scpn_quantum_control.phase import run_sync_witness_suite


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build one real BL-18 evidence payload through the public facade."""
    suite = run_sync_witness_suite()
    return sync_witness_evidence_payload(suite, artifact_id="pytest-bl18")


def test_sync_witness_payload_carries_rows_and_boundaries(payload: dict[str, Any]) -> None:
    assert payload["schema"] == SYNC_WITNESS_EVIDENCE_SCHEMA
    assert payload["artifact_id"] == "pytest-bl18"
    assert payload["artifact_date"] == "2026-07-09"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["production_eligible"] is False
    assert payload["promotion_ready"] is False
    assert payload["passed"] is True
    assert len(payload["rows"]) == 3
    assert len(payload["boundary_rows"]) == 2
    assert payload["boundary_rows"][0]["boundary_id"] == "high_dimensional_manifold_boundary"


def test_sync_witness_markdown_renders_evidence(payload: dict[str, Any]) -> None:
    markdown = render_sync_witness_evidence_markdown(payload)

    assert "# Synchronisation-Witness Evidence" in markdown
    assert "`functional_non_isolated`" in markdown
    assert "`synchronised_eight_node`" in markdown
    assert "`desynchronised_eight_node`" in markdown
    assert "`hardware_phase_tomography_boundary`" in markdown


def test_sync_witness_writer_creates_json_and_markdown(tmp_path: Path) -> None:
    artifact = write_sync_witness_evidence_artifact(
        tmp_path / "sync_witness_evidence.json",
        artifact_id="pytest-writer",
    )

    assert artifact.row_count == 3
    assert artifact.boundary_count == 2
    assert artifact.classification == "functional_non_isolated"
    assert artifact.to_dict()["artifact_id"] == "pytest-writer"
    payload = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    markdown = artifact.markdown_path.read_text(encoding="utf-8")
    assert payload["artifact_id"] == "pytest-writer"
    assert payload["passed"] is True
    assert "Synchronisation-Witness Evidence" in markdown


def test_sync_witness_writer_rejects_invalid_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_path must end with .json"):
        write_sync_witness_evidence_artifact(tmp_path / "artifact.txt")

    with pytest.raises(ValueError, match="markdown_path must end with .md"):
        write_sync_witness_evidence_artifact(
            tmp_path / "artifact.json",
            markdown_path=tmp_path / "artifact.txt",
        )


def test_sync_witness_payload_rejects_blank_artifact_id() -> None:
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        sync_witness_evidence_payload(artifact_id=" ")


def test_sync_witness_payload_rejects_wrong_suite_classification() -> None:
    suite = run_sync_witness_suite()
    object.__setattr__(suite, "evidence_class", "isolated_affinity")

    with pytest.raises(ValueError, match="suite evidence_class"):
        sync_witness_evidence_payload(suite)
