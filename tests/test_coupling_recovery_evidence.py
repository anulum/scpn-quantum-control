# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coupling-Recovery Evidence Tests
"""Tests for BL-17 coupling-recovery artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import (
    COUPLING_RECOVERY_EVIDENCE_SCHEMA,
    coupling_recovery_evidence_payload,
    render_coupling_recovery_evidence_markdown,
    write_coupling_recovery_evidence_artifact,
)
from scpn_quantum_control.phase import run_coupling_recovery_suite


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build one real BL-17 evidence payload through the public facade."""
    suite = run_coupling_recovery_suite()
    return coupling_recovery_evidence_payload(suite, artifact_id="pytest-bl17")


def test_coupling_recovery_payload_carries_rows_and_boundaries(
    payload: dict[str, Any],
) -> None:
    assert payload["schema"] == COUPLING_RECOVERY_EVIDENCE_SCHEMA
    assert payload["artifact_id"] == "pytest-bl17"
    assert payload["artifact_date"] == "2026-07-09"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["production_eligible"] is False
    assert payload["promotion_ready"] is False
    assert payload["passed"] is True
    assert len(payload["rows"]) == 3
    assert len(payload["boundary_rows"]) == 2
    assert payload["boundary_rows"][0]["boundary_id"] == "partial_observation_inference_boundary"


def test_coupling_recovery_markdown_renders_evidence(payload: dict[str, Any]) -> None:
    markdown = render_coupling_recovery_evidence_markdown(payload)

    assert "# Coupling-Recovery Evidence" in markdown
    assert "`functional_non_isolated`" in markdown
    assert "`kuramoto_clean_three_node`" in markdown
    assert "`xy_pair_energy`" in markdown
    assert "`hardware_hamiltonian_learning_boundary`" in markdown


def test_coupling_recovery_writer_creates_json_and_markdown(tmp_path: Path) -> None:
    artifact = write_coupling_recovery_evidence_artifact(
        tmp_path / "coupling_recovery_evidence.json",
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
    assert "Coupling-Recovery Evidence" in markdown


def test_coupling_recovery_writer_rejects_invalid_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_path must end with .json"):
        write_coupling_recovery_evidence_artifact(tmp_path / "artifact.txt")

    with pytest.raises(ValueError, match="markdown_path must end with .md"):
        write_coupling_recovery_evidence_artifact(
            tmp_path / "artifact.json",
            markdown_path=tmp_path / "artifact.txt",
        )


def test_coupling_recovery_payload_rejects_blank_artifact_id() -> None:
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        coupling_recovery_evidence_payload(artifact_id=" ")


def test_coupling_recovery_payload_rejects_wrong_suite_classification() -> None:
    suite = run_coupling_recovery_suite()
    object.__setattr__(suite, "evidence_class", "isolated_affinity")

    with pytest.raises(ValueError, match="suite evidence_class"):
        coupling_recovery_evidence_payload(suite)
