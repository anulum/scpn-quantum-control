# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-System Objective Evidence Tests
"""Tests for BL-16 open-system objective artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import (
    OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA,
    open_system_objective_evidence_payload,
    render_open_system_objective_evidence_markdown,
    write_open_system_objective_evidence_artifact,
)
from scpn_quantum_control.phase import run_open_system_objective_suite


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build one real BL-16 evidence payload through the public facade."""
    suite = run_open_system_objective_suite()
    return open_system_objective_evidence_payload(suite, artifact_id="pytest-bl16")


def test_open_system_payload_carries_rows_and_boundaries(payload: dict[str, Any]) -> None:
    assert payload["schema"] == OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA
    assert payload["artifact_id"] == "pytest-bl16"
    assert payload["artifact_date"] == "2026-07-09"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["production_eligible"] is False
    assert payload["promotion_ready"] is False
    assert payload["passed"] is True
    assert len(payload["rows"]) == 4
    assert len(payload["boundary_rows"]) == 2
    assert payload["boundary_rows"][0]["failure_class"] == "unsupported_adjoint_lindblad_gradient"


def test_open_system_markdown_renders_evidence(payload: dict[str, Any]) -> None:
    markdown = render_open_system_objective_evidence_markdown(payload)

    assert "# Open-System Objective Evidence" in markdown
    assert "`functional_non_isolated`" in markdown
    assert "`two_qubit_relaxing_sync`" in markdown
    assert "`lindblad_density`" in markdown
    assert "`mcwf_ensemble`" in markdown
    assert "`hardware_open_system_gradient_boundary`" in markdown


def test_open_system_writer_creates_json_and_markdown(tmp_path: Path) -> None:
    artifact = write_open_system_objective_evidence_artifact(
        tmp_path / "open_system_objective_evidence.json",
        artifact_id="pytest-writer",
    )

    assert artifact.row_count == 4
    assert artifact.boundary_count == 2
    assert artifact.classification == "functional_non_isolated"
    assert artifact.to_dict()["artifact_id"] == "pytest-writer"
    payload = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    markdown = artifact.markdown_path.read_text(encoding="utf-8")
    assert payload["artifact_id"] == "pytest-writer"
    assert payload["passed"] is True
    assert "Open-System Objective Evidence" in markdown


def test_open_system_writer_rejects_invalid_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_path must end with .json"):
        write_open_system_objective_evidence_artifact(tmp_path / "artifact.txt")

    with pytest.raises(ValueError, match="markdown_path must end with .md"):
        write_open_system_objective_evidence_artifact(
            tmp_path / "artifact.json",
            markdown_path=tmp_path / "artifact.txt",
        )


def test_open_system_payload_rejects_blank_artifact_id() -> None:
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        open_system_objective_evidence_payload(artifact_id=" ")


def test_open_system_payload_rejects_wrong_suite_classification() -> None:
    suite = run_open_system_objective_suite()
    object.__setattr__(suite, "evidence_class", "isolated_affinity")

    with pytest.raises(ValueError, match="suite evidence_class"):
        open_system_objective_evidence_payload(suite)
