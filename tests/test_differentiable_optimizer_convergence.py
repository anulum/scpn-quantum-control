# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Optimizer Convergence Artifact Tests
"""Tests for BL-15 optimizer convergence artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import (
    GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA,
    ground_state_optimizer_convergence_payload,
    render_ground_state_optimizer_convergence_markdown,
    write_ground_state_optimizer_convergence_artifact,
)
from scpn_quantum_control.phase import run_ground_state_optimizer_convergence_suite


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build one real BL-15 convergence payload through the public facade."""
    suite = run_ground_state_optimizer_convergence_suite()
    return ground_state_optimizer_convergence_payload(suite, artifact_id="pytest-bl15")


def test_optimizer_convergence_payload_carries_rows_and_boundary(
    payload: dict[str, Any],
) -> None:
    assert payload["schema"] == GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA
    assert payload["artifact_id"] == "pytest-bl15"
    assert payload["artifact_date"] == "2026-07-09"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["production_eligible"] is False
    assert payload["promotion_ready"] is False
    assert payload["passed"] is True
    assert len(payload["rows"]) == 10
    assert len(payload["boundary_rows"]) == 1
    assert payload["boundary_rows"][0]["failure_class"] == "unsupported_qjit_metric_fusion"


def test_optimizer_convergence_markdown_renders_evidence(
    payload: dict[str, Any],
) -> None:
    markdown = render_ground_state_optimizer_convergence_markdown(payload)

    assert "# Ground-State Optimizer Convergence Evidence" in markdown
    assert "`functional_non_isolated`" in markdown
    assert "`qng_qjit_class_boundary`" in markdown
    assert "`single_qubit_z_rotation_ground`" in markdown
    assert "`two_qubit_product_ising_ground`" in markdown


def test_optimizer_convergence_writer_creates_json_and_markdown(
    tmp_path: Path,
) -> None:
    artifact = write_ground_state_optimizer_convergence_artifact(
        tmp_path / "ground_state_optimizer_convergence.json",
        artifact_id="pytest-writer",
    )

    assert artifact.row_count == 10
    assert artifact.boundary_count == 1
    assert artifact.classification == "functional_non_isolated"
    assert artifact.to_dict()["artifact_id"] == "pytest-writer"
    payload = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    markdown = artifact.markdown_path.read_text(encoding="utf-8")
    assert payload["artifact_id"] == "pytest-writer"
    assert payload["passed"] is True
    assert "Ground-State Optimizer Convergence Evidence" in markdown


def test_optimizer_convergence_writer_rejects_invalid_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_path must end with .json"):
        write_ground_state_optimizer_convergence_artifact(tmp_path / "artifact.txt")

    with pytest.raises(ValueError, match="markdown_path must end with .md"):
        write_ground_state_optimizer_convergence_artifact(
            tmp_path / "artifact.json",
            markdown_path=tmp_path / "artifact.txt",
        )


def test_optimizer_convergence_payload_rejects_blank_artifact_id() -> None:
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        ground_state_optimizer_convergence_payload(artifact_id=" ")


def test_optimizer_convergence_payload_rejects_wrong_suite_classification() -> None:
    suite = run_ground_state_optimizer_convergence_suite()
    object.__setattr__(suite, "evidence_class", "isolated_affinity")

    with pytest.raises(ValueError, match="suite evidence_class"):
        ground_state_optimizer_convergence_payload(suite)
