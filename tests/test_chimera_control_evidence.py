# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera evidence tests
"""Measured-regime, rendering, and deterministic custody tests for chimera-control."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
import pytest

from scpn_quantum_control.chimera_control.evidence import (
    CHIMERA_CONTROL_EVIDENCE_DATE,
    CHIMERA_CONTROL_EVIDENCE_SCHEMA,
    ChimeraMultiscaleEvidence,
    ChimeraSupportRow,
    SyntheticRegimeEvidence,
    build_chimera_multiscale_evidence,
    render_chimera_multiscale_markdown,
    write_chimera_multiscale_evidence,
)
from scpn_quantum_control.chimera_control.schema import SyntheticRegime


class RegimeOverrides(TypedDict, total=False):
    """Type-preserving overrides for constructor contract tests."""

    regime: SyntheticRegime
    trajectory_digest: str
    trajectory_samples: int
    population_mean: tuple[float, float]
    population_min: tuple[float, float]
    population_max: tuple[float, float]
    population_std: tuple[float, float]
    chimera_index: float
    metastability_index: float
    community_metastability: float
    global_order_mean: float
    objective_before: float
    objective_after: float
    proposal_step_size: float
    proposal_accepted: bool


def _regime_evidence(**overrides: Unpack[RegimeOverrides]) -> SyntheticRegimeEvidence:
    """Construct real evidence with exactly typed field overrides."""
    values: RegimeOverrides = {
        "regime": SyntheticRegime.CHIMERA_TRANSIENT,
        "trajectory_digest": "a" * 64,
        "trajectory_samples": 2,
        "population_mean": (1.0, 0.5),
        "population_min": (1.0, 0.1),
        "population_max": (1.0, 0.9),
        "population_std": (0.0, 0.2),
        "chimera_index": 0.1,
        "metastability_index": 0.1,
        "community_metastability": 0.1,
        "global_order_mean": 0.7,
        "objective_before": 0.2,
        "objective_after": 0.1,
        "proposal_step_size": 0.25,
        "proposal_accepted": True,
    }
    values.update(overrides)
    return SyntheticRegimeEvidence(**values)


@pytest.fixture(scope="module")
def frozen_evidence() -> ChimeraMultiscaleEvidence:
    """Build the committed 64-per-population evidence once."""
    return build_chimera_multiscale_evidence()


def test_frozen_regimes_separate_chimera_transient_from_sync_control(
    frozen_evidence: ChimeraMultiscaleEvidence,
) -> None:
    """Distinguish the frozen chimera transient from its synchronised control."""
    chimera = frozen_evidence.chimera
    synchronised = frozen_evidence.synchronised_control

    assert frozen_evidence.schema_version == CHIMERA_CONTROL_EVIDENCE_SCHEMA
    assert frozen_evidence.generated_on == CHIMERA_CONTROL_EVIDENCE_DATE
    assert frozen_evidence.population_size == 64
    assert chimera.population_min[0] > 0.99
    assert chimera.population_mean[1] < 0.75
    assert chimera.population_min[1] < 0.2
    assert chimera.population_std[1] > 0.15
    assert chimera.chimera_index > 0.05
    assert synchronised.population_mean[0] > 0.99
    assert synchronised.population_mean[1] > 0.95
    assert synchronised.population_std[1] < 0.05
    assert synchronised.chimera_index < 0.001
    assert chimera.proposal_accepted and synchronised.proposal_accepted
    assert chimera.objective_after < chimera.objective_before
    assert synchronised.objective_after < synchronised.objective_before


def test_frozen_gradient_projection_and_scope_rows_are_fail_closed(
    frozen_evidence: ChimeraMultiscaleEvidence,
) -> None:
    """Require bounded gradients, projected topology, and fail-closed support."""
    assert frozen_evidence.gradient_max_abs_error < 1.0e-8
    assert frozen_evidence.topology_violation_before > 100.0
    assert frozen_evidence.topology_violation_after < 1.0e-10
    statuses = {row.capability: row.status for row in frozen_evidence.support}
    assert statuses["optional challenge-registry extension"] == "descoped"
    assert statuses["topology-constraint interaction"] == "bounded"
    assert len(frozen_evidence.content_digest) == 64


def test_evidence_serialisation_and_markdown_expose_exact_claim_boundary(
    frozen_evidence: ChimeraMultiscaleEvidence,
) -> None:
    """Expose deterministic custody and non-claims in both evidence formats."""
    payload = frozen_evidence.to_dict()
    markdown = render_chimera_multiscale_markdown(frozen_evidence)

    assert payload["content_digest"] == frozen_evidence.content_digest
    assert payload["chimera"] == frozen_evidence.chimera.to_dict()
    assert "finite-N synthetic" in str(payload["claim_boundary"])
    assert "# Chimera and Multiscale Control Evidence" in markdown
    assert frozen_evidence.content_digest in markdown
    assert "not an attractor, generalisation, or physical-domain claim" in markdown
    assert markdown.endswith("\n")


def test_writer_round_trips_and_fails_closed_on_drift(
    tmp_path: Path,
    frozen_evidence: ChimeraMultiscaleEvidence,
) -> None:
    """Write reproducible evidence and fail closed on missing or changed bytes."""
    json_path = tmp_path / "nested/evidence.json"
    markdown_path = tmp_path / "nested/evidence.md"
    written = write_chimera_multiscale_evidence(
        frozen_evidence,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert written == (json_path, markdown_path)
    assert (
        write_chimera_multiscale_evidence(
            frozen_evidence,
            json_path=json_path,
            markdown_path=markdown_path,
            check=True,
        )
        == written
    )
    json_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence drift"):
        write_chimera_multiscale_evidence(
            frozen_evidence,
            json_path=json_path,
            markdown_path=markdown_path,
            check=True,
        )
    json_path.unlink()
    with pytest.raises(RuntimeError, match="evidence drift"):
        write_chimera_multiscale_evidence(
            frozen_evidence,
            json_path=json_path,
            markdown_path=markdown_path,
            check=True,
        )


def test_support_row_contract_rejects_invalid_status_and_blank_fields() -> None:
    """Reject unsupported status values and blank support-row fields."""
    with pytest.raises(ValueError, match="status"):
        # Deliberately violate the status Literal; the runtime guard must reject it.
        ChimeraSupportRow("cap", "invalid", "evidence", "non-claim")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capability"):
        ChimeraSupportRow(" ", "supported", "evidence", "non-claim")
    with pytest.raises(ValueError, match="evidence"):
        ChimeraSupportRow("cap", "supported", " ", "non-claim")
    with pytest.raises(ValueError, match="non_claim"):
        ChimeraSupportRow("cap", "supported", "evidence", " ")


def test_regime_evidence_contract_rejects_invalid_custody() -> None:
    """Reject malformed regime metrics, digests, and objective claims."""
    assert _regime_evidence().trajectory_samples == 2
    with pytest.raises(ValueError, match="trajectory_digest"):
        _regime_evidence(trajectory_digest="bad")
    with pytest.raises(ValueError, match="trajectory_digest"):
        _regime_evidence(trajectory_digest="z" * 64)
    with pytest.raises(ValueError, match="trajectory_samples"):
        _regime_evidence(trajectory_samples=True)
    with pytest.raises(ValueError, match="trajectory_samples"):
        # Deliberate fractional sample count violates the integer parameter.
        _regime_evidence(trajectory_samples=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="population_mean"):
        # Deliberate one-element tuple violates the two-population contract.
        _regime_evidence(population_mean=(1.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="population_min"):
        _regime_evidence(population_min=(1.0, np.nan))
    with pytest.raises(ValueError, match=r"population_mean values must lie in \[0, 1\]"):
        _regime_evidence(population_mean=(1.1, 0.5))
    with pytest.raises(ValueError, match=r"population_min values must lie in \[0, 1\]"):
        _regime_evidence(population_min=(-0.1, 0.1))
    with pytest.raises(ValueError, match=r"population_max values must lie in \[0, 1\]"):
        _regime_evidence(population_max=(1.1, 0.9))
    with pytest.raises(ValueError, match="population_std values"):
        _regime_evidence(population_std=(-0.1, 0.2))
    with pytest.raises(ValueError, match="min <= mean <= max"):
        _regime_evidence(population_mean=(0.5, 0.5))
    with pytest.raises(ValueError, match=r"global_order_mean must lie in \[0, 1\]"):
        _regime_evidence(global_order_mean=1.1)
    with pytest.raises(ValueError, match="strict objective decrease"):
        _regime_evidence(objective_after=0.2)
    with pytest.raises(ValueError, match="objective_after"):
        _regime_evidence(objective_after=-1.0)


def test_complete_evidence_contract_rejects_invalid_top_level_fields(
    frozen_evidence: ChimeraMultiscaleEvidence,
) -> None:
    """Reject invalid top-level schema, metric, support, and digest fields."""
    # Dataclass replacement invokes the real constructor and __post_init__;
    # all fields below have correct Python types but violate value invariants.
    assert replace(frozen_evidence) == frozen_evidence
    with pytest.raises(ValueError, match="schema_version"):
        replace(frozen_evidence, schema_version="bad")
    with pytest.raises(ValueError, match="generated_on"):
        replace(frozen_evidence, generated_on=" ")
    with pytest.raises(ValueError, match="population_size"):
        replace(frozen_evidence, population_size=1)
    with pytest.raises(ValueError, match="gradient_max_abs_error"):
        replace(frozen_evidence, gradient_max_abs_error=-1.0)
    with pytest.raises(ValueError, match="topology_digest"):
        replace(frozen_evidence, topology_digest="bad")
    with pytest.raises(ValueError, match="support"):
        replace(frozen_evidence, support=())
    with pytest.raises(ValueError, match="claim_boundary"):
        replace(frozen_evidence, claim_boundary=" ")
    with pytest.raises(ValueError, match="content_digest"):
        replace(frozen_evidence, content_digest="bad")
