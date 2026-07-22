# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable environment evidence tests.
"""Tests for version-pin and execution-route environment evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableDependencyEnvironmentEvidence,
    build_differentiable_dependency_environment_evidence,
)
from scpn_quantum_control.differentiable_dependency_environment_evidence import (
    DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY,
    REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLASSIFICATIONS,
    REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS,
    DifferentiableDependencyEnvironmentEvidenceCategory,
    DifferentiableDependencyEnvironmentEvidenceStatus,
)


def _evidence_record(
    *,
    evidence_id: object = "jax_cpu",
    title: object = "JAX CPU overlay",
    category: object = "toolchain",
    classification: object = "locked_versions",
    version_pins: object = ("jax==0.10.1",),
    evidence_paths: object = ("requirements.txt",),
    evidence_sha256: object = ("0" * 64,),
    evidence_status: object = "locked",
    blockers: object = (),
    claim_boundary: object = (DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY),
) -> DifferentiableDependencyEnvironmentEvidence:
    """Construct one evidence row while permitting malformed test values."""
    return DifferentiableDependencyEnvironmentEvidence(
        evidence_id=cast(str, evidence_id),
        title=cast(str, title),
        category=cast(DifferentiableDependencyEnvironmentEvidenceCategory, category),
        classification=cast(str, classification),
        version_pins=cast(tuple[str, ...], version_pins),
        evidence_paths=cast(tuple[str, ...], evidence_paths),
        evidence_sha256=cast(tuple[str, ...], evidence_sha256),
        evidence_status=cast(
            DifferentiableDependencyEnvironmentEvidenceStatus,
            evidence_status,
        ),
        blockers=cast(tuple[str, ...], blockers),
        claim_boundary=cast(str, claim_boundary),
    )


def test_dependency_environment_evidence_inventory_is_complete_and_bound() -> None:
    """Every required toolchain and route must bind current source-file digests."""
    records = build_differentiable_dependency_environment_evidence()

    assert tuple(record.evidence_id for record in records) == (
        REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS
    )
    assert all(
        record.classification
        == REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLASSIFICATIONS[record.evidence_id]
        for record in records
    )
    for record in records:
        for path, digest in zip(record.evidence_paths, record.evidence_sha256, strict=True):
            assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()

    by_id = {record.evidence_id: record for record in records}
    assert by_id["local_cpu"].environment_ready
    assert by_id["gpu_overlay"].classification == "declared_unlocked"
    assert not by_id["gpu_overlay"].environment_ready
    assert by_id["gtx1060_workstation"].classification == "unsupported_local_hardware"
    assert by_id["jarvislabs_cloud"].classification == "cloud_only"
    assert by_id["ml350_isolated"].classification == "isolated_host_only"
    assert by_id["ml350_isolated"].evidence_paths == (
        "docs/differentiable_reviewer_evidence.md",
        "data/differentiable_phase_qnode/ml350_full_framework_catalyst_baseline_20260705/phase_qnode_affinity_validation_isolated_required.json",
    )
    assert all(
        not path.startswith("docs/internal/")
        for record in records
        for path in record.evidence_paths
    )
    assert "DifferentiableDependencyEnvironmentEvidence" in scpn.__all__
    assert (
        scpn.build_differentiable_dependency_environment_evidence
        is build_differentiable_dependency_environment_evidence
    )


def test_dependency_environment_evidence_serializes_locked_and_blocked_rows() -> None:
    """JSON-ready rows must preserve versions, digests, blockers, and readiness."""
    locked = _evidence_record()
    blocked = _evidence_record(
        evidence_id="gpu_overlay",
        classification="declared_unlocked",
        evidence_status="hard_gap",
        blockers=("missing exact GPU lock",),
    )

    assert locked.to_dict()["environment_ready"] is True
    assert blocked.to_dict() == {
        "evidence_id": "gpu_overlay",
        "title": "JAX CPU overlay",
        "category": "toolchain",
        "classification": "declared_unlocked",
        "version_pins": ["jax==0.10.1"],
        "evidence_paths": ["requirements.txt"],
        "evidence_sha256": ["0" * 64],
        "evidence_status": "hard_gap",
        "blockers": ["missing exact GPU lock"],
        "environment_ready": False,
        "claim_boundary": DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY,
    }


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(lambda: _evidence_record(evidence_id=1), "evidence_id must", id="id-type"),
        pytest.param(lambda: _evidence_record(title=" "), "title must", id="blank-title"),
        pytest.param(
            lambda: _evidence_record(classification=""),
            "classification must",
            id="blank-classification",
        ),
        pytest.param(
            lambda: _evidence_record(category="unknown"),
            "category must be toolchain or execution_route",
            id="unknown-category",
        ),
        pytest.param(
            lambda: _evidence_record(version_pins=[]),
            "version_pins must be a tuple",
            id="pin-list",
        ),
        pytest.param(
            lambda: _evidence_record(version_pins=()),
            "version_pins must be non-empty",
            id="empty-toolchain-pins",
        ),
        pytest.param(
            lambda: _evidence_record(version_pins=("jax==1", "jax==1")),
            "version_pins must contain unique entries",
            id="duplicate-pins",
        ),
        pytest.param(
            lambda: _evidence_record(evidence_paths=(" ",)),
            "evidence_paths must contain non-empty strings",
            id="blank-path",
        ),
        pytest.param(
            lambda: _evidence_record(evidence_sha256=[]),
            "evidence_sha256 must be a tuple",
            id="digest-list",
        ),
        pytest.param(
            lambda: _evidence_record(evidence_sha256=("0" * 64, "1" * 64)),
            "evidence_sha256 must align",
            id="digest-count",
        ),
        pytest.param(
            lambda: _evidence_record(evidence_sha256=("G" * 64,)),
            "lowercase SHA-256",
            id="invalid-digest",
        ),
        pytest.param(
            lambda: _evidence_record(blockers=[]),
            "blockers must be a tuple",
            id="blocker-list",
        ),
        pytest.param(
            lambda: _evidence_record(evidence_status="unknown"),
            "evidence_status must be locked or hard_gap",
            id="unknown-status",
        ),
        pytest.param(
            lambda: _evidence_record(blockers=("unexpected",)),
            "locked evidence must not carry blockers",
            id="locked-blocker",
        ),
        pytest.param(
            lambda: _evidence_record(evidence_status="hard_gap"),
            "hard-gap evidence must list blockers",
            id="gap-without-blocker",
        ),
        pytest.param(
            lambda: _evidence_record(claim_boundary="test boundary"),
            "canonical evidence boundary",
            id="wrong-boundary",
        ),
    ],
)
def test_dependency_environment_evidence_rejects_malformed_rows(
    factory: Callable[[], DifferentiableDependencyEnvironmentEvidence],
    message: str,
) -> None:
    """Claim-bearing rows must reject coercion and structural contradictions."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_dependency_environment_route_accepts_empty_version_inventory() -> None:
    """Execution routes carry classifications and evidence rather than versions."""
    record = _evidence_record(
        evidence_id="local_cpu",
        category="execution_route",
        classification="locally_runnable",
        version_pins=(),
    )

    assert record.version_pins == ()


def test_dependency_environment_evidence_builder_fails_on_missing_sources(
    tmp_path: Path,
) -> None:
    """The inventory must not emit rows when a cited source file is absent."""
    with pytest.raises(FileNotFoundError):
        build_differentiable_dependency_environment_evidence(repo_root=tmp_path)
