# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-kernel evidence tests
"""Evidence construction, custody, rendering, and fail-closed tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.topology_kernel_product import (
    TOPOLOGY_KERNEL_EVIDENCE_DATE,
    TOPOLOGY_KERNEL_EVIDENCE_SCHEMA,
    KernelEvaluation,
    KernelSupportRow,
    TopologyKernelConfig,
    TopologyKernelEvidence,
    build_topology_kernel_evidence,
    render_topology_kernel_markdown,
    write_topology_kernel_evidence,
)


@pytest.fixture(scope="module")
def evidence() -> TopologyKernelEvidence:
    """Build the frozen evidence once for this module."""
    return build_topology_kernel_evidence()


def _low_accuracy(name: str, labels: NDArray[np.int64]) -> KernelEvaluation:
    predictions = np.ones_like(labels)
    correct = int(np.sum(predictions == labels))
    return KernelEvaluation(
        name=name,
        predictions=predictions,
        labels=labels,
        correct=correct,
        total=labels.size,
        accuracy=correct / labels.size,
        kernel_digest="a" * 64,
    )


def test_frozen_evidence_matches_registered_metrics(evidence: TopologyKernelEvidence) -> None:
    assert evidence.schema_version == TOPOLOGY_KERNEL_EVIDENCE_SCHEMA
    assert evidence.generated_on == TOPOLOGY_KERNEL_EVIDENCE_DATE
    assert (
        evidence.content_digest
        == "deea53654166e4909a73d2e16e2fb4e935fc64a0a0e2f9bb97f85c9ecb664904"
    )
    assert evidence.ring.accuracy == pytest.approx(1.0)
    assert evidence.path.accuracy == pytest.approx(0.25)
    assert evidence.complete.accuracy == pytest.approx(0.5625)
    assert evidence.zero.accuracy == pytest.approx(0.5)
    assert evidence.classical_rbf.accuracy == pytest.approx(0.5)
    assert evidence.minimum_teacher_margin == pytest.approx(0.22085677522668157)
    assert evidence.gram_minimum_eigenvalue > 0.0
    assert evidence.permutation_max_abs_error < 1.0e-12
    assert evidence.to_dict(include_content_digest=False).get("content_digest") is None


def test_evidence_payload_normalises_subprecision_runtime_drift(
    evidence: TopologyKernelEvidence,
) -> None:
    perturbed = replace(
        evidence,
        gram_minimum_eigenvalue=evidence.gram_minimum_eigenvalue + 1.0e-14,
    )
    assert perturbed.to_dict() == evidence.to_dict()


def test_evidence_rejects_stale_schema_and_claim_contract_drift(
    evidence: TopologyKernelEvidence,
) -> None:
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        replace(evidence, schema_version="topology_aware_quantum_kernel_evidence_v1")
    with pytest.raises(ValueError, match="canonical evidence contract"):
        replace(
            evidence,
            support=(replace(evidence.support[0], capability="feature-map work package"),)
            + evidence.support[1:],
        )
    with pytest.raises(ValueError, match="topology-kernel contract"):
        replace(evidence, claim_boundary="broader claim")


def test_render_and_write_reject_content_digest_drift(
    evidence: TopologyKernelEvidence,
    tmp_path: Path,
) -> None:
    drifted = replace(evidence, content_digest="b" * 64)
    with pytest.raises(ValueError, match="canonical evidence fields"):
        render_topology_kernel_markdown(drifted)
    with pytest.raises(ValueError, match="canonical evidence fields"):
        write_topology_kernel_evidence(
            drifted,
            tmp_path / "evidence.json",
            tmp_path / "evidence.md",
        )


def test_support_row_normalises_and_serialises() -> None:
    row = KernelSupportRow(" cap ", "supported", " evidence ", " boundary ")
    assert row.to_dict() == {
        "capability": "cap",
        "status": "supported",
        "evidence": "evidence",
        "boundary": "boundary",
    }


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: KernelSupportRow("", "supported", "evidence", "boundary"),
        lambda: KernelSupportRow("capability", "supported", "", "boundary"),
        lambda: KernelSupportRow("capability", "supported", "evidence", ""),
        lambda: KernelSupportRow(
            "capability",
            cast(Literal["supported", "descoped"], "bad"),
            "evidence",
            "boundary",
        ),
    ],
)
def test_support_row_rejects_invalid_fields(
    constructor: Callable[[], KernelSupportRow],
) -> None:
    with pytest.raises(ValueError):
        constructor()


def test_evidence_markdown_and_json_are_deterministic(
    evidence: TopologyKernelEvidence,
    tmp_path: Path,
) -> None:
    markdown = render_topology_kernel_markdown(evidence)
    assert "representability, not independent generalisation" in markdown
    assert "| `ring` | 16 | 16 | 1.000000 |" in markdown
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    returned = write_topology_kernel_evidence(evidence, json_path, markdown_path)
    assert returned == (json_path, markdown_path)
    assert json.loads(json_path.read_text()) == evidence.to_dict()
    assert markdown_path.read_text() == markdown
    assert write_topology_kernel_evidence(evidence, json_path, markdown_path, check=True) == (
        json_path,
        markdown_path,
    )
    markdown_path.write_text("stale")
    with pytest.raises(RuntimeError, match="byte check failed"):
        write_topology_kernel_evidence(evidence, json_path, markdown_path, check=True)


def test_evidence_helpers_reject_wrong_types_and_equal_paths(
    evidence: TopologyKernelEvidence,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        build_topology_kernel_evidence(config=cast(TopologyKernelConfig, object()))
    with pytest.raises(ValueError):
        render_topology_kernel_markdown(cast(TopologyKernelEvidence, object()))
    with pytest.raises(ValueError):
        write_topology_kernel_evidence(
            cast(TopologyKernelEvidence, object()),
            tmp_path / "a",
            tmp_path / "b",
        )
    with pytest.raises(ValueError):
        write_topology_kernel_evidence(evidence, tmp_path / "same", tmp_path / "same")
    with pytest.raises(RuntimeError, match="byte check failed"):
        write_topology_kernel_evidence(
            evidence,
            tmp_path / "missing.json",
            tmp_path / "missing.md",
            check=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: replace(item, schema_version="bad"),
        lambda item: replace(item, generated_on="2026-01-01"),
        lambda item: replace(item, seed=True),
        lambda item: replace(item, n_qubits=0),
        lambda item: replace(item, dataset_digest="bad"),
        lambda item: replace(item, ring_gram_digest="bad"),
        lambda item: replace(item, content_digest="bad"),
        lambda item: replace(item, minimum_teacher_margin=np.nan),
        lambda item: replace(item, gram_symmetry_max_abs_error=np.inf),
        lambda item: replace(item, gram_diagonal_max_abs_error=np.nan),
        lambda item: replace(item, gram_minimum_eigenvalue=np.nan),
        lambda item: replace(item, permutation_max_abs_error=np.inf),
        lambda item: replace(item, minimum_teacher_margin=0.0),
        lambda item: replace(item, gram_symmetry_max_abs_error=-1.0),
        lambda item: replace(item, support=()),
        lambda item: replace(item, claim_boundary=""),
    ],
)
def test_evidence_rejects_invalid_scalar_contract(
    evidence: TopologyKernelEvidence,
    mutate: Callable[[TopologyKernelEvidence], TopologyKernelEvidence],
) -> None:
    with pytest.raises(ValueError):
        mutate(evidence)


def test_evidence_rejects_evaluation_name_count_and_accuracy_gates(
    evidence: TopologyKernelEvidence,
) -> None:
    with pytest.raises(ValueError, match="names or order"):
        replace(evidence, ring=replace(evidence.ring, name="wrong"))
    with pytest.raises(ValueError, match="entire test split"):
        replace(evidence, test_count=15)
    low_ring = _low_accuracy("ring", evidence.ring.labels)
    with pytest.raises(ValueError, match="90%"):
        replace(evidence, ring=low_ring)
    perfect_control = replace(evidence.ring, name="path")
    with pytest.raises(ValueError, match="strictly exceed"):
        replace(evidence, path=perfect_control)


def test_evidence_requires_unique_support_and_descoped_row(
    evidence: TopologyKernelEvidence,
) -> None:
    duplicate = evidence.support + (evidence.support[0],)
    with pytest.raises(ValueError, match="canonical evidence contract"):
        replace(evidence, support=duplicate)
    supported_only = tuple(replace(row, status="supported") for row in evidence.support)
    with pytest.raises(ValueError, match="canonical evidence contract"):
        replace(evidence, support=supported_only)
