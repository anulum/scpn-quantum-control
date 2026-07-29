# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-88 evidence tests
"""Evidence construction, custody, rendering, and fail-closed tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scpn_quantum_control.topology_kernel_product import (
    BL88_EVIDENCE_DATE,
    BL88_EVIDENCE_SCHEMA,
    KernelEvaluation,
    KernelSupportRow,
    TopologyKernelEvidence,
    build_topology_kernel_evidence,
    render_topology_kernel_markdown,
    write_topology_kernel_evidence,
)


@pytest.fixture(scope="module")
def evidence() -> TopologyKernelEvidence:
    """Build the frozen evidence once for this module."""
    return build_topology_kernel_evidence()


def _low_accuracy(name: str, labels: np.ndarray) -> KernelEvaluation:
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
    assert evidence.schema_version == BL88_EVIDENCE_SCHEMA
    assert evidence.generated_on == BL88_EVIDENCE_DATE
    assert (
        evidence.content_digest
        == "a960ec0386d892518548c0d00cb8bc765301768ef52c4b29a6922050eb1d2c22"
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


def test_support_row_normalises_and_serialises() -> None:
    row = KernelSupportRow(" cap ", "supported", " evidence ", " boundary ")
    assert row.to_dict() == {
        "capability": "cap",
        "status": "supported",
        "evidence": "evidence",
        "boundary": "boundary",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capability": ""},
        {"evidence": ""},
        {"boundary": ""},
        {"status": "bad"},
    ],
)
def test_support_row_rejects_invalid_fields(kwargs: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "capability": "cap",
        "status": "supported",
        "evidence": "evidence",
        "boundary": "boundary",
    }
    fields.update(kwargs)
    with pytest.raises(ValueError):
        KernelSupportRow(**fields)  # type: ignore[arg-type]


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
        build_topology_kernel_evidence(config=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        render_topology_kernel_markdown(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        write_topology_kernel_evidence(object(), tmp_path / "a", tmp_path / "b")  # type: ignore[arg-type]
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
    "changes",
    [
        {"schema_version": "bad"},
        {"generated_on": "2026-01-01"},
        {"seed": True},
        {"n_qubits": 0},
        {"dataset_digest": "bad"},
        {"ring_gram_digest": "bad"},
        {"content_digest": "bad"},
        {"minimum_teacher_margin": np.nan},
        {"gram_symmetry_max_abs_error": np.inf},
        {"gram_diagonal_max_abs_error": np.nan},
        {"gram_minimum_eigenvalue": np.nan},
        {"permutation_max_abs_error": np.inf},
        {"minimum_teacher_margin": 0.0},
        {"gram_symmetry_max_abs_error": -1.0},
        {"support": ()},
        {"claim_boundary": ""},
    ],
)
def test_evidence_rejects_invalid_scalar_contract(
    evidence: TopologyKernelEvidence,
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        replace(evidence, **changes)


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
    with pytest.raises(ValueError, match="capability-unique"):
        replace(evidence, support=duplicate)
    supported_only = tuple(replace(row, status="supported") for row in evidence.support)
    with pytest.raises(ValueError, match="descoped"):
        replace(evidence, support=supported_only)
