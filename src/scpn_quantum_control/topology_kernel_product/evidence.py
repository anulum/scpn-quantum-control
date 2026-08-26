# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-kernel deterministic evidence
"""Deterministic evidence and claim-boundary rendering for topology-kernel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .classifier import evaluate_kernel_ridge, fit_kernel_ridge
from .kernels import (
    fidelity_kernel_matrix,
    permute_edge_features,
    permute_topology,
    rbf_kernel_matrix,
)
from .schema import (
    TOPOLOGY_KERNEL_CLAIM_BOUNDARY,
    KernelEvaluation,
    TopologyKernelConfig,
    TopologyKernelDataset,
)
from .synthetic import (
    build_teacher_aligned_dataset,
    complete_topology,
    path_topology,
    ring_topology,
    zero_topology,
)

TOPOLOGY_KERNEL_EVIDENCE_SCHEMA = "topology_aware_quantum_kernel_evidence_v2"
TOPOLOGY_KERNEL_EVIDENCE_DATE = "2026-07-29"

SupportStatus = Literal["supported", "descoped"]
_NUMERIC_CUSTODY_DECIMALS = 12
_TOPOLOGY_KERNEL_SUPPORT_CONTRACT: tuple[tuple[str, SupportStatus, str, str], ...] = (
    (
        "edge-aligned feature mapping",
        "supported",
        "one feature modulates each canonical undirected XY coupling entry",
        "exact local statevectors only; no provider or hardware execution",
    ),
    (
        "topology-aware kernel classification",
        "supported",
        "custody-checked Gram/cross matrices and regularised kernel ridge fitting",
        "teacher-aligned synthetic representability, not independent generalisation",
    ),
    (
        "application-domain transfer",
        "descoped",
        "no typed application-domain kit or product consumer is implemented",
        "no tokamak, EEG, grid, or other application-domain fitness claim",
    ),
    (
        "controls and numerical invariants",
        "supported",
        "path, complete, zero-coupling, classical RBF, PSD, and relabeling checks",
        "control separation does not establish computational or quantum advantage",
    ),
)


def _canonicalise_evidence_numbers(value: object) -> object:
    """Normalise sub-precision platform drift before evidence custody."""
    if isinstance(value, float):
        rounded = round(value, _NUMERIC_CUSTODY_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {str(key): _canonicalise_evidence_numbers(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise_evidence_numbers(child) for child in value]
    return value


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True, slots=True)
class KernelSupportRow:
    """One explicit implementation or descoping decision.

    Parameters
    ----------
    capability:
        Stable work-package capability name.
    status:
        ``supported`` by this product slice or deliberately ``descoped``.
    evidence:
        Exact implementation or verification fact.
    boundary:
        Interpretation the row does not establish.
    """

    capability: str
    status: SupportStatus
    evidence: str
    boundary: str

    def __post_init__(self) -> None:
        for name in ("capability", "evidence", "boundary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.status not in {"supported", "descoped"}:
            raise ValueError("status must be supported or descoped")

    def to_dict(self) -> dict[str, str]:
        """Return deterministic JSON-compatible support fields."""
        return {
            "capability": self.capability,
            "status": self.status,
            "evidence": self.evidence,
            "boundary": self.boundary,
        }


def _evaluation_dict(evaluation: KernelEvaluation) -> dict[str, object]:
    return {
        "name": evaluation.name,
        "predictions": evaluation.predictions.tolist(),
        "labels": evaluation.labels.tolist(),
        "correct": evaluation.correct,
        "total": evaluation.total,
        "accuracy": evaluation.accuracy,
        "kernel_digest": evaluation.kernel_digest,
    }


@dataclass(frozen=True, slots=True)
class TopologyKernelEvidence:
    """Frozen teacher-aligned synthetic result, checks, and controls.

    Parameters
    ----------
    schema_version, generated_on:
        Versioned evidence format and fixed generation date.
    seed, n_qubits, feature_dim, train_count, test_count:
        Frozen synthetic task dimensions.
    dataset_digest, ring_gram_digest:
        SHA-256 custody for the exact dataset and primary training Gram matrix.
    ring, path, complete, zero, classical_rbf:
        Test-set predictions and accuracies from independently fitted kernel
        ridge models. The first four differ only in graph coupling topology.
    minimum_teacher_margin:
        Smallest absolute prototype-similarity difference in selected data.
    gram_symmetry_max_abs_error, gram_diagonal_max_abs_error:
        Numerical primary-Gram invariants.
    gram_minimum_eigenvalue:
        Smallest eigenvalue of the symmetrised primary Gram matrix.
    permutation_max_abs_error:
        Fidelity change after simultaneous node and edge-feature relabeling.
    support:
        Ordered implementation and descoping ledger.
    claim_boundary:
        Mandatory limit on scientific and operational interpretation.
    content_digest:
        SHA-256 of every preceding canonical evidence field.
    """

    schema_version: str
    generated_on: str
    seed: int
    n_qubits: int
    feature_dim: int
    train_count: int
    test_count: int
    dataset_digest: str
    ring_gram_digest: str
    ring: KernelEvaluation
    path: KernelEvaluation
    complete: KernelEvaluation
    zero: KernelEvaluation
    classical_rbf: KernelEvaluation
    minimum_teacher_margin: float
    gram_symmetry_max_abs_error: float
    gram_diagonal_max_abs_error: float
    gram_minimum_eigenvalue: float
    permutation_max_abs_error: float
    support: tuple[KernelSupportRow, ...]
    claim_boundary: str
    content_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TOPOLOGY_KERNEL_EVIDENCE_SCHEMA:
            raise ValueError("schema_version is unsupported")
        if self.generated_on != TOPOLOGY_KERNEL_EVIDENCE_DATE:
            raise ValueError("generated_on must match the frozen evidence date")
        for name in ("seed", "n_qubits", "feature_dim", "train_count", "test_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dataset_digest", "ring_gram_digest", "content_digest"):
            if not _is_digest(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        evaluations = (self.ring, self.path, self.complete, self.zero, self.classical_rbf)
        expected_names = ("ring", "path", "complete", "zero", "classical_rbf")
        if tuple(item.name for item in evaluations) != expected_names:
            raise ValueError("evaluation names or order do not match the evidence schema")
        if any(item.total != self.test_count for item in evaluations):
            raise ValueError("every evaluation must cover the entire test split")
        metrics = (
            self.minimum_teacher_margin,
            self.gram_symmetry_max_abs_error,
            self.gram_diagonal_max_abs_error,
            self.gram_minimum_eigenvalue,
            self.permutation_max_abs_error,
        )
        if not all(np.isfinite(value) for value in metrics):
            raise ValueError("numerical evidence metrics must be finite")
        if self.minimum_teacher_margin <= 0.0:
            raise ValueError("minimum_teacher_margin must be positive")
        if (
            min(
                self.gram_symmetry_max_abs_error,
                self.gram_diagonal_max_abs_error,
                self.permutation_max_abs_error,
            )
            < 0.0
        ):
            raise ValueError("absolute-error metrics must be non-negative")
        if self.ring.accuracy < 0.9:
            raise ValueError("ring representability gate requires at least 90% accuracy")
        if self.ring.accuracy <= max(item.accuracy for item in evaluations[1:]):
            raise ValueError("ring accuracy must strictly exceed every frozen control")
        actual_support = tuple(
            (row.capability, row.status, row.evidence, row.boundary) for row in self.support
        )
        if actual_support != _TOPOLOGY_KERNEL_SUPPORT_CONTRACT:
            raise ValueError("support rows must match the canonical evidence contract")
        if self.claim_boundary != TOPOLOGY_KERNEL_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the topology-kernel contract")

    def to_dict(self, *, include_content_digest: bool = True) -> dict[str, object]:
        """Return canonical JSON-compatible evidence fields.

        Parameters
        ----------
        include_content_digest:
            Exclude only the outer digest when recomputing custody.
        """
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "generated_on": self.generated_on,
            "seed": self.seed,
            "n_qubits": self.n_qubits,
            "feature_dim": self.feature_dim,
            "train_count": self.train_count,
            "test_count": self.test_count,
            "dataset_digest": self.dataset_digest,
            "ring_gram_digest": self.ring_gram_digest,
            "evaluations": {
                "ring": _evaluation_dict(self.ring),
                "path": _evaluation_dict(self.path),
                "complete": _evaluation_dict(self.complete),
                "zero": _evaluation_dict(self.zero),
                "classical_rbf": _evaluation_dict(self.classical_rbf),
            },
            "minimum_teacher_margin": self.minimum_teacher_margin,
            "gram_symmetry_max_abs_error": self.gram_symmetry_max_abs_error,
            "gram_diagonal_max_abs_error": self.gram_diagonal_max_abs_error,
            "gram_minimum_eigenvalue": self.gram_minimum_eigenvalue,
            "permutation_max_abs_error": self.permutation_max_abs_error,
            "support": [row.to_dict() for row in self.support],
            "claim_boundary": self.claim_boundary,
        }
        if include_content_digest:
            payload["content_digest"] = self.content_digest
        return cast(dict[str, object], _canonicalise_evidence_numbers(payload))


def _content_digest(evidence: TopologyKernelEvidence) -> str:
    canonical = json.dumps(
        evidence.to_dict(include_content_digest=False),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _require_content_digest(evidence: TopologyKernelEvidence) -> None:
    if evidence.content_digest != _content_digest(evidence):
        raise ValueError("content_digest does not match the canonical evidence fields")


def _evaluate_quantum_control(
    name: str,
    dataset: TopologyKernelDataset,
    topology: NDArray[np.float64],
    config: TopologyKernelConfig,
) -> tuple[KernelEvaluation, str]:
    train_kernel = fidelity_kernel_matrix(
        dataset.train_features,
        dataset.train_features,
        topology,
        config,
        row_ids=dataset.train_ids,
        column_ids=dataset.train_ids,
    )
    cross_kernel = fidelity_kernel_matrix(
        dataset.test_features,
        dataset.train_features,
        topology,
        config,
        row_ids=dataset.test_ids,
        column_ids=dataset.train_ids,
    )
    model = fit_kernel_ridge(train_kernel, dataset.train_labels, alpha=config.ridge)
    evaluation = evaluate_kernel_ridge(name, model, cross_kernel, dataset.test_labels)
    return evaluation, train_kernel.content_digest


def _evaluate_rbf_control(
    dataset: TopologyKernelDataset,
    config: TopologyKernelConfig,
    *,
    gamma: float,
) -> KernelEvaluation:
    train_kernel = rbf_kernel_matrix(
        dataset.train_features,
        dataset.train_features,
        config,
        gamma=gamma,
        row_ids=dataset.train_ids,
        column_ids=dataset.train_ids,
    )
    cross_kernel = rbf_kernel_matrix(
        dataset.test_features,
        dataset.train_features,
        config,
        gamma=gamma,
        row_ids=dataset.test_ids,
        column_ids=dataset.train_ids,
    )
    model = fit_kernel_ridge(train_kernel, dataset.train_labels, alpha=config.ridge)
    return evaluate_kernel_ridge("classical_rbf", model, cross_kernel, dataset.test_labels)


def build_topology_kernel_evidence(
    *,
    config: TopologyKernelConfig | None = None,
    seed: int = 880,
) -> TopologyKernelEvidence:
    """Build deterministic topology-kernel evidence with four controls.

    The frozen defaults use four qubits, six canonical edge features, 32 train
    samples, 16 test samples, seed 880, and RBF gamma 0.2. The ring kernel is
    the label-generating teacher; path, complete, zero-coupling, and RBF
    evaluations are explicit controls. All quantum values are exact local
    statevector fidelities.

    Returns
    -------
    TopologyKernelEvidence
        Custody-bound metrics, predictions, invariants, and support ledger.
    """
    policy = TopologyKernelConfig() if config is None else config
    if not isinstance(policy, TopologyKernelConfig):
        raise ValueError("config must be a TopologyKernelConfig or None")
    dataset = build_teacher_aligned_dataset(policy, seed=seed)
    topologies = {
        "ring": ring_topology(policy.n_qubits),
        "path": path_topology(policy.n_qubits),
        "complete": complete_topology(policy.n_qubits),
        "zero": zero_topology(policy.n_qubits),
    }
    ring, ring_gram_digest = _evaluate_quantum_control("ring", dataset, topologies["ring"], policy)
    path, _ = _evaluate_quantum_control("path", dataset, topologies["path"], policy)
    complete, _ = _evaluate_quantum_control("complete", dataset, topologies["complete"], policy)
    zero, _ = _evaluate_quantum_control("zero", dataset, topologies["zero"], policy)
    classical_rbf = _evaluate_rbf_control(dataset, policy, gamma=0.2)

    gram = fidelity_kernel_matrix(
        dataset.train_features,
        dataset.train_features,
        topologies["ring"],
        policy,
        row_ids=dataset.train_ids,
        column_ids=dataset.train_ids,
    )
    symmetry_error = float(np.max(np.abs(gram.values - gram.values.T)))
    diagonal_error = float(np.max(np.abs(np.diag(gram.values) - 1.0)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh((gram.values + gram.values.T) / 2.0)))

    all_features = np.vstack((dataset.train_features, dataset.test_features))
    all_ids = dataset.train_ids + dataset.test_ids
    teacher = fidelity_kernel_matrix(
        all_features,
        dataset.teacher_prototypes,
        topologies["ring"],
        policy,
        row_ids=all_ids,
        column_ids=("teacher-positive", "teacher-negative"),
    )
    minimum_margin = float(np.min(np.abs(teacher.values[:, 0] - teacher.values[:, 1])))

    permutation = tuple(range(1, policy.n_qubits)) + (0,)
    probe_features = dataset.test_features[:8]
    probe_ids = dataset.test_ids[:8]
    original = fidelity_kernel_matrix(
        probe_features,
        probe_features,
        topologies["ring"],
        policy,
        row_ids=probe_ids,
        column_ids=probe_ids,
    )
    relabelled = fidelity_kernel_matrix(
        permute_edge_features(probe_features, permutation, policy),
        permute_edge_features(probe_features, permutation, policy),
        permute_topology(topologies["ring"], permutation, policy),
        policy,
        row_ids=probe_ids,
        column_ids=probe_ids,
    )
    permutation_error = float(np.max(np.abs(original.values - relabelled.values)))
    support = tuple(KernelSupportRow(*row) for row in _TOPOLOGY_KERNEL_SUPPORT_CONTRACT)
    provisional = TopologyKernelEvidence(
        schema_version=TOPOLOGY_KERNEL_EVIDENCE_SCHEMA,
        generated_on=TOPOLOGY_KERNEL_EVIDENCE_DATE,
        seed=seed,
        n_qubits=policy.n_qubits,
        feature_dim=policy.feature_dim,
        train_count=dataset.train_features.shape[0],
        test_count=dataset.test_features.shape[0],
        dataset_digest=dataset.content_digest,
        ring_gram_digest=ring_gram_digest,
        ring=ring,
        path=path,
        complete=complete,
        zero=zero,
        classical_rbf=classical_rbf,
        minimum_teacher_margin=minimum_margin,
        gram_symmetry_max_abs_error=symmetry_error,
        gram_diagonal_max_abs_error=diagonal_error,
        gram_minimum_eigenvalue=minimum_eigenvalue,
        permutation_max_abs_error=permutation_error,
        support=support,
        claim_boundary=TOPOLOGY_KERNEL_CLAIM_BOUNDARY,
        content_digest="0" * 64,
    )
    return replace(provisional, content_digest=_content_digest(provisional))


def render_topology_kernel_markdown(evidence: TopologyKernelEvidence) -> str:
    """Render a concise human-readable companion to canonical JSON evidence."""
    if not isinstance(evidence, TopologyKernelEvidence):
        raise ValueError("evidence must be TopologyKernelEvidence")
    _require_content_digest(evidence)
    rows = [
        "| Kernel | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for evaluation in (
        evidence.ring,
        evidence.path,
        evidence.complete,
        evidence.zero,
        evidence.classical_rbf,
    ):
        rows.append(
            f"| `{evaluation.name}` | {evaluation.correct} | {evaluation.total} | "
            f"{evaluation.accuracy:.6f} |"
        )
    support_rows = [
        "| Capability | Status | Evidence | Boundary |",
        "|---|---|---|---|",
    ]
    support_rows.extend(
        f"| {row.capability} | `{row.status}` | {row.evidence} | {row.boundary} |"
        for row in evidence.support
    )
    return "\n".join(
        [
            "# Topology-aware quantum-kernel evidence",
            "",
            f"- Schema: `{evidence.schema_version}`",
            f"- Generated: `{evidence.generated_on}`",
            f"- Dataset digest: `{evidence.dataset_digest}`",
            f"- Evidence digest: `{evidence.content_digest}`",
            "",
            "The labels are generated by the same ring-kernel family evaluated below. "
            "The result therefore tests declared inductive-bias representability, not "
            "independent generalisation or quantum advantage.",
            "",
            *rows,
            "",
            "## Numerical invariants",
            "",
            f"- Minimum teacher margin: `{evidence.minimum_teacher_margin:.16g}`",
            f"- Gram symmetry max error: `{evidence.gram_symmetry_max_abs_error:.3e}`",
            f"- Gram diagonal max error: `{evidence.gram_diagonal_max_abs_error:.3e}`",
            f"- Gram minimum eigenvalue: `{evidence.gram_minimum_eigenvalue:.16g}`",
            f"- Simultaneous relabeling max error: `{evidence.permutation_max_abs_error:.3e}`",
            "",
            "## Scope ledger",
            "",
            *support_rows,
            "",
            f"Claim boundary: {evidence.claim_boundary}.",
            "",
        ]
    )


def write_topology_kernel_evidence(
    evidence: TopologyKernelEvidence,
    json_path: Path,
    markdown_path: Path,
    *,
    check: bool = False,
) -> tuple[Path, Path]:
    """Write or byte-check canonical JSON and Markdown evidence.

    Parameters
    ----------
    evidence:
        Frozen record to serialise.
    json_path, markdown_path:
        Distinct explicit output targets.
    check:
        When true, require both existing files to exactly equal regenerated
        bytes and raise without writing on any difference. When false, parent
        directories are created and only the two exact targets are replaced.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        JSON and Markdown paths in argument order.

    Raises
    ------
    ValueError
        If evidence is the wrong type or paths are equal.
    RuntimeError
        In check mode when a file is absent or its bytes differ.
    """
    if not isinstance(evidence, TopologyKernelEvidence):
        raise ValueError("evidence must be TopologyKernelEvidence")
    _require_content_digest(evidence)
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    if json_target == markdown_target:
        raise ValueError("json_path and markdown_path must be distinct")
    json_text = json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
    markdown_text = render_topology_kernel_markdown(evidence)
    if check:
        for target, expected in ((json_target, json_text), (markdown_target, markdown_text)):
            if not target.is_file() or target.read_text() != expected:
                raise RuntimeError(f"evidence byte check failed: {target}")
        return json_target, markdown_target
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json_text)
    markdown_target.write_text(markdown_text)
    return json_target, markdown_target
