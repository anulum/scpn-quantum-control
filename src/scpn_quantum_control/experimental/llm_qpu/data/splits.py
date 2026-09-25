# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — deterministic source-group partitions
"""Freeze synthetic memory group splits and answer-free pilot inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..contracts import ArtifactHeader, SplitManifest, TaskSpec, canonical_bytes
from ..contracts.wire import SPLIT_SCHEMA
from .dedup import validate_source_families
from .tasks import MemoryRecord, generate_memory_records


def _dataset_digest(records: Sequence[MemoryRecord]) -> str:
    return hashlib.sha256(
        canonical_bytes(
            [
                {**record.public_wire(), "target_text": record.target_text}
                for record in sorted(records, key=lambda row: row.sample_id)
            ]
        )
    ).hexdigest()


def build_memory_split(
    task: TaskSpec,
    records: Sequence[MemoryRecord],
    *,
    seed: int,
    generation_seed: int,
    train_count: int,
    dev_count: int,
    test_count: int,
    base_repo_commit: str,
    implementation_revision: str,
) -> SplitManifest:
    """Freeze exact disjoint source groups and private target custody."""
    if (
        type(task) is not TaskSpec
        or task.source_kind != "synthetic_classical"
        or task.target_origin != "classical_generator"
    ):
        raise ValueError("memory split requires a synthetic TaskSpec")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("split seed must be a nonnegative signed 64-bit integer")
    if any(type(count) is not int or count <= 0 for count in (train_count, dev_count, test_count)):
        raise ValueError("every split needs a positive source count")
    validate_source_families(records)
    groups = {record.group_id for record in records}
    if len(groups) != train_count + dev_count + test_count:
        raise ValueError("split counts must cover every source group exactly")
    expected = generate_memory_records(seed=generation_seed, source_count=len(groups))
    if tuple(sorted(records, key=lambda row: row.sample_id)) != tuple(
        sorted(expected, key=lambda row: row.sample_id)
    ):
        raise ValueError("memory records differ from independent classical generator")
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            canonical_bytes({"seed": seed, "group_id": group})
        ).digest(),
    )
    train_groups = tuple(sorted(ranked[:train_count]))
    dev_groups = tuple(sorted(ranked[train_count : train_count + dev_count]))
    test_groups = tuple(sorted(ranked[train_count + dev_count :]))
    task_digest = hashlib.sha256(canonical_bytes(task.to_wire())).hexdigest()
    dataset_digest = _dataset_digest(records)
    content = {
        "schema": SPLIT_SCHEMA,
        "object_kind": "split_manifest",
        "task_digest": task_digest,
        "dataset_digest": dataset_digest,
        "train_groups": list(train_groups),
        "dev_groups": list(dev_groups),
        "test_groups": list(test_groups),
        "seed": seed,
        "dedup_rule": "normalized_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    header = ArtifactHeader(
        object_kind="split_manifest",
        content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest(),
        parents=tuple(sorted((task_digest, dataset_digest))),
        base_repo_commit=base_repo_commit,
        implementation_revision=implementation_revision,
        execution_origin="offline_design",
        data_origin="synthetic_classical",
        claim_scope="design_only",
    )
    return SplitManifest(
        task_digest=task_digest,
        dataset_digest=dataset_digest,
        train_groups=train_groups,
        dev_groups=dev_groups,
        test_groups=test_groups,
        seed=seed,
        dedup_rule="normalized_source_digest",
        test_target_custodian="separate_locked_evaluator",
        transform_fit_split="train_only",
        header=header,
    )


def pilot_inputs(
    records: Sequence[MemoryRecord], split: SplitManifest, *, count: int = 16
) -> tuple[dict[str, object], ...]:
    """Select one answer-free dev prefix per source for the engineering pilot."""
    if type(split) is not SplitManifest or split.header.data_origin != "synthetic_classical":
        raise ValueError("pilot requires a synthetic group split")
    if type(count) is not int or not 0 < count <= len(split.dev_groups):
        raise ValueError("pilot count exceeds dev groups")
    validate_source_families(records)
    if _dataset_digest(records) != split.dataset_digest:
        raise ValueError("pilot records differ from frozen split dataset")
    available = {record.group_id: record for record in records if record.variant_id == 0}
    selected = split.dev_groups[:count]
    if any(group not in available for group in selected):
        raise ValueError("pilot source group is missing its primary prompt")
    return tuple(available[group].public_wire() for group in selected)
