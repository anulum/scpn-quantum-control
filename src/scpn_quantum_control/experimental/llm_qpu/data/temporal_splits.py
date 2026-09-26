# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal source-group custody
"""Bind paired temporal histories to a disjoint source-group split."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..contracts import (
    ArtifactHeader,
    SplitManifest,
    TaskSpec,
    canonical_bytes,
    validate_task_split,
)
from ..contracts.wire import SPLIT_SCHEMA
from .dedup import normalized_source_digest
from .temporal_task import TemporalRecord, build_temporal_task, generate_temporal_records
from .temporal_transfer import build_transfer_task, generate_transfer_records


def _dataset_digest(records: Sequence[TemporalRecord]) -> str:
    """Hash ordered full records without a monolithic JSON size ceiling."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row.sample_id):
        prompt_bytes = record.prefix_text.encode("utf-8")
        identity = record.public_wire()
        identity.pop("prefix_text")
        encoded = canonical_bytes(
            {
                **identity,
                "prefix_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "prefix_byte_length": len(prompt_bytes),
                "target_text": record.target_text,
            }
        )
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_inventory(records: Sequence[TemporalRecord]) -> None:
    """Reject duplicate IDs, cross-group sources and normalized prompt copies."""
    if not records or len(records) > 14_336:
        raise ValueError("temporal inventory out of bounds")
    samples: set[str] = set()
    source_groups: dict[str, str] = {}
    prompt_groups: dict[str, str] = {}
    for record in records:
        if type(record) is not TemporalRecord:
            raise ValueError("temporal inventory contains an unvalidated record")
        if record.sample_id in samples:
            raise ValueError("duplicate temporal sample ID")
        samples.add(record.sample_id)
        previous_group = source_groups.setdefault(record.source_id, record.group_id)
        if previous_group != record.group_id:
            raise ValueError("one temporal source appears in multiple groups")
        prompt_digest = normalized_source_digest(record.prefix_text)
        previous_prompt_group = prompt_groups.setdefault(prompt_digest, record.group_id)
        if previous_prompt_group != record.group_id:
            raise ValueError("temporal prompt duplicated across source groups")


def build_temporal_split(
    task: TaskSpec,
    records: Sequence[TemporalRecord],
    *,
    seed: int,
    generation_seed: int,
    train_count: int,
    dev_count: int,
    test_count: int,
    base_repo_commit: str,
    implementation_revision: str,
) -> SplitManifest:
    """Freeze the exact generated records and disjoint counterfactual groups."""
    if type(task) is not TaskSpec or task != build_temporal_task(
        base_repo_commit=task.header.base_repo_commit,
        implementation_revision=task.header.implementation_revision,
    ):
        raise ValueError("temporal split requires its exact task")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("split seed must be a nonnegative signed 64-bit integer")
    if any(type(count) is not int or count <= 0 for count in (train_count, dev_count, test_count)):
        raise ValueError("every split needs a positive source count")
    _validate_inventory(records)
    groups = {record.group_id for record in records}
    if len(groups) != train_count + dev_count + test_count:
        raise ValueError("split counts must cover every source group exactly")
    expected = generate_temporal_records(seed=generation_seed, source_count=len(groups))
    if tuple(sorted(records, key=lambda row: row.sample_id)) != tuple(
        sorted(expected, key=lambda row: row.sample_id)
    ):
        raise ValueError("temporal records differ from the independent classical generator")
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


def temporal_pilot_inputs(
    task: TaskSpec,
    records: Sequence[TemporalRecord],
    split: SplitManifest,
    *,
    group_count: int = 8,
) -> tuple[dict[str, object], ...]:
    """Select paired answer-free dev histories in frozen group order."""
    if type(split) is not SplitManifest or split.header.data_origin != "synthetic_classical":
        raise ValueError("temporal pilot requires a synthetic group split")
    validate_task_split(task, split)
    if type(group_count) is not int or not 0 < group_count <= len(split.dev_groups):
        raise ValueError("temporal pilot group count exceeds dev groups")
    _validate_inventory(records)
    if _dataset_digest(records) != split.dataset_digest:
        raise ValueError("pilot records differ from frozen temporal dataset")
    primary = {
        (record.group_id, record.scene_id): record for record in records if record.variant_id == 0
    }
    selected: list[dict[str, object]] = []
    for group in split.dev_groups[:group_count]:
        for scene_id in (0, 1):
            record = primary.get((group, scene_id))
            if record is None:
                raise ValueError("paired temporal history is missing")
            selected.append(record.public_wire())
    return tuple(selected)


def build_transfer_split(
    task: TaskSpec,
    records: Sequence[TemporalRecord],
    *,
    seed: int,
    generation_seed: int,
    train_count: int,
    dev_count: int,
    test_count: int,
    base_repo_commit: str,
    implementation_revision: str,
) -> SplitManifest:
    """Freeze disjoint source groups for the three-copy task version."""
    if (
        type(task) is not TaskSpec
        or task
        != build_transfer_task(
            base_repo_commit=task.header.base_repo_commit,
            implementation_revision=task.header.implementation_revision,
        )
        or (
            task.header.base_repo_commit != base_repo_commit
            or task.header.implementation_revision != implementation_revision
        )
    ):
        raise ValueError("transfer split requires its exact task")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("split seed must be a nonnegative signed 64-bit integer")
    if any(type(count) is not int or count <= 0 for count in (train_count, dev_count, test_count)):
        raise ValueError("every split needs a positive source count")
    _validate_inventory(records)
    groups = {record.group_id for record in records}
    if len(groups) != train_count + dev_count + test_count:
        raise ValueError("split counts must cover every source group exactly")
    expected = generate_transfer_records(seed=generation_seed, source_count=len(groups))
    if tuple(sorted(records, key=lambda row: row.sample_id)) != tuple(
        sorted(expected, key=lambda row: row.sample_id)
    ):
        raise ValueError("transfer records differ from the independent classical generator")
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


def transfer_pilot_inputs(
    task: TaskSpec,
    records: Sequence[TemporalRecord],
    split: SplitManifest,
    *,
    group_count: int = 8,
) -> tuple[dict[str, object], ...]:
    """Select paired answer-free copy-task dev prompts only."""
    if type(task) is not TaskSpec or task != build_transfer_task(
        base_repo_commit=task.header.base_repo_commit,
        implementation_revision=task.header.implementation_revision,
    ):
        raise ValueError("transfer pilot requires its exact task")
    return temporal_pilot_inputs(task, records, split, group_count=group_count)
