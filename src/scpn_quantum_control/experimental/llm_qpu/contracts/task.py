# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Task and group-safe dataset split contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .wire import SPLIT_SCHEMA, TASK_SCHEMA, ArtifactHeader, _digest, _text, canonical_bytes


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Freeze a non-QPU task target and its causal observation boundary."""

    task_id: str
    objective: str
    source_kind: str
    target_origin: str
    label_schema_digest: str
    causal_cutoff: int
    primary_metric: str
    group_definition: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Require an independently sourced target and a bounded cutoff."""
        for name in ("task_id", "objective", "group_definition"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.source_kind not in ("owner_dataset", "external_dataset", "synthetic_classical"):
            raise ValueError("task source kind must be declared")
        if self.target_origin not in ("independent_ground_truth", "classical_generator"):
            raise ValueError("QPU-generated or unknown task target is forbidden")
        _digest(self.label_schema_digest, name="label schema")
        if type(self.causal_cutoff) is not int or not 0 <= self.causal_cutoff <= 1_000_000:
            raise ValueError("causal cutoff must be a bounded nonnegative token index")
        if self.primary_metric not in ("accuracy", "balanced_accuracy", "f1", "mse", "mae"):
            raise ValueError("primary metric must be fixed and known")
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "task_spec":
            raise ValueError("TaskSpec requires its exact artifact header")
        if self.header.data_origin != self.source_kind:
            raise ValueError("TaskSpec data origin mismatch")
        self.header.validate_content(self._scientific_wire(), parents=(self.label_schema_digest,))

    def _scientific_wire(self) -> dict[str, object]:
        """Return the content hashed by the artifact header."""
        return {
            "schema": TASK_SCHEMA,
            "object_kind": "task_spec",
            **{
                name: getattr(self, name)
                for name in (
                    "task_id",
                    "objective",
                    "source_kind",
                    "target_origin",
                    "label_schema_digest",
                    "causal_cutoff",
                    "primary_metric",
                    "group_definition",
                )
            },
        }

    def to_wire(self) -> dict[str, object]:
        """Return the exact versioned task specification."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> TaskSpec:
        """Reject unknown fields and validate every task field."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("TaskSpec fields mismatch")
        if value["schema"] != TASK_SCHEMA or value["object_kind"] != "task_spec":
            raise ValueError("unknown TaskSpec schema")
        return cls(
            **{name: value[name] for name in cls.__dataclass_fields__ if name != "header"},
            header=ArtifactHeader.from_wire(value["header"]),
        )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Freeze source-group splits and keep test targets with a separate custodian."""

    task_digest: str
    dataset_digest: str
    train_groups: tuple[str, ...]
    dev_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    seed: int
    dedup_rule: str
    test_target_custodian: str
    transform_fit_split: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse group leakage and test-driven transform fitting."""
        _digest(self.task_digest, name="task")
        _digest(self.dataset_digest, name="dataset")
        groups: list[str] = []
        for name in ("train_groups", "dev_groups", "test_groups"):
            group_set = getattr(self, name)
            if type(group_set) is not tuple or not group_set or len(group_set) > 4096:
                raise ValueError(f"{name} must be a nonempty bounded tuple")
            validated = tuple(_text(item, name="source group") for item in group_set)
            if validated != tuple(sorted(validated)) or len(validated) != len(set(validated)):
                raise ValueError("source groups must be sorted and unique")
            object.__setattr__(self, name, validated)
            groups.extend(validated)
        if len(groups) != len(set(groups)):
            raise ValueError("source groups overlap across splits")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("split seed must be a nonnegative signed 64-bit integer")
        if self.dedup_rule not in ("exact_source_digest", "normalized_source_digest"):
            raise ValueError("unknown source dedup rule")
        if self.test_target_custodian != "separate_locked_evaluator":
            raise ValueError("test targets require separate locked evaluator custody")
        if self.transform_fit_split != "train_only":
            raise ValueError("transforms must fit on train groups only")
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "split_manifest":
            raise ValueError("SplitManifest requires its exact artifact header")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.task_digest, self.dataset_digest)
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the content hashed by the artifact header."""
        return {
            "schema": SPLIT_SCHEMA,
            "object_kind": "split_manifest",
            **{
                name: list(getattr(self, name))
                if name.endswith("_groups")
                else getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "header"
            },
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached, versioned split data."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> SplitManifest:
        """Reject unknown fields and validate disjoint source groups."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("SplitManifest fields mismatch")
        if value["schema"] != SPLIT_SCHEMA or value["object_kind"] != "split_manifest":
            raise ValueError("unknown SplitManifest schema")
        for name in ("train_groups", "dev_groups", "test_groups"):
            if type(value[name]) is not list:
                raise ValueError("split groups must be lists on wire")
        return cls(
            task_digest=value["task_digest"],
            dataset_digest=value["dataset_digest"],
            train_groups=tuple(value["train_groups"]),
            dev_groups=tuple(value["dev_groups"]),
            test_groups=tuple(value["test_groups"]),
            seed=value["seed"],
            dedup_rule=value["dedup_rule"],
            test_target_custodian=value["test_target_custodian"],
            transform_fit_split=value["transform_fit_split"],
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_task_split(task: TaskSpec, split: SplitManifest) -> None:
    """Bind a split to the exact frozen task bytes before downstream use."""
    if type(task) is not TaskSpec or type(split) is not SplitManifest:
        raise ValueError("task and split must be validated contracts")
    if hashlib.sha256(canonical_bytes(task.to_wire())).hexdigest() != split.task_digest:
        raise ValueError("split task digest does not bind TaskSpec")
    if task.header.data_origin != split.header.data_origin:
        raise ValueError("split data origin does not bind TaskSpec")
