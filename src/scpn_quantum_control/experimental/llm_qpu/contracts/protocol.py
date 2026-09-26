# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Immutable, design-only analysis protocol for the LLM-QPU lane."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .task import SplitManifest, TaskSpec, validate_task_split
from .wire import EXPERIMENT_PROTOCOL_SCHEMA, ArtifactHeader, _digest, _text, canonical_bytes


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """Freeze the intended comparison before an evaluator opens outcomes.

    Construction proves structural integrity only. A separate custody record must
    establish that this exact digest predates access to held-out outcomes.
    """

    experiment_id: str
    evaluation_id: str
    task_digest: str
    split_digest: str
    model_digest: str
    compressor_digest: str
    kernel_plan_digest: str
    measurement_plan_digest: str
    analysis_plan_digest: str
    mode: str
    prediction_mode: str
    arms: tuple[str, ...]
    classical_arm: str
    quantum_arm: str
    primary_metric: str
    max_dev_fits_per_arm: int
    max_qpu_evaluations: int
    stopping_rule: str
    failure_policy: str
    inference_unit: str
    analysis_scope: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Reject unfrozen comparisons, ambiguous budgets and silent attrition."""
        for name in ("experiment_id", "evaluation_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "task_digest",
            "split_digest",
            "model_digest",
            "compressor_digest",
            "kernel_plan_digest",
            "measurement_plan_digest",
            "analysis_plan_digest",
        ):
            _digest(getattr(self, name), name=name)
        digests = self._parents()
        if len(set(digests)) != len(digests):
            raise ValueError("protocol parent digests must identify distinct artifacts")
        if self.mode not in ("contextual_latent_transform", "chunk_isolated_sequence"):
            raise ValueError("unknown memory experiment mode")
        if self.prediction_mode not in ("teacher_forced_trace", "free_generation"):
            raise ValueError("unknown prediction mode")
        if type(self.arms) is not tuple or not 2 <= len(self.arms) <= 32:
            raise ValueError("protocol needs bounded comparison arms")
        arms = tuple(_text(arm, name="arm") for arm in self.arms)
        if arms != tuple(sorted(set(arms))):
            raise ValueError("arms must be sorted and unique")
        object.__setattr__(self, "arms", arms)
        if (
            self.classical_arm not in arms
            or self.quantum_arm not in arms
            or self.classical_arm == self.quantum_arm
        ):
            raise ValueError("primary contrast must name distinct registered arms")
        if self.primary_metric not in (
            "mean_log_loss",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "mse",
            "mae",
        ):
            raise ValueError("unknown primary metric")
        for name in ("max_dev_fits_per_arm", "max_qpu_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 1_000_000:
                raise ValueError(f"{name} must be a positive bounded budget")
        if self.stopping_rule != "fixed_split_no_optional_stopping":
            raise ValueError("outcome-dependent stopping is forbidden")
        if self.failure_policy != "retain_all_ids_report_missing_and_sensitivity":
            raise ValueError("missing results require explicit inventory and sensitivity")
        if self.inference_unit != "source_group_paired":
            raise ValueError("shots and tokens are not independent inference units")
        if self.analysis_scope not in ("exploratory", "confirmatory"):
            raise ValueError("analysis scope must be declared")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "experiment_protocol"
        ):
            raise ValueError("ExperimentProtocol requires its exact artifact header")
        self.header.validate_content(self._scientific_wire(), parents=digests)

    def _parents(self) -> tuple[str, ...]:
        """Return the complete sorted artifact lineage."""
        return tuple(
            sorted(
                (
                    self.task_digest,
                    self.split_digest,
                    self.model_digest,
                    self.compressor_digest,
                    self.kernel_plan_digest,
                    self.measurement_plan_digest,
                    self.analysis_plan_digest,
                )
            )
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the exact bytes covered by the content digest."""
        return {
            "schema": EXPERIMENT_PROTOCOL_SCHEMA,
            "object_kind": "experiment_protocol",
            **{
                name: list(getattr(self, name)) if name == "arms" else getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "header"
            },
        }

    def to_wire(self) -> dict[str, object]:
        """Return a detached, versioned record."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> ExperimentProtocol:
        """Reject unknown fields and noncanonical arm representation."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ExperimentProtocol fields mismatch")
        if (
            value["schema"] != EXPERIMENT_PROTOCOL_SCHEMA
            or value["object_kind"] != "experiment_protocol"
        ):
            raise ValueError("unknown ExperimentProtocol schema")
        if type(value["arms"]) is not list:
            raise ValueError("protocol arms must be a list on wire")
        return cls(
            experiment_id=value["experiment_id"],
            evaluation_id=value["evaluation_id"],
            task_digest=value["task_digest"],
            split_digest=value["split_digest"],
            model_digest=value["model_digest"],
            compressor_digest=value["compressor_digest"],
            kernel_plan_digest=value["kernel_plan_digest"],
            measurement_plan_digest=value["measurement_plan_digest"],
            analysis_plan_digest=value["analysis_plan_digest"],
            mode=value["mode"],
            prediction_mode=value["prediction_mode"],
            arms=tuple(value["arms"]),
            classical_arm=value["classical_arm"],
            quantum_arm=value["quantum_arm"],
            primary_metric=value["primary_metric"],
            max_dev_fits_per_arm=value["max_dev_fits_per_arm"],
            max_qpu_evaluations=value["max_qpu_evaluations"],
            stopping_rule=value["stopping_rule"],
            failure_policy=value["failure_policy"],
            inference_unit=value["inference_unit"],
            analysis_scope=value["analysis_scope"],
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_experiment_protocol(
    protocol: ExperimentProtocol, task: TaskSpec, split: SplitManifest
) -> None:
    """Bind the frozen comparison to its exact task and locked group split."""
    if type(protocol) is not ExperimentProtocol:
        raise ValueError("protocol must be a validated contract")
    validate_task_split(task, split)
    if protocol.task_digest != hashlib.sha256(canonical_bytes(task.to_wire())).hexdigest():
        raise ValueError("protocol task digest mismatch")
    if protocol.split_digest != hashlib.sha256(canonical_bytes(split.to_wire())).hexdigest():
        raise ValueError("protocol split digest mismatch")
    if protocol.primary_metric != task.primary_metric:
        raise ValueError("protocol primary metric differs from frozen task")
    if protocol.header.data_origin != task.source_kind:
        raise ValueError("protocol data origin differs from frozen task")
