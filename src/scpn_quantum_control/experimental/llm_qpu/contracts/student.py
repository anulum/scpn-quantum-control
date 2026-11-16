# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Offline classical students distilled from quantum features; never quantum themselves."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .features import FeatureBatch
from .protocol import ExperimentProtocol
from .task import SplitManifest
from .wire import (
    STUDENT_ARTIFACT_SCHEMA,
    ArtifactHeader,
    _digest,
    _strict_json,
    _text,
    canonical_bytes,
    f64,
)

_TEACHERS = {"hardware_features": "provider_result", "simulated_features": "synthetic_classical"}
_FIDELITY = ("feature_mse", "feature_mae")
_UTILITY = ("accuracy", "mean_log_loss", "mse", "mae")
_FLOATS = ("learning_rate", "fidelity", "utility")
_DIGESTS = ("protocol_digest", "model_digest", "compressor_digest", "teacher_digest")


def _finite(value: object, *, name: str) -> float:
    """Return one finite float, refusing integers and booleans."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


@dataclass(frozen=True, slots=True)
class StudentArtifact:
    """Design-only classical student trained on train groups to mimic a feature teacher."""

    student_id: str
    protocol_digest: str
    model_digest: str
    compressor_digest: str
    teacher_kind: str
    teacher_digest: str
    targets: tuple[str, ...]
    architecture: str
    weights_digest: str
    config_digest: str
    training_groups: tuple[str, ...]
    optimizer: str
    learning_rate: float
    epochs: int
    seed: int
    device: str
    runtime: str
    runtime_version: str
    heldout_split: str
    fidelity_metric: str
    fidelity: float
    utility_metric: str
    utility: float
    output_class: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse a quantum or hardware label, open-ended training and unbounded scores."""
        if self.output_class in ("quantum", "hardware", "quantum_hardware"):
            raise ValueError("a student runs classically; its output is never quantum or hardware")
        if self.output_class != "classical_offline":
            raise ValueError("unknown student output class")
        for name in (
            "student_id",
            "architecture",
            "optimizer",
            "device",
            "runtime",
            "runtime_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (*_DIGESTS, "weights_digest", "config_digest"):
            _digest(getattr(self, name), name=name)
        if self.teacher_kind not in _TEACHERS:
            raise ValueError("a teacher is hardware or simulated quantum features")
        for name in ("targets", "training_groups"):
            values = getattr(self, name)
            if type(values) is not tuple or not 0 < len(values) <= 4096:
                raise ValueError(f"{name} must be a nonempty tuple")
            if len({_text(value, name=name) for value in values}) != len(values):
                raise ValueError(f"{name} must be unique")
        if self.training_groups != tuple(sorted(self.training_groups)):
            raise ValueError("training groups must be sorted")
        self._check_training_and_scores()
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "student_artifact"
        ):
            raise ValueError("StudentArtifact requires its exact artifact header")
        if self.header.data_origin != _TEACHERS[self.teacher_kind]:
            raise ValueError("a student's data origin is its teacher's")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.protocol_digest, self.teacher_digest)
        )

    def _check_training_and_scores(self) -> None:
        """Bound the optimiser, the held-out split and the fidelity and utility scores."""
        if _finite(self.learning_rate, name="learning rate") <= 0.0:
            raise ValueError("learning rate must be positive")
        if type(self.epochs) is not int or not 0 < self.epochs <= 1_000_000:
            raise ValueError("epochs must be a bounded positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a non-negative 63-bit integer")
        if self.heldout_split not in ("dev", "test"):
            raise ValueError("held-out scores come from the dev or the test split")
        if self.fidelity_metric not in _FIDELITY:
            raise ValueError("unknown fidelity metric")
        if _finite(self.fidelity, name="fidelity") < 0.0:
            raise ValueError("a feature error cannot be negative")
        if self.utility_metric not in _UTILITY:
            raise ValueError("unknown utility metric")
        utility = _finite(self.utility, name="utility")
        if utility < 0.0 or (self.utility_metric == "accuracy" and utility > 1.0):
            raise ValueError("utility must be non-negative, and at most 1 for accuracy")

    def _scientific_wire(self) -> dict[str, object]:
        """Return the student content hashed by its header."""
        return {
            "schema": STUDENT_ARTIFACT_SCHEMA,
            "object_kind": "student_artifact",
            **{
                name: list(value)
                if type(value) is tuple
                else f64(value)
                if name in _FLOATS
                else value
                for name in self.__dataclass_fields__
                if name != "header"
                for value in (getattr(self, name),)
            },
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned student record."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> StudentArtifact:
        """Reject unknown fields and restore tuples and f64 values."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("StudentArtifact fields mismatch")
        if (
            value["schema"] != STUDENT_ARTIFACT_SCHEMA
            or value["object_kind"] != "student_artifact"
        ):
            raise ValueError("unknown StudentArtifact schema")
        if type(value["targets"]) is not list or type(value["training_groups"]) is not list:
            raise ValueError("targets and training groups must be lists on wire")
        restored: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            item = value[name]
            if name in ("targets", "training_groups"):
                restored[name] = tuple(item)
            elif name in _FLOATS:
                if (
                    type(item) is not dict
                    or set(item) != {"$f64"}
                    or type(item["$f64"]) is not str
                ):
                    raise ValueError(f"{name} must be an f64 tag on wire")
                restored[name] = float.fromhex(item["$f64"])
            elif name == "header":
                restored[name] = ArtifactHeader.from_wire(item)
            else:
                restored[name] = item
        return cls(**restored)


def validate_student_artifact(
    student: StudentArtifact,
    protocol: ExperimentProtocol,
    split: SplitManifest,
    teacher: FeatureBatch,
) -> None:
    """Refuse a student of another model, teacher or split, or a relabelled teacher."""
    if type(student) is not StudentArtifact:
        raise ValueError("student validation needs a StudentArtifact")
    if student.protocol_digest != hashlib.sha256(canonical_bytes(protocol.to_wire())).hexdigest():
        raise ValueError("the student names a different frozen protocol")
    if protocol.split_digest != hashlib.sha256(canonical_bytes(split.to_wire())).hexdigest():
        raise ValueError("the protocol binds a different split")
    if (student.model_digest, student.compressor_digest) != (
        protocol.model_digest,
        protocol.compressor_digest,
    ):
        raise ValueError("the student names another model or compressor than the protocol")
    if student.teacher_digest != hashlib.sha256(canonical_bytes(teacher.to_wire())).hexdigest():
        raise ValueError("the student names a different teacher feature batch")
    if _TEACHERS[student.teacher_kind] != teacher.header.data_origin:
        raise ValueError("teacher kind does not match the teacher's origin; no relabelling")
    if list(student.targets) != [
        name for name in teacher.observable_ids if name in student.targets
    ]:
        raise ValueError("student targets must be teacher observables, in teacher order")
    if not set(student.training_groups) <= set(split.train_groups):
        raise ValueError("a student trains on train groups only")
    if student.utility_metric != protocol.primary_metric:
        raise ValueError("student utility must use the protocol's primary metric")


def decode_student_artifact(raw: bytes) -> StudentArtifact:
    """Decode one student artifact from strict canonical wire bytes."""
    return StudentArtifact.from_wire(_strict_json(raw))
