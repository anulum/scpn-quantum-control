# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Strict, provider-free identities and cell contracts for the opt-in lane."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "scpn.experimental.llm_qpu.planned_cell.v1"
ARRAY_SCHEMA = "scpn.experimental.llm_qpu.array_descriptor.v1"
TASK_SCHEMA = "scpn.experimental.llm_qpu.task_spec.v2"
SPLIT_SCHEMA = "scpn.experimental.llm_qpu.split_manifest.v2"
HEADER_SCHEMA = "scpn.experimental.llm_qpu.artifact_header.v1"
MODEL_SCHEMA = "scpn.experimental.llm_qpu.model_descriptor.v1"
LATENT_SCHEMA = "scpn.experimental.llm_qpu.latent_batch.v1"
_KEY_FIELDS = (
    "experiment_id",
    "source_sample_id",
    "sequence_id",
    "arm_id",
    "basis_id",
    "replicate_id",
    "planned_epoch_id",
    "parameter_binding_id",
)
_KERNELS = frozenset(
    {
        "xy_static_digital_v1",
        "xy_sequence_digital_v1",
        "xy_monitored_sequence_v1",
        "conditional_born_policy_v1",
    }
)
_RESERVED_METADATA = frozenset(
    {"shots", "provider", "device", "job_id", "origin", "approval", "status"}
)
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DTYPE_SIZE = {"<f8": 8, "<f4": 4, "<i8": 8, "<i4": 4, "<u8": 8, "<u4": 4}
_MAX_WIRE_BYTES = 1_048_576
_MAX_ARRAY_BYTES = 16_777_216
_MAX_METADATA_BYTES = 8192


def _positive_int(value: object, *, name: str, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{name} must be a bounded positive integer")
    return value


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be text")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or len(normalized) > 256 or any(ord(ch) < 32 for ch in normalized):
        raise ValueError(f"{name} must be bounded nonempty text")
    return normalized


def f64(value: object) -> dict[str, str]:
    """Tag a finite scientific float without JSON's lossy numeric encoding."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("scientific f64 must be finite and exactly float")
    return {"$f64": (0.0 if value == 0.0 else value).hex()}


def _canonical(value: object, *, depth: int = 0) -> object:
    if depth > 16:
        raise ValueError("wire nesting exceeds limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError("wire integer exceeds signed 64-bit range")
        return value
    if type(value) is str:
        return _text(value, name="wire string")
    if type(value) is float:
        raise ValueError("raw JSON floats are forbidden; use f64")
    if type(value) is list or type(value) is tuple:
        if len(value) > 4096:
            raise ValueError("wire list exceeds item limit")
        return [_canonical(item, depth=depth + 1) for item in value]
    if type(value) is dict:
        if len(value) > 4096:
            raise ValueError("wire object exceeds item limit")
        if set(value) == {"$f64"}:
            token = value["$f64"]
            if type(token) is not str:
                raise ValueError("f64 tag must contain hex text")
            try:
                parsed = float.fromhex(token)
            except ValueError as exc:
                raise ValueError("invalid f64 hex") from exc
            if not math.isfinite(parsed) or f64(parsed)["$f64"] != token:
                raise ValueError("noncanonical or nonfinite f64 hex")
            return {"$f64": token}
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or _FIELD_NAME.fullmatch(key) is None:
                raise ValueError("wire field names must be ASCII identifiers")
            result[key] = _canonical(item, depth=depth + 1)
        return result
    raise ValueError("unsupported wire value")


def canonical_bytes(value: object) -> bytes:
    """Encode only tagged, bounded primitives in deterministic UTF-8 JSON."""
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_WIRE_BYTES:
        raise ValueError("canonical wire exceeds byte limit")
    return encoded


def _strict_json(raw: bytes) -> object:
    if type(raw) is not bytes or len(raw) > _MAX_WIRE_BYTES:
        raise ValueError("wire must be bounded bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def refuse_constant(_: str) -> object:
        raise ValueError("nonfinite JSON")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=refuse_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    if canonical_bytes(value) != raw:
        raise ValueError("noncanonical JSON wire")
    return value


@dataclass(frozen=True, slots=True)
class CellKey:
    """Complete scientific cell identity, independent of provider ordering."""

    experiment_id: str
    source_sample_id: str
    sequence_id: str
    arm_id: str
    basis_id: str
    replicate_id: str
    planned_epoch_id: str
    parameter_binding_id: str

    def __post_init__(self) -> None:
        """Normalize and bound every identity component."""
        for name in _KEY_FIELDS:
            object.__setattr__(self, name, _text(getattr(self, name), name=name))

    def to_wire(self) -> dict[str, str]:
        """Return a detached mapping with every identity component."""
        return {name: getattr(self, name) for name in _KEY_FIELDS}

    @classmethod
    def from_wire(cls, value: object) -> CellKey:
        """Reject missing or unknown identity components."""
        if type(value) is not dict or set(value) != set(_KEY_FIELDS):
            raise ValueError("CellKey fields mismatch")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PlannedCell:
    """Immutable cell and shot plan; metadata is copied before hashing."""

    key: CellKey
    circuit_plan_digest: str
    shots: int
    measurement_width: int
    role: str
    metadata_blob: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Reject invalid plans and authority fields in frozen metadata."""
        if type(self.key) is not CellKey:
            raise ValueError("planned cell requires CellKey")
        _digest(self.circuit_plan_digest, name="circuit plan")
        _positive_int(self.shots, name="shots")
        _positive_int(self.measurement_width, name="measurement width", maximum=64)
        if self.role not in ("data", "sentinel", "calibration"):
            raise ValueError("unknown cell role")
        if type(self.metadata_blob) is not bytes or len(self.metadata_blob) > _MAX_METADATA_BYTES:
            raise ValueError("metadata bytes exceed limit")
        metadata = _strict_json(self.metadata_blob)
        if type(metadata) is not dict or _RESERVED_METADATA.intersection(metadata):
            raise ValueError("metadata shadows a reserved field")

    @classmethod
    def create(
        cls,
        *,
        key: CellKey,
        circuit_plan_digest: str,
        shots: int,
        measurement_width: int,
        role: str,
        metadata: Mapping[str, object],
    ) -> PlannedCell:
        """Deep-copy only approved primitive metadata into immutable bytes."""
        if type(metadata) is not dict:
            raise ValueError("metadata must be a plain mapping")
        return cls(
            key=key,
            circuit_plan_digest=circuit_plan_digest,
            shots=shots,
            measurement_width=measurement_width,
            role=role,
            metadata_blob=canonical_bytes(metadata),
        )

    def to_wire(self) -> dict[str, object]:
        """Return a detached, versioned wire record."""
        return {
            "schema": SCHEMA,
            "object_kind": "planned_cell",
            "key": self.key.to_wire(),
            "circuit_plan_digest": self.circuit_plan_digest,
            "shots": self.shots,
            "measurement_width": self.measurement_width,
            "role": self.role,
            "metadata": json.loads(self.metadata_blob),
        }

    @classmethod
    def from_wire(cls, value: object) -> PlannedCell:
        """Validate schema, field inventory and semantics before use."""
        fields = {
            "schema",
            "object_kind",
            "key",
            "circuit_plan_digest",
            "shots",
            "measurement_width",
            "role",
            "metadata",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("planned cell fields mismatch")
        if value["schema"] != SCHEMA or value["object_kind"] != "planned_cell":
            raise ValueError("unknown planned cell schema or object kind")
        return cls.create(
            key=CellKey.from_wire(value["key"]),
            circuit_plan_digest=value["circuit_plan_digest"],
            shots=value["shots"],
            measurement_width=value["measurement_width"],
            role=value["role"],
            metadata=value["metadata"],
        )


@dataclass(frozen=True, slots=True)
class ArrayDescriptor:
    """Bounded little-endian binary payload descriptor; never loads pickle."""

    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        """Refuse unsupported layouts and mismatched byte counts."""
        if (
            type(self.dtype) is not str
            or self.dtype not in _DTYPE_SIZE
            or type(self.shape) is not tuple
        ):
            raise ValueError("unsupported array dtype or shape")
        if not 1 <= len(self.shape) <= 4:
            raise ValueError("array rank out of bounds")
        elements = 1
        for axis in self.shape:
            elements *= _positive_int(axis, name="array axis")
            if elements > 1_000_000:
                raise ValueError("array element limit exceeded")
        expected = elements * _DTYPE_SIZE[self.dtype]
        if type(self.byte_length) is not int or expected != self.byte_length:
            raise ValueError("array byte length mismatch")
        if expected > _MAX_ARRAY_BYTES:
            raise ValueError("array byte limit exceeded")
        _digest(self.sha256, name="array payload")

    def validate_payload(self, payload: bytes) -> None:
        """Check exact bytes and finite float elements before consumers load."""
        if type(payload) is not bytes or len(payload) != self.byte_length:
            raise ValueError("array payload byte length mismatch")
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError("array payload digest mismatch")
        if self.dtype in ("<f8", "<f4"):
            code = "<d" if self.dtype == "<f8" else "<f"
            if any(not math.isfinite(value[0]) for value in struct.iter_unpack(code, payload)):
                raise ValueError("array contains nonfinite float")

    def to_wire(self) -> dict[str, object]:
        """Return the versioned descriptor without embedding binary bytes."""
        return {
            "schema": ARRAY_SCHEMA,
            "object_kind": "array_descriptor",
            "dtype": self.dtype,
            "shape": list(self.shape),
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }

    @classmethod
    def from_wire(cls, value: object) -> ArrayDescriptor:
        """Reject unknown fields or an incompatible descriptor schema."""
        fields = {"schema", "object_kind", "dtype", "shape", "byte_length", "sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("array descriptor fields mismatch")
        if value["schema"] != ARRAY_SCHEMA or value["object_kind"] != "array_descriptor":
            raise ValueError("unknown array descriptor schema or object kind")
        if type(value["dtype"]) is not str or type(value["sha256"]) is not str:
            raise ValueError("array descriptor dtype and digest must be text")
        if type(value["shape"]) is not list:
            raise ValueError("array shape must be a list")
        return cls(value["dtype"], tuple(value["shape"]), value["byte_length"], value["sha256"])


def content_id(
    *,
    kernel_id: str,
    map_version: str,
    protocol_digest: str,
    scientific_fields: Mapping[str, object],
    output_dir: str | None = None,
) -> str:
    """Hash scientific input only; output location is deliberately excluded."""
    if kernel_id not in _KERNELS:
        raise ValueError("unknown kernel ID")
    _text(map_version, name="map version")
    _digest(protocol_digest, name="protocol")
    if type(scientific_fields) is not dict:
        raise ValueError("scientific fields must be a plain mapping")
    if output_dir is not None and type(output_dir) is not str:
        raise ValueError("output directory must be text")
    payload = {
        "schema": "scpn.experimental.llm_qpu.content_identity.v1",
        "kernel_id": kernel_id,
        "map_version": map_version,
        "protocol_digest": protocol_digest,
        "scientific_fields": scientific_fields,
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def request_id(content_digest: str, cell: PlannedCell, *, kernel_id: str) -> str:
    """Bind kernel, complete cell identity and exact shot plan."""
    _digest(content_digest, name="content")
    if kernel_id not in _KERNELS or type(cell) is not PlannedCell:
        raise ValueError("unknown kernel or planned cell")
    payload = {
        "schema": "scpn.experimental.llm_qpu.request_identity.v1",
        "content_digest": content_digest,
        "kernel_id": kernel_id,
        "cell": cell.to_wire(),
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def run_id(request_digest: str, *, repetition: int, calibration_epoch_id: str) -> str:
    """Bind a planned repetition and calibration epoch to the request."""
    _digest(request_digest, name="request")
    _positive_int(repetition, name="repetition")
    epoch = _text(calibration_epoch_id, name="calibration epoch")
    return hashlib.sha256(
        canonical_bytes(
            {"request_digest": request_digest, "repetition": repetition, "epoch": epoch}
        )
    ).hexdigest()


def attempt_id(run_digest: str, *, nonce: str) -> str:
    """Name one planned submit boundary; caller must journal it first."""
    _digest(run_digest, name="run")
    if type(nonce) is not str:
        raise ValueError("attempt nonce must be UUID text")
    try:
        parsed = uuid.UUID(nonce)
    except ValueError as exc:
        raise ValueError("attempt nonce must be UUID text") from exc
    if parsed.version != 4 or str(parsed) != nonce:
        raise ValueError("attempt nonce must be canonical UUIDv4")
    return hashlib.sha256(canonical_bytes({"run_digest": run_digest, "nonce": nonce})).hexdigest()


def validate_completion(
    expected: Sequence[PlannedCell], received: Sequence[Mapping[str, Any]]
) -> None:
    """Reject missing, duplicate, coerced or shape-mismatched raw results."""
    if not expected or not received or len(expected) != len(received) or len(expected) > 4096:
        raise ValueError("incomplete cell inventory")
    by_key = {canonical_bytes(cell.key.to_wire()): cell for cell in expected}
    if len(by_key) != len(expected):
        raise ValueError("duplicate planned cell")
    seen_keys: set[bytes] = set()
    seen_provider: set[tuple[str, int, int]] = set()
    fields = {
        "key",
        "provider_job_id",
        "pub_index",
        "binding_index",
        "counts",
        "actual_shots",
        "joint_shape",
    }
    for item in received:
        if type(item) is not dict or set(item) != fields:
            raise ValueError("raw result fields mismatch")
        key = canonical_bytes(CellKey.from_wire(item["key"]).to_wire())
        if key not in by_key or key in seen_keys:
            raise ValueError("unknown or duplicate result cell")
        seen_keys.add(key)
        cell = by_key[key]
        job_id = _text(item["provider_job_id"], name="provider job ID")
        pub_index = item["pub_index"]
        binding_index = item["binding_index"]
        if (
            type(pub_index) is not int
            or not 0 <= pub_index <= 1_000_000
            or type(binding_index) is not int
            or not 0 <= binding_index <= 1_000_000
        ):
            raise ValueError("provider indices must be nonnegative integers")
        provider_key = (job_id, pub_index, binding_index)
        if provider_key in seen_provider:
            raise ValueError("provider mapping is not bijective")
        seen_provider.add(provider_key)
        if type(item["actual_shots"]) is not int or item["actual_shots"] != cell.shots:
            raise ValueError("actual shots mismatch")
        counts = item["counts"]
        if type(counts) is not dict or not counts or len(counts) > 4096:
            raise ValueError("raw counts must be nonempty")
        total = 0
        for bits, count in counts.items():
            if (
                type(bits) is not str
                or len(bits) != cell.measurement_width
                or set(bits) - {"0", "1"}
            ):
                raise ValueError("raw bitstring width mismatch")
            if type(count) is not int or not 0 <= count <= cell.shots:
                raise ValueError("raw counts must be nonnegative integers")
            total += count
        if total != cell.shots:
            raise ValueError("raw counts shot sum mismatch")
        shape = item["joint_shape"]
        if shape is not None and (
            type(shape) is not list or shape != [cell.shots, cell.measurement_width]
        ):
            raise ValueError("joint shot shape mismatch")
    if len(seen_keys) != len(expected):
        raise ValueError("incomplete cell inventory")


def decode_contract(
    raw: bytes,
) -> PlannedCell | ArrayDescriptor | TaskSpec | SplitManifest | ModelDescriptor | LatentBatch:
    """Decode only implemented records; all other chapter objects refuse."""
    value = _strict_json(raw)
    if type(value) is not dict:
        raise ValueError("contract wire must be an object")
    if value.get("schema") == SCHEMA:
        return PlannedCell.from_wire(value)
    if value.get("schema") == ARRAY_SCHEMA:
        return ArrayDescriptor.from_wire(value)
    if value.get("schema") == TASK_SCHEMA:
        return TaskSpec.from_wire(value)
    if value.get("schema") == SPLIT_SCHEMA:
        return SplitManifest.from_wire(value)
    if value.get("schema") == MODEL_SCHEMA:
        return ModelDescriptor.from_wire(value)
    if value.get("schema") == LATENT_SCHEMA:
        return LatentBatch.from_wire(value)
    raise ValueError("unsupported contract schema")


@dataclass(frozen=True, slots=True)
class ArtifactHeader:
    """Bind a design artifact to its exact content and code provenance."""

    object_kind: str
    content_digest: str
    parents: tuple[str, ...]
    base_repo_commit: str
    implementation_revision: str
    execution_origin: str
    data_origin: str
    claim_scope: str

    def __post_init__(self) -> None:
        """Refuse unknown origins, revisions and unordered parent lineage."""
        if self.object_kind not in (
            "task_spec",
            "split_manifest",
            "model_descriptor",
            "latent_batch",
        ):
            raise ValueError("unsupported artifact object kind")
        _digest(self.content_digest, name="artifact content")
        if type(self.parents) is not tuple or len(self.parents) > 32:
            raise ValueError("artifact parents must be a bounded tuple")
        for parent in self.parents:
            _digest(parent, name="artifact parent")
        if self.parents != tuple(sorted(set(self.parents))):
            raise ValueError("artifact parents must be sorted and unique")
        for name in ("base_repo_commit", "implementation_revision"):
            value = getattr(self, name)
            if type(value) is not str or _COMMIT.fullmatch(value) is None:
                raise ValueError(f"{name} must be a full lowercase Git commit")
        if self.base_repo_commit == self.implementation_revision:
            raise ValueError("implementation revision must differ from base commit")
        if self.execution_origin != "offline_design":
            raise ValueError("design artifact cannot claim a live execution origin")
        if self.data_origin not in (
            "owner_dataset",
            "external_dataset",
            "synthetic_classical",
            "owner_checkpoint",
        ):
            raise ValueError("unknown artifact data origin")
        if self.claim_scope != "design_only":
            raise ValueError("design artifact cannot claim confirmation")

    def to_wire(self) -> dict[str, object]:
        """Return an exact detached provenance header."""
        return {
            "schema": HEADER_SCHEMA,
            "lane_id": "llm-qpu",
            "object_kind": self.object_kind,
            "content_digest": self.content_digest,
            "parents": list(self.parents),
            "base_repo_commit": self.base_repo_commit,
            "implementation_revision": self.implementation_revision,
            "execution_origin": self.execution_origin,
            "data_origin": self.data_origin,
            "claim_scope": self.claim_scope,
        }

    @classmethod
    def from_wire(cls, value: object) -> ArtifactHeader:
        """Decode only the reviewed v1 header shape."""
        fields = set(cls.__dataclass_fields__) | {"schema", "lane_id"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ArtifactHeader fields mismatch")
        if value["schema"] != HEADER_SCHEMA or value["lane_id"] != "llm-qpu":
            raise ValueError("unknown ArtifactHeader schema or lane")
        if type(value["parents"]) is not list:
            raise ValueError("artifact parents must be a list on wire")
        return cls(
            object_kind=value["object_kind"],
            content_digest=value["content_digest"],
            parents=tuple(value["parents"]),
            base_repo_commit=value["base_repo_commit"],
            implementation_revision=value["implementation_revision"],
            execution_origin=value["execution_origin"],
            data_origin=value["data_origin"],
            claim_scope=value["claim_scope"],
        )

    def validate_content(self, content: Mapping[str, object], *, parents: tuple[str, ...]) -> None:
        """Bind the header to exact scientific bytes and expected lineage."""
        if type(content) is not dict or self.parents != tuple(sorted(parents)):
            raise ValueError("artifact content or parent lineage mismatch")
        if hashlib.sha256(canonical_bytes(content)).hexdigest() != self.content_digest:
            raise ValueError("artifact content digest mismatch")


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


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Design-only identity for a future locally probed hidden-state source."""

    model_id: str
    checkpoint_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    runtime_build_digest: str
    loader_id: str
    quantization: str
    tensor_dtype: str
    block_count: int
    hidden_width: int
    tap_block_index: int
    tap_stream: str
    tap_boundary: str
    probe_evidence_digest: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse inferred dimensions, completion-only taps and missing lineage."""
        for name in ("model_id", "loader_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "checkpoint_digest",
            "tokenizer_digest",
            "chat_template_digest",
            "runtime_build_digest",
            "probe_evidence_digest",
        ):
            _digest(getattr(self, name), name=name)
        if self.quantization not in ("none", "gguf_q6_k", "gguf_q4_k_m"):
            raise ValueError("unknown model quantization")
        if self.tensor_dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError("unknown hidden-state dtype")
        _positive_int(self.block_count, name="model block count", maximum=1024)
        _positive_int(self.hidden_width, name="observed hidden width", maximum=65_536)
        if (
            type(self.tap_block_index) is not int
            or not 0 <= self.tap_block_index < self.block_count
        ):
            raise ValueError("tap block index must be within observed model blocks")
        if self.tap_stream != "residual_hidden_state":
            raise ValueError("completion or embedding fallback is not a hidden-state tap")
        if self.tap_boundary not in ("before_norm", "after_norm"):
            raise ValueError("tap normalization boundary must be explicit")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "model_descriptor"
        ):
            raise ValueError("ModelDescriptor requires its exact artifact header")
        if self.header.data_origin != "owner_checkpoint":
            raise ValueError("ModelDescriptor requires owner-checkpoint origin")
        self.header.validate_content(
            self._scientific_wire(),
            parents=(
                self.checkpoint_digest,
                self.tokenizer_digest,
                self.chat_template_digest,
                self.runtime_build_digest,
                self.probe_evidence_digest,
            ),
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return only the model identity and tap semantics hashed as content."""
        return {
            "schema": MODEL_SCHEMA,
            "object_kind": "model_descriptor",
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "header"
            },
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached versioned model metadata, with no weights or prompts."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> ModelDescriptor:
        """Reject unknown fields and restore the frozen model descriptor."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ModelDescriptor fields mismatch")
        if value["schema"] != MODEL_SCHEMA or value["object_kind"] != "model_descriptor":
            raise ValueError("unknown ModelDescriptor schema")
        return cls(
            **{name: value[name] for name in cls.__dataclass_fields__ if name != "header"},
            header=ArtifactHeader.from_wire(value["header"]),
        )


@dataclass(frozen=True, slots=True)
class LatentBatch:
    """Design-only, ordered hidden-state tensor with explicit causal positions."""

    task_digest: str
    split_digest: str
    model_digest: str
    split_name: str
    layout: str
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    mask: tuple[tuple[bool, ...], ...]
    token_positions: tuple[tuple[int | None, ...], ...]
    answer_start_positions: tuple[tuple[int | None, ...], ...]
    tap_block_index: int
    tap_boundary: str
    tensor: ArrayDescriptor
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Validate local shape, row identity, mask, cutoff and content digest."""
        for name in ("task_digest", "split_digest", "model_digest"):
            _digest(getattr(self, name), name=name)
        if self.split_name not in ("train", "dev", "test"):
            raise ValueError("unknown latent split")
        if self.layout not in ("contextual", "chunk_isolated"):
            raise ValueError("unknown latent layout")
        if type(self.tensor) is not ArrayDescriptor or self.tensor.dtype != "<f4":
            raise ValueError("latent tensor must be little-endian float32")
        shape = self.tensor.shape
        if self.layout == "contextual" and len(shape) != 2:
            raise ValueError("contextual latent tensor must have [N,d] shape")
        if self.layout == "chunk_isolated" and len(shape) != 3:
            raise ValueError("sequence latent tensor must have [N,T,d] shape")
        rows = shape[0]
        steps = 1 if self.layout == "contextual" else shape[1]
        for name in ("sample_ids", "source_ids", "group_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != rows:
                raise ValueError(f"{name} must match latent row count")
            object.__setattr__(self, name, tuple(_text(value, name=name) for value in values))
        if len(set(self.sample_ids)) != rows:
            raise ValueError("duplicate latent sample ID")
        if type(self.lengths) is not tuple or len(self.lengths) != rows:
            raise ValueError("latent lengths must match row count")
        for name in ("mask", "token_positions", "answer_start_positions"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) != rows
                or any(type(row) is not tuple or len(row) != steps for row in values)
            ):
                raise ValueError(f"latent {name} must match tensor steps")
        for index, length in enumerate(self.lengths):
            if type(length) is not int or not 0 < length <= steps:
                raise ValueError("latent length exceeds tensor steps")
            if self.layout == "contextual" and length != 1:
                raise ValueError("contextual latent length must be one")
            for step in range(steps):
                selected = self.token_positions[index][step]
                answer = self.answer_start_positions[index][step]
                active = self.mask[index][step]
                if type(active) is not bool or active != (step < length):
                    raise ValueError("latent mask and length disagree")
                if not active:
                    if selected is not None or answer is not None:
                        raise ValueError("latent padding carries token positions")
                    continue
                if (
                    type(selected) is not int
                    or type(answer) is not int
                    or not 0 <= selected < answer <= 1_000_001
                ):
                    raise ValueError("latent token position reaches answer")
        if type(self.tap_block_index) is not int or not 0 <= self.tap_block_index < 1024:
            raise ValueError("invalid latent tap block index")
        if self.tap_boundary not in ("before_norm", "after_norm"):
            raise ValueError("latent tap boundary missing")
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "latent_batch":
            raise ValueError("LatentBatch requires its exact artifact header")
        self.header.validate_content(
            self._scientific_wire(),
            parents=(self.task_digest, self.split_digest, self.model_digest, self.tensor.sha256),
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the exact tensor interpretation hashed as scientific content."""
        return {
            "schema": LATENT_SCHEMA,
            "object_kind": "latent_batch",
            "task_digest": self.task_digest,
            "split_digest": self.split_digest,
            "model_digest": self.model_digest,
            "split_name": self.split_name,
            "layout": self.layout,
            "sample_ids": list(self.sample_ids),
            "source_ids": list(self.source_ids),
            "group_ids": list(self.group_ids),
            "lengths": list(self.lengths),
            "mask": [list(row) for row in self.mask],
            "token_positions": [list(row) for row in self.token_positions],
            "answer_start_positions": [list(row) for row in self.answer_start_positions],
            "tap_block_index": self.tap_block_index,
            "tap_boundary": self.tap_boundary,
            "tensor": self.tensor.to_wire(),
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached v1 content and provenance."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> LatentBatch:
        """Decode only the frozen field inventory and nested tuple shape."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("LatentBatch fields mismatch")
        if value["schema"] != LATENT_SCHEMA or value["object_kind"] != "latent_batch":
            raise ValueError("unknown LatentBatch schema")
        flat = ("sample_ids", "source_ids", "group_ids", "lengths")
        nested = ("mask", "token_positions", "answer_start_positions")
        if any(type(value[name]) is not list for name in (*flat, *nested)):
            raise ValueError("latent rows must be lists on wire")
        if any(any(type(row) is not list for row in value[name]) for name in nested):
            raise ValueError("latent steps must be lists on wire")
        return cls(
            task_digest=value["task_digest"],
            split_digest=value["split_digest"],
            model_digest=value["model_digest"],
            split_name=value["split_name"],
            layout=value["layout"],
            sample_ids=tuple(value["sample_ids"]),
            source_ids=tuple(value["source_ids"]),
            group_ids=tuple(value["group_ids"]),
            lengths=tuple(value["lengths"]),
            mask=tuple(tuple(row) for row in value["mask"]),
            token_positions=tuple(tuple(row) for row in value["token_positions"]),
            answer_start_positions=tuple(tuple(row) for row in value["answer_start_positions"]),
            tap_block_index=value["tap_block_index"],
            tap_boundary=value["tap_boundary"],
            tensor=ArrayDescriptor.from_wire(value["tensor"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_latent_batch(
    task: TaskSpec,
    split: SplitManifest,
    model: ModelDescriptor,
    batch: LatentBatch,
    payload: bytes,
) -> None:
    """Bind a private finite tensor to frozen task, split and model records."""
    if (
        type(task) is not TaskSpec
        or type(split) is not SplitManifest
        or type(model) is not ModelDescriptor
        or type(batch) is not LatentBatch
    ):
        raise ValueError("latent validation requires frozen contracts")
    validate_task_split(task, split)
    for name, record in (("task", task), ("split", split), ("model", model)):
        expected = hashlib.sha256(canonical_bytes(record.to_wire())).hexdigest()
        if getattr(batch, f"{name}_digest") != expected:
            raise ValueError(f"latent {name} digest mismatch")
    if (
        batch.header.data_origin != task.source_kind
        or split.header.data_origin != task.source_kind
    ):
        raise ValueError("latent data origin mismatch")
    if model.tensor_dtype != "float32" or batch.tensor.shape[-1] != model.hidden_width:
        raise ValueError("latent dtype or hidden width mismatch")
    if batch.tap_block_index != model.tap_block_index or batch.tap_boundary != model.tap_boundary:
        raise ValueError("latent tap differs from model descriptor")
    allowed_groups = getattr(split, f"{batch.split_name}_groups")
    if any(group not in allowed_groups for group in batch.group_ids):
        raise ValueError("latent group outside frozen split")
    for positions in batch.token_positions:
        if any(position is not None and position > task.causal_cutoff for position in positions):
            raise ValueError("latent token position exceeds causal cutoff")
    batch.tensor.validate_payload(payload)
    if batch.layout == "chunk_isolated":
        _, steps, width = batch.tensor.shape
        row_bytes = width * 4
        for index, length in enumerate(batch.lengths):
            for step in range(length, steps):
                offset = (index * steps + step) * row_bytes
                if payload[offset : offset + row_bytes] != bytes(row_bytes):
                    raise ValueError("latent padding must be zero")
