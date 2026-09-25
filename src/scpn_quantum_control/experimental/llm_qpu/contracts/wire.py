# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Canonical wire, arrays, and artifact provenance for LLM-QPU contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA = "scpn.experimental.llm_qpu.planned_cell.v1"
ARRAY_SCHEMA = "scpn.experimental.llm_qpu.array_descriptor.v1"
TASK_SCHEMA = "scpn.experimental.llm_qpu.task_spec.v2"
SPLIT_SCHEMA = "scpn.experimental.llm_qpu.split_manifest.v2"
HEADER_SCHEMA = "scpn.experimental.llm_qpu.artifact_header.v1"
MODEL_SCHEMA = "scpn.experimental.llm_qpu.model_descriptor.v1"
LATENT_SCHEMA = "scpn.experimental.llm_qpu.latent_batch.v1"
COMPRESSOR_SCHEMA = "scpn.experimental.llm_qpu.compressor_artifact.v1"
COMPRESSED_SCHEMA = "scpn.experimental.llm_qpu.compressed_latent_batch.v1"
RESERVOIR_SCHEMA = "scpn.experimental.llm_qpu.reservoir_spec.v1"
STATIC_PLAN_SCHEMA = "scpn.experimental.llm_qpu.static_circuit_plan.v1"
MEASUREMENT_PLAN_SCHEMA = "scpn.experimental.llm_qpu.measurement_plan.v1"
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
            "compressor_artifact",
            "compressed_latent_batch",
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
