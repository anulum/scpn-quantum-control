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


def decode_contract(raw: bytes) -> PlannedCell | ArrayDescriptor:
    """Decode only implemented records; all other chapter objects refuse."""
    value = _strict_json(raw)
    if type(value) is not dict:
        raise ValueError("contract wire must be an object")
    if value.get("schema") == SCHEMA:
        return PlannedCell.from_wire(value)
    if value.get("schema") == ARRAY_SCHEMA:
        return ArrayDescriptor.from_wire(value)
    raise ValueError("unsupported contract schema")
