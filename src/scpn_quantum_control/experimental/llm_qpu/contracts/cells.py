# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Scientific cell identity, plans, and raw-result completeness."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .wire import (
    _KERNELS,
    _KEY_FIELDS,
    _MAX_METADATA_BYTES,
    _RESERVED_METADATA,
    SCHEMA,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
)


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
