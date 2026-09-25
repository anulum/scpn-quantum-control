# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Radian latent batches and exact frozen-map validation."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from .compressor import CompressorArtifact, _positive_float_hex, validate_compressor_artifact
from .latent import LatentBatch, validate_latent_batch
from .model import ModelDescriptor
from .task import SplitManifest, TaskSpec
from .wire import (
    COMPRESSED_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    _digest,
    _text,
    canonical_bytes,
)


@dataclass(frozen=True, slots=True)
class CompressedLatentBatch:
    """Retain causal row identities for bounded radian input angles."""

    latent_digest: str
    compressor_digest: str
    split_name: str
    layout: str
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    mask: tuple[tuple[bool, ...], ...]
    token_positions: tuple[tuple[int | None, ...], ...]
    answer_start_positions: tuple[tuple[int | None, ...], ...]
    angle_unit: str
    comparison_tolerance_hex: str
    tensor: ArrayDescriptor
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Validate radian tensor identity, causal rows and exact header."""
        _digest(self.latent_digest, name="source latent")
        _digest(self.compressor_digest, name="compressor")
        if self.split_name not in ("train", "dev", "test"):
            raise ValueError("unknown compressed split")
        if self.layout not in ("contextual", "chunk_isolated"):
            raise ValueError("unknown compressed layout")
        if self.angle_unit != "radian":
            raise ValueError("compressed angles must be radians")
        _positive_float_hex(self.comparison_tolerance_hex, name="angle tolerance", maximum=1e-5)
        if type(self.tensor) is not ArrayDescriptor or self.tensor.dtype != "<f4":
            raise ValueError("compressed tensor must be little-endian float32")
        shape = self.tensor.shape
        if len(shape) != (2 if self.layout == "contextual" else 3):
            raise ValueError("compressed tensor rank differs from layout")
        rows = shape[0]
        steps = 1 if self.layout == "contextual" else shape[1]
        for name in ("sample_ids", "source_ids", "group_ids", "lengths"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != rows:
                raise ValueError("compressed row identity length mismatch")
        for name in ("sample_ids", "source_ids", "group_ids"):
            object.__setattr__(
                self, name, tuple(_text(item, name=name) for item in getattr(self, name))
            )
        if len(set(self.sample_ids)) != rows:
            raise ValueError("duplicate compressed sample ID")
        for name in ("mask", "token_positions", "answer_start_positions"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) != rows
                or any(type(row) is not tuple or len(row) != steps for row in values)
            ):
                raise ValueError("compressed step identity length mismatch")
        for index, length in enumerate(self.lengths):
            if type(length) is not int or not 0 < length <= steps:
                raise ValueError("compressed length exceeds steps")
            if self.layout == "contextual" and length != 1:
                raise ValueError("contextual compressed length must be one")
            for step in range(steps):
                active = self.mask[index][step]
                selected = self.token_positions[index][step]
                answer = self.answer_start_positions[index][step]
                if type(active) is not bool or active != (step < length):
                    raise ValueError("compressed mask and length disagree")
                if not active and (selected is not None or answer is not None):
                    raise ValueError("compressed padding carries token positions")
                if active and (
                    type(selected) is not int
                    or type(answer) is not int
                    or not 0 <= selected < answer <= 1_000_001
                ):
                    raise ValueError("compressed token position reaches answer")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "compressed_latent_batch"
        ):
            raise ValueError("CompressedLatentBatch requires exact header")
        self.header.validate_content(
            self._scientific_wire(),
            parents=(self.latent_digest, self.compressor_digest, self.tensor.sha256),
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return exact compressed row content hashed by its header."""
        return {
            "schema": COMPRESSED_SCHEMA,
            "object_kind": "compressed_latent_batch",
            "latent_digest": self.latent_digest,
            "compressor_digest": self.compressor_digest,
            "split_name": self.split_name,
            "layout": self.layout,
            "sample_ids": list(self.sample_ids),
            "source_ids": list(self.source_ids),
            "group_ids": list(self.group_ids),
            "lengths": list(self.lengths),
            "mask": [list(row) for row in self.mask],
            "token_positions": [list(row) for row in self.token_positions],
            "answer_start_positions": [list(row) for row in self.answer_start_positions],
            "angle_unit": self.angle_unit,
            "comparison_tolerance_hex": self.comparison_tolerance_hex,
            "tensor": self.tensor.to_wire(),
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached radian batch descriptors and provenance."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> CompressedLatentBatch:
        """Decode only the exact radian batch field inventory."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("CompressedLatentBatch fields mismatch")
        if (
            value["schema"] != COMPRESSED_SCHEMA
            or value["object_kind"] != "compressed_latent_batch"
        ):
            raise ValueError("unknown CompressedLatentBatch schema")
        flat = ("sample_ids", "source_ids", "group_ids", "lengths")
        nested = ("mask", "token_positions", "answer_start_positions")
        if any(type(value[name]) is not list for name in (*flat, *nested)) or any(
            any(type(row) is not list for row in value[name]) for name in nested
        ):
            raise ValueError("compressed rows and steps must be lists on wire")
        return cls(
            latent_digest=value["latent_digest"],
            compressor_digest=value["compressor_digest"],
            split_name=value["split_name"],
            layout=value["layout"],
            sample_ids=tuple(value["sample_ids"]),
            source_ids=tuple(value["source_ids"]),
            group_ids=tuple(value["group_ids"]),
            lengths=tuple(value["lengths"]),
            mask=tuple(tuple(row) for row in value["mask"]),
            token_positions=tuple(tuple(row) for row in value["token_positions"]),
            answer_start_positions=tuple(tuple(row) for row in value["answer_start_positions"]),
            angle_unit=value["angle_unit"],
            comparison_tolerance_hex=value["comparison_tolerance_hex"],
            tensor=ArrayDescriptor.from_wire(value["tensor"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_compressed_latent_batch(
    task: TaskSpec,
    split: SplitManifest,
    model: ModelDescriptor,
    latent: LatentBatch,
    compressor: CompressorArtifact,
    compressed: CompressedLatentBatch,
    latent_payload: bytes,
    centering_payload: bytes,
    projection_payload: bytes,
    scales_payload: bytes,
    angles_payload: bytes,
) -> None:
    """Recompute every active angle from private bytes and refuse row drift."""
    if type(compressed) is not CompressedLatentBatch:
        raise ValueError("compressed validation requires frozen contract")
    validate_latent_batch(task, split, model, latent, latent_payload)
    validate_compressor_artifact(
        split, model, compressor, centering_payload, projection_payload, scales_payload
    )
    if compressed.latent_digest != hashlib.sha256(canonical_bytes(latent.to_wire())).hexdigest():
        raise ValueError("compressed source latent digest mismatch")
    if (
        compressed.compressor_digest
        != hashlib.sha256(canonical_bytes(compressor.to_wire())).hexdigest()
    ):
        raise ValueError("compressed compressor digest mismatch")
    for name in (
        "split_name",
        "layout",
        "sample_ids",
        "source_ids",
        "group_ids",
        "lengths",
        "mask",
        "token_positions",
        "answer_start_positions",
    ):
        if getattr(compressed, name) != getattr(latent, name):
            raise ValueError(f"compressed {name} differs from source latent")
    if compressed.tensor.shape != (*latent.tensor.shape[:-1], compressor.output_width):
        raise ValueError("compressed output shape mismatch")
    if compressed.header.data_origin != latent.header.data_origin:
        raise ValueError("compressed data origin mismatch")
    compressed.tensor.validate_payload(angles_payload)
    source_values = tuple(item[0] for item in struct.iter_unpack("<f", latent_payload))
    centers = tuple(item[0] for item in struct.iter_unpack("<f", centering_payload))
    projection = tuple(item[0] for item in struct.iter_unpack("<f", projection_payload))
    scales = tuple(item[0] for item in struct.iter_unpack("<f", scales_payload))
    angles = tuple(item[0] for item in struct.iter_unpack("<f", angles_payload))
    width = compressor.input_width
    output_width = compressor.output_width
    steps = 1 if latent.layout == "contextual" else latent.tensor.shape[1]
    epsilon = _positive_float_hex(compressor.epsilon_hex, name="epsilon", maximum=1.0)
    tolerance = _positive_float_hex(
        compressed.comparison_tolerance_hex, name="angle tolerance", maximum=1e-5
    )
    for row_index, length in enumerate(latent.lengths):
        for step in range(steps):
            flat_row = row_index * steps + step
            source = source_values[flat_row * width : (flat_row + 1) * width]
            output = angles[flat_row * output_width : (flat_row + 1) * output_width]
            if step >= length:
                if any(value != 0.0 for value in output):
                    raise ValueError("compressed padding must be zero")
                continue
            rms = math.sqrt(math.fsum(value * value for value in source) / width + epsilon)
            normalized = tuple(value / rms - centers[index] for index, value in enumerate(source))
            for column, actual in enumerate(output):
                if not -math.pi / 2 < actual < math.pi / 2:
                    raise ValueError("compressed angle outside open radian interval")
                projected = math.fsum(
                    projection[index * output_width + column] * normalized[index]
                    for index in range(width)
                )
                expected = (math.pi / 2) * math.tanh(projected / scales[column])
                if abs(actual - expected) > tolerance:
                    raise ValueError("compressed angle differs from frozen map")
