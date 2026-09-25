# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Causal hidden-state batch descriptors and lineage validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .model import ModelDescriptor
from .task import SplitManifest, TaskSpec, validate_task_split
from .wire import LATENT_SCHEMA, ArrayDescriptor, ArtifactHeader, _digest, _text, canonical_bytes


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
