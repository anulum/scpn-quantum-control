# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Frozen compiled bundles bound to one target snapshot and one SDK lock."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

from .target import TargetSnapshot
from .wire import (
    COMPILED_BUNDLE_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
)

_CELL_FIELDS = (
    "planned_cell_digest",
    "logical_plan_digest",
    "compiled_payload_digest",
    "binding_digest",
)
_OPTION_TYPES = (int, str, bool)


@dataclass(frozen=True, slots=True)
class CompiledCell:
    """One planned cell with its logical plan, compiled payload, binding and shots."""

    planned_cell_digest: str
    logical_plan_digest: str
    compiled_payload_digest: str
    binding_digest: str
    shots: int

    def __post_init__(self) -> None:
        """Require four content digests and bounded integer shots."""
        for name in _CELL_FIELDS:
            _digest(getattr(self, name), name=name)
        _positive_int(self.shots, name="compiled cell shots", maximum=10_000_000)

    def to_wire(self) -> dict[str, object]:
        """Return the exact cell binding."""
        return {**{name: getattr(self, name) for name in _CELL_FIELDS}, "shots": self.shots}

    @classmethod
    def from_wire(cls, value: object) -> CompiledCell:
        """Reject a cell binding with missing or extra fields."""
        if type(value) is not dict or set(value) != {*_CELL_FIELDS, "shots"}:
            raise ValueError("CompiledCell fields mismatch")
        return cls(**{name: value[name] for name in (*_CELL_FIELDS, "shots")})


@dataclass(frozen=True, slots=True)
class CompiledBundle:
    """Design-only compiled payload for one target; no field may change after approval."""

    target_snapshot_digest: str
    provider: str
    device_name: str
    sdk_lock_digest: str
    cells: tuple[CompiledCell, ...]
    layout: tuple[int, ...]
    logical_gates: tuple[str, ...]
    compiled_gate_counts: tuple[tuple[str, int], ...]
    substitutions: tuple[tuple[str, tuple[str, ...]], ...]
    compiled_depth: int
    compile_log_digest: str
    provider_options: tuple[tuple[str, int | str | bool], ...]
    payload_digest: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse targetless compiles, unreported substitutions and edited payloads."""
        for name in ("target_snapshot_digest", "sdk_lock_digest", "compile_log_digest"):
            _digest(getattr(self, name), name=name)
        object.__setattr__(self, "provider", _text(self.provider, name="provider"))
        object.__setattr__(self, "device_name", _text(self.device_name, name="device name"))
        for name in ("compiled_gate_counts", "substitutions", "provider_options"):
            _require_pairs(getattr(self, name), name=name)
        self._check_cells_and_layout()
        self._check_gates()
        _positive_int(self.compiled_depth, name="compiled depth", maximum=10_000_000)
        self._check_options()
        if self.payload_digest != _payload_digest(self):
            raise ValueError("compiled payload digest does not match the bundle")
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "compiled_bundle":
            raise ValueError("CompiledBundle requires its exact artifact header")
        if self.header.data_origin not in ("provider_metadata", "synthetic_classical"):
            raise ValueError("a compiled bundle derives from provider metadata or a fixture")
        self.header.validate_content(
            self._scientific_wire(),
            parents=(self.target_snapshot_digest, self.sdk_lock_digest, self.compile_log_digest),
        )

    def _check_cells_and_layout(self) -> None:
        """Require ordered unique cells and an injective physical layout."""
        if type(self.cells) is not tuple or not self.cells or len(self.cells) > 4096:
            raise ValueError("compiled cells must be a bounded nonempty tuple")
        if any(type(cell) is not CompiledCell for cell in self.cells):
            raise ValueError("compiled cells must be CompiledCell records")
        planned = [cell.planned_cell_digest for cell in self.cells]
        if len(set(planned)) != len(planned):
            raise ValueError("a planned cell appears twice in the bundle")
        if type(self.layout) is not tuple or not self.layout:
            raise ValueError("layout must be a nonempty tuple of physical qubits")
        if any(type(qubit) is not int or qubit < 0 for qubit in self.layout):
            raise ValueError("layout entries must be non-negative physical qubits")
        if len(set(self.layout)) != len(self.layout):
            raise ValueError("layout must map logical qubits to distinct physical qubits")

    def _check_gates(self) -> None:
        """Require every compiled gate to be logical or a declared substitution."""
        logical = _sorted_names(self.logical_gates, name="logical gates")
        if type(self.compiled_gate_counts) is not tuple or not self.compiled_gate_counts:
            raise ValueError("compiled gate counts must be a nonempty tuple")
        names = tuple(name for name, _ in self.compiled_gate_counts)
        _sorted_names(names, name="compiled gates")
        for _, count in self.compiled_gate_counts:
            _positive_int(count, name="compiled gate count", maximum=10**9)
        replaced = tuple(source for source, _ in self.substitutions)
        _sorted_names(replaced, name="substituted gates", allow_empty=True)
        introduced: set[str] = set()
        for source, targets in self.substitutions:
            if source not in logical:
                raise ValueError("a substitution must replace a logical gate")
            introduced.update(_sorted_names(targets, name="substitution targets"))
        unreported = set(names) - set(logical) - introduced
        if unreported:
            raise ValueError(f"unreported gate substitution: {', '.join(sorted(unreported))}")
        missing = set(logical) - set(names) - set(replaced)
        if missing:
            raise ValueError(
                f"logical gate vanished without substitution: {', '.join(sorted(missing))}"
            )

    def _check_options(self) -> None:
        """Require sorted, uniquely named scalar provider options without shot overrides."""
        if type(self.provider_options) is not tuple or len(self.provider_options) > 64:
            raise ValueError("provider options must be a bounded tuple")
        keys = tuple(key for key, _ in self.provider_options)
        _sorted_names(keys, name="provider option names", allow_empty=True)
        if "shots" in keys:
            raise ValueError("shots belong to compiled cells, not provider options")
        for _, value in self.provider_options:
            if type(value) not in _OPTION_TYPES:
                raise ValueError("provider option values must be integers, text or booleans")
            if type(value) is str:
                _text(value, name="provider option value")

    def _scientific_wire(self) -> dict[str, object]:
        """Return the compiled payload identity hashed as content."""
        return {
            "schema": COMPILED_BUNDLE_SCHEMA,
            "object_kind": "compiled_bundle",
            "target_snapshot_digest": self.target_snapshot_digest,
            "provider": self.provider,
            "device_name": self.device_name,
            "sdk_lock_digest": self.sdk_lock_digest,
            "compile_log_digest": self.compile_log_digest,
            "payload_digest": self.payload_digest,
            **_payload_wire(self),
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached versioned bundle metadata; compiled payloads stay private."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> CompiledBundle:
        """Reject unknown fields and restore the frozen bundle."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("CompiledBundle fields mismatch")
        if value["schema"] != COMPILED_BUNDLE_SCHEMA or value["object_kind"] != "compiled_bundle":
            raise ValueError("unknown CompiledBundle schema")
        lists = ("cells", "layout", "logical_gates", "compiled_gate_counts")
        if any(
            type(value[name]) is not list for name in (*lists, "substitutions", "provider_options")
        ):
            raise ValueError("bundle sequences must be lists on wire")
        return cls(
            target_snapshot_digest=value["target_snapshot_digest"],
            provider=value["provider"],
            device_name=value["device_name"],
            sdk_lock_digest=value["sdk_lock_digest"],
            cells=tuple(CompiledCell.from_wire(cell) for cell in value["cells"]),
            layout=tuple(value["layout"]),
            logical_gates=tuple(value["logical_gates"]),
            compiled_gate_counts=cast(
                "tuple[tuple[str, int], ...]", _pairs(value["compiled_gate_counts"])
            ),
            substitutions=tuple(
                (cast(str, pair[0]), tuple(cast("list[str]", pair[1])))
                for pair in _pairs(value["substitutions"], nested=True)
            ),
            compiled_depth=value["compiled_depth"],
            compile_log_digest=value["compile_log_digest"],
            provider_options=cast(
                "tuple[tuple[str, int | str | bool], ...]", _pairs(value["provider_options"])
            ),
            payload_digest=value["payload_digest"],
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _require_pairs(values: object, *, name: str) -> None:
    """Refuse a sequence that is not a tuple of two-item tuples."""
    if type(values) is not tuple or any(
        type(item) is not tuple or len(item) != 2 for item in values
    ):
        raise ValueError(f"{name} must be a tuple of pairs")


def _sorted_names(values: object, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    """Return a sorted unique tuple of bounded names, or refuse."""
    if type(values) is not tuple or (not values and not allow_empty):
        raise ValueError(f"{name} must be a nonempty tuple")
    texts = tuple(_text(value, name=name) for value in values)
    if texts != tuple(sorted(set(texts))):
        raise ValueError(f"{name} must be sorted and unique")
    return texts


def _pairs(values: list[object], *, nested: bool = False) -> tuple[tuple[object, ...], ...]:
    """Return wire pairs as tuples, refusing anything that is not a two-item list."""
    pairs: list[tuple[object, ...]] = []
    for item in values:
        if type(item) is not list or len(item) != 2 or (nested and type(item[1]) is not list):
            raise ValueError("bundle pairs must be two-item lists on wire")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _payload_wire(bundle: CompiledBundle) -> dict[str, object]:
    """Return the ordered fields a later approval binds byte for byte."""
    return {
        "cells": [cell.to_wire() for cell in bundle.cells],
        "layout": list(bundle.layout),
        "logical_gates": list(bundle.logical_gates),
        "compiled_gate_counts": [list(pair) for pair in bundle.compiled_gate_counts],
        "substitutions": [[source, list(targets)] for source, targets in bundle.substitutions],
        "compiled_depth": bundle.compiled_depth,
        "provider_options": [list(pair) for pair in bundle.provider_options],
    }


def _payload_digest(bundle: CompiledBundle) -> str:
    """Return the SHA-256 of the ordered approvable payload."""
    return hashlib.sha256(canonical_bytes(_payload_wire(bundle))).hexdigest()


def compiled_payload_digest(bundle: CompiledBundle) -> str:
    """Return the digest an approval must quote for this exact bundle payload."""
    return _payload_digest(bundle)


def validate_compiled_bundle(bundle: CompiledBundle, target: TargetSnapshot) -> None:
    """Refuse a bundle compiled for another target or outside its capabilities."""
    if type(bundle) is not CompiledBundle or type(target) is not TargetSnapshot:
        raise ValueError("bundle validation needs a CompiledBundle and its TargetSnapshot")
    target_digest = hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest()
    if bundle.target_snapshot_digest != target_digest:
        raise ValueError("bundle was compiled against a different target snapshot")
    if (bundle.provider, bundle.device_name) != (target.provider, target.device_name):
        raise ValueError("bundle provider or device differs from its target")
    if max(bundle.layout) >= target.num_qubits:
        raise ValueError("layout uses a qubit the target does not have")
    outside = {name for name, _ in bundle.compiled_gate_counts} - set(target.basis_gates)
    if outside:
        raise ValueError(f"compiled gates outside the target basis: {', '.join(sorted(outside))}")
    if bundle.header.data_origin != target.header.data_origin:
        raise ValueError("bundle and target data origins differ")
    limit = target.max_shots
    if limit.state == "known" and max(cell.shots for cell in bundle.cells) > int(
        limit.require(name="max_shots")
    ):
        raise ValueError("a compiled cell requests more shots than the target allows")


def decode_compiled_bundle(raw: bytes) -> CompiledBundle:
    """Decode one compiled bundle from strict canonical wire bytes."""
    return CompiledBundle.from_wire(_strict_json(raw))
