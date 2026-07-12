# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD effect IR module
# scpn-quantum-control -- Program AD effect IR records and parser
"""Validated Program AD effect-IR records and metadata parser."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class ProgramADSSAValue:
    """One versioned SSA value emitted by program AD graph capture."""

    name: str
    producer: int | None
    version: int
    shape: tuple[int, ...]
    dtype: str
    effect: int | None = None

    def __post_init__(self) -> None:
        """Validate SSA metadata at construction time."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("program AD SSA value name must be a non-empty string")
        if self.producer is not None and self.producer < 0:
            raise ValueError("program AD SSA value producer must be non-negative or None")
        if self.version < 0:
            raise ValueError("program AD SSA value version must be non-negative")
        if any(not isinstance(dimension, int) or dimension < 0 for dimension in self.shape):
            raise ValueError("program AD SSA value shape dimensions must be non-negative ints")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("program AD SSA value dtype must be a non-empty string")
        if self.effect is not None and self.effect < 0:
            raise ValueError("program AD SSA value effect must be non-negative or None")


@dataclass(frozen=True)
class ProgramADEffect:
    """One ordered effect or pure operation in program AD graph capture."""

    index: int
    kind: str
    target: str
    inputs: tuple[str, ...]
    version: int
    ordering: int
    operation: str | None = None

    def __post_init__(self) -> None:
        """Validate effect metadata at construction time."""
        if self.index < 0:
            raise ValueError("program AD effect index must be non-negative")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD effect kind must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("program AD effect target must be a non-empty string")
        if any(not isinstance(item, str) or not item for item in self.inputs):
            raise ValueError("program AD effect inputs must be non-empty strings")
        if self.version < 0:
            raise ValueError("program AD effect version must be non-negative")
        if self.ordering < 0:
            raise ValueError("program AD effect ordering must be non-negative")
        if self.operation is not None and (
            not isinstance(self.operation, str) or not self.operation
        ):
            raise ValueError("program AD effect operation must be non-empty or None")


@dataclass(frozen=True)
class ProgramADAliasEdge:
    """One alias or mutation-version edge in program AD graph capture."""

    source: str
    target: str
    kind: str
    version: int

    def __post_init__(self) -> None:
        """Validate alias-edge metadata at construction time."""
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("program AD alias source must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("program AD alias target must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD alias kind must be a non-empty string")
        if self.version < 0:
            raise ValueError("program AD alias version must be non-negative")


@dataclass(frozen=True)
class ProgramADPhiNode:
    """One metadata-only control-join phi record in program AD graph capture."""

    index: int
    target: str
    incoming: tuple[str, ...]
    control_region: int | None
    selected: str | None
    source_line: int | None

    def __post_init__(self) -> None:
        """Validate phi-node metadata at construction time."""
        if self.index < 0:
            raise ValueError("program AD phi node index must be non-negative")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("program AD phi node target must be a non-empty string")
        if len(self.incoming) < 2:
            raise ValueError(
                "program AD phi node incoming values must contain at least two entries"
            )
        if any(not isinstance(item, str) or not item for item in self.incoming):
            raise ValueError("program AD phi node incoming values must be non-empty strings")
        if self.control_region is not None and self.control_region < 0:
            raise ValueError("program AD phi node control_region must be non-negative or None")
        if self.selected is not None and (not isinstance(self.selected, str) or not self.selected):
            raise ValueError("program AD phi node selected value must be non-empty or None")
        if self.source_line is not None and self.source_line <= 0:
            raise ValueError("program AD phi node source_line must be positive or None")


@dataclass(frozen=True)
class ProgramADControlRegion:
    """One source or runtime control-flow region in program AD graph capture."""

    index: int
    kind: str
    predicate: str | None
    entered: bool
    source_line: int | None

    def __post_init__(self) -> None:
        """Validate control-region metadata at construction time."""
        if self.index < 0:
            raise ValueError("program AD control region index must be non-negative")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD control region kind must be a non-empty string")
        if self.predicate is not None and (
            not isinstance(self.predicate, str) or not self.predicate
        ):
            raise ValueError("program AD control region predicate must be non-empty or None")
        if not isinstance(self.entered, bool):
            raise ValueError("program AD control region entered must be a boolean")
        if self.source_line is not None and self.source_line <= 0:
            raise ValueError("program AD control region source_line must be positive or None")


@dataclass(frozen=True)
class ProgramADEffectIR:
    """Deterministic SSA/effect IR emitted by program AD graph capture."""

    ssa_values: tuple[ProgramADSSAValue, ...]
    effects: tuple[ProgramADEffect, ...]
    alias_edges: tuple[ProgramADAliasEdge, ...]
    control_regions: tuple[ProgramADControlRegion, ...]
    serialization: str
    phi_nodes: tuple[ProgramADPhiNode, ...] = ()

    def __post_init__(self) -> None:
        """Validate effect-IR record contents at construction time."""
        if any(not isinstance(value, ProgramADSSAValue) for value in self.ssa_values):
            raise ValueError("program AD IR ssa_values must contain ProgramADSSAValue entries")
        if any(not isinstance(effect, ProgramADEffect) for effect in self.effects):
            raise ValueError("program AD IR effects must contain ProgramADEffect entries")
        if any(not isinstance(edge, ProgramADAliasEdge) for edge in self.alias_edges):
            raise ValueError("program AD IR alias_edges must contain ProgramADAliasEdge entries")
        if any(not isinstance(region, ProgramADControlRegion) for region in self.control_regions):
            raise ValueError(
                "program AD IR control_regions must contain ProgramADControlRegion entries"
            )
        if any(not isinstance(phi, ProgramADPhiNode) for phi in self.phi_nodes):
            raise ValueError("program AD IR phi_nodes must contain ProgramADPhiNode entries")
        if not isinstance(self.serialization, str) or not self.serialization:
            raise ValueError("program AD IR serialization must be a non-empty string")


def parse_program_ad_effect_ir(serialization: str) -> ProgramADEffectIR:
    """Parse a bounded ``program_ad_effect_ir.v1`` JSON payload.

    This is a metadata round-trip parser for emitted Program AD evidence. It is
    intentionally narrow: unknown formats, malformed rows, and unsupported JSON
    shapes fail closed instead of being treated as compiler frontend input.
    """
    if not isinstance(serialization, str) or not serialization:
        raise ValueError("program AD IR serialization must be a non-empty string")
    try:
        payload = json.loads(serialization)
    except json.JSONDecodeError as exc:
        raise ValueError("program AD IR serialization must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("program AD IR serialization must decode to an object")
    if payload.get("format") != "program_ad_effect_ir.v1":
        raise ValueError("program AD IR format must be program_ad_effect_ir.v1")

    ssa_values = _parse_program_ad_ssa_values(payload.get("ssa_values"))
    effects = _parse_program_ad_effects(payload.get("effects"))
    alias_edges = _parse_program_ad_alias_edges(payload.get("alias_edges"))
    control_regions = _parse_program_ad_control_regions(payload.get("control_regions"))
    phi_nodes = _parse_program_ad_phi_nodes(payload.get("phi_nodes", []))
    _parse_program_ad_bytecode_offsets(payload.get("bytecode_offsets"))
    return ProgramADEffectIR(
        ssa_values=ssa_values,
        effects=effects,
        alias_edges=alias_edges,
        control_regions=control_regions,
        serialization=serialization,
        phi_nodes=phi_nodes,
    )


def _require_program_ad_ir_rows(name: str, value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError(f"program AD IR {name} must be a list")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"program AD IR {name} entries must be objects")
        rows.append(cast(Mapping[str, object], item))
    return tuple(rows)


def _parse_program_ad_optional_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"program AD IR {name} must be an integer or null")
    return value


def _parse_program_ad_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"program AD IR {name} must be an integer")
    return value


def _parse_program_ad_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"program AD IR {name} must be a string")
    return value


def _parse_program_ad_optional_str(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _parse_program_ad_str(name, value)


def _parse_program_ad_str_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"program AD IR {name} must be a list")
    return tuple(_parse_program_ad_str(name, item) for item in value)


def _parse_program_ad_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("program AD IR shape must be a list")
    return tuple(_parse_program_ad_int("shape dimension", item) for item in value)


def _parse_program_ad_ssa_values(value: object) -> tuple[ProgramADSSAValue, ...]:
    return tuple(
        ProgramADSSAValue(
            name=_parse_program_ad_str("ssa value name", row.get("name")),
            producer=_parse_program_ad_optional_int("ssa value producer", row.get("producer")),
            version=_parse_program_ad_int("ssa value version", row.get("version")),
            shape=_parse_program_ad_shape(row.get("shape")),
            dtype=_parse_program_ad_str("ssa value dtype", row.get("dtype")),
            effect=_parse_program_ad_optional_int("ssa value effect", row.get("effect")),
        )
        for row in _require_program_ad_ir_rows("ssa_values", value)
    )


def _parse_program_ad_effects(value: object) -> tuple[ProgramADEffect, ...]:
    return tuple(
        ProgramADEffect(
            index=_parse_program_ad_int("effect index", row.get("index")),
            kind=_parse_program_ad_str("effect kind", row.get("kind")),
            target=_parse_program_ad_str("effect target", row.get("target")),
            inputs=_parse_program_ad_str_tuple("effect inputs", row.get("inputs")),
            version=_parse_program_ad_int("effect version", row.get("version")),
            ordering=_parse_program_ad_int("effect ordering", row.get("ordering")),
            operation=_parse_program_ad_optional_str("effect operation", row.get("operation")),
        )
        for row in _require_program_ad_ir_rows("effects", value)
    )


def _parse_program_ad_alias_edges(value: object) -> tuple[ProgramADAliasEdge, ...]:
    return tuple(
        ProgramADAliasEdge(
            source=_parse_program_ad_str("alias source", row.get("source")),
            target=_parse_program_ad_str("alias target", row.get("target")),
            kind=_parse_program_ad_str("alias kind", row.get("kind")),
            version=_parse_program_ad_int("alias version", row.get("version")),
        )
        for row in _require_program_ad_ir_rows("alias_edges", value)
    )


def _parse_program_ad_control_regions(value: object) -> tuple[ProgramADControlRegion, ...]:
    regions: list[ProgramADControlRegion] = []
    for row in _require_program_ad_ir_rows("control_regions", value):
        entered = row.get("entered")
        if not isinstance(entered, bool):
            raise ValueError("program AD IR control region entered must be a boolean")
        regions.append(
            ProgramADControlRegion(
                index=_parse_program_ad_int("control region index", row.get("index")),
                kind=_parse_program_ad_str("control region kind", row.get("kind")),
                predicate=_parse_program_ad_optional_str(
                    "control region predicate", row.get("predicate")
                ),
                entered=entered,
                source_line=_parse_program_ad_optional_int(
                    "control region source_line", row.get("source_line")
                ),
            )
        )
    return tuple(regions)


def _parse_program_ad_phi_nodes(value: object) -> tuple[ProgramADPhiNode, ...]:
    return tuple(
        ProgramADPhiNode(
            index=_parse_program_ad_int("phi node index", row.get("index")),
            target=_parse_program_ad_str("phi node target", row.get("target")),
            incoming=_parse_program_ad_str_tuple("phi node incoming", row.get("incoming")),
            control_region=_parse_program_ad_optional_int(
                "phi node control_region", row.get("control_region")
            ),
            selected=_parse_program_ad_optional_str("phi node selected", row.get("selected")),
            source_line=_parse_program_ad_optional_int(
                "phi node source_line", row.get("source_line")
            ),
        )
        for row in _require_program_ad_ir_rows("phi_nodes", value)
    )


def _parse_program_ad_bytecode_offsets(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("program AD IR bytecode_offsets must be a list")
    return tuple(_parse_program_ad_int("bytecode offset", item) for item in value)


__all__ = [
    "ProgramADAliasEdge",
    "ProgramADControlRegion",
    "ProgramADEffect",
    "ProgramADEffectIR",
    "ProgramADPhiNode",
    "ProgramADSSAValue",
    "parse_program_ad_effect_ir",
]
