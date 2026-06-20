# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD effect IR tests
"""Tests for Program AD effect-IR emission, serialization, and validation."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import program_ad_effect_ir as effect_ir_module
from scpn_quantum_control.differentiable import (
    Parameter,
    ProgramADAliasEdge,
    ProgramADControlRegion,
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADPhiNode,
    ProgramADSSAValue,
    analyze_program_ad_alias_effects,
    parse_program_ad_effect_ir,
    whole_program_value_and_grad,
)


def test_program_ad_effect_ir_exports_stable_facade_identities() -> None:
    """Program AD effect-IR records should share identities across public surfaces."""

    assert ProgramADSSAValue is effect_ir_module.ProgramADSSAValue
    assert ProgramADEffect is effect_ir_module.ProgramADEffect
    assert ProgramADAliasEdge is effect_ir_module.ProgramADAliasEdge
    assert ProgramADPhiNode is effect_ir_module.ProgramADPhiNode
    assert ProgramADControlRegion is effect_ir_module.ProgramADControlRegion
    assert ProgramADEffectIR is effect_ir_module.ProgramADEffectIR
    assert parse_program_ad_effect_ir is effect_ir_module.parse_program_ad_effect_ir
    assert scpn.ProgramADSSAValue is ProgramADSSAValue
    assert scpn.ProgramADEffect is ProgramADEffect
    assert scpn.ProgramADAliasEdge is ProgramADAliasEdge
    assert scpn.ProgramADPhiNode is ProgramADPhiNode
    assert scpn.ProgramADControlRegion is ProgramADControlRegion
    assert scpn.ProgramADEffectIR is ProgramADEffectIR
    assert scpn.parse_program_ad_effect_ir is parse_program_ad_effect_ir


def test_whole_program_ad_emits_deterministic_ssa_effect_ir() -> None:
    """Program AD should expose deterministic SSA, alias, mutation, and control metadata."""

    def objective(values: Any) -> object:
        alias = values.copy()
        total = values[0]
        for index in range(1, 3):
            total = total + alias[index] * float(index)
        if total > 0.0:
            alias[0] = total
        return alias[0] + np.sin(values[2])

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )

    assert result.program_ir is not None
    assert result.program_ir.serialization.startswith('{"alias_edges"')
    assert "program_ad_effect_ir.v1" in result.program_ir.serialization
    assert result.program_ir.ssa_values[0] == ProgramADSSAValue(
        name="%0",
        producer=0,
        version=0,
        shape=(),
        dtype="float64",
        effect=0,
    )
    assert [effect.ordering for effect in result.program_ir.effects] == list(
        range(len(result.program_ir.effects))
    )
    assert any(effect.kind == "mutation" for effect in result.program_ir.effects)
    assert any(effect.kind == "control_branch" for effect in result.program_ir.effects)
    assert any(edge.kind == "mutation_version" for edge in result.program_ir.alias_edges)
    assert result.program_ir.alias_edges
    assert any(region.kind == "runtime_branch" for region in result.program_ir.control_regions)
    assert any(region.kind.startswith("source_") for region in result.program_ir.control_regions)
    assert result.program_ir.phi_nodes
    assert any(phi.target.startswith("phi:runtime_branch") for phi in result.program_ir.phi_nodes)
    assert any(phi.target.startswith("phi:source:") for phi in result.program_ir.phi_nodes)
    assert all(phi.control_region is not None for phi in result.program_ir.phi_nodes)
    np.testing.assert_allclose(
        result.gradient,
        [1.0, 1.0, 2.0 + math.cos(0.75)],
        atol=1.0e-12,
    )


def test_program_ad_effect_ir_serialization_round_trips_metadata() -> None:
    """Program AD effect IR serialization should parse back into validated records."""

    def objective(values: Any) -> object:
        total = values[0]
        if values[1] > 0.0:
            total = total + np.sin(values[1])
        return total + values[2]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None

    parsed = parse_program_ad_effect_ir(result.program_ir.serialization)

    assert parsed.ssa_values == result.program_ir.ssa_values
    assert parsed.effects == result.program_ir.effects
    assert parsed.alias_edges == result.program_ir.alias_edges
    assert parsed.control_regions == result.program_ir.control_regions
    assert parsed.phi_nodes == result.program_ir.phi_nodes
    assert parsed.serialization == result.program_ir.serialization
    assert analyze_program_ad_alias_effects(parsed).claim_boundary == (
        "metadata_only_no_general_alias_lattice"
    )


def test_program_ad_effect_ir_parser_fails_closed_on_malformed_payloads() -> None:
    """Program AD effect IR parsing should reject unsupported or malformed metadata."""

    valid: dict[str, Any] = {
        "format": "program_ad_effect_ir.v1",
        "ssa_values": [
            {
                "name": "%0",
                "producer": 0,
                "version": 0,
                "shape": [],
                "dtype": "float64",
                "effect": 0,
            }
        ],
        "effects": [
            {
                "index": 0,
                "kind": "pure",
                "target": "%0",
                "inputs": [],
                "version": 0,
                "ordering": 0,
            }
        ],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [
            {
                "index": 0,
                "target": "phi:source:control_flow:1",
                "incoming": ["executed_path", "non_executed_path"],
                "control_region": None,
                "selected": "executed_path",
                "source_line": 1,
            }
        ],
        "bytecode_offsets": [0],
    }
    parsed = parse_program_ad_effect_ir(json.dumps(valid, sort_keys=True, separators=(",", ":")))
    assert parsed.ssa_values[0].name == "%0"
    assert parsed.phi_nodes[0] == ProgramADPhiNode(
        index=0,
        target="phi:source:control_flow:1",
        incoming=("executed_path", "non_executed_path"),
        control_region=None,
        selected="executed_path",
        source_line=1,
    )

    malformed_payloads: tuple[tuple[str | dict[str, Any], str], ...] = (
        ("", "non-empty"),
        ("not-json", "valid JSON"),
        ("[]", "decode to an object"),
        ({**valid, "format": "program_ad_effect_ir.v2"}, "format"),
        ({**valid, "ssa_values": {}}, "ssa_values"),
        ({**valid, "ssa_values": [{**valid["ssa_values"][0], "shape": [True]}]}, "integer"),
        ({**valid, "effects": [{**valid["effects"][0], "inputs": {}}]}, "inputs"),
        ({**valid, "control_regions": [{"entered": "yes"}]}, "entered"),
        (
            {
                **valid,
                "phi_nodes": [
                    {
                        **valid["phi_nodes"][0],
                        "incoming": ["only_one"],
                    }
                ],
            },
            "incoming values",
        ),
        ({**valid, "bytecode_offsets": ["zero"]}, "bytecode offset"),
    )
    for payload, message in malformed_payloads:
        serialized = payload if isinstance(payload, str) else json.dumps(payload)
        with pytest.raises(ValueError, match=message):
            parse_program_ad_effect_ir(serialized)


def test_program_ad_effect_ir_validation_paths() -> None:
    """Program AD IR dataclasses should fail closed on malformed compiler metadata."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=("theta",),
        version=0,
        ordering=0,
    )
    edge = ProgramADAliasEdge(source="alias", target="%0", kind="source_alias", version=0)
    region = ProgramADControlRegion(
        index=0,
        kind="runtime_branch",
        predicate="%0:gt:0.0",
        entered=True,
        source_line=None,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(edge,),
        control_regions=(region,),
        serialization="program_ad_effect_ir.v1",
        phi_nodes=(
            ProgramADPhiNode(
                index=0,
                target="phi:runtime_branch:0",
                incoming=("executed_true", "executed_false"),
                control_region=0,
                selected="executed_true",
                source_line=None,
            ),
        ),
    )

    assert ir.ssa_values == (value,)
    assert ir.phi_nodes[0].selected == "executed_true"
    with pytest.raises(ValueError, match="SSA value name"):
        ProgramADSSAValue("", producer=0, version=0, shape=(), dtype="float64")
    with pytest.raises(ValueError, match="effect kind"):
        ProgramADEffect(index=0, kind="", target="%0", inputs=(), version=0, ordering=0)
    with pytest.raises(ValueError, match="alias source"):
        ProgramADAliasEdge(source="", target="%0", kind="source_alias", version=0)
    with pytest.raises(ValueError, match="control region kind"):
        ProgramADControlRegion(index=0, kind="", predicate=None, entered=True, source_line=None)
    with pytest.raises(ValueError, match="incoming values"):
        ProgramADPhiNode(
            index=0,
            target="phi:bad",
            incoming=("only_one",),
            control_region=None,
            selected=None,
            source_line=None,
        )
    with pytest.raises(ValueError, match="serialization"):
        ProgramADEffectIR(
            ssa_values=(value,),
            effects=(effect,),
            alias_edges=(edge,),
            control_regions=(region,),
            serialization="",
        )
