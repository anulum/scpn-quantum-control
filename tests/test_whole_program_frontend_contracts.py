# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program frontend contracts tests
# scpn-quantum-control -- whole-program frontend contract tests
"""Validation and public-identity tests for whole-program frontend contracts."""

from __future__ import annotations

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable import (
    compile_whole_program_frontend as facade_compile_whole_program_frontend,
)
from scpn_quantum_control.whole_program_frontend import (
    WholeProgramBytecodeBasicBlock,
    WholeProgramBytecodeInstruction,
    WholeProgramCompilerFrontendReport,
    WholeProgramSemanticsReport,
    WholeProgramSourceBytecodeLineMap,
    WholeProgramSourceIRFeature,
    WholeProgramSourceRegion,
    WholeProgramSymbolScopeEntry,
    WholeProgramUnsupportedSemanticDiagnostic,
    compile_whole_program_frontend,
)


def _instruction() -> WholeProgramBytecodeInstruction:
    """Return one valid bytecode instruction for contract tests."""
    return WholeProgramBytecodeInstruction(0, "RETURN_VALUE", "", 1)


def _block(**overrides: object) -> WholeProgramBytecodeBasicBlock:
    """Build one valid basic block with optional invalid overrides."""
    values: dict[str, object] = {
        "label": "bb0",
        "start_offset": 0,
        "end_offset": 0,
        "instruction_offsets": (0,),
        "successor_offsets": (),
        "terminating_opname": "RETURN_VALUE",
    }
    values.update(overrides)
    return WholeProgramBytecodeBasicBlock(**values)  # type: ignore[arg-type]


def _region(**overrides: object) -> WholeProgramSourceRegion:
    """Build one valid source region with optional invalid overrides."""
    values: dict[str, object] = {
        "region_id": "region:entry",
        "kind": "function_entry",
        "detail": "objective",
        "line_start": 1,
        "line_end": 1,
        "parent_region_id": None,
        "feature_kinds": (),
    }
    values.update(overrides)
    return WholeProgramSourceRegion(**values)  # type: ignore[arg-type]


def _line_map(**overrides: object) -> WholeProgramSourceBytecodeLineMap:
    """Build one valid source-bytecode crosswalk with optional overrides."""
    values: dict[str, object] = {
        "line_number": 1,
        "absolute_line_number": 1,
        "instruction_offsets": (0,),
        "region_ids": ("region:entry",),
        "feature_kinds": (),
    }
    values.update(overrides)
    return WholeProgramSourceBytecodeLineMap(**values)  # type: ignore[arg-type]


def _scope_entry(**overrides: object) -> WholeProgramSymbolScopeEntry:
    """Build one valid symbol-scope entry with optional overrides."""
    values: dict[str, object] = {
        "symbol": "theta",
        "roles": ("parameter",),
        "line_numbers": (1,),
        "bytecode_offsets": (0,),
        "region_ids": ("region:entry",),
    }
    values.update(overrides)
    return WholeProgramSymbolScopeEntry(**values)  # type: ignore[arg-type]


def _diagnostic(**overrides: object) -> WholeProgramUnsupportedSemanticDiagnostic:
    """Build one valid unsupported-semantics diagnostic with overrides."""
    values: dict[str, object] = {
        "semantic": "yield",
        "detail": "yield expression",
        "line_number": 1,
        "absolute_line_number": 1,
        "region_ids": ("region:entry",),
        "bytecode_offsets": (0,),
    }
    values.update(overrides)
    return WholeProgramUnsupportedSemanticDiagnostic(**values)  # type: ignore[arg-type]


def _semantics(**overrides: object) -> WholeProgramSemanticsReport:
    """Build one valid semantics report with optional overrides."""
    values: dict[str, object] = {
        "bytecode_frontend": True,
        "source_frontend": True,
        "graph_capture": True,
        "aliasing_observed": False,
        "mutation_observed": False,
        "loop_observed": False,
        "control_flow_observed": False,
        "numpy_observed": False,
        "differentiation_semantics": "executed_trace",
        "accepted_python_semantics": (),
        "unsupported_python_semantics": (),
    }
    values.update(overrides)
    return WholeProgramSemanticsReport(**values)  # type: ignore[arg-type]


def _frontend_report(**overrides: object) -> WholeProgramCompilerFrontendReport:
    """Build one internally consistent frontend report with overrides."""
    values: dict[str, object] = {
        "function_name": "objective",
        "bytecode_instructions": (_instruction(),),
        "bytecode_basic_blocks": (_block(),),
        "source_ir_features": (WholeProgramSourceIRFeature("call", "np.sin", 1),),
        "source_regions": (_region(),),
        "source_bytecode_line_map": (_line_map(),),
        "symbol_scope_entries": (_scope_entry(),),
        "unsupported_semantic_diagnostics": (),
        "semantics_report": _semantics(),
        "source_available": True,
        "source_sha256": "a" * 64,
        "source_start_line": 1,
        "source_end_line": 1,
        "bytecode_digest": "b" * 64,
        "frontend_digest": "c" * 64,
        "ast_node_count": 1,
        "hard_gaps": (),
        "claim_boundary": "static inspection only",
    }
    values.update(overrides)
    return WholeProgramCompilerFrontendReport(**values)  # type: ignore[arg-type]


def test_whole_program_frontend_dataclasses_fail_closed() -> None:
    """Frontend value objects should reject inconsistent static metadata."""
    with pytest.raises(ValueError, match="instruction_offsets must be sorted"):
        WholeProgramBytecodeBasicBlock(
            label="bb",
            start_offset=1,
            end_offset=2,
            instruction_offsets=(2, 1),
            successor_offsets=(),
            terminating_opname="RETURN_VALUE",
        )

    with pytest.raises(ValueError, match="feature_kinds must be sorted and unique"):
        WholeProgramSourceRegion(
            region_id="region:bad",
            kind="entry",
            detail="module",
            line_start=1,
            line_end=1,
            parent_region_id=None,
            feature_kinds=("loop", "loop"),
        )

    with pytest.raises(ValueError, match="instruction_offsets must be sorted"):
        WholeProgramSourceBytecodeLineMap(
            line_number=1,
            absolute_line_number=1,
            instruction_offsets=(4, 2),
            region_ids=("region:entry",),
            feature_kinds=(),
        )

    with pytest.raises(ValueError, match="roles must be sorted and unique"):
        WholeProgramSymbolScopeEntry(
            symbol="values",
            roles=("parameter", "parameter"),
            line_numbers=(1,),
            bytecode_offsets=(),
            region_ids=("region:entry",),
        )

    with pytest.raises(ValueError, match="bytecode_offsets must be sorted"):
        WholeProgramUnsupportedSemanticDiagnostic(
            semantic="filtered_comprehension",
            detail="filtered_comprehension",
            line_number=1,
            absolute_line_number=10,
            region_ids=("region:entry",),
            bytecode_offsets=(4, 2),
        )

    with pytest.raises(ValueError, match="accepted_python_semantics entries"):
        WholeProgramSemanticsReport(
            bytecode_frontend=True,
            source_frontend=True,
            graph_capture=True,
            aliasing_observed=False,
            mutation_observed=False,
            loop_observed=False,
            control_flow_observed=False,
            numpy_observed=False,
            differentiation_semantics="bounded",
            accepted_python_semantics=("",),
            unsupported_python_semantics=(),
        )


def test_whole_program_frontend_exports_stay_crosswired() -> None:
    """Package-root, facade, and module exports should share object identity."""
    assert scpn.compile_whole_program_frontend is compile_whole_program_frontend
    assert facade_compile_whole_program_frontend is compile_whole_program_frontend
    assert scpn.WholeProgramBytecodeBasicBlock is WholeProgramBytecodeBasicBlock
    assert scpn.WholeProgramCompilerFrontendReport is WholeProgramCompilerFrontendReport
    assert scpn.WholeProgramSourceBytecodeLineMap is WholeProgramSourceBytecodeLineMap
    assert scpn.WholeProgramSourceRegion is WholeProgramSourceRegion
    assert scpn.WholeProgramSymbolScopeEntry is WholeProgramSymbolScopeEntry
    assert (
        scpn.WholeProgramUnsupportedSemanticDiagnostic is WholeProgramUnsupportedSemanticDiagnostic
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WholeProgramBytecodeInstruction(-1, "LOAD", "", 1), "offset"),
        (lambda: WholeProgramBytecodeInstruction(0, "", "", 1), "opname"),
        (lambda: WholeProgramBytecodeInstruction(0, "LOAD", 1, 1), "argrepr"),
        (lambda: WholeProgramBytecodeInstruction(0, "LOAD", "", 0), "line_number"),
        (lambda: WholeProgramBytecodeInstruction(0, "LOAD", "", 1, -1), "jump_target"),
        (lambda: WholeProgramSourceIRFeature("", "detail", 1), "kind"),
        (lambda: WholeProgramSourceIRFeature("call", "", 1), "detail"),
        (lambda: WholeProgramSourceIRFeature("call", "detail", 0), "line_number"),
    ],
)
def test_instruction_and_feature_contracts_reject_invalid_fields(
    factory: object, message: str
) -> None:
    """Reject each invalid instruction and source-feature field."""
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"label": ""}, "label"),
        ({"start_offset": -1}, "offsets"),
        ({"end_offset": -1}, "offsets"),
        ({"start_offset": 2, "end_offset": 1}, "end_offset"),
        ({"instruction_offsets": ()}, "must be non-empty"),
        ({"instruction_offsets": (1, 0)}, "must be sorted"),
        (
            {"start_offset": 1, "end_offset": 1, "instruction_offsets": (0,)},
            "start_offset must match",
        ),
        ({"end_offset": 1}, "end_offset must match"),
        ({"successor_offsets": (-1,)}, "successor_offsets must be non-negative"),
        ({"successor_offsets": (1, 1)}, "sorted and unique"),
        ({"terminating_opname": ""}, "terminating_opname"),
    ],
)
def test_basic_block_contract_rejects_invalid_fields(
    overrides: dict[str, object], message: str
) -> None:
    """Reject every malformed basic-block invariant."""
    with pytest.raises(ValueError, match=message):
        _block(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"region_id": ""}, "region_id"),
        ({"kind": ""}, "kind"),
        ({"detail": ""}, "detail"),
        ({"line_start": 0}, "line numbers"),
        ({"line_end": 0}, "line numbers"),
        ({"line_start": 2}, "line_end"),
        ({"parent_region_id": ""}, "parent_region_id"),
        ({"feature_kinds": ("",)}, "entries"),
        ({"feature_kinds": ("loop", "loop")}, "sorted and unique"),
    ],
)
def test_source_region_contract_rejects_invalid_fields(
    overrides: dict[str, object], message: str
) -> None:
    """Reject every malformed source-region invariant."""
    with pytest.raises(ValueError, match=message):
        _region(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"line_number": 0}, "line_number"),
        ({"absolute_line_number": 0}, "absolute_line_number"),
        ({"instruction_offsets": ()}, "must be non-empty"),
        ({"instruction_offsets": (1, 0)}, "must be sorted"),
        ({"instruction_offsets": (-1,)}, "must be non-negative"),
        ({"region_ids": ("",)}, "region_ids entries"),
        ({"region_ids": ("r", "r")}, "sorted and unique"),
        ({"feature_kinds": ("",)}, "feature_kinds entries"),
        ({"feature_kinds": ("loop", "loop")}, "sorted and unique"),
    ],
)
def test_line_map_contract_rejects_invalid_fields(
    overrides: dict[str, object], message: str
) -> None:
    """Reject every malformed source-bytecode crosswalk invariant."""
    with pytest.raises(ValueError, match=message):
        _line_map(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"symbol": ""}, "symbol"),
        ({"roles": ()}, "roles must be non-empty"),
        ({"roles": ("parameter", "parameter")}, "sorted and unique"),
        ({"roles": ("",)}, "roles entries"),
        ({"line_numbers": (0,)}, "line_numbers must be positive"),
        ({"line_numbers": (1, 1)}, "line_numbers must be sorted and unique"),
        ({"bytecode_offsets": (-1,)}, "bytecode_offsets must be non-negative"),
        ({"bytecode_offsets": (0, 0)}, "bytecode_offsets must be sorted and unique"),
        ({"region_ids": ("",)}, "region_ids entries"),
        ({"region_ids": ("r", "r")}, "region_ids must be sorted and unique"),
    ],
)
def test_symbol_scope_contract_rejects_invalid_fields(
    overrides: dict[str, object], message: str
) -> None:
    """Reject every malformed symbol-scope invariant."""
    with pytest.raises(ValueError, match=message):
        _scope_entry(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"semantic": ""}, "semantic must be non-empty"),
        ({"detail": ""}, "detail"),
        ({"line_number": 0}, "line_number"),
        ({"absolute_line_number": 0}, "absolute_line_number"),
        ({"region_ids": ("",)}, "region_ids entries"),
        ({"region_ids": ("r", "r")}, "region_ids must be sorted and unique"),
        ({"bytecode_offsets": (-1,)}, "bytecode_offsets must be non-negative"),
        ({"bytecode_offsets": (0, 0)}, "bytecode_offsets must be sorted and unique"),
    ],
)
def test_unsupported_diagnostic_contract_rejects_invalid_fields(
    overrides: dict[str, object], message: str
) -> None:
    """Reject every malformed unsupported-semantics diagnostic invariant."""
    with pytest.raises(ValueError, match=message):
        _diagnostic(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bytecode_frontend": 1}, "bytecode_frontend"),
        ({"source_frontend": 1}, "source_frontend"),
        ({"graph_capture": 1}, "graph_capture"),
        ({"aliasing_observed": 1}, "aliasing_observed"),
        ({"mutation_observed": 1}, "mutation_observed"),
        ({"loop_observed": 1}, "loop_observed"),
        ({"control_flow_observed": 1}, "control_flow_observed"),
        ({"numpy_observed": 1}, "numpy_observed"),
        ({"differentiation_semantics": ""}, "differentiation_semantics"),
        ({"accepted_python_semantics": []}, "accepted_python_semantics must be a tuple"),
        ({"unsupported_python_semantics": []}, "unsupported_python_semantics must be a tuple"),
        ({"accepted_python_semantics": (1,)}, "accepted_python_semantics entries"),
        ({"unsupported_python_semantics": ("",)}, "unsupported_python_semantics entries"),
    ],
)
def test_semantics_report_rejects_invalid_fields(
    overrides: dict[str, object], message: str
) -> None:
    """Reject non-boolean flags and malformed semantic label collections."""
    with pytest.raises(ValueError, match=message):
        _semantics(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"function_name": 1}, "function_name"),
        ({"function_name": ""}, "function_name"),
        ({"bytecode_instructions": (object(),)}, "bytecode_instructions"),
        ({"bytecode_basic_blocks": (object(),)}, "bytecode_basic_blocks"),
        (
            {
                "bytecode_basic_blocks": (
                    _block(start_offset=1, end_offset=1, instruction_offsets=(1,)),
                )
            },
            "known instructions",
        ),
        ({"bytecode_basic_blocks": (_block(successor_offsets=(1,)),)}, "known successors"),
        ({"source_ir_features": (object(),)}, "source_ir_features"),
        ({"source_regions": (object(),)}, "source_regions"),
        ({"source_regions": (_region(parent_region_id="missing"),)}, "known parents"),
        ({"source_bytecode_line_map": (object(),)}, "source_bytecode_line_map"),
        ({"source_bytecode_line_map": (_line_map(region_ids=()),)}, "attach source regions"),
        (
            {"source_bytecode_line_map": (_line_map(instruction_offsets=(1,)),)},
            "known instructions",
        ),
        ({"source_bytecode_line_map": (_line_map(region_ids=("missing",)),)}, "known regions"),
        ({"symbol_scope_entries": (object(),)}, "symbol_scope_entries"),
        ({"symbol_scope_entries": (_scope_entry(region_ids=()),)}, "attach source regions"),
        ({"symbol_scope_entries": (_scope_entry(bytecode_offsets=(1,)),)}, "known instructions"),
        ({"symbol_scope_entries": (_scope_entry(region_ids=("missing",)),)}, "known regions"),
        ({"unsupported_semantic_diagnostics": (object(),)}, "unsupported_semantic_diagnostics"),
        (
            {"unsupported_semantic_diagnostics": (_diagnostic(region_ids=()),)},
            "must attach regions",
        ),
        (
            {"unsupported_semantic_diagnostics": (_diagnostic(region_ids=("missing",)),)},
            "known regions",
        ),
        (
            {"unsupported_semantic_diagnostics": (_diagnostic(bytecode_offsets=(1,)),)},
            "known instructions",
        ),
        ({"semantics_report": object()}, "semantics_report"),
        ({"source_available": 1}, "source_available"),
        ({"source_sha256": None}, "requires sha256"),
        ({"source_sha256": "short"}, "requires sha256"),
        ({"source_start_line": None}, "requires start line"),
        ({"source_start_line": 0}, "requires start line"),
        ({"source_end_line": None}, "requires valid end line"),
        ({"source_start_line": 2, "source_end_line": 1}, "requires valid end line"),
        (
            {
                "source_available": False,
                "source_sha256": "a" * 64,
                "source_start_line": None,
                "source_end_line": None,
            },
            "must not carry sha256",
        ),
        (
            {
                "source_available": False,
                "source_sha256": None,
                "source_start_line": 1,
                "source_end_line": None,
            },
            "must not carry line bounds",
        ),
        ({"bytecode_digest": "short"}, "bytecode_digest"),
        ({"frontend_digest": "short"}, "frontend_digest"),
        ({"ast_node_count": -1}, "ast_node_count"),
        ({"hard_gaps": ("",)}, "hard_gaps"),
        (
            {"semantics_report": _semantics(unsupported_python_semantics=("yield",))},
            "must be recorded as hard gaps",
        ),
        (
            {
                "unsupported_semantic_diagnostics": (_diagnostic(),),
                "semantics_report": _semantics(),
            },
            "must match diagnostics",
        ),
        ({"claim_boundary": ""}, "claim_boundary"),
    ],
)
def test_compiler_frontend_report_rejects_inconsistent_records(
    overrides: dict[str, object], message: str
) -> None:
    """Reject every inconsistent cross-record compiler frontend invariant."""
    with pytest.raises(ValueError, match=message):
        _frontend_report(**overrides)


def test_compiler_frontend_report_serializes_all_contract_rows() -> None:
    """Serialize a ready report and preserve every typed count and row."""
    report = _frontend_report()
    payload = report.to_dict()

    assert report.frontend_ready is True
    assert report.bytecode_instruction_count == 1
    assert report.source_feature_count == 1
    assert report.bytecode_basic_block_count == 1
    assert report.source_region_count == 1
    assert report.source_bytecode_line_map_count == 1
    assert report.symbol_scope_entry_count == 1
    assert report.unsupported_semantic_diagnostic_count == 0
    assert payload["frontend_ready"] is True
    assert payload["bytecode_instructions"][0]["opname"] == "RETURN_VALUE"  # type: ignore[index]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "source_available": False,
            "source_sha256": None,
            "source_start_line": None,
            "source_end_line": None,
        },
        {
            "bytecode_instructions": (),
            "bytecode_basic_blocks": (),
            "source_bytecode_line_map": (),
            "symbol_scope_entries": (),
        },
        {"bytecode_basic_blocks": ()},
        {"source_regions": (), "source_bytecode_line_map": (), "symbol_scope_entries": ()},
        {"source_bytecode_line_map": ()},
        {"symbol_scope_entries": ()},
        {"semantics_report": _semantics(bytecode_frontend=False)},
        {"semantics_report": _semantics(source_frontend=False)},
        {"hard_gaps": ("source_unavailable",)},
    ],
)
def test_frontend_ready_requires_every_static_admission_condition(
    overrides: dict[str, object],
) -> None:
    """Return false when any one frontend-readiness condition is absent."""
    assert _frontend_report(**overrides).frontend_ready is False
