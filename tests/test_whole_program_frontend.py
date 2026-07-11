# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program frontend analysis tests
"""Execution-free tests for whole-program source and bytecode frontend inspection."""

from __future__ import annotations

import dis
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    compile_whole_program_frontend as facade_compile_whole_program_frontend,
)
from scpn_quantum_control.differentiable import (
    whole_program_value_and_grad,
)
from scpn_quantum_control.whole_program_frontend import (
    WholeProgramBytecodeBasicBlock,
    WholeProgramCompilerFrontendReport,
    WholeProgramSourceBytecodeLineMap,
    WholeProgramSourceRegion,
    WholeProgramSymbolScopeEntry,
    WholeProgramUnsupportedSemanticDiagnostic,
    _instruction_line_number,
    compile_whole_program_frontend,
)


def test_whole_program_frontend_module_matches_facade_report() -> None:
    """The extracted module and compatibility facade should inspect the same objective."""

    calls = {"count": 0}

    def objective(values: NDArray[np.float64]) -> object:
        calls["count"] += 1
        total = values[0]
        for index in range(1, 3):
            total = total + np.sin(values[index])
        if total > 0.0:
            total = total * values[0]
        return total

    module_report = compile_whole_program_frontend(objective)
    facade_report = facade_compile_whole_program_frontend(objective)
    payload = module_report.to_dict()

    assert calls == {"count": 0}
    assert module_report == facade_report
    assert isinstance(module_report, WholeProgramCompilerFrontendReport)
    assert module_report.frontend_ready is True
    assert module_report.source_available is True
    assert module_report.source_sha256 is not None
    assert len(module_report.source_sha256) == 64
    assert module_report.source_start_line is not None
    source_start_line = module_report.source_start_line
    assert module_report.source_end_line is not None
    assert module_report.source_start_line < module_report.source_end_line
    assert len(module_report.bytecode_digest) == 64
    assert len(module_report.frontend_digest) == 64
    assert module_report.bytecode_instruction_count > 0
    assert module_report.bytecode_basic_block_count > 1
    assert module_report.source_feature_count > 0
    assert module_report.source_region_count > 1
    assert module_report.source_bytecode_line_map_count > 0
    assert module_report.symbol_scope_entry_count > 0
    assert module_report.ast_node_count > 0
    assert module_report.hard_gaps == ()
    assert all(
        isinstance(block, WholeProgramBytecodeBasicBlock)
        for block in module_report.bytecode_basic_blocks
    )
    assert all(
        isinstance(region, WholeProgramSourceRegion) for region in module_report.source_regions
    )
    assert all(
        isinstance(line_map, WholeProgramSourceBytecodeLineMap)
        for line_map in module_report.source_bytecode_line_map
    )
    assert all(
        isinstance(entry, WholeProgramSymbolScopeEntry)
        for entry in module_report.symbol_scope_entries
    )
    assert any(block.successor_offsets for block in module_report.bytecode_basic_blocks)
    assert any(len(block.successor_offsets) == 2 for block in module_report.bytecode_basic_blocks)
    assert {"entry", "function", "loop", "control_flow"}.issubset(
        {region.kind for region in module_report.source_regions}
    )
    source_line_count = max(region.line_end for region in module_report.source_regions)
    assert all(
        1 <= line_map.line_number <= source_line_count
        for line_map in module_report.source_bytecode_line_map
    )
    assert all(line_map.region_ids for line_map in module_report.source_bytecode_line_map)
    assert any(
        line_map.absolute_line_number is not None
        and line_map.absolute_line_number > line_map.line_number
        for line_map in module_report.source_bytecode_line_map
    )
    assert all(
        line_map.absolute_line_number is None or line_map.absolute_line_number >= source_start_line
        for line_map in module_report.source_bytecode_line_map
    )
    assert any(
        entry.symbol == "values" and entry.region_ids
        for entry in module_report.symbol_scope_entries
    )
    assert module_report.semantics_report.bytecode_frontend is True
    assert module_report.semantics_report.source_frontend is True
    assert module_report.semantics_report.loop_observed is True
    assert module_report.semantics_report.control_flow_observed is True
    assert module_report.semantics_report.numpy_observed is True
    assert {"loop", "control_flow", "numpy"}.issubset(
        {feature.kind for feature in module_report.source_ir_features}
    )
    assert payload["frontend_ready"] is True
    assert str(payload["function_name"]).endswith("objective")
    assert payload["source_start_line"] == module_report.source_start_line
    assert payload["source_end_line"] == module_report.source_end_line
    assert payload["bytecode_instruction_count"] == module_report.bytecode_instruction_count
    assert payload["bytecode_basic_block_count"] == module_report.bytecode_basic_block_count
    assert payload["source_region_count"] == module_report.source_region_count
    assert payload["source_bytecode_line_map_count"] == (
        module_report.source_bytecode_line_map_count
    )
    assert payload["symbol_scope_entry_count"] == module_report.symbol_scope_entry_count
    assert (
        payload["unsupported_semantic_diagnostic_count"]
        == module_report.unsupported_semantic_diagnostic_count
        == 0
    )
    assert payload["frontend_digest"] == module_report.frontend_digest
    bytecode_basic_blocks = payload["bytecode_basic_blocks"]
    assert isinstance(bytecode_basic_blocks, list)
    assert bytecode_basic_blocks
    assert isinstance(bytecode_basic_blocks[0], dict)
    assert bytecode_basic_blocks[0]["label"] == module_report.bytecode_basic_blocks[0].label
    source_regions = payload["source_regions"]
    assert isinstance(source_regions, list)
    assert source_regions
    assert isinstance(source_regions[0], dict)
    assert source_regions[0]["kind"] == "entry"
    source_bytecode_line_map = payload["source_bytecode_line_map"]
    assert isinstance(source_bytecode_line_map, list)
    assert source_bytecode_line_map
    assert isinstance(source_bytecode_line_map[0], dict)
    assert source_bytecode_line_map[0]["instruction_offsets"]
    assert source_bytecode_line_map[0]["region_ids"]
    symbol_scope_entries = payload["symbol_scope_entries"]
    assert isinstance(symbol_scope_entries, list)
    assert any(
        isinstance(entry, dict) and entry["symbol"] == "values" and "parameter" in entry["roles"]
        for entry in symbol_scope_entries
    )
    assert "does not execute objectives" in module_report.claim_boundary


def test_whole_program_frontend_reports_located_unsupported_semantics() -> None:
    """Unsupported source constructs should become located hard gaps."""

    def objective(values: NDArray[np.float64]) -> object:
        return sum([item for item in values if item > 0.0])

    report = compile_whole_program_frontend(objective)
    payload = report.to_dict()

    assert report.frontend_ready is False
    assert report.semantics_report.unsupported_python_semantics == ("filtered_comprehension",)
    assert report.hard_gaps == ("unsupported_python_semantics:filtered_comprehension",)
    assert report.unsupported_semantic_diagnostic_count == 1
    diagnostic = report.unsupported_semantic_diagnostics[0]
    assert isinstance(diagnostic, WholeProgramUnsupportedSemanticDiagnostic)
    assert diagnostic.semantic == "filtered_comprehension"
    assert diagnostic.detail == "filtered_comprehension"
    assert diagnostic.line_number > 0
    assert diagnostic.absolute_line_number is not None
    assert diagnostic.region_ids
    assert isinstance(diagnostic.bytecode_offsets, tuple)
    assert diagnostic.bytecode_offsets
    assert report.frontend_digest
    hard_gaps = payload["hard_gaps"]
    assert isinstance(hard_gaps, list)
    assert "unsupported_python_semantics:filtered_comprehension" in hard_gaps
    assert payload["unsupported_semantic_diagnostic_count"] == 1
    diagnostics = payload["unsupported_semantic_diagnostics"]
    assert isinstance(diagnostics, list)
    assert diagnostics
    first_diagnostic = diagnostics[0]
    assert isinstance(first_diagnostic, dict)
    assert first_diagnostic["semantic"] == "filtered_comprehension"
    assert first_diagnostic["line_number"] == diagnostic.line_number
    assert any(
        feature.kind == "unsupported_python_semantics"
        and feature.detail == "filtered_comprehension"
        and feature.line_number == diagnostic.line_number
        for feature in report.source_ir_features
    )


def test_whole_program_frontend_rejects_async_objective_before_execution() -> None:
    """Async whole-program objectives should fail the frontend gate."""

    async def helper(value: object) -> object:
        return value

    async def objective(values: NDArray[np.float64]) -> object:
        return await helper(values[0])

    objective_callable = cast(Callable[..., object], objective)
    report = compile_whole_program_frontend(objective_callable)

    assert report.frontend_ready is False
    assert report.semantics_report.unsupported_python_semantics == (
        "async_function",
        "await_expression",
    )
    assert report.hard_gaps == (
        "unsupported_python_semantics:async_function",
        "unsupported_python_semantics:await_expression",
    )
    diagnostics = {
        diagnostic.semantic: diagnostic for diagnostic in report.unsupported_semantic_diagnostics
    }
    assert set(diagnostics) == {"async_function", "await_expression"}
    for diagnostic in diagnostics.values():
        assert diagnostic.line_number > 0
        assert diagnostic.absolute_line_number is not None
        assert diagnostic.region_ids
    assert any(diagnostic.bytecode_offsets for diagnostic in diagnostics.values())
    assert any(
        feature.kind == "unsupported_python_semantics" and feature.detail == "async_function"
        for feature in report.source_ir_features
    )
    assert any(
        feature.kind == "unsupported_python_semantics" and feature.detail == "await_expression"
        for feature in report.source_ir_features
    )

    with pytest.raises(ValueError) as exc_info:
        whole_program_value_and_grad(objective_callable, np.array([1.0], dtype=np.float64))

    message = str(exc_info.value)
    assert "whole-program AD frontend execution gate rejected objective" in message
    assert "unsupported_python_semantics:async_function" in message
    assert "unsupported_python_semantics:await_expression" in message
    assert "semantic=async_function" in message
    assert "semantic=await_expression" in message


def test_whole_program_frontend_reports_async_iteration_as_unsupported() -> None:
    """Async iteration should be located as an unsupported frontend construct."""

    class AsyncItems:
        def __aiter__(self) -> AsyncIterator[object]:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    async def objective(values: AsyncItems) -> object:
        total: object = None
        async for item in values:
            total = item
        return total

    report = compile_whole_program_frontend(cast(Callable[..., object], objective))

    assert report.frontend_ready is False
    assert report.semantics_report.unsupported_python_semantics == (
        "async_for",
        "async_function",
    )
    diagnostics = {
        diagnostic.semantic: diagnostic for diagnostic in report.unsupported_semantic_diagnostics
    }
    assert set(diagnostics) == {"async_for", "async_function"}
    async_for_diagnostic = diagnostics["async_for"]
    assert async_for_diagnostic.line_number > 0
    assert async_for_diagnostic.absolute_line_number is not None
    assert async_for_diagnostic.region_ids
    assert any(
        feature.kind == "loop" and feature.detail == "async_for"
        for feature in report.source_ir_features
    )


def _line_marker_instruction(starts_line: bool | int, positions: dis.Positions) -> dis.Instruction:
    """Return a ``dis.Instruction`` stand-in carrying only the line-marker fields.

    ``dis.Instruction``'s concrete field set changed across CPython releases —
    ``is_jump_target`` was dropped in 3.13 — so constructing it with fixed keyword
    arguments is not portable. ``_instruction_line_number`` reads only
    ``starts_line`` and ``positions``, so a stand-in carrying those two attributes
    exercises the same code on every supported interpreter.
    """

    return cast(dis.Instruction, SimpleNamespace(starts_line=starts_line, positions=positions))


def test_whole_program_frontend_normalises_python313_boolean_line_markers() -> None:
    """Bytecode line capture should survive CPython 3.13 boolean line markers."""

    python313_instruction = _line_marker_instruction(
        starts_line=True,
        positions=dis.Positions(lineno=123, end_lineno=123, col_offset=4, end_col_offset=10),
    )
    legacy_instruction = _line_marker_instruction(
        starts_line=77,
        positions=dis.Positions(lineno=123, end_lineno=123, col_offset=4, end_col_offset=10),
    )
    missing_instruction = _line_marker_instruction(
        starts_line=False,
        positions=dis.Positions(
            lineno=None, end_lineno=None, col_offset=None, end_col_offset=None
        ),
    )

    assert _instruction_line_number(python313_instruction) == 123
    assert _instruction_line_number(legacy_instruction) == 77
    assert _instruction_line_number(missing_instruction) is None
