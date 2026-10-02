# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program frontend tests
# scpn-quantum-control -- whole-program frontend analysis tests
"""Execution-free tests for whole-program source and bytecode frontend inspection."""

from __future__ import annotations

import ast
import dis
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from functools import wraps
from pathlib import Path
from threading import Event
from types import CodeType, FrameType, FunctionType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.whole_program_frontend as frontend_module
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import (
    compile_whole_program_frontend as facade_compile_whole_program_frontend,
)
from scpn_quantum_control.differentiable import (
    program_adjoint_replay_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
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
    assert module_report.frontend_ready is False
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
    assert module_report.hard_gaps == ("unsupported_python_semantics:captured_mutation",)
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
    assert payload["frontend_ready"] is False
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
        == 1
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


def test_frontend_metadata_and_introspection_fail_closed(
    tmp_path: Path,
) -> None:
    """Validate source metadata and unavailable source/bytecode fallbacks."""
    metadata_type = frontend_module._ObjectiveSourceMetadata
    invalid_metadata: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: metadata_type("", 1, 1), "source"),
        (lambda: metadata_type("x", 0, 1), "start_line"),
        (lambda: metadata_type("x", 2, 1), "end_line"),
    )
    for factory, message in invalid_metadata:
        with pytest.raises(ValueError, match=message):
            factory()

    path = tmp_path / "unavailable_source.py"
    path.write_text("def objective(value):\n    return value\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("unavailable_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path.write_text("  \n", encoding="utf-8")
    assert facade_compile_whole_program_frontend(module.objective).source_available is False
    path.unlink()
    assert facade_compile_whole_program_frontend(module.objective).source_available is False
    unavailable = facade_compile_whole_program_frontend(len)
    assert unavailable.source_available is False
    assert unavailable.bytecode_instructions == ()
    assert frontend_module._normalise_positive_line_number(None) is None
    assert frontend_module._normalise_positive_line_number(0) is None


def test_frontend_source_helpers_cover_alias_effect_and_region_variants() -> None:
    """Parse representative alias, mutation, async, loop, and region syntax."""
    source = """
async def objective(values):
    items = []
    alias = items
    obj = Box()
    obj.value = values[0]
    copied = obj.value
    external.value = copied
    external_copy = external.value
    other[0] = copied
    items.append(copied)
    if values[0]:
        obj.value = copied
        (left_value, right_value) = values
        alias[0] = copied
    else:
        copied += 1
    for left, right in []:
        copied = copied + left
        if right:
            continue
        break
    while copied:
        del items[0]
        copied = await obj.step()
    return np.sin(values[0]) if copied else numpy.cos(values[0])
"""
    tree = ast.parse(source)
    features = frontend_module._source_ir_features(source)
    details = {(feature.kind, feature.detail) for feature in features}
    assert any(kind == "list_alias" for kind, _detail in details)
    assert any(kind == "object_attribute_alias" for kind, _detail in details)
    assert any(kind == "control_path_alias" for kind, _detail in details)
    assert any(kind == "loop_carried_state" for kind, _detail in details)
    assert {"break", "continue"}.issubset({detail for kind, detail in details if kind == "loop"})
    assert frontend_module._source_regions(source, features)
    unsupported = frontend_module._source_ir_features(
        None, unsupported_python_semantics=("synthetic",)
    )
    assert unsupported[0].detail == "synthetic"
    assert frontend_module._source_ir_features("def broken(") == ()
    assert frontend_module._source_regions("def broken(", ()) == ()

    assignment = ast.parse("target = " + "+".join(["value"] * 50)).body[0]
    assert isinstance(assignment, ast.Assign)
    assert frontend_module._stable_ast_expression_label(assignment.value, 1).endswith("...")
    call = ast.parse("Box()", mode="eval").body
    assert frontend_module._is_local_object_constructor_call(call)
    assert not frontend_module._is_local_object_constructor_call(ast.Constant(value=1))
    assert not frontend_module._is_local_object_constructor_call(
        ast.parse("np.sin(1)", mode="eval").body
    )
    assert not frontend_module._is_local_object_constructor_call(
        ast.parse("(lambda: value)()", mode="eval").body
    )

    attribute = ast.parse("root.child.value", mode="eval").body
    subscript = ast.parse("root.child[0][1]", mode="eval").body
    assert isinstance(attribute, ast.Attribute)
    assert isinstance(subscript, ast.Subscript)
    assert frontend_module._ast_attribute_root(attribute) == "root"
    factory_attribute = ast.parse("factory().value", mode="eval").body
    assert isinstance(factory_attribute, ast.Attribute)
    assert frontend_module._ast_attribute_root(factory_attribute) == ""
    assert frontend_module._ast_subscript_root(subscript) == "root"
    factory_subscript = ast.parse("factory()[0]", mode="eval").body
    assert isinstance(factory_subscript, ast.Subscript)
    assert frontend_module._ast_subscript_root(factory_subscript) == ""
    child_call = ast.parse("root.child()", mode="eval").body
    assert isinstance(child_call, ast.Call)
    assert frontend_module._ast_call_name(child_call.func) == "root.child"
    lambda_call = ast.parse("(lambda: value)()", mode="eval").body
    assert isinstance(lambda_call, ast.Call)
    assert frontend_module._ast_call_name(lambda_call.func) == ""
    assert tree


def test_frontend_semantics_cover_signatures_and_unsupported_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify accepted signatures and every located unsupported construct."""
    captured = SimpleNamespace(value=1.0)

    def objective(
        value: object = None, *args: object, flag: bool = True, **kwargs: object
    ) -> object:
        del args, flag, kwargs
        return captured.value if value is None else value

    accepted = frontend_module._accepted_python_semantics(
        objective,
        "def f():\n return [item for item in (x for x in values)]",
    )
    assert {
        "closure",
        "default_argument",
        "keyword_only_parameter",
        "var_keyword_parameter",
        "var_positional_parameter",
        "list_comprehension",
        "generator_expression",
    }.issubset(set(accepted))

    monkeypatch.setattr(
        inspect,
        "signature",
        lambda _value: (_ for _ in ()).throw(ValueError("no signature")),
    )
    assert frontend_module._accepted_python_semantics(objective, None) == ("closure",)

    source = """
@decorator
async def objective(value):
    assert value
    with value:
        try:
            await value.step()
            async for item in value:
                yield {entry for entry in item}
        except Exception:
            raise RuntimeError
    return objective(value)

@decorator
def synchronous_objective():
    return captured.value
    """
    diagnostics = frontend_module._unsupported_python_semantic_diagnostics(
        objective=objective,
        source=source,
        source_start_line=10,
        bytecode_instructions=(),
        source_regions=(),
    )
    assert {
        "async_function",
        "decorator",
        "await_expression",
        "async_for",
        "set_or_dict_comprehension",
        "generator",
        "context_manager",
        "exception_control_flow",
        "recursion",
        "object_attribute",
    }.issubset({diagnostic.semantic for diagnostic in diagnostics})
    assert "filtered_comprehension" not in frontend_module._unsupported_python_semantics(
        objective, "def objective(values):\n return [value for value in values]"
    )
    assert (
        frontend_module._unsupported_python_semantic_diagnostics(
            objective=objective,
            source="def broken(",
            source_start_line=None,
            bytecode_instructions=(),
            source_regions=(),
        )
        == ()
    )


def test_frontend_compile_reports_each_missing_static_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public compiler records absent static surfaces without execution."""
    with pytest.raises(ValueError, match="callable"):
        compile_whole_program_frontend(cast(Callable[..., object], 1))

    def objective(value: object) -> object:
        return value

    monkeypatch.setattr(frontend_module, "_objective_source_metadata", lambda _value: None)
    monkeypatch.setattr(frontend_module, "_objective_bytecode", lambda _value: ())
    monkeypatch.setattr(frontend_module, "_symbol_scope_entries", lambda **_kwargs: ())
    report = compile_whole_program_frontend(objective)
    assert {
        "bytecode_frontend_missing",
        "bytecode_basic_blocks_missing",
        "source_frontend_missing",
        "symbol_scope_entries_missing",
    }.issubset(set(report.hard_gaps))

    metadata = frontend_module._ObjectiveSourceMetadata("def broken(", 1, 1)
    monkeypatch.setattr(frontend_module, "_objective_source_metadata", lambda _value: metadata)
    report = compile_whole_program_frontend(objective)
    assert "source_regions_missing" in report.hard_gaps
    assert "source_ast_parse_failed" in report.hard_gaps

    metadata = frontend_module._ObjectiveSourceMetadata(
        "def objective(value):\n return value", 1, 2
    )
    monkeypatch.setattr(frontend_module, "_objective_source_metadata", lambda _value: metadata)
    monkeypatch.setattr(
        frontend_module,
        "_source_regions",
        lambda _source, _features: (
            frontend_module.WholeProgramSourceRegion(
                region_id="entry",
                kind="entry",
                detail="module",
                line_start=1,
                line_end=2,
                parent_region_id=None,
                feature_kinds=(),
            ),
        ),
    )
    monkeypatch.setattr(frontend_module, "_source_bytecode_line_map", lambda **_kwargs: ())
    report = compile_whole_program_frontend(objective)
    assert "source_bytecode_line_map_missing" in report.hard_gaps


def test_frontend_bytecode_and_scalar_helpers_cover_defensive_edges() -> None:
    """Decode synthetic jumps, symbols, roles, lines, and invalid source text."""
    instruction_type = frontend_module.WholeProgramBytecodeInstruction
    jump = instruction_type(0, "JUMP_FORWARD", "to 8", 1, None)
    bad_jump = instruction_type(2, "JUMP_FORWARD", "to target", 1, None)
    negative_jump = instruction_type(4, "JUMP_FORWARD", "to -1", 1, None)
    plain = instruction_type(6, "LOAD_CONST", "1", 1, None)
    assert frontend_module._bytecode_jump_target(jump) == 8
    assert frontend_module._bytecode_jump_target(bad_jump) is None
    assert frontend_module._bytecode_jump_target(negative_jump) is None
    assert frontend_module._bytecode_jump_target(plain) is None
    assert (
        frontend_module._bytecode_jump_target(
            instruction_type(8, "JUMP_FORWARD", "forward 8", 1, None)
        )
        is None
    )
    assert frontend_module._bytecode_basic_blocks(()) == ()
    assert frontend_module._bytecode_is_unconditional_jump("JUMP_FORWARD")
    assert not frontend_module._bytecode_is_unconditional_jump("JUMP_IF_FALSE")

    assert (
        frontend_module._bytecode_symbol_name(
            instruction_type(0, "LOAD_GLOBAL", "NULL + value", 1, None)
        )
        == "value"
    )
    assert frontend_module._bytecode_symbol_name(plain) is None
    assert (
        frontend_module._bytecode_symbol_name(instruction_type(0, "LOAD_GLOBAL", "()", 1, None))
        is None
    )
    assert (
        frontend_module._bytecode_symbol_name(
            instruction_type(0, "LOAD_GLOBAL", "NULL + 1", 1, None)
        )
        is None
    )
    assert frontend_module._bytecode_symbol_role("LOAD_FAST") == "bytecode_load"
    assert frontend_module._bytecode_symbol_role("STORE_FAST") == "bytecode_store"
    assert frontend_module._bytecode_symbol_role("DELETE_FAST") == "bytecode_delete"
    assert frontend_module._bytecode_symbol_role("OTHER") == "bytecode_reference"
    assert frontend_module._ast_name_role(ast.Load()) == "source_load"
    assert frontend_module._ast_name_role(ast.Store()) == "source_store"
    assert frontend_module._ast_name_role(ast.Del()) == "source_delete"
    assert frontend_module._single_absolute_line({1, 2}) is None
    assert frontend_module._source_relative_line(2, 5) == 2
    assert frontend_module._source_relative_line(5, None) == 5
    assert frontend_module._source_ast_node_count(None) == 0
    assert frontend_module._source_ast_node_count("def broken(") == 0
    assert frontend_module._source_parse_failed(None) is False
    assert frontend_module._source_parse_failed("def broken(") is True
    assert frontend_module._source_has_node(None, ast.If) is False
    assert frontend_module._source_has_node("if value", ast.If) is True
    assert frontend_module._source_mentions_numpy(None) is False
    assert frontend_module._ast_name_role(ast.expr_context()) == "source_reference"


def test_frontend_line_map_scope_and_capture_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover line remapping, uninspectable callables, and source parse fallbacks."""
    instruction_type = frontend_module.WholeProgramBytecodeInstruction
    line_map = frontend_module._source_bytecode_line_map(
        bytecode_instructions=(instruction_type(0, "LOAD_FAST", "value", 2, None),),
        source_ir_features=(),
        source_regions=(),
        source_start_line=5,
    )
    assert line_map[0].absolute_line_number == 6

    class CallableWithoutCode:
        def __call__(self, value: object) -> object:
            return value

    opaque = CallableWithoutCode()
    assert frontend_module._captured_or_global_names(opaque) == set()
    assert frontend_module._symbol_scope_entries(
        objective=opaque,
        source=None,
        bytecode_instructions=(),
        source_regions=(),
        source_start_line=None,
    )

    def objective(value: object) -> object:
        def nested() -> object:
            return value

        return nested()

    monkeypatch.setattr(
        inspect,
        "signature",
        lambda _value: (_ for _ in ()).throw(TypeError("no signature")),
    )
    entries = frontend_module._symbol_scope_entries(
        objective=objective,
        source="def broken(",
        bytecode_instructions=frontend_module._objective_bytecode(objective),
        source_regions=(),
        source_start_line=None,
    )
    assert any("cell" in entry.roles for entry in entries if entry.symbol == "value")

    def closure_factory() -> Callable[[], object]:
        token = object()

        def closure() -> object:
            return token

        return closure

    closure = closure_factory()
    # A nested `def` is a function object, and this stand-in copies its code
    # object; `Callable` alone does not carry `__code__`.
    assert isinstance(closure, FunctionType)

    class NonMappingGlobals:
        __code__ = closure.__code__
        __globals__: list[object] = []

    # The stand-in is deliberately not callable — a list for `__globals__` is
    # the shape under test — and mypy cannot express that negative case.
    assert "token" in frontend_module._captured_or_global_names(
        NonMappingGlobals()  # type: ignore[arg-type]
    )


def test_public_frontend_digest_matches_canonical_json_oracle() -> None:
    """Actual public compiler metadata hashes retain their existing canonical wire."""

    def objective(values: NDArray[np.float64]) -> object:
        label = 'Δ\\"𐀀'
        if len(label) > 0:
            return values[0] * values[0]
        return values[0]

    baseline = active_reserved_bytes()
    report = facade_compile_whole_program_frontend(objective)
    instruction_payload = [
        {
            "offset": item.offset,
            "opname": item.opname,
            "argrepr": item.argrepr,
            "line_number": item.line_number,
            "jump_target_offset": item.jump_target_offset,
        }
        for item in report.bytecode_instructions
    ]
    encoded = json.dumps(instruction_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert report.bytecode_digest == hashlib.sha256(encoded).hexdigest()
    payload = {
        "source_sha256": report.source_sha256,
        "source_start_line": report.source_start_line,
        "source_end_line": report.source_end_line,
        "bytecode_instructions": [
            {
                "offset": item.offset,
                "opname": item.opname,
                "argrepr": item.argrepr,
                "line_number": item.line_number,
            }
            for item in report.bytecode_instructions
        ],
        "bytecode_basic_blocks": [item.to_dict() for item in report.bytecode_basic_blocks],
        "source_ir_features": [
            {"kind": item.kind, "detail": item.detail, "line_number": item.line_number}
            for item in report.source_ir_features
        ],
        "source_regions": [item.to_dict() for item in report.source_regions],
        "source_bytecode_line_map": [item.to_dict() for item in report.source_bytecode_line_map],
        "symbol_scope_entries": [item.to_dict() for item in report.symbol_scope_entries],
        "unsupported_semantic_diagnostics": [
            item.to_dict() for item in report.unsupported_semantic_diagnostics
        ],
        "semantics": {
            "accepted": list(report.semantics_report.accepted_python_semantics),
            "unsupported": list(report.semantics_report.unsupported_python_semantics),
            "bytecode_frontend": report.semantics_report.bytecode_frontend,
            "source_frontend": report.semantics_report.source_frontend,
        },
        "hard_gaps": list(report.hard_gaps),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert report.frontend_digest == hashlib.sha256(canonical).hexdigest()
    assert active_reserved_bytes() == baseline


def test_public_frontend_digest_inherits_owner_cancellation_and_recovers() -> None:
    """Compiler invoked within a real owner refuses cancellation and disposes child charge."""

    def objective(values: NDArray[np.float64]) -> object:
        return values[0] * values[0]

    signal = Event()
    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("frontend_owner", "forward", (1,), "uint8"),))
    with reserve_execution_memory(plan, cancelled=signal):
        signal.set()
        with pytest.raises(ExecutionCancelledError):
            facade_compile_whole_program_frontend(objective)
    assert active_reserved_bytes() == baseline
    report = facade_compile_whole_program_frontend(objective)
    assert report.frontend_ready
    assert active_reserved_bytes() == baseline


def test_public_frontend_nested_code_reports_match_across_native_processes() -> None:
    """The actual QEC method retains code provenance with stable complete reports."""
    from scpn_quantum_control.qec.fault_tolerant import RepetitionCodeUPDE

    objective = RepetitionCodeUPDE.step_with_qec
    direct = compile_whole_program_frontend(objective)
    facade = facade_compile_whole_program_frontend(objective)
    assert direct == facade
    nested_code = [value for value in objective.__code__.co_consts if isinstance(value, CodeType)]
    assert nested_code
    for code in nested_code:
        assert any(
            instruction.opname == "LOAD_CONST"
            and code.co_name in instruction.argrepr
            and code.co_filename in instruction.argrepr
            and str(code.co_firstlineno) in instruction.argrepr
            for instruction in direct.bytecode_instructions
        )

    program = """
import json
from scpn_quantum_control.differentiable import compile_whole_program_frontend
from scpn_quantum_control.qec.fault_tolerant import RepetitionCodeUPDE

report = compile_whole_program_frontend(RepetitionCodeUPDE.step_with_qec)
print(json.dumps(report.to_dict(), sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    expected = direct.to_dict()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", program],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("change", ["append", "overwrite", "replace", "remove"])
def test_public_frontend_rejects_source_change_during_read_and_recovers(
    tmp_path: Path, change: str
) -> None:
    """Real source-open changes reject stale admission without executing the objective."""
    source_path = tmp_path / "changing_objective.py"
    source_path.write_text(
        "calls = 0\n"
        "def objective(values):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    return values[0] * values[0]\n",
        encoding="utf-8",
    )
    program = r"""
import importlib.util
import json
import linecache
import sys
from pathlib import Path
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import compile_whole_program_frontend
from scpn_quantum_control.execution_reservations import active_reserved_bytes

path = Path(sys.argv[1])
change = sys.argv[2]
original = path.read_text(encoding="utf-8")
replacement = path.with_suffix(".replacement")
replacement.write_text(original, encoding="utf-8")
spec = importlib.util.spec_from_file_location("changing_objective", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
linecache.clearcache()
baseline = active_reserved_bytes()
armed = True
with path.open("r+", encoding="utf-8") as writer:
    def observe_open(event, args):
        global armed
        if armed and event == "open" and args[0] == str(path):
            armed = False
            if change == "append":
                writer.seek(0, 2)
                writer.write("# file grew at source open\n")
                writer.flush()
            elif change == "overwrite":
                writer.seek(0)
                writer.write("calls = 1\n")
                writer.flush()
            elif change == "replace":
                replacement.replace(path)
            elif change == "remove":
                path.unlink()
            else:
                raise AssertionError("unknown filesystem change")
    sys.addaudithook(observe_open)
    try:
        compile_whole_program_frontend(module.objective)
    except DenseAllocationError as exc:
        refusal = str(exc)
    else:
        raise AssertionError("changed source was admitted")
assert not armed
assert active_reserved_bytes() == baseline
if not path.exists():
    path.write_text(original, encoding="utf-8")
report = compile_whole_program_frontend(module.objective)
assert report.source_available
assert module.calls == 0
assert active_reserved_bytes() == baseline
print(json.dumps({"refusal": refusal, "recovered": report.source_available}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", program, str(source_path), change],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    if change == "remove":
        assert "source became unavailable during read" in payload["refusal"]
    else:
        assert "source changed before bounded read" in payload["refusal"]
    assert payload["recovered"] is True


@pytest.mark.parametrize("source_kind", ["unavailable", "directory"])
def test_public_frontend_preserves_unavailable_source_observation(
    tmp_path: Path,
    source_kind: str,
) -> None:
    """Real dynamic code and directory filenames never manufacture a source snapshot."""
    filename = "<scpn-dynamic-objective>" if source_kind == "unavailable" else str(tmp_path)
    namespace: dict[str, object] = {}
    exec(compile("def objective(values):\n    return values[0]\n", filename, "exec"), namespace)
    objective = cast(Callable[..., object], namespace["objective"])
    baseline = active_reserved_bytes()
    report = facade_compile_whole_program_frontend(objective)
    assert not report.source_available
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("change", ["remove", "append"])
def test_public_frontend_refuses_source_change_after_actual_block_extraction(
    tmp_path: Path,
    change: str,
) -> None:
    """A captured block cannot outlive the identity check on its actual source file."""
    path = tmp_path / "post_block_source.py"
    original = "def objective(values):\n    return values[0] * values[0]\n"
    path.write_text(original)
    spec = importlib.util.spec_from_file_location("post_block_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed: list[str] = []
    baseline = active_reserved_bytes()

    def profile(frame: FrameType, event: str, argument: object) -> None:
        del argument
        if event == "return" and frame.f_code.co_name == "objective_source_block" and not observed:
            observed.append(change)
            if change == "remove":
                path.unlink()
            else:
                with path.open("a") as source:
                    source.write("# modified after extraction\n")

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(DenseAllocationError, match="source disappeared|source changed"):
            facade_compile_whole_program_frontend(module.objective)
    finally:
        sys.setprofile(previous)
    assert observed == [change]
    assert active_reserved_bytes() == baseline
    path.write_text(original)
    assert facade_compile_whole_program_frontend(module.objective).source_available
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("fault", ["grow", "truncate"])
def test_public_frontend_digest_refuses_actual_payload_change_and_recovers(fault: str) -> None:
    """Canonical encoding cannot publish a digest inconsistent with admitted payload bytes."""

    def objective(values: NDArray[np.float64]) -> object:
        return values[0] * values[0]

    observed: list[int] = []
    baseline = active_reserved_bytes()

    def profile(frame: FrameType, event: str, argument: object) -> None:
        del argument
        caller = frame.f_back
        if (
            event == "call"
            and frame.f_code.co_name == "iterencode"
            and caller is not None
            and caller.f_code.co_name == "_frontend_json_digest"
            and not observed
        ):
            payload = frame.f_locals["o"]
            assert isinstance(payload, (dict, list))
            observed.append(caller.f_locals["encoded_size"])
            if fault == "grow":
                extra = "x" * (observed[0] + 1)
                if isinstance(payload, dict):
                    payload["transport_fault"] = extra
                else:
                    payload.append(extra)
            else:
                payload.clear()

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="digest encoding exceeded|digest encoding differs"):
            facade_compile_whole_program_frontend(objective)
    finally:
        sys.setprofile(previous)
    assert len(observed) == 1
    assert active_reserved_bytes() == baseline
    report = facade_compile_whole_program_frontend(objective)
    assert report.frontend_ready
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("entry_point", ["compiler", "runtime"])
def test_public_frontend_refuses_foreign_filename_before_protocols(entry_point: str) -> None:
    """Refuse filename subclasses before introspection and recover the public gradient.

    Parameters
    ----------
    entry_point
        Public static compiler or numerical differentiation entry point.

    """
    calls: list[str] = []

    class Filename(str):
        """Observe protocols that source admission must never invoke."""

        def __hash__(self) -> int:
            """Record an unwanted source-cache hash."""
            calls.append("hashing")
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            """Record an unwanted source-path comparison."""
            calls.append("equality")
            return super().__eq__(other)

    def objective(values: NDArray[np.float64]) -> object:
        """Provide a source-visible linear objective for refusal and recovery."""
        return values[0] * 3.0

    function = cast(FunctionType, objective)
    original_code = function.__code__
    function.__code__ = original_code.replace(co_filename=Filename(original_code.co_filename))
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="objective source filename must be a plain string"):
        if entry_point == "compiler":
            facade_compile_whole_program_frontend(function)
        else:
            whole_program_value_and_grad(function, np.array([2.0]), trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline

    function.__code__ = original_code
    assert facade_compile_whole_program_frontend(function).frontend_ready
    result = whole_program_value_and_grad(function, np.array([2.0]), trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("form", ["method", "wrapped", "class"])
def test_public_frontend_filename_admission_preserves_callable_forms(form: str) -> None:
    """Retain source lookup for bound methods, classes and unwrapped functions.

    Parameters
    ----------
    form
        Existing callable form accepted by the source introspection path.

    """

    class Objective:
        """Expose an ordinary class and bound-method source owner."""

        def __call__(self, values: NDArray[np.float64]) -> object:
            """Return the supported scalar objective."""
            return values[0] * 3.0

    def objective(values: NDArray[np.float64]) -> object:
        """Provide the original source owner retained by the decorated form."""
        return values[0] * 3.0

    @wraps(objective)
    def wrapped(values: NDArray[np.float64]) -> object:
        """Forward to the function whose source metadata unwrap inspects."""
        return objective(values)

    selected: Callable[..., object]
    if form == "method":
        selected = Objective().__call__
        function = cast(FunctionType, Objective.__call__)
    elif form == "wrapped":
        selected = wrapped
        function = cast(FunctionType, objective)
    else:
        selected = Objective
        function = cast(FunctionType, objective)

    baseline = active_reserved_bytes()
    assert facade_compile_whole_program_frontend(selected).source_available
    assert active_reserved_bytes() == baseline
    if form != "class":
        original_code = function.__code__

        class Filename(str):
            """Reject accidental lookup of the foreign source-cache key."""

            def __hash__(self) -> int:
                """Fail if introspection reaches an unadmitted source cache key."""
                raise AssertionError("filename subclass reached source cache")

        function.__code__ = original_code.replace(co_filename=Filename(original_code.co_filename))
        with pytest.raises(ValueError, match="objective source filename must be a plain string"):
            facade_compile_whole_program_frontend(selected)
        assert active_reserved_bytes() == baseline
        function.__code__ = original_code
        assert facade_compile_whole_program_frontend(selected).source_available
        assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("constant_kind", ["object", "tuple", "frozenset", "nested_code"])
def test_public_frontend_refuses_foreign_constants_without_protocols(constant_kind: str) -> None:
    """Keep a located source refusal before disassembly can represent a foreign value.

    Parameters
    ----------
    constant_kind
        Direct, container-nested or nested-code foreign constant.

    """
    calls: list[str] = []

    class ForeignConstant:
        """Observe protocols that refused source metadata must never execute."""

        def __repr__(self) -> str:
            """Record an unwanted disassembler representation."""
            calls.append("representation")
            return "foreign"

        def __eq__(self, other: object) -> bool:
            """Record an unwanted loaded/source constant comparison."""
            calls.append("equality")
            return False

        def __hash__(self) -> int:
            """Record hashing separately from immutable fixture construction."""
            calls.append("hashing")
            return 1

    def objective(values: NDArray[np.float64]) -> object:
        """Provide the real source whose changed constant must be refused."""
        return values[0] * 3.0

    function = cast(FunctionType, objective)
    original_code = function.__code__
    constant: object = ForeignConstant()
    if constant_kind == "tuple":
        constant = (constant,)
    elif constant_kind == "frozenset":
        constant = frozenset([constant])
    elif constant_kind == "nested_code":
        constant = original_code.replace(co_consts=(None, constant))
    function.__code__ = original_code.replace(co_consts=(None, 0, constant))
    calls.clear()
    baseline = active_reserved_bytes()
    report = facade_compile_whole_program_frontend(function)
    assert not report.frontend_ready
    diagnostic = next(
        row
        for row in report.unsupported_semantic_diagnostics
        if row.semantic == "external_callback"
    )
    assert diagnostic.detail == "external callback source does not match captured function"
    assert diagnostic.absolute_line_number is not None
    assert diagnostic.region_ids
    with pytest.raises(ValueError, match="external_callback"):
        whole_program_value_and_grad(function, np.array([2.0]), trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline

    function.__code__ = original_code
    assert facade_compile_whole_program_frontend(function).frontend_ready
    result = whole_program_value_and_grad(function, np.array([2.0]), trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    assert calls == []
    assert active_reserved_bytes() == baseline


def test_public_frontend_pure_captured_helper_preserves_real_gradient() -> None:
    """Admit a source-visible numeric helper and exercise the public AD runtime."""
    coefficient = [2.0]

    def helper(value: object) -> object:
        return coefficient[0] * cast(float, value) ** 2

    def objective(values: NDArray[np.float64]) -> object:
        return helper(values[0]) + np.sin(values[1])

    report = compile_whole_program_frontend(objective)
    assert report.frontend_ready
    assert report.unsupported_semantic_diagnostics == ()
    result = whole_program_value_and_grad(objective, np.array([0.3, -0.2]))
    assert result.value == pytest.approx(2.0 * 0.3**2 + np.sin(-0.2))
    np.testing.assert_allclose(result.gradient, [1.2, np.cos(-0.2)], atol=1e-14)
    assert coefficient == [2.0]


def test_public_frontend_refuses_callback_before_runtime_execution() -> None:
    """Bind caller locations and refuse a real external write before evaluation."""
    ledger: list[str] = []

    def callback(value: object) -> object:
        ledger.append("executed")
        return value

    def objective(values: NDArray[np.float64]) -> object:
        return callback(values[0])

    report = compile_whole_program_frontend(objective)
    assert not report.frontend_ready
    assert compile_whole_program_frontend(objective) == report
    diagnostic = next(
        item
        for item in report.unsupported_semantic_diagnostics
        if item.semantic == "external_callback"
    )
    assert diagnostic.absolute_line_number is not None
    assert diagnostic.region_ids and diagnostic.bytecode_offsets
    with pytest.raises(ValueError, match="external_callback"):
        whole_program_value_and_grad(objective, np.array([0.3]))
    assert ledger == []


def test_public_frontend_refuses_captured_alias_write_before_runtime() -> None:
    """Preserve captured storage across static inspection and numerical refusal."""
    state = [2.0]

    def objective(values: NDArray[np.float64]) -> object:
        alias = state
        alias[0] = 5.0
        return values[0] * state[0]

    report = compile_whole_program_frontend(objective)
    assert not report.frontend_ready
    assert "captured_mutation" in report.semantics_report.unsupported_python_semantics
    with pytest.raises(ValueError, match="captured_mutation"):
        whole_program_value_and_grad(objective, np.array([0.3]))
    assert state == [2.0]


def test_public_frontend_refuses_ambient_rng_without_state_advance() -> None:
    """The public frontend and runtime refuse ambient draws without performing one."""

    def objective(values: NDArray[np.float64]) -> object:
        return values[0] + np.random.random()

    before = np.random.get_state(legacy=True)
    report = compile_whole_program_frontend(objective)
    assert not report.frontend_ready
    assert "ambient_rng" in report.semantics_report.unsupported_python_semantics
    with pytest.raises(ValueError, match="ambient_rng"):
        whole_program_value_and_grad(objective, np.array([0.3]))
    after = np.random.get_state(legacy=True)
    assert isinstance(before, tuple) and isinstance(after, tuple)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_public_frontend_refuses_active_integer_shape_conversion() -> None:
    """Locate parameter-dependent integer metadata before entering the AD runtime."""

    def objective(values: NDArray[np.float64]) -> object:
        count = int(values[0])
        return np.sum(np.zeros(count)) + values[0]

    report = compile_whole_program_frontend(objective)
    assert not report.frontend_ready
    assert "dynamic_integer" in report.semantics_report.unsupported_python_semantics
    with pytest.raises(ValueError, match="dynamic_integer"):
        whole_program_value_and_grad(objective, np.array([2.3]))


@pytest.mark.parametrize(
    ("literal", "key", "outer_indent", "body_indent", "newline"),
    [
        (
            '"""coefficient\n        preserved"""',
            "coefficient\n        preserved",
            "    ",
            "        ",
            "\n",
        ),
        ('"""coefficient\npreserved"""', "coefficient\npreserved", "    ", "        ", "\n"),
        (
            '"""coefficient\n        \n        preserved"""',
            "coefficient\n        \n        preserved",
            "    ",
            "        ",
            "\n",
        ),
        (
            'b"""coefficient\n        preserved"""',
            b"coefficient\n        preserved",
            "    ",
            "        ",
            "\n",
        ),
        (
            '"""coefficient\\\n        preserved"""',
            "coefficient        preserved",
            "    ",
            "        ",
            "\n",
        ),
        (
            'r"""coefficient\\n\n        preserved"""',
            "coefficient\\n\n        preserved",
            "    ",
            "        ",
            "\n",
        ),
        ('f"""coefficient\n        {2}"""', "coefficient\n        2", "    ", "        ", "\n"),
        (
            'f"""coefficient\n        {f\'{2}\'}"""',
            "coefficient\n        2",
            "    ",
            "        ",
            "\n",
        ),
        ('"""coefficient\n\t\tpreserved"""', "coefficient\n\t\tpreserved", "\t", "\t\t", "\n"),
        (
            '"""coefficient\n        preserved"""',
            "coefficient\n        preserved",
            "    ",
            "    \t",
            "\n",
        ),
        (
            '"""coefficient\n        preserved"""',
            "coefficient\n        preserved",
            "    ",
            "  \t\t\t",
            "\n",
        ),
        (
            '"""coefficient\n        preserved"""',
            "coefficient\n        preserved",
            "    ",
            "        ",
            "\r\n",
        ),
    ],
    ids=[
        "margin",
        "no-margin",
        "blank-line",
        "bytes",
        "escaped-line",
        "raw",
        "formatted",
        "nested-formatted",
        "tabs",
        "mixed-tabs",
        "tab-crossing-margin",
        "crlf",
    ],
)
def test_public_frontend_keeps_nested_string_contents_and_source_coordinates(
    tmp_path: Path,
    literal: str,
    key: str | bytes,
    outer_indent: str,
    body_indent: str,
    newline: str,
) -> None:
    """Preserve real source literals through compiler, differentiation and replay.

    Parameters
    ----------
    tmp_path
        Owned location of the real Python objective module.
    literal
        Original multiline token, including its actual interior whitespace.
    key
        Independently specified dictionary key consumed by the objective.
    outer_indent
        Actual lexical nesting margin of the objective definition.
    body_indent
        Actual statement indentation, including legal mixed tab margins.
    newline
        Actual source file line ending.

    """
    source = (
        "def build():\n"
        f"{outer_indent}def objective(values):\n"
        f'{body_indent}"""Original documentation.\n        Retain this margin.\n        """\n'
        f"{body_indent}options = {{{literal}: 2.0}}\n"
        f"{body_indent}return values[0] * options[{key!r}]\n"
        f"{outer_indent}return objective\n"
        "objective = build()\n"
    )
    path = tmp_path / "literal_objective.py"
    path.write_bytes(source.replace("\n", newline).encode("utf-8"))
    specification = importlib.util.spec_from_file_location("literal_objective", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    function = module.objective
    assert isinstance(function, FunctionType)
    baseline = active_reserved_bytes()
    original_constants = function.__code__.co_consts
    report = compile_whole_program_frontend(function)
    assert report.frontend_ready
    assert report.source_start_line == function.__code__.co_firstlineno == 2
    assert report.source_end_line == source.splitlines().index(f"{outer_indent}return objective")
    assert function.__code__.co_consts == original_constants
    result = whole_program_value_and_grad(function, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_public_frontend_keeps_outside_margin_expression_continuations(tmp_path: Path) -> None:
    """Retain legal implicit continuations through source binding and native replay.

    Parameters
    ----------
    tmp_path
        Owned location for a normally imported nested objective module.

    """
    source = (
        "def build():\n"
        "    def objective(values):\n"
        "        return (\n"
        "values[0] *\n"
        "  2.0\n"
        "        )\n"
        "    return objective\n"
        "objective = build()\n"
    )
    path = tmp_path / "continuation_objective.py"
    path.write_text(source, encoding="utf-8")
    specification = importlib.util.spec_from_file_location("continuation_objective", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    function = module.objective
    assert isinstance(function, FunctionType)
    baseline = active_reserved_bytes()
    report = facade_compile_whole_program_frontend(function)
    assert report.frontend_ready
    assert report.source_start_line == function.__code__.co_firstlineno == 2
    assert report.source_end_line == 6
    result = whole_program_value_and_grad(function, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "code_shape", ["co_name", "co_qualname", "wide", "deep", "tuple_subclass"]
)
def test_public_frontend_refuses_non_native_or_unbounded_code_metadata(code_shape: str) -> None:
    """Refuse real loaded metadata before protocols, then recover ordinary execution.

    Parameters
    ----------
    code_shape
        CPython-admitted foreign name, container subclass or oversized constant graph.

    """
    calls: list[str] = []

    class ForeignName(str):
        """Observe unwanted protocols on an admitted CPython code-name subclass."""

        def __repr__(self) -> str:
            """Record accidental metadata formatting."""
            calls.append("name_representation")
            return "foreign"

        def __eq__(self, other: object) -> bool:
            """Record accidental source-identity comparison."""
            calls.append("name_equality")
            return False

        def __hash__(self) -> int:
            """Record accidental metadata hashing."""
            calls.append("name_hashing")
            return 1

    class ForeignTuple(tuple[object, ...]):
        """Observe container protocols before a native-type refusal."""

        def __len__(self) -> int:
            """Record accidental constant-container traversal."""
            calls.append("container_length")
            return super().__len__()

    def objective(values: NDArray[np.float64]) -> object:
        """Provide genuine source for the loaded metadata refusal and recovery."""
        return values[0] * 3.0

    function = cast(FunctionType, objective)
    original_code = function.__code__
    if code_shape == "co_name":
        changed_code = original_code.replace(co_name=ForeignName(original_code.co_name))
    elif code_shape == "co_qualname":
        changed_code = original_code.replace(co_qualname=ForeignName(original_code.co_qualname))
    else:
        constant: object = tuple(range(4097))
        if code_shape == "deep":
            constant = 3.0
            for _ in range(65):
                constant = (constant,)
        elif code_shape == "tuple_subclass":
            constant = ForeignTuple((3.0,))
        changed_code = original_code.replace(co_consts=(None, 0, constant))
    function.__code__ = changed_code
    calls.clear()
    baseline = active_reserved_bytes()
    report = facade_compile_whole_program_frontend(function)
    assert not report.frontend_ready
    assert report.bytecode_instructions == ()
    assert "unsupported_python_semantics:external_callback" in report.hard_gaps
    with pytest.raises(ValueError, match="external_callback"):
        whole_program_value_and_grad(function, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline
    function.__code__ = original_code
    assert facade_compile_whole_program_frontend(function).frontend_ready
    result = whole_program_value_and_grad(function, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "replacement", ["def objective(\n", 'def objective(values):\n    """', ""]
)
def test_public_frontend_refuses_malformed_current_source_and_recovers(
    tmp_path: Path, replacement: str
) -> None:
    """Refuse actual malformed on-disk source before the loaded objective runs.

    Parameters
    ----------
    tmp_path
        Owned location for the real normally imported source module.
    replacement
        Current malformed or absent definition replacing the loaded source bytes.

    """
    source = "def objective(values):\n    return values[0] * 3.0\n"
    path = tmp_path / "changed_source_objective.py"
    path.write_text(source, encoding="utf-8")
    specification = importlib.util.spec_from_file_location("changed_source_objective", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    function = module.objective
    assert isinstance(function, FunctionType)
    original_code = function.__code__
    path.write_text(replacement, encoding="utf-8")
    baseline = active_reserved_bytes()
    calls: list[str] = []

    def profile(frame: FrameType, event: str, argument: object) -> None:
        """Observe actual entry into the already loaded objective code."""
        del argument
        if event == "call" and frame.f_code is original_code:
            calls.append(event)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        report = facade_compile_whole_program_frontend(function)
        assert not report.frontend_ready
        assert "source_frontend_missing" in report.hard_gaps
        with pytest.raises(ValueError, match="source_frontend_missing"):
            whole_program_value_and_grad(function, [2.0], trace=False)
    finally:
        sys.setprofile(previous)
    assert calls == []
    assert function.__code__ is original_code
    assert active_reserved_bytes() == baseline
    path.write_text(source, encoding="utf-8")
    assert facade_compile_whole_program_frontend(function).frontend_ready
    sys.setprofile(profile)
    try:
        result = whole_program_value_and_grad(function, [2.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    finally:
        sys.setprofile(previous)
    assert calls == ["call"]
    assert active_reserved_bytes() == baseline
