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
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.whole_program_frontend as frontend_module
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


def test_frontend_metadata_and_introspection_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate source metadata and unavailable source/bytecode fallbacks."""
    metadata_type = frontend_module._ObjectiveSourceMetadata
    for factory, message in (
        (lambda: metadata_type("", 1, 1), "source"),
        (lambda: metadata_type("x", 0, 1), "start_line"),
        (lambda: metadata_type("x", 2, 1), "end_line"),
    ):
        with pytest.raises(ValueError, match=message):
            factory()

    def objective(value: object) -> object:
        return value

    monkeypatch.setattr(frontend_module.inspect, "getsourcelines", lambda _value: (["  \n"], 1))
    assert frontend_module._objective_source_metadata(objective) is None
    assert frontend_module._objective_source(objective) is None

    def unavailable(_value: object) -> object:
        raise TypeError("unavailable")

    monkeypatch.setattr(frontend_module.inspect, "getsourcelines", unavailable)
    assert frontend_module._objective_source_metadata(objective) is None
    monkeypatch.setattr(frontend_module.dis, "get_instructions", unavailable)
    assert frontend_module._objective_bytecode(objective) == ()
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
    assert frontend_module._ast_subscript_root(ast.parse("factory()[0]", mode="eval").body) == ""
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
        frontend_module.inspect,
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
        frontend_module.inspect,
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

    def closure_factory() -> object:
        token = object()

        def closure() -> object:
            return token

        return closure

    closure = closure_factory()

    class NonMappingGlobals:
        __code__ = closure.__code__
        __globals__: list[object] = []

    assert "token" in frontend_module._captured_or_global_names(NonMappingGlobals())
