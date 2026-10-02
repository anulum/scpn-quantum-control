# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-visible function effect analysis
"""Bind helper source, captured namespaces and argument origins for effect admission."""

from __future__ import annotations
import __future__

import ast
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from tokenize import TokenError
from types import CodeType, FunctionType
from typing import TYPE_CHECKING, cast

from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import reserve_execution_memory
from .program_ad_effect_source_binding import _source_ast_matches
from .program_ad_effect_values import (
    _UNKNOWN,
    _is_static_mapping_key,
    _merge_values,
    _Storage,
    _Value,
)
from .source_admission import objective_source_block, read_source_lines

if TYPE_CHECKING:
    from .program_ad_effect_admission import _Visitor

_FUTURE_FLAGS = sum(
    feature.compiler_flag
    for feature in (
        __future__.nested_scopes,
        __future__.generators,
        __future__.division,
        __future__.absolute_import,
        __future__.with_statement,
        __future__.print_function,
        __future__.unicode_literals,
        __future__.barry_as_FLUFL,
        __future__.generator_stop,
        __future__.annotations,
    )
)


def _plain_code_metadata(code: CodeType) -> bool:
    """Admit bounded native symbols before any source lookup or comparison."""
    names = (code.co_name, getattr(code, "co_qualname", code.co_name), code.co_filename)
    symbols = (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)
    return (
        all(type(name) is str for name in names)
        and all(
            type(group) is tuple
            and len(group) <= 4096
            and all(type(name) is str for name in group)
            for group in symbols
        )
        and type(code.co_code) is bytes
        and type(getattr(code, "co_exceptiontable", b"")) is bytes
    )


def _code_signature(code: CodeType) -> object:
    """Compare bounded immutable code graphs without calling constant protocols."""
    remaining = 4096

    def project(value: object, depth: int) -> object:
        """Retain executable fields and reject foreign constants before equality."""
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            return _UNKNOWN
        if type(value) is CodeType:
            current = value
            if not _plain_code_metadata(current):
                return _UNKNOWN
            constants = project(current.co_consts, depth + 1)
            if constants is _UNKNOWN:
                return _UNKNOWN
            return (
                current.co_code,
                constants,
                current.co_names,
                current.co_varnames,
                current.co_freevars,
                current.co_cellvars,
                current.co_argcount,
                current.co_posonlyargcount,
                current.co_kwonlyargcount,
                current.co_flags & ~inspect.CO_NESTED,
                getattr(current, "co_exceptiontable", b""),
            )
        if type(value) in (tuple, frozenset):
            sequence = cast(tuple[object, ...], value)
            if len(sequence) > remaining:
                return _UNKNOWN
            items = []
            for item in sequence:
                incoming = project(item, depth + 1)
                if incoming is _UNKNOWN:
                    return _UNKNOWN
                items.append(incoming)
            return frozenset(items) if type(value) is frozenset else tuple(items)
        return value if _is_static_mapping_key(value) or value is Ellipsis else _UNKNOWN

    return project(code, 0)


def _matches_source_code(function: FunctionType, source: list[str]) -> bool:
    """Compile actual file or enclosing lexical source without executing its code."""
    loaded = function.__code__
    signature = _code_signature(loaded)
    if signature is _UNKNOWN:
        return False
    compiled = compile(
        "".join(source),
        loaded.co_filename,
        "exec",
        flags=loaded.co_flags & _FUTURE_FLAGS,
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    pending = [compiled]
    inspected = 0
    while pending and inspected < 4096:
        candidate = pending.pop()
        inspected += 1
        if (
            candidate.co_name == loaded.co_name
            and candidate.co_firstlineno == loaded.co_firstlineno
            and getattr(candidate, "co_qualname", candidate.co_name)
            == getattr(loaded, "co_qualname", loaded.co_name)
        ):
            return _code_signature(candidate) == signature
        pending.extend(item for item in candidate.co_consts if type(item) is CodeType)
    return False


@dataclass(frozen=True, slots=True)
class ProgramADEffectFinding:
    """A refused source effect at the objective's caller location.

    Parameters
    ----------
    node
        Original AST node used by the frontend to bind line and bytecode offsets.
    semantic
        Stable unsupported-effect category.
    detail
        Deliberately authored explanation; no arbitrary object representation.

    """

    node: ast.AST
    semantic: str
    detail: str


def _source_definition(
    function: FunctionType, tree: ast.AST
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | None:
    name = function.__code__.co_name
    if type(name) is not str:
        return None
    if name == "<lambda>":
        candidates = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
        return candidates[0] if len(candidates) == 1 else None
    return next(
        (
            node
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
        ),
        None,
    )


def _plain_namespace(namespace: object) -> bool:
    return (
        type(namespace) is dict
        and len(namespace) <= 4096
        and all(type(key) is str for key in namespace)
    )


def _referenced_environment(
    function: FunctionType, builtin_namespace: dict[str, object]
) -> dict[str, _Value]:
    """Retain referenced native bindings across the admitted executable code graph.

    Parameters
    ----------
    function
        Source-qualified function whose immutable code graph is already bounded.
    builtin_namespace
        Admitted plain builtin dictionary used after ordinary global lookup.

    Returns
    -------
    dict
        Abstract bindings for referenced globals and builtins, including nested
        class and comprehension code. Closure and argument bindings are added
        by the owning function analysis. Unused module names allocate no values.

    """
    environment: dict[str, _Value] = {}
    pending = [function.__code__]
    while pending:
        code = pending.pop()
        for name in code.co_names:
            if name in environment:
                continue
            if name in function.__globals__:
                environment[name] = _Value(function.__globals__[name], captured=True)
            elif name in builtin_namespace:
                environment[name] = _Value(builtin_namespace[name], captured=True)
        pending.extend(item for item in code.co_consts if type(item) is CodeType)
    return environment


def _module_import_scope(statement: ast.stmt) -> bool:
    """Keep module import provenance without borrowing function or class imports."""
    pending: list[ast.AST] = [statement]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            return True
        if not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            pending.extend(ast.iter_child_nodes(node))
    return False


class _Analysis(_Storage):
    def __init__(self, visitor_type: type[_Visitor]) -> None:
        """Own one bounded function analysis using the caller's AST visitor type."""
        self.visitor_type = visitor_type
        self.findings: list[ProgramADEffectFinding] = []
        self.inspections = 0
        super().__init__()

    def add(self, node: ast.AST, semantic: str, detail: str) -> None:
        """Retain one diagnostic for each distinct source node and effect reason."""
        if not any(
            finding.node is node and finding.semantic == semantic and finding.detail == detail
            for finding in self.findings
        ):
            self.findings.append(ProgramADEffectFinding(node, semantic, detail))

    def root_source_matches(self, function: FunctionType, node: ast.AST) -> bool:
        """Bind root code to its actual file before admitting source-level effects."""
        try:
            observation = Path(function.__code__.co_filename).stat()
        except OSError:
            self.add(node, "external_callback", "external callback source is unavailable")
            return False
        plan = ExecutionMemoryPlan(
            (
                ExecutionBuffer(
                    "effect_root_source", "intermediate", (max(1, observation.st_size),), "uint8"
                ),
            )
        )
        with reserve_execution_memory(plan) as reservation:
            lines, source_plan = read_source_lines(
                function.__code__.co_filename, observation, reservation
            )
            try:
                code = function.__code__
                tree = cast(
                    ast.Module,
                    compile(
                        "".join(lines),
                        code.co_filename,
                        "exec",
                        flags=ast.PyCF_ONLY_AST | (code.co_flags & _FUTURE_FLAGS),
                        dont_inherit=True,
                        optimize=sys.flags.optimize,
                    ),
                )
                matching = False
                for statement in tree.body:
                    start = statement.lineno
                    if isinstance(
                        statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                    ):
                        start = min([start, *(item.lineno for item in statement.decorator_list)])
                    end = statement.end_lineno or statement.lineno
                    if not start <= code.co_firstlineno <= end:
                        continue
                    selected = ["\n"] * len(lines)
                    for imported in tree.body:
                        if _module_import_scope(imported):
                            stop = imported.end_lineno or imported.lineno
                            selected[imported.lineno - 1 : stop] = lines[
                                imported.lineno - 1 : stop
                            ]
                    selected[start - 1 : end] = lines[start - 1 : end]
                    capacity = sum(len(line) * (1 if line.isascii() else 4) for line in selected)
                    reservation.resize(
                        ExecutionMemoryPlan(
                            (
                                *source_plan.buffers,
                                ExecutionBuffer(
                                    "effect_root_compilation",
                                    "intermediate",
                                    (max(1, capacity),),
                                    "uint8",
                                    128,
                                ),
                            )
                        )
                    )
                    matching = _source_ast_matches(code, node, statement) and _matches_source_code(
                        function, selected
                    )
                    break
            except (SyntaxError, ValueError):
                matching = False
            if not matching:
                self.add(
                    node,
                    "external_callback",
                    "external callback source does not match captured function",
                )
            return matching

    def function(
        self,
        function: FunctionType,
        definition: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        arguments: list[_Value],
        keywords: dict[str, _Value],
        stack: tuple[FunctionType, ...],
        location: ast.AST | None,
        variadic: _Value | None = None,
    ) -> _Value:
        """Inspect a source-bound callable with propagated arguments and captures."""
        self.inspections += 1
        root = definition if location is None else location
        if isinstance(definition, ast.AsyncFunctionDef):
            if location is None:
                self.add(root, "async_function", "async_function")
            else:
                self.add(
                    root, "external_callback", "asynchronous external callback is unsupported"
                )
            return _Value()
        if any(function is previous for previous in stack) or self.inspections > 64:
            self.add(
                root,
                "external_callback",
                "external callback recursion or inspection bound is unsupported",
            )
            return _Value()
        if not _plain_code_metadata(function.__code__):
            self.add(
                root,
                "external_callback",
                "external callback code metadata must use plain immutable values",
            )
            return _Value()
        if location is None and not self.root_source_matches(function, root):
            return _Value()
        builtin_namespace: object = FunctionType.__getattribute__(function, "__builtins__")
        namespaces = [function.__globals__, builtin_namespace, function.__dict__]
        if function.__kwdefaults__ is not None:
            namespaces.append(function.__kwdefaults__)
        if any(not _plain_namespace(namespace) for namespace in namespaces) or (
            function.__defaults__ is not None and type(function.__defaults__) is not tuple
        ):
            self.add(
                root,
                "external_callback",
                "external callback namespace must be a bounded plain dictionary",
            )
            return _Value()
        environment = _referenced_environment(function, cast(dict[str, object], builtin_namespace))
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        ):
            try:
                environment[name] = _Value(cell.cell_contents, captured=True)
            except ValueError:
                self.add(
                    root, "external_callback", "external callback has an empty captured binding"
                )
                return _Value()
        parameters = [*definition.args.posonlyargs, *definition.args.args]
        defaults = function.__defaults__ or ()
        for index, parameter in enumerate(parameters):
            default_index = index - (len(parameters) - len(defaults))
            fallback = (
                _Value(defaults[default_index], captured=True) if default_index >= 0 else _Value()
            )
            if index < len(arguments):
                environment[parameter.arg] = arguments[index]
            elif variadic is not None:
                candidates = [variadic]
                if parameter.arg in keywords:
                    candidates.append(keywords[parameter.arg])
                elif default_index >= 0:
                    candidates.append(fallback)
                environment[parameter.arg] = _merge_values(candidates)
            else:
                environment[parameter.arg] = keywords.get(parameter.arg, fallback)
        if definition.args.vararg is not None:
            remaining = tuple(arguments[len(parameters) :])
            environment[definition.args.vararg.arg] = _Value(
                active=any(value.active for value in remaining)
                or (variadic is not None and variadic.active),
                local=True,
                elements=remaining,
                variadic=variadic,
                container_kind="tuple",
            )
        for parameter in definition.args.kwonlyargs:
            environment[parameter.arg] = keywords.get(
                parameter.arg,
                _Value(
                    (function.__kwdefaults__ or {}).get(parameter.arg, _UNKNOWN), captured=True
                ),
            )
        if definition.args.kwarg is not None:
            consumed = {
                parameter.arg for parameter in (*definition.args.args, *definition.args.kwonlyargs)
            }
            environment[definition.args.kwarg.arg] = _Value(
                local=True,
                container_kind="dict",
                keywords=tuple(
                    (name, value) for name, value in keywords.items() if name not in consumed
                ),
            )
        visitor = self.visitor_type(self, environment, (*stack, function), location)
        if isinstance(definition, ast.Lambda):
            visitor.visit(definition.body)
            visitor.returns.append(visitor.value(definition.body))
        else:
            for statement in definition.body:
                visitor.visit(statement)
        return _merge_values(visitor.returns)

    def helper(
        self,
        function: FunctionType,
        node: ast.Call,
        visitor: _Visitor,
        callback_argument: _Value | None = None,
    ) -> _Value:
        """Inspect ordinary helper calls or one-element native callback bindings."""
        if callback_argument is None:
            keywords = visitor.keyword_arguments(node)
            if keywords is None:
                return _Value()
            arguments, variadic = visitor.positional_values(node.args)
        else:
            keywords = {}
            arguments, variadic = (callback_argument,), None
        if not _plain_code_metadata(function.__code__):
            self.add(
                visitor.location or node,
                "external_callback",
                "external callback code metadata must use plain immutable values",
            )
            return _Value()
        if not _plain_namespace(function.__dict__) or "__wrapped__" in function.__dict__:
            self.add(
                visitor.location or node,
                "external_callback",
                "external callback wrapper identity is unsupported",
            )
            return _Value()
        try:
            observation = Path(function.__code__.co_filename).stat()
        except OSError:
            self.add(
                visitor.location or node,
                "external_callback",
                "external callback source is unavailable",
            )
            return _Value()
        plan = ExecutionMemoryPlan(
            (
                ExecutionBuffer(
                    "effect_helper_source", "intermediate", (max(1, observation.st_size),), "uint8"
                ),
            )
        )
        with reserve_execution_memory(plan) as reservation:
            lines, source_plan = read_source_lines(
                function.__code__.co_filename, observation, reservation
            )
            reservation.resize(
                ExecutionMemoryPlan(
                    (
                        *source_plan.buffers,
                        ExecutionBuffer(
                            "effect_helper_compilation",
                            "intermediate",
                            (max(1, observation.st_size),),
                            "uint8",
                            128,
                        ),
                    )
                )
            )
            try:
                block, _ = objective_source_block(function, lines)
                source = "".join(block)
                indented = source.startswith((" ", "\t"))
                tree = ast.parse(("if True:\n" if indented else "") + source)
                if indented:
                    tree = ast.Module(body=cast(ast.If, tree.body[0]).body, type_ignores=[])
            except (OSError, SyntaxError, TokenError):
                self.add(
                    visitor.location or node,
                    "external_callback",
                    "external callback source cannot be inspected",
                )
                return _Value()
            definition = _source_definition(function, tree)
            if definition is None:
                self.add(
                    visitor.location or node,
                    "external_callback",
                    "external callback source definition is unavailable",
                )
                return _Value()
            try:
                matching_source = _matches_source_code(function, lines)
            except (SyntaxError, ValueError):
                matching_source = False
            if not matching_source:
                self.add(
                    visitor.location or node,
                    "external_callback",
                    "external callback source does not match captured function",
                )
                return _Value()
            return self.function(
                function,
                definition,
                list(arguments),
                keywords,
                visitor.stack,
                visitor.location or node,
                variadic,
            )
