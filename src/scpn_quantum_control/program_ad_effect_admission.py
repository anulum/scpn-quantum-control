# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — objective effect admission
"""Inspect source-visible calls and external writes without executing objectives.

Unknown callable identities refuse. Source-visible helpers are inspected with
argument provenance; their findings bind to the caller's source location.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import BuiltinMethodType, FunctionType, ModuleType
from typing import cast

import numpy as np

from .program_ad_captured_state import (
    _NUMPY_CALLABLE_IDS,
    _batching_transform,
    _is_passive_local_class,
)
from .program_ad_effect_analysis import (
    ProgramADEffectFinding,
    _Analysis,
    _plain_namespace,
    _source_definition,
)
from .program_ad_effect_call_binding import _container_call_signature_matches
from .program_ad_effect_dispatch import (
    _ARRAY_ALLOCATOR,
    _ARRAY_OUTPUT_POSITIONS,
    _MULTI_VIEW_IDS,
    _NATIVE_EXCEPTION_IDS,
    _NUMPY_OUTPUT_POSITIONS,
    _PURE_IDS,
    _RANDOM_MODULES,
    _READ_METHODS,
    _SCATTER_ADD,
    _UFUNC_AT_METHODS,
    _VIEW_IDS,
    _WRITE_METHODS,
)
from .program_ad_effect_values import (
    _UNKNOWN,
    _is_static_mapping_key,
    _merge_values,
    _Value,
    _value_signature,
)


@dataclass(frozen=True, slots=True)
class _MappedHelper:
    """A source-visible function wrapped by the package's own batching transform."""

    function: FunctionType


def find_objective_effects(
    objective: Callable[..., object], tree: ast.AST
) -> tuple[ProgramADEffectFinding, ...]:
    """Find unsupported external effects before numerical execution.

    Parameters
    ----------
    objective
        Source-visible Python function whose bindings are inspected by identity.
    tree
        Parsed source of that same function, already obtained by the frontend.

    Returns
    -------
    tuple of ProgramADEffectFinding
        Deterministically ordered callback, captured-mutation, ambient-randomness
        and active integer-conversion refusals. Pure helpers and supported local
        operations remain eligible for the existing numerical runtime gate.

    Notes
    -----
    This performs no callback execution, arbitrary attribute resolution or array
    conversion. The existing primitive registry names IR operations, not Python
    callable identities; a registered IR name alone cannot admit an opaque call.

    """
    if type(objective) is not FunctionType:
        return (
            ProgramADEffectFinding(
                tree,
                "external_callback",
                "external callback requires source-visible function identity",
            ),
        )
    function = objective
    definition = _source_definition(function, tree)
    if definition is None:
        return (
            ProgramADEffectFinding(
                tree, "external_callback", "external callback source definition is unavailable"
            ),
        )
    analysis = _Analysis(_Visitor)
    arguments = [_Value(active=True, local=True)]
    analysis.function(function, definition, arguments, {}, (), None)
    return tuple(
        sorted(
            analysis.findings,
            key=lambda finding: (
                getattr(finding.node, "lineno", 1),
                finding.semantic,
                finding.detail,
            ),
        )
    )


class _Visitor(ast.NodeVisitor):
    def __init__(
        self,
        analysis: _Analysis,
        environment: dict[str, _Value],
        stack: tuple[FunctionType, ...],
        location: ast.AST | None,
    ) -> None:
        self.analysis = analysis
        self.environment = environment
        self.stack = stack
        self.location = location
        self.external_names: set[str] = set()
        self.returns: list[_Value] = []
        self.expression_values: dict[ast.AST, _Value] = {}

    def add(self, node: ast.AST, semantic: str, detail: str) -> None:
        """Retain an effect diagnostic at its objective or helper call location."""
        self.analysis.add(self.location or node, semantic, detail)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Refuse captured class reads without invoking attribute descriptors."""
        receiver = self.value(node.value)
        if (
            receiver.captured
            and type(receiver.value) is type
            and id(receiver.value) not in _NUMPY_CALLABLE_IDS
            and not _is_passive_local_class(receiver.value)
        ):
            self.add(
                node,
                "object_attribute",
                "captured class attributes require a supported storage contract",
            )
        self.generic_visit(node)

    def value(self, node: ast.AST) -> _Value:
        """Resolve expression activity and storage provenance without execution."""
        if node in self.expression_values:
            return self.analysis.resolve(self.expression_values[node])
        if isinstance(node, ast.Name):
            return self.analysis.resolve(self.environment.get(node.id, _Value()))
        if isinstance(node, ast.Constant):
            return _Value(node.value, local=True)
        if isinstance(node, ast.IfExp):
            selected = _merge_values([self.value(node.body), self.value(node.orelse)])
            return replace(selected, active=selected.active or self.value(node.test).active)
        if isinstance(node, ast.Starred):
            return self.value(node.value)
        if isinstance(node, ast.Tuple | ast.List):
            elements, variadic = self.positional_values(node.elts)
            return _Value(
                active=any(element.active for element in elements)
                or (variadic is not None and variadic.active),
                local=True,
                elements=elements,
                variadic=variadic,
                container_kind="tuple" if isinstance(node, ast.Tuple) else "list",
            )
        if isinstance(node, ast.Dict):
            literal_fields: dict[object, _Value] = {}
            elements = tuple(self.value(value) for value in node.values)
            key_active = any(self.value(key).active for key in node.keys if key is not None)
            for literal_key, literal_item in zip(node.keys, node.values, strict=True):
                if literal_key is None:
                    expansion = self.mapping_storage(self.value(literal_item))
                    if expansion is None:
                        return _Value(
                            active=key_active,
                            local=True,
                            elements=(),
                            variadic=_merge_values(list(elements)),
                            container_kind="dict",
                        )
                    literal_fields.update(expansion)
                else:
                    field_name = self.value(literal_key).value
                    if not _is_static_mapping_key(field_name):
                        return _Value(
                            active=key_active,
                            local=True,
                            elements=(),
                            variadic=_merge_values(list(elements)),
                            container_kind="dict",
                        )
                    literal_fields[field_name] = self.value(literal_item)
            return _Value(
                active=key_active,
                local=True,
                elements=elements,
                keywords=tuple(literal_fields.items()),
                container_kind="dict",
            )
        if isinstance(node, ast.Call) and any(
            self.value(node.func).value is constructor for constructor in (list, tuple, dict)
        ):
            constructor = self.value(node.func).value
            arguments, variadic = self.positional_values(node.args)
            source = (
                arguments[0]
                if arguments
                else variadic
                if variadic is not None
                else _Value(local=True, elements=())
            )
            if constructor is dict:
                fields = self.mapping_storage(source) if node.args else {}
                keywords = self.keyword_arguments(node)
                if fields is not None and keywords is not None:
                    fields.update(keywords)
                    return _Value(
                        local=True,
                        elements=tuple(fields.values()),
                        keywords=tuple(fields.items()),
                        container_kind="dict",
                    )
                return _Value(
                    active=source.active,
                    local=True,
                    elements=(),
                    variadic=source,
                    container_kind="dict",
                )
            sequence, sequence_tail = self.positional_storage(source)
            return _Value(
                active=source.active,
                local=True,
                elements=sequence,
                variadic=sequence_tail,
                container_kind="list" if constructor is list else "tuple",
            )
        if isinstance(node, ast.Call) and self.value(node.func).value is _ARRAY_ALLOCATOR:
            array_keywords = self.keyword_arguments(node) or {}
            array_arguments, array_variadic = self.positional_values(node.args)
            source = (
                array_arguments[0]
                if array_arguments
                else array_variadic
                if array_variadic is not None
                else array_keywords.get("object", _Value())
            )
            if array_keywords.get("copy", _Value(True, local=True)).value is True:
                return _Value(active=source.active, local=True)
            return _Value(captured=source.captured, active=source.active, local=source.local)
        if isinstance(node, ast.Attribute):
            receiver = self.value(node.value)
            if node.attr == "at":
                for ufunc, method in _UFUNC_AT_METHODS:
                    if receiver.value is ufunc:
                        return _Value(
                            method,
                            captured=receiver.captured,
                            active=receiver.active,
                            local=receiver.local,
                        )
            if _is_passive_local_class(receiver.value):
                namespace = type.__getattribute__(cast(type, receiver.value), "__dict__")
                if node.attr in namespace:
                    return _Value(namespace[node.attr], captured=True)
            if type(receiver.value) is ModuleType:
                module = receiver.value
                namespace = vars(module)
                return (
                    _Value(namespace.get(node.attr, _UNKNOWN), captured=True)
                    if _plain_namespace(namespace)
                    else _Value(captured=True)
                )
            return _Value(captured=receiver.captured, active=receiver.active, local=receiver.local)
        if isinstance(node, ast.Subscript):
            receiver = self.value(node.value)
            subscript_key = self.value(node.slice).value
            projected_fields = self.mapping_storage(receiver)
            if projected_fields is not None:
                if _is_static_mapping_key(subscript_key):
                    return self.analysis.resolve(projected_fields.get(subscript_key, _Value()))
                return self.analysis.resolve(_merge_values(list(projected_fields.values())))
            if receiver.elements is not None:
                if (
                    receiver.container_kind != "dict"
                    and type(subscript_key) is int
                    and -len(receiver.elements) <= subscript_key < len(receiver.elements)
                    and (subscript_key >= 0 or receiver.variadic is None)
                ):
                    return self.analysis.resolve(receiver.elements[subscript_key])
                candidates = list(receiver.elements)
                if receiver.variadic is not None:
                    candidates.append(receiver.variadic)
                return self.analysis.resolve(_merge_values(candidates))
            return _Value(captured=receiver.captured, active=receiver.active, local=receiver.local)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and 1 <= len(node.args) <= 2
        ):
            getter_fields = self.mapping_storage(self.value(node.func.value))
            lookup_key = self.value(node.args[0]).value
            if getter_fields is not None:
                default = (
                    self.value(node.args[1]) if len(node.args) == 2 else _Value(None, local=True)
                )
                selected = (
                    getter_fields.get(lookup_key, default)
                    if _is_static_mapping_key(lookup_key)
                    else _merge_values([*getter_fields.values(), default])
                )
                return self.analysis.resolve(selected)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"copy", "flatten"}
        ):
            source = self.value(node.func.value)
            if node.func.attr == "copy" and (
                source.elements is not None or source.keywords is not None
            ):
                return _Value(
                    active=source.active,
                    local=True,
                    elements=source.elements,
                    keywords=source.keywords,
                    variadic=source.variadic,
                    container_kind=source.container_kind,
                )
            return _Value(active=source.active, local=True)
        if (
            isinstance(node, ast.Call)
            and self.value(node.func).value is cast
            and len(node.args) == 2
        ):
            return self.value(node.args[1])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _READ_METHODS
            and (
                self.value(node.func.value).local
                or type(self.value(node.func.value).value) in (list, tuple, dict, np.ndarray)
            )
        ):
            receiver = self.value(node.func.value)
            return _Value(captured=receiver.captured, active=receiver.active, local=receiver.local)
        if isinstance(node, ast.Call) and id(self.value(node.func).value) in _VIEW_IDS:
            view_arguments, view_variadic = self.positional_values(node.args)
            if id(self.value(node.func).value) in _MULTI_VIEW_IDS:
                view_sources = list(view_arguments)
                if view_variadic is not None:
                    view_sources.append(view_variadic)
            elif view_arguments:
                view_sources = [view_arguments[0]]
            elif view_variadic is not None:
                view_sources = [view_variadic]
            else:
                view_keywords = self.keyword_arguments(node) or {}
                view_sources = [
                    view_keywords[name]
                    for name in ("a", "ary", "array", "m")
                    if name in view_keywords
                ]
            return _Value(
                captured=any(argument.captured for argument in view_sources),
                active=any(argument.active for argument in view_sources),
                local=all(argument.local for argument in view_sources),
            )
        children = [
            self.value(child)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        ]
        return _Value(active=any(child.active for child in children), local=True)

    def write(self, target: ast.AST, node: ast.AST) -> None:
        """Locate unsupported writes to external names or captured storage."""
        if isinstance(target, ast.Name):
            if target.id in self.external_names:
                self.add(node, "captured_mutation", "captured or global mutation is unsupported")
        elif isinstance(target, ast.Subscript | ast.Attribute):
            receiver = self.value(target.value)
            if receiver.captured or not receiver.local:
                self.add(
                    node, "captured_mutation", "captured mutation through an alias is unsupported"
                )
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self.write(element, node)

    def visit_Global(self, node: ast.Global) -> None:
        """Record global names so assignments retain external write provenance."""
        self.external_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Record nonlocal names so assignments retain captured write provenance."""
        self.external_names.update(node.names)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Inspect assignment effects and bind the incoming storage provenance."""
        self.visit(node.value)
        value = self.analysis.resolve(self.value(node.value))
        for target in node.targets:
            self.write(target, node)
            self.bind_target(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Inspect an annotated write and bind its value when one is present."""
        self.write(node.target, node)
        if node.value is not None:
            self.visit(node.value)
            self.bind_target(node.target, self.value(node.value))

    def bind_target(self, target: ast.AST, value: _Value) -> None:
        """Retain element storage identity through ordinary and unpacked bindings."""
        if isinstance(target, ast.Name):
            self.environment[target.id] = self.analysis.resolve(value)
        elif isinstance(target, ast.Subscript):
            self.assign_element(target, value)
        elif isinstance(target, ast.Tuple | ast.List):
            elements, variadic = self.positional_storage(value)
            starred = next(
                (index for index, item in enumerate(target.elts) if isinstance(item, ast.Starred)),
                None,
            )
            exact = variadic is None and (
                len(elements) == len(target.elts)
                if starred is None
                else len(elements) >= len(target.elts) - 1
            )
            if exact and starred is None:
                for item, incoming in zip(target.elts, elements, strict=True):
                    self.bind_target(item, incoming)
            elif exact and starred is not None:
                suffix = len(target.elts) - starred - 1
                for index, item in enumerate(target.elts):
                    if index < starred:
                        incoming = elements[index]
                    elif index == starred:
                        incoming = _Value(
                            local=True,
                            elements=elements[starred : len(elements) - suffix],
                            container_kind="list",
                        )
                    else:
                        incoming = elements[len(elements) - len(target.elts) + index]
                    self.bind_target(
                        item.value if isinstance(item, ast.Starred) else item, incoming
                    )
            else:
                candidates = list(elements)
                if variadic is not None:
                    candidates.append(variadic)
                incoming = _merge_values(candidates)
                for item in target.elts:
                    self.bind_target(
                        item.value if isinstance(item, ast.Starred) else item, incoming
                    )

    def static_slice(self, node: ast.Slice) -> slice | None:
        """Read exact integer slice bounds without conversion or user protocols."""
        bounds = [
            self.value(bound).value if bound is not None else None
            for bound in (node.lower, node.upper, node.step)
        ]
        if not all(bound is None or type(bound) is int for bound in bounds) or bounds[2] == 0:
            return None
        return slice(*cast(list[int | None], bounds))

    def assign_element(self, target: ast.Subscript, incoming: _Value) -> None:
        """Update local container metadata through its shared storage identity."""
        receiver = self.value(target.value)
        if receiver.captured or not receiver.local:
            return
        key = self.value(target.slice).value
        if receiver.keywords is not None:
            fields = dict(receiver.keywords)
            if _is_static_mapping_key(key):
                fields[key] = incoming
            else:
                fields = {name: _merge_values([value, incoming]) for name, value in fields.items()}
            updated = replace(
                receiver, elements=tuple(fields.values()), keywords=tuple(fields.items())
            )
        elif receiver.elements is not None:
            elements = list(receiver.elements)
            if isinstance(target.slice, ast.Slice):
                selection = self.static_slice(target.slice)
                incoming_elements, incoming_tail = self.positional_storage(incoming)
                if selection is not None and receiver.variadic is None and incoming_tail is None:
                    try:
                        elements[selection] = incoming_elements
                    except ValueError:
                        self.add(
                            target, "external_callback", "slice update cardinality is unsupported"
                        )
                        return
                    updated = replace(receiver, elements=tuple(elements))
                else:
                    candidates = [*elements, *incoming_elements]
                    if incoming_tail is not None:
                        candidates.append(incoming_tail)
                    if receiver.variadic is not None:
                        candidates.append(receiver.variadic)
                    updated = replace(receiver, elements=(), variadic=_merge_values(candidates))
            elif (
                type(key) is int
                and -len(elements) <= key < len(elements)
                and (key >= 0 or receiver.variadic is None)
            ):
                elements[key] = incoming
                updated = replace(receiver, elements=tuple(elements))
            else:
                candidates = [*elements, incoming]
                if receiver.variadic is not None:
                    candidates.append(receiver.variadic)
                updated = replace(receiver, elements=(), variadic=_merge_values(candidates))
        else:
            return
        self.analysis.store(receiver, updated)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Propagate augmented values while refusing captured in-place writes."""
        self.write(node.target, node)
        if isinstance(node.target, ast.Name | ast.Subscript):
            value = self.value(node.target)
            if value.captured and type(value.value) not in (
                int,
                float,
                complex,
                str,
                bytes,
                bool,
                tuple,
            ):
                self.add(
                    node,
                    "captured_mutation",
                    "captured mutation through in-place augmentation is unsupported",
                )
        self.visit(node.value)
        if isinstance(node.target, ast.Name | ast.Subscript):
            old = self.value(node.target)
            incoming = self.value(node.value)
            if old.elements is not None and isinstance(node.op, ast.Add):
                elements, tail = self.positional_storage(incoming)
                if old.variadic is None:
                    updated = replace(
                        old,
                        elements=(*old.elements, *elements),
                        variadic=tail,
                        active=old.active or incoming.active,
                    )
                else:
                    candidates = [old.variadic, *elements]
                    if tail is not None:
                        candidates.append(tail)
                    updated = replace(old, variadic=_merge_values(candidates))
                if old.container_kind == "tuple":
                    self.bind_target(node.target, replace(updated, storage_origins=()))
                else:
                    self.analysis.store(old, updated)
                    self.bind_target(node.target, self.analysis.resolve(old))
            else:
                self.bind_target(
                    node.target,
                    _Value(
                        captured=old.captured,
                        active=old.active or incoming.active,
                        local=old.local,
                    ),
                )

    def visit_Delete(self, node: ast.Delete) -> None:
        """Validate deletion receivers and update supported local storage."""
        for target in node.targets:
            self.write(target, node)
            if isinstance(target, ast.Name):
                self.environment.pop(target.id, None)
            elif isinstance(target, ast.Subscript):
                receiver = self.value(target.value)
                if receiver.captured or not receiver.local:
                    continue
                if receiver.elements is None or receiver.container_kind in {"tuple", "generator"}:
                    self.add(
                        node,
                        "external_callback",
                        "local deletion requires known mutable container storage",
                    )
                    continue
                key = self.value(target.slice).value
                if receiver.keywords is not None and _is_static_mapping_key(key):
                    fields = dict(receiver.keywords)
                    fields.pop(key, None)
                    self.analysis.store(
                        receiver,
                        replace(
                            receiver,
                            elements=tuple(fields.values()),
                            keywords=tuple(fields.items()),
                        ),
                    )
                else:
                    elements = list(receiver.elements)
                    selection = (
                        self.static_slice(target.slice)
                        if isinstance(target.slice, ast.Slice)
                        else key
                    )
                    if receiver.variadic is None and (
                        isinstance(selection, slice)
                        or (type(selection) is int and -len(elements) <= selection < len(elements))
                    ):
                        del elements[selection]
                        updated = replace(receiver, elements=tuple(elements))
                    else:
                        candidates = elements + (
                            [receiver.variadic] if receiver.variadic is not None else []
                        )
                        updated = replace(
                            receiver, elements=(), variadic=_merge_values(candidates)
                        )
                    self.analysis.store(receiver, updated)

    def visit_Return(self, node: ast.Return) -> None:
        """Inspect a return expression and retain its value for helper callers."""
        if node.value is not None:
            self.visit(node.value)
            self.returns.append(self.value(node.value))

    def visit_If(self, node: ast.If) -> None:
        """Join both branch environments without dropping captured alias origins."""
        self.visit(node.test)
        before = self.environment.copy()
        storage_before = self.analysis.storage.copy()
        for statement in node.body:
            self.visit(statement)
        body_storage = self.analysis.storage.copy()
        body = {
            name: self.analysis.snapshot(value, body_storage)
            for name, value in self.environment.items()
        }
        self.environment = before.copy()
        self.analysis.storage = storage_before.copy()
        for statement in node.orelse:
            self.visit(statement)
        alternate_storage = self.analysis.storage.copy()
        alternate = {
            name: self.analysis.snapshot(value, alternate_storage)
            for name, value in self.environment.items()
        }
        self.analysis.join_storage(body_storage, alternate_storage)
        self.environment = {
            name: _merge_values([body.get(name, _Value()), alternate.get(name, _Value())])
            for name in body.keys() | alternate.keys()
        }

    def visit_For(self, node: ast.For) -> None:
        """Bind loop items and inspect loop-carried effects and the else body."""
        self.visit(node.iter)
        self.write(node.target, node)
        value = self.iterated_value(self.value(node.iter))
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                self.environment[target.id] = value
        self.loop(node, node.body)
        for statement in node.orelse:
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        """Inspect repeated condition and body effects before the else body."""
        self.loop(node, [ast.Expr(value=node.test), *node.body])
        for statement in node.orelse:
            self.visit(statement)

    def loop(self, node: ast.AST, statements: list[ast.stmt]) -> None:
        """Join loop-carried aliases to a bounded fixed point or refuse the loop."""
        for _ in range(16):
            storage_before = self.analysis.storage.copy()
            before = {
                name: self.analysis.snapshot(value, storage_before)
                for name, value in self.environment.items()
            }
            self.expression_values.clear()
            for statement in statements:
                self.visit(statement)
            storage_after = self.analysis.storage.copy()
            after = {
                name: self.analysis.snapshot(value, storage_after)
                for name, value in self.environment.items()
            }
            self.analysis.join_storage(storage_before, storage_after)
            self.environment = {
                name: _merge_values([before.get(name, _Value()), after.get(name, _Value())])
                for name in before.keys() | after.keys()
            }
            if all(
                name in before and _value_signature(value) == _value_signature(before[name])
                for name, value in self.environment.items()
            ):
                return
        self.add(
            node,
            "external_callback",
            "loop alias effect inspection did not converge within its bound",
        )

    def visit_ListComp(self, node: ast.ListComp | ast.GeneratorExp) -> None:
        """Propagate comprehension item activity within its temporary scope."""
        before = self.environment.copy()
        for generator in node.generators:
            self.visit(generator.iter)
            value = self.iterated_value(self.value(generator.iter))
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name):
                    self.environment[target.id] = value
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.elt)
        element = self.value(node.elt)
        self.expression_values[node] = _Value(
            active=element.active,
            local=True,
            elements=(),
            variadic=element,
            container_kind="list" if isinstance(node, ast.ListComp) else "generator",
        )
        self.environment = before

    visit_GeneratorExp = visit_ListComp

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Locate nested callback definitions without an admitted effect contract."""
        self.add(
            node,
            "external_callback",
            "nested callback definition requires an explicit effect contract",
        )

    def mapping_storage(self, value: _Value) -> dict[object, _Value] | None:
        """Project bounded plain dictionaries without invoking captured key protocols."""
        if value.keywords is not None:
            return dict(value.keywords)
        if type(value.value) is dict:
            mapping = cast(dict[object, object], value.value)
            if len(mapping) > 4096 or not all(_is_static_mapping_key(key) for key in mapping):
                return None
            return {
                name: _Value(item, captured=value.captured, active=value.active, local=value.local)
                for name, item in mapping.items()
            }
        return None

    def keyword_storage(self, value: _Value) -> dict[str, _Value] | None:
        """Require string keys at the actual callback keyword-expansion boundary."""
        fields = self.mapping_storage(value)
        if fields is None or any(type(name) is not str for name in fields):
            return None
        return {cast(str, name): item for name, item in fields.items()}

    def keyword_arguments(self, node: ast.Call) -> dict[str, _Value] | None:
        """Resolve explicit and expanded helper keywords with their storage origin."""
        fields: dict[str, _Value] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                expansion = self.keyword_storage(self.value(keyword.value))
                if expansion is None:
                    self.add(
                        node,
                        "external_callback",
                        "keyword expansion requires known plain mapping storage",
                    )
                    return None
                incoming = expansion
            else:
                incoming = {keyword.arg: self.value(keyword.value)}
            if fields.keys() & incoming.keys():
                self.add(
                    node, "external_callback", "duplicate callback keyword binding is unsupported"
                )
                return None
            fields.update(incoming)
        return fields

    def positional_storage(self, value: _Value) -> tuple[tuple[_Value, ...], _Value | None]:
        """Project known sequence elements and retain an unknown iterable's storage origin."""
        fields = self.mapping_storage(value)
        if fields is not None:
            return tuple(_Value(name, local=True) for name in fields), None
        if value.container_kind == "dict":
            return (), _Value(active=value.active, local=True)
        if value.elements is not None:
            return value.elements, value.variadic
        sequence = value.value
        if type(sequence) in (tuple, list) and len(cast(tuple[object, ...], sequence)) <= 4096:
            return (
                tuple(
                    _Value(item, captured=value.captured, active=value.active, local=value.local)
                    for item in cast(tuple[object, ...], sequence)
                ),
                None,
            )
        if type(sequence) is np.ndarray and sequence.ndim == 1 and sequence.size <= 4096:
            return (
                tuple(
                    _Value(item, captured=value.captured, active=value.active, local=value.local)
                    for item in sequence
                ),
                None,
            )
        return (), _Value(captured=value.captured, active=value.active, local=value.local)

    def positional_values(
        self, expressions: list[ast.expr]
    ) -> tuple[tuple[_Value, ...], _Value | None]:
        """Expand a known prefix and join all possible operands after unknown arity."""
        prefix: list[_Value] = []
        variadic: _Value | None = None
        for expression in expressions:
            if isinstance(expression, ast.Starred):
                elements, tail = self.positional_storage(self.value(expression.value))
            else:
                elements, tail = (self.value(expression),), None
            if variadic is None:
                prefix.extend(elements)
                variadic = tail
            else:
                candidates = [variadic, *elements]
                if tail is not None:
                    candidates.append(tail)
                variadic = _merge_values(candidates)
        return tuple(prefix), variadic

    def iterated_value(self, value: _Value) -> _Value:
        """Project possible iteration elements instead of inheriting container ownership."""
        elements, variadic = self.positional_storage(value)
        candidates = list(elements)
        if variadic is not None:
            candidates.append(variadic)
        return self.analysis.resolve(_merge_values(candidates))

    def container_keywords(self, node: ast.Call, receiver: _Value) -> dict[str, _Value] | None:
        """Bind native container signatures and retain admitted keyword storage.

        Parameters
        ----------
        node
            Source-visible method call whose arguments are inspected statically.
        receiver
            Abstract storage with its known native container kind, when available.

        Returns
        -------
        dict or None
            Admitted keywords, or None after an authored binding refusal.
            Non-container native array methods retain their existing domain.

        """
        assert isinstance(node.func, ast.Attribute)
        keywords = self.keyword_arguments(node)
        if keywords is None:
            return None
        if receiver.container_kind is not None:
            arguments, variadic = self.positional_values(node.args)
            positional_count = len(arguments)
            if variadic is not None:
                positional_count = sum(
                    len(self.positional_storage(self.value(expression.value))[0])
                    if isinstance(expression, ast.Starred)
                    else 1
                    for expression in node.args
                )
            if not _container_call_signature_matches(
                receiver.container_kind,
                node.func.attr,
                positional_count,
                variadic is not None,
                tuple(keywords),
            ):
                self.add(
                    node, "external_callback", "local container call signature is unsupported"
                )
                return None
        return keywords

    def mutate_container(self, node: ast.Call, receiver: _Value) -> None:
        """Track structural updates and returned storage without evaluating protocols."""
        assert isinstance(node.func, ast.Attribute)
        arguments, variadic = self.positional_values(node.args)
        method = node.func.attr
        keywords = self.container_keywords(node, receiver)
        if keywords is None:
            return
        if method == "sort":
            sort_key = keywords.get("key", _Value(None, local=True))
            if sort_key.value is not None and id(sort_key.value) not in _PURE_IDS:
                if type(sort_key.value) is FunctionType:
                    self.analysis.helper(sort_key.value, node, self, self.iterated_value(receiver))
                else:
                    self.add(
                        node,
                        "external_callback",
                        "sort key callback identity has no source-visible effect contract",
                    )
        if method == "clear":
            self.analysis.store(
                receiver,
                replace(
                    receiver,
                    elements=() if receiver.elements is not None else None,
                    keywords=() if receiver.keywords is not None else None,
                    variadic=None,
                ),
            )
        elif method == "pop" and receiver.keywords is not None:
            fields = dict(receiver.keywords)
            key = arguments[0].value if arguments else _UNKNOWN
            default = arguments[1] if len(arguments) > 1 else variadic
            if _is_static_mapping_key(key) and variadic is None:
                result = fields.pop(key, default if default is not None else _Value())
                self.analysis.store(
                    receiver,
                    replace(
                        receiver, elements=tuple(fields.values()), keywords=tuple(fields.items())
                    ),
                )
            else:
                candidates = list(fields.values())
                if default is not None:
                    candidates.append(default)
                result = _merge_values(candidates)
            self.expression_values[node] = self.analysis.resolve(result)
        elif (
            method in {"insert", "pop", "remove", "reverse", "sort"}
            and receiver.elements is not None
        ):
            elements = list(receiver.elements)
            if method == "insert":
                incoming = arguments[1] if len(arguments) > 1 else variadic
                index = arguments[0].value if arguments else _UNKNOWN
                if incoming is None:
                    self.add(
                        node, "external_callback", "local container call signature is unsupported"
                    )
                    return
                if type(index) is int and variadic is None and receiver.variadic is None:
                    elements.insert(index, incoming)
                    updated = replace(receiver, elements=tuple(elements))
                else:
                    candidates = [*elements, incoming]
                    if receiver.variadic is not None:
                        candidates.append(receiver.variadic)
                    updated = replace(receiver, elements=(), variadic=_merge_values(candidates))
            elif method == "pop":
                index = arguments[0].value if arguments else -1
                if (
                    type(index) is int
                    and variadic is None
                    and receiver.variadic is None
                    and -len(elements) <= index < len(elements)
                ):
                    result = elements.pop(index)
                    updated = replace(receiver, elements=tuple(elements))
                else:
                    candidates = elements + (
                        [receiver.variadic] if receiver.variadic is not None else []
                    )
                    result = _merge_values(candidates)
                    updated = replace(receiver, elements=(), variadic=result)
                self.expression_values[node] = self.analysis.resolve(result)
            elif method == "reverse" and receiver.variadic is None:
                updated = replace(receiver, elements=tuple(reversed(elements)))
            elif method == "remove":
                sought = arguments[0].value if arguments else _UNKNOWN
                index = next(
                    (
                        index
                        for index, element in enumerate(elements)
                        if sought is not _UNKNOWN and element.value is sought
                    ),
                    None,
                )
                if index is not None and receiver.variadic is None:
                    del elements[index]
                    updated = replace(receiver, elements=tuple(elements))
                else:
                    candidates = elements + (
                        [receiver.variadic] if receiver.variadic is not None else []
                    )
                    updated = replace(receiver, elements=(), variadic=_merge_values(candidates))
            else:
                candidates = elements + (
                    [receiver.variadic] if receiver.variadic is not None else []
                )
                updated = replace(receiver, elements=(), variadic=_merge_values(candidates))
            self.analysis.store(receiver, updated)
        elif method == "update" and receiver.keywords is not None:
            fields = dict(receiver.keywords)
            if arguments:
                incoming_fields = self.mapping_storage(arguments[0])
                if incoming_fields is None:
                    pairs, pair_tail = self.positional_storage(arguments[0])
                    projected_pairs: dict[object, _Value] = {}
                    for pair in pairs:
                        entries, entry_tail = self.positional_storage(self.analysis.resolve(pair))
                        if entry_tail is not None or len(entries) != 2:
                            break
                        name = entries[0].value
                        if not _is_static_mapping_key(name):
                            break
                        projected_pairs[name] = entries[1]
                    else:
                        if pair_tail is None:
                            incoming_fields = projected_pairs
                if incoming_fields is None:
                    self.add(
                        node, "external_callback", "mapping update requires known plain storage"
                    )
                    return
                fields.update(incoming_fields)
            fields.update(keywords)
            self.analysis.store(
                receiver,
                replace(receiver, elements=tuple(fields.values()), keywords=tuple(fields.items())),
            )
        elif node.func.attr in {"append", "extend"} and receiver.elements is not None:
            incoming_value = arguments[0] if arguments else variadic
            if incoming_value is None:
                self.add(
                    node, "external_callback", "local container call signature is unsupported"
                )
                return
            if node.func.attr == "append":
                incoming_elements: tuple[_Value, ...]
                incoming_elements, incoming_tail = (incoming_value,), None
            else:
                incoming_elements, incoming_tail = self.positional_storage(incoming_value)
            if receiver.variadic is None:
                updated = replace(
                    receiver,
                    elements=(*receiver.elements, *incoming_elements),
                    variadic=incoming_tail,
                )
            else:
                candidates = [receiver.variadic, *incoming_elements]
                if incoming_tail is not None:
                    candidates.append(incoming_tail)
                updated = replace(receiver, variadic=_merge_values(candidates))
            self.analysis.store(receiver, updated)

    def output_storage(self, node: ast.Call, position: int | None) -> None:
        """Refuse external output storage before a supported native call can write it."""
        keywords = self.keyword_arguments(node)
        if keywords is None:
            return
        targets = [keywords["out"]] if "out" in keywords else []
        if position is not None:
            arguments, variadic = self.positional_values(node.args)
            if len(arguments) > position:
                targets.append(arguments[position])
            elif variadic is not None:
                targets.append(variadic)
        pending = list(targets)
        while pending:
            target = pending.pop()
            if target.elements is not None:
                pending.extend(target.elements)
                if target.variadic is not None:
                    pending.append(target.variadic)
                continue
            if target.value is not None and (target.captured or not target.local):
                self.add(
                    node,
                    "captured_mutation",
                    "captured mutation through NumPy output storage is unsupported",
                )

    def visit_Call(self, node: ast.Call) -> None:
        """Admit registered calls and refuse unsupported callbacks or output writes."""
        self.generic_visit(node)
        callee = self.value(node.func)
        if id(callee.value) in _PURE_IDS:
            if id(callee.value) in _NUMPY_CALLABLE_IDS:
                position = (
                    callee.value.nin
                    if type(callee.value) is np.ufunc
                    else _NUMPY_OUTPUT_POSITIONS.get(id(callee.value))
                )
                self.output_storage(node, position)
            if callee.value is int:
                arguments, variadic = self.positional_values(node.args)
                if any(argument.active for argument in arguments) or (
                    variadic is not None and variadic.active
                ):
                    self.add(
                        node,
                        "dynamic_integer",
                        "active integer shape or index conversion is unsupported",
                    )
            return
        if id(callee.value) in _NATIVE_EXCEPTION_IDS and not node.keywords:
            # A native exception is built without foreign code; its payload
            # reaches the caller, so a traced value may not travel in it.
            arguments, variadic = self.positional_values(node.args)
            if any(argument.active for argument in arguments) or (
                variadic is not None and variadic.active
            ):
                self.add(
                    node,
                    "external_callback",
                    "exception payload must not carry a traced value",
                )
            return
        if (
            callee.value is type
            and len(node.args) == 1
            and not node.keywords
            and not isinstance(node.args[0], ast.Starred)
        ):
            # One operand reads a class; three operands would create one.
            return
        if type(callee.value) is _MappedHelper:
            arguments, variadic = self.positional_values(node.args)
            if node.keywords or variadic is not None or len(arguments) != 1:
                self.add(
                    node,
                    "external_callback",
                    "mapped callback is called with one positional batch only",
                )
                return
            # A row is a view of the batch, so the helper is inspected with the
            # batch's own activity and storage provenance.
            mapped = self.analysis.helper(callee.value.function, node, self, arguments[0])
            self.expression_values[node] = _Value(
                active=mapped.active or arguments[0].active, local=True
            )
            return
        if callee.value is _batching_transform():
            arguments, variadic = self.positional_values(node.args)
            if (
                node.keywords
                or variadic is not None
                or len(arguments) != 1
                or type(arguments[0].value) is not FunctionType
            ):
                self.add(
                    node,
                    "external_callback",
                    "batching transform effect contract covers one source-visible "
                    "function mapped over the leading axis",
                )
                return
            self.expression_values[node] = _Value(_MappedHelper(arguments[0].value), local=True)
            return
        if type(callee.value) is BuiltinMethodType:
            method = callee.value
            if (
                type(method.__self__) is np.ufunc
                and id(method.__self__) in _PURE_IDS
                and method.__name__ == "at"
            ):
                if method.__self__ is not _SCATTER_ADD:
                    self.add(
                        node, "external_callback", "in-place ufunc calls support only np.add.at"
                    )
                else:
                    arguments, variadic = self.positional_values(node.args)
                    destination = arguments[0] if arguments else variadic
                    if destination is not None and (destination.captured or not destination.local):
                        self.add(
                            node,
                            "captured_mutation",
                            "captured mutation through scatter is unsupported",
                        )
                return
        if isinstance(node.func, ast.Attribute):
            receiver = self.value(node.func.value)
            if any(receiver.value is module for module in _RANDOM_MODULES):
                self.add(node, "ambient_rng", "ambient random state or rng is unsupported")
                return
            if node.func.attr in _WRITE_METHODS:
                if receiver.captured or not receiver.local:
                    self.add(
                        node,
                        "external_callback",
                        "external callback mutation of captured storage is unsupported",
                    )
                else:
                    self.mutate_container(node, receiver)
                return
            if node.func.attr in _READ_METHODS and (
                receiver.local or type(receiver.value) in (list, tuple, dict, np.ndarray)
            ):
                if (
                    receiver.container_kind is not None
                    and self.container_keywords(node, receiver) is None
                ):
                    return
                if node.func.attr in _ARRAY_OUTPUT_POSITIONS:
                    self.output_storage(node, _ARRAY_OUTPUT_POSITIONS[node.func.attr])
                return
            if (
                node.func.attr == "__array_ufunc__"
                and receiver.active
                and receiver.local
                and not receiver.captured
                and node.args
                and id(self.value(node.args[0]).value) in _PURE_IDS
            ):
                return
            if (
                node.func.attr == "__array_function__"
                and receiver.active
                and receiver.local
                and not receiver.captured
                and node.args
                and id(self.value(node.args[0]).value) in _NUMPY_CALLABLE_IDS
            ):
                # The traced array's own dispatch receives a native identity and
                # never calls it: it traces a supported function and refuses any
                # other by name.
                return
        if _is_passive_local_class(callee.value) and not node.args and not node.keywords:
            self.expression_values[node] = _Value(callee.value, local=True)
            return
        if type(callee.value) is FunctionType:
            self.expression_values[node] = self.analysis.helper(callee.value, node, self)
            return
        self.add(
            node,
            "external_callback",
            "external callback identity has no source-visible effect contract",
        )
