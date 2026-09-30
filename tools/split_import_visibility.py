# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dynamic import visibility
"""Expose nonliteral import sites and dependencies in literal lazy tables."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from importlib.util import resolve_name

PACKAGE = "scpn_quantum_control"
IMPORT_OPERATIONS = frozenset({"import_module", "find_spec", "__import__"})


@dataclass(frozen=True)
class DynamicImportSite:
    """An unresolved importer bound to its call AST and complete source text."""

    source: str
    scope: str
    callee: str
    argument: str
    line: int
    call_sha256: str
    source_sha256: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the line-independent identity used by reviewed-site counts."""
        return self.source, self.scope, self.call_sha256


@dataclass
class ImportVisibility:
    """Nonliteral sites, expanded module dependencies and declaration errors."""

    sites: list[DynamicImportSite] = field(default_factory=list)
    dependencies: list[tuple[str, int, str, bool]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _digest(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _last_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return ast.unparse(node)
    return None


def _aliases(tree: ast.Module) -> set[str]:
    aliases = set(IMPORT_OPERATIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"importlib", "importlib.util"}:
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in IMPORT_OPERATIONS
            )
    # Constructor-injected importers and assignment aliases retain their provenance.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.arguments):
                arguments = [*node.posonlyargs, *node.args]
                pairs = list(zip(arguments[-len(node.defaults) :], node.defaults, strict=False))
                pairs.extend(
                    (argument, default)
                    for argument, default in zip(node.kwonlyargs, node.kw_defaults, strict=True)
                    if default is not None
                )
                for argument, default in pairs:
                    if _last_name(default) in aliases:
                        before = len(aliases)
                        aliases.add(argument.arg)
                        changed |= len(aliases) != before
            if isinstance(node, ast.Assign | ast.AnnAssign):
                value = node.value
                if value is None or _last_name(value) not in aliases:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    before = len(aliases)
                    aliases.add(ast.unparse(target).split(".")[-1])
                    changed |= len(aliases) != before
    return aliases


def _table(tree: ast.Module, name: str, result: ImportVisibility) -> ast.Dict | None:
    declarations: list[ast.Assign | ast.AnnAssign] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(ast.unparse(target) == name for target in targets):
                declarations.append(node)
        if (
            isinstance(node, ast.Name | ast.Attribute)
            and ast.unparse(node) == name
            and isinstance(node.ctx, ast.Store | ast.Del)
            and not any(
                node is target
                for declaration in declarations
                for target in (
                    declaration.targets
                    if isinstance(declaration, ast.Assign)
                    else [declaration.target]
                )
            )
        ):
            result.problems.append(f"mutated lazy declaration: {name}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) == name
            and node.func.attr
            not in {
                "items",
                "get",
                "keys",
                "values",
            }
        ):
            result.problems.append(f"mutated lazy declaration: {name}")
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store | ast.Del)
            and ast.unparse(node.value) == name
        ):
            result.problems.append(f"mutated lazy declaration: {name}")
    if not declarations:
        return None
    if len(declarations) != 1 or not isinstance(declarations[0].value, ast.Dict):
        result.problems.append(f"expected one literal lazy declaration: {name}")
        return None
    return declarations[0].value


def _entries(table: ast.Dict, name: str, result: ImportVisibility) -> list[tuple[str, ast.AST]]:
    entries: list[tuple[str, ast.AST]] = []
    seen: set[str] = set()
    for key, value in zip(table.keys, table.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            result.problems.append(f"nonliteral key in lazy declaration: {name}")
        elif key.value in seen:
            result.problems.append(f"duplicate key in lazy declaration: {name}: {key.value}")
        else:
            seen.add(key.value)
            entries.append((key.value, value))
    return entries


def _exports(tree: ast.Module, module: str, result: ImportVisibility) -> None:
    table = _table(tree, "_EXPORT_GROUPS", result)
    if table is None:
        return
    index = "{export_name: module_name for module_name, export_names in _EXPORT_GROUPS.items() for export_name in export_names}"
    indexes = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_EXPORT_MODULES"
            for target in node.targets
        )
    ]
    resolvers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    ]
    expected_lookup = _digest(ast.parse("module_name = _EXPORT_MODULES.get(name)").body[0])
    expected_load = _digest(
        ast.parse('import_module(f"{__name__}.{module_name}")', mode="eval").body
    )
    if (
        len(indexes) != 1
        or _digest(indexes[0]) != _digest(ast.parse(index, mode="eval").body)
        or len(resolvers) != 1
        or not any(_digest(node) == expected_lookup for node in resolvers[0].body)
        or not any(_digest(node) == expected_load for node in ast.walk(resolvers[0]))
        or sum(
            isinstance(node, ast.Name)
            and node.id == "module_name"
            and isinstance(node.ctx, ast.Store)
            for node in ast.walk(resolvers[0])
        )
        != 1
        or sum(
            isinstance(node, ast.Name)
            and node.id == "_EXPORT_MODULES"
            and isinstance(node.ctx, ast.Store | ast.Del)
            for node in ast.walk(tree)
        )
        != 1
        or any(
            isinstance(node, ast.Subscript)
            and ast.unparse(node.value) == "_EXPORT_MODULES"
            and isinstance(node.ctx, ast.Store | ast.Del)
            or isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) == "_EXPORT_MODULES"
            and node.func.attr != "get"
            for node in ast.walk(tree)
        )
    ):
        result.problems.append("lazy export table/resolver mismatch")
    names: set[str] = set()
    for child, value in _entries(table, "_EXPORT_GROUPS", result):
        if not child or not all(part.isidentifier() for part in child.split(".")):
            result.problems.append(f"invalid lazy export module: {child}")
            continue
        if not isinstance(value, ast.Tuple | ast.List) or not value.elts:
            result.problems.append(f"expected nonempty literal export names: {child}")
            continue
        for export in value.elts:
            if (
                not isinstance(export, ast.Constant)
                or not isinstance(export.value, str)
                or not export.value.isidentifier()
            ):
                result.problems.append(f"invalid literal export name: {child}")
            elif export.value in names:
                result.problems.append(f"duplicate lazy export: {export.value}")
            else:
                names.add(export.value)
        result.dependencies.append((module + "." + child, value.lineno, "lazy", False))


def _plugins(tree: ast.Module, result: ImportVisibility) -> None:
    table = _table(tree, "self._lazy_loaders", result)
    if table is None:
        return
    expected_binding = _digest(
        ast.parse("module_path, class_name = self._lazy_loaders[name]").body[0]
    )
    consumers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            isinstance(child, ast.Subscript)
            and ast.unparse(child.value) == "self._lazy_loaders"
            and isinstance(child.ctx, ast.Load)
            for child in ast.walk(node)
        )
    ]
    if not consumers or any(
        not any(_digest(child) == expected_binding for child in ast.walk(consumer))
        or sum(
            isinstance(child, ast.Name)
            and child.id == "module_path"
            and isinstance(child.ctx, ast.Store)
            for child in ast.walk(consumer)
        )
        != 1
        or not any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func).split(".")[-1] == "import_module"
            and len(child.args) == 1
            and isinstance(child.args[0], ast.Name)
            and child.args[0].id == "module_path"
            for child in ast.walk(consumer)
        )
        for consumer in consumers
    ):
        result.problems.append("plugin loader table/resolver mismatch")
    for backend, value in _entries(table, "self._lazy_loaders", result):
        if (
            not isinstance(value, ast.Tuple | ast.List)
            or len(value.elts) != 2
            or not all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in value.elts
            )
        ):
            result.problems.append(f"expected literal module/class loader pair: {backend}")
            continue
        target = value.elts[0]
        assert isinstance(target, ast.Constant) and isinstance(target.value, str)
        class_name = value.elts[1]
        assert isinstance(class_name, ast.Constant) and isinstance(class_name.value, str)
        if not class_name.value.isidentifier():
            result.problems.append(f"invalid plugin loader class name: {backend}")
        elif not target.value.startswith(PACKAGE + "."):
            result.problems.append(f"first-party plugin loader required: {backend}")
        else:
            result.dependencies.append((target.value, value.lineno, "lazy", False))


def inspect_import_visibility(
    tree: ast.Module, source: str, path: str, module: str, package: str
) -> ImportVisibility:
    """Inspect an already parsed production module without importing its code.

    Parameters
    ----------
    tree
        AST owned by the dependency scanner.
    source
        Complete source text binding semantic review to callers and declarations.
    path
        Repository-relative production path.
    module
        Absolute dotted module name, with package initialisers already resolved.
    package
        Absolute package name used to resolve literal relative import calls.

    Returns
    -------
    ImportVisibility
        All nonliteral importer sites, literal dependencies and malformed tables.

    """
    result = ImportVisibility()
    aliases = _aliases(tree)
    source_digest = hashlib.sha256(source.encode()).hexdigest()

    def visit(node: ast.AST, scope: tuple[str, ...], typecheck: bool = False) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = (*scope, node.name)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            typecheck = True
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in aliases:
            argument = (
                node.args[0]
                if node.args
                else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "name"), None
                )
            )
            relative_target: str | None = None
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith(".")
            ):
                package_argument = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "package"),
                    node.args[1] if len(node.args) > 1 else None,
                )
                declared_package = (
                    package
                    if isinstance(package_argument, ast.Name)
                    and package_argument.id == "__package__"
                    else package_argument.value
                    if isinstance(package_argument, ast.Constant)
                    and isinstance(package_argument.value, str)
                    else None
                )
                if declared_package:
                    try:
                        relative_target = resolve_name(argument.value, declared_package)
                    except ImportError:
                        result.problems.append(
                            "relative dynamic import escapes its declared package"
                        )
                if relative_target is not None:
                    if relative_target == PACKAGE or relative_target.startswith(PACKAGE + "."):
                        result.dependencies.append(
                            (relative_target, node.lineno, "dynamic", typecheck)
                        )
                    for child in ast.iter_child_nodes(node):
                        visit(child, scope, typecheck)
                    return
            if (
                not isinstance(argument, ast.Constant)
                or not isinstance(argument.value, str)
                or argument.value.startswith(".")
            ):
                result.sites.append(
                    DynamicImportSite(
                        path,
                        ".".join(scope) or "<module>",
                        ast.unparse(node.func),
                        ast.unparse(argument) if argument is not None else "<missing>",
                        node.lineno,
                        _digest(node),
                        source_digest,
                    )
                )
            elif argument.value == PACKAGE or (
                argument.value.startswith(PACKAGE + ".")
                and (
                    ast.unparse(node.func).split(".")[-1] not in IMPORT_OPERATIONS or not node.args
                )
            ):
                result.dependencies.append((argument.value, node.lineno, "dynamic", typecheck))
        for child in ast.iter_child_nodes(node):
            visit(child, scope, typecheck)

    visit(tree, ())
    _exports(tree, module, result)
    _plugins(tree, result)
    # The provider SDK probe table includes the package root, outside dotted string refs.
    sdk = _table(tree, "_SDK_IMPORTS", result)
    if sdk is not None:
        for _, value in _entries(sdk, "_SDK_IMPORTS", result):
            if not isinstance(value, ast.Tuple | ast.List) or not all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in value.elts
            ):
                result.problems.append("expected literal SDK probe module names")
                continue
            for item in value.elts:
                if isinstance(item, ast.Constant) and item.value == PACKAGE:
                    result.dependencies.append((PACKAGE, item.lineno, "dynamic", False))
    return result
