# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — split ownership and dependency inventory
"""Inventory who owns every tracked path and how the package's units depend on each other.

The workbench is planned to split into focused ``SCPN-QC-*`` repositories. Before any
boundary work can be reviewed, every tracked path needs exactly one current owner and one
proposed destination, and every import that crosses a domain boundary needs to be known with
its kind. This tool produces that inventory from the Git index and the source text alone; it
never imports the package unless ``--measure-import-cost`` is given.

The domain assignment is data (``data/split_preparation/split_domain_map.json``), not code.
A unit whose files belong to different targets is listed under ``split_units`` with every
file assigned explicitly; a file of such a unit that the map does not list is a problem.
The tool fails closed: an unknown unit, a unit listed twice, a unit in the map that no longer
exists, a tracked path no rule owns, or a Python file that does not parse is an error, and the
command exits non-zero after writing the problems it found.

Edge kinds for imports of package units:

``module``      top-level statement;
``module_try``  top-level statement inside ``try``;
``lazy``        inside a function body;
``typecheck``   under ``if TYPE_CHECKING``;
``dynamic``     ``importlib.import_module`` / ``find_spec`` / ``__import__`` with a literal name;
``string_ref``  a non-docstring string literal naming ``scpn_quantum_control.<unit>``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.split_import_visibility import (
    DynamicImportSite,
    ImportVisibility,
    inspect_import_visibility,
)

PACKAGE = "scpn_quantum_control"
SOURCE_PREFIX = f"src/{PACKAGE}/"
DEFAULT_MAP = Path("data/split_preparation/split_domain_map.json")
RUNTIME_KINDS = frozenset({"module", "module_try", "lazy", "dynamic"})
MODULE_KINDS = frozenset({"module", "module_try"})
UMBRELLA = "workbench-umbrella"
UNCLASSIFIED = "unclassified"
DYNAMIC_CALLS = frozenset({"import_module", "find_spec", "__import__"})
_STRING_REF = re.compile(
    rf"{PACKAGE}\.([A-Za-z_][A-Za-z0-9_]*)(?:(?!\.{PACKAGE}\.)\.[A-Za-z_][A-Za-z0-9_]*)*"
)
_SOURCE_PATH_REF = re.compile(rf"src/{PACKAGE}/[A-Za-z0-9_/]+\.py")
_TEXT_SCAN_LIMIT = 8 * 1024 * 1024
_STDLIB = frozenset(sys.stdlib_module_names)


class OwnershipError(ValueError):
    """Raised when the domain map itself is malformed."""


@dataclass(frozen=True)
class PathRule:
    """One ordered ownership rule for tracked paths outside the package source."""

    prefix: str
    kind: str
    target: str
    root_files_only: bool = False


@dataclass(frozen=True)
class DomainMap:
    """Validated unit → domain → target assignment."""

    unit_domain: dict[str, str]
    domain_target: dict[str, str]
    targets: frozenset[str]
    empty_targets: dict[str, str]
    path_rules: tuple[PathRule, ...]
    open_classifications: tuple[tuple[str, str, str], ...]
    extra_import_names: dict[str, tuple[str, ...]]
    split_files: dict[str, str] = field(default_factory=dict)

    def target_of_unit(self, unit: str) -> str | None:
        """Return the target repository of ``unit``, or ``None`` when the unit is unmapped."""
        domain = self.unit_domain.get(unit)
        return None if domain is None else self.domain_target[domain]


@dataclass(frozen=True)
class Edge:
    """One import of a package unit found in a tracked Python file."""

    source_path: str
    target_unit: str
    kind: str
    line: int
    target_module: str = ""
    root_export: bool = False
    import_names: tuple[str, ...] = ()
    typecheck_context: bool = False
    literal_table: bool = False


@dataclass
class FileRecord:
    """Ownership record for one tracked path."""

    path: str
    sha256: str
    kind: str
    unit: str
    domain: str
    target: str
    reason: str = ""


@dataclass
class Inventory:
    """Everything one run found; ``problems`` empty means the inventory is complete."""

    head: str
    files: list[FileRecord] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    external_imports: dict[str, set[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    dynamic_sites: list[DynamicImportSite] = field(default_factory=list)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OwnershipError(f"duplicate key in domain map: {key!r}")
        result[key] = value
    return result


def load_domain_map(path: Path) -> DomainMap:
    """Load and validate the domain map.

    Parameters
    ----------
    path
        JSON file with schema ``scpn_qc_split_domain_map_v2``.

    Returns
    -------
    DomainMap
        The validated map.

    Raises
    ------
    OwnershipError
        On duplicate keys, a unit listed under two domains, an unknown target, a split file
        listed twice or under an unknown domain or unit, or an unsupported schema.

    """
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if raw.get("schema") != "scpn_qc_split_domain_map_v2":
        raise OwnershipError(f"unsupported domain-map schema: {raw.get('schema')!r}")
    targets = frozenset(raw["targets"]) | {UMBRELLA}
    unit_domain: dict[str, str] = {}
    domain_target: dict[str, str] = {}
    for domain, spec in raw["domains"].items():
        if spec["target"] not in targets:
            raise OwnershipError(f"domain {domain!r} names unknown target {spec['target']!r}")
        domain_target[domain] = spec["target"]
        for unit in spec["units"]:
            if unit in unit_domain:
                raise OwnershipError(
                    f"unit {unit!r} assigned to both {unit_domain[unit]!r} and {domain!r}"
                )
            unit_domain[unit] = domain
    rules = []
    for rule in raw["path_rules"]:
        if rule["target"] not in targets | {"by-unit", "by-imports"}:
            raise OwnershipError(f"path rule {rule['prefix']!r} names unknown target")
        rules.append(
            PathRule(
                prefix=rule["prefix"],
                kind=rule["kind"],
                target=rule["target"],
                root_files_only=bool(rule.get("root_files_only", False)),
            )
        )
    for name in raw.get("empty_targets", {}):
        if name not in targets:
            raise OwnershipError(f"empty target {name!r} is not a declared target")
    split_files: dict[str, str] = {}
    for unit, by_domain in raw.get("split_units", {}).items():
        if unit not in unit_domain:
            raise OwnershipError(f"split unit {unit!r} is not a mapped unit")
        for domain, names in by_domain.items():
            if domain not in domain_target:
                raise OwnershipError(f"split unit {unit!r} names unknown domain {domain!r}")
            for name in names:
                key = f"{unit}/{name}"
                if key in split_files:
                    raise OwnershipError(f"split file {key!r} assigned twice")
                split_files[key] = domain
    return DomainMap(
        unit_domain=unit_domain,
        domain_target=domain_target,
        targets=targets,
        empty_targets=dict(raw.get("empty_targets", {})),
        path_rules=tuple(rules),
        open_classifications=tuple(
            (item["unit"], item["reason"], item.get("decision", "open"))
            for item in raw.get("open_classifications", [])
        ),
        extra_import_names={
            dist: tuple(names) for dist, names in raw.get("extra_import_names", {}).items()
        },
        split_files=split_files,
    )


def tracked_files(repo: Path) -> list[str]:
    """Return every path in the Git index of ``repo``, sorted."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return sorted(p for p in completed.stdout.decode("utf-8").split("\0") if p)


def git_head(repo: Path) -> str:
    """Return the full commit id of ``HEAD`` in ``repo``."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def unit_of(source_path: str) -> str:
    """Return the top-level package unit of a path under ``src/scpn_quantum_control/``."""
    parts = source_path[len(SOURCE_PREFIX) :].split("/")
    if len(parts) == 1:
        return parts[0].removesuffix(".py")
    return parts[0]


def _module_parts(path: str) -> tuple[list[str], bool]:
    """Return the dotted module parts of a package path and whether it is a package init."""
    parts = path[len("src/") :].removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        return parts[:-1], True
    return parts, False


def _import_kind(stack: Sequence[ast.AST]) -> str:
    for node in stack:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            return "typecheck"
    if any(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) for n in stack):
        return "lazy"
    if any(isinstance(n, ast.Try) for n in stack):
        return "module_try"
    return "module"


def _resolve_from(node: ast.ImportFrom, module: list[str], is_package: bool) -> list[list[str]]:
    if node.level == 0:
        base = (node.module or "").split(".")
    else:
        package = module if is_package else module[:-1]
        up = node.level - 1
        anchor = package[: len(package) - up] if up else package
        base = anchor + (node.module.split(".") if node.module else [])
    if base == [PACKAGE]:
        return [[PACKAGE, alias.name] for alias in node.names]
    return [base]


def scan_python(
    source: str,
    path: str,
    module: list[str] | None = None,
    is_package: bool = False,
    *,
    visibility: ImportVisibility | None = None,
) -> tuple[list[Edge], set[str]]:
    """Find package-unit imports and third-party top-level imports in one Python file.

    Parameters
    ----------
    source
        File text.
    path
        Repository-relative path, recorded on every edge.
    module
        Dotted module parts of the file when it lies inside the package (needed to resolve
        relative imports); ``None`` for files outside the package.
    is_package
        Whether the file is a package ``__init__``.
    visibility
        Optional accumulator for nonliteral production import sites and table errors.

    Returns
    -------
    tuple[list[Edge], set[str]]
        Edges to package units, and the set of third-party top-level import names.

    Raises
    ------
    SyntaxError
        When the file does not parse.

    """
    tree = ast.parse(source, filename=path)
    here = module or []
    edges: list[Edge] = []
    external: set[str] = set()

    def record(
        target: list[str],
        kind: str,
        line: int,
        *,
        root_export: bool = False,
        import_names: tuple[str, ...] = (),
        typecheck_context: bool = False,
    ) -> None:
        if target and target[0] == PACKAGE:
            if len(target) > 1:
                edges.append(
                    Edge(
                        path,
                        target[1],
                        kind,
                        line,
                        ".".join(target),
                        root_export,
                        import_names,
                        typecheck_context,
                    )
                )
        elif target and target[0] and target[0] not in _STDLIB and target[0] != "__future__":
            external.add(target[0])

    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        docstring_holder = isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
        for index, child in enumerate(ast.iter_child_nodes(node)):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    record(alias.name.split("."), _import_kind(stack), child.lineno)
            elif isinstance(child, ast.ImportFrom):
                if child.level and module is None:
                    continue
                for target in _resolve_from(child, here, is_package):
                    root_export = len(target) == 2 and (
                        child.module == PACKAGE
                        or (
                            child.module is None
                            and bool(child.level)
                            and len(here) - int(not is_package) == child.level
                        )
                    )
                    names = tuple(alias.name for alias in child.names) if not root_export else ()
                    record(
                        target,
                        _import_kind(stack),
                        child.lineno,
                        root_export=root_export,
                        import_names=names,
                    )
            elif isinstance(child, ast.Call):
                name = ast.unparse(child.func).split(".")[-1]
                first = child.args[0] if child.args else None
                if (
                    name in DYNAMIC_CALLS
                    and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                ):
                    record(
                        first.value.split("."),
                        "dynamic",
                        child.lineno,
                        typecheck_context=_import_kind(stack) == "typecheck",
                    )
                    continue
            elif (
                isinstance(child, ast.Expr)
                and isinstance(child.value, ast.Constant)
                and isinstance(child.value.value, str)
                and docstring_holder
                and index == 0
            ):
                continue
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                for match in _STRING_REF.finditer(child.value):
                    edges.append(
                        Edge(
                            path,
                            match.group(1),
                            "string_ref",
                            child.lineno,
                            match.group(0),
                            typecheck_context=_import_kind(stack) == "typecheck",
                        )
                    )
            visit(child, [*stack, child])

    visit(tree, [])
    if module is not None:
        discovered = inspect_import_visibility(
            tree, source, path, ".".join(module), ".".join(module if is_package else module[:-1])
        )
        for target, line, kind, typecheck in discovered.dependencies:
            parts = target.split(".")
            edges.append(
                Edge(
                    path,
                    parts[1] if len(parts) > 1 else "__init__",
                    kind,
                    line,
                    target,
                    root_export=len(parts) == 1,
                    typecheck_context=typecheck,
                    literal_table=True,
                )
            )
        if visibility is not None:
            visibility.sites.extend(discovered.sites)
            visibility.dependencies.extend(discovered.dependencies)
            visibility.problems.extend(f"{path}: {problem}" for problem in discovered.problems)
    return edges, external


def _rule_for(path: str, rules: Iterable[PathRule]) -> PathRule | None:
    for rule in rules:
        if rule.root_files_only:
            if "/" not in path:
                return rule
            continue
        if path.startswith(rule.prefix):
            return rule
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_inventory(repo: Path, domain_map: DomainMap) -> Inventory:
    """Classify every tracked path of ``repo`` and collect its package-unit imports.

    Problems (unknown units, stale map units, unowned paths, parse errors) are collected in
    ``Inventory.problems`` rather than raised, so one run reports all of them.
    """
    inventory = Inventory(head=git_head(repo))
    paths = tracked_files(repo)
    source_units: set[str] = set()
    python_files: list[tuple[str, list[str] | None, bool]] = []
    for path in paths:
        if path.startswith(SOURCE_PREFIX) and path.endswith(".py"):
            source_units.add(unit_of(path))
            module, is_package = _module_parts(path)
            python_files.append((path, module, is_package))
        elif path.endswith(".py") and not path.startswith(
            ("oscillatools/", "scpn_quantum_engine/")
        ):
            python_files.append((path, None, False))

    for unit in sorted(source_units - domain_map.unit_domain.keys()):
        inventory.problems.append(f"unknown unit (not in domain map): {unit}")
    for unit in sorted(domain_map.unit_domain.keys() - source_units):
        inventory.problems.append(f"stale unit (in domain map, not in tree): {unit}")

    imports_by_file: dict[str, set[str]] = defaultdict(set)
    for path, file_module, file_is_package in python_files:
        try:
            source = (repo / path).read_text(encoding="utf-8")
            visibility = ImportVisibility()
            edges, external = scan_python(
                source, path, file_module, file_is_package, visibility=visibility
            )
            inventory.dynamic_sites.extend(visibility.sites)
            inventory.problems.extend(visibility.problems)
        except (SyntaxError, UnicodeDecodeError) as exc:
            inventory.problems.append(f"parse error: {path}: {type(exc).__name__}: {exc}")
            continue
        # A name imported from the package root that is not a unit (a re-exported class or
        # function) is an import of the facade, exactly as Python executes it.
        edges = [
            e
            if e.target_unit in source_units
            else Edge(
                e.source_path,
                "__init__",
                e.kind,
                e.line,
                e.target_module,
                e.root_export,
                e.import_names,
                e.typecheck_context,
                e.literal_table,
            )
            for e in edges
        ]
        inventory.edges.extend(edges)
        for name in external:
            inventory.external_imports.setdefault(name, set()).add(path)
        for edge in edges:
            if edge.kind != "string_ref" and edge.target_unit in source_units:
                imports_by_file[path].add(edge.target_unit)

    for path in paths:
        record = _classify(path, repo, domain_map, imports_by_file.get(path, set()))
        if record is None:
            inventory.problems.append(f"unowned path (no rule): {path}")
            continue
        if record.target == UNCLASSIFIED:
            inventory.problems.append(f"unclassified file in split unit: {path}")
        inventory.files.append(record)
    present = {p[len(SOURCE_PREFIX) :] for p in paths if p.startswith(SOURCE_PREFIX)}
    for key in sorted(domain_map.split_files.keys() - present):
        inventory.problems.append(f"stale split file (in domain map, not in tree): {key}")
    inventory.edges.sort(key=lambda e: (e.source_path, e.line, e.target_unit, e.kind))
    return inventory


def _classify(
    path: str, repo: Path, domain_map: DomainMap, imported_units: set[str]
) -> FileRecord | None:
    rule = _rule_for(path, domain_map.path_rules)
    if rule is None:
        return None
    digest = _sha256(repo / path) if (repo / path).is_file() else "absent-in-worktree"
    if rule.target == "by-unit":
        unit = unit_of(path)
        domain = domain_map.unit_domain.get(unit, "")
        if any(key.startswith(unit + "/") for key in domain_map.split_files):
            split_domain = domain_map.split_files.get(path[len(SOURCE_PREFIX) :])
            if split_domain is None:
                return FileRecord(path, digest, rule.kind, unit, "", UNCLASSIFIED)
            target = domain_map.domain_target[split_domain]
            return FileRecord(path, digest, rule.kind, unit, split_domain, target, "split unit")
        target = domain_map.target_of_unit(unit) or "unmapped"
        return FileRecord(path, digest, rule.kind, unit, domain, target)
    if rule.target == "by-imports":
        targets = sorted({domain_map.target_of_unit(u) or "unmapped" for u in imported_units})
        if not targets:
            return FileRecord(path, digest, rule.kind, "", "", UMBRELLA, "imports no package unit")
        if len(targets) == 1:
            return FileRecord(path, digest, rule.kind, "", "", targets[0], "single target")
        reason = "integration: " + ",".join(targets)
        return FileRecord(path, digest, rule.kind, "", "", UMBRELLA, reason)
    return FileRecord(path, digest, rule.kind, "", "", rule.target, "path rule " + rule.prefix)


def domain_weights(
    edges: Iterable[Edge], domain_map: DomainMap, kinds: frozenset[str]
) -> dict[str, dict[str, int]]:
    """Count distinct (source file, target unit) pairs between different source domains.

    Only edges from package source files count; the facade domain is excluded, as in the
    planning snapshot, because it re-exports everything by design.
    """
    pairs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for edge in edges:
        if edge.kind not in kinds or not edge.source_path.startswith(SOURCE_PREFIX):
            continue
        source_domain = domain_map.unit_domain.get(unit_of(edge.source_path))
        target_domain = domain_map.unit_domain.get(edge.target_unit)
        if not source_domain or not target_domain or source_domain == target_domain:
            continue
        if "facade" in (source_domain, target_domain):
            continue
        pairs[(source_domain, target_domain)].add((edge.source_path, edge.target_unit))
    weights: dict[str, dict[str, int]] = defaultdict(dict)
    for (source_domain, target_domain), members in pairs.items():
        weights[source_domain][target_domain] = len(members)
    return dict(weights)


def minimum_backward_edges(weights: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Return an exact minimum-weight layer order and the edges it makes backward.

    Dynamic programming over subsets: the order is built from the bottom, and an edge
    ``a → b`` is backward when ``a`` sits below ``b`` (a lower layer depending on a higher
    one). Ties are broken by domain name, so the result is deterministic.
    """
    domains = sorted(set(weights) | {b for row in weights.values() for b in row})
    count = len(domains)
    cost = [[weights.get(a, {}).get(b, 0) for b in domains] for a in domains]
    unreachable = 1 << 62
    best = [unreachable] * (1 << count)
    choice = [-1] * (1 << count)
    best[0] = 0
    for subset in range(1 << count):
        if best[subset] == unreachable:
            continue
        for candidate in range(count):
            if subset >> candidate & 1:
                continue
            added = sum(cost[u][candidate] for u in range(count) if subset >> u & 1)
            grown = subset | 1 << candidate
            if best[subset] + added < best[grown]:
                best[grown] = best[subset] + added
                choice[grown] = candidate
    order: list[str] = []
    subset = (1 << count) - 1
    while subset:
        candidate = choice[subset]
        order.append(domains[candidate])
        subset &= ~(1 << candidate)
    order.reverse()
    position = {name: index for index, name in enumerate(order)}
    ranked = sorted(
        (
            (-w, a, b)
            for a, row in weights.items()
            for b, w in row.items()
            if position[a] < position[b]
        ),
    )
    backward = [{"from": a, "to": b, "weight": -w} for w, a, b in ranked]
    return {
        "layer_order_bottom_to_top": order,
        "backward_weight": best[(1 << count) - 1] if count else 0,
        "total_weight": sum(w for row in weights.values() for w in row.values()),
        "backward_edges": backward,
    }


def extras_map(repo: Path, domain_map: DomainMap, inventory: Inventory) -> dict[str, Any]:
    """Map each optional extra to its import names and every file that references it.

    A reference is a third-party import in a tracked Python file, an import line in a tracked
    notebook, or the extra's name inside an install specifier (``[extra]``) in any tracked
    text file. "No importer in ``src``" is reported separately and is never proof of death.
    """
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    extras: dict[str, list[str]] = pyproject["project"].get("optional-dependencies", {})
    notebook_imports = _notebook_imports(repo, inventory)
    specifier_refs = _extra_specifier_refs(repo, inventory, set(extras))
    result: dict[str, Any] = {}
    for extra, requirements in sorted(extras.items()):
        dists = sorted({_dist_name(r) for r in requirements if not r.startswith(PACKAGE)})
        names: set[str] = set()
        unmapped: list[str] = []
        for dist in dists:
            if dist == "scpn-quantum-control":
                continue
            mapped = domain_map.extra_import_names.get(dist)
            if mapped is None:
                unmapped.append(dist)
            else:
                names.update(mapped)
        files: set[str] = set()
        for name in names:
            files.update(inventory.external_imports.get(name, set()))
            files.update(notebook_imports.get(name, set()))
        by_area: dict[str, int] = defaultdict(int)
        for path in files:
            by_area[path.split("/", 1)[0] if "/" in path else "(root)"] += 1
        result[extra] = {
            "distributions": dists,
            "import_names": sorted(names),
            "unmapped_distributions": unmapped,
            "referencing_files_by_area": dict(sorted(by_area.items())),
            "src_importers": len([p for p in files if p.startswith("src/")]),
            "install_specifier_refs": sorted(specifier_refs.get(extra, set())),
        }
    return result


def _dist_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
    return match.group(0).lower() if match else requirement


def _notebook_imports(repo: Path, inventory: Inventory) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    for record in inventory.files:
        if not record.path.endswith(".ipynb"):
            continue
        try:
            notebook = json.loads((repo / record.path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            text = "".join(cell.get("source", []))
            for match in pattern.finditer(text):
                found[match.group(1)].add(record.path)
    return found


def _extra_specifier_refs(
    repo: Path, inventory: Inventory, extras: set[str]
) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r"\[([A-Za-z0-9_,\- ]+)\]")
    text_suffixes = (".md", ".yml", ".yaml", ".toml", ".txt", ".sh", ".py", ".cfg", ".ipynb")
    for record in inventory.files:
        if not record.path.endswith(text_suffixes) or record.path == "pyproject.toml":
            continue
        text = _read_text(repo / record.path)
        if text is None:
            continue
        for match in pattern.finditer(text):
            for name in match.group(1).replace(" ", "").split(","):
                if name in extras:
                    found[name].add(record.path)
    return found


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > _TEXT_SCAN_LIMIT:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def digest_bound_paths(repo: Path, inventory: Inventory) -> dict[str, Any]:
    """List data and generated files that name package source paths.

    Such files (evidence manifests, coverage registers, generated inventories) break when a
    source file moves; QSP cards regenerate them through their producers. Files too large or
    not UTF-8 are listed as skipped with the reason, never silently treated as clean.
    """
    scanned_areas = ("data/", "docs/_generated/", "results/", "tools/")
    bound: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for record in inventory.files:
        if not record.path.startswith(scanned_areas) or record.path.endswith(".py"):
            continue
        text = _read_text(repo / record.path)
        if text is None:
            skipped[record.path] = "larger than scan limit or not UTF-8 text"
            continue
        names = sorted(set(_SOURCE_PATH_REF.findall(text)))
        if names:
            bound[record.path] = names
    return {
        "scanned_areas": list(scanned_areas),
        "files_naming_source_paths": bound,
        "skipped": dict(sorted(skipped.items())),
    }


def measure_import_cost(repo: Path) -> dict[str, Any]:
    """Import the package in a fresh interpreter and count the package modules it loads."""
    script = (
        "import json, sys\n"
        f"import {PACKAGE}\n"
        f"print(json.dumps(sorted(m for m in sys.modules if m == '{PACKAGE}' "
        f"or m.startswith('{PACKAGE}.'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"status": "failed", "stderr_tail": completed.stderr[-2000:]}
    modules = json.loads(completed.stdout.strip().splitlines()[-1])
    return {"status": "measured", "python": sys.version.split()[0], "loaded_modules": len(modules)}


def _json(data: Any) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_outputs(
    repo: Path,
    domain_map: DomainMap,
    inventory: Inventory,
    out_dir: Path,
    import_cost: dict[str, Any] | None,
) -> list[Path]:
    """Write the deterministic inventory outputs into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["path", "sha256", "kind", "unit", "domain", "target", "reason"])
    for record in inventory.files:
        writer.writerow(
            [
                record.path,
                record.sha256,
                record.kind,
                record.unit,
                record.domain,
                record.target,
                record.reason,
            ]
        )
    written.append(_write(out_dir / "ownership.csv", buffer.getvalue()))

    edges = [
        {
            "source": e.source_path,
            "target_unit": e.target_unit,
            "kind": e.kind,
            "line": e.line,
            "target_module": e.target_module,
            "root_export": e.root_export,
            "import_names": list(e.import_names),
            "typecheck_context": e.typecheck_context,
            "literal_table": e.literal_table,
        }
        for e in inventory.edges
    ]
    written.append(_write(out_dir / "dependency_edges.json", _json(edges)))
    written.append(
        _write(
            out_dir / "dynamic_import_sites.json",
            _json([asdict(site) for site in inventory.dynamic_sites]),
        )
    )

    module_weights = domain_weights(inventory.edges, domain_map, MODULE_KINDS)
    runtime_weights = domain_weights(inventory.edges, domain_map, RUNTIME_KINDS)
    cycles = {
        "module_level": minimum_backward_edges(module_weights),
        "runtime": minimum_backward_edges(runtime_weights),
    }
    written.append(_write(out_dir / "cycle_minimum.json", _json(cycles)))
    written.append(
        _write(out_dir / "extras_map.json", _json(extras_map(repo, domain_map, inventory)))
    )
    written.append(
        _write(out_dir / "digest_bound_paths.json", _json(digest_bound_paths(repo, inventory)))
    )
    if import_cost is not None:
        written.append(_write(out_dir / "import_cost.json", _json(import_cost)))

    target_units: dict[str, list[str]] = defaultdict(list)
    for unit, domain in domain_map.unit_domain.items():
        target_units[domain_map.domain_target[domain]].append(unit)
    kinds: dict[str, int] = defaultdict(int)
    source_files: dict[str, int] = defaultdict(int)
    for record in inventory.files:
        kinds[record.kind] += 1
        if record.path.startswith(SOURCE_PREFIX):
            source_files[record.target] += 1
    summary = {
        "schema": "scpn_qc_split_ownership_summary_v1",
        "head": inventory.head,
        "tracked_paths": len(inventory.files),
        "paths_by_kind": dict(sorted(kinds.items())),
        "source_units": len(domain_map.unit_domain),
        "edges": len(inventory.edges),
        "units_per_target": {t: len(target_units.get(t, [])) for t in sorted(domain_map.targets)},
        "source_files_per_target": {t: source_files.get(t, 0) for t in sorted(domain_map.targets)},
        "empty_targets": domain_map.empty_targets,
        "problems": inventory.problems,
    }
    written.append(_write(out_dir / "summary.json", _json(summary)))

    lines = ["# Stage-0 classifications (owner review)", ""]
    lines += [
        f"- `{unit}` — {reason} **Decision:** {decision}"
        for unit, reason, decision in domain_map.open_classifications
    ]
    written.append(_write(out_dir / "open_classifications.md", "\n".join(lines) + "\n"))
    return written


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inventory; exit 0 when complete, 1 when problems were found, 2 on a bad map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--map", type=Path, default=None, help="domain map (default in repo)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--measure-import-cost",
        action="store_true",
        help="also import the package in a fresh interpreter and count loaded modules",
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        domain_map = load_domain_map(args.map or repo / DEFAULT_MAP)
    except OwnershipError as exc:
        print(f"domain map error: {exc}", file=sys.stderr)
        return 2
    inventory = build_inventory(repo, domain_map)
    import_cost = measure_import_cost(repo) if args.measure_import_cost else None
    write_outputs(repo, domain_map, inventory, args.out, import_cost)
    for problem in inventory.problems:
        print(problem, file=sys.stderr)
    print(
        f"{len(inventory.files)} paths, {len(inventory.edges)} edges, "
        f"{len(inventory.problems)} problems -> {args.out}"
    )
    return 1 if inventory.problems else 0


# Entry guard only; the CLI behaviour is exercised by the subprocess tests in
# tests/test_audit_split_ownership.py (tools/ is outside the coverage source).
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
