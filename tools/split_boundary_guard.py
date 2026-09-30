# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Split dependency boundary guard
"""Reject new, widened or stale exceptions to the declared dependency DAG.

Package targets, rather than a minimum feedback-arc order, define direction.
Both AD domains belong to the same target. Split-unit files are resolved using
full import candidates retained by the ownership scanner. Tests and evidence
consumers are reported separately and never become runtime exceptions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_split_ownership import (
    DEFAULT_MAP,
    PACKAGE,
    SOURCE_PREFIX,
    UMBRELLA,
    DomainMap,
    Inventory,
    build_inventory,
    load_domain_map,
)

DEFAULT_BASELINE = Path("data/split_preparation/boundary_baseline.json")
BLOCKING_KINDS = frozenset({"module", "module_try", "lazy", "dynamic", "string_ref"})
REMOVAL_OWNERS = frozenset(
    (
        "lazy-subpackage-exports",
        "generic-hamiltonian-separation",
        "shared-contract-nucleus",
        "differentiable-contract-inversion",
        "hardware-isolation",
        "domain-public-api-freeze",
        "optional-dependency-installation",
        "domain-test-ci-partition",
        "repository-extraction-rollback",
        "residual-dependency-inversions",
    )
)
BoundaryKey = tuple[str, str, str]


@dataclass(frozen=True)
class ExceptionRow:
    """One counted dependency exception with its removal owner."""

    source: str
    target: str
    kind: str
    count: int
    reason: str
    removal_owner: str

    @property
    def key(self) -> BoundaryKey:
        """Line-independent identity that the ratchet counts."""
        return self.source, self.target, self.kind


@dataclass(frozen=True)
class BoundaryPolicy:
    """Validated acyclic target graph and existing dependency exceptions."""

    dependencies: dict[str, frozenset[str]]
    exceptions: tuple[ExceptionRow, ...]
    non_module_references: tuple[ExceptionRow, ...] = ()


@dataclass
class BoundaryReport:
    """Source findings and the separate nonblocking consumer inventory."""

    problems: list[str] = field(default_factory=list)
    violations: Counter[BoundaryKey] = field(default_factory=Counter)
    locations: dict[BoundaryKey, list[int]] = field(default_factory=dict)
    typecheck_edges: int = 0
    consumer_edges: int = 0
    non_module_references: Counter[BoundaryKey] = field(default_factory=Counter)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object with string keys")
    return cast(dict[str, object], value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty string")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate policy key: {key}")
        result[key] = value
    return result


def load_policy(path: Path, domain_map: DomainMap) -> BoundaryPolicy:
    """Read a baseline and reject malformed exceptions or an invalid target DAG.

    Parameters
    ----------
    path
        JSON baseline using ``scpn_qc_boundary_baseline_v2``.
    domain_map
        Current ownership assignments against which all targets are checked.

    Returns
    -------
    BoundaryPolicy
        Graph with its transitive dependency closure and counted exceptions.

    Raises
    ------
    ValueError
        If schema, targets, graph direction or exception records are invalid.

    """
    raw = _mapping(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
    if raw.get("schema") != "scpn_qc_boundary_baseline_v2":
        raise ValueError("unsupported boundary baseline schema")
    graph = _mapping(raw.get("dependencies"))
    targets = frozenset(domain_map.domain_target.values()) - {UMBRELLA}
    if set(graph) != targets:
        raise ValueError("dependency graph must name exactly the current split targets")
    closure: dict[str, set[str]] = {}
    for source, value in graph.items():
        if not isinstance(value, list):
            raise ValueError(f"dependencies of {source} must be a list")
        members = [_text(item) for item in value]
        if len(members) != len(set(members)) or not set(members) <= targets:
            raise ValueError(f"duplicate or unknown dependency of {source}")
        closure[source] = set(members)
    for _ in targets:
        for source in sorted(targets):
            closure[source].update(
                target for member in tuple(closure[source]) for target in closure[member]
            )
    if any(source in members for source, members in closure.items()):
        raise ValueError("dependency graph must be acyclic")
    values = raw.get("exceptions")
    if not isinstance(values, list):
        raise ValueError("exceptions must be a list")
    rows: list[ExceptionRow] = []
    seen: set[BoundaryKey] = set()
    for value in values:
        row = _mapping(value)
        count = row.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("exception count must be a positive integer")
        item = ExceptionRow(
            _text(row.get("source")),
            _text(row.get("target")),
            _text(row.get("kind")),
            count,
            _text(row.get("reason")),
            _text(row.get("removal_owner")),
        )
        if (
            not item.source.startswith(SOURCE_PREFIX)
            or not item.source.endswith(".py")
            or ".." in Path(item.source).parts
            or not item.target.startswith(SOURCE_PREFIX)
            or not item.target.endswith(".py")
            or ".." in Path(item.target).parts
            or item.kind not in BLOCKING_KINDS
            or item.removal_owner not in REMOVAL_OWNERS
            or item.key in seen
        ):
            raise ValueError(f"invalid or duplicate exception: {item.key}")
        seen.add(item.key)
        rows.append(item)
    references = raw.get("non_module_references", [])
    if not isinstance(references, list):
        raise ValueError("non-module references must be a list")
    non_modules: list[ExceptionRow] = []
    for value in references:
        row = _mapping(value)
        count = row.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("non-module reference count must be a positive integer")
        item = ExceptionRow(
            _text(row.get("source")),
            _text(row.get("target_module")),
            "string_ref",
            count,
            _text(row.get("reason")),
            "not-a-dependency",
        )
        if (
            not item.source.startswith(SOURCE_PREFIX)
            or ".." in Path(item.source).parts
            or not item.target.startswith(PACKAGE + ".")
            or item.key in seen
        ):
            raise ValueError(f"invalid or duplicate non-module reference: {item.key}")
        seen.add(item.key)
        non_modules.append(item)
    return BoundaryPolicy(
        {target: frozenset(members) for target, members in closure.items()},
        tuple(rows),
        tuple(non_modules),
    )


def inspect_boundaries(
    inventory: Inventory, domain_map: DomainMap, policy: BoundaryPolicy
) -> BoundaryReport:
    """Resolve source-file owners and count edges forbidden by the target graph.

    Parameters
    ----------
    inventory
        Output from the existing ownership scanner, including full import candidates.
    domain_map
        Owner map with split-file decisions.
    policy
        Validated dependency closure; exceptions are evaluated by ``check_boundaries``.

    Returns
    -------
    BoundaryReport
        Findings, violation counts, source line locations and consumer totals.

    """
    report = BoundaryReport(problems=list(inventory.problems))
    records = {
        record.path: record for record in inventory.files if record.path.startswith(SOURCE_PREFIX)
    }
    modules = {
        path[len("src/") :].removesuffix(".py").removesuffix("/__init__").replace("/", "."): path
        for path in records
    }
    non_modules = {row.key for row in policy.non_module_references}
    for edge in inventory.edges:
        if not edge.source_path.startswith(SOURCE_PREFIX):
            report.consumer_edges += 1
            continue
        if edge.kind == "typecheck" or edge.typecheck_context:
            report.typecheck_edges += 1
            continue
        if edge.kind not in BLOCKING_KINDS:
            report.problems.append(
                f"unknown edge kind: {edge.source_path}:{edge.line}: {edge.kind}"
            )
            continue
        candidates = (
            [
                edge.target_module if name == "*" else edge.target_module + "." + name
                for name in edge.import_names
            ]
            if edge.import_names
            else [edge.target_module]
        )
        for imported_module in dict.fromkeys(candidates):
            candidate = imported_module
            target_path = modules.get(candidate)
            while target_path is None and candidate.count(".") > 1:
                candidate = candidate.rsplit(".", 1)[0]
                target_path = modules.get(candidate)
            if target_path is None and edge.root_export:
                target_path = modules.get(PACKAGE)
            if target_path is None:
                key = (edge.source_path, edge.target_module, edge.kind)
                if key in non_modules:
                    report.non_module_references[key] += 1
                    continue
                report.problems.append(
                    f"unknown import target: {edge.source_path}:{edge.line}: {edge.target_module}"
                )
                continue
            source_record = records[edge.source_path]
            target_record = records[target_path]
            source = source_record.target
            target = target_record.target
            if source in (UMBRELLA, target):
                continue
            if (
                source not in policy.dependencies
                or target not in policy.dependencies
                and target != UMBRELLA
            ):
                report.problems.append(
                    f"unknown owner: {edge.source_path}:{edge.line}: {source} -> {target}"
                )
                continue
            if target in policy.dependencies[source]:
                continue
            key = (edge.source_path, target_path, edge.kind)
            report.violations[key] += 1
            report.locations.setdefault(key, []).append(edge.line)
    return report


def check_boundaries(report: BoundaryReport, policy: BoundaryPolicy) -> list[str]:
    """Return errors for new, increased, decreased or absent baseline exceptions.

    Parameters
    ----------
    report
        Current source scan, with counted forbidden dependencies.
    policy
        Existing exceptions, each with an explicit removal card.

    Returns
    -------
    list[str]
        All scan and ratchet failures. An empty list means the guard passes.

    """
    errors = list(report.problems)
    baseline = {row.key: row for row in policy.exceptions}
    for key, count in sorted(report.violations.items()):
        locations = ",".join(map(str, report.locations[key]))
        row = baseline.get(key)
        if row is None:
            errors.append(
                f"new backward edge: {key[0]}:{locations} -> {key[1]} ({key[2]}, count={count})"
            )
        elif count != row.count:
            errors.append(
                f"baseline count changed: {key[0]}:{locations} -> {key[1]} ({key[2]}): {row.count} -> {count}; prune decreases"
            )
    for key in sorted(baseline.keys() - report.violations.keys()):
        errors.append(f"stale baseline row: {key[0]} -> {key[1]} ({key[2]}); prune it")
    for row in policy.non_module_references:
        observed = report.non_module_references[row.key]
        if observed != row.count:
            errors.append(
                f"non-module reference count changed: {row.source} -> {row.target}: {row.count} -> {observed}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the source inventory and baseline ratchet with a failing shell status.

    Parameters
    ----------
    argv
        Optional CLI arguments; default is the process argument vector.

    Returns
    -------
    int
        Zero for an exact passing baseline, one for source or policy violations.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--domain-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)
    try:
        domain_map = load_domain_map(args.repo / args.domain_map)
        policy = load_policy(args.repo / args.baseline, domain_map)
        inventory = build_inventory(args.repo, domain_map)
        report = inspect_boundaries(inventory, domain_map, policy)
        errors = check_boundaries(report, policy)
    except (OSError, ValueError) as exc:
        print(f"boundary guard failed: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"Split boundary guard: {len(report.violations)} exception rows, {sum(report.violations.values())} occurrences; {report.typecheck_edges} type-checking and {report.consumer_edges} consumer edges reported separately; {len(errors)} problems"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
