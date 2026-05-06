# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — full-suite ordering-state audit helper
"""Inventory likely test-ordering and hidden-global-state hazards."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

MONKEYPATCH_STATE_METHODS = {
    "delattr",
    "delenv",
    "delitem",
    "setattr",
    "setenv",
    "setitem",
    "syspath_prepend",
}

GLOBAL_STATE_TERMS = (
    "_available",
    "_backend",
    "_cache",
    "_client",
    "_engine",
    "_has_",
    "_instance",
    "_registry",
    "_service",
    "backend",
    "cache",
    "registry",
)


@dataclass(frozen=True)
class OrderingFinding:
    """One potential test-ordering hazard."""

    path: str
    line: int
    category: str
    name: str
    context: str


def _call_name(node: ast.AST) -> str:
    """Return the final dotted name component for a call expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _full_attr_name(node: ast.AST) -> str:
    """Return a dotted attribute/name expression when statically available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _full_attr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _source_segment(source: str, node: ast.AST) -> str:
    """Return compact source text for a node."""
    segment = ast.get_source_segment(source, node) or ""
    return " ".join(segment.split())


def _literal_text(node: ast.AST) -> str:
    """Return string literal content when available."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _monkeypatch_category(node: ast.Call, context: str) -> str:
    """Classify one monkeypatch call."""
    name = _call_name(node.func)
    lowered = context.lower()
    first_arg = _literal_text(node.args[0]).lower() if node.args else ""
    if "sys.modules" in lowered:
        return "module_injection"
    if name in {"setenv", "delenv"}:
        return "environment_mutation"
    if name == "syspath_prepend":
        return "import_path_mutation"
    if first_arg.startswith("scpn_"):
        return "environment_mutation"
    if any(term in lowered for term in GLOBAL_STATE_TERMS):
        return "global_state_mutation"
    return "monkeypatch_state"


def _assignment_category(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> str:
    """Classify assignment targets that mutate imported module state."""
    targets: list[ast.AST]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        targets = [node.target]
    names = [_full_attr_name(target).lower() for target in targets]
    if any(name.startswith("sys.modules") for name in names):
        return "module_injection"
    if any(any(term in name for term in GLOBAL_STATE_TERMS) for name in names):
        return "global_state_assignment"
    return ""


def audit_ordering_file(path: Path) -> tuple[OrderingFinding, ...]:
    """Audit one test module for likely ordering hazards."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[OrderingFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            full_name = _full_attr_name(node.func)
            call_name = _call_name(node.func)
            context = _source_segment(source, node)
            if full_name == "importlib.reload" or call_name == "reload":
                findings.append(
                    OrderingFinding(str(path), node.lineno, "module_reload", full_name, context)
                )
            elif full_name.startswith("monkeypatch.") and call_name in MONKEYPATCH_STATE_METHODS:
                findings.append(
                    OrderingFinding(
                        str(path),
                        node.lineno,
                        _monkeypatch_category(node, context),
                        full_name,
                        context,
                    )
                )
            elif full_name in {"np.random.seed", "numpy.random.seed", "random.seed"}:
                findings.append(
                    OrderingFinding(
                        str(path), node.lineno, "random_seed_mutation", full_name, context
                    )
                )
            elif full_name.endswith(".clear") and any(
                term in context.lower() for term in ("backend", "cache", "registry")
            ):
                findings.append(
                    OrderingFinding(
                        str(path), node.lineno, "global_clear_call", full_name, context
                    )
                )
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            category = _assignment_category(node)
            if category:
                findings.append(
                    OrderingFinding(
                        str(path),
                        node.lineno,
                        category,
                        "assignment",
                        _source_segment(source, node),
                    )
                )
    return tuple(sorted(findings, key=lambda item: (item.path, item.line, item.category)))


def audit_ordering_tree(root: Path) -> tuple[OrderingFinding, ...]:
    """Audit all top-level pytest modules under a test root."""
    findings: list[OrderingFinding] = []
    for path in sorted(root.glob("test_*.py")):
        if path.is_file():
            findings.extend(audit_ordering_file(path))
    return tuple(findings)


def _finding_to_dict(finding: OrderingFinding) -> dict[str, object]:
    """Convert one finding to JSON-compatible data."""
    return {
        "path": finding.path,
        "line": finding.line,
        "category": finding.category,
        "name": finding.name,
        "context": finding.context,
    }


def findings_to_json(findings: Sequence[OrderingFinding]) -> str:
    """Serialise ordering findings as deterministic JSON."""
    return json.dumps(
        [_finding_to_dict(item) for item in findings],
        indent=2,
        sort_keys=True,
    )


def category_counts(findings: Iterable[OrderingFinding]) -> dict[str, int]:
    """Return deterministic category counts."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
    return dict(sorted(counts.items()))


def format_findings(findings: Iterable[OrderingFinding]) -> str:
    """Render a compact ordering-state audit summary."""
    items = tuple(findings)
    lines = [
        "Full-suite ordering-state audit summary:",
        f"- findings: {len(items)}",
    ]
    for category, count in category_counts(items).items():
        lines.append(f"- {category}: {count}")
    for item in items[:30]:
        lines.append(f"- {item.path}:{item.line} {item.category} {item.name}")
    if len(items) > 30:
        lines.append(f"- additional_findings: {len(items) - 30}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests",
        help="Directory containing pytest modules.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--fail-on-module-reload",
        action="store_true",
        help="Return non-zero when module reloads are found.",
    )
    args = parser.parse_args(argv)

    findings = audit_ordering_tree(args.tests_root)
    print(findings_to_json(findings) if args.json else format_findings(findings))
    has_reload = any(item.category == "module_reload" for item in findings)
    return 1 if args.fail_on_module_reload and has_reload else 0


if __name__ == "__main__":
    raise SystemExit(main())
