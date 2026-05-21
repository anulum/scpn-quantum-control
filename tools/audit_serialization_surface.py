# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Serialization attack-surface audit helper
"""AST audit for unsafe deserialisation surfaces."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

UNSAFE_CALLS = {
    "pickle.load",
    "pickle.loads",
    "dill.load",
    "dill.loads",
    "joblib.load",
    "cloudpickle.load",
    "cloudpickle.loads",
    "marshal.load",
    "marshal.loads",
    "shelve.open",
    "qiskit.qpy.load",
    "qpy.load",
}
APPROVED_WRAPPER_FUNCTIONS = {
    ("src/scpn_quantum_control/hardware/hal_qiskit.py", "_reviewed_qpy_load_circuits"): {
        "qiskit.qpy.load",
        "qpy.load",
    },
}
DEFAULT_ROOTS = ("src", "scripts", "tools", "tests")
EXCLUDED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}


@dataclass(frozen=True)
class SerializationFinding:
    """One unsafe serialisation-surface finding."""

    path: str
    line: int
    column: int
    symbol: str
    reason: str


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path.as_posix()
        self.findings: list[tuple[int, int, str, str]] = []
        self.import_aliases: dict[str, str] = {}
        self._function_stack: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Track import aliases introduced by ``import`` statements."""
        for alias in node.names:
            self.import_aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track import aliases introduced by ``from`` imports."""
        if node.module is None:
            return
        for alias in node.names:
            full_name = f"{node.module}.{alias.name}"
            self.import_aliases[alias.asname or alias.name] = full_name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Record calls to unsafe serialization APIs."""
        symbol = _call_name(node.func, self.import_aliases)
        if symbol in UNSAFE_CALLS and not self._is_approved_wrapper_call(symbol):
            self.findings.append(
                (
                    node.lineno,
                    node.col_offset,
                    symbol,
                    "Unsafe deserialisation API requires an explicit reviewed wrapper.",
                )
            )
        if symbol in {"numpy.load", "np.load"} and _has_allow_pickle_true(node):
            self.findings.append(
                (
                    node.lineno,
                    node.col_offset,
                    symbol,
                    "np.load(..., allow_pickle=True) can execute object deserialisation payloads.",
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track the enclosing function for reviewed wrapper allow-listing."""
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def _is_approved_wrapper_call(self, symbol: str) -> bool:
        if not self._function_stack:
            return False
        approved = APPROVED_WRAPPER_FUNCTIONS.get((self.path, self._function_stack[-1]))
        return approved is not None and symbol in approved


def _call_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value, aliases)
        full_name = f"{parent}.{node.attr}" if parent else node.attr
        parts = full_name.split(".")
        if parts:
            parts[0] = aliases.get(parts[0], parts[0])
        return ".".join(parts)
    return ""


def _has_allow_pickle_true(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg != "allow_pickle":
            continue
        return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return False


def _normalise(path: Path) -> str:
    return path.as_posix()


def candidate_files(project_root: Path, roots: Sequence[str] = DEFAULT_ROOTS) -> tuple[Path, ...]:
    """Return deterministic Python files under audited roots."""
    files: list[Path] = []
    for root in roots:
        base = project_root / root
        if base.is_file() and base.suffix == ".py":
            files.append(base.relative_to(project_root))
            continue
        if not base.exists():
            continue
        files.extend(
            path.relative_to(project_root)
            for path in base.rglob("*.py")
            if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts)
        )
    return tuple(sorted(set(files), key=_normalise))


def scan_text(path: Path, text: str) -> tuple[SerializationFinding, ...]:
    """Scan one Python source string."""
    try:
        tree = ast.parse(text, filename=_normalise(path))
    except SyntaxError as exc:
        return (
            SerializationFinding(
                path=_normalise(path),
                line=exc.lineno or 0,
                column=exc.offset or 0,
                symbol="syntax_error",
                reason=str(exc),
            ),
        )
    visitor = _Visitor(path)
    visitor.visit(tree)
    return tuple(
        SerializationFinding(
            path=_normalise(path),
            line=line,
            column=column,
            symbol=symbol,
            reason=reason,
        )
        for line, column, symbol, reason in visitor.findings
    )


def scan_files(project_root: Path, files: Iterable[Path]) -> tuple[SerializationFinding, ...]:
    """Scan repository-relative files."""
    findings: list[SerializationFinding] = []
    for relative in files:
        text = (project_root / relative).read_text(encoding="utf-8")
        findings.extend(scan_text(relative, text))
    return tuple(
        sorted(findings, key=lambda item: (item.path, item.line, item.column, item.symbol))
    )


def findings_to_json(findings: Sequence[SerializationFinding]) -> str:
    """Serialise findings as deterministic JSON."""
    return json.dumps(
        [
            {
                "column": item.column,
                "line": item.line,
                "path": item.path,
                "reason": item.reason,
                "symbol": item.symbol,
            }
            for item in findings
        ],
        indent=2,
        sort_keys=True,
    )


def format_findings(findings: Sequence[SerializationFinding]) -> str:
    """Render a compact text report."""
    if not findings:
        return "Serialization surface audit: no unsafe deserialisation calls found"
    lines = ["Serialization surface audit findings:"]
    for item in findings:
        lines.append(f"- {item.path}:{item.line}:{item.column} {item.symbol}: {item.reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input", type=Path, help="Scan one Python file instead of default roots."
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("roots", nargs="*", help="Repository-relative roots to scan.")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    if args.input is not None:
        input_path = args.input.resolve()
        relative = (
            input_path.relative_to(project_root)
            if input_path.is_relative_to(project_root)
            else input_path
        )
        findings = scan_text(relative, input_path.read_text(encoding="utf-8"))
    else:
        findings = scan_files(
            project_root, candidate_files(project_root, tuple(args.roots) or DEFAULT_ROOTS)
        )

    print(findings_to_json(findings) if args.json else format_findings(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
