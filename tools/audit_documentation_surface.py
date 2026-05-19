# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- documentation surface audit helper
"""Inventory repository documentation coverage and stale documentation markers."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PYTHON_ROOTS = ("src", "scripts", "tools")
DEFAULT_MARKDOWN_ROOTS = ("README.md", "CHANGELOG.md", "docs")
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-linux",
    "__pycache__",
    "site",
}
STALE_STATUS_RE = re.compile(r"Status Snapshot --? (20\d{2}-\d{2}-\d{2})")
TITLE_RE = re.compile(r"^#\s+\S+", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class DocumentationFinding:
    """One documentation-surface audit finding."""

    path: str
    line: int
    kind: str
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class DocumentationAllowlistEntry:
    """One explicitly accepted documentation-audit finding pattern."""

    path_pattern: str
    kind: str
    symbol: str
    reason: str


def _normalise(path: Path) -> str:
    return path.as_posix()


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def candidate_python_files(
    project_root: Path,
    roots: Sequence[str] = DEFAULT_PYTHON_ROOTS,
) -> tuple[Path, ...]:
    """Return deterministic repository-relative Python files to audit."""
    files: list[Path] = []
    for root in roots:
        base = project_root / root
        if base.is_file() and base.suffix == ".py" and not _is_excluded(base):
            files.append(base.relative_to(project_root))
            continue
        if not base.exists():
            continue
        files.extend(
            path.relative_to(project_root)
            for path in base.rglob("*.py")
            if path.is_file() and not _is_excluded(path)
        )
    return tuple(sorted(set(files), key=_normalise))


def candidate_markdown_files(
    project_root: Path,
    roots: Sequence[str] = DEFAULT_MARKDOWN_ROOTS,
) -> tuple[Path, ...]:
    """Return deterministic repository-relative Markdown files to audit."""
    files: list[Path] = []
    for root in roots:
        base = project_root / root
        if base.is_file() and base.suffix.lower() in {".md", ".rst"}:
            files.append(base.relative_to(project_root))
            continue
        if not base.exists():
            continue
        files.extend(
            path.relative_to(project_root)
            for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".rst"} and not _is_excluded(path)
        )
    return tuple(sorted(set(files), key=_normalise))


def _qualified_name(stack: Sequence[tuple[str, str]], name: str) -> str:
    names = tuple(item_name for _, item_name in stack)
    return ".".join((*names, name)) if names else name


def _is_public_callable_scope(stack: Sequence[tuple[str, str]]) -> bool:
    if not stack:
        return True
    return stack[-1][0] == "class" and all(kind == "class" for kind, _ in stack)


class _DocstringVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[DocumentationFinding] = []
        self.stack: list[tuple[str, str]] = []

    def visit_Module(self, node: ast.Module) -> None:
        if ast.get_docstring(node) is None:
            self.findings.append(
                DocumentationFinding(
                    path=_normalise(self.path),
                    line=1,
                    kind="module",
                    symbol=_normalise(self.path),
                    reason="module is missing a docstring",
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if _is_public_name(node.name) and ast.get_docstring(node) is None:
            self.findings.append(
                DocumentationFinding(
                    path=_normalise(self.path),
                    line=node.lineno,
                    kind="class",
                    symbol=_qualified_name(self.stack, node.name),
                    reason="public class is missing a docstring",
                )
            )
        self.stack.append(("class", node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if (
            _is_public_name(node.name)
            and _is_public_callable_scope(self.stack)
            and ast.get_docstring(node) is None
        ):
            self.findings.append(
                DocumentationFinding(
                    path=_normalise(self.path),
                    line=node.lineno,
                    kind="function",
                    symbol=_qualified_name(self.stack, node.name),
                    reason="public function or method is missing a docstring",
                )
            )
        self.stack.append(("function", node.name))
        self.generic_visit(node)
        self.stack.pop()


def audit_python_text(path: Path, text: str) -> tuple[DocumentationFinding, ...]:
    """Audit one Python source string for public docstring coverage."""
    try:
        tree = ast.parse(text, filename=_normalise(path))
    except SyntaxError as exc:
        return (
            DocumentationFinding(
                path=_normalise(path),
                line=exc.lineno or 0,
                kind="syntax_error",
                symbol=_normalise(path),
                reason=str(exc),
            ),
        )
    visitor = _DocstringVisitor(path)
    visitor.visit(tree)
    return tuple(visitor.findings)


def audit_markdown_text(
    path: Path, text: str, *, current_date: str
) -> tuple[DocumentationFinding, ...]:
    """Audit one Markdown/RST source string for title and stale status markers."""
    findings: list[DocumentationFinding] = []
    if path.suffix.lower() == ".md" and not TITLE_RE.search(text):
        findings.append(
            DocumentationFinding(
                path=_normalise(path),
                line=1,
                kind="markdown_title",
                symbol=_normalise(path),
                reason="Markdown page is missing an H1 title",
            )
        )
    for match in STALE_STATUS_RE.finditer(text):
        status_date = match.group(1)
        if status_date != current_date:
            line = text.count("\n", 0, match.start()) + 1
            findings.append(
                DocumentationFinding(
                    path=_normalise(path),
                    line=line,
                    kind="stale_status_snapshot",
                    symbol=status_date,
                    reason=f"status snapshot date differs from {current_date}",
                )
            )
    return tuple(findings)


def audit_files(
    project_root: Path,
    *,
    python_files: Iterable[Path],
    markdown_files: Iterable[Path],
    current_date: str,
) -> tuple[DocumentationFinding, ...]:
    """Audit selected repository-relative files."""
    findings: list[DocumentationFinding] = []
    for relative in python_files:
        findings.extend(
            audit_python_text(relative, (project_root / relative).read_text(encoding="utf-8"))
        )
    for relative in markdown_files:
        findings.extend(
            audit_markdown_text(
                relative,
                (project_root / relative).read_text(encoding="utf-8"),
                current_date=current_date,
            )
        )
    return tuple(sorted(findings, key=lambda item: (item.path, item.line, item.kind, item.symbol)))


def load_allowlist(path: Path) -> tuple[DocumentationAllowlistEntry, ...]:
    """Load and validate documentation-audit allow-list entries."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("entries")
    if not isinstance(raw, list):
        raise ValueError("documentation allowlist must be a JSON list or object with entries")
    entries: list[DocumentationAllowlistEntry] = []
    required = ("path_pattern", "kind", "symbol", "reason")
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"allowlist entry {index} must be an object")
        values: dict[str, str] = {}
        for key in required:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"allowlist entry {index} requires non-empty {key}")
            values[key] = value.strip()
        entries.append(DocumentationAllowlistEntry(**values))
    return tuple(entries)


def _is_allowed(
    finding: DocumentationFinding,
    allowlist: Sequence[DocumentationAllowlistEntry],
) -> bool:
    return any(
        fnmatch.fnmatchcase(finding.path, entry.path_pattern)
        and finding.kind == entry.kind
        and finding.symbol == entry.symbol
        for entry in allowlist
    )


def filter_allowed_findings(
    findings: Sequence[DocumentationFinding],
    allowlist: Sequence[DocumentationAllowlistEntry],
) -> tuple[DocumentationFinding, ...]:
    """Remove findings that match an explicit allow-list entry."""
    return tuple(finding for finding in findings if not _is_allowed(finding, allowlist))


def findings_to_json(findings: Sequence[DocumentationFinding]) -> str:
    """Serialise findings as deterministic JSON."""
    return json.dumps(
        [
            {
                "kind": item.kind,
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


def format_findings(findings: Sequence[DocumentationFinding], *, limit: int = 40) -> str:
    """Render a compact documentation audit report."""
    if not findings:
        return "Documentation surface audit: no findings"
    by_kind = Counter(item.kind for item in findings)
    by_root = Counter(
        Path(item.path).parts[0] if Path(item.path).parts else item.path for item in findings
    )
    lines = [
        "Documentation surface audit findings:",
        f"- total_findings: {len(findings)}",
        "- by_kind: " + ", ".join(f"{key}={by_kind[key]}" for key in sorted(by_kind)),
        "- by_root: " + ", ".join(f"{key}={by_root[key]}" for key in sorted(by_root)),
        "",
        "## First findings",
    ]
    for item in findings[:limit]:
        lines.append(f"- {item.path}:{item.line} {item.kind} {item.symbol}: {item.reason}")
    if len(findings) > limit:
        lines.append(f"- additional_findings: {len(findings) - limit}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--current-date", default="2026-05-18")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--allowlist", type=Path, action="append", default=[])
    parser.add_argument("--python-root", action="append", dest="python_roots")
    parser.add_argument("--markdown-root", action="append", dest="markdown_roots")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    python_files = candidate_python_files(
        project_root,
        tuple(args.python_roots) if args.python_roots else DEFAULT_PYTHON_ROOTS,
    )
    markdown_files = candidate_markdown_files(
        project_root,
        tuple(args.markdown_roots) if args.markdown_roots else DEFAULT_MARKDOWN_ROOTS,
    )
    findings = audit_files(
        project_root,
        python_files=python_files,
        markdown_files=markdown_files,
        current_date=args.current_date,
    )
    allowlist = tuple(
        entry for allowlist_path in args.allowlist for entry in load_allowlist(allowlist_path)
    )
    if allowlist:
        findings = filter_allowed_findings(findings, allowlist)
    print(findings_to_json(findings) if args.json else format_findings(findings, limit=args.limit))
    return 1 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
