# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — descriptive production naming audit
"""Reject internal work-item codes from production-facing names."""

from __future__ import annotations

import ast
import json
import re
import subprocess  # nosec B404
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_TASK_CODE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:bl|st|dp|fu|rg|hg)[_-]?\d"
)
_PATH_TASK_CODE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:^|[/_.-])(?:bl|st|dp|fu|rg|hg)[_-]?\d"
)
_MACHINE_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.:/-]+")
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PRODUCTION_ROOTS: Final[frozenset[str]] = frozenset(
    {"src", "tools", "scripts", "studio-web", "scpn_quantum_engine"}
)
_GENERIC_CODE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".jl",
        ".js",
        ".jsx",
        ".rs",
        ".sv",
        ".ts",
        ".tsx",
        ".v",
    }
)


@dataclass(frozen=True, order=True)
class NamingFinding:
    """One production-facing name that exposes an internal work-item code."""

    path: str
    line: int
    kind: str
    value: str

    def render(self) -> str:
        """Return a stable command-line representation."""
        return f"{self.path}:{self.line}: {self.kind}: {self.value}"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Return the identities of all Python docstring constants."""
    nodes: set[int] = set()
    for owner in ast.walk(tree):
        if not isinstance(
            owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not owner.body:
            continue
        first = owner.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            nodes.add(id(first.value))
    return nodes


def _python_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded Python identifiers, headings, and machine names."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=display_path)
    lines = text.splitlines()
    if len(lines) >= 7 and _TASK_CODE.search(lines[6]):
        yield NamingFinding(display_path, 7, "module heading", lines[6].strip())
    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc and _TASK_CODE.search(module_doc):
        first = tree.body[0]
        yield NamingFinding(
            display_path, first.lineno, "module description", module_doc.splitlines()[0]
        )
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = (node.name,)
        elif isinstance(node, ast.Name):
            names = (node.id,)
        elif isinstance(node, ast.arg):
            names = (node.arg,)
        elif isinstance(node, ast.Attribute):
            names = (node.attr,)
        for name in names:
            if _TASK_CODE.search(name):
                yield NamingFinding(
                    display_path,
                    getattr(node, "lineno", 0),
                    "Python identifier",
                    name,
                )
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and not node.value.startswith("docs/internal/")
            and _MACHINE_NAME.fullmatch(node.value)
            and _TASK_CODE.search(node.value)
        ):
            yield NamingFinding(display_path, node.lineno, "machine-facing string", node.value)


def _json_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded JSON keys and identifier-like values."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return

    def walk(value: object) -> Iterator[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                if _MACHINE_NAME.fullmatch(key) and _TASK_CODE.search(key):
                    yield key
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
        elif (
            isinstance(value, str) and _MACHINE_NAME.fullmatch(value) and _TASK_CODE.search(value)
        ):
            yield value

    for value in walk(payload):
        yield NamingFinding(display_path, 1, "JSON machine name", value)


def _workflow_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded workflow job IDs and step names."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("name:") or stripped.startswith("- name:"):
            if _TASK_CODE.search(stripped):
                yield NamingFinding(display_path, line_number, "workflow name", stripped)
        elif line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            job_id = stripped[:-1]
            if _TASK_CODE.search(job_id):
                yield NamingFinding(display_path, line_number, "workflow job ID", job_id)


def _documentation_headings(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield public documentation headings that expose task codes."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("#") and _TASK_CODE.search(line):
            yield NamingFinding(display_path, line_number, "documentation heading", line)


def _generic_code_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded identifiers from non-Python production sources."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for name in _IDENTIFIER.findall(line):
            if _TASK_CODE.search(name):
                yield NamingFinding(display_path, line_number, "source identifier", name)


def audit_paths(root: Path, relative_paths: Iterable[str]) -> tuple[NamingFinding, ...]:
    """Audit the supplied repository-relative paths."""
    findings: set[NamingFinding] = set()
    for relative in relative_paths:
        if _PATH_TASK_CODE.search(relative):
            findings.add(NamingFinding(relative, 0, "tracked path", relative))
        path = root / relative
        if not path.is_file():
            continue
        first_part = Path(relative).parts[0] if Path(relative).parts else ""
        production = first_part in _PRODUCTION_ROOTS
        if production and path.suffix == ".py":
            findings.update(_python_names(path, relative))
        elif production and path.suffix in _GENERIC_CODE_SUFFIXES:
            findings.update(_generic_code_names(path, relative))
        elif relative.startswith(".github/workflows/") and path.suffix in {".yml", ".yaml"}:
            findings.update(_workflow_names(path, relative))
        elif (
            first_part == "docs"
            and "internal" not in Path(relative).parts
            and path.suffix == ".md"
        ):
            findings.update(_documentation_headings(path, relative))
        if path.suffix == ".json":
            findings.update(_json_names(path, relative))
    return tuple(sorted(findings))


def tracked_paths(root: Path) -> tuple[str, ...]:
    """Return tracked and non-ignored untracked paths from Git."""
    completed = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)


def audit_repository(root: Path) -> tuple[NamingFinding, ...]:
    """Audit all current repository paths relevant to production naming."""
    return audit_paths(root, tracked_paths(root))


def main() -> int:
    """Fail when an internal work-item code leaks into production naming."""
    root = Path(__file__).resolve().parents[1]
    findings = audit_repository(root)
    if findings:
        for finding in findings:
            print(finding.render())
        print(f"descriptive production naming audit failed: {len(findings)} finding(s)")
        return 1
    print("descriptive production naming audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
