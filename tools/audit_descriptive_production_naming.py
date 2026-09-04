# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — descriptive production naming audit
"""Reject internal work-item codes from production-facing names."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess  # nosec B404
import tokenize
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_TASK_CODE: Final[re.Pattern[str]] = re.compile(
    r"(?ix)(?<![A-Za-z0-9])(?:"
    r"qwc[_-]?\d+(?:\.\d+)+(?:[._-]?[a-z])?"
    r"|(?:bl|st|dp|rg|hg|qwc|ws|lock|kimi|aud)[_-]?\d+(?:[._-]?[a-z])?"
    r"|kt-\d+(?:[._-]?[a-z])?"
    r"|fu[_-](?:\d+|[a-z])"
    r"|co\d+[_-][a-z]+[_-]\d+"
    r"|sec[_-]?\d+"
    r"|s\d+\.\d+"
    r")(?![A-Za-z0-9])"
)
_PATH_TASK_CODE: Final[re.Pattern[str]] = re.compile(
    r"(?ix)(?:^|[/_.-])(?:"
    r"qwc[_-]?\d+(?:\.\d+)+(?:[._-]?[a-z])?"
    r"|(?:bl|st|dp|rg|hg|qwc|ws|lock|kimi|aud)[_-]?\d+(?:[._-]?[a-z])?"
    r"|kt-\d+(?:[._-]?[a-z])?"
    r"|fu[_-](?:\d+|[a-z])"
    r"|co\d+[_-][a-z]+[_-]\d+"
    r"|sec[_-]?\d+"
    r"|s\d+\.\d+"
    r")(?![A-Za-z0-9])"
)
_CAMPAIGN_STAGE_PROSE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(?:W\d+|(?:post|pre)[-_]?[Ww]\d+)(?![A-Za-z0-9])"
)
_PATH_CAMPAIGN_STAGE: Final[re.Pattern[str]] = re.compile(r"(?i)(?:^|[/_.-])w\d+(?=$|[/_.-])")
_QUALIFIED_CAMPAIGN_STAGE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:^|_)(?:pre|post)_?w\d+(?=$|_)"
)
_CAMPAIGN_STAGE_CONTEXT: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:analys|calibrat|campaign|count|epoch|hardware|observ|result|sensitiv|status|submit|window)"
)
_MACHINE_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.:/-]+")
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PRODUCTION_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "oscillatools",
        "scpn_quantum_engine",
        "scripts",
        "src",
        "studio-web",
        "tests",
        "tools",
    }
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
_BASELINE_SCHEMA: Final = "scpn_qc.descriptive_production_naming_baseline.v1"
_BASELINE_PATH: Final = Path("tools/descriptive_production_naming_baseline.json")
_PUBLIC_ROOT_DOCUMENTS: Final[frozenset[str]] = frozenset(
    {
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "README.md",
        "ROADMAP.md",
        "VALIDATION.md",
    }
)
_EXACT_NEGATIVE_FIXTURES: Final[frozenset[str]] = frozenset(
    {
        "tests/test_audit_descriptive_production_naming.py",
        "tests/test_bench_cli_branches.py",
    }
)
_EXACT_NEGATIVE_VALUES: Final[frozenset[tuple[str, str]]] = frozenset(
    ("tests/test_binding_spec.py", f"ws_{index}") for index in range(3)
)


def _contains_path_campaign_stage(value: str) -> bool:
    """Return whether a path-like value contains abbreviated window staging."""
    path_like = "/" in value or value.endswith((".json", ".md", ".yaml", ".yml"))
    return path_like and bool(_PATH_CAMPAIGN_STAGE.search(value))


def _contains_identifier_campaign_stage(value: str) -> bool:
    """Reject stage shorthand inside descriptive names, but allow bare math weights."""
    if _QUALIFIED_CAMPAIGN_STAGE.search(value):
        return True
    return bool(_PATH_CAMPAIGN_STAGE.search(value) and _CAMPAIGN_STAGE_CONTEXT.search(value))


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
    """Yield task-coded Python identifiers, docs, comments, and runtime strings."""
    text = path.read_text(encoding="utf-8")
    if not (
        _TASK_CODE.search(text)
        or _CAMPAIGN_STAGE_PROSE.search(text)
        or _PATH_CAMPAIGN_STAGE.search(text)
    ):
        return
    tree = ast.parse(text, filename=display_path)
    lines = text.splitlines()
    if len(lines) >= 7 and _TASK_CODE.search(lines[6]):
        yield NamingFinding(display_path, 7, "module heading", lines[6].strip())
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if (
            token.type == tokenize.COMMENT
            and token.start[0] != 7
            and (_TASK_CODE.search(token.string) or _CAMPAIGN_STAGE_PROSE.search(token.string))
        ):
            yield NamingFinding(
                display_path,
                token.start[0],
                "source comment",
                token.string.strip(),
            )
    module_doc = ast.get_docstring(tree, clean=False)
    module_doc_node_id: int | None = None
    if module_doc and (_TASK_CODE.search(module_doc) or _CAMPAIGN_STAGE_PROSE.search(module_doc)):
        first = tree.body[0]
        module_doc_node_id = id(first.value) if isinstance(first, ast.Expr) else None
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
            if _TASK_CODE.search(name) or _contains_identifier_campaign_stage(name):
                yield NamingFinding(
                    display_path,
                    getattr(node, "lineno", 0),
                    "Python identifier",
                    name,
                )
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            if id(node) != module_doc_node_id and (
                _TASK_CODE.search(node.value) or _CAMPAIGN_STAGE_PROSE.search(node.value)
            ):
                yield NamingFinding(
                    display_path,
                    node.lineno,
                    "production docstring",
                    node.value,
                )
            continue
        path_stage = _contains_path_campaign_stage(node.value)
        if node.value.startswith("docs/internal/") or not (
            _TASK_CODE.search(node.value) or _CAMPAIGN_STAGE_PROSE.search(node.value) or path_stage
        ):
            continue
        kind = (
            "machine-facing string"
            if _MACHINE_NAME.fullmatch(node.value)
            else "runtime or user-facing string"
        )
        yield NamingFinding(display_path, node.lineno, kind, node.value)


def _json_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded JSON keys and string values."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return

    def walk(value: object) -> Iterator[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                if _TASK_CODE.search(key) or _contains_identifier_campaign_stage(key):
                    yield key
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
        elif isinstance(value, str) and len(value) <= 4096:
            path_stage = _contains_path_campaign_stage(value)
            if (
                _TASK_CODE.search(value)
                or (len(value) <= 512 and _CAMPAIGN_STAGE_PROSE.search(value))
                or path_stage
            ):
                yield value

    for value in walk(payload):
        yield NamingFinding(display_path, 1, "JSON machine name", value)


def _workflow_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded workflow names, comments, values, and job IDs."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("name:") or stripped.startswith("- name:"):
            if _TASK_CODE.search(stripped) or _CAMPAIGN_STAGE_PROSE.search(stripped):
                yield NamingFinding(display_path, line_number, "workflow name", stripped)
        elif line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            job_id = stripped[:-1]
            if _TASK_CODE.search(job_id) or _contains_identifier_campaign_stage(job_id):
                yield NamingFinding(display_path, line_number, "workflow job ID", job_id)
        elif _TASK_CODE.search(line) or _CAMPAIGN_STAGE_PROSE.search(line):
            yield NamingFinding(display_path, line_number, "workflow text", stripped)


def _documentation_text(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield public documentation text that exposes internal task codes."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not (
            _TASK_CODE.search(line)
            or _CAMPAIGN_STAGE_PROSE.search(line)
            or _contains_path_campaign_stage(line)
        ):
            continue
        kind = "documentation heading" if line.startswith("#") else "public documentation text"
        yield NamingFinding(display_path, line_number, kind, line)


def _generic_code_names(path: Path, display_path: str) -> Iterator[NamingFinding]:
    """Yield task-coded text from non-Python production sources."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _TASK_CODE.search(line) or _CAMPAIGN_STAGE_PROSE.search(line):
            yield NamingFinding(display_path, line_number, "source text", line.strip())
        for name in _IDENTIFIER.findall(line):
            if _TASK_CODE.search(name) or _contains_identifier_campaign_stage(name):
                yield NamingFinding(display_path, line_number, "source identifier", name)


def audit_paths(root: Path, relative_paths: Iterable[str]) -> tuple[NamingFinding, ...]:
    """Audit the supplied repository-relative paths."""
    findings: set[NamingFinding] = set()
    for relative in relative_paths:
        if relative in _EXACT_NEGATIVE_FIXTURES:
            continue
        if _PATH_TASK_CODE.search(relative) or _PATH_CAMPAIGN_STAGE.search(relative):
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
        elif path.suffix == ".md" and (
            relative in _PUBLIC_ROOT_DOCUMENTS
            or (first_part == "docs" and "internal" not in Path(relative).parts)
        ):
            findings.update(_documentation_text(path, relative))
        public_json = (
            first_part in _PRODUCTION_ROOTS
            or first_part == "data"
            or first_part == "notebooks"
            or (first_part == "docs" and "internal" not in Path(relative).parts)
        )
        if public_json and path.suffix in {".ipynb", ".json"}:
            findings.update(_json_names(path, relative))
    return tuple(
        sorted(
            finding
            for finding in findings
            if (finding.path, finding.value) not in _EXACT_NEGATIVE_VALUES
        )
    )


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


def finding_fingerprint(finding: NamingFinding) -> str:
    """Return a line-number-independent identity for one naming violation."""
    payload = "\0".join((finding.path, finding.kind, finding.value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def baseline_payload(findings: Iterable[NamingFinding]) -> dict[str, object]:
    """Return a stable counted baseline for existing repository debt."""
    counts = Counter(finding_fingerprint(finding) for finding in findings)
    return {
        "schema": _BASELINE_SCHEMA,
        "known_finding_counts": dict(sorted(counts.items())),
    }


def load_baseline(path: Path) -> Counter[str]:
    """Load and validate the counted naming-debt baseline."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _BASELINE_SCHEMA:
        raise ValueError("unexpected descriptive production naming baseline schema")
    raw_counts = payload.get("known_finding_counts")
    if not isinstance(raw_counts, dict):
        raise ValueError("known_finding_counts must be an object")
    counts: Counter[str] = Counter()
    for fingerprint, count in raw_counts.items():
        if (
            not isinstance(fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
        ):
            raise ValueError("baseline entries must map SHA-256 strings to positive integers")
        counts[fingerprint] = count
    return counts


def unexpected_findings(
    findings: Iterable[NamingFinding], known_counts: Mapping[str, int]
) -> tuple[NamingFinding, ...]:
    """Return findings that exceed the exact pre-existing debt allowance."""
    seen: Counter[str] = Counter()
    unexpected: list[NamingFinding] = []
    for finding in findings:
        fingerprint = finding_fingerprint(finding)
        seen[fingerprint] += 1
        if seen[fingerprint] > known_counts.get(fingerprint, 0):
            unexpected.append(finding)
    return tuple(unexpected)


def main() -> int:
    """Fail on new internal-code leakage or write an explicit debt baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="replace the counted baseline with the current repository findings",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    findings = audit_repository(root)
    baseline_path = root / _BASELINE_PATH
    if args.write_baseline:
        baseline_path.write_text(
            json.dumps(baseline_payload(findings), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(findings)} known findings to {baseline_path}")
        return 0
    try:
        known_counts = load_baseline(baseline_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"descriptive production naming baseline is invalid: {exc}")
        return 1
    unexpected = unexpected_findings(findings, known_counts)
    if unexpected:
        for finding in unexpected:
            print(finding.render())
        print(f"descriptive production naming audit failed: {len(unexpected)} new finding(s)")
        return 1
    print(f"descriptive production naming audit passed ({len(findings)} known finding(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
