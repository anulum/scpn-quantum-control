# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tracked module-size responsibility audit
"""Validate the tracked oversized-code inventory and its architecture decisions.

The audit treats line count as a review trigger, not a defect. Every tracked
code file above the configured threshold must have one registry row naming its
responsibility, dependency boundary, disposition, and reassessment trigger.
The default gate permits only the explicitly recorded refactor debt. ``--strict``
additionally requires that no refactor-required rows remain.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "tools" / "module_size_policy.json"

FileKind = Literal["source", "test", "facade", "entrypoint"]
Disposition = Literal["cohesive", "facade", "entrypoint", "refactor_required"]

_FILE_KINDS = frozenset({"source", "test", "facade", "entrypoint"})
_DISPOSITIONS = frozenset({"cohesive", "facade", "entrypoint", "refactor_required"})


@dataclass(frozen=True)
class PolicyEntry:
    """One reviewed oversized code surface."""

    path: str
    lines: int
    kind: FileKind
    disposition: Disposition
    responsibility: str
    dependency_boundary: str
    reassess_when: str
    refactor_target: str


@dataclass(frozen=True)
class ModuleSizePolicy:
    """Parsed module-size policy registry."""

    threshold_lines: int
    extensions: frozenset[str]
    open_refactor_limit: int
    entries: tuple[PolicyEntry, ...]


@dataclass(frozen=True)
class AuditResult:
    """Repository comparison against the reviewed policy registry."""

    inventory: tuple[tuple[str, int], ...]
    entries: tuple[PolicyEntry, ...]
    errors: tuple[str, ...]
    open_refactors: tuple[PolicyEntry, ...]


def count_physical_lines(path: Path) -> int:
    """Return the physical line count without requiring text decoding."""
    content = path.read_bytes()
    if not content:
        return 0
    return content.count(b"\n") + int(not content.endswith(b"\n"))


def tracked_paths(repo_root: Path) -> tuple[str, ...]:
    """Return the repository-relative paths tracked by Git."""
    result = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(sorted(path.decode("utf-8") for path in result.stdout.split(b"\0") if path))


def build_inventory(
    repo_root: Path,
    paths: tuple[str, ...],
    threshold_lines: int,
    extensions: frozenset[str],
) -> tuple[tuple[str, int], ...]:
    """Return tracked code files whose physical size exceeds the threshold."""
    inventory: list[tuple[str, int]] = []
    for relative in paths:
        if Path(relative).suffix not in extensions:
            continue
        absolute = repo_root / relative
        if not absolute.is_file():
            continue
        lines = count_physical_lines(absolute)
        if lines > threshold_lines:
            inventory.append((relative, lines))
    return tuple(sorted(inventory))


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return cast(list[object], value)


def _text(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _integer(mapping: dict[str, object], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _parse_entry(value: object, index: int, threshold_lines: int) -> PolicyEntry:
    context = f"files[{index}]"
    row = _mapping(value, context)
    path = _text(row, "path", context)
    path_value = Path(path)
    if path_value.is_absolute() or ".." in path_value.parts:
        raise ValueError(f"{context}.path must be repository-relative")
    lines = _integer(row, "lines", context)
    if lines <= threshold_lines:
        raise ValueError(f"{context}.lines must exceed threshold_lines")
    kind_value = _text(row, "kind", context)
    if kind_value not in _FILE_KINDS:
        raise ValueError(f"{context}.kind is unsupported: {kind_value}")
    disposition_value = _text(row, "disposition", context)
    if disposition_value not in _DISPOSITIONS:
        raise ValueError(f"{context}.disposition is unsupported: {disposition_value}")
    kind = cast(FileKind, kind_value)
    disposition = cast(Disposition, disposition_value)
    if disposition == "facade" and kind != "facade":
        raise ValueError(f"{context}: facade disposition requires facade kind")
    if disposition == "entrypoint" and kind != "entrypoint":
        raise ValueError(f"{context}: entrypoint disposition requires entrypoint kind")
    if disposition == "cohesive" and kind not in {"source", "test"}:
        raise ValueError(f"{context}: cohesive disposition requires source or test kind")
    if disposition == "refactor_required" and kind not in {"source", "test"}:
        raise ValueError(f"{context}: refactor-required disposition requires source or test kind")
    refactor_value = row.get("refactor_target", "")
    if not isinstance(refactor_value, str):
        raise ValueError(f"{context}.refactor_target must be a string")
    refactor_target = refactor_value.strip()
    if disposition == "refactor_required" and not refactor_target:
        raise ValueError(f"{context}: refactor-required rows need refactor_target")
    if disposition != "refactor_required" and refactor_target:
        raise ValueError(f"{context}: only refactor-required rows may set refactor_target")
    return PolicyEntry(
        path=path,
        lines=lines,
        kind=kind,
        disposition=disposition,
        responsibility=_text(row, "responsibility", context),
        dependency_boundary=_text(row, "dependency_boundary", context),
        reassess_when=_text(row, "reassess_when", context),
        refactor_target=refactor_target,
    )


def parse_policy(payload: object) -> ModuleSizePolicy:
    """Parse and validate a decoded module-size policy document."""
    root = _mapping(payload, "registry")
    threshold_lines = _integer(root, "threshold_lines", "registry")
    if threshold_lines <= 0:
        raise ValueError("registry.threshold_lines must be positive")
    open_refactor_limit = _integer(root, "open_refactor_limit", "registry")
    if open_refactor_limit < 0:
        raise ValueError("registry.open_refactor_limit must be non-negative")
    extension_values = _sequence(root.get("extensions"), "registry.extensions")
    extensions: set[str] = set()
    for index, value in enumerate(extension_values):
        if not isinstance(value, str) or not value.startswith(".") or len(value) < 2:
            raise ValueError(f"registry.extensions[{index}] must be a dotted suffix")
        extensions.add(value)
    if not extensions:
        raise ValueError("registry.extensions must not be empty")
    entry_values = _sequence(root.get("files"), "registry.files")
    entries = tuple(
        _parse_entry(value, index, threshold_lines) for index, value in enumerate(entry_values)
    )
    paths = [entry.path for entry in entries]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise ValueError(f"registry.files contains duplicate paths: {duplicates}")
    if paths != sorted(paths):
        raise ValueError("registry.files must be sorted by path")
    return ModuleSizePolicy(
        threshold_lines=threshold_lines,
        extensions=frozenset(extensions),
        open_refactor_limit=open_refactor_limit,
        entries=entries,
    )


def load_policy(path: Path) -> ModuleSizePolicy:
    """Load a UTF-8 JSON module-size policy registry."""
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    return parse_policy(payload)


def audit_policy(inventory: tuple[tuple[str, int], ...], policy: ModuleSizePolicy) -> AuditResult:
    """Compare a live oversized-file inventory with its reviewed policy."""
    actual = dict(inventory)
    registered = {entry.path: entry for entry in policy.entries}
    errors: list[str] = []
    for path in sorted(actual.keys() - registered.keys()):
        errors.append(f"unreviewed oversized file: {path} ({actual[path]} lines)")
    for path in sorted(registered.keys() - actual.keys()):
        errors.append(f"stale oversized-file registry row: {path}")
    for path in sorted(actual.keys() & registered.keys()):
        expected = registered[path].lines
        observed = actual[path]
        if expected != observed:
            errors.append(f"line-count drift: {path} records {expected}, observed {observed}")
    open_refactors = tuple(
        entry for entry in policy.entries if entry.disposition == "refactor_required"
    )
    if len(open_refactors) != policy.open_refactor_limit:
        errors.append(
            "open-refactor ratchet drift: "
            f"registry expects {policy.open_refactor_limit}, observed {len(open_refactors)}"
        )
    return AuditResult(
        inventory=inventory,
        entries=policy.entries,
        errors=tuple(errors),
        open_refactors=open_refactors,
    )


def audit_repository(repo_root: Path, registry_path: Path) -> AuditResult:
    """Load policy, derive tracked inventory, and return their comparison."""
    policy = load_policy(registry_path)
    inventory = build_inventory(
        repo_root,
        tracked_paths(repo_root),
        policy.threshold_lines,
        policy.extensions,
    )
    return audit_policy(inventory, policy)


def format_result(result: AuditResult) -> str:
    """Render a compact human-readable audit result."""
    if result.errors:
        return "module-size policy audit failed:\n" + "\n".join(
            f"- {error}" for error in result.errors
        )
    lines = [
        "module-size policy inventory current: "
        f"{len(result.inventory)} file(s), {len(result.open_refactors)} open refactor(s)"
    ]
    lines.extend(
        f"- OPEN {entry.path}: {entry.refactor_target}" for entry in result.open_refactors
    )
    return "\n".join(lines)


def result_exit_code(result: AuditResult, strict: bool) -> int:
    """Return the process status for inventory-only or strict certification mode."""
    if result.errors:
        return 1
    if strict and result.open_refactors:
        return 1
    return 0


def _json_result(result: AuditResult, strict: bool) -> str:
    return json.dumps(
        {
            "status": "fail" if result_exit_code(result, strict) else "pass",
            "strict": strict,
            "oversized_files": len(result.inventory),
            "open_refactors": [entry.path for entry in result.open_refactors],
            "errors": list(result.errors),
        },
        indent=2,
        sort_keys=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail while any reviewed refactor-required rows remain",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tracked module-size responsibility audit."""
    args = _parser().parse_args(argv)
    try:
        result = audit_repository(args.repo_root.resolve(), args.registry.resolve())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        if args.json:
            print(json.dumps({"errors": [str(exc)], "status": "error"}, indent=2, sort_keys=True))
        else:
            print(f"module-size policy audit failed:\n- {exc}")
        return 2
    print(_json_result(result, args.strict) if args.json else format_result(result))
    return result_exit_code(result, args.strict)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
