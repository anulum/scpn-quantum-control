# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Rust FFI safety audit helper
"""Static Rust FFI safety inventory for the PyO3 extension boundary."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "scpn-rust-ffi-safety-audit/v1"
CLAIM_BOUNDARY = (
    "Static Rust FFI inventory only: records PyO3 boundaries, registration drift, "
    'extern "C" declarations, and literal unsafe tokens. It is not a formal proof, '
    "Miri run, sanitizer run, fuzz campaign, or dynamic memory-safety guarantee."
)
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    "__pycache__",
}
PYFUNCTION_RE = re.compile(r"^\s*#\s*\[\s*pyfunction\b")
PYMODULE_RE = re.compile(r"^\s*#\s*\[\s*pymodule\b")
FN_RE = re.compile(r"\b(?:pub\s+)?fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
UNSAFE_RE = re.compile(r"\bunsafe\b")
EXTERN_C_RE = re.compile(r'extern\s+"C"')
WRAP_RE = re.compile(
    r"wrap_pyfunction!\(\s*(?P<path>[A-Za-z_][A-Za-z0-9_:]*)\s*,\s*m\s*\)",
    re.DOTALL,
)


@dataclass(frozen=True)
class RustFfiBoundary:
    """One PyO3 boundary discovered in Rust source."""

    file: str
    line: int
    kind: str
    symbol: str
    registered: bool


@dataclass(frozen=True)
class UnsafeOccurrence:
    """One literal Rust unsafe token discovered in source."""

    file: str
    line: int
    symbol: str
    occurrence_kind: str
    text: str


@dataclass(frozen=True)
class RustFfiSafetyAudit:
    """Machine-readable Rust FFI safety audit result."""

    schema: str
    crate_root: str
    status: str
    claim_boundary: str
    pyfunction_count: int
    pymodule_count: int
    extern_c_count: int
    unsafe_occurrence_count: int
    unregistered_pyfunction_count: int
    pyo3_boundaries: tuple[RustFfiBoundary, ...]
    unsafe_occurrences: tuple[UnsafeOccurrence, ...]


def rust_files(crate_root: Path) -> tuple[Path, ...]:
    """Return deterministic Rust source files below the crate root."""
    src = crate_root / "src"
    if not src.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in src.rglob("*.rs")
                if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts)
            ),
            key=lambda item: item.relative_to(crate_root).as_posix(),
        )
    )


def registered_pyfunctions(crate_root: Path) -> set[str]:
    """Return PyO3 functions registered by the module initializer."""
    lib_rs = crate_root / "src" / "lib.rs"
    if not lib_rs.exists():
        return set()
    text = lib_rs.read_text(encoding="utf-8")
    return {match.group("path").rsplit("::", 1)[-1] for match in WRAP_RE.finditer(text)}


def _relative(crate_root: Path, path: Path) -> str:
    return path.relative_to(crate_root).as_posix()


def _next_function_name(lines: Sequence[str], start_index: int) -> str:
    for line in lines[start_index:]:
        match = FN_RE.search(line)
        if match is not None:
            return match.group("name")
    return "<unknown>"


def _enclosing_function(lines: Sequence[str], start_index: int) -> str:
    for line in reversed(lines[: start_index + 1]):
        match = FN_RE.search(line)
        if match is not None:
            return match.group("name")
    return "<module>"


def _occurrence_kind(line: str) -> str:
    return "block" if re.search(r"\bunsafe\s*\{", line) else "token"


def _scan_file(
    crate_root: Path,
    path: Path,
    registrations: set[str],
) -> tuple[tuple[RustFfiBoundary, ...], tuple[UnsafeOccurrence, ...], int]:
    relative = _relative(crate_root, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    boundaries: list[RustFfiBoundary] = []
    unsafe_occurrences: list[UnsafeOccurrence] = []
    extern_c_count = 0
    for index, line in enumerate(lines):
        line_number = index + 1
        if PYFUNCTION_RE.search(line):
            symbol = _next_function_name(lines, index + 1)
            boundaries.append(
                RustFfiBoundary(
                    file=relative,
                    line=line_number,
                    kind="pyfunction",
                    symbol=symbol,
                    registered=symbol in registrations,
                )
            )
        if PYMODULE_RE.search(line):
            boundaries.append(
                RustFfiBoundary(
                    file=relative,
                    line=line_number,
                    kind="pymodule",
                    symbol=_next_function_name(lines, index + 1),
                    registered=True,
                )
            )
        if EXTERN_C_RE.search(line):
            extern_c_count += 1
        if UNSAFE_RE.search(line):
            unsafe_occurrences.append(
                UnsafeOccurrence(
                    file=relative,
                    line=line_number,
                    symbol=_enclosing_function(lines, index),
                    occurrence_kind=_occurrence_kind(line),
                    text=line.strip(),
                )
            )
    return tuple(boundaries), tuple(unsafe_occurrences), extern_c_count


def scan_crate(crate_root: Path) -> RustFfiSafetyAudit:
    """Scan one Rust crate for PyO3 boundary and unsafe-token evidence."""
    reported_root = crate_root.as_posix()
    resolved_root = crate_root.resolve()
    registrations = registered_pyfunctions(resolved_root)
    boundaries: list[RustFfiBoundary] = []
    unsafe_occurrences: list[UnsafeOccurrence] = []
    extern_c_count = 0
    for path in rust_files(resolved_root):
        file_boundaries, file_unsafe, file_extern_c_count = _scan_file(
            resolved_root, path, registrations
        )
        boundaries.extend(file_boundaries)
        unsafe_occurrences.extend(file_unsafe)
        extern_c_count += file_extern_c_count

    boundaries.sort(key=lambda item: (item.file, item.line, item.kind, item.symbol))
    unsafe_occurrences.sort(key=lambda item: (item.file, item.line, item.symbol))
    unregistered = [
        boundary
        for boundary in boundaries
        if boundary.kind == "pyfunction" and not boundary.registered
    ]
    status = "fail" if unsafe_occurrences or unregistered else "pass"
    return RustFfiSafetyAudit(
        schema=SCHEMA,
        crate_root=reported_root,
        status=status,
        claim_boundary=CLAIM_BOUNDARY,
        pyfunction_count=sum(boundary.kind == "pyfunction" for boundary in boundaries),
        pymodule_count=sum(boundary.kind == "pymodule" for boundary in boundaries),
        extern_c_count=extern_c_count,
        unsafe_occurrence_count=len(unsafe_occurrences),
        unregistered_pyfunction_count=len(unregistered),
        pyo3_boundaries=tuple(boundaries),
        unsafe_occurrences=tuple(unsafe_occurrences),
    )


def audit_to_json(audit: RustFfiSafetyAudit) -> str:
    """Serialize an audit result as deterministic JSON."""
    return json.dumps(asdict(audit), indent=2, sort_keys=True)


def format_audit(audit: RustFfiSafetyAudit) -> str:
    """Render a compact CLI summary."""
    lines = [
        f"Rust FFI safety audit: {audit.status}",
        f"- PyO3 functions: {audit.pyfunction_count}",
        f"- PyO3 modules: {audit.pymodule_count}",
        f"- unregistered PyO3 functions: {audit.unregistered_pyfunction_count}",
        f"- unsafe occurrences: {audit.unsafe_occurrence_count}",
        f"- extern C declarations: {audit.extern_c_count}",
    ]
    for occurrence in audit.unsafe_occurrences:
        lines.append(
            f"  unsafe: {occurrence.file}:{occurrence.line} "
            f"{occurrence.symbol} ({occurrence.occurrence_kind})"
        )
    for boundary in audit.pyo3_boundaries:
        if boundary.kind == "pyfunction" and not boundary.registered:
            lines.append(f"  unregistered: {boundary.file}:{boundary.line} {boundary.symbol}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crate-root", type=Path, default=Path("scpn_quantum_engine"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    audit = scan_crate(args.crate_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(audit_to_json(audit) + "\n", encoding="utf-8")

    print(audit_to_json(audit) if args.json and args.output is None else format_audit(audit))
    return 0 if audit.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
