# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Rust kernel execution-mode audit helper
"""Static SIMD/threading inventory for Rust PyO3 kernel claims."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import tomllib

SCHEMA = "scpn-rust-kernel-execution-audit/v1"
CLAIM_BOUNDARY = (
    "Static Rust source inventory only: records PyO3 kernel execution-mode "
    "evidence tokens for scalar/unknown, ndarray dot, rayon-threaded, and explicit "
    "SIMD paths. It is not an isolated benchmark, throughput claim, compiler-vectorization "
    "proof, or CPU-affinity artefact."
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
FN_RE = re.compile(r"\b(?:pub\s+)?fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
WRAP_RE = re.compile(
    r"wrap_pyfunction!\(\s*(?P<path>[A-Za-z_][A-Za-z0-9_:]*)\s*,\s*m\s*\)",
    re.DOTALL,
)
RAYON_TOKENS = ("rayon::prelude", ".par_iter(", ".into_par_iter(")
SIMD_TOKENS = ("std::simd::Simd", "std::simd", "Simd<", "packed_simd", "portable_simd")
NDARRAY_DOT_TOKENS = (".dot(", "ndarray::linalg")


@dataclass(frozen=True)
class RustKernelExecutionRecord:
    """One Rust PyO3 kernel execution-mode inventory row."""

    file: str
    line: int
    symbol: str
    registered: bool
    execution_mode: str
    evidence_tokens: tuple[str, ...]
    performance_claim_eligible: bool
    claim_boundary: str


@dataclass(frozen=True)
class RustKernelExecutionAudit:
    """Machine-readable Rust SIMD/threading evidence audit."""

    schema: str
    crate_root: str
    status: str
    claim_boundary: str
    pyfunction_count: int
    rayon_threaded_count: int
    explicit_simd_count: int
    ndarray_dot_count: int
    scalar_or_unknown_count: int
    performance_claim_eligible_count: int
    kernel_records: tuple[RustKernelExecutionRecord, ...]


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
    """Return every PyO3 function registered by the module initializer."""
    lib_rs = crate_root / "src" / "lib.rs"
    if not lib_rs.exists():
        return set()
    text = lib_rs.read_text(encoding="utf-8")
    return {match.group("path").rsplit("::", 1)[-1] for match in WRAP_RE.finditer(text)}


def pyo3_member_crate_roots(crate_root: Path) -> tuple[Path, ...]:
    """Return in-tree path-dependency crates compiled with the ``pyo3`` feature.

    After the program-AD replay extraction the ``scpn_quantum_engine`` PyO3 surface
    is compiled from its own ``src/`` plus every in-tree path dependency it enables
    the ``pyo3`` feature on. Those member crates ship into the same Python extension,
    so the audit scans them alongside the primary crate rather than undercounting the
    relocated kernels. Only path dependencies nested under the crate root are returned,
    keeping every scanned file addressable relative to the reported crate.
    """
    manifest = crate_root / "Cargo.toml"
    if not manifest.exists():
        return ()
    base = crate_root.resolve()
    dependencies = tomllib.loads(manifest.read_text(encoding="utf-8")).get("dependencies", {})
    roots: list[Path] = []
    for spec in dependencies.values():
        if not isinstance(spec, dict):
            continue
        path = spec.get("path")
        if not path or "pyo3" not in spec.get("features", ()):
            continue
        member = (base / path).resolve()
        if base in member.parents and member not in roots:
            roots.append(member)
    return tuple(roots)


def _relative(crate_root: Path, path: Path) -> str:
    return path.relative_to(crate_root).as_posix()


def _next_function_name(lines: Sequence[str], start_index: int) -> str:
    for line in lines[start_index:]:
        match = FN_RE.search(line)
        if match is not None:
            return match.group("name")
    return "<unknown>"


def _collect_tokens(text: str, tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in tokens if token in text)


def _classify_execution_mode(text: str) -> tuple[str, tuple[str, ...]]:
    simd_tokens = _collect_tokens(text, SIMD_TOKENS)
    if simd_tokens:
        return "explicit_simd", simd_tokens
    rayon_tokens = _collect_tokens(text, RAYON_TOKENS)
    if rayon_tokens:
        return "rayon_threaded", rayon_tokens
    ndarray_tokens = _collect_tokens(text, NDARRAY_DOT_TOKENS)
    if ndarray_tokens:
        return "ndarray_dot", ndarray_tokens
    return "scalar_or_unknown", ()


def _scan_file(
    crate_root: Path,
    path: Path,
    registrations: set[str],
) -> tuple[RustKernelExecutionRecord, ...]:
    relative = _relative(crate_root, path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    execution_mode, evidence_tokens = _classify_execution_mode(text)
    records: list[RustKernelExecutionRecord] = []
    for index, line in enumerate(lines):
        if not PYFUNCTION_RE.search(line):
            continue
        symbol = _next_function_name(lines, index + 1)
        records.append(
            RustKernelExecutionRecord(
                file=relative,
                line=index + 1,
                symbol=symbol,
                registered=symbol in registrations,
                execution_mode=execution_mode,
                evidence_tokens=evidence_tokens,
                performance_claim_eligible=False,
                claim_boundary=(
                    "Execution-mode evidence only. Promotion of speedup or production "
                    "performance claims requires separate isolated_affinity benchmark artefacts."
                ),
            )
        )
    return tuple(records)


def scan_crate(crate_root: Path) -> RustKernelExecutionAudit:
    """Scan a Rust crate and its in-tree PyO3 member crates for SIMD/threading evidence."""
    reported_root = crate_root.as_posix()
    resolved_root = crate_root.resolve()
    scan_roots = (resolved_root, *pyo3_member_crate_roots(resolved_root))
    registrations: set[str] = set()
    for root in scan_roots:
        registrations |= registered_pyfunctions(root)
    records: list[RustKernelExecutionRecord] = []
    for root in scan_roots:
        for path in rust_files(root):
            records.extend(_scan_file(resolved_root, path, registrations))
    records.sort(key=lambda item: (item.file, item.line, item.symbol))
    unregistered = [record for record in records if not record.registered]
    status = "fail" if unregistered else "pass"
    return RustKernelExecutionAudit(
        schema=SCHEMA,
        crate_root=reported_root,
        status=status,
        claim_boundary=CLAIM_BOUNDARY,
        pyfunction_count=len(records),
        rayon_threaded_count=sum(record.execution_mode == "rayon_threaded" for record in records),
        explicit_simd_count=sum(record.execution_mode == "explicit_simd" for record in records),
        ndarray_dot_count=sum(record.execution_mode == "ndarray_dot" for record in records),
        scalar_or_unknown_count=sum(
            record.execution_mode == "scalar_or_unknown" for record in records
        ),
        performance_claim_eligible_count=sum(
            record.performance_claim_eligible for record in records
        ),
        kernel_records=tuple(records),
    )


def audit_to_json(audit: RustKernelExecutionAudit) -> str:
    """Serialize an audit result as deterministic JSON."""
    return json.dumps(asdict(audit), indent=2, sort_keys=True)


def format_audit(audit: RustKernelExecutionAudit) -> str:
    """Render a compact CLI summary."""
    return "\n".join(
        (
            f"Rust kernel execution audit: {audit.status}",
            f"- PyO3 functions: {audit.pyfunction_count}",
            f"- rayon-threaded records: {audit.rayon_threaded_count}",
            f"- explicit SIMD records: {audit.explicit_simd_count}",
            f"- ndarray dot records: {audit.ndarray_dot_count}",
            f"- scalar/unknown records: {audit.scalar_or_unknown_count}",
            f"- performance-claim eligible records: {audit.performance_claim_eligible_count}",
        )
    )


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
