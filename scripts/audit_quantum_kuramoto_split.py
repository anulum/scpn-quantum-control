# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — audit quantum kuramoto split script
# scpn-quantum-control -- S6 quantum-kuramoto split audit
"""Audit the feasible boundary for a decoupled quantum-kuramoto package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

DATE = "2026-05-07"
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "scpn_quantum_control"
OUT_DIR = REPO_ROOT / "data" / "s6_quantum_kuramoto_split"
DOC_PATH = (
    REPO_ROOT
    / "docs"
    / "internal"
    / "audits"
    / "split_package"
    / f"quantum_kuramoto_split_audit_{DATE}.md"
)
CANDIDATE_PACKAGES = ("phase", "bridge", "hardware", "accel")
CANDIDATE_ROOT_MODULES = ("_rust_accel",)
COMPATIBILITY_ALIASES = {
    "scpn_quantum_control.accel.rust_import": "oscillatools.accel.rust_import",
}
SCPN_MARKERS = (
    "ssgf",
    "snn",
    "sc_to_quantum",
    "orchestrator",
    "control_plasma",
    "build_knm_paper27",
    "OMEGA_N_16",
    "fim",
    "feedback",
)
ALLOWED_FOUNDATION_IMPORTS = (
    "scpn_quantum_control._rust_accel",
    "scpn_quantum_control.accel",
    "scpn_quantum_control.bridge.knm_hamiltonian",
    "scpn_quantum_control.bridge.sparse_hamiltonian",
    "scpn_quantum_control.dense_budget",
    "scpn_quantum_control.hardware",
    "scpn_quantum_control.phase",
)

SplitStatus = Literal["reusable", "needs_review", "scpn_specific"]


@dataclass(frozen=True, slots=True)
class SplitAuditRow:
    """One source-backed candidate module observation.

    Attributes
    ----------
    module
        Requested import address or declared static source address.
    path
        Repository-relative source or explicit external provider reference.
    status
        Conservative reuse classification.
    reasons
        Source observations supporting the classification.
    internal_imports
        Resolved first-party import addresses.
    external_import_roots
        External dependency roots, including relative provider imports.
    canonical_module
        Actual alias provider or declared static package context.
    source_sha256
        Digest of the inspected physical source bytes.

    """

    module: str
    path: str
    status: SplitStatus
    reasons: tuple[str, ...]
    internal_imports: tuple[str, ...]
    external_import_roots: tuple[str, ...]
    canonical_module: str | None = None
    source_sha256: str | None = None


def _candidate_files(source_root: Path) -> list[Path]:
    """List candidate source files and the required root foundation.

    Parameters
    ----------
    source_root
        Physical package root under inspection.

    Returns
    -------
    list[Path]
        Existing package sources excluding cache directories, followed by the
        required foundation paths whose absence will fail the source audit.

    """
    files: list[Path] = []
    for package in CANDIDATE_PACKAGES:
        root = source_root / package
        if root.is_dir():
            files.extend(
                path for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts
            )
    files.extend(source_root / f"{name}.py" for name in CANDIDATE_ROOT_MODULES)
    return files


def _imports(path: Path, *, module_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Inspect import statements in a physical source or an alias provider.

    Parameters
    ----------
    path
        Existing Python source file to inspect.
    module_name
        Canonical provider address for a source outside the parent package.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        Sorted internal addresses and external import roots.

    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    internal: set[str] = set()
    external: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if alias.name.startswith("scpn_quantum_control"):
                    internal.add(alias.name)
                else:
                    external.add(root)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = _resolve_relative_module(path, node.level, module, module_name=module_name)
                if base.startswith("scpn_quantum_control"):
                    internal.add(base)
                else:
                    external.add(base.split(".")[0])
            elif module.startswith("scpn_quantum_control"):
                internal.add(module)
            else:
                external.add(module.split(".")[0])
    return tuple(sorted(internal)), tuple(sorted(external))


def _resolve_relative_module(path: Path, level: int, module: str, *, module_name: str) -> str:
    """Resolve a relative import in its actual owning package.

    Parameters
    ----------
    path
        Existing source file under inspection.
    level
        Number of relative import dots.
    module
        Relative module suffix, which may be empty.
    module_name
        Canonical address of an external compatibility provider.

    Returns
    -------
    str
        Resolved address in the declared owning package.

    Raises
    ------
    ValueError
        If the relative import leaves its declared package.

    """
    address_parts = module_name.split(".")
    package_parts = (
        address_parts
        if path.name == "__init__.py" and address_parts[-1] != "__init__"
        else address_parts[:-1]
    )
    if level > len(package_parts):
        raise ValueError(f"relative import leaves declared package: {module_name}")
    base = package_parts[: len(package_parts) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _status(
    path: Path,
    internal_imports: tuple[str, ...],
    *,
    module_name: str,
) -> tuple[SplitStatus, tuple[str, ...]]:
    """Classify inspected source without treating an alias as a physical file.

    Parameters
    ----------
    path
        Actual source file containing the implementation.
    internal_imports
        Resolved imports from the source inspection.
    module_name
        Canonical provider address when inspecting an external alias.

    Returns
    -------
    tuple[SplitStatus, tuple[str, ...]]
        Scientific reuse posture and the observations supporting it.

    """
    text = path.read_text(encoding="utf-8")
    module = module_name
    reasons: list[str] = []
    lower_module = module.lower()
    if any(marker.lower() in lower_module for marker in SCPN_MARKERS):
        reasons.append("module_name_contains_scpn_specific_marker")
    if any(marker in text for marker in SCPN_MARKERS):
        reasons.append("source_contains_scpn_specific_marker")
    unsupported_internal = tuple(
        item
        for item in internal_imports
        if not item.startswith(ALLOWED_FOUNDATION_IMPORTS)
        and item != "scpn_quantum_control"
        and not item.startswith(module.rsplit(".", 1)[0])
    )
    if unsupported_internal:
        reasons.append("imports_non_foundation_scpn_module")
    if module.endswith(("phase.xy_kuramoto", "phase.xy_compiler", "phase.trotter_error")):
        reasons.append("core_kuramoto_candidate")
    if module.endswith(("hardware.runner", "hardware.async_runner", "hardware.backends")):
        reasons.append("hardware_core_candidate")
    if module.endswith(("accel.dispatcher", "accel.rust_import", "accel.rust_kuramoto_classical")):
        reasons.append("acceleration_candidate")

    if (
        "module_name_contains_scpn_specific_marker" in reasons
        or "source_contains_scpn_specific_marker" in reasons
    ):
        return "scpn_specific", tuple(reasons)
    if "imports_non_foundation_scpn_module" in reasons:
        return "needs_review", tuple(reasons)
    return "reusable", tuple(reasons or ["no_scpn_specific_marker_detected"])


def audit_module_source(module: str, path: Path) -> SplitAuditRow:
    """Audit a physical source in its declared canonical package context.

    Parameters
    ----------
    module
        Canonical dotted address used to resolve source imports.
    path
        Existing Python source, including an offline source snapshot.

    Returns
    -------
    SplitAuditRow
        Static import classification, physical provenance and exact byte digest.
        This source audit does not assert runtime importability.

    Raises
    ------
    ValueError
        If an import leaves the declared package.
    SyntaxError
        If the source cannot be parsed as Python.

    """
    path = path.resolve()
    internal, external = _imports(path, module_name=module)
    status, reasons = _status(path, internal, module_name=module)
    reference = (
        str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(REPO_ROOT)
        else f"external:{module}"
    )
    return SplitAuditRow(
        module=module,
        path=reference,
        status=status,
        reasons=reasons,
        internal_imports=internal,
        external_import_roots=external,
        canonical_module=module,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _compatibility_alias_row(requested: str, canonical: str) -> SplitAuditRow:
    """Inspect the real implementation behind an existing module alias.

    Parameters
    ----------
    requested
        Existing compatibility import address retained by the parent package.
    canonical
        Reviewed owning address the import must resolve to.

    Returns
    -------
    SplitAuditRow
        Alias row with canonical identity, actual source digest and import audit.

    Raises
    ------
    ImportError
        If the existing compatibility surface cannot be imported.
    ValueError
        If the alias resolves to another owner or has no physical source.

    """
    provider = importlib.import_module(requested)
    source_file = getattr(provider, "__file__", None)
    if provider.__name__ != canonical or not isinstance(source_file, str):
        raise ValueError("compatibility alias has no verified canonical source")
    row = audit_module_source(canonical, Path(source_file))
    return replace(
        row,
        module=requested,
        reasons=(*row.reasons, "verified_compatibility_alias"),
    )


def build_split_audit(
    *, source_root: Path = SRC_ROOT, compatibility_aliases: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Build the split audit from physical sources and verified public aliases.

    Parameters
    ----------
    source_root
        Physical package root or offline source snapshot. The root accelerator
        foundation must exist; optional package directories may be absent.
    compatibility_aliases
        Requested addresses and their reviewed providers; defaults to the
        existing accelerator compatibility contract.

    Returns
    -------
    dict[str, object]
        Import classifications with source-backed compatibility provenance.

    """
    rows: list[SplitAuditRow] = []
    for path in _candidate_files(source_root):
        relative = path.relative_to(source_root).with_suffix("")
        module = ".".join(("scpn_quantum_control", *relative.parts))
        rows.append(audit_module_source(module, path))
    rows.extend(
        _compatibility_alias_row(requested, canonical)
        for requested, canonical in (
            COMPATIBILITY_ALIASES if compatibility_aliases is None else compatibility_aliases
        ).items()
    )
    counts = {status: sum(1 for row in rows if row.status == status) for status in _STATUSES}
    return {
        "schema": "s6_quantum_kuramoto_split_audit_v1",
        "date": DATE,
        "source_root": str(source_root.resolve()),
        "candidate_packages": list(CANDIDATE_PACKAGES),
        "candidate_root_modules": list(CANDIDATE_ROOT_MODULES),
        "statuses": counts,
        "acceptance_boundary": {
            "safe_to_publish_package_now": False,
            "reason": "first-pass import audit only; no package skeleton or publish workflow yet",
            "required_next_steps": [
                "manually review needs_review rows",
                "define stable public API for reusable rows",
                "create package skeleton only after boundary review",
                "add import-compatibility tests for scpn_quantum_control re-exports",
            ],
        },
        "rows": [asdict(row) for row in rows],
    }


_STATUSES: tuple[SplitStatus, ...] = ("reusable", "needs_review", "scpn_specific")


def _markdown(payload: dict[str, object]) -> str:
    statuses = payload["statuses"]
    if not isinstance(statuses, dict):
        raise TypeError("statuses must be a dictionary")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    lines = [
        "# S6 Quantum-Kuramoto Split Audit",
        "",
        "This is a first-pass import-graph and marker audit for a future decoupled `quantum-kuramoto` package. It does not create or publish a second package.",
        "",
        "## Status Counts",
        f"- Reusable: `{statuses.get('reusable', 0)}`",
        f"- Needs review: `{statuses.get('needs_review', 0)}`",
        f"- SCPN-specific: `{statuses.get('scpn_specific', 0)}`",
        "",
        "## Boundary",
        "- Safe to publish now: `False`",
        "- Reason: first-pass import audit only; no package skeleton or publish workflow yet.",
        "",
        "## Reusable Candidates",
    ]
    for row in rows:
        if isinstance(row, dict) and row.get("status") == "reusable":
            lines.append(f"- `{row['module']}` — {', '.join(row['reasons'])}")
    lines.extend(["", "## Review or Exclusion Rows"])
    for row in rows:
        if isinstance(row, dict) and row.get("status") != "reusable":
            lines.append(f"- `{row['module']}` — `{row['status']}` — {', '.join(row['reasons'])}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse the quantum-Kuramoto split audit CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def write_split_audit(
    payload: dict[str, object], *, json_path: Path, doc_path: Path
) -> tuple[str, str]:
    """Export an audit object as validated Markdown and digest-linked JSON.

    Parameters
    ----------
    payload
        Current or previously stored split audit object.
    json_path
        Destination JSON file.
    doc_path
        Destination Markdown report.

    Returns
    -------
    tuple[str, str]
        SHA256 digests of the actual JSON and Markdown bytes.

    Raises
    ------
    TypeError
        If status counts or module rows cannot be rendered.

    """
    document = _markdown(payload)
    return _write_json(json_path, payload), _write_text(doc_path, document)


def main() -> int:
    """Run the quantum-Kuramoto split audit and write public artefacts."""
    args = parse_args()
    payload = build_split_audit(source_root=args.source_root)
    json_path = args.out_dir / f"quantum_kuramoto_split_audit_{DATE}.json"
    sha_json, sha_md = write_split_audit(payload, json_path=json_path, doc_path=args.doc_path)
    print(f"wrote {json_path} sha256={sha_json}")
    print(f"wrote {args.doc_path} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
