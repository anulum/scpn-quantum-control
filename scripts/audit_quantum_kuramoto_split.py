# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S6 quantum-kuramoto split audit
"""Audit the feasible boundary for a decoupled quantum-kuramoto package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

DATE = "2026-05-07"
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "scpn_quantum_control"
OUT_DIR = REPO_ROOT / "data" / "s6_quantum_kuramoto_split"
DOC_PATH = REPO_ROOT / "docs" / f"quantum_kuramoto_split_audit_{DATE}.md"
CANDIDATE_PACKAGES = ("phase", "bridge", "hardware", "accel")
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
    """One candidate module row for the split audit."""

    module: str
    path: str
    status: SplitStatus
    reasons: tuple[str, ...]
    internal_imports: tuple[str, ...]
    external_import_roots: tuple[str, ...]


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT / "src").with_suffix("")
    return ".".join(rel.parts)


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    for package in CANDIDATE_PACKAGES:
        root = SRC_ROOT / package
        if root.is_dir():
            files.extend(
                path for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts
            )
    return files


def _imports(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
                base = _resolve_relative_module(path, node.level, module)
                if base:
                    internal.add(base)
            elif module.startswith("scpn_quantum_control"):
                internal.add(module)
            elif module:
                external.add(module.split(".")[0])
    return tuple(sorted(internal)), tuple(sorted(external))


def _resolve_relative_module(path: Path, level: int, module: str) -> str | None:
    package_parts = _module_name(path).split(".")[:-1]
    if level > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(base) if base else None


def _status(path: Path, internal_imports: tuple[str, ...]) -> tuple[SplitStatus, tuple[str, ...]]:
    text = path.read_text(encoding="utf-8")
    module = _module_name(path)
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


def build_split_audit() -> dict[str, object]:
    """Build the S6 split audit payload."""
    rows: list[SplitAuditRow] = []
    for path in _candidate_files():
        internal, external = _imports(path)
        status, reasons = _status(path, internal)
        rows.append(
            SplitAuditRow(
                module=_module_name(path),
                path=str(path.relative_to(REPO_ROOT)),
                status=status,
                reasons=reasons,
                internal_imports=internal,
                external_import_roots=external,
            )
        )
    counts = {status: sum(1 for row in rows if row.status == status) for status in _STATUSES}
    return {
        "schema": "s6_quantum_kuramoto_split_audit_v1",
        "date": DATE,
        "candidate_packages": list(CANDIDATE_PACKAGES),
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
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the quantum-Kuramoto split audit and write public artefacts."""

    args = parse_args()
    payload = build_split_audit()
    json_path = args.out_dir / f"quantum_kuramoto_split_audit_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, _markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
