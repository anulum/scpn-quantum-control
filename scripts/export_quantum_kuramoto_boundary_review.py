# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — export quantum kuramoto boundary review script
# scpn-quantum-control -- S6 quantum-kuramoto boundary review
"""Export the manual S6 boundary review and proposed public API surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DATE = "2026-05-07"
REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPO_ROOT / "data" / "s6_quantum_kuramoto_split" / f"quantum_kuramoto_split_audit_{DATE}.json"
)
OUT_DIR = REPO_ROOT / "data" / "s6_quantum_kuramoto_split"
DOC_PATH = REPO_ROOT / "docs" / f"quantum_kuramoto_boundary_review_{DATE}.md"

PUBLIC_API_MODULES = (
    "scpn_quantum_control.phase.xy_kuramoto",
    "scpn_quantum_control.phase.xy_compiler",
    "scpn_quantum_control.phase.trotter_error",
    "scpn_quantum_control.phase.structured_ansatz",
    "scpn_quantum_control.phase.phase_vqe",
    "scpn_quantum_control.phase.pulse_shaping",
    "scpn_quantum_control.phase.lindblad",
    "scpn_quantum_control.phase.mps_evolution",
    "scpn_quantum_control.hardware.backends",
    "scpn_quantum_control.hardware.async_runner",
    "scpn_quantum_control.hardware.qubit_mapper",
    "scpn_quantum_control.hardware.qasm_export",
    "scpn_quantum_control.hardware.circuit_export",
    "scpn_quantum_control.hardware.analog_kuramoto",
    "scpn_quantum_control.hardware.plugin_registry",
    "scpn_quantum_control.accel.rust_import",
)

NEEDS_REVIEW_DECISIONS: dict[str, dict[str, str]] = {
    "scpn_quantum_control.phase.backend_selector": {
        "decision": "defer",
        "reason": "depends on analysis-sector helpers outside the proposed lightweight package",
        "required_refactor": "move sector summaries behind optional adapter or duplicate minimal selector logic",
    },
    "scpn_quantum_control.bridge.sparse_hamiltonian": {
        "decision": "defer",
        "reason": "depends on analysis.magnetisation_sectors",
        "required_refactor": "extract magnetisation-sector primitive into the reusable boundary",
    },
    "scpn_quantum_control.hardware.hybrid_digital_analog": {
        "decision": "promote_after_facade",
        "reason": "scientifically reusable, but imports kuramoto_core facade from the parent package",
        "required_refactor": "route through quantum_kuramoto public facade after skeleton exists",
    },
    "scpn_quantum_control.hardware.provenance": {
        "decision": "defer",
        "reason": "depends on parent package configuration semantics",
        "required_refactor": "define package-local provenance config or keep provenance in parent package",
    },
    "scpn_quantum_control.hardware.runner": {
        "decision": "defer",
        "reason": "depends on parent config, logging setup, provenance, and mitigation modules",
        "required_refactor": "start with AsyncHardwareRunner and add full runner only after config/provenance split",
    },
}


def _load_audit(path: Path = AUDIT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S6 split audit must be a JSON object")
    return payload


def build_boundary_review(audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the manual boundary-review payload."""
    payload = _load_audit() if audit is None else audit
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("S6 split audit must contain rows")
    by_module = {row["module"]: row for row in rows if isinstance(row, dict) and "module" in row}
    missing_api = [module for module in PUBLIC_API_MODULES if module not in by_module]
    if missing_api:
        raise ValueError(f"public API modules missing from audit: {missing_api}")
    api_surface = [
        {
            "module": module,
            "current_status": by_module[module]["status"],
            "proposed_export": module.replace("scpn_quantum_control", "quantum_kuramoto", 1),
        }
        for module in PUBLIC_API_MODULES
    ]
    review_rows = []
    for module, decision in NEEDS_REVIEW_DECISIONS.items():
        row = by_module.get(module)
        if row is None:
            raise ValueError(f"needs-review module missing from audit: {module}")
        review_rows.append({"module": module, "audit_status": row["status"], **decision})
    return {
        "schema": "s6_quantum_kuramoto_boundary_review_v1",
        "date": DATE,
        "source_audit": str(AUDIT_PATH.relative_to(REPO_ROOT)),
        "package_skeleton_allowed": False,
        "reason": "manual boundary review still requires refactors for config/provenance/analysis-dependent rows",
        "proposed_public_api": api_surface,
        "needs_review_decisions": review_rows,
        "compatibility_requirements": [
            "existing scpn_quantum_control imports must remain unchanged",
            "quantum_kuramoto must not import SCPN-specific SSGF/SNN/orchestrator modules",
            "runner skeleton must start with async/backend abstractions before parent config is split",
            "separate pyproject is allowed only after import-compatibility tests pass",
        ],
        "next_steps": [
            "add API-surface contract tests for proposed exports",
            "extract or wrap magnetisation-sector primitive if sparse_hamiltonian is promoted",
            "define package-local config/provenance policy before promoting hardware.runner",
            "create skeleton only after the above blockers are closed",
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# S6 Quantum-Kuramoto Boundary Review",
        "",
        "This review converts the import-graph audit into a conservative package-boundary decision. It does not create a `quantum_kuramoto` package skeleton.",
        "",
        "## Decision",
        f"- Package skeleton allowed: `{payload['package_skeleton_allowed']}`",
        f"- Reason: {payload['reason']}",
        "",
        "## Proposed Public API Surface",
    ]
    lines.extend(
        f"- `{row['proposed_export']}` from `{row['module']}` (`{row['current_status']}`)"
        for row in payload["proposed_public_api"]
    )
    lines.extend(["", "## Needs-Review Decisions"])
    lines.extend(
        f"- `{row['module']}` — `{row['decision']}` — {row['reason']} — refactor: {row['required_refactor']}"
        for row in payload["needs_review_decisions"]
    )
    lines.extend(["", "## Compatibility Requirements"])
    lines.extend(f"- {item}" for item in payload["compatibility_requirements"])
    lines.extend(["", "## Next Steps"])
    lines.extend(f"- {item}" for item in payload["next_steps"])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse the quantum Kuramoto boundary-review export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the quantum Kuramoto boundary-review artefact."""
    args = parse_args()
    audit = _load_audit(args.audit_path)
    payload = build_boundary_review(audit)
    json_path = args.out_dir / f"quantum_kuramoto_boundary_review_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, _markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
