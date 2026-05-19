# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S6 quantum-kuramoto API contract
"""Export the S6 API-surface contract before package skeleton creation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

DATE = "2026-05-07"
REPO_ROOT = Path(__file__).resolve().parents[1]
S6_DIR = REPO_ROOT / "data" / "s6_quantum_kuramoto_split"
BOUNDARY_REVIEW_PATH = S6_DIR / f"quantum_kuramoto_boundary_review_{DATE}.json"
DOC_PATH = REPO_ROOT / "docs" / f"quantum_kuramoto_api_contract_{DATE}.md"

TARGET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
TARGET_PREFIX = "quantum_kuramoto."
BLOCKED_SOURCE_MODULES = frozenset(
    {
        "scpn_quantum_control.phase.backend_selector",
        "scpn_quantum_control.bridge.sparse_hamiltonian",
        "scpn_quantum_control.hardware.provenance",
        "scpn_quantum_control.hardware.runner",
    }
)


def _load_review(path: Path = BOUNDARY_REVIEW_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("boundary review must be a JSON object")
    return payload


def _public_member_count(module_name: str) -> tuple[bool, int, str | None]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, 0, f"{exc.__class__.__name__}: {exc}"
    return True, len([name for name in dir(module) if not name.startswith("_")]), None


def _target_valid(target: str) -> bool:
    return bool(TARGET_RE.match(target)) and target.startswith(TARGET_PREFIX)


def build_api_contract(review: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the future ``quantum_kuramoto`` API contract payload."""

    payload = _load_review() if review is None else review
    if payload.get("schema") != "s6_quantum_kuramoto_boundary_review_v1":
        raise ValueError("unexpected boundary-review schema")
    proposed = payload.get("proposed_public_api")
    if not isinstance(proposed, list):
        raise ValueError("boundary review must contain proposed_public_api")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    seen_exports: set[str] = set()
    for item in proposed:
        if not isinstance(item, dict):
            errors.append("proposed API row is not an object")
            continue
        module = str(item.get("module", ""))
        target = str(item.get("proposed_export", ""))
        status = str(item.get("current_status", ""))
        importable, public_count, import_error = _public_member_count(module)
        target_valid = _target_valid(target)
        duplicate_export = target in seen_exports
        seen_exports.add(target)
        blocked_source = module in BLOCKED_SOURCE_MODULES
        immediately_promotable = (
            importable
            and target_valid
            and not duplicate_export
            and not blocked_source
            and status == "reusable"
        )
        row = {
            "module": module,
            "proposed_export": target,
            "current_status": status,
            "importable": importable,
            "public_member_count": public_count,
            "import_error": import_error,
            "target_valid": target_valid,
            "duplicate_export": duplicate_export,
            "blocked_source": blocked_source,
            "immediately_promotable": immediately_promotable,
        }
        rows.append(row)
        if not importable:
            errors.append(f"{module} is not importable: {import_error}")
        if not target_valid:
            errors.append(f"{target} is not a valid quantum_kuramoto export")
        if duplicate_export:
            errors.append(f"{target} is declared more than once")
        if blocked_source:
            errors.append(f"{module} is blocked by the boundary review")
        if status != "reusable":
            warnings.append(
                f"{module} has status {status}; require facade or isolation before promotion"
            )

    return {
        "schema": "s6_quantum_kuramoto_api_contract_v1",
        "date": DATE,
        "source_review": str(BOUNDARY_REVIEW_PATH.relative_to(REPO_ROOT)),
        "contract_passed": not errors,
        "package_skeleton_allowed": False,
        "reason": "API targets are being contracted before any separate package skeleton is created",
        "rows": rows,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "proposed_exports": len(rows),
            "importable_exports": sum(1 for row in rows if row["importable"]),
            "immediately_promotable_exports": sum(
                1 for row in rows if row["immediately_promotable"]
            ),
            "warning_count": len(warnings),
            "error_count": len(errors),
        },
        "next_steps": [
            "extract package-local facades for non-reusable proposed rows",
            "add import-compatibility tests before skeleton creation",
            "create the skeleton only after contract_passed remains true and boundary blockers are closed",
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# S6 Quantum-Kuramoto API Contract",
        "",
        "This contract validates the proposed `quantum_kuramoto` export surface before any package skeleton is created.",
        "",
        "## Summary",
        f"- Contract passed: `{payload['contract_passed']}`",
        f"- Package skeleton allowed: `{payload['package_skeleton_allowed']}`",
        f"- Proposed exports: `{summary['proposed_exports']}`",
        f"- Importable source modules: `{summary['importable_exports']}`",
        f"- Immediately promotable exports: `{summary['immediately_promotable_exports']}`",
        f"- Warnings: `{summary['warning_count']}`",
        f"- Errors: `{summary['error_count']}`",
        "",
        "## Export Rows",
    ]
    lines.extend(
        "- `{target}` from `{module}`: importable=`{importable}`, status=`{status}`, "
        "promotable=`{promotable}`, public_members=`{members}`".format(
            target=row["proposed_export"],
            module=row["module"],
            importable=row["importable"],
            status=row["current_status"],
            promotable=row["immediately_promotable"],
            members=row["public_member_count"],
        )
        for row in payload["rows"]
    )
    if payload["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    if payload["errors"]:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in payload["errors"])
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
    """Parse the quantum Kuramoto API-contract export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-path", type=Path, default=BOUNDARY_REVIEW_PATH)
    parser.add_argument("--out-dir", type=Path, default=S6_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    return parser.parse_args()


def main() -> int:
    """Write the quantum Kuramoto API-contract artefacts."""
    args = parse_args()
    review = _load_review(args.review_path)
    payload = build_api_contract(review)
    json_path = args.out_dir / f"quantum_kuramoto_api_contract_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, _markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
