#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S2 claim-boundary report
"""Generate claim-boundary report for S2 scaling rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmarks.advantage_protocol import (
    default_s2_scaling_protocol,
    validate_scaling_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DATE = "2026-05-06"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=OUT_DIR / f"s2_scaling_lite_rows_{DATE}.json",
        help="S2 rows JSON file.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    raise ValueError("rows payload must be a list or contain a rows list")


def build_claim_boundary_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a claim-boundary report for validated S2 rows."""
    protocol = default_s2_scaling_protocol()
    validation = validate_scaling_rows(protocol, rows)
    statuses = {str(row.get("baseline")): str(row.get("status")) for row in rows}
    has_hardware_ok = any(
        row.get("baseline") == "qpu_hardware" and row.get("status") == "ok" for row in rows
    )
    has_mps_ok = any(
        row.get("baseline") == "mps_tensor_network" and row.get("status") == "ok" for row in rows
    )
    allowed = [
        "Report that the S2 protocol, schema, and lite no-QPU rehearsal path are operational.",
        "Report measured small-size classical ODE and dense exact-diagonalisation timings where rows are ok.",
        "Report explicit skipped rows as coverage of the validation contract, not as performance data.",
    ]
    forbidden = [
        "Do not claim broad quantum advantage from lite rows.",
        "Do not claim hardware scaling because no QPU rows are present.",
        "Do not claim tensor-network hardness unless MPS/TN rows are measured and non-spoofable.",
        "Do not extrapolate skipped rows into crossover estimates.",
    ]
    blockers = [
        "Run full required MPS/TN baseline rows.",
        "Run Aer/statevector rows for the selected size grid.",
        "Measure sparse eigensolver rows or record justified size-gated failures.",
        "Add hardware rows only after preregistration, QPU approval, raw-count storage, and validation gates.",
    ]
    if not validation.valid:
        forbidden.append("Do not promote figures or tables until row validation passes.")
        blockers.append("Fix validation failures before any S2 promotion.")
    if has_hardware_ok:
        allowed.append("Report hardware rows only for the exact preregistered sizes present.")
    if has_mps_ok:
        allowed.append("Discuss MPS/TN spoofability only for measured MPS/TN rows.")
    return {
        "date": DATE,
        "protocol_id": protocol.protocol_id,
        "validation": validation.to_dict(),
        "baseline_statuses": statuses,
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
        "remaining_blockers": blockers,
        "protocol_claim_boundary": protocol.claim_boundary,
        "hardware_submission": False,
        "advantage_claim": False,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# S2 Scaling Claim-boundary Report",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        f"Validation: `{report['validation']['valid']}`",
        "",
        "## Allowed Claims",
        *[f"- {item}" for item in report["allowed_claims"]],
        "",
        "## Forbidden Claims",
        *[f"- {item}" for item in report["forbidden_claims"]],
        "",
        "## Remaining Blockers",
        *[f"- {item}" for item in report["remaining_blockers"]],
        "",
        "## Protocol Claim Boundary",
        str(report["protocol_claim_boundary"]),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(args.rows.read_text(encoding="utf-8"))
    report = build_claim_boundary_report(_rows_from_payload(payload))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"s2_scaling_claim_boundary_{DATE}.json"
    md_path = args.out_dir / f"s2_scaling_claim_boundary_{DATE}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0 if report["validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
