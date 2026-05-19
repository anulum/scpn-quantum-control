#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S3 design readiness export
"""Export S3 pulse/ansatz design-readiness artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scpn_quantum_control.benchmarks.s3_design_protocol import (
    default_s3_design_protocol,
    score_s3_candidates,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s3_pulse_ansatz_design"
DATE = "2026-05-06"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown(summary: dict[str, object]) -> str:
    protocol = summary["protocol"]
    if not isinstance(protocol, dict):
        raise TypeError("protocol must be a mapping")
    rows = summary["rows"]
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    lines = [
        "# S3 Pulse / Ansatz Design Readiness",
        "",
        f"Protocol ID: `{summary['protocol_id']}`",
        "",
        "Submission state: no hardware submission; deterministic candidate scoring only.",
        "",
        "## Objective",
        str(protocol["objective"]),
        "",
        "## Candidate Scores",
        "",
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("row must be a mapping")
        metrics = row["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("row metrics must be a mapping")
        lines.append(
            f"- `{row['candidate_label']}` ({row['family']}): score={row['score']}; "
            f"metrics={json.dumps(metrics, sort_keys=True)}"
        )
    lines.extend(["", "## Forbidden Claims"])
    forbidden = protocol["forbidden_claims"]
    if not isinstance(forbidden, list):
        raise TypeError("forbidden_claims must be a list")
    lines.extend(f"- {claim}" for claim in forbidden)
    lines.extend(["", "## Required Follow-ups"])
    followups = protocol["required_followups"]
    if not isinstance(followups, list):
        raise TypeError("required_followups must be a list")
    lines.extend(f"- {item}" for item in followups)
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the S3 design-readiness artefacts."""
    args = _parse_args()
    if args.n_qubits < 2:
        raise ValueError("--n-qubits must be at least 2")
    protocol = default_s3_design_protocol()
    k_matrix = build_knm_paper27(args.n_qubits)
    omega = np.asarray(OMEGA_N_16[: args.n_qubits], dtype=np.float64)
    rows = score_s3_candidates(protocol, k_matrix, omega)
    summary = {
        "date": DATE,
        "protocol_id": protocol.protocol_id,
        "script": "scripts/export_s3_design_readiness.py",
        "hardware_submission": False,
        "ml_training_performed": False,
        "n_qubits": args.n_qubits,
        "protocol": protocol.to_dict(),
        "rows": [row.to_dict() for row in rows],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"s3_design_readiness_{DATE}.json"
    md_path = args.out_dir / f"s3_design_readiness_{DATE}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
