#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM IBM pilot protocol generator
"""Prepare a non-submitting IBM pilot protocol for the SCPN/FIM paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
DATE = "2026-05-05"

STATE_BY_MAGNETISATION = {
    4: "0000",
    2: "0001",
    0: "0011",
    -2: "0111",
    -4: "1111",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(lambdas: list[float], depths: list[int], shots: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for lambda_fim in lambdas:
        for depth in depths:
            for magnetisation, bitstring in STATE_BY_MAGNETISATION.items():
                rows.append(
                    {
                        "protocol_arm": "fim_sector_survival_pilot",
                        "n_qubits": 4,
                        "lambda_fim": float(lambda_fim),
                        "depth": depth,
                        "initial_bitstring": bitstring,
                        "magnetisation": magnetisation,
                        "popcount": bitstring.count("1"),
                        "shots": shots,
                        "submission_status": "not_submitted",
                    }
                )
    readout_rows = [
        {
            "protocol_arm": "readout_baseline",
            "n_qubits": 4,
            "lambda_fim": None,
            "depth": 0,
            "initial_bitstring": format(index, "04b"),
            "magnetisation": 4 - 2 * format(index, "04b").count("1"),
            "popcount": format(index, "04b").count("1"),
            "shots": shots,
            "submission_status": "not_submitted",
        }
        for index in range(16)
    ]
    rows.extend(readout_rows)
    total_circuits = len(rows)
    return {
        "schema": "scpn_fim_ibm_candidate_protocol_v1",
        "date": DATE,
        "command": "python scripts/prepare_fim_ibm_pilot.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "submission_status": "not_submitted",
        "requires_user_approval_before_qpu": True,
        "n_qubits": 4,
        "lambda_values": lambdas,
        "depth_values": depths,
        "shots_per_circuit": shots,
        "total_circuits": total_circuits,
        "total_shots": total_circuits * shots,
        "live_gate_before_submission": [
            "select backend and calibration window",
            "live transpilation depth and two-qubit gate count check",
            "layout and readout calibration capture",
            "QPU-time estimate",
            "explicit user approval",
        ],
        "scientific_boundary": (
            "This is a candidate protocol only. The ideal H_XY + H_FIM model "
            "conserves magnetisation; the pilot tests whether hardware execution "
            "shows lambda-correlated survival or leakage differences after equal-depth, "
            "popcount, layout, and readout controls."
        ),
        "falsification_rule": (
            "If lambda-dependent survival differences are absent, inconsistent in sign, "
            "or dominated by readout/layout/popcount controls, the paper must retain "
            "offline-only FIM claims and report the hardware protocol as falsified or "
            "inconclusive."
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", default="0,1,4")
    parser.add_argument("--depths", default="2,4,6")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    lambdas = _parse_csv_floats(ns.lambdas)
    depths = _parse_csv_ints(ns.depths)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(lambdas, depths, ns.shots)

    json_path = ns.output_dir / f"fim_ibm_candidate_protocol_{DATE}.json"
    csv_path = ns.output_dir / f"fim_ibm_candidate_protocol_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(summary["rows"]))
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
