#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- repeated SCPN/FIM IBM follow-up protocol
"""Prepare a repeated/randomized SCPN/FIM IBM follow-up protocol.

The first IBM pilot was valid but descriptive only: one sample per
lambda/depth/state condition and no positive hardware-protection signal. This
script prepares the next statistically meaningful step without submitting it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
DATE = "2026-05-05"
DEFAULT_SEED = 2026050503
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


def generate(
    lambdas: list[float],
    depths: list[int],
    reps: int,
    shots: int,
    seed: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for rep in range(reps):
        for lambda_fim in lambdas:
            for depth in depths:
                for magnetisation, bitstring in STATE_BY_MAGNETISATION.items():
                    rows.append(
                        {
                            "protocol_arm": "fim_repeated_followup",
                            "n_qubits": 4,
                            "lambda_fim": float(lambda_fim),
                            "depth": int(depth),
                            "initial_bitstring": bitstring,
                            "magnetisation": int(magnetisation),
                            "popcount": bitstring.count("1"),
                            "replicate": rep,
                            "shots": shots,
                            "submission_status": "not_submitted",
                        }
                    )

    rng = random.Random(seed)
    rng.shuffle(rows)
    for circuit_index, row in enumerate(rows):
        row["randomized_order"] = circuit_index

    readout_rows = [
        {
            "protocol_arm": "readout_baseline",
            "n_qubits": 4,
            "lambda_fim": None,
            "depth": 0,
            "initial_bitstring": format(index, "04b"),
            "magnetisation": 4 - 2 * format(index, "04b").count("1"),
            "popcount": format(index, "04b").count("1"),
            "replicate": 0,
            "shots": shots,
            "randomized_order": len(rows) + index,
            "submission_status": "not_submitted",
        }
        for index in range(16)
    ]
    rows.extend(readout_rows)
    total_circuits = len(rows)
    return {
        "schema": "scpn_fim_ibm_repeated_followup_protocol_v1",
        "date": DATE,
        "command": "python scripts/prepare_fim_ibm_repeated_followup.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "submission_status": "not_submitted",
        "requires_live_backend_transpile_before_qpu": True,
        "requires_user_approval_before_qpu": True,
        "n_qubits": 4,
        "randomization_seed": seed,
        "lambda_values": lambdas,
        "depth_values": depths,
        "replicates_per_condition": reps,
        "shots_per_circuit": shots,
        "main_circuits": total_circuits - len(readout_rows),
        "readout_circuits": len(readout_rows),
        "total_circuits": total_circuits,
        "total_shots": total_circuits * shots,
        "estimated_statistical_use": (
            "Repeated samples permit paired/descriptive uncertainty estimates and "
            "Welch or permutation tests per depth/state family. The design is still "
            "small enough to abort after live transpilation if depth or queue cost is poor."
        ),
        "primary_test": (
            "Compare lambda=4 against lambda=0 for magnetisation leakage and exact-state "
            "retention within matched initial state and depth, using replicate rows."
        ),
        "falsification_rule": (
            "If lambda=4 again increases leakage consistently, or if the sign is unstable "
            "after readout correction, the manuscript must report no hardware evidence for "
            "FIM coherence protection on this backend/circuit family."
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default="0,4")
    parser.add_argument("--depths", default="2,4,6")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    lambdas = _parse_csv_floats(ns.lambdas)
    depths = _parse_csv_ints(ns.depths)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(lambdas, depths, ns.reps, ns.shots, ns.seed)
    json_path = ns.output_dir / f"fim_ibm_repeated_followup_protocol_{DATE}.json"
    csv_path = ns.output_dir / f"fim_ibm_repeated_followup_protocol_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(summary["rows"]))
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    print(f"total_circuits={summary['total_circuits']}")
    print(f"total_shots={summary['total_shots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
