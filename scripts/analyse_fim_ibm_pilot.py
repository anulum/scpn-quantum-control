#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- SCPN/FIM IBM pilot analysis
"""Analyse the SCPN/FIM n=4 IBM pilot raw counts.

This is intentionally a descriptive pilot analysis. The submitted campaign has
one hardware sample for each lambda/depth/state condition, plus readout-only
calibrations for all 16 basis states. It can detect gross implementation
failures and guide the next campaign, but it cannot support formal p-values or
publication-grade hardware protection claims without repeated randomized
hardware samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_ID = "d7t53ofljm6s73bc6bj0"
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "scpn_fim_hamiltonian"
    / f"fim_ibm_pilot_raw_counts_2026-05-05_{JOB_ID}.json"
)
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
SUMMARY_OUT = OUT_DIR / f"fim_ibm_pilot_analysis_2026-05-05_{JOB_ID}.json"
ROW_CSV_OUT = OUT_DIR / f"fim_ibm_pilot_row_metrics_2026-05-05_{JOB_ID}.csv"
TREND_CSV_OUT = OUT_DIR / f"fim_ibm_pilot_lambda_trends_2026-05-05_{JOB_ID}.csv"
PUBLISHED_SHA256 = "be284b9b2f71dfecd978703d979a8893e79b35dcc4537d7a372b83ba48305790"


@dataclass(frozen=True)
class RowMetric:
    circuit_index: int
    protocol_arm: str
    n_qubits: int
    lambda_fim: float | None
    depth: int | None
    initial_bitstring: str
    magnetisation: int
    popcount: int
    shots: int
    state_retention: float
    magnetisation_leakage: float
    parity_leakage: float
    mean_output_magnetisation: float
    readout_state_retention: float | None
    readout_magnetisation_leakage: float | None
    readout_parity_leakage: float | None
    parity_leakage_readout_corrected: float | None
    live_transpiled_depth: int | None
    live_two_qubit_gates: int | None


@dataclass(frozen=True)
class LambdaTrend:
    initial_bitstring: str
    magnetisation: int
    popcount: int
    depth: int
    lambda_left: float
    lambda_right: float
    state_retention_delta_right_minus_left: float
    magnetisation_leakage_delta_right_minus_left: float
    parity_leakage_delta_right_minus_left: float
    corrected_parity_leakage_delta_right_minus_left: float | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(bitstring: str) -> str:
    return bitstring.replace(" ", "")


def _total(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def _popcount(bitstring: str) -> int:
    return _clean(bitstring).count("1")


def _magnetisation(bitstring: str) -> int:
    clean = _clean(bitstring)
    return len(clean) - 2 * clean.count("1")


def _parity(bitstring: str) -> int:
    return _popcount(bitstring) % 2


def _state_retention(counts: dict[str, int], initial: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    return counts.get(initial[::-1], 0) / total


def _magnetisation_leakage(counts: dict[str, int], initial: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    target = _magnetisation(initial)
    leaked = sum(
        count for bitstring, count in counts.items() if _magnetisation(bitstring) != target
    )
    return leaked / total


def _parity_leakage(counts: dict[str, int], initial: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    target = _parity(initial)
    leaked = sum(count for bitstring, count in counts.items() if _parity(bitstring) != target)
    return leaked / total


def _mean_output_magnetisation(counts: dict[str, int]) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    return sum(_magnetisation(bitstring) * count for bitstring, count in counts.items()) / total


def _correct_parity_leakage(observed: float, readout_flip: float) -> float | None:
    denominator = 1.0 - 2.0 * readout_flip
    if denominator <= 0.0:
        return None
    return min(1.0, max(0.0, (observed - readout_flip) / denominator))


def _two_qubit_gates(ops: dict[str, int] | None) -> int | None:
    if ops is None:
        return None
    return int(sum(value for name, value in ops.items() if name in {"cx", "cz", "ecr"}))


def _row_metric(
    row: dict[str, Any],
    readout: dict[str, dict[str, float]],
) -> RowMetric:
    meta = row["metadata"]
    counts = {str(key): int(value) for key, value in row["counts"].items()}
    initial = str(meta["initial_bitstring"])
    shots = _total(counts)
    baseline = readout.get(initial)
    raw_parity_leakage = _parity_leakage(counts, initial)
    readout_parity = None if baseline is None else baseline["parity_leakage"]
    corrected = (
        None
        if readout_parity is None
        else _correct_parity_leakage(raw_parity_leakage, readout_parity)
    )
    return RowMetric(
        circuit_index=int(meta["circuit_index"]),
        protocol_arm=str(meta["protocol_arm"]),
        n_qubits=int(meta["n_qubits"]),
        lambda_fim=None if meta.get("lambda_fim") is None else float(meta["lambda_fim"]),
        depth=None if meta.get("depth") is None else int(meta["depth"]),
        initial_bitstring=initial,
        magnetisation=int(meta["magnetisation"]),
        popcount=int(meta["popcount"]),
        shots=shots,
        state_retention=_state_retention(counts, initial),
        magnetisation_leakage=_magnetisation_leakage(counts, initial),
        parity_leakage=raw_parity_leakage,
        mean_output_magnetisation=_mean_output_magnetisation(counts),
        readout_state_retention=None if baseline is None else baseline["state_retention"],
        readout_magnetisation_leakage=None
        if baseline is None
        else baseline["magnetisation_leakage"],
        readout_parity_leakage=readout_parity,
        parity_leakage_readout_corrected=corrected,
        live_transpiled_depth=(
            None
            if meta.get("live_transpiled_depth") is None
            else int(meta["live_transpiled_depth"])
        ),
        live_two_qubit_gates=_two_qubit_gates(meta.get("live_transpiled_ops")),
    )


def _build_readout(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    readout: dict[str, dict[str, float]] = {}
    for row in rows:
        meta = row["metadata"]
        if meta.get("protocol_arm") != "readout_baseline":
            continue
        initial = str(meta["initial_bitstring"])
        counts = {str(key): int(value) for key, value in row["counts"].items()}
        readout[initial] = {
            "state_retention": _state_retention(counts, initial),
            "magnetisation_leakage": _magnetisation_leakage(counts, initial),
            "parity_leakage": _parity_leakage(counts, initial),
        }
    return readout


def _trend_rows(metrics: list[RowMetric]) -> list[LambdaTrend]:
    main = [row for row in metrics if row.protocol_arm == "fim_sector_survival_pilot"]
    by_condition: dict[tuple[str, int, float], RowMetric] = {}
    for row in main:
        if row.depth is None or row.lambda_fim is None:
            continue
        by_condition[(row.initial_bitstring, row.depth, row.lambda_fim)] = row

    trends: list[LambdaTrend] = []
    for initial, depth, left_lambda in sorted(by_condition):
        if left_lambda != 0.0:
            continue
        left = by_condition[(initial, depth, left_lambda)]
        for right_lambda in (1.0, 4.0):
            right = by_condition.get((initial, depth, right_lambda))
            if right is None:
                continue
            corrected_delta = None
            if (
                left.parity_leakage_readout_corrected is not None
                and right.parity_leakage_readout_corrected is not None
            ):
                corrected_delta = (
                    right.parity_leakage_readout_corrected - left.parity_leakage_readout_corrected
                )
            trends.append(
                LambdaTrend(
                    initial_bitstring=initial,
                    magnetisation=left.magnetisation,
                    popcount=left.popcount,
                    depth=depth,
                    lambda_left=left_lambda,
                    lambda_right=right_lambda,
                    state_retention_delta_right_minus_left=right.state_retention
                    - left.state_retention,
                    magnetisation_leakage_delta_right_minus_left=right.magnetisation_leakage
                    - left.magnetisation_leakage,
                    parity_leakage_delta_right_minus_left=right.parity_leakage
                    - left.parity_leakage,
                    corrected_parity_leakage_delta_right_minus_left=corrected_delta,
                )
            )
    return trends


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(mean(clean))


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["result_rows"]
    readout = _build_readout(rows)
    metrics = [_row_metric(row, readout) for row in rows]
    trends = _trend_rows(metrics)
    main = [row for row in metrics if row.protocol_arm == "fim_sector_survival_pilot"]
    l4_trends = [row for row in trends if row.lambda_right == 4.0]
    l1_trends = [row for row in trends if row.lambda_right == 1.0]

    return {
        "analysis": "fim_ibm_pilot_descriptive",
        "backend": payload.get("backend"),
        "job_id": payload.get("job_id"),
        "created_utc": payload.get("created_utc"),
        "completed_utc": payload.get("completed_utc"),
        "n_result_rows": len(rows),
        "n_main_rows": len(main),
        "n_readout_rows": len(readout),
        "shots_total": sum(row.shots for row in metrics),
        "main_shots_total": sum(row.shots for row in main),
        "readout_basis_states": sorted(readout),
        "main_depths": sorted({row.depth for row in main if row.depth is not None}),
        "main_lambdas": sorted({row.lambda_fim for row in main if row.lambda_fim is not None}),
        "live_depth_max": max(row.live_transpiled_depth or 0 for row in metrics),
        "live_two_qubit_gate_max": max(row.live_two_qubit_gates or 0 for row in metrics),
        "readout_summary": {
            "mean_state_retention": float(
                mean(row["state_retention"] for row in readout.values())
            ),
            "mean_magnetisation_leakage": float(
                mean(row["magnetisation_leakage"] for row in readout.values())
            ),
            "mean_parity_leakage": float(mean(row["parity_leakage"] for row in readout.values())),
        },
        "lambda_1_vs_0": {
            "n_comparisons": len(l1_trends),
            "mean_state_retention_delta": _mean_or_none(
                [row.state_retention_delta_right_minus_left for row in l1_trends]
            ),
            "mean_magnetisation_leakage_delta": _mean_or_none(
                [row.magnetisation_leakage_delta_right_minus_left for row in l1_trends]
            ),
            "mean_parity_leakage_delta": _mean_or_none(
                [row.parity_leakage_delta_right_minus_left for row in l1_trends]
            ),
            "n_lower_magnetisation_leakage": sum(
                row.magnetisation_leakage_delta_right_minus_left < 0.0 for row in l1_trends
            ),
            "n_higher_magnetisation_leakage": sum(
                row.magnetisation_leakage_delta_right_minus_left > 0.0 for row in l1_trends
            ),
        },
        "lambda_4_vs_0": {
            "n_comparisons": len(l4_trends),
            "mean_state_retention_delta": _mean_or_none(
                [row.state_retention_delta_right_minus_left for row in l4_trends]
            ),
            "mean_magnetisation_leakage_delta": _mean_or_none(
                [row.magnetisation_leakage_delta_right_minus_left for row in l4_trends]
            ),
            "mean_parity_leakage_delta": _mean_or_none(
                [row.parity_leakage_delta_right_minus_left for row in l4_trends]
            ),
            "mean_corrected_parity_leakage_delta": _mean_or_none(
                [row.corrected_parity_leakage_delta_right_minus_left for row in l4_trends]
            ),
            "n_lower_magnetisation_leakage": sum(
                row.magnetisation_leakage_delta_right_minus_left < 0.0 for row in l4_trends
            ),
            "n_higher_magnetisation_leakage": sum(
                row.magnetisation_leakage_delta_right_minus_left > 0.0 for row in l4_trends
            ),
        },
        "claim_boundary": {
            "outcome": "descriptive_pilot_only",
            "supports": [
                "The submitted circuits executed and returned complete count dictionaries.",
                "The data are sufficient for sanity checks, readout-baseline comparison, and planning repeated randomized campaigns.",
            ],
            "does_not_support": [
                "No formal hardware p-values, because there is one sample per lambda/depth/state condition.",
                "No claim that the FIM term improves hardware coherence.",
                "No claim of many-body localisation on hardware.",
                "No full confusion-matrix readout mitigation; only basis-state readout baselines and parity-flip correction are available.",
            ],
        },
        "row_metrics": [asdict(row) for row in metrics],
        "lambda_trends": [asdict(row) for row in trends],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--verify-integrity", action="store_true")
    args = parser.parse_args()

    if args.verify_integrity:
        actual = _sha256(args.input)
        if actual != PUBLISHED_SHA256:
            raise SystemExit(f"SHA-256 mismatch: {actual} != {PUBLISHED_SHA256}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = _summarise(payload)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(ROW_CSV_OUT, summary["row_metrics"])
    _write_csv(TREND_CSV_OUT, summary["lambda_trends"])

    print("SCPN/FIM IBM pilot descriptive analysis")
    print(f"  backend: {summary['backend']}")
    print(f"  job ID:  {summary['job_id']}")
    print(f"  rows:    {summary['n_result_rows']}")
    print(f"  output:  {SUMMARY_OUT}")
    print(f"  outcome: {summary['claim_boundary']['outcome']}")
    print(
        "  lambda=4 vs lambda=0 mean magnetisation-leakage delta: "
        f"{summary['lambda_4_vs_0']['mean_magnetisation_leakage_delta']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
