#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analyse FIM IBM repeated followup script
# scpn-quantum-control -- repeated SCPN/FIM IBM follow-up analysis
"""Analyse the repeated/randomized SCPN/FIM IBM follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, cast

from scipy.stats import combine_pvalues, ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_ID = "ibm-run-cf4835290f607387"
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "scpn_fim_hamiltonian"
    / f"fim_ibm_repeated_followup_raw_counts_2026-05-05_{JOB_ID}.json"
)
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
SUMMARY_OUT = OUT_DIR / f"fim_ibm_repeated_followup_analysis_2026-05-05_{JOB_ID}.json"
ROW_CSV_OUT = OUT_DIR / f"fim_ibm_repeated_followup_row_metrics_2026-05-05_{JOB_ID}.csv"
COMPARISON_CSV_OUT = OUT_DIR / f"fim_ibm_repeated_followup_comparisons_2026-05-05_{JOB_ID}.csv"
PUBLISHED_SHA256 = "6e4df78f1c679cd29b9c503bf1fecf39be76e707b1e6c4df99bbcc87b8e50d44"


@dataclass(frozen=True)
class RowMetric:
    """Per-circuit metric row for the repeated SCPN/FIM hardware follow-up."""

    circuit_index: int
    protocol_arm: str
    lambda_fim: float | None
    depth: int | None
    initial_bitstring: str
    magnetisation: int
    popcount: int
    replicate: int
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
class Comparison:
    """Welch comparison between lambda-zero and lambda-four replicates."""

    initial_bitstring: str
    magnetisation: int
    popcount: int
    depth: int
    observable: str
    lambda_0_mean: float
    lambda_4_mean: float
    delta_lambda_4_minus_0: float
    relative_delta_to_lambda_0: float
    welch_t: float
    welch_p: float
    n_lambda_0: int
    n_lambda_4: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _total(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def _clean(bitstring: str) -> str:
    return bitstring.replace(" ", "")


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


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


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


def _row_metric(row: dict[str, Any], readout: dict[str, dict[str, float]]) -> RowMetric:
    meta = row["metadata"]
    counts = {str(key): int(value) for key, value in row["counts"].items()}
    initial = str(meta["initial_bitstring"])
    baseline = readout.get(initial)
    raw_parity = _parity_leakage(counts, initial)
    readout_parity = None if baseline is None else baseline["parity_leakage"]
    corrected = (
        None if readout_parity is None else _correct_parity_leakage(raw_parity, readout_parity)
    )
    return RowMetric(
        circuit_index=int(meta["circuit_index"]),
        protocol_arm=str(meta["protocol_arm"]),
        lambda_fim=None if meta.get("lambda_fim") is None else float(meta["lambda_fim"]),
        depth=None if meta.get("depth") is None else int(meta["depth"]),
        initial_bitstring=initial,
        magnetisation=int(meta["magnetisation"]),
        popcount=int(meta["popcount"]),
        replicate=int(meta.get("replicate", 0)),
        shots=_total(counts),
        state_retention=_state_retention(counts, initial),
        magnetisation_leakage=_magnetisation_leakage(counts, initial),
        parity_leakage=raw_parity,
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


def _observable(row: RowMetric, name: str) -> float | None:
    return cast(float | None, getattr(row, name))


def _comparison(
    left: list[RowMetric],
    right: list[RowMetric],
    observable: str,
) -> Comparison | None:
    left_values = [_observable(row, observable) for row in left]
    right_values = [_observable(row, observable) for row in right]
    if any(value is None for value in left_values + right_values):
        return None
    left_clean = [float(value) for value in left_values if value is not None]
    right_clean = [float(value) for value in right_values if value is not None]
    if len(left_clean) < 2 or len(right_clean) < 2:
        return None
    welch = ttest_ind(left_clean, right_clean, equal_var=False)
    left_mean = float(mean(left_clean))
    right_mean = float(mean(right_clean))
    denominator = left_mean if abs(left_mean) > 1e-15 else 1.0
    ref = left[0]
    return Comparison(
        initial_bitstring=ref.initial_bitstring,
        magnetisation=ref.magnetisation,
        popcount=ref.popcount,
        depth=int(ref.depth or 0),
        observable=observable,
        lambda_0_mean=left_mean,
        lambda_4_mean=right_mean,
        delta_lambda_4_minus_0=right_mean - left_mean,
        relative_delta_to_lambda_0=(right_mean - left_mean) / denominator,
        welch_t=float(welch.statistic),
        welch_p=float(welch.pvalue),
        n_lambda_0=len(left_clean),
        n_lambda_4=len(right_clean),
    )


def _comparisons(metrics: list[RowMetric]) -> list[Comparison]:
    main = [row for row in metrics if row.protocol_arm == "fim_repeated_followup"]
    grouped: dict[tuple[str, int, float], list[RowMetric]] = {}
    for row in main:
        if row.depth is None or row.lambda_fim is None:
            continue
        grouped.setdefault((row.initial_bitstring, row.depth, row.lambda_fim), []).append(row)
    comparisons: list[Comparison] = []
    for initial, depth, lambda_fim in sorted(grouped):
        if lambda_fim != 0.0:
            continue
        left = grouped[(initial, depth, lambda_fim)]
        right = grouped.get((initial, depth, 4.0))
        if right is None:
            continue
        for observable in (
            "state_retention",
            "magnetisation_leakage",
            "parity_leakage",
            "parity_leakage_readout_corrected",
        ):
            comparison = _comparison(left, right, observable)
            if comparison is not None:
                comparisons.append(comparison)
    return comparisons


def _fisher(comparisons: list[Comparison], observable: str) -> dict[str, float | int]:
    selected = [row for row in comparisons if row.observable == observable]
    if not selected:
        return {"n_tests": 0}
    chi2, p_value = combine_pvalues([row.welch_p for row in selected], method="fisher")
    return {
        "n_tests": len(selected),
        "chi2": float(chi2),
        "p": float(p_value),
        "n_delta_positive": sum(row.delta_lambda_4_minus_0 > 0.0 for row in selected),
        "n_delta_negative": sum(row.delta_lambda_4_minus_0 < 0.0 for row in selected),
        "mean_delta": float(mean(row.delta_lambda_4_minus_0 for row in selected)),
        "sem_delta": float(_sem([row.delta_lambda_4_minus_0 for row in selected])),
    }


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["result_rows"]
    readout = _build_readout(rows)
    metrics = [_row_metric(row, readout) for row in rows]
    comparisons = _comparisons(metrics)
    main = [row for row in metrics if row.protocol_arm == "fim_repeated_followup"]
    readout_rows = [row for row in metrics if row.protocol_arm == "readout_baseline"]
    return {
        "analysis": "fim_ibm_repeated_followup",
        "backend": payload.get("backend"),
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "total_circuits": payload.get("total_circuits"),
        "shots_per_circuit": payload.get("shots_per_circuit"),
        "total_shots": payload.get("total_shots"),
        "wait_wall_time_s": payload.get("wait_wall_time_s"),
        "n_main_rows": len(main),
        "n_readout_rows": len(readout_rows),
        "max_live_transpiled_depth": max(row.live_transpiled_depth or 0 for row in metrics),
        "max_live_two_qubit_gates": max(row.live_two_qubit_gates or 0 for row in metrics),
        "readout_summary": {
            "mean_state_retention": float(
                mean(row["state_retention"] for row in readout.values())
            ),
            "mean_magnetisation_leakage": float(
                mean(row["magnetisation_leakage"] for row in readout.values())
            ),
            "mean_parity_leakage": float(mean(row["parity_leakage"] for row in readout.values())),
        },
        "fisher_by_observable": {
            observable: _fisher(comparisons, observable)
            for observable in (
                "state_retention",
                "magnetisation_leakage",
                "parity_leakage",
                "parity_leakage_readout_corrected",
            )
        },
        "claim_boundary": {
            "outcome": "repeated_hardware_followup_completed",
            "primary_interpretation_rule": (
                "A positive magnetisation-leakage delta means lambda=4 leaked more than "
                "lambda=0. This falsifies the simple hardware-protection interpretation "
                "for this backend/circuit family."
            ),
            "blocked_claims": [
                "No backend-general FIM protection claim.",
                "No hardware many-body-localisation claim.",
                "No full confusion-matrix mitigation claim.",
            ],
        },
        "row_metrics": [asdict(row) for row in metrics],
        "comparisons": [asdict(row) for row in comparisons],
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
    """Analyse repeated follow-up raw counts and write summary artefacts."""
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
    _write_csv(COMPARISON_CSV_OUT, summary["comparisons"])
    mag = summary["fisher_by_observable"]["magnetisation_leakage"]
    print("SCPN/FIM repeated IBM follow-up analysis")
    print(f"  backend: {summary['backend']}")
    print(f"  job ID:  {summary['job_id']}")
    print(f"  output:  {SUMMARY_OUT}")
    print(f"  magnetisation leakage mean delta: {mag['mean_delta']:.6f}")
    print(f"  magnetisation leakage Fisher p:   {mag['p']:.6e}")
    print(f"  positive deltas: {mag['n_delta_positive']}/{mag['n_tests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
