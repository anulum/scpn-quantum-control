#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analyse FIM readout matrix mitigation script
# scpn-quantum-control -- full-basis FIM readout mitigation analysis
"""Apply full-basis readout-matrix mitigation to the SCPN/FIM follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import combine_pvalues, ttest_ind

from scpn_quantum_control.mitigation.readout_matrix import (
    build_readout_confusion_matrix,
    computational_basis_labels,
    mitigate_counts,
    probability_magnetisation_leakage,
    probability_mean_magnetisation,
    probability_parity_leakage,
    probability_state_retention,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_ID = "ibm-run-cf4835290f607387"
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "scpn_fim_hamiltonian"
    / f"fim_ibm_repeated_followup_raw_counts_2026-05-05_{JOB_ID}.json"
)
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
SUMMARY_OUT = OUT_DIR / f"fim_readout_matrix_mitigation_summary_2026-05-05_{JOB_ID}.json"
ROW_CSV_OUT = OUT_DIR / f"fim_readout_matrix_mitigation_rows_2026-05-05_{JOB_ID}.csv"
COMPARISON_CSV_OUT = OUT_DIR / f"fim_readout_matrix_mitigation_comparisons_2026-05-05_{JOB_ID}.csv"
PUBLISHED_SHA256 = "6e4df78f1c679cd29b9c503bf1fecf39be76e707b1e6c4df99bbcc87b8e50d44"


@dataclass(frozen=True)
class MitigatedRow:
    """Per-circuit full-matrix readout mitigated observables."""

    circuit_index: int
    protocol_arm: str
    lambda_fim: float | None
    depth: int | None
    initial_bitstring: str
    observed_target_bitstring: str
    magnetisation: int
    popcount: int
    replicate: int
    shots: int
    state_retention_matrix_mitigated: float
    magnetisation_leakage_matrix_mitigated: float
    parity_leakage_matrix_mitigated: float
    mean_output_magnetisation_matrix_mitigated: float
    raw_state_retention: float
    raw_magnetisation_leakage: float
    raw_parity_leakage: float
    live_transpiled_depth: int | None
    live_two_qubit_gates: int | None


@dataclass(frozen=True)
class MitigatedComparison:
    """Lambda=4 minus lambda=0 comparison for a mitigated observable."""

    initial_bitstring: str
    observed_target_bitstring: str
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


def _popcount(bitstring: str) -> int:
    return bitstring.replace(" ", "").count("1")


def _magnetisation(bitstring: str) -> int:
    clean = bitstring.replace(" ", "")
    return len(clean) - 2 * clean.count("1")


def _parity(bitstring: str) -> int:
    return _popcount(bitstring) % 2


def _state_retention(counts: dict[str, int], target_observed_bitstring: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    return counts.get(target_observed_bitstring, 0) / total


def _magnetisation_leakage(counts: dict[str, int], target_observed_bitstring: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    target = _magnetisation(target_observed_bitstring)
    return (
        sum(count for bitstring, count in counts.items() if _magnetisation(bitstring) != target)
        / total
    )


def _parity_leakage(counts: dict[str, int], target_observed_bitstring: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    target = _parity(target_observed_bitstring)
    return (
        sum(count for bitstring, count in counts.items() if _parity(bitstring) != target) / total
    )


def _two_qubit_gates(ops: dict[str, int] | None) -> int | None:
    if ops is None:
        return None
    return int(sum(value for name, value in ops.items() if name in {"cx", "cz", "ecr"}))


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _calibration_counts(rows: list[dict[str, Any]], n_qubits: int) -> dict[str, dict[str, int]]:
    labels = set(computational_basis_labels(n_qubits))
    calibrations: dict[str, dict[str, int]] = {}
    for row in rows:
        meta = row["metadata"]
        if meta.get("protocol_arm") != "readout_baseline":
            continue
        prepared_observed_order = str(meta["initial_bitstring"])[::-1]
        if prepared_observed_order not in labels:
            raise ValueError(f"invalid calibration state {prepared_observed_order!r}")
        calibrations[prepared_observed_order] = {
            str(bitstring): int(count) for bitstring, count in row["counts"].items()
        }
    return calibrations


def _row_metric(row: dict[str, Any], matrix: Any) -> MitigatedRow:
    meta = row["metadata"]
    counts = {str(key): int(value) for key, value in row["counts"].items()}
    initial = str(meta["initial_bitstring"])
    target_observed = initial[::-1]
    probabilities = mitigate_counts(counts, matrix)
    return MitigatedRow(
        circuit_index=int(meta["circuit_index"]),
        protocol_arm=str(meta["protocol_arm"]),
        lambda_fim=None if meta.get("lambda_fim") is None else float(meta["lambda_fim"]),
        depth=None if meta.get("depth") is None else int(meta["depth"]),
        initial_bitstring=initial,
        observed_target_bitstring=target_observed,
        magnetisation=int(meta["magnetisation"]),
        popcount=int(meta["popcount"]),
        replicate=int(meta.get("replicate", 0)),
        shots=_total(counts),
        state_retention_matrix_mitigated=probability_state_retention(
            probabilities,
            matrix.labels,
            target_observed,
        ),
        magnetisation_leakage_matrix_mitigated=probability_magnetisation_leakage(
            probabilities,
            matrix.labels,
            target_observed,
        ),
        parity_leakage_matrix_mitigated=probability_parity_leakage(
            probabilities,
            matrix.labels,
            target_observed,
        ),
        mean_output_magnetisation_matrix_mitigated=probability_mean_magnetisation(
            probabilities,
            matrix.labels,
        ),
        raw_state_retention=_state_retention(counts, target_observed),
        raw_magnetisation_leakage=_magnetisation_leakage(counts, target_observed),
        raw_parity_leakage=_parity_leakage(counts, target_observed),
        live_transpiled_depth=(
            None
            if meta.get("live_transpiled_depth") is None
            else int(meta["live_transpiled_depth"])
        ),
        live_two_qubit_gates=_two_qubit_gates(meta.get("live_transpiled_ops")),
    )


def _comparison(
    left: list[MitigatedRow],
    right: list[MitigatedRow],
    observable: str,
) -> MitigatedComparison:
    left_values = [float(getattr(row, observable)) for row in left]
    right_values = [float(getattr(row, observable)) for row in right]
    welch = ttest_ind(left_values, right_values, equal_var=False)
    left_mean = float(mean(left_values))
    right_mean = float(mean(right_values))
    denominator = left_mean if abs(left_mean) > 1e-15 else 1.0
    ref = left[0]
    return MitigatedComparison(
        initial_bitstring=ref.initial_bitstring,
        observed_target_bitstring=ref.observed_target_bitstring,
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
        n_lambda_0=len(left_values),
        n_lambda_4=len(right_values),
    )


def _comparisons(metrics: list[MitigatedRow]) -> list[MitigatedComparison]:
    main = [row for row in metrics if row.protocol_arm == "fim_repeated_followup"]
    grouped: dict[tuple[str, int, float], list[MitigatedRow]] = {}
    for row in main:
        if row.depth is None or row.lambda_fim is None:
            continue
        grouped.setdefault((row.initial_bitstring, row.depth, row.lambda_fim), []).append(row)

    comparisons: list[MitigatedComparison] = []
    for initial, depth, lambda_fim in sorted(grouped):
        if lambda_fim != 0.0:
            continue
        left = grouped[(initial, depth, lambda_fim)]
        right = grouped.get((initial, depth, 4.0))
        if right is None or len(left) < 2 or len(right) < 2:
            continue
        for observable in (
            "state_retention_matrix_mitigated",
            "magnetisation_leakage_matrix_mitigated",
            "parity_leakage_matrix_mitigated",
        ):
            comparisons.append(_comparison(left, right, observable))
    return comparisons


def _fisher(
    comparisons: list[MitigatedComparison],
    observable: str,
) -> dict[str, float | int]:
    selected = [row for row in comparisons if row.observable == observable]
    if not selected:
        return {"n_tests": 0}
    chi2, p_value = combine_pvalues([row.welch_p for row in selected], method="fisher")
    deltas = [row.delta_lambda_4_minus_0 for row in selected]
    return {
        "n_tests": len(selected),
        "chi2": float(chi2),
        "p": float(p_value),
        "n_delta_positive": sum(delta > 0.0 for delta in deltas),
        "n_delta_negative": sum(delta < 0.0 for delta in deltas),
        "mean_delta": float(mean(deltas)),
        "sem_delta": float(_sem(deltas)),
    }


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["result_rows"]
    n_qubits = int(payload.get("n_qubits", 4))
    calibrations = _calibration_counts(rows, n_qubits)
    matrix = build_readout_confusion_matrix(calibrations, n_qubits)
    metrics = [
        _row_metric(row, matrix)
        for row in rows
        if row["metadata"].get("protocol_arm") == "fim_repeated_followup"
    ]
    comparisons = _comparisons(metrics)
    observables = (
        "state_retention_matrix_mitigated",
        "magnetisation_leakage_matrix_mitigated",
        "parity_leakage_matrix_mitigated",
    )
    return {
        "analysis": "fim_readout_matrix_mitigation",
        "backend": payload.get("backend"),
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "n_qubits": n_qubits,
        "n_main_rows": len(metrics),
        "n_calibration_rows": len(calibrations),
        "calibration_basis_complete": len(calibrations) == 2**n_qubits,
        "calibration_prepared_state_convention": "metadata.initial_bitstring[::-1]",
        "matrix_convention": "observed_probabilities = confusion_matrix @ true_probabilities",
        "condition_number": matrix.condition_number,
        "mean_calibration_shots": float(mean(matrix.shots_by_prepared_state.values())),
        "fisher_by_observable": {
            observable: _fisher(comparisons, observable) for observable in observables
        },
        "claim_boundary": {
            "outcome": "full_basis_readout_matrix_crosscheck_completed",
            "interpretation": (
                "The repeated FIM follow-up has all 16 n=4 basis-state readout "
                "calibration circuits, so this analysis applies a full 2^n by 2^n "
                "readout confusion matrix offline. It does not add new QPU data."
            ),
            "blocked_claims": [
                "No hardware-protection claim for lambda=4.",
                "No backend-general readout-mitigation claim.",
                "No correction for coherent gate errors or Trotter overhead.",
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
    """Apply full-basis readout mitigation to the repeated follow-up artefact."""
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

    mag = summary["fisher_by_observable"]["magnetisation_leakage_matrix_mitigated"]
    retention = summary["fisher_by_observable"]["state_retention_matrix_mitigated"]
    print("SCPN/FIM full-basis readout-matrix mitigation")
    print(f"  backend: {summary['backend']}")
    print(f"  job ID:  {summary['job_id']}")
    print(f"  output:  {SUMMARY_OUT}")
    print(f"  condition number: {summary['condition_number']:.3f}")
    print(f"  magnetisation leakage mean delta: {mag['mean_delta']:.6f}")
    print(f"  magnetisation leakage Fisher p:   {mag['p']:.6e}")
    print(f"  state retention mean delta:       {retention['mean_delta']:.6f}")
    print(f"  state retention Fisher p:         {retention['p']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
