#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 2 popcount-control reproducer
"""Recompute Phase 2 popcount-control statistics from raw IBM counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import combine_pvalues, ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "phase2_popcount_control"
    / "phase2_popcount_control_2026-05-05T135318Z.json"
)
PUBLISHED_SHA256 = "f43cbd7e466a3267847b44a750aeba7801cbc52ef10e9808573ef7ed01ec3cf0"


@dataclass(frozen=True)
class StateSummary:
    depth: int
    state_label: str
    initial: str
    sector: str
    popcount: int
    mean_parity_leakage: float
    sem_parity_leakage: float
    n_reps: int


@dataclass(frozen=True)
class ComparisonSummary:
    name: str
    depth: int
    left_label: str
    right_label: str
    left_mean: float
    right_mean: float
    difference: float
    relative_to_right: float
    welch_t: float
    welch_p: float
    n_left: int
    n_right: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _total_counts(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def _retention(counts: dict[str, int], initial: str) -> float:
    total = _total_counts(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    return counts.get(initial[::-1], 0) / total


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("experiment") != "phase2_popcount_control":
        raise ValueError(f"unexpected experiment: {payload.get('experiment')}")
    if payload.get("backend") != "ibm_kingston":
        raise ValueError(f"unexpected backend: {payload.get('backend')}")
    if payload.get("job_ids") != ["ibm-run-7d468e2b1e44b406", "ibm-run-b3424c38cfe03c86"]:
        raise ValueError("job IDs do not match the promoted popcount-control run")
    if payload.get("n_circuits") != 365:
        raise ValueError(f"unexpected circuit count: {payload.get('n_circuits')}")

    rows = payload["circuits"]
    main_rows = [r for r in rows if r["meta"]["block"] == "parity_leakage"]
    readout_rows = [r for r in rows if r["meta"]["block"] == "readout"]
    if len(main_rows) != 360 or len(readout_rows) != 5:
        raise ValueError(f"unexpected split: main={len(main_rows)}, readout={len(readout_rows)}")

    buckets: dict[tuple[int, str], list[float]] = {}
    meta_by_label: dict[str, dict[str, Any]] = {}
    for row in main_rows:
        meta = row["meta"]
        counts = row["counts"]
        total = _total_counts(counts)
        if total != 4096:
            raise ValueError(f"unexpected main shots: {total}")
        label = str(meta["state_label"])
        key = (int(meta["depth"]), label)
        buckets.setdefault(key, []).append(float(row["stats"]["parity_leakage"]))
        meta_by_label[label] = {
            "initial": str(meta["initial"]),
            "sector": str(meta["sector"]),
            "popcount": int(meta["popcount"]),
        }

    state_summaries: list[StateSummary] = []
    for depth, label in sorted(buckets):
        values = buckets[(depth, label)]
        state_summaries.append(
            StateSummary(
                depth=depth,
                state_label=label,
                **meta_by_label[label],
                mean_parity_leakage=float(mean(values)),
                sem_parity_leakage=float(_sem(values)),
                n_reps=len(values),
            )
        )

    comparisons = {
        "original_E0_minus_O0": ("E0_original_even", "O0_original_odd"),
        "within_even_E0_minus_E1": ("E0_original_even", "E1_even_swap"),
        "within_odd_O0_minus_O1": ("O0_original_odd", "O1_odd_swap"),
        "excitation_inversion_E0_minus_O3": (
            "E0_original_even",
            "O3_odd_high_excitation",
        ),
    }
    comparison_summaries: list[ComparisonSummary] = []
    p_values: dict[str, list[float]] = {name: [] for name in comparisons}
    for depth in sorted({d for d, _ in buckets}):
        for name, (left, right) in comparisons.items():
            left_values = buckets[(depth, left)]
            right_values = buckets[(depth, right)]
            left_mean = mean(left_values)
            right_mean = mean(right_values)
            welch = ttest_ind(left_values, right_values, equal_var=False)
            p_values[name].append(float(welch.pvalue))
            comparison_summaries.append(
                ComparisonSummary(
                    name=name,
                    depth=depth,
                    left_label=left,
                    right_label=right,
                    left_mean=float(left_mean),
                    right_mean=float(right_mean),
                    difference=float(left_mean - right_mean),
                    relative_to_right=float((left_mean - right_mean) / right_mean),
                    welch_t=float(welch.statistic),
                    welch_p=float(welch.pvalue),
                    n_left=len(left_values),
                    n_right=len(right_values),
                )
            )

    fisher_by_comparison: dict[str, dict[str, float | int]] = {}
    for name, values in p_values.items():
        chi2, p = combine_pvalues(values, method="fisher")
        fisher_by_comparison[name] = {
            "chi2": float(chi2),
            "p": float(p),
            "n_depths_significant_at_0_05": sum(v < 0.05 for v in values),
            "n_depths_tested": len(values),
        }

    readout = []
    for row in readout_rows:
        meta = row["meta"]
        counts = row["counts"]
        total = _total_counts(counts)
        if total != 8192:
            raise ValueError(f"unexpected readout shots: {total}")
        readout.append(
            {
                "state_label": meta["state_label"],
                "initial": meta["initial"],
                "retention": _retention(counts, meta["initial"]),
                "total_shots": total,
            }
        )

    return {
        "backend": payload["backend"],
        "job_ids": payload["job_ids"],
        "n_circuits": payload["n_circuits"],
        "live_depth_summary": payload["live_depth_summary"],
        "live_gate_summary": payload["live_gate_summary"],
        "state_summaries": [asdict(s) for s in state_summaries],
        "comparison_summaries": [asdict(s) for s in comparison_summaries],
        "fisher_by_comparison": fisher_by_comparison,
        "readout": readout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--verify-integrity", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.verify_integrity:
        actual = _sha256(args.input)
        if actual != PUBLISHED_SHA256:
            raise SystemExit(f"SHA-256 mismatch: {actual} != {PUBLISHED_SHA256}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = _summarise(payload)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("Phase 2 popcount-control reproduction")
    print(f"  backend:  {summary['backend']}")
    print(f"  job IDs:  {', '.join(summary['job_ids'])}")
    print(f"  circuits: {summary['n_circuits']}")
    print()
    for name, stats in summary["fisher_by_comparison"].items():
        print(
            f"{name}: chi2={stats['chi2']:.6f}, p={stats['p']:.6e}, "
            f"sig={stats['n_depths_significant_at_0_05']}/"
            f"{stats['n_depths_tested']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
