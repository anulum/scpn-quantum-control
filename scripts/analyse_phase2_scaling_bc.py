#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 B-C scaling raw-count reproducer
"""Recompute Phase 2 B-C scaling statistics from raw IBM counts."""

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


@dataclass(frozen=True)
class ScalingDepthSummary:
    """Phase 2 B-C leakage and Welch-test summary for one width/depth pair."""

    n_qubits: int
    depth: int
    leakage_even: float
    leakage_odd: float
    sem_even: float
    sem_odd: float
    asymmetry_relative: float
    welch_t: float
    welch_p: float
    n_even: int
    n_odd: int


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


def _total(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarise Phase 2 B-C scaling rows from a raw-count payload."""

    rows = [
        c
        for c in payload["circuits"]
        if c["meta"]["experiment"] in {"B_scaling_n6_phase2", "C_scaling_n8_phase2"}
    ]
    if len(rows) != 280:
        raise ValueError(f"expected 280 B-C circuits, got {len(rows)}")

    buckets: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        meta = row["meta"]
        if _total(row["counts"]) != 4096:
            raise ValueError(f"unexpected shot count for {meta}")
        key = (int(meta["n_qubits"]), int(meta["depth"]), str(meta["sector"]))
        buckets.setdefault(key, []).append(float(row["stats"]["parity_leakage"]))

    summaries: list[ScalingDepthSummary] = []
    p_values_by_n: dict[int, list[float]] = {}
    for n_qubits, depth in sorted({(k[0], k[1]) for k in buckets}):
        even = buckets[(n_qubits, depth, "even")]
        odd = buckets[(n_qubits, depth, "odd")]
        even_mean = mean(even)
        odd_mean = mean(odd)
        welch = ttest_ind(even, odd, equal_var=False)
        p_values_by_n.setdefault(n_qubits, []).append(float(welch.pvalue))
        summaries.append(
            ScalingDepthSummary(
                n_qubits=n_qubits,
                depth=depth,
                leakage_even=float(even_mean),
                leakage_odd=float(odd_mean),
                sem_even=float(_sem(even)),
                sem_odd=float(_sem(odd)),
                asymmetry_relative=float((even_mean - odd_mean) / odd_mean),
                welch_t=float(welch.statistic),
                welch_p=float(welch.pvalue),
                n_even=len(even),
                n_odd=len(odd),
            )
        )

    fisher_by_n = {}
    for n_qubits, p_values in sorted(p_values_by_n.items()):
        chi2, p_combined = combine_pvalues(p_values, method="fisher")
        fisher_by_n[str(n_qubits)] = {
            "chi2": float(chi2),
            "p": float(p_combined),
            "n_depths_significant_at_0_05": sum(p < 0.05 for p in p_values),
            "n_depths_tested": len(p_values),
        }

    return {
        "backend": payload.get("backend"),
        "job_ids": payload.get("job_ids", []),
        "n_circuits": len(rows),
        "depth_summaries": [asdict(s) for s in summaries],
        "fisher_by_n": fisher_by_n,
    }


def main() -> int:
    """Run the Phase 2 B-C scaling raw-count reproduction CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--sha256", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.sha256:
        actual = _sha256(args.input)
        if actual != args.sha256:
            raise SystemExit(f"SHA-256 mismatch: {actual} != {args.sha256}")

    summary = summarise(json.loads(args.input.read_text(encoding="utf-8")))
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("Phase 2 B-C scaling reproduction")
    print(f"  backend:  {summary['backend']}")
    print(f"  job IDs:  {', '.join(summary['job_ids'])}")
    print(f"  circuits: {summary['n_circuits']}")
    for n_qubits, fisher in summary["fisher_by_n"].items():
        print(
            f"  n={n_qubits}: Fisher chi2={fisher['chi2']:.6f}, "
            f"p={fisher['p']:.6e}, "
            f"sig={fisher['n_depths_significant_at_0_05']}/{fisher['n_depths_tested']}"
        )
    print()
    print(f"{'n':>3} {'depth':>6} {'even':>10} {'odd':>10} {'asym_rel':>10} {'p':>12}")
    for row in summary["depth_summaries"]:
        print(
            f"{row['n_qubits']:>3d} {row['depth']:>6d} "
            f"{row['leakage_even']:>10.6f} {row['leakage_odd']:>10.6f} "
            f"{100 * row['asymmetry_relative']:>9.2f}% {row['welch_p']:>12.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
