#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 DLA parity raw-count reproducer
"""Recompute Phase 2 DLA parity statistics from raw IBM counts."""

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
    REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_2026-05-05T121357Z.json"
)
PUBLISHED_SHA256 = "7c5f2a32d5a113d916d84d26d27a69336846364d5ee23ba4621b059125e0f5d5"


@dataclass(frozen=True)
class DepthSummary:
    """Phase 2 leakage, uncertainty, and Welch-test result for one depth."""

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


@dataclass(frozen=True)
class ReadoutSummary:
    """Readout-retention baseline for one prepared computational basis state."""

    n_qubits: int
    initial: str
    retention: float
    total_shots: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _total_counts(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def _retention(counts: dict[str, int], initial: str) -> float:
    total = _total_counts(counts)
    if total <= 0:
        raise ValueError("empty count dictionary")
    return counts.get(initial[::-1], 0) / total


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("backend") != "ibm_kingston":
        raise ValueError(f"unexpected backend: {payload.get('backend')}")
    if payload.get("n_circuits") != 612:
        raise ValueError(f"unexpected circuit count: {payload.get('n_circuits')}")
    if payload.get("job_ids") != ["ibm-run-7da8644af35021fb", "ibm-run-6f9990bba1d90a12"]:
        raise ValueError("job ID list does not match the preregistered Phase 2 run")

    phase_rows = [
        c for c in payload["circuits"] if c["meta"]["experiment"] == "A_dla_parity_n4_phase2"
    ]
    readout_rows = [
        c for c in payload["circuits"] if c["meta"]["experiment"] == "G_readout_baseline"
    ]
    if len(phase_rows) != 600 or len(readout_rows) != 12:
        raise ValueError(f"unexpected split: phase={len(phase_rows)}, readout={len(readout_rows)}")

    buckets: dict[tuple[int, str], list[float]] = {}
    for row in phase_rows:
        meta = row["meta"]
        stats = row["stats"]
        counts = row["counts"]
        total = _total_counts(counts)
        if total != 4096:
            raise ValueError(f"unexpected main shots at depth {meta['depth']}: {total}")
        leakage = float(stats["parity_leakage"])
        key = (int(meta["depth"]), str(meta["sector"]))
        buckets.setdefault(key, []).append(leakage)

    depth_summaries: list[DepthSummary] = []
    p_values: list[float] = []
    for depth in sorted({k[0] for k in buckets}):
        even = buckets[(depth, "even")]
        odd = buckets[(depth, "odd")]
        even_mean = mean(even)
        odd_mean = mean(odd)
        welch = ttest_ind(even, odd, equal_var=False)
        p_values.append(float(welch.pvalue))
        depth_summaries.append(
            DepthSummary(
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

    fisher_chi2, fisher_p = combine_pvalues(p_values, method="fisher")
    readout: list[ReadoutSummary] = []
    for row in readout_rows:
        meta = row["meta"]
        counts = row["counts"]
        total = _total_counts(counts)
        if total != 8192:
            raise ValueError(f"unexpected readout shots for {meta['initial']}: {total}")
        readout.append(
            ReadoutSummary(
                n_qubits=int(meta["n_qubits"]),
                initial=str(meta["initial"]),
                retention=float(_retention(counts, str(meta["initial"]))),
                total_shots=total,
            )
        )

    return {
        "backend": payload["backend"],
        "job_ids": payload["job_ids"],
        "n_circuits": payload["n_circuits"],
        "depth_summaries": [asdict(s) for s in depth_summaries],
        "fisher_chi2": float(fisher_chi2),
        "fisher_p": float(fisher_p),
        "n_depths_significant_at_0_05": sum(p < 0.05 for p in p_values),
        "n_depths_tested": len(p_values),
        "readout": [asdict(r) for r in readout],
    }


def main() -> int:
    """Run the Phase 2 DLA parity raw-count reproduction CLI."""

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

    print("Phase 2 DLA parity reduced A+G reproduction")
    print(f"  backend:              {summary['backend']}")
    print(f"  job IDs:              {', '.join(summary['job_ids'])}")
    print(f"  circuits:             {summary['n_circuits']}")
    print(f"  Fisher chi2:          {summary['fisher_chi2']:.6f}")
    print(f"  Fisher p:             {summary['fisher_p']:.6e}")
    print(
        "  significant depths:   "
        f"{summary['n_depths_significant_at_0_05']} / {summary['n_depths_tested']}"
    )
    print()
    print(f"{'depth':>6} {'even':>10} {'odd':>10} {'asym_rel':>10} {'p':>12}")
    for row in summary["depth_summaries"]:
        print(
            f"{row['depth']:>6d} {row['leakage_even']:>10.6f} "
            f"{row['leakage_odd']:>10.6f} {100 * row['asymmetry_relative']:>9.2f}% "
            f"{row['welch_p']:>12.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
