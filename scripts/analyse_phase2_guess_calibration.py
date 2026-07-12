#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analyse phase2 guess calibration script
# scpn-quantum-control -- Offline parity-decay calibration for GUESS follow-up
"""Fit parity-survival decay curves as an offline GUESS-readiness check.

This is not a zero-noise extrapolation claim because the datasets do not include
deliberate noise-scale folding. It asks a narrower question: is parity leakage a
smooth enough symmetry-decay witness, as a function of circuit depth, to justify
a future GUESS-style hardware protocol?
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AG_SUMMARY = REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_summary_2026-05-05.json"
POPCOUNT_SUMMARY = (
    REPO_ROOT
    / "data"
    / "phase2_popcount_control"
    / "phase2_popcount_control_summary_2026-05-05.json"
)
OUT_PATH = (
    REPO_ROOT
    / "data"
    / "phase2_guess_calibration"
    / "phase2_guess_calibration_summary_2026-05-05.json"
)


@dataclass(frozen=True)
class DecayFit:
    """Log-linear parity-survival decay fit used for GUESS-readiness triage."""

    dataset: str
    series: str
    n_points: int
    alpha_per_depth: float
    intercept: float
    r_squared: float
    rmse_log_survival: float
    min_survival: float
    max_survival: float
    usable_for_future_guess: bool


def _fit_decay(dataset: str, series: str, rows: list[tuple[int, float]]) -> DecayFit:
    x = np.asarray([depth for depth, _ in rows], dtype=float)
    survival = np.asarray([max(1.0 - 2.0 * leakage, 1e-9) for _, leakage in rows], dtype=float)
    y = np.log(survival)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    rmse = math.sqrt(ss_res / len(rows))
    return DecayFit(
        dataset=dataset,
        series=series,
        n_points=len(rows),
        alpha_per_depth=float(max(0.0, -slope)),
        intercept=float(intercept),
        r_squared=float(r_squared),
        rmse_log_survival=float(rmse),
        min_survival=float(np.min(survival)),
        max_survival=float(np.max(survival)),
        usable_for_future_guess=bool(
            r_squared >= 0.90 and rmse <= 0.08 and np.min(survival) > 0.05
        ),
    )


def _ag_series(summary: dict[str, Any]) -> list[DecayFit]:
    rows = summary["depth_summaries"]
    even = [(int(row["depth"]), float(row["leakage_even"])) for row in rows]
    odd = [(int(row["depth"]), float(row["leakage_odd"])) for row in rows]
    mean_series = [
        (depth, mean([even_value, odd_value]))
        for (depth, even_value), (_, odd_value) in zip(even, odd)
    ]
    return [
        _fit_decay("phase2_ag_n4", "even", even),
        _fit_decay("phase2_ag_n4", "odd", odd),
        _fit_decay("phase2_ag_n4", "sector_mean", mean_series),
    ]


def _popcount_series(summary: dict[str, Any]) -> list[DecayFit]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in summary["state_summaries"]:
        grouped.setdefault(str(row["state_label"]), []).append(
            (int(row["depth"]), float(row["mean_parity_leakage"]))
        )
    return [
        _fit_decay("phase2_popcount", label, sorted(rows))
        for label, rows in sorted(grouped.items())
    ]


def main() -> int:
    """Run the offline Phase 2 GUESS-readiness calibration summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args()

    ag_summary = json.loads(AG_SUMMARY.read_text(encoding="utf-8"))
    popcount_summary = json.loads(POPCOUNT_SUMMARY.read_text(encoding="utf-8"))
    fits = _ag_series(ag_summary) + _popcount_series(popcount_summary)
    summary = {
        "method": "log_linear_fit_of_parity_survival_vs_depth",
        "not_zero_noise_extrapolation": True,
        "interpretation": (
            "Depth is used only as a cumulative-noise proxy. Future GUESS validation "
            "requires explicit folded noise scales or another controlled noise knob."
        ),
        "fits": [asdict(fit) for fit in fits],
        "usable_series": [fit.series for fit in fits if fit.usable_for_future_guess],
    }
    if args.write_json:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Phase 2 offline GUESS-readiness calibration")
    for fit in fits:
        usable = "usable" if fit.usable_for_future_guess else "not-ready"
        print(
            f"  {fit.dataset}:{fit.series}: alpha={fit.alpha_per_depth:.6f}, "
            f"R2={fit.r_squared:.4f}, rmse={fit.rmse_log_survival:.4f}, {usable}"
        )
    if args.write_json:
        print(f"  wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
