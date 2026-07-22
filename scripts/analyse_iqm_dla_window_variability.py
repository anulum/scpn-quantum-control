# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — window-variability frozen analysis
"""Frozen analysis for the window-variability campaign (FU-W).

Implements ``docs/campaigns/iqm_dla_window_variability_prereg_2026-07-22.md``
verbatim. Per window ``w`` and depth ``d`` the four repetitions pool into
``leak_even`` / ``leak_odd`` (4,096 shots per arm) and the difference
``Δ(d, w) = leak_even − leak_odd`` with binomial variance.

- **Primary:** Cochran's Q homogeneity test of ``Δ(10, w)`` across windows
  against the shot-noise-only null (α = 0.05).
- **S1:** the same Q test at depths 4, 8, 12, Holm–Bonferroni adjusted.
- **S2:** DerSimonian–Laird ``τ̂(d)`` point estimates (descriptive).
- **S3:** fraction of windows with ``Δ(4, w) > 0`` with an exact
  Clopper–Pearson 95 % interval.
- **S4:** descriptive per-window calibration covariates (optional input).
- Per-window drift table; per-window four-state parity-confusion summary
  from that window's readout block (descriptive; raw-count endpoints stay
  primary exactly as in the powered block).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import beta, chi2

DEPTHS = (4, 8, 10, 12)
PRIMARY_DEPTH = 10
SECONDARY_DEPTHS = (4, 8, 12)
REPETITIONS = (1, 2, 3, 4)
ALPHA = 0.05
SECTORS = {"even": "0011", "odd": "0001"}
READOUT_STATES = ("0011", "0001", "0000", "1111")
MINIMUM_WINDOWS = 6


def _parity(bitstring: str) -> int:
    return bitstring.replace(" ", "").count("1") % 2


def _leak(counts: dict[str, int], initial: str) -> tuple[int, int]:
    total = sum(int(v) for v in counts.values())
    expected = _parity(initial)
    leaked = sum(int(v) for k, v in counts.items() if _parity(k) != expected)
    return leaked, total


def _window_arms(counts: dict[str, dict[str, int]], depth: int) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for sector, initial in SECTORS.items():
        leaked = total = 0
        for rep in REPETITIONS:
            label = f"main_d{depth}_{sector}_rep{rep}"
            if label in counts:
                lk, tt = _leak(counts[label], initial)
                leaked += lk
                total += tt
        out[sector] = (leaked, total)
    return out


def _delta_and_variance(arms: dict[str, tuple[int, int]]) -> tuple[float, float] | None:
    (le, ne), (lo, no) = arms["even"], arms["odd"]
    if ne == 0 or no == 0:
        return None
    p_e, p_o = le / ne, lo / no
    variance = p_e * (1 - p_e) / ne + p_o * (1 - p_o) / no
    return p_e - p_o, variance


def _cochran_q(deltas: list[float], variances: list[float]) -> dict[str, Any]:
    weights = [1.0 / v for v in variances]
    mean = sum(w * d for w, d in zip(weights, deltas, strict=True)) / sum(weights)
    q = sum(w * (d - mean) ** 2 for w, d in zip(weights, deltas, strict=True))
    df = len(deltas) - 1
    p_value = float(chi2.sf(q, df))
    # DerSimonian–Laird method-of-moments between-window variance.
    s1, s2 = sum(weights), sum(w * w for w in weights)
    c = s1 - s2 / s1
    tau_squared = max(0.0, (q - df) / c) if c > 0 else 0.0
    return {
        "windows": len(deltas),
        "weighted_mean_difference": mean,
        "cochran_q": q,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "tau_dl": math.sqrt(tau_squared),
        "mean_shot_noise_se": math.sqrt(len(variances) / s1),
    }


def _holm(p_values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (depth, p) in enumerate(ordered):
        running = max(running, (m - rank) * p)
        adjusted[depth] = min(1.0, running)
    return adjusted


def _clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(ALPHA / 2, successes, total - successes + 1))
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1 - ALPHA / 2, successes + 1, total - successes))
    )
    return lower, upper


def _readout_parity_confusion(counts: dict[str, dict[str, int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for state in READOUT_STATES:
        label = f"readout_{state}"
        if label in counts:
            wrong, total = _leak(counts[label], state)
            out[state] = wrong / total
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the frozen window-variability analysis and write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-counts",
        required=True,
        nargs="+",
        help="per-window counts JSONs in window order (window 1 first)",
    )
    parser.add_argument(
        "--covariates",
        default=None,
        help="optional JSON mapping window index to calibration covariates (S4)",
    )
    parser.add_argument("--out", required=True, help="output analysis JSON")
    args = parser.parse_args(argv)

    windows: list[dict[str, Any]] = []
    for index, path in enumerate(args.window_counts, start=1):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        windows.append(
            {
                "window": index,
                "counts": payload["counts"],
                "job_ids": payload.get("job_ids", []),
                "date": payload.get("date"),
                "layout": payload.get("layout"),
            }
        )

    achieved = len(windows)
    analysable = achieved >= MINIMUM_WINDOWS

    per_window: list[dict[str, Any]] = []
    per_depth_inputs: dict[int, tuple[list[float], list[float]]] = {d: ([], []) for d in DEPTHS}
    for window in windows:
        row: dict[str, Any] = {
            "window": window["window"],
            "date": window["date"],
            "job_ids": window["job_ids"],
            "readout_parity_confusion": _readout_parity_confusion(window["counts"]),
            "depths": {},
        }
        for depth in DEPTHS:
            arms = _window_arms(window["counts"], depth)
            resolved = _delta_and_variance(arms)
            if resolved is None:
                continue
            delta, variance = resolved
            (le, ne), (lo, no) = arms["even"], arms["odd"]
            row["depths"][str(depth)] = {
                "leak_even": le / ne,
                "leak_odd": lo / no,
                "difference_even_minus_odd": delta,
                "shot_noise_se": math.sqrt(variance),
                "shots_per_arm": ne,
            }
            deltas, variances = per_depth_inputs[depth]
            deltas.append(delta)
            variances.append(variance)
        per_window.append(row)

    heterogeneity: dict[str, Any] = {}
    secondary_p: dict[int, float] = {}
    for depth in DEPTHS:
        deltas, variances = per_depth_inputs[depth]
        if len(deltas) < 2:
            continue
        result = _cochran_q(deltas, variances)
        heterogeneity[str(depth)] = result
        if depth in SECONDARY_DEPTHS:
            secondary_p[depth] = result["p_value"]

    primary = heterogeneity.get(str(PRIMARY_DEPTH), {})
    primary_decision = {
        **primary,
        "alpha": ALPHA,
        "analysable": analysable,
        "achieved_windows": achieved,
        "minimum_windows": MINIMUM_WINDOWS,
        "drift_exceeds_shot_noise": bool(analysable and primary and primary["p_value"] < ALPHA),
    }

    holm = _holm(secondary_p) if secondary_p else {}
    s1 = {
        str(depth): {
            "p_value": secondary_p[depth],
            "holm_adjusted_p": holm[depth],
            "rejects_homogeneity": bool(analysable and holm[depth] < ALPHA),
        }
        for depth in secondary_p
    }

    s2 = {
        str(depth): {
            "tau_dl": heterogeneity[str(depth)]["tau_dl"],
            "mean_shot_noise_se": heterogeneity[str(depth)]["mean_shot_noise_se"],
        }
        for depth in DEPTHS
        if str(depth) in heterogeneity
    }

    d4_deltas = per_depth_inputs[4][0]
    positive = sum(1 for delta in d4_deltas if delta > 0)
    s3 = {
        "windows": len(d4_deltas),
        "positive_windows": positive,
        "fraction_positive": positive / len(d4_deltas) if d4_deltas else None,
        "clopper_pearson_95": _clopper_pearson(positive, len(d4_deltas)) if d4_deltas else None,
    }

    covariates = None
    if args.covariates:
        covariates = json.loads(Path(args.covariates).read_text(encoding="utf-8"))
    s4 = {
        "caveat": "descriptive only per the preregistration; no test, no claim",
        "covariates": covariates,
    }

    report = {
        "campaign": "iqm_dla_window_variability_prereg_2026-07-22",
        "kind": "primary_decision_rule",
        "achieved_windows": achieved,
        "analysable": analysable,
        "primary_d10_heterogeneity": primary_decision,
        "s1_per_depth_heterogeneity_holm": s1,
        "s2_tau_profile": s2,
        "s3_d4_sign_stability": s3,
        "s4_calibration_covariates": s4,
        "per_window": per_window,
        "batching_disclosure": (
            "each window is one batched pass (mains one job, readout one "
            "job); repetitions are execution-order replicates inside the "
            "window and the WINDOW is the independent unit (frozen in the "
            "preregistration)"
        ),
        "interpretation_boundary": (
            "device-noise statement only: exact statevector baseline fixes "
            "noiseless parity leakage at zero (AUD-6); S4 is descriptive, "
            "no causal attribution of drift"
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"analysis: {out_path}")
    if primary:
        print(
            f"PRIMARY d10: Q {primary['cochran_q']:.3f} on {primary['degrees_of_freedom']} df, "
            f"p {primary['p_value']:.3e}, tau_DL {primary['tau_dl']:.4f} "
            f"(mean shot-noise se {primary['mean_shot_noise_se']:.4f}) -> "
            f"drift_exceeds_shot_noise={primary_decision['drift_exceeds_shot_noise']}"
        )
    for depth in SECONDARY_DEPTHS:
        if str(depth) in s1:
            entry = s1[str(depth)]
            print(
                f"S1 d{depth}: p {entry['p_value']:.3e} (Holm {entry['holm_adjusted_p']:.3e}) "
                f"-> rejects={entry['rejects_homogeneity']}"
            )
    if s3["windows"]:
        print(
            f"S3 d4 sign stability: {s3['positive_windows']}/{s3['windows']} positive "
            f"(CP95 {s3['clopper_pearson_95']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
