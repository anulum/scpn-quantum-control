# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — calibration-epoch sensitivity analysis
"""Pool same-calibration IQM windows before heterogeneity analysis.

This companion leaves the frozen window-level analysis untouched. It applies
the prospective post-W7 amendment in
``docs/campaigns/iqm_dla_window_variability_epoch_amendment_2026-09-04.md``:
windows with the same exact ``calibration_set_id`` are technical replicates,
pooled at the raw-count level, and contribute one calibration epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import analyse_iqm_dla_window_variability as frozen

SCHEMA = "scpn.iqm-window-variability-epoch-sensitivity.v1"
AMENDMENT = "iqm_dla_window_variability_epoch_amendment_2026-09-04"
MINIMUM_EPOCHS = 6


def _report_status(records: list[dict[str, Any]]) -> str:
    """Label prospective W1-W7 evidence separately from post-amendment runs."""
    if records and max(record["window"] for record in records) <= 7:
        return "post_w7_pre_w8_sensitivity"
    return "post_amendment_epoch_sensitivity"


def _object(path: Path) -> dict[str, Any]:
    """Load one JSON object, rejecting non-object roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _records(count_paths: list[Path], calibration_paths: list[Path]) -> list[dict[str, Any]]:
    """Pair and validate nominal-window counts with calibration evidence."""
    if len(count_paths) != len(calibration_paths):
        raise ValueError("counts and calibration path counts differ")
    records: list[dict[str, Any]] = []
    seen_windows: set[int] = set()
    for count_path, calibration_path in zip(count_paths, calibration_paths, strict=True):
        counts = _object(count_path)
        calibration = _object(calibration_path)
        window = counts.get("window")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"{count_path} has no positive integer window")
        if window in seen_windows:
            raise ValueError(f"duplicate nominal window {window}")
        seen_windows.add(window)
        calibration_set_id = calibration.get("calibration_set_id")
        if not isinstance(calibration_set_id, str) or not calibration_set_id.strip():
            raise ValueError(f"{calibration_path} has no calibration_set_id")
        if counts.get("date") != calibration.get("date"):
            raise ValueError(f"window {window} counts/calibration dates differ")
        raw_counts = counts.get("counts")
        if not isinstance(raw_counts, dict):
            raise ValueError(f"{count_path} has no counts object")
        records.append(
            {
                "window": window,
                "date": counts.get("date"),
                "layout": counts.get("layout"),
                "calibration_set_id": calibration_set_id,
                "counts": raw_counts,
            }
        )
    return sorted(records, key=lambda record: record["window"])


def _pooled_arms(records: list[dict[str, Any]], depth: int) -> dict[str, tuple[int, int]]:
    """Pool leaked and total shots across technical replicates in one epoch."""
    totals = {sector: [0, 0] for sector in frozen.SECTORS}
    for record in records:
        arms = frozen._window_arms(record["counts"], depth)
        for sector, (leaked, shots) in arms.items():
            totals[sector][0] += leaked
            totals[sector][1] += shots
    return {sector: (values[0], values[1]) for sector, values in totals.items()}


def _epoch_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build epoch mapping, pooled effects, and heterogeneity statistics."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["calibration_set_id"], []).append(record)

    per_depth: dict[int, tuple[list[float], list[float]]] = {
        depth: ([], []) for depth in frozen.DEPTHS
    }
    per_epoch: list[dict[str, Any]] = []
    window_to_epoch: list[dict[str, Any]] = []
    for epoch, (calibration_set_id, members) in enumerate(grouped.items(), start=1):
        windows = [member["window"] for member in members]
        for member in members:
            window_to_epoch.append(
                {
                    "window": member["window"],
                    "date": member["date"],
                    "epoch": epoch,
                    "calibration_set_id": calibration_set_id,
                }
            )
        row: dict[str, Any] = {
            "epoch": epoch,
            "calibration_set_id": calibration_set_id,
            "nominal_windows": windows,
            "technical_replicates": len(windows),
            "depths": {},
        }
        for depth in frozen.DEPTHS:
            arms = _pooled_arms(members, depth)
            resolved = frozen._delta_and_variance(arms)
            if resolved is None:
                continue
            delta, variance = resolved
            (leaked_even, shots_even), (leaked_odd, shots_odd) = arms["even"], arms["odd"]
            row["depths"][str(depth)] = {
                "leak_even": leaked_even / shots_even,
                "leak_odd": leaked_odd / shots_odd,
                "difference_even_minus_odd": delta,
                "shot_noise_se": variance**0.5,
                "shots_even": shots_even,
                "shots_odd": shots_odd,
            }
            deltas, variances = per_depth[depth]
            deltas.append(delta)
            variances.append(variance)
        per_epoch.append(row)

    heterogeneity: dict[str, dict[str, Any]] = {}
    secondary_p: dict[int, float] = {}
    for depth, (deltas, variances) in per_depth.items():
        if len(deltas) < 2:
            continue
        result = frozen._cochran_q(deltas, variances)
        result["epochs"] = result.pop("windows")
        heterogeneity[str(depth)] = result
        if depth in frozen.SECONDARY_DEPTHS:
            secondary_p[depth] = result["p_value"]

    achieved_epochs = len(grouped)
    analysable = achieved_epochs >= MINIMUM_EPOCHS
    primary = heterogeneity.get(str(frozen.PRIMARY_DEPTH), {})
    primary_decision = {
        **primary,
        "alpha": frozen.ALPHA,
        "analysable": analysable,
        "achieved_epochs": achieved_epochs,
        "minimum_epochs": MINIMUM_EPOCHS,
        "drift_exceeds_shot_noise": bool(
            analysable and primary and primary["p_value"] < frozen.ALPHA
        ),
    }
    holm = frozen._holm(secondary_p) if secondary_p else {}
    s1 = {
        str(depth): {
            "p_value": secondary_p[depth],
            "holm_adjusted_p": holm[depth],
            "rejects_homogeneity": bool(analysable and holm[depth] < frozen.ALPHA),
        }
        for depth in secondary_p
    }
    s2 = {
        str(depth): {
            "tau_dl": result["tau_dl"],
            "mean_shot_noise_se": result["mean_shot_noise_se"],
        }
        for depth, result in ((depth, heterogeneity[str(depth)]) for depth in frozen.DEPTHS)
    }
    d4_deltas = per_depth[4][0]
    positive = sum(delta > 0 for delta in d4_deltas)
    s3 = {
        "epochs": len(d4_deltas),
        "positive_epochs": positive,
        "fraction_positive": positive / len(d4_deltas) if d4_deltas else None,
        "clopper_pearson_95": (
            frozen._clopper_pearson(positive, len(d4_deltas)) if d4_deltas else None
        ),
    }
    return {
        "nominal_windows": len(records),
        "calibration_epochs": achieved_epochs,
        "analysable": analysable,
        "window_to_epoch": window_to_epoch,
        "technical_replicate_groups": [
            {
                "calibration_set_id": calibration_set_id,
                "nominal_windows": [member["window"] for member in members],
            }
            for calibration_set_id, members in grouped.items()
            if len(members) > 1
        ],
        "per_epoch": per_epoch,
        "primary_d10_heterogeneity": primary_decision,
        "s1_per_depth_heterogeneity_holm": s1,
        "s2_tau_profile": s2,
        "s3_d4_sign_stability": s3,
    }


def main(argv: list[str] | None = None) -> int:
    """Write the deterministic side-by-side epoch sensitivity artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-counts", required=True, nargs="+")
    parser.add_argument("--calibrations", required=True, nargs="+")
    parser.add_argument("--frozen-analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    frozen_path = Path(args.frozen_analysis)
    frozen_report = _object(frozen_path)
    records = _records(
        [Path(path) for path in args.window_counts],
        [Path(path) for path in args.calibrations],
    )
    if frozen_report.get("achieved_windows") != len(records):
        raise ValueError("frozen analysis and epoch inputs cover different window counts")

    report = {
        "schema": SCHEMA,
        "amendment": AMENDMENT,
        "status": _report_status(records),
        "frozen_window_level_reference": {
            "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "primary_d10_heterogeneity": frozen_report.get("primary_d10_heterogeneity"),
        },
        "epoch_level_sensitivity": _epoch_report(records),
        "decision_boundary": (
            "always report frozen window-level and epoch-pooled results together; "
            "same-calibration technical replicates add precision but no epoch degree of freedom"
        ),
        "interpretation_boundary": (
            "device-noise variability sensitivity only; a changed calibration-set ID is an "
            "operational epoch marker, not proof of complete physical independence"
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(frozen._portable_json_value(report), indent=2) + "\n",
        encoding="utf-8",
    )
    primary = report["epoch_level_sensitivity"]["primary_d10_heterogeneity"]
    print(
        f"epoch sensitivity: {output} — {len(records)} nominal windows, "
        f"{primary['achieved_epochs']} calibration epochs, d10 p={primary.get('p_value')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
