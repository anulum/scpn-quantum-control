# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — depth-profile campaign frozen primary analysis
"""Frozen analysis for the parity-asymmetry depth-profile campaign.

Implements ``docs/campaigns/iqm_dla_depth_profile_prereg_2026-07-22.md``
verbatim:

- **Primary (decay ordering):** one-sided z-test that the pooled
  even-minus-odd leakage difference at depth 8 exceeds that at depth 12
  (``delta_8 > delta_12``, alpha = 0.05, difference-of-differences with
  binomial variances, 4,096 shots per arm per depth).
- **S1 (crossing localisation):** per-depth signs with Wilson 95 %
  intervals over the JOINT profile {4, 6, 8, 10, 12}; depths 4/6/10 come
  from the executed powered block (2026-07-21) — the cross-window caveat is
  part of the frozen wording and is emitted into the artefact.
- **S2 (equilibration check):** mean total leakage per depth with Wilson
  intervals, reported descriptively against saturation toward 0.5.
- Per-repetition drift table.

Raw-count treatment matches the executed powered-block primary (identical
across arms; readout calibration circuits committed alongside). No
coherent-dynamics claim in any branch (noiseless leakage is exactly zero).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

NEW_DEPTHS = (8, 12)
PRIOR_DEPTHS = (4, 6, 10)
REPETITIONS = (1, 2, 3, 4)
ALPHA = 0.05
SECTORS = {"even": "0011", "odd": "0001"}


def _parity(bitstring: str) -> int:
    return bitstring.replace(" ", "").count("1") % 2


def _leak(counts: dict[str, int], initial: str) -> tuple[int, int]:
    total = sum(int(v) for v in counts.values())
    expected = _parity(initial)
    leaked = sum(int(v) for k, v in counts.items() if _parity(k) != expected)
    return leaked, total


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def _pooled(counts: dict[str, dict[str, int]], depth: int) -> dict[str, tuple[int, int]]:
    """Pooled (leaked, total) per sector for one depth over all repetitions."""
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


def main(argv: list[str] | None = None) -> int:
    """Run the frozen depth-profile analysis and write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", required=True, nargs="+", help="depth-profile counts JSONs (all reps)"
    )
    parser.add_argument(
        "--powered-counts",
        required=True,
        nargs="+",
        help="executed powered-block counts JSONs (d4/6/10, PRIOR window — S1 joint profile)",
    )
    parser.add_argument("--out", required=True, help="output analysis JSON")
    args = parser.parse_args(argv)

    counts: dict[str, dict[str, int]] = {}
    job_ids: list[str] = []
    for path in args.counts:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        counts.update(payload["counts"])
        job_ids.extend(payload.get("job_ids", []))
    prior_counts: dict[str, dict[str, int]] = {}
    prior_jobs: list[str] = []
    for path in args.powered_counts:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        prior_counts.update(payload["counts"])
        prior_jobs.extend(payload.get("job_ids", []))

    missing = [
        f"main_d{d}_{s}_rep{r}"
        for d in NEW_DEPTHS
        for s in SECTORS
        for r in REPETITIONS
        if f"main_d{d}_{s}_rep{r}" not in counts
    ]
    complete = not missing

    # Primary: delta_8 > delta_12 on pooled proportions.
    pooled = {depth: _pooled(counts, depth) for depth in NEW_DEPTHS}
    deltas: dict[int, float] = {}
    variances: dict[int, float] = {}
    for depth in NEW_DEPTHS:
        (le, ne), (lo, no) = pooled[depth]["even"], pooled[depth]["odd"]
        p_e, p_o = le / ne, lo / no
        deltas[depth] = p_e - p_o
        variances[depth] = p_e * (1 - p_e) / ne + p_o * (1 - p_o) / no
    z = (deltas[8] - deltas[12]) / math.sqrt(variances[8] + variances[12])
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    primary = {
        "delta_8": deltas[8],
        "delta_12": deltas[12],
        "one_sided_z": z,
        "one_sided_p": p_value,
        "alpha": ALPHA,
        "rejects_null": bool(complete and deltas[8] > deltas[12] and p_value < ALPHA),
    }

    # S1: joint profile with Wilson intervals; new window + prior window.
    joint_profile: list[dict[str, Any]] = []
    for depth, source, source_counts, window in [
        *[(d, "depth_profile_2026-07-22", counts, "this campaign") for d in NEW_DEPTHS],
        *[(d, "powered_block_2026-07-21", prior_counts, "prior window") for d in PRIOR_DEPTHS],
    ]:
        sector_totals = _pooled(source_counts, depth)
        (le, ne), (lo, no) = sector_totals["even"], sector_totals["odd"]
        if not ne or not no:
            continue
        p_e, p_o = le / ne, lo / no
        joint_profile.append(
            {
                "depth": depth,
                "window": window,
                "source": source,
                "leak_even": p_e,
                "leak_odd": p_o,
                "relative_asymmetry": (p_e - p_o) / p_o,
                "wilson95_even": _wilson(le, ne),
                "wilson95_odd": _wilson(lo, no),
                "sign_positive": p_e > p_o,
                "total_leakage": (p_e + p_o) / 2.0,
            }
        )
    joint_profile.sort(key=lambda row: row["depth"])

    drift_table = []
    for depth in NEW_DEPTHS:
        for rep in REPETITIONS:
            row: dict[str, Any] = {"depth": depth, "repetition": rep}
            for sector, initial in SECTORS.items():
                label = f"main_d{depth}_{sector}_rep{rep}"
                if label in counts:
                    lk, tt = _leak(counts[label], initial)
                    row[f"leak_{sector}"] = lk / tt
            if "leak_even" in row and "leak_odd" in row:
                drift_table.append(row)

    report = {
        "campaign": "iqm_dla_depth_profile_prereg_2026-07-22",
        "kind": "primary_decision_rule",
        "job_ids": job_ids,
        "prior_window_job_ids": prior_jobs,
        "matrix_complete": complete,
        "missing_labels": missing,
        "primary_decay_ordering": primary,
        "joint_profile_with_cross_window_caveat": {
            "caveat": (
                "depths 4/6/10 come from the 2026-07-21 powered block (prior "
                "execution window, same calibration set 246a4930); depths 8/12 "
                "from this campaign — cross-window comparison per the frozen "
                "preregistration wording"
            ),
            "profile": joint_profile,
        },
        "per_repetition_drift": drift_table,
        "interpretation_boundary": (
            "device-noise statement only: exact statevector baseline fixes "
            "noiseless parity leakage at zero (AUD-6)"
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"analysis: {out_path}")
    print(
        f"PRIMARY delta_8 {deltas[8]:+.4f} vs delta_12 {deltas[12]:+.4f} "
        f"z {z:.3f} p {p_value:.3e} -> rejects_null={primary['rejects_null']}"
    )
    for row in joint_profile:
        print(
            f"d{row['depth']:>2} [{row['window']}]: rel {row['relative_asymmetry']:+.4f} "
            f"total {row['total_leakage']:.4f} sign+ {row['sign_positive']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
