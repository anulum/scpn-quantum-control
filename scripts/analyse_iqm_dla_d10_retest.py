# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — d10 sign-replication frozen analysis
"""Frozen analysis for the depth-10 sign-replication campaign.

Implements ``docs/campaigns/iqm_dla_d10_retest_prereg_2026-07-22.md``
verbatim: primary one-sided replication of the NEGATIVE asymmetry
(``leak_even < leak_odd``) pooled over the 8 execution-order repetitions;
secondaries S1 (two-sided + Wilson intervals), S2 (cross-window
difference-of-differences against the 2026-07-21 depth-10 arms), S3 (total
leakage versus the prior window), and the per-repetition drift table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEPTH = 10
REPETITIONS = tuple(range(1, 9))
PRIOR_REPETITIONS = tuple(range(1, 5))
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


def _pooled(
    counts: dict[str, dict[str, int]], repetitions: tuple[int, ...]
) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for sector, initial in SECTORS.items():
        leaked = total = 0
        for rep in repetitions:
            label = f"main_d{DEPTH}_{sector}_rep{rep}"
            if label in counts:
                lk, tt = _leak(counts[label], initial)
                leaked += lk
                total += tt
        out[sector] = (leaked, total)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the frozen d10 sign-replication analysis and write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, help="d10-retest counts JSON")
    parser.add_argument(
        "--prior-counts",
        required=True,
        nargs="+",
        help="2026-07-21 powered-block counts JSONs (prior-window d10 arms)",
    )
    parser.add_argument("--out", required=True, help="output analysis JSON")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.counts).read_text(encoding="utf-8"))
    counts = payload["counts"]
    prior_counts: dict[str, dict[str, int]] = {}
    for path in args.prior_counts:
        prior_counts.update(json.loads(Path(path).read_text(encoding="utf-8"))["counts"])

    missing = [
        f"main_d{DEPTH}_{s}_rep{r}"
        for s in SECTORS
        for r in REPETITIONS
        if f"main_d{DEPTH}_{s}_rep{r}" not in counts
    ]
    complete = not missing

    pooled = _pooled(counts, REPETITIONS)
    (le, ne), (lo, no) = pooled["even"], pooled["odd"]
    p_e, p_o = le / ne, lo / no
    pooled_p = (le + lo) / (ne + no)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / ne + 1 / no))
    # Primary is one-sided for the NEGATIVE asymmetry: leak_even < leak_odd.
    z_negative = (p_o - p_e) / se
    p_negative = 0.5 * math.erfc(z_negative / math.sqrt(2))
    primary = {
        "shots_even": ne,
        "shots_odd": no,
        "leak_even": p_e,
        "leak_odd": p_o,
        "difference_even_minus_odd": p_e - p_o,
        "relative_asymmetry": (p_e - p_o) / p_o,
        "one_sided_z_for_negative": z_negative,
        "one_sided_p_for_negative": p_negative,
        "alpha": ALPHA,
        "negative_sign_replicates": bool(complete and p_e < p_o and p_negative < ALPHA),
    }

    z_two = abs(p_e - p_o) / se
    s1 = {
        "two_sided_p": math.erfc(z_two / math.sqrt(2)),
        "wilson95_even": _wilson(le, ne),
        "wilson95_odd": _wilson(lo, no),
    }

    prior = _pooled(prior_counts, PRIOR_REPETITIONS)
    (ple, pne), (plo, pno) = prior["even"], prior["odd"]
    pp_e, pp_o = ple / pne, plo / pno
    diff_now, diff_prior = p_e - p_o, pp_e - pp_o
    var_now = p_e * (1 - p_e) / ne + p_o * (1 - p_o) / no
    var_prior = pp_e * (1 - pp_e) / pne + pp_o * (1 - pp_o) / pno
    z_cross = (diff_now - diff_prior) / math.sqrt(var_now + var_prior)
    s2 = {
        "difference_this_window": diff_now,
        "difference_prior_window": diff_prior,
        "cross_window_z": z_cross,
        "cross_window_two_sided_p": math.erfc(abs(z_cross) / math.sqrt(2)),
        "caveat": "prior window = 2026-07-21 powered block, same calibration set",
    }
    s3 = {
        "total_leakage_this_window": (p_e + p_o) / 2.0,
        "total_leakage_prior_window": (pp_e + pp_o) / 2.0,
    }

    drift = []
    for rep in REPETITIONS:
        row: dict[str, Any] = {"repetition": rep}
        for sector, initial in SECTORS.items():
            label = f"main_d{DEPTH}_{sector}_rep{rep}"
            if label in counts:
                lk, tt = _leak(counts[label], initial)
                row[f"leak_{sector}"] = lk / tt
        if "leak_even" in row and "leak_odd" in row:
            row["difference"] = row["leak_even"] - row["leak_odd"]
            drift.append(row)

    report = {
        "campaign": "iqm_dla_d10_retest_prereg_2026-07-22",
        "kind": "primary_decision_rule",
        "job_ids": payload.get("job_ids", []),
        "layout": payload.get("layout"),
        "matrix_complete": complete,
        "missing_labels": missing,
        "primary_sign_replication": primary,
        "s1_two_sided": s1,
        "s2_cross_window": s2,
        "s3_total_leakage": s3,
        "per_repetition_drift": drift,
        "batching_disclosure": (
            "8 repetitions are execution-order replicates inside one batched "
            "job (frozen in the preregistration); cross-window independence "
            "comes from the 2026-07-21 arms"
        ),
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
        f"PRIMARY d10: leak_even {p_e:.4f} vs leak_odd {p_o:.4f} "
        f"(diff {p_e - p_o:+.4f}, rel {(p_e - p_o) / p_o:+.4f}); "
        f"one-sided-negative z {z_negative:.3f} p {p_negative:.3e} -> "
        f"negative_sign_replicates={primary['negative_sign_replicates']}"
    )
    print(f"S2 cross-window: now {diff_now:+.4f} vs prior {diff_prior:+.4f} (z {z_cross:.3f})")
    print(
        f"S3 total leakage: now {s3['total_leakage_this_window']:.4f} "
        f"vs prior {s3['total_leakage_prior_window']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
