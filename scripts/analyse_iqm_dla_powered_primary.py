# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — powered DLA backend-sensitivity primary analysis
"""Frozen primary analysis for the powered DLA backend-sensitivity block.

Implements the analysis plan of
``docs/campaigns/iqm_dla_backend_sensitivity_powered_prereg_2026-07-21.md``
(extending ``scripts/analyse_iqm_dla_parity.py``):

- primary: pooled across the three depths and four repetitions, one-sided
  two-proportion z-test of ``leak_even > leak_odd`` at ``alpha = 0.05``;
- relative asymmetry metric ``(leak_even − leak_odd) / leak_odd``;
- secondaries: per-depth Wilson intervals and one-sided tests, a
  per-repetition drift table, and sign agreement with the IBM Phase 2 A+G
  per-depth reference directions (all positive).

Decision rule (frozen): reject `H0` only if the pooled one-sided p < 0.05
with a positive direction; otherwise report the bounded null with the same
prominence. Either way no coherent-dynamics claim exists — the exact
baseline pins noiseless leakage at zero, so every asymmetry statement is a
statement about device noise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEPTHS = (4, 6, 10)
REPETITIONS = (1, 2, 3, 4)
ALPHA = 0.05


def _parity(bitstring: str) -> int:
    return bitstring.replace(" ", "").count("1") % 2


def _leak(counts: dict[str, int], initial: str) -> tuple[int, int]:
    """Return (leaked shots, total shots) for one circuit's counts."""
    total = sum(int(v) for v in counts.values())
    expected = _parity(initial)
    leaked = sum(int(v) for k, v in counts.items() if _parity(k) != expected)
    return leaked, total


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("empty sample")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def _one_sided_two_proportion(leak_a: int, n_a: int, leak_b: int, n_b: int) -> tuple[float, float]:
    """One-sided z-test of p_a > p_b; returns (z, p_value)."""
    p_a, p_b = leak_a / n_a, leak_b / n_b
    pooled = (leak_a + leak_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0.0:
        raise ValueError("degenerate pooled proportion")
    z = (p_a - p_b) / se
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return z, p_value


def main(argv: list[str] | None = None) -> int:
    """Run the frozen primary analysis and write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        required=True,
        nargs="+",
        help="hardware counts JSONs (repetition blocks, any order)",
    )
    parser.add_argument("--out", required=True, help="output analysis JSON")
    args = parser.parse_args(argv)

    counts: dict[str, dict[str, int]] = {}
    job_ids: list[str] = []
    layouts: set[tuple[int, ...]] = set()
    for path in args.counts:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        counts.update(payload["counts"])
        job_ids.extend(payload.get("job_ids", []))
        layouts.add(tuple(payload.get("layout", ())))
    if len(layouts) != 1:
        raise ValueError(f"repetition blocks disagree on layout: {sorted(layouts)}")

    missing = [
        f"main_d{d}_{s}_rep{r}"
        for d in DEPTHS
        for s in ("even", "odd")
        for r in REPETITIONS
        if f"main_d{d}_{s}_rep{r}" not in counts
    ]
    complete = not missing
    if missing:
        print(f"note: matrix incomplete — missing {missing}; primary needs all repetitions")

    initials = {"even": "0011", "odd": "0001"}
    pooled = {"even": [0, 0], "odd": [0, 0]}
    per_depth: dict[str, Any] = {}
    drift_table: list[dict[str, Any]] = []
    for depth in DEPTHS:
        depth_tot = {"even": [0, 0], "odd": [0, 0]}
        for rep in REPETITIONS:
            row: dict[str, Any] = {"depth": depth, "repetition": rep}
            for sector, initial in initials.items():
                label = f"main_d{depth}_{sector}_rep{rep}"
                if label not in counts:
                    continue
                leaked, total = _leak(counts[label], initial)
                pooled[sector][0] += leaked
                pooled[sector][1] += total
                depth_tot[sector][0] += leaked
                depth_tot[sector][1] += total
                row[f"leak_{sector}"] = leaked / total
            if "leak_even" in row and "leak_odd" in row:
                row["relative_asymmetry"] = (row["leak_even"] - row["leak_odd"]) / row["leak_odd"]
                drift_table.append(row)
        (le, ne), (lo, no) = depth_tot["even"], depth_tot["odd"]
        if ne and no:
            z, p = _one_sided_two_proportion(le, ne, lo, no)
            per_depth[str(depth)] = {
                "leak_even": le / ne,
                "leak_odd": lo / no,
                "relative_asymmetry": (le / ne - lo / no) / (lo / no),
                "wilson95_even": _wilson(le, ne),
                "wilson95_odd": _wilson(lo, no),
                "one_sided_z": z,
                "one_sided_p": p,
                "sign_matches_ibm_positive": (le / ne - lo / no) > 0,
            }

    (le, ne), (lo, no) = pooled["even"], pooled["odd"]
    z, p = _one_sided_two_proportion(le, ne, lo, no)
    leak_even, leak_odd = le / ne, lo / no
    primary = {
        "shots_even": ne,
        "shots_odd": no,
        "leak_even": leak_even,
        "leak_odd": leak_odd,
        "relative_asymmetry": (leak_even - leak_odd) / leak_odd,
        "one_sided_z": z,
        "one_sided_p": p,
        "alpha": ALPHA,
        "rejects_h0": bool(complete and leak_even > leak_odd and p < ALPHA),
    }

    report = {
        "campaign": "iqm_dla_backend_sensitivity_powered_prereg_2026-07-21",
        "kind": "primary_decision_rule",
        "layout": list(next(iter(layouts))),
        "job_ids": job_ids,
        "matrix_complete": complete,
        "missing_labels": missing,
        "primary_pooled": primary,
        "per_depth": per_depth,
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
        f"pooled: leak_even {leak_even:.4f} leak_odd {leak_odd:.4f} "
        f"rel {primary['relative_asymmetry']:+.4f} z {z:.3f} p {p:.3e}"
    )
    for depth, row in per_depth.items():
        print(
            f"d{depth}: rel {row['relative_asymmetry']:+.4f} p {row['one_sided_p']:.3e} "
            f"sign+ {row['sign_matches_ibm_positive']}"
        )
    print(f"PRIMARY REJECTS H0 (backend-universal direction): {primary['rejects_h0']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
