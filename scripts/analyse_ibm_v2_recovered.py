#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Reproduce the IBM v2 aggregates from recovered raw counts
"""Recompute the IBM v2 fair-experiment aggregates from the recovered raw counts.

This is the referee-facing reproducer: given the public recovered pack
(`data/ibm_hardware_v2_recovered_2026-07-18/`), it recomputes each experiment's
survival observable directly from the raw counts and compares it to the value
that was committed in the aggregate-only quarantined pack. Eight of the nine
experiments reproduce to |Δmean| < 1e-4; the ninth (`A_odd`) reproduces to a
few percent under the natural odd-parity-subspace definition because its
committed aggregate applied an original mitigation whose exact form is not in
the pack — the residual is reported honestly rather than fitted away.

Claim boundary: reproducing `F_FIM = 0.9158 > F_XY = 0.8484` (all-zero survival)
from real hardware counts establishes only that the *observation* is genuine.
It is NOT evidence of a coherence-protection ("DUAL PROTECTION") mechanism: the
two circuits differ in depth/structure, and the coherence-protection hypothesis
was tested properly on the promoted `ibm_kingston` SCPN/FIM campaign and
committed as a negative/falsification result (see
`docs/campaigns/scpn_fim_claim_boundary_2026-05-05.md`).

Usage:
  python scripts/analyse_ibm_v2_recovered.py
  python scripts/analyse_ibm_v2_recovered.py --pack DIR --output PATH
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = REPO_ROOT / "data" / "ibm_hardware_v2_recovered_2026-07-18"

#: Per-experiment survival observable. "all_zero"/"sector" survive a single
#: computational-basis target bitstring; "odd_parity" sums the odd-weight
#: subspace. Targets are the prepared-state / sector labels of the protocol.
EXPERIMENT_OBSERVABLE: dict[str, tuple[str, str | None]] = {
    "A_even": ("bitstring", "0000"),
    "A_odd": ("odd_parity", None),
    "C_xy": ("bitstring", "0000"),
    "C_fim": ("bitstring", "0000"),
    "B_M+4": ("bitstring", "0000"),
    "B_M+2": ("bitstring", "0001"),
    "B_M0": ("bitstring", "0110"),
    "B_M-2": ("bitstring", "1011"),
    "B_M-4": ("bitstring", "1111"),
}

#: |Δmean| at or below this counts as a bit-faithful reproduction.
EXACT_TOLERANCE = 1e-3


def _bitstring_survival(counts: Mapping[str, int], target: str) -> float:
    """Probability of a single computational-basis target state."""
    shots = sum(counts.values())
    return counts.get(target, 0) / shots if shots else 0.0


def _odd_parity_survival(counts: Mapping[str, int]) -> float:
    """Total probability in the odd-parity (odd Hamming-weight) subspace."""
    shots = sum(counts.values())
    if not shots:
        return 0.0
    odd = sum(c for b, c in counts.items() if sum(int(x) for x in b) % 2 == 1)
    return odd / shots


def per_pub_survival(experiment: str, per_pub_counts: Sequence[Mapping[str, int]]) -> list[float]:
    """Compute the survival observable for every pub of one experiment."""
    kind, target = EXPERIMENT_OBSERVABLE[experiment]
    if kind == "odd_parity":
        return [_odd_parity_survival(c) for c in per_pub_counts]
    assert target is not None
    return [_bitstring_survival(c, target) for c in per_pub_counts]


def reproduce(pack: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every experiment's mean survival and compare to the committed value."""
    rows = []
    exact = 0
    for entry in pack["experiments"]:
        name = entry["experiment"]
        survivals = per_pub_survival(name, entry["per_pub_counts"])
        recovered_mean = statistics.mean(survivals) if survivals else 0.0
        committed_mean = entry.get("committed_aggregate_mean")
        delta = abs(recovered_mean - committed_mean) if committed_mean is not None else None
        is_exact = delta is not None and delta <= EXACT_TOLERANCE
        exact += int(is_exact)
        rows.append(
            {
                "experiment": name,
                "observable": EXPERIMENT_OBSERVABLE[name][0],
                "recovered_mean": recovered_mean,
                "committed_mean": committed_mean,
                "abs_delta": delta,
                "reproduced": is_exact,
            }
        )
    by_name = {row["experiment"]: row for row in rows}
    f_fim = by_name["C_fim"]["recovered_mean"]
    f_xy = by_name["C_xy"]["recovered_mean"]
    return {
        "n_experiments": len(rows),
        "n_reproduced_exactly": exact,
        "rows": rows,
        "dual_protection_observation": {
            "F_FIM": f_fim,
            "F_XY": f_xy,
            "F_FIM_gt_F_XY": f_fim > f_xy,
            "claim_boundary": (
                "Genuine hardware observation reproduced from public raw counts. "
                "NOT evidence of coherence protection: circuits differ in depth/"
                "structure; the protection hypothesis was falsified on the promoted "
                "ibm_kingston SCPN/FIM campaign."
            ),
        },
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    pack = json.loads((args.pack / "recovered_raw_counts.json").read_text(encoding="utf-8"))
    report = reproduce(pack)
    for row in report["rows"]:
        mark = "OK  " if row["reproduced"] else "≈   "
        delta = "n/a" if row["abs_delta"] is None else f"{row['abs_delta']:.5f}"
        print(
            f"  {mark}{row['experiment']:7s} recovered={row['recovered_mean']:.4f} "
            f"committed={row['committed_mean']}  |Δ|={delta}"
        )
    obs = report["dual_protection_observation"]
    print(
        f"\n{report['n_reproduced_exactly']}/{report['n_experiments']} reproduce to "
        f"|Δ|<{EXACT_TOLERANCE}. F_FIM={obs['F_FIM']:.4f} > F_XY={obs['F_XY']:.4f} "
        f"= {obs['F_FIM_gt_F_XY']} (real observation; NOT coherence protection)."
    )
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
