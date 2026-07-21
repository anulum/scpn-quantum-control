# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer primary decision-rule analysis
"""Frozen primary decision rule for the IQM layout-transfer campaign.

Implements exactly the preregistered rule of
``docs/campaigns/iqm_layout_transfer_square_lattice_prereg_2026-07-21.md``:

- per size, readout-corrected order-parameter error
  ``err(arm, n) = |R_hw(arm, n) − R_exact(n)|``;
- primary passes iff ``err(optimised) < err(default)`` at ALL preregistered
  sizes AND the bootstrap 90 % confidence interval (10,000 shot-level
  resamples, frozen seed) of the pooled error difference
  ``mean_n[err(default) − err(optimised)]`` excludes zero;
- secondary: the same comparison for ``default`` versus ``naive``.

The bootstrap resamples every circuit's counts (main arms and both readout
calibration circuits) as multinomials, so shot noise in the readout
correction propagates into the interval. Runs in the main ``.venv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark import (  # noqa: E402
    corrected_order_parameter,
    per_qubit_readout_errors,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260721
CONFIDENCE = 0.90


def _resample(counts: dict[str, int], rng: np.random.Generator) -> dict[str, int]:
    keys = list(counts.keys())
    values = np.array([counts[k] for k in keys], dtype=np.int64)
    total = int(values.sum())
    drawn = rng.multinomial(total, values / total)
    return {k: int(v) for k, v in zip(keys, drawn, strict=True) if v > 0}


def _block_errors(
    block: dict[str, Any],
    counts: dict[str, dict[str, int]],
    rng: np.random.Generator | None,
) -> dict[str, float]:
    """Readout-corrected error per arm for one size (optionally resampled)."""
    n = block["n"]
    readout_qubits = tuple(int(q) for q in block["readout_qubits"])
    zeros = counts[f"readout_n{n}_zeros"]
    ones = counts[f"readout_n{n}_ones"]
    if rng is not None:
        zeros, ones = _resample(zeros, rng), _resample(ones, rng)
    e01, e10 = per_qubit_readout_errors(zeros, ones, readout_qubits)
    errors: dict[str, float] = {}
    for arm in block["arms"]:
        measured = tuple(int(q) for q in arm["measured_qubits"])
        arm_counts = counts[f"main_n{n}_{arm['arm']}"]
        if rng is not None:
            arm_counts = _resample(arm_counts, rng)
        value = corrected_order_parameter(arm_counts, measured, e01, e10)
        errors[arm["arm"]] = abs(value - float(block["exact_reference"]))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Evaluate the frozen primary rule and write the analysis artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="live plan artefact JSON")
    parser.add_argument(
        "--counts",
        required=True,
        nargs="+",
        help="hardware counts JSONs (one per size block, any order)",
    )
    parser.add_argument("--out", required=True, help="output analysis JSON")
    args = parser.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    job_ids: list[str] = []
    backends: set[str] = set()
    for path in args.counts:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        counts.update(payload["counts"])
        job_ids.extend(payload.get("job_ids", []))
        backends.add(str(payload.get("backend", "unknown")))

    blocks = [b for b in plan["blocks"] if f"main_n{b['n']}_optimised" in counts]
    if len(blocks) != len(plan["blocks"]):
        missing = [b["n"] for b in plan["blocks"] if b not in blocks]
        print(
            f"note: counts cover {len(blocks)} of {len(plan['blocks'])} sizes "
            f"(missing n={missing}) — primary rule needs all sizes"
        )

    point = {block["n"]: _block_errors(block, counts, None) for block in blocks}
    wins_primary = all(e["optimised"] < e["default"] for e in point.values())
    wins_secondary = all(e["default"] < e["naive"] for e in point.values())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pooled_primary = np.empty(BOOTSTRAP_RESAMPLES)
    pooled_secondary = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        diffs_p, diffs_s = [], []
        for block in blocks:
            errors = _block_errors(block, counts, rng)
            diffs_p.append(errors["default"] - errors["optimised"])
            diffs_s.append(errors["naive"] - errors["default"])
        pooled_primary[i] = float(np.mean(diffs_p))
        pooled_secondary[i] = float(np.mean(diffs_s))

    lo, hi = (1.0 - CONFIDENCE) / 2.0 * 100.0, (1.0 + CONFIDENCE) / 2.0 * 100.0
    ci_primary = [float(np.percentile(pooled_primary, q)) for q in (lo, hi)]
    ci_secondary = [float(np.percentile(pooled_secondary, q)) for q in (lo, hi)]
    complete = len(blocks) == len(plan["blocks"])
    primary_passes = bool(
        complete and wins_primary and (ci_primary[0] > 0.0 or ci_primary[1] < 0.0)
    )

    report = {
        "campaign": plan["campaign"],
        "kind": "primary_decision_rule",
        "backends": sorted(backends),
        "job_ids": job_ids,
        "sizes_covered": [b["n"] for b in blocks],
        "matrix_complete": complete,
        "point_errors": {str(n): e for n, e in point.items()},
        "pooled_difference_default_minus_optimised": {
            "point": float(np.mean([e["default"] - e["optimised"] for e in point.values()])),
            "bootstrap_ci90": ci_primary,
        },
        "pooled_difference_naive_minus_default": {
            "point": float(np.mean([e["naive"] - e["default"] for e in point.values()])),
            "bootstrap_ci90": ci_secondary,
        },
        "wins_all_sizes_optimised_vs_default": wins_primary,
        "wins_all_sizes_default_vs_naive": wins_secondary,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "primary_rule_passes": primary_passes,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"analysis: {out_path}")
    for n, errors in sorted(point.items()):
        print(f"n={n}: " + " ".join(f"{arm}={err:.4f}" for arm, err in errors.items()))
    print(f"pooled default−optimised: {report['pooled_difference_default_minus_optimised']}")
    print(f"pooled naive−default:     {report['pooled_difference_naive_minus_default']}")
    print(f"PRIMARY RULE PASSES: {primary_passes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
