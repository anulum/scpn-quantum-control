#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA-parity validation suite CLI
"""End-to-end CLI runner for the DLA-parity validation pathway.

Loads the published dataset, recomputes the statistical summaries,
verifies them against the published figures within tolerance, and
computes the noiseless classical reference. Prints a table and
exits non-zero on any tolerance or invariant breach.

Usage
-----

::

    python scripts/run_dla_parity_suite.py
    python scripts/run_dla_parity_suite.py --verify-integrity
    python scripts/run_dla_parity_suite.py --backend qutip
    python scripts/run_dla_parity_suite.py --data-dir /path/to/data
    python scripts/run_dla_parity_suite.py --json > result.json

The ``--json`` mode prints a machine-readable summary suitable for
downstream CI checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from scpn_quantum_control.dla_parity import (
    FullHarnessResult,
    available_baselines,
    run_full_harness,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the DLA-parity dataset validation suite end-to-end.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override default data/phase1_dla_parity/ location.",
    )
    p.add_argument(
        "--verify-integrity",
        action="store_true",
        help="Enforce SHA-256 digests against the embedded PUBLISHED_SHA256 table.",
    )
    p.add_argument(
        "--published-summary",
        type=Path,
        default=None,
        help="Override default figures/phase1/phase1_dla_parity_summary.json.",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "numpy", "qutip"),
        default="auto",
        help="Classical-baseline backend (default: auto).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout instead of a table.",
    )
    return p.parse_args(argv)


def _emit_table(result: FullHarnessResult) -> None:
    print("DLA-parity validation suite — summary")
    print(f"  dataset circuits:         {result.dataset.n_circuits_total}")
    print(f"  sub-phase runs:           {len(result.dataset.runs)}")
    print(f"  hardware backends:        {sorted(result.dataset.backends)}")
    print(
        f"  Fisher chi²:              {result.reproduction.fisher.chi2:.4f} "
        f"(df={result.reproduction.fisher.degrees_of_freedom}, "
        f"sig@0.05: {result.reproduction.fisher.n_depths_significant_at_0_05} / "
        f"{result.reproduction.fisher.n_depths_tested})",
    )
    print(
        f"  peak asymmetry:           "
        f"{100 * result.reproduction.peak_asymmetry_relative:+.2f}% "
        f"(depth={result.reproduction.peak_asymmetry_depth})",
    )
    print(f"  mean asymmetry:           {100 * result.reproduction.mean_asymmetry_relative:+.2f}%")
    print(
        f"  published claims checked: {len(result.reproduction.claims_checked)}; "
        f"all within tolerance.",
    )
    print(
        f"  classical reference:      backend={result.classical_reference.backend}, "
        f"max|leakage|={result.classical_reference.max_abs_leakage:.3e} "
        f"(zero within 1e-10: {result.classical_reference.is_zero_within_tolerance})",
    )
    print()
    print(
        f"{'depth':>6} {'even':>10} {'odd':>10} {'asym_rel':>10} {'welch_t':>10} {'welch_p':>12}"
    )
    for s in result.reproduction.depth_summaries:
        print(
            f"{s.depth:>6d} "
            f"{s.leakage_even:>10.4f} "
            f"{s.leakage_odd:>10.4f} "
            f"{100 * s.asymmetry_relative:>9.2f}% "
            f"{s.welch_t:>10.3f} "
            f"{s.welch_p:>12.3e}",
        )


def _emit_json(result: FullHarnessResult) -> None:
    payload = {
        "n_circuits": result.dataset.n_circuits_total,
        "backends": sorted(result.dataset.backends),
        "fisher": asdict(result.reproduction.fisher),
        "peak_asymmetry_relative": result.reproduction.peak_asymmetry_relative,
        "peak_asymmetry_depth": result.reproduction.peak_asymmetry_depth,
        "mean_asymmetry_relative": result.reproduction.mean_asymmetry_relative,
        "depth_summaries": [asdict(s) for s in result.reproduction.depth_summaries],
        "classical_reference": {
            "backend": result.classical_reference.backend,
            "max_abs_leakage": result.classical_reference.max_abs_leakage,
            "is_zero_within_tolerance": result.classical_reference.is_zero_within_tolerance,
        },
        "available_baselines": available_baselines(),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_full_harness(
            data_dir=args.data_dir,
            verify_integrity=args.verify_integrity,
            published_summary=args.published_summary,
            baselines_backend=args.backend,
        )
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"FAIL (missing data): {exc}", file=sys.stderr)
        return 3
    if args.json:
        _emit_json(result)
    else:
        _emit_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
