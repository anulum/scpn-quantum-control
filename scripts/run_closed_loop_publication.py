#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — closed-loop publication artifact run script (RC-2)
"""Run the reproducible software-in-the-loop closed-loop publication package and emit an artifact.

Measures the software-in-the-loop latency report (local statevector simulator
wall clock — a software budget, not a hardware measurement), builds the
publication scaffold with its fail-closed claim ledger, and exports the
dynamic-circuit templates (mid-circuit measurement + ``if_test`` conditionals)
as OpenQASM 3 — exportable but un-run. Writes one JSON artifact and prints the
publication scaffold summary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.benchmarks.closed_loop_publication_run import (
    ClosedLoopRunConfig,
    run_closed_loop_publication,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "closed_loop_publication"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options controlling the run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=4, help="Ring size (Kuramoto oscillators).")
    parser.add_argument("--coupling", type=float, default=0.6, help="Ring coupling strength.")
    parser.add_argument("--target-r", type=float, default=0.6, help="Order-parameter setpoint.")
    parser.add_argument(
        "--rounds", type=int, default=32, help="Measured software-in-the-loop feedback rounds."
    )
    parser.add_argument(
        "--template-rounds",
        type=int,
        default=3,
        help="Feedback rounds in the exported dynamic-circuit templates.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Ring and controller seed.")
    parser.add_argument(
        "--reserved-core",
        type=int,
        default=0,
        help="CPU core whose isolation state sets the timing grade.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the closed-loop publication package and write the artifact.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = run_closed_loop_publication(
        ClosedLoopRunConfig(
            n_oscillators=args.n,
            coupling=args.coupling,
            target_r=args.target_r,
            n_rounds=args.rounds,
            dynamic_circuit_rounds=args.template_rounds,
            seed=args.seed,
            reserved_core=args.reserved_core,
        )
    )

    payload = artifact.to_dict()
    out_path = out_dir / f"closed_loop_publication_n{args.n}_seed{args.seed}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(artifact.package_markdown)
    print(f"timing_grade: {artifact.timing_grade}")
    for note in artifact.notes:
        print(f"  - {note}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
