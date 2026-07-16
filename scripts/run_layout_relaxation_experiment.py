#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KT-4 relaxation seed-sweep experiment run script
"""Run the preregistered KT-4 relaxation seed-sweep experiment (RESEARCH).

Executes the preregistered instances — the synthetic two-cluster topology
under a seed sweep with both search arms in the DynQ region, plus one
full-device instance where the candidate set is at least twice the logical
width — and writes the experiment artifact with the verdict against the
preregistered null hypothesis. "No gain" over the KT-3 discrete baseline is
a valid, publishable outcome; nothing here promotes the relaxation beyond a
research observation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scpn_quantum_control.benchmarks.layout_method_comparison import (  # noqa: E402
    LayoutComparisonConfig,
)
from scpn_quantum_control.benchmarks.layout_relaxation_experiment import (  # noqa: E402
    preregistered_instances,
    run_layout_relaxation_experiment,
)
from scripts.run_layout_method_comparison import (  # noqa: E402
    two_cluster_gate_errors,
    two_cluster_readout_errors,
)

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "layout_relaxation_experiment"


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
    parser.add_argument("--n", type=int, default=4, help="Logical qubit count of the XY problem.")
    parser.add_argument("--t", type=float, default=0.1, help="Evolution time.")
    parser.add_argument("--reps", type=int, default=5, help="Trotter repetitions.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seed sweep for the DynQ-region instances (preregistered default 0..9).",
    )
    parser.add_argument(
        "--full-device-seed",
        type=int,
        default=0,
        help="Seed of the single full-device (m >= 2n) instance.",
    )
    parser.add_argument(
        "--reserved-core",
        type=int,
        default=0,
        help="CPU core whose isolation state sets the timing grade.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preregistered experiment and write the artifact.

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

    n = int(args.n)
    K = (np.ones((n, n)) - np.eye(n)).astype(np.float64)
    omega = np.linspace(0.1, 0.4, n, dtype=np.float64)
    seeds = tuple(int(seed) for seed in args.seeds)

    artifact = run_layout_relaxation_experiment(
        two_cluster_gate_errors(),
        K,
        omega,
        readout_errors=two_cluster_readout_errors(),
        base_config=LayoutComparisonConfig(
            t=args.t,
            reps=args.reps,
            reserved_core=args.reserved_core,
        ),
        instances=preregistered_instances(
            seeds=seeds, full_device_seed=int(args.full_device_seed)
        ),
    )

    payload = artifact.to_dict()
    seed_span = f"{min(seeds)}-{max(seeds)}" if len(seeds) > 1 else str(seeds[0])
    out_path = out_dir / f"layout_relaxation_experiment_n{n}_seeds{seed_span}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(artifact.render_markdown_table())
    print(
        f"baseline mean±std: {artifact.baseline_mean_cost:.6f} ± {artifact.baseline_cost_std:.6f}"
    )
    print(
        "relaxation mean±std: "
        f"{artifact.relaxation_mean_cost:.6f} ± {artifact.relaxation_cost_std:.6f}"
    )
    print(f"wins/ties/losses: {artifact.wins}/{artifact.ties}/{artifact.losses}")
    print(f"verdict: {artifact.verdict}")
    print(f"timing_grade: {artifact.timing_grade}")
    for note in artifact.notes:
        print(f"  - {note}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
