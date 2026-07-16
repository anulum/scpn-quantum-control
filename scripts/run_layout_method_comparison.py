#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — layout-method comparison run script (KT-3)
"""Run the layout-method comparison benchmark and emit an artifact.

Compares DynQ, DynQ + Kuramoto discrete optimiser, and SABRE on the synthetic
two-cluster calibration topology (the same shape as ``dynq_qubit_mapping.md``
§6.5): a low-error cluster, a high-error bridge coupler, and a noisier second
cluster. Writes a JSON artifact with measured depths, the analytic
success-probability model, and honest labelling, and prints the Markdown table
used in the documentation benchmarks section.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scpn_quantum_control.benchmarks.layout_method_comparison import (
    GateErrors,
    LayoutComparisonConfig,
    run_layout_method_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "layout_method_comparison"


def two_cluster_gate_errors() -> GateErrors:
    """Return the synthetic two-cluster calibration used by the comparison.

    A low-error 4-qubit cluster (qubits 0–3), a high-error bridge coupler
    (3, 4), and a noisier 4-qubit cluster (qubits 4–7). Synthetic by
    construction — labelled as such in the documentation, never presented as
    device calibration.

    Returns
    -------
    GateErrors
        Per-edge two-qubit gate errors.
    """
    return {
        (0, 1): 0.002,
        (1, 2): 0.003,
        (2, 3): 0.002,
        (0, 2): 0.004,
        (1, 3): 0.003,
        (3, 4): 0.05,
        (4, 5): 0.01,
        (5, 6): 0.012,
        (6, 7): 0.011,
        (4, 6): 0.013,
        (5, 7): 0.012,
    }


def two_cluster_readout_errors() -> dict[int, float]:
    """Return the synthetic per-qubit readout errors for the same topology."""
    return {qubit: 0.01 + 0.001 * qubit for qubit in range(8)}


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
    parser.add_argument("--seed", type=int, default=7, help="Shared DynQ/optimiser/SABRE seed.")
    parser.add_argument(
        "--reserved-core",
        type=int,
        default=0,
        help="CPU core whose isolation state sets the timing grade.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the layout-method comparison and write the artifact.

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

    artifact = run_layout_method_comparison(
        two_cluster_gate_errors(),
        K,
        omega,
        readout_errors=two_cluster_readout_errors(),
        config=LayoutComparisonConfig(
            t=args.t,
            reps=args.reps,
            seed=args.seed,
            reserved_core=args.reserved_core,
        ),
    )

    payload = artifact.to_dict()
    out_path = out_dir / f"layout_method_comparison_n{n}_seed{args.seed}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(artifact.render_markdown_table())
    print(f"timing_grade: {artifact.timing_grade}")
    for note in artifact.notes:
        print(f"  - {note}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
