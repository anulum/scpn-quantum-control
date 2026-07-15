#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — decisive-advantage classical-baseline run script
"""Run the classical baselines for the decisive-advantage gate and emit an artifact.

Without ``--run`` the script emits the preregistered protocol manifest — the
public commitment before any measurement. With ``--run`` it executes the
classical baselines at the preregistered decision size and writes a reproducible
artifact (measured timings, provenance, host-isolation grade) plus the
fail-closed decision label. With no QPU credits the decision is ``inconclusive``
and no quantum advantage is claimed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.benchmarks.decisive_advantage_protocol import (
    default_decisive_advantage_protocol,
)
from scpn_quantum_control.benchmarks.decisive_run_harness import (
    DecisiveRunConfig,
    run_decisive_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "decisive_advantage_gate"


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
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the classical baselines; omit for a preregistration-only manifest.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-max", type=float, default=1.0, help="Total evolution time.")
    parser.add_argument("--dt", type=float, default=0.1, help="Evolution time step.")
    parser.add_argument("--mps-bond-dim", type=int, default=32, help="MPS TEBD bond dimension.")
    parser.add_argument(
        "--no-mps",
        action="store_true",
        help="Emit the MPS row as a configuration-gated skip instead of running it.",
    )
    parser.add_argument(
        "--reserved-core",
        type=int,
        default=0,
        help="CPU core whose isolation state sets the timing grade.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the decisive-advantage classical-baseline harness.

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
    protocol = default_decisive_advantage_protocol()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.run:
        payload = {"protocol": protocol.to_dict()}
        out_path = out_dir / f"{protocol.protocol.protocol_id}.manifest.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"protocol: {protocol.protocol.protocol_id}")
        print("preregistration only (pass --run to measure the classical baselines)")
        print(f"written: {out_path}")
        return 0

    config = DecisiveRunConfig(
        t_max=args.t_max,
        dt=args.dt,
        mps_bond_dim=args.mps_bond_dim,
        include_mps=not args.no_mps,
        reserved_core=args.reserved_core,
    )
    artifact = run_decisive_benchmark(protocol, config)
    payload = artifact.to_dict()
    out_path = out_dir / f"{protocol.protocol.protocol_id}.artifact.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"protocol: {protocol.protocol.protocol_id}")
    print(f"timing_grade: {payload['timing_grade']}")
    print(f"decision: {payload['decision']['label']}")
    for reason in payload["decision"]["reasons"]:
        print(f"  - {reason}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
