#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HLS co-simulation evidence run script (RC-3)
"""Run the HLS software co-simulation and emit the hash-bound handoff artifact.

Generates the pulse-player bundle for a canonical half-sine envelope, compiles
its testbench under the host compiler against the packaged non-synthesis shim,
executes it (bit-true ``PASS <n>`` verdict), and writes the hash-bound
evidence + SC-NEUROCORE handoff artifact. The artifact states its boundary
explicitly: codegen + software co-simulation only — no synthesis, no timing
closure, no board execution.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scpn_quantum_control.benchmarks.hls_cosimulation_evidence import (
    HLSCosimulationConfig,
    run_hls_cosimulation_handoff,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "hls_cosimulation"


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
    parser.add_argument("--samples", type=int, default=256, help="Envelope sample count.")
    parser.add_argument("--amplitude", type=float, default=0.8, help="Peak envelope amplitude.")
    parser.add_argument(
        "--sample-rate-hz", type=float, default=250e6, help="Pulse replay sample rate."
    )
    parser.add_argument(
        "--target-sku", choices=("zu3eg", "zu9eg"), default="zu3eg", help="Target device SKU."
    )
    parser.add_argument("--compiler", default="g++", help="Host compiler executable.")
    parser.add_argument(
        "--reserved-core",
        type=int,
        default=0,
        help="CPU core whose isolation state sets the timing grade.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the co-simulation evidence build and write the artifact.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    int
        Process exit code (``0`` when the co-simulation passed, ``1``
        otherwise — the artifact is written either way; failure evidence is
        still evidence).
    """
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = run_hls_cosimulation_handoff(
        HLSCosimulationConfig(
            n_samples=args.samples,
            amplitude=args.amplitude,
            sample_rate_hz=args.sample_rate_hz,
            target_sku=args.target_sku,
            compiler=args.compiler,
            reserved_core=args.reserved_core,
        )
    )

    payload = artifact.to_dict()
    out_path = out_dir / f"hls_cosimulation_{args.target_sku}_n{args.samples}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    evidence = payload["evidence"]
    print(f"co-simulation passed: {evidence['passed']} ({evidence['samples_streamed']} samples)")
    print(f"compiler: {evidence['compiler_version']}")
    print(f"timing_grade: {payload['timing_grade']}")
    for note in payload["notes"]:
        print(f"  - {note}")
    print(f"written: {out_path}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
