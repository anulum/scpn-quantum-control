#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — manifest-bound UltraScale+ HLS artifact exporter
"""Export a versioned UltraScale+ HLS artifact directory.

The runner emits source assets plus a manifest for SC-NEUROCORE's decoupled
``hdl_gen.hls_ingest`` lane. It does not invoke Vivado or claim timing closure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from scpn_quantum_control.codegen.ultrascale_hls import (
    TargetSku,
    emit_versioned_hls_artifact,
    verify_hls_artifact_manifest,
)
from scpn_quantum_control.phase.pulse_shaping import build_hypergeometric_pulse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ultrascale_hls_artifacts"),
        help="parent directory for the versioned artifact directory",
    )
    parser.add_argument(
        "--artifact-id",
        default="ultrascale-hls-pulse-axi-v1",
        help="single directory segment used as the artifact identifier",
    )
    parser.add_argument("--target-sku", choices=("zu3eg", "zu9eg"), default="zu3eg")
    parser.add_argument("--sample-rate-hz", type=float, default=125e6)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--fifo-depth", type=int, default=1024)
    parser.add_argument("--fixed-point-width", type=int, default=16)
    parser.add_argument("--fixed-point-frac-bits", type=int, default=8)
    parser.add_argument("--t-total", type=float, default=1.0)
    parser.add_argument("--omega-0", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser


def main() -> None:
    """Run the HLS artifact exporter."""
    args = _parser().parse_args()
    pulse = build_hypergeometric_pulse(
        t_total=args.t_total,
        omega_0=args.omega_0,
        alpha=args.alpha,
        beta=args.beta,
        n_points=args.n_samples,
    )
    manifest = emit_versioned_hls_artifact(
        pulse.envelope,
        args.output_dir,
        artifact_id=args.artifact_id,
        sample_rate_hz=args.sample_rate_hz,
        target_sku=cast(TargetSku, args.target_sku),
        fifo_depth=args.fifo_depth,
        fixed_point_width=args.fixed_point_width,
        fixed_point_frac_bits=args.fixed_point_frac_bits,
    )
    manifest_path = args.output_dir / args.artifact_id / "manifest.json"
    verification = verify_hls_artifact_manifest(manifest_path)
    if not verification.valid:
        raise SystemExit("manifest verification failed: " + "; ".join(verification.errors))

    print(f"artifact_id: {manifest.artifact_id}")
    print(f"manifest: {manifest_path}")
    print(f"consumer_contract_version: {manifest.consumer_contract_version}")
    print(f"schema_version: {manifest.schema_version}")
    print("files:")
    for file_record in manifest.files:
        print(f"  - {file_record.role}: {file_record.path} {file_record.sha256}")
    print(f"claim_boundary: {manifest.claim_boundary}")


if __name__ == "__main__":
    main()
