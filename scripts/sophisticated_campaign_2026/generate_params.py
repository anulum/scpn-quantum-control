#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated campaign parameter generator
"""Fail-closed parameter cache generator for the sophisticated campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent


def _random_symmetric(rng: np.random.Generator, n: int) -> np.ndarray:
    matrix = rng.random((n, n))
    return (matrix + matrix.T) / 2.0


def _write_arrays(output_path: Path, arrays: dict[str, np.ndarray], seed: int) -> None:
    provenance = {
        "schema": "campaign.synthetic_parameters.v1",
        "campaign": "sophisticated_campaign_2026",
        "allow_synthetic": True,
        "seed": seed,
        "files": [],
    }
    for filename, payload in arrays.items():
        np.save(output_path / filename, payload)
        provenance["files"].append(
            {
                "filename": filename,
                "shape": list(payload.shape),
                "source_mode": "synthetic",
            }
        )
    (output_path / "PARAMETER_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def generate_all_params(
    output_dir: str | Path | None = None,
    *,
    allow_synthetic: bool = False,
    seed: int = 42,
) -> Path:
    """Generate labelled smoke parameters only when explicitly requested."""
    if not allow_synthetic:
        raise RuntimeError(
            "Refusing silent synthetic fallback. Pass --allow-synthetic only for "
            "labelled smoke-test parameters, not publication QPU inputs."
        )

    rng = np.random.default_rng(seed)
    output_path = Path(output_dir) if output_dir is not None else CAMPAIGN_DIR / "params"
    output_path.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "tokamak_Knm_16x16.npy": _random_symmetric(rng, 16),
        "tokamak_omega.npy": rng.random(16),
        "c_elegans_subnetwork_14x14.npy": _random_symmetric(rng, 14),
        "resource_Knm_12x12.npy": _random_symmetric(rng, 12),
        "internet_timing_20x20.npy": _random_symmetric(rng, 20),
        "thermo_Knm_16x16.npy": _random_symmetric(rng, 16),
        "hyper_pairwise.npy": _random_symmetric(rng, 12),
        "hyper_3body.npy": rng.random((12, 12, 12)),
        "hyper_directed.npy": rng.random((12, 12)),
        "logical_Knm_12x12.npy": _random_symmetric(rng, 12),
    }

    _write_arrays(output_path, arrays, seed)
    print(f"Sophisticated synthetic smoke parameters generated in {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sophisticated campaign parameters.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Generate labelled synthetic smoke-test parameters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_all_params(
        output_dir=args.output_dir,
        allow_synthetic=args.allow_synthetic,
        seed=args.seed,
    )
