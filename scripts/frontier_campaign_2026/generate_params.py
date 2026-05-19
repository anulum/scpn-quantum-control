#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Generate source-backed frontier campaign parameter matrices."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from campaign_io import campaign_path
from scpn_neurocore.bridge import (
    load_connectome,
    load_power_grid,
    load_tokamak_data,
)  # SC-NeuroCore bridges


def _matrix_from_source(
    value: Any, *, source_name: str, expected_n: int | None = None
) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("K_nm", value.get("knm"))
    if hasattr(value, "K_nm"):
        value = value.K_nm

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{source_name} must produce a square K_nm matrix, got {matrix.shape}")
    if expected_n is not None and matrix.shape != (expected_n, expected_n):
        raise ValueError(
            f"{source_name} must produce shape {(expected_n, expected_n)}, got {matrix.shape}"
        )
    return matrix


def _omega_from_source(value: Any, *, source_name: str, expected_n: int) -> np.ndarray | None:
    omega: Any = None
    if isinstance(value, tuple) and len(value) > 1:
        omega = value[1]
    elif isinstance(value, dict):
        omega = value.get("omega", value.get("omega_rad_s"))
    elif hasattr(value, "omega"):
        omega = value.omega

    if omega is None:
        return None

    vector = np.asarray(omega, dtype=np.float64)
    if vector.shape != (expected_n,):
        raise ValueError(
            f"{source_name} must produce omega shape {(expected_n,)}, got {vector.shape}"
        )
    return vector


def _synthetic_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.uniform(0.5, 2.0, size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _load_or_fail(
    loader: Any,
    *args: Any,
    source_name: str,
    expected_n: int | None = None,
    allow_synthetic: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    try:
        payload = loader(*args)
    except Exception as exc:
        if not allow_synthetic:
            raise RuntimeError(
                f"{source_name} unavailable. Refusing silent synthetic fallback; "
                "pass --allow-synthetic for smoke-test parameters only."
            ) from exc
        if expected_n is None:
            raise RuntimeError(
                f"{source_name} synthetic fallback requires an expected size"
            ) from exc
        return _synthetic_matrix(expected_n, rng), None, "synthetic"

    matrix = _matrix_from_source(payload, source_name=source_name, expected_n=expected_n)
    omega = _omega_from_source(payload, source_name=source_name, expected_n=matrix.shape[0])
    return matrix, omega, "bridge"


def generate_all_params(
    output_dir: str | Path | None = None,
    *,
    allow_synthetic: bool = False,
    seed: int = 42,
) -> None:
    """Write all frontier campaign parameter arrays and provenance metadata."""
    output_path = Path(output_dir) if output_dir is not None else campaign_path("params")
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    provenance: list[dict[str, Any]] = []

    sizes = [12, 14, 16, 20, 40, 80, 160]

    for N in sizes:
        source_name = "c_elegans_sub" if N == 14 else f"power_grid_{N}"
        loader = load_connectome if N == 14 else load_power_grid
        args = ("c_elegans_sub", N) if N == 14 else (N,)
        K, omega, source_mode = _load_or_fail(
            loader,
            *args,
            source_name=source_name,
            expected_n=N,
            allow_synthetic=allow_synthetic,
            rng=rng,
        )

        if omega is None:
            if not allow_synthetic:
                raise RuntimeError(
                    f"{source_name} did not provide omega. Refusing synthetic omega fallback; "
                    "pass --allow-synthetic for smoke-test parameters only."
                )
            omega = rng.normal(0.0, 0.3, N)
            omega_mode = "synthetic"
        else:
            omega_mode = source_mode

        matrix_path = f"scale_Knm_{N}x{N}.npy"
        omega_path = f"scale_omega_{N}.npy"
        arrays[matrix_path] = K
        arrays[omega_path] = omega
        provenance.extend(
            [
                {
                    "file": matrix_path,
                    "source_name": source_name,
                    "source_mode": source_mode,
                    "shape": list(K.shape),
                },
                {
                    "file": omega_path,
                    "source_name": source_name,
                    "source_mode": omega_mode,
                    "shape": list(omega.shape),
                },
            ]
        )

    # Specific applied matrices
    applied_sources = [
        ("c_elegans_subnetwork_14x14.npy", load_connectome, ("c_elegans_sub", 14), 14),
        ("tokamak_Knm_16x16.npy", load_tokamak_data, (), 16),
        ("distributed_Knm_20x20.npy", load_power_grid, (20,), 20),
        ("distill_Knm_12x12.npy", load_power_grid, (12,), 12),
        ("tn_Knm_64x64.npy", load_power_grid, (64,), 64),
        ("pt_Knm_12x12.npy", load_power_grid, (12,), 12),
        ("logical_Knm_12x12.npy", load_power_grid, (12,), 12),
        ("hyper_pairwise.npy", load_power_grid, (12,), 12),
    ]

    for filename, loader, args, expected_n in applied_sources:
        K, _, source_mode = _load_or_fail(
            loader,
            *args,
            source_name=filename,
            expected_n=expected_n,
            allow_synthetic=allow_synthetic,
            rng=rng,
        )
        arrays[filename] = K
        provenance.append(
            {
                "file": filename,
                "source_name": filename.removesuffix(".npy"),
                "source_mode": source_mode,
                "shape": list(K.shape),
            }
        )

    if allow_synthetic:
        synthetic_terms = {
            "hyper_3body.npy": rng.random((12, 12, 12)),
            "hyper_directed.npy": rng.random((12, 12)),
        }
        for filename, payload in synthetic_terms.items():
            arrays[filename] = payload
            provenance.append(
                {
                    "file": filename,
                    "source_name": filename.removesuffix(".npy"),
                    "source_mode": "synthetic",
                    "shape": list(payload.shape),
                }
            )

    for filename, payload in arrays.items():
        np.save(output_path / filename, payload)

    provenance_path = output_path / "PARAMETER_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "scpn-quantum-control.frontier-params.v1",
                "allow_synthetic": allow_synthetic,
                "seed": seed if allow_synthetic else None,
                "files": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    mode = "synthetic smoke-test" if allow_synthetic else "source-backed"
    print(f"Generated {mode} .npy files in {output_path} with provenance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate frontier campaign parameter files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow deterministic synthetic smoke-test arrays when source artifacts are missing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_all_params(args.output_dir, allow_synthetic=args.allow_synthetic, seed=args.seed)
