# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological QEC CLI
"""CLI utilities for biological surface-code execution artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .biological_pipeline import run_biological_qec_execution


def _load_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix in {".csv", ".txt"}:
        matrix = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("K input must be .npy, .csv, or .txt")
    return np.asarray(matrix, dtype=float)


def _load_vector(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        vector = np.load(path)
    elif suffix in {".csv", ".txt"}:
        vector = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("z_errors input must be .npy, .csv, or .txt")
    arr = np.asarray(vector, dtype=np.int8)
    if arr.ndim != 1:
        raise ValueError("z_errors must be a one-dimensional vector.")
    return arr


def _load_domains(path: Path | None) -> dict[int, str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    mapping: dict[int, str] = {}
    for k, v in payload.items():
        mapping[int(k)] = str(v)
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scpn-biological-qec-report",
        description="Generate biological surface-code execution artefact payload.",
    )
    parser.add_argument(
        "--k", required=True, type=Path, help="Coupling matrix file (.npy/.csv/.txt)"
    )
    parser.add_argument(
        "--z-errors",
        required=True,
        type=Path,
        help="Edge error vector file (.npy/.csv/.txt)",
    )
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument(
        "--domains",
        type=Path,
        default=None,
        help='Optional JSON node-domain mapping, e.g. {"0":"L1"}',
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON metadata payload",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    K = _load_matrix(args.k)
    z_errors = _load_vector(args.z_errors)
    node_domains = _load_domains(args.domains)
    metadata: dict[str, Any] | None = None
    if args.metadata is not None:
        with args.metadata.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    result = run_biological_qec_execution(
        K,
        z_errors,
        threshold=float(args.threshold),
        node_domains=node_domains,
        metadata=metadata,
    )
    payload = result.to_payload()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


__all__ = ["build_parser", "main"]
