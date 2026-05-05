#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM level-spacing artefact harness
"""Generate adjacent-gap-ratio artefacts for H_XY + H_FIM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from scpn_quantum_control.analysis.fim_hamiltonian import (
    add_fim_feedback,
    adjacent_gap_ratio,
    magnetisation_sector_indices,
)
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_dense_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(n_values: list[int], lambdas: list[float], tolerance: float) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        k_matrix = build_knm_paper27(n_qubits)
        omega = OMEGA_N_16[:n_qubits]
        base_hamiltonian = knm_to_dense_matrix(k_matrix, omega)
        sectors = magnetisation_sector_indices(n_qubits)
        for lambda_fim in lambdas:
            hamiltonian = add_fim_feedback(base_hamiltonian, lambda_fim)
            full_stats = adjacent_gap_ratio(np.linalg.eigvalsh(hamiltonian), tolerance)
            rows.append(
                {
                    "n_qubits": n_qubits,
                    "lambda_fim": float(lambda_fim),
                    "scope": "full_spectrum",
                    "magnetisation": None,
                    **full_stats,
                }
            )
            for magnetisation, indices in sectors.items():
                block = hamiltonian[np.ix_(indices, indices)]
                sector_stats = adjacent_gap_ratio(np.linalg.eigvalsh(block), tolerance)
                rows.append(
                    {
                        "n_qubits": n_qubits,
                        "lambda_fim": float(lambda_fim),
                        "scope": "magnetisation_sector",
                        "magnetisation": int(magnetisation),
                        **sector_stats,
                    }
                )
    return {
        "schema": "scpn_fim_level_spacing_v1",
        "date": DATE,
        "command": "python scripts/analyse_fim_level_spacing.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "tolerance": tolerance,
        "claim_boundary": (
            "Adjacent-gap ratios are exact small-n diagnostics. They are evidence "
            "about spectral structure only and do not by themselves prove strict "
            "many-body localisation or hardware protection."
        ),
        "n_values": n_values,
        "lambda_values": lambdas,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4,6,8")
    parser.add_argument("--lambdas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    lambdas = _parse_csv_floats(ns.lambdas)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(n_values, lambdas, ns.tolerance)

    json_path = ns.output_dir / f"fim_level_spacing_summary_{DATE}.json"
    csv_path = ns.output_dir / f"fim_level_spacing_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(summary["rows"]))
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
