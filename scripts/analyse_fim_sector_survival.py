#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM sector-survival artefact harness
"""Generate conservative sector-conservation artefacts for H_XY + H_FIM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

from scpn_quantum_control.analysis.fim_hamiltonian import (
    add_fim_feedback,
    commutator_frobenius_norm_with_diagonal,
    magnetisation_operator_diagonal,
    sector_coupling_rows,
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


def generate(n_values: list[int], lambdas: list[float]) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        k_matrix = build_knm_paper27(n_qubits)
        omega = OMEGA_N_16[:n_qubits]
        base_hamiltonian = knm_to_dense_matrix(k_matrix, omega)
        magnetisation = magnetisation_operator_diagonal(n_qubits)
        magnetisation_squared = magnetisation**2
        for lambda_fim in lambdas:
            hamiltonian = add_fim_feedback(base_hamiltonian, lambda_fim)
            comm_m = commutator_frobenius_norm_with_diagonal(hamiltonian, magnetisation)
            comm_m2 = commutator_frobenius_norm_with_diagonal(hamiltonian, magnetisation_squared)
            rows = sector_coupling_rows(hamiltonian, lambda_fim)
            max_off_sector = max(float(row["off_sector_frobenius_norm"]) for row in rows)
            summary_rows.append(
                {
                    "n_qubits": n_qubits,
                    "lambda_fim": float(lambda_fim),
                    "commutator_norm_M": comm_m,
                    "commutator_norm_M_squared": comm_m2,
                    "max_off_sector_frobenius_norm": max_off_sector,
                    "ideal_unitary_sector_leakage": 0.0,
                }
            )
            sector_rows.extend(rows)
    return {
        "schema": "scpn_fim_sector_survival_v1",
        "date": DATE,
        "command": "python scripts/analyse_fim_sector_survival.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim_boundary": (
            "The ideal dense Hamiltonian conserves total magnetisation. This artefact "
            "therefore supports sector-conservation and energy-barrier claims only. "
            "Any IBM sector leakage or survival asymmetry must be attributed to "
            "noise, state preparation, transpilation, layout, readout, or a separate "
            "open-system model, not to ideal H_XY + H_FIM unitary leakage."
        ),
        "n_values": n_values,
        "lambda_values": lambdas,
        "summary_rows": summary_rows,
        "sector_rows": sector_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4,6,8")
    parser.add_argument("--lambdas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    lambdas = _parse_csv_floats(ns.lambdas)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(n_values, lambdas)

    json_path = ns.output_dir / f"fim_sector_survival_prediction_{DATE}.json"
    summary_csv = ns.output_dir / f"fim_sector_survival_summary_{DATE}.csv"
    sector_csv = ns.output_dir / f"fim_sector_survival_rows_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(summary_csv, list(summary["summary_rows"]))
    _write_csv(sector_csv, list(summary["sector_rows"]))
    print(f"wrote_json={json_path}")
    print(f"wrote_summary_csv={summary_csv}")
    print(f"wrote_sector_csv={sector_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_summary_csv={_sha256(summary_csv)}")
    print(f"sha256_sector_csv={_sha256(sector_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
