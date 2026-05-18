#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM spectrum artefact harness
"""Generate exact spectrum and magnetisation-sector artefacts for H_XY + H_FIM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from scpn_quantum_control.analysis.fim_hamiltonian import (
    add_fim_feedback,
    sector_spectrum_rows,
    summarise_spectrum,
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
    """Generate exact spectrum and magnetisation-sector summary artefacts.

    The output captures small-system eigenvalue summaries for the XY Hamiltonian
    with FIM feedback. It is a reproducibility and manuscript-support artefact,
    not a hardware-evidence or quantum-advantage result.
    """
    spectrum_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        k_matrix = build_knm_paper27(n_qubits)
        omega = OMEGA_N_16[:n_qubits]
        base_hamiltonian = knm_to_dense_matrix(k_matrix, omega)
        for lambda_fim in lambdas:
            hamiltonian = add_fim_feedback(base_hamiltonian, lambda_fim)
            eigenvalues = np.linalg.eigvalsh(hamiltonian)
            row = asdict(summarise_spectrum(eigenvalues, n_qubits, lambda_fim))
            row.update(
                {
                    "k_base": 0.45,
                    "k_alpha": 0.3,
                    "omega_source": "OMEGA_N_16_prefix",
                    "hamiltonian": "H_XY - lambda*M^2/n",
                }
            )
            spectrum_rows.append(row)
            sector_rows.extend(sector_spectrum_rows(hamiltonian, lambda_fim))
    return {
        "schema": "scpn_fim_spectrum_v1",
        "date": DATE,
        "command": "python scripts/analyse_fim_spectrum.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim_boundary": (
            "Exact small-n offline spectra only. These artefacts do not establish "
            "hardware robustness, quantum advantage, or strict many-body localisation."
        ),
        "n_values": n_values,
        "lambda_values": lambdas,
        "spectrum_rows": spectrum_rows,
        "sector_rows": sector_rows,
    }


def main() -> int:
    """Run the spectrum artefact generator from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4,6,8")
    parser.add_argument("--lambdas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    lambdas = _parse_csv_floats(ns.lambdas)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(n_values, lambdas)

    json_path = ns.output_dir / f"fim_spectrum_summary_{DATE}.json"
    spectrum_csv = ns.output_dir / f"fim_spectrum_summary_{DATE}.csv"
    sector_csv = ns.output_dir / f"fim_sector_spectrum_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(spectrum_csv, list(cast(list[dict[str, object]], summary["spectrum_rows"])))
    _write_csv(sector_csv, list(cast(list[dict[str, object]], summary["sector_rows"])))
    print(f"wrote_json={json_path}")
    print(f"wrote_spectrum_csv={spectrum_csv}")
    print(f"wrote_sector_csv={sector_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_spectrum_csv={_sha256(spectrum_csv)}")
    print(f"sha256_sector_csv={_sha256(sector_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
