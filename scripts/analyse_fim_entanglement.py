#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM entanglement artefact harness
"""Generate exact eigenstate bipartition-entropy artefacts for H_XY + H_FIM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import cast

import numpy as np

from scpn_quantum_control.analysis.fim_hamiltonian import (
    add_fim_feedback,
    bipartite_entropy_from_statevector,
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


def generate(n_values: list[int], lambdas: list[float], eigenstates: int) -> dict[str, object]:
    """Generate exact low-energy bipartition entropy summaries.

    The calculation uses the committed Paper-27 coupling builder and the FIM
    feedback Hamiltonian path, then diagonalises each requested small system
    exactly. The returned payload is an artefact bundle for reproducibility and
    manuscript support; its claim boundary is restricted to exact small-system
    diagnostics.
    """
    rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        k_matrix = build_knm_paper27(n_qubits)
        omega = OMEGA_N_16[:n_qubits]
        base_hamiltonian = knm_to_dense_matrix(k_matrix, omega)
        keep = list(range(max(1, n_qubits // 2)))
        for lambda_fim in lambdas:
            hamiltonian = add_fim_feedback(base_hamiltonian, lambda_fim)
            values, vectors = np.linalg.eigh(hamiltonian)
            limit = min(eigenstates, values.size)
            entropies: list[float] = []
            for index in range(limit):
                entropy = bipartite_entropy_from_statevector(vectors[:, index], n_qubits, keep)
                entropies.append(entropy)
                rows.append(
                    {
                        "n_qubits": n_qubits,
                        "lambda_fim": float(lambda_fim),
                        "eigenstate_index": index,
                        "energy": float(values[index]),
                        "bipartition_keep": ",".join(str(item) for item in keep),
                        "entropy_bits": entropy,
                    }
                )
            aggregate_rows.append(
                {
                    "n_qubits": n_qubits,
                    "lambda_fim": float(lambda_fim),
                    "n_eigenstates": limit,
                    "ground_entropy_bits": entropies[0] if entropies else None,
                    "mean_low_energy_entropy_bits": float(np.mean(entropies))
                    if entropies
                    else None,
                    "max_low_energy_entropy_bits": float(np.max(entropies)) if entropies else None,
                }
            )
    return {
        "schema": "scpn_fim_entanglement_v1",
        "date": DATE,
        "command": "python scripts/analyse_fim_entanglement.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim_boundary": (
            "Exact small-n bipartition entropies of low-energy eigenstates. These "
            "diagnostics do not by themselves prove hardware robustness or strict "
            "many-body localisation."
        ),
        "n_values": n_values,
        "lambda_values": lambdas,
        "eigenstates_per_hamiltonian": eigenstates,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
    }


def main() -> int:
    """Run the FIM entanglement artefact generator from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4,6,8")
    parser.add_argument("--lambdas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--eigenstates", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    lambdas = _parse_csv_floats(ns.lambdas)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(n_values, lambdas, ns.eigenstates)

    json_path = ns.output_dir / f"fim_entanglement_summary_{DATE}.json"
    rows_csv = ns.output_dir / f"fim_entanglement_rows_{DATE}.csv"
    aggregate_csv = ns.output_dir / f"fim_entanglement_aggregate_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(rows_csv, list(cast(list[dict[str, object]], summary["rows"])))
    _write_csv(aggregate_csv, list(cast(list[dict[str, object]], summary["aggregate_rows"])))
    print(f"wrote_json={json_path}")
    print(f"wrote_rows_csv={rows_csv}")
    print(f"wrote_aggregate_csv={aggregate_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_rows_csv={_sha256(rows_csv)}")
    print(f"sha256_aggregate_csv={_sha256(aggregate_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
