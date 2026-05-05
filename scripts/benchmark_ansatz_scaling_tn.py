#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- ansatz scaling and tensor-network baseline harness
"""Generate ansatz-scaling and MPS-truncation diagnostics for n=4--12."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import efficient_su2, n_local

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_dense_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _make_ansatz(name: str, n_qubits: int, reps: int):
    k_matrix = build_knm_paper27(n_qubits)
    if name == "knm_informed":
        return knm_to_ansatz(k_matrix, reps=reps)
    if name == "two_local":
        return n_local(
            n_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            reps=reps,
        )
    if name == "efficient_su2":
        return efficient_su2(n_qubits, reps=reps)
    raise ValueError(name)


def ansatz_scaling_rows(n_values: list[int], reps_values: list[int]) -> list[dict[str, object]]:
    """Return circuit-size rows for the three ansatz families."""

    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        for reps in reps_values:
            for name in ["knm_informed", "two_local", "efficient_su2"]:
                circuit = _make_ansatz(name, n_qubits, reps)
                transpiled = transpile(
                    circuit,
                    basis_gates=["rz", "sx", "x", "cx"],
                    optimization_level=1,
                )
                ops = circuit.count_ops()
                tops = transpiled.count_ops()
                rows.append(
                    {
                        "ansatz": name,
                        "n_qubits": n_qubits,
                        "reps": reps,
                        "parameters": circuit.num_parameters,
                        "raw_depth": circuit.depth(),
                        "raw_two_qubit_gates": int(
                            sum(v for k, v in ops.items() if k in {"cx", "cz", "rzz", "ecr"})
                        ),
                        "transpiled_depth": transpiled.depth(),
                        "transpiled_two_qubit_gates": int(
                            sum(v for k, v in tops.items() if k in {"cx", "cz", "rzz", "ecr"})
                        ),
                    }
                )
    return rows


def _ground_state(n_qubits: int) -> tuple[float, np.ndarray]:
    k_matrix = build_knm_paper27(n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    hamiltonian = knm_to_dense_matrix(k_matrix, omega)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=np.complex128)


def _schmidt_values(state: np.ndarray, n_qubits: int, cut: int) -> np.ndarray:
    left_dim = 2**cut
    right_dim = 2 ** (n_qubits - cut)
    matrix = state.reshape(left_dim, right_dim)
    return np.linalg.svd(matrix, compute_uv=False)


def _discarded_weight(singular_values: np.ndarray, max_bond: int) -> float:
    if max_bond >= singular_values.size:
        return 0.0
    tail = singular_values[max_bond:]
    return float(np.sum(np.abs(tail) ** 2))


def mps_truncation_rows(
    n_values: list[int],
    max_bonds: list[int],
    exact_max_qubits: int,
) -> list[dict[str, object]]:
    """Return exact-ground-state MPS truncation diagnostics where feasible."""

    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        if n_qubits > exact_max_qubits:
            rows.append(
                {
                    "n_qubits": n_qubits,
                    "status": "skipped",
                    "reason": "above_exact_max_qubits",
                    "exact_max_qubits": exact_max_qubits,
                    "ground_energy": None,
                    "max_bond": None,
                    "worst_cut_discarded_weight": None,
                    "max_midchain_entropy_bits": None,
                }
            )
            continue
        ground_energy, state = _ground_state(n_qubits)
        cut_spectra = [_schmidt_values(state, n_qubits, cut) for cut in range(1, n_qubits)]
        entropies = []
        for spectrum in cut_spectra:
            probabilities = np.square(np.abs(spectrum))
            nonzero = probabilities[probabilities > 1e-15]
            entropies.append(float(-np.sum(nonzero * np.log2(nonzero))))
        for max_bond in max_bonds:
            rows.append(
                {
                    "n_qubits": n_qubits,
                    "status": "ok",
                    "reason": None,
                    "exact_max_qubits": exact_max_qubits,
                    "ground_energy": ground_energy,
                    "max_bond": max_bond,
                    "worst_cut_discarded_weight": float(
                        max(_discarded_weight(spectrum, max_bond) for spectrum in cut_spectra)
                    ),
                    "max_midchain_entropy_bits": float(max(entropies)),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4,6,8,10,12")
    parser.add_argument("--reps-values", default="1,2")
    parser.add_argument("--max-bonds", default="4,8,16")
    parser.add_argument(
        "--exact-max-qubits",
        type=int,
        default=8,
        help="Largest n for exact ground-state generation before rows are marked skipped.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    reps_values = _parse_csv_ints(ns.reps_values)
    max_bonds = _parse_csv_ints(ns.max_bonds)
    ns.output_dir.mkdir(parents=True, exist_ok=True)

    ansatz_rows = ansatz_scaling_rows(n_values, reps_values)
    tn_rows = mps_truncation_rows(n_values, max_bonds, ns.exact_max_qubits)
    summary = {
        "date": DATE,
        "schema": "scpn_ansatz_scaling_tn_v1",
        "command": "python scripts/benchmark_ansatz_scaling_tn.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim_boundary": (
            "Circuit-size rows cover n=4--12. Tensor-network diagnostics are "
            "MPS truncation diagnostics computed from exact ground states only "
            "up to exact_max_qubits; skipped rows are not extrapolated."
        ),
        "n_values": n_values,
        "reps_values": reps_values,
        "max_bonds": max_bonds,
        "exact_max_qubits": ns.exact_max_qubits,
        "ansatz_rows": ansatz_rows,
        "tensor_network_rows": tn_rows,
    }

    json_path = ns.output_dir / f"ansatz_scaling_tn_summary_{DATE}.json"
    ansatz_csv = ns.output_dir / f"ansatz_scaling_summary_{DATE}.csv"
    tn_csv = ns.output_dir / f"tn_truncation_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(ansatz_csv, ansatz_rows)
    _write_csv(tn_csv, tn_rows)
    print(f"wrote_json={json_path}")
    print(f"wrote_ansatz_csv={ansatz_csv}")
    print(f"wrote_tn_csv={tn_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_ansatz_csv={_sha256(ansatz_csv)}")
    print(f"sha256_tn_csv={_sha256(tn_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
