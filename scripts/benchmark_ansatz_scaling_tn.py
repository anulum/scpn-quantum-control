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
from scipy.sparse.linalg import eigsh

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_dense_matrix,
    knm_to_sparse_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"
VQE_SUMMARY_PATH = OUT_DIR / f"vqe_benchmark_summary_{DATE}.json"


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


def _dense_ground_state(n_qubits: int) -> tuple[float, np.ndarray]:
    k_matrix = build_knm_paper27(n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    hamiltonian = knm_to_dense_matrix(k_matrix, omega)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=np.complex128)


def _sparse_ground_state(n_qubits: int) -> tuple[float, np.ndarray, float]:
    k_matrix = build_knm_paper27(n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    hamiltonian = knm_to_sparse_matrix(k_matrix, omega)
    eigenvalues, eigenvectors = eigsh(hamiltonian, k=1, which="SA", tol=1e-10)
    eigenvalue = float(eigenvalues[0])
    eigenvector = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    residual = hamiltonian @ eigenvector - eigenvalue * eigenvector
    return eigenvalue, eigenvector, float(np.linalg.norm(residual))


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
    sparse_max_qubits: int,
) -> list[dict[str, object]]:
    """Return exact-ground-state MPS truncation diagnostics where feasible."""

    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        if n_qubits <= exact_max_qubits:
            ground_energy, state = _dense_ground_state(n_qubits)
            solver = "dense_eigh"
            residual_norm = 0.0
        elif n_qubits <= sparse_max_qubits:
            ground_energy, state, residual_norm = _sparse_ground_state(n_qubits)
            solver = "sparse_eigsh"
        else:
            rows.append(
                {
                    "n_qubits": n_qubits,
                    "status": "skipped",
                    "reason": "above_exact_max_qubits",
                    "exact_max_qubits": exact_max_qubits,
                    "sparse_max_qubits": sparse_max_qubits,
                    "solver": None,
                    "eigen_residual_norm": None,
                    "ground_energy": None,
                    "max_bond": None,
                    "worst_cut_discarded_weight": None,
                    "max_midchain_entropy_bits": None,
                }
            )
            continue
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
                    "sparse_max_qubits": sparse_max_qubits,
                    "solver": solver,
                    "eigen_residual_norm": residual_norm,
                    "ground_energy": ground_energy,
                    "max_bond": max_bond,
                    "worst_cut_discarded_weight": float(
                        max(_discarded_weight(spectrum, max_bond) for spectrum in cut_spectra)
                    ),
                    "max_midchain_entropy_bits": float(max(entropies)),
                }
            )
    return rows


def _load_best_vqe_reference_rows(path: Path) -> dict[int, dict[str, object]]:
    """Return the best committed VQE aggregate row for each qubit count."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_by_n: dict[int, dict[str, object]] = {}
    for row in payload.get("aggregate", []):
        n_qubits = int(str(row["n_qubits"]))
        current = best_by_n.get(n_qubits)
        current_error = float("inf")
        if current is not None:
            current_error = float(str(current["median_relative_error_pct"]))
        candidate_error = float(str(row["median_relative_error_pct"]))
        if candidate_error < current_error:
            best_by_n[n_qubits] = row
    return best_by_n


def reference_comparison_rows(
    n_values: list[int],
    tn_rows: list[dict[str, object]],
    vqe_summary_path: Path = VQE_SUMMARY_PATH,
) -> list[dict[str, object]]:
    """Pair tensor-network diagnostics with committed VQE aggregate references.

    Missing VQE rows are recorded as skipped rows. This avoids presenting
    unrun optimisation data for larger systems as a measured result.
    """

    best_vqe = _load_best_vqe_reference_rows(vqe_summary_path)
    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        ok_tn_rows = [
            row for row in tn_rows if row["n_qubits"] == n_qubits and row["status"] == "ok"
        ]
        if ok_tn_rows:
            tn_reference = max(ok_tn_rows, key=lambda row: int(str(row["max_bond"])))
            discarded_weight = float(str(tn_reference["worst_cut_discarded_weight"]))
            retained_weight = float(max(0.0, 1.0 - discarded_weight))
            tn_status = "ok"
        else:
            tn_reference = next(
                (row for row in tn_rows if row["n_qubits"] == n_qubits),
                {},
            )
            discarded_weight = None
            retained_weight = None
            tn_status = "skipped"

        vqe_reference = best_vqe.get(n_qubits)
        row: dict[str, object] = {
            "n_qubits": n_qubits,
            "tn_status": tn_status,
            "tn_solver": tn_reference.get("solver"),
            "tn_ground_energy": tn_reference.get("ground_energy"),
            "tn_eigen_residual_norm": tn_reference.get("eigen_residual_norm"),
            "tn_max_bond": tn_reference.get("max_bond"),
            "tn_worst_cut_discarded_weight": discarded_weight,
            "tn_retained_weight_lower_bound": retained_weight,
            "vqe_status": "skipped",
            "vqe_skip_reason": "no_committed_vqe_reference",
            "vqe_best_ansatz": None,
            "vqe_reps": None,
            "vqe_n_seeds": None,
            "vqe_best_energy": None,
            "vqe_median_relative_error_pct": None,
            "vqe_best_relative_error_pct": None,
        }
        if vqe_reference is not None:
            row.update(
                {
                    "vqe_status": "ok",
                    "vqe_skip_reason": None,
                    "vqe_best_ansatz": vqe_reference["ansatz"],
                    "vqe_reps": vqe_reference["reps"],
                    "vqe_n_seeds": vqe_reference["n_seeds"],
                    "vqe_best_energy": vqe_reference["best_energy"],
                    "vqe_median_relative_error_pct": vqe_reference["median_relative_error_pct"],
                    "vqe_best_relative_error_pct": vqe_reference["best_relative_error_pct"],
                }
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run the ansatz/tensor-network scaling benchmark CLI."""

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
    parser.add_argument(
        "--sparse-max-qubits",
        type=int,
        default=12,
        help="Largest n for sparse ground-state generation before rows are marked skipped.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--vqe-summary", type=Path, default=VQE_SUMMARY_PATH)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    reps_values = _parse_csv_ints(ns.reps_values)
    max_bonds = _parse_csv_ints(ns.max_bonds)
    ns.output_dir.mkdir(parents=True, exist_ok=True)

    ansatz_rows = ansatz_scaling_rows(n_values, reps_values)
    tn_rows = mps_truncation_rows(
        n_values,
        max_bonds,
        ns.exact_max_qubits,
        ns.sparse_max_qubits,
    )
    comparison_rows = reference_comparison_rows(n_values, tn_rows, ns.vqe_summary)
    summary = {
        "date": DATE,
        "schema": "scpn_ansatz_scaling_tn_v2",
        "command": "python scripts/benchmark_ansatz_scaling_tn.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim_boundary": (
            "Circuit-size rows cover n=4--12. Tensor-network diagnostics are "
            "MPS truncation diagnostics computed from dense exact ground states "
            "up to exact_max_qubits and sparse eigensolver ground states up to "
            "sparse_max_qubits. VQE reference comparisons use only committed "
            "aggregate rows from vqe_benchmark_summary_2026-05-05.json; missing "
            "larger-n VQE rows are marked skipped, not extrapolated."
        ),
        "n_values": n_values,
        "reps_values": reps_values,
        "max_bonds": max_bonds,
        "exact_max_qubits": ns.exact_max_qubits,
        "sparse_max_qubits": ns.sparse_max_qubits,
        "ansatz_rows": ansatz_rows,
        "tensor_network_rows": tn_rows,
        "reference_comparison_rows": comparison_rows,
    }

    json_path = ns.output_dir / f"ansatz_scaling_tn_summary_{DATE}.json"
    ansatz_csv = ns.output_dir / f"ansatz_scaling_summary_{DATE}.csv"
    tn_csv = ns.output_dir / f"tn_truncation_summary_{DATE}.csv"
    comparison_csv = ns.output_dir / f"ansatz_tn_reference_comparison_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(ansatz_csv, ansatz_rows)
    _write_csv(tn_csv, tn_rows)
    _write_csv(comparison_csv, comparison_rows)
    print(f"wrote_json={json_path}")
    print(f"wrote_ansatz_csv={ansatz_csv}")
    print(f"wrote_tn_csv={tn_csv}")
    print(f"wrote_comparison_csv={comparison_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_ansatz_csv={_sha256(ansatz_csv)}")
    print(f"sha256_tn_csv={_sha256(tn_csv)}")
    print(f"sha256_comparison_csv={_sha256(comparison_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
