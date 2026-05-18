#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM VQE ground-state artefact harness
"""Generate small-n VQE artefacts for the FIM-augmented Hamiltonian."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from qiskit.circuit.library import efficient_su2, n_local
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from scpn_quantum_control.analysis.fim_hamiltonian import add_fim_feedback
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
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


def _make_ansatz(name: str, n_qubits: int, reps: int):
    k_matrix = build_knm_paper27(n_qubits)
    if name == "knm_informed":
        return knm_to_ansatz(k_matrix, reps=reps)
    if name == "two_local":
        return n_local(n_qubits, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=reps)
    if name == "efficient_su2":
        return efficient_su2(n_qubits, reps=reps)
    raise ValueError(name)


def _expectation_from_state(ansatz, params: np.ndarray, hamiltonian: np.ndarray) -> float:
    state = Statevector.from_instruction(ansatz.assign_parameters(params)).data
    return float(np.vdot(state, hamiltonian @ state).real)


def _run_vqe(
    ansatz_name: str,
    n_qubits: int,
    lambda_fim: float,
    reps: int,
    seed: int,
    maxiter: int,
) -> dict[str, object]:
    k_matrix = build_knm_paper27(n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    base_hamiltonian = knm_to_dense_matrix(k_matrix, omega)
    hamiltonian = add_fim_feedback(base_hamiltonian, lambda_fim)
    exact_values = np.linalg.eigvalsh(hamiltonian)
    exact_energy = float(np.min(exact_values))
    ansatz = _make_ansatz(ansatz_name, n_qubits, reps)
    history: list[float] = []

    def cost(params: np.ndarray) -> float:
        energy = _expectation_from_state(ansatz, params, hamiltonian)
        history.append(energy)
        return energy

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    start = time.perf_counter()
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    energy = float(result.fun)
    absolute_error = abs(energy - exact_energy)
    return {
        "ansatz": ansatz_name,
        "n_qubits": n_qubits,
        "lambda_fim": float(lambda_fim),
        "reps": reps,
        "seed": seed,
        "maxiter": maxiter,
        "parameters": ansatz.num_parameters,
        "energy": energy,
        "exact_energy": exact_energy,
        "absolute_error": absolute_error,
        "relative_error_pct": absolute_error / abs(exact_energy) * 100.0
        if abs(exact_energy) > 1e-15
        else None,
        "n_evals": int(result.nfev),
        "optimizer_success": bool(result.success),
        "elapsed_ms": float(elapsed_ms),
        "initial_energy": float(history[0]) if history else None,
        "best_history_energy": float(min(history)) if history else None,
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["ansatz"], row["n_qubits"], row["lambda_fim"], row["reps"])
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (ansatz, n_qubits, lambda_fim, reps), group in sorted(grouped.items()):
        errors = [
            float(str(row["relative_error_pct"]))
            for row in group
            if row["relative_error_pct"] is not None
        ]
        energies = [float(str(row["energy"])) for row in group]
        out.append(
            {
                "ansatz": ansatz,
                "n_qubits": n_qubits,
                "lambda_fim": lambda_fim,
                "reps": reps,
                "n_seeds": len(group),
                "mean_energy": float(statistics.mean(energies)),
                "best_energy": float(min(energies)),
                "mean_relative_error_pct": float(statistics.mean(errors)) if errors else None,
                "best_relative_error_pct": float(min(errors)) if errors else None,
                "median_relative_error_pct": float(statistics.median(errors)) if errors else None,
            }
        )
    return out


def generate(
    n_values: list[int], lambdas: list[float], seeds: list[int], reps: int, maxiter: int
) -> dict[str, object]:
    """Generate FIM-regularised VQE benchmark rows and aggregate summaries."""

    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        for lambda_fim in lambdas:
            for ansatz_name in ["knm_informed", "two_local", "efficient_su2"]:
                for seed in seeds:
                    rows.append(_run_vqe(ansatz_name, n_qubits, lambda_fim, reps, seed, maxiter))
    return {
        "schema": "scpn_fim_vqe_ground_state_v1",
        "date": DATE,
        "command": "python scripts/benchmark_fim_vqe_ground_state.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim_boundary": (
            "Small-n VQE scoring against exact dense diagonalisation. This is an "
            "ansatz and optimisation diagnostic only; it is not hardware evidence "
            "and not a quantum-advantage claim."
        ),
        "n_values": n_values,
        "lambda_values": lambdas,
        "seeds": seeds,
        "reps": reps,
        "maxiter": maxiter,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }


def main() -> int:
    """Run the FIM VQE ground-state benchmark CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4")
    parser.add_argument("--lambdas", default="0,1,4")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    n_values = _parse_csv_ints(ns.n_values)
    lambdas = _parse_csv_floats(ns.lambdas)
    seeds = _parse_csv_ints(ns.seeds)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(n_values, lambdas, seeds, ns.reps, ns.maxiter)

    json_path = ns.output_dir / f"fim_vqe_ground_state_summary_{DATE}.json"
    rows_csv = ns.output_dir / f"fim_vqe_ground_state_rows_{DATE}.csv"
    aggregate_csv = ns.output_dir / f"fim_vqe_ground_state_aggregate_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows_payload = summary["rows"]
    aggregate_payload = summary["aggregate"]
    if not isinstance(rows_payload, list) or not isinstance(aggregate_payload, list):
        raise TypeError("generated benchmark payload must contain row lists")
    _write_csv(rows_csv, rows_payload)
    _write_csv(aggregate_csv, aggregate_payload)
    print(f"wrote_json={json_path}")
    print(f"wrote_rows_csv={rows_csv}")
    print(f"wrote_aggregate_csv={aggregate_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_rows_csv={_sha256(rows_csv)}")
    print(f"sha256_aggregate_csv={_sha256(aggregate_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
