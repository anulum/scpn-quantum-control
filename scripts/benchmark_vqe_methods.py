#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- VQE methods benchmark harness
"""Generate VQE quality tables for the methods paper."""

from __future__ import annotations

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

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from scpn_quantum_control.hardware.classical import classical_exact_diag

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_ansatz(name: str, n: int, reps: int):
    k = build_knm_paper27(n)
    if name == "knm_informed":
        return knm_to_ansatz(k, reps=reps)
    if name == "two_local":
        return n_local(n, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=reps)
    if name == "efficient_su2":
        return efficient_su2(n, reps=reps)
    raise ValueError(name)


def _run_vqe(name: str, n: int, reps: int, seed: int, maxiter: int) -> dict[str, object]:
    k = build_knm_paper27(n)
    omega = OMEGA_N_16[:n]
    hamiltonian = knm_to_hamiltonian(k, omega)
    exact_energy = float(classical_exact_diag(n, K=k, omega=omega)["ground_energy"])
    ansatz = _make_ansatz(name, n, reps)
    history = []

    def cost(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        energy = float(sv.expectation_value(hamiltonian).real)
        history.append(energy)
        return energy

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    start = time.perf_counter()
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    energy = float(result.fun)
    abs_error = abs(energy - exact_energy)
    return {
        "ansatz": name,
        "n_qubits": n,
        "reps": reps,
        "seed": seed,
        "maxiter": maxiter,
        "parameters": ansatz.num_parameters,
        "energy": energy,
        "exact_energy": exact_energy,
        "absolute_error": abs_error,
        "relative_error_pct": abs_error / abs(exact_energy) * 100.0
        if abs(exact_energy) > 1e-15
        else None,
        "n_evals": int(result.nfev),
        "optimizer_success": bool(result.success),
        "elapsed_ms": float(elapsed_ms),
        "initial_energy": float(history[0]) if history else None,
        "best_history_energy": float(min(history)) if history else None,
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["ansatz"], row["n_qubits"], row["reps"]), []).append(row)
    out = []
    for (ansatz, n, reps), group in sorted(grouped.items()):
        errors = [
            float(row["relative_error_pct"])
            for row in group
            if row["relative_error_pct"] is not None
        ]
        energies = [float(row["energy"]) for row in group]
        out.append(
            {
                "ansatz": ansatz,
                "n_qubits": n,
                "reps": reps,
                "n_seeds": len(group),
                "mean_energy": float(statistics.mean(energies)),
                "best_energy": float(min(energies)),
                "mean_relative_error_pct": float(statistics.mean(errors)),
                "best_relative_error_pct": float(min(errors)),
                "median_relative_error_pct": float(statistics.median(errors)),
            }
        )
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [3, 4]:
        for reps in [1, 2]:
            for name in ["knm_informed", "two_local", "efficient_su2"]:
                for seed in [11, 23, 37]:
                    rows.append(_run_vqe(name, n, reps, seed, maxiter=80))
    aggregate = _aggregate(rows)
    summary = {
        "date": DATE,
        "command": "PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/benchmark_vqe_methods.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timing_caveat": (
            "Opportunistic local timing on a shared workstation. CPU load from other "
            "processes was not pinned or isolated; publication-grade timing numbers "
            "should be rerun on an isolated benchmark host with governor/load metadata. "
            "Energy/error comparisons are less sensitive but still depend on optimiser "
            "budget and random seeds."
        ),
        "rows": rows,
        "aggregate": aggregate,
    }
    json_path = OUT_DIR / f"vqe_benchmark_summary_{DATE}.json"
    rows_csv = OUT_DIR / f"vqe_benchmark_rows_{DATE}.csv"
    agg_csv = OUT_DIR / f"vqe_benchmark_aggregate_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with rows_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with agg_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in aggregate for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"wrote_json={json_path}")
    print(f"wrote_rows_csv={rows_csv}")
    print(f"wrote_aggregate_csv={agg_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_rows_csv={_sha256(rows_csv)}")
    print(f"sha256_aggregate_csv={_sha256(agg_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
