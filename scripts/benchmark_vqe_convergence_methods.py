#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- VQE convergence and scaling evidence
"""Generate methods-paper VQE convergence, timing uncertainty, and scaling artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
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
DATE = "2026-05-20"
ANSATZ_FAMILIES = ("knm_informed", "two_local", "efficient_su2")
DEFAULT_SEEDS = (11, 23, 37)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local VQE convergence traces and ansatz scaling probes for the "
            "rust_vqe_methods paper."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--n-values", type=int, nargs="+", default=[4])
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--timing-repeats", type=int, default=50)
    parser.add_argument("--scaling-n", type=int, nargs="+", default=[4, 6, 8, 12, 16, 18, 20])
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_ansatz(name: str, n_qubits: int, reps: int) -> QuantumCircuit:
    """Build one of the paper benchmark ansatz families."""
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
    raise ValueError(f"unsupported ansatz family: {name}")


def _relative_error_pct(energy: float, exact_energy: float) -> float | None:
    if abs(exact_energy) <= 1e-15:
        return None
    return abs(energy - exact_energy) / abs(exact_energy) * 100.0


def _sample_trace(
    history: Sequence[float], exact_energy: float, checkpoints: Iterable[int]
) -> list[dict[str, Any]]:
    best = float("inf")
    rows: list[dict[str, Any]] = []
    history_by_eval = {index + 1: float(value) for index, value in enumerate(history)}
    for eval_index in checkpoints:
        if eval_index < 1:
            continue
        usable = [value for index, value in enumerate(history, start=1) if index <= eval_index]
        if usable:
            best = min(best, min(usable))
            energy = history_by_eval.get(eval_index, float(usable[-1]))
            rows.append(
                {
                    "eval": int(eval_index),
                    "energy": energy,
                    "best_energy": float(best),
                    "absolute_error": abs(best - exact_energy),
                    "relative_error_pct": _relative_error_pct(best, exact_energy),
                }
            )
    return rows


def run_vqe_trace(
    *,
    ansatz_name: str,
    n_qubits: int,
    reps: int,
    seed: int,
    maxiter: int,
) -> dict[str, Any]:
    """Run one VQE trace and preserve the optimiser history and parameters."""
    k_matrix = build_knm_paper27(n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    hamiltonian = knm_to_hamiltonian(k_matrix, omega)
    exact_energy = float(classical_exact_diag(n_qubits, K=k_matrix, omega=omega)["ground_energy"])
    ansatz = make_ansatz(ansatz_name, n_qubits, reps)
    history: list[float] = []

    def cost(params: np.ndarray) -> float:
        bound = ansatz.assign_parameters(params)
        state = Statevector.from_instruction(bound)
        energy = float(state.expectation_value(hamiltonian).real)
        history.append(energy)
        return energy

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    started = time.perf_counter()
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    best_energy = float(min(history)) if history else float(result.fun)
    checkpoints = [1, 5, 10, 20, 40, 80, 160, 320]
    checkpoints = [item for item in checkpoints if item <= maxiter]
    return {
        "ansatz": ansatz_name,
        "n_qubits": n_qubits,
        "reps": reps,
        "seed": seed,
        "maxiter": maxiter,
        "parameters": int(ansatz.num_parameters),
        "exact_energy": exact_energy,
        "final_energy": float(result.fun),
        "best_energy": best_energy,
        "absolute_error": abs(best_energy - exact_energy),
        "relative_error_pct": _relative_error_pct(best_energy, exact_energy),
        "n_evals": int(result.nfev),
        "optimizer_success": bool(result.success),
        "elapsed_ms": float(elapsed_ms),
        "optimal_params": [float(value) for value in np.asarray(result.x, dtype=float)],
        "trace": _sample_trace(history, exact_energy, checkpoints),
    }


def summarise_final_errors(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate final VQE errors by ansatz and qubit count."""
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["ansatz"]), int(row["n_qubits"]), int(row["reps"])), []
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (ansatz, n_qubits, reps), group in sorted(grouped.items()):
        errors = [
            float(row["relative_error_pct"])
            for row in group
            if row["relative_error_pct"] is not None
        ]
        best_errors = [float(row["absolute_error"]) for row in group]
        elapsed = [float(row["elapsed_ms"]) for row in group]
        summary.append(
            {
                "ansatz": ansatz,
                "n_qubits": n_qubits,
                "reps": reps,
                "n_seeds": len(group),
                "mean_relative_error_pct": float(statistics.mean(errors)),
                "stdev_relative_error_pct": float(statistics.stdev(errors))
                if len(errors) > 1
                else 0.0,
                "stderr_relative_error_pct": float(statistics.stdev(errors) / len(errors) ** 0.5)
                if len(errors) > 1
                else 0.0,
                "best_relative_error_pct": float(min(errors)),
                "mean_absolute_error": float(statistics.mean(best_errors)),
                "mean_elapsed_ms": float(statistics.mean(elapsed)),
                "stderr_elapsed_ms": float(statistics.stdev(elapsed) / len(elapsed) ** 0.5)
                if len(elapsed) > 1
                else 0.0,
            }
        )
    return summary


def _time_values_ms(fn, repeats: int) -> list[float]:
    values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return values


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarise benchmark timings with uncertainty metrics."""
    if not values:
        raise ValueError("values must be non-empty")
    sorted_values = sorted(float(value) for value in values)
    return {
        "mean_ms": float(statistics.mean(sorted_values)),
        "median_ms": float(statistics.median(sorted_values)),
        "stdev_ms": float(statistics.stdev(sorted_values)) if len(sorted_values) > 1 else 0.0,
        "stderr_ms": float(statistics.stdev(sorted_values) / len(sorted_values) ** 0.5)
        if len(sorted_values) > 1
        else 0.0,
        "min_ms": float(sorted_values[0]),
        "max_ms": float(sorted_values[-1]),
        "repeats": float(len(sorted_values)),
    }


def run_scaling_probe(
    *,
    n_values: Sequence[int],
    reps: int,
    timing_repeats: int,
) -> list[dict[str, Any]]:
    """Measure ansatz construction and local transpilation resource scaling."""
    rows: list[dict[str, Any]] = []
    for n_qubits in n_values:
        for ansatz_name in ANSATZ_FAMILIES:
            values = _time_values_ms(
                lambda ansatz_name=ansatz_name, n_qubits=n_qubits: make_ansatz(
                    ansatz_name,
                    n_qubits,
                    reps,
                ),
                timing_repeats,
            )
            circuit = make_ansatz(ansatz_name, n_qubits, reps)
            transpiled = transpile(
                circuit,
                basis_gates=["rz", "sx", "x", "cx"],
                optimization_level=1,
                seed_transpiler=20260520,
            )
            raw_ops = circuit.count_ops()
            transpiled_ops = transpiled.count_ops()
            rows.append(
                {
                    "ansatz": ansatz_name,
                    "n_qubits": n_qubits,
                    "reps": reps,
                    "parameters": int(circuit.num_parameters),
                    "raw_depth": int(circuit.depth()),
                    "raw_two_qubit_gates": int(
                        sum(
                            value
                            for key, value in raw_ops.items()
                            if key in {"cx", "cz", "rzz", "ecr"}
                        )
                    ),
                    "transpiled_depth": int(transpiled.depth()),
                    "transpiled_two_qubit_gates": int(
                        sum(
                            value
                            for key, value in transpiled_ops.items()
                            if key in {"cx", "cz", "rzz", "ecr"}
                        )
                    ),
                    **{f"build_{key}": value for key, value in timing_summary(values).items()},
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the methods-paper convergence and scaling benchmark."""
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    for n_qubits in args.n_values:
        for ansatz_name in ANSATZ_FAMILIES:
            for seed in args.seeds:
                trace_rows.append(
                    run_vqe_trace(
                        ansatz_name=ansatz_name,
                        n_qubits=n_qubits,
                        reps=args.reps,
                        seed=seed,
                        maxiter=args.maxiter,
                    )
                )
    aggregate = summarise_final_errors(trace_rows)
    scaling_rows = run_scaling_probe(
        n_values=args.scaling_n,
        reps=args.reps,
        timing_repeats=args.timing_repeats,
    )
    payload = {
        "schema": "scpn_rust_vqe_methods_convergence_v1",
        "date": args.date,
        "command": " ".join(
            ["scripts/benchmark_vqe_convergence_methods.py", *([] if argv is None else argv)]
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "claim_boundary": (
            "Dense statevector VQE convergence is reported only for the selected small "
            "n values. The n=16--20 lane is construction/transpilation scaling, not "
            "dense energy optimisation."
        ),
        "vqe_rows": trace_rows,
        "vqe_aggregate": aggregate,
        "scaling_rows": scaling_rows,
    }
    json_path = args.out_dir / f"vqe_convergence_methods_{args.date}.json"
    aggregate_csv = args.out_dir / f"vqe_convergence_aggregate_{args.date}.csv"
    scaling_csv = args.out_dir / f"ansatz_scaling_error_bars_{args.date}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(aggregate_csv, aggregate)
    _write_csv(scaling_csv, scaling_rows)
    print(f"wrote_json={json_path}")
    print(f"wrote_aggregate_csv={aggregate_csv}")
    print(f"wrote_scaling_csv={scaling_csv}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_aggregate_csv={_sha256(aggregate_csv)}")
    print(f"sha256_scaling_csv={_sha256(scaling_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
