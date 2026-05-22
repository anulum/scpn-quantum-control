#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IBM runtime latency campaign
"""Run a reproducible IBM Runtime latency scenario matrix.

The campaign measures externally visible latency windows, not intra-shot control
electronics latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.ibm_latency_probe import (
    derive_timing_windows,
    extract_job_telemetry,
)
from scpn_quantum_control.hardware.s1_feedback_ibm import build_s1_feedback_arm_circuits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_s1_ibm_live_readiness import (  # noqa: E402
    DEFAULT_CREDENTIALS_VAULT,
    SEED_TRANSPILER,
    load_authenticated_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_TIMEOUT = 3600.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-qpu", action="store_true")
    parser.add_argument("--s1-shots-grid", default="256,512,1024")
    parser.add_argument("--s1-repetitions-grid", default="4,8")
    parser.add_argument("--capacity-shots", type=int, default=256)
    parser.add_argument(
        "--capacity-width-fractions",
        default="0.1,0.25,0.5,1.0",
        help="Fractions of backend qubits for width sweep.",
    )
    parser.add_argument("--capacity-depth", type=int, default=2)
    parser.add_argument("--capacity-trials", type=int, default=2)
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array(
            [[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
    )


def _parse_int_grid(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError(f"invalid integer grid: {raw!r}")
    return values


def _parse_fraction_grid(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values or any((not math.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError(f"invalid fraction grid: {raw!r}")
    return values


def _submit_batch(
    backend: Any,
    circuits: Sequence[QuantumCircuit],
    shots: int,
    timeout_s: float,
) -> dict[str, Any]:
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    submit_started = time.monotonic()
    job = sampler.run(list(circuits))
    submit_finished = time.monotonic()
    result_started = time.monotonic()
    _ = job.result(timeout=timeout_s)
    result_finished = time.monotonic()
    telemetry = extract_job_telemetry(job)
    windows = derive_timing_windows(telemetry)
    return {
        "job_id": telemetry.get("job_id"),
        "submit_overhead_seconds": submit_finished - submit_started,
        "submit_to_result_seconds": result_finished - submit_started,
        "result_wait_seconds": result_finished - result_started,
        "provider_windows": windows,
        "telemetry": telemetry,
    }


def _summarise_latency_windows(measurements: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = [float(row["submit_to_result_seconds"]) for row in measurements]
    wait = [float(row["result_wait_seconds"]) for row in measurements]
    submit = [float(row["submit_overhead_seconds"]) for row in measurements]
    provider_run = [
        float(row["provider_windows"]["provider_run_seconds"])
        for row in measurements
        if row["provider_windows"]["provider_run_seconds"] is not None
    ]
    summary: dict[str, Any] = {
        "count": len(measurements),
        "submit_to_result_mean_s": statistics.mean(total),
        "submit_to_result_std_s": statistics.stdev(total) if len(total) > 1 else 0.0,
        "submit_to_result_p95_s": float(np.quantile(np.array(total), 0.95)),
        "result_wait_mean_s": statistics.mean(wait),
        "submit_overhead_mean_s": statistics.mean(submit),
        "job_ids": [row["job_id"] for row in measurements],
    }
    if provider_run:
        summary["provider_run_mean_s"] = statistics.mean(provider_run)
        summary["provider_run_std_s"] = (
            statistics.stdev(provider_run) if len(provider_run) > 1 else 0.0
        )
    return summary


def _build_capacity_circuit(width: int, depth: int, seed: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, width)
    for layer in range(depth):
        for qubit in range(width):
            angle = (seed + 17 * (layer + 1) + qubit) * 0.001
            circuit.ry(angle, qubit)
        for qubit in range(width - 1):
            circuit.cx(qubit, qubit + 1)
    circuit.measure(range(width), range(width))
    return circuit


def _capacity_widths(backend_qubits: int, fractions: Sequence[float]) -> list[int]:
    widths = {
        max(2, min(backend_qubits, int(round(backend_qubits * fraction))))
        for fraction in fractions
    }
    widths.add(backend_qubits)
    return sorted(widths)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the IBM runtime latency matrix and persist campaign artefacts."""
    args = _parse_args(argv)
    if args.submit and not args.confirm_qpu:
        print("ERROR: --submit requires --confirm-qpu")
        return 2

    payload: dict[str, Any] = {
        "schema": "scpn_ibm_runtime_latency_campaign_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "hardware_submission": bool(args.submit),
        "claim_boundary": (
            "Measures externally visible IBM runtime latency windows. "
            "Does not directly measure intra-shot control-electronics feedforward latency."
        ),
    }

    try:
        backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    except Exception as exc:
        payload["status"] = "blocked_dependency_or_auth"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output = args.out_dir / f"ibm_runtime_latency_campaign_{args.backend}_{_timestamp()}.json"
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={output}")
        print(f"sha256_json={_sha256(output)}")
        print("hardware_submission=false")
        return 3

    backend_qubits = int(getattr(backend, "num_qubits", 0))
    payload["backend_qubits"] = backend_qubits

    shots_grid = _parse_int_grid(args.s1_shots_grid)
    reps_grid = _parse_int_grid(args.s1_repetitions_grid)
    width_fractions = _parse_fraction_grid(args.capacity_width_fractions)
    capacity_widths = _capacity_widths(backend_qubits, width_fractions)

    payload["scenario_matrix"] = {
        "s1_shots_grid": shots_grid,
        "s1_repetitions_grid": reps_grid,
        "capacity_widths": capacity_widths,
        "capacity_depth": args.capacity_depth,
        "capacity_trials": args.capacity_trials,
    }

    s1_results: list[dict[str, Any]] = []
    capacity_results: list[dict[str, Any]] = []
    controller = _controller()

    for shots in shots_grid:
        for repetitions in reps_grid:
            feedback_arm, control_arm = build_s1_feedback_arm_circuits(
                controller,
                n_rounds=3,
                shots=shots,
                repetitions=repetitions,
            )
            feedback_isa = transpile(
                feedback_arm.circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=SEED_TRANSPILER,
            )
            control_isa = transpile(
                control_arm.circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=SEED_TRANSPILER,
            )
            entry: dict[str, Any] = {
                "shots": shots,
                "repetitions": repetitions,
                "feedback_depth": int(feedback_isa.depth()),
                "control_depth": int(control_isa.depth()),
                "feedback_two_qubit_ops": int(feedback_isa.num_nonlocal_gates()),
                "control_two_qubit_ops": int(control_isa.num_nonlocal_gates()),
            }
            if args.submit:
                feedback_batch = [
                    feedback_isa.copy(name=f"feedback_s{shots}_r{repetitions}_{idx:02d}")
                    for idx in range(repetitions)
                ]
                control_batch = [
                    control_isa.copy(name=f"control_s{shots}_r{repetitions}_{idx:02d}")
                    for idx in range(repetitions)
                ]
                feedback_runtime = _submit_batch(backend, feedback_batch, shots, args.timeout_s)
                control_runtime = _submit_batch(backend, control_batch, shots, args.timeout_s)
                entry["runtime"] = {
                    "feedback_dynamic": feedback_runtime,
                    "control_open_loop": control_runtime,
                }
            s1_results.append(entry)

    for width in capacity_widths:
        trial_measurements: list[dict[str, Any]] = []
        transpiled_depth = None
        two_qubit_ops = None
        for trial in range(args.capacity_trials):
            circuit = _build_capacity_circuit(width, args.capacity_depth, seed=20260522 + trial)
            isa = transpile(
                circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=SEED_TRANSPILER + trial,
            )
            if transpiled_depth is None:
                transpiled_depth = int(isa.depth())
                two_qubit_ops = int(isa.num_nonlocal_gates())
            if args.submit:
                trial_measurements.append(
                    _submit_batch(
                        backend,
                        [isa.copy(name=f"capacity_w{width}_t{trial:02d}")],
                        args.capacity_shots,
                        args.timeout_s,
                    )
                )
        capacity_entry: dict[str, Any] = {
            "width": width,
            "depth": args.capacity_depth,
            "shots": args.capacity_shots,
            "trials": args.capacity_trials,
            "transpiled_depth": transpiled_depth,
            "two_qubit_ops": two_qubit_ops,
        }
        if args.submit:
            capacity_entry["measurements"] = trial_measurements
            capacity_entry["summary"] = _summarise_latency_windows(trial_measurements)
        capacity_results.append(capacity_entry)

    payload["status"] = "submitted_and_completed" if args.submit else "ready_no_submit"
    payload["s1_matrix"] = s1_results
    payload["capacity_sweep"] = capacity_results
    if not args.submit:
        payload["submission_instructions"] = (
            "Run with --submit --confirm-qpu to execute the matrix and collect IBM windows."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"ibm_runtime_latency_campaign_{args.backend}_{_timestamp()}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={output}")
    print(f"sha256_json={_sha256(output)}")
    print(f"hardware_submission={str(args.submit).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
