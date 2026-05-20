#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM replication and ZNE IBM lane
"""Prepare, submit, retrieve, and analyse FIM replication/ZNE IBM lanes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from scpn_quantum_control.hardware.runner import _extract_counts
from scpn_quantum_control.mitigation.readout_matrix import (
    build_readout_confusion_matrix,
    computational_basis_labels,
    mitigate_counts,
    probability_magnetisation_leakage,
    probability_parity_leakage,
    probability_state_retention,
)
from scpn_quantum_control.mitigation.zne import zne_extrapolate

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
DEFAULT_CREDENTIALS_VAULT = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
DEFAULT_LAYOUTS = {
    "ibm_marrakesh": (7, 17, 6, 8),
    "ibm_fez": (21, 22, 23, 24),
}
N_QUBITS = 4
DEFAULT_STATES = ("0000", "0001", "0111", "1111")
DEFAULT_DEPTHS = (2, 4)
DEFAULT_LAMBDAS = (0.0, 4.0)
DEFAULT_SCALES = (1, 3, 5)
SEED_TRANSPILER = 20260520
T_STEP = 0.3


@dataclass(frozen=True)
class FIMCircuitEntry:
    """One FIM replication/ZNE circuit entry."""

    block: str
    lambda_fim: float | None
    depth: int | None
    initial_bitstring: str
    replicate: int
    noise_scale: int
    circuit: QuantumCircuit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIM second-backend replication and ZNE lane.")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--experiment-id", default="scpn_fim_replication_zne_2026-05-20")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--noise-scales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--physical-qubits", type=int, nargs="+")
    parser.add_argument("--max-depth", type=int, default=2500)
    parser.add_argument("--max-qpu-seconds", type=float, default=160.0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--retrieve-submission", type=Path)
    return parser.parse_args(argv)


def _parse_vault(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    phase1_path = REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py"
    spec = importlib.util.spec_from_file_location("phase1_mini_bench_ibm_kingston", phase1_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {phase1_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    token, instance = module.parse_vault(path)
    return token, instance


def _load_backend(backend_name: str, instance: str | None, credentials_vault: Path):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token, vault_instance = _parse_vault(credentials_vault)
    service_kwargs: dict[str, str] = {"channel": "ibm_cloud"} if token else {}
    if token:
        service_kwargs["token"] = token
    selected_instance = instance or vault_instance
    if selected_instance:
        service_kwargs["instance"] = selected_instance
    service = QiskitRuntimeService(**service_kwargs)
    return service.backend(backend_name), service


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return _sha256(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(key): int(value) for key, value in circuit.count_ops().items()}


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    return int(
        sum(value for key, value in circuit.count_ops().items() if key in {"cx", "cz", "ecr"})
    )


def _magnetisation(bitstring: str) -> int:
    clean = bitstring.replace(" ", "")
    return len(clean) - 2 * clean.count("1")


def _prep_bitstring(circuit: QuantumCircuit, bitstring: str) -> None:
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            circuit.x(qubit)


def _kuramoto_k_matrix() -> np.ndarray:
    k_matrix = np.zeros((N_QUBITS, N_QUBITS), dtype=np.float64)
    for i in range(N_QUBITS):
        for j in range(N_QUBITS):
            if i != j:
                k_matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return k_matrix


def build_fim_trotter_circuit(
    initial_bitstring: str,
    depth: int,
    lambda_fim: float,
    t_step: float = T_STEP,
) -> QuantumCircuit:
    """Build the n=4 Kuramoto-XY plus FIM digital Trotter circuit."""
    circuit = QuantumCircuit(N_QUBITS, N_QUBITS)
    _prep_bitstring(circuit, initial_bitstring)
    k_matrix = _kuramoto_k_matrix()
    omega = np.linspace(0.8, 1.2, N_QUBITS)
    fim_theta = -4.0 * float(lambda_fim) * t_step / float(N_QUBITS)
    for _ in range(depth):
        for qubit in range(N_QUBITS):
            circuit.rz(2.0 * omega[qubit] * t_step, qubit)
        for i in range(N_QUBITS - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * t_step
            circuit.rxx(theta, i, j)
            circuit.ryy(theta, i, j)
        if abs(fim_theta) > 1e-15:
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    circuit.rzz(fim_theta, i, j)
    circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def build_readout_circuit(initial_bitstring: str) -> QuantumCircuit:
    """Build a computational-basis readout calibration circuit."""
    circuit = QuantumCircuit(N_QUBITS, N_QUBITS)
    _prep_bitstring(circuit, initial_bitstring)
    circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def _validate_scales(scales: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(scale) for scale in scales)
    if not resolved:
        raise ValueError("noise scales must be non-empty")
    if any(scale < 1 or scale % 2 == 0 for scale in resolved):
        raise ValueError("noise scales must be odd positive integers")
    return resolved


def locally_fold_circuit(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Fold invertible operations locally while preserving final measurements."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd positive integer")
    folded = circuit.copy_empty_like(name=f"{circuit.name}_lf{scale}")
    qubit_map = {bit: folded.qubits[index] for index, bit in enumerate(circuit.qubits)}
    clbit_map = {bit: folded.clbits[index] for index, bit in enumerate(circuit.clbits)}
    folds = (scale - 1) // 2
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [qubit_map[bit] for bit in instruction.qubits]
        clbits = [clbit_map[bit] for bit in instruction.clbits]
        folded.append(operation.copy(), qubits, clbits)
        if folds == 0 or clbits or operation.name in {"measure", "reset", "barrier", "delay"}:
            continue
        try:
            inverse = operation.inverse()
        except Exception:
            continue
        for _ in range(folds):
            folded.append(inverse.copy(), qubits, [])
            folded.append(operation.copy(), qubits, [])
    return folded


def build_entries(
    *,
    states: Sequence[str],
    depths: Sequence[int],
    replicates: int,
    noise_scales: Sequence[int],
) -> list[FIMCircuitEntry]:
    """Build main FIM ZNE circuits plus full 16-state readout calibration."""
    scales = _validate_scales(noise_scales)
    entries: list[FIMCircuitEntry] = []
    for initial in states:
        if len(initial) != N_QUBITS or any(bit not in {"0", "1"} for bit in initial):
            raise ValueError(f"invalid initial bitstring: {initial}")
        for depth in depths:
            for lambda_fim in DEFAULT_LAMBDAS:
                for replicate in range(replicates):
                    base = build_fim_trotter_circuit(initial, int(depth), float(lambda_fim))
                    base.name = f"fim_l{lambda_fim:g}_d{depth}_{initial}_r{replicate}"
                    for scale in scales:
                        folded = locally_fold_circuit(base, scale)
                        entries.append(
                            FIMCircuitEntry(
                                block="main",
                                lambda_fim=float(lambda_fim),
                                depth=int(depth),
                                initial_bitstring=initial,
                                replicate=replicate,
                                noise_scale=scale,
                                circuit=folded,
                            )
                        )
    for state in computational_basis_labels(N_QUBITS):
        # build_readout_circuit expects logical preparation order; counts are observed reversed.
        logical_state = state[::-1]
        circuit = build_readout_circuit(logical_state)
        circuit.name = f"fim_readout_{state}"
        entries.append(
            FIMCircuitEntry(
                block="readout_calibration",
                lambda_fim=None,
                depth=None,
                initial_bitstring=logical_state,
                replicate=0,
                noise_scale=1,
                circuit=circuit,
            )
        )
    return entries


def _transpile_entries(
    backend: Any,
    entries: Sequence[FIMCircuitEntry],
    physical_qubits: Sequence[int],
) -> list[QuantumCircuit]:
    return [
        transpile(
            entry.circuit,
            backend=backend,
            initial_layout=list(physical_qubits),
            optimization_level=2,
            seed_transpiler=SEED_TRANSPILER,
        )
        for entry in entries
    ]


def _backend_status(backend: Any) -> dict[str, Any]:
    status_fn = getattr(backend, "status", None)
    if not callable(status_fn):
        return {"available": None, "pending_jobs": None, "status_msg": "status unavailable"}
    status = status_fn()
    return {
        "available": getattr(status, "operational", None),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "status_msg": getattr(status, "status_msg", None),
    }


def _metadata_row(entry: FIMCircuitEntry, circuit: QuantumCircuit, index: int) -> dict[str, Any]:
    return {
        "circuit_index": index,
        "block": entry.block,
        "protocol_arm": entry.block,
        "lambda_fim": entry.lambda_fim,
        "depth": entry.depth,
        "initial_bitstring": entry.initial_bitstring,
        "observed_target_bitstring": entry.initial_bitstring[::-1],
        "magnetisation": _magnetisation(entry.initial_bitstring),
        "popcount": entry.initial_bitstring.count("1"),
        "replicate": entry.replicate,
        "zne_noise_scale": entry.noise_scale,
        "transpiled_depth": int(circuit.depth()),
        "transpiled_size": int(circuit.size()),
        "transpiled_two_qubit_gates": _two_qubit_count(circuit),
        "transpiled_ops": _operation_counts(circuit),
    }


def _readiness_payload(
    *,
    args: argparse.Namespace,
    backend: Any,
    entries: Sequence[FIMCircuitEntry],
    isa_circuits: Sequence[QuantumCircuit],
    physical_qubits: Sequence[int],
) -> dict[str, Any]:
    rows = [
        _metadata_row(entry, circuit, index)
        for index, (entry, circuit) in enumerate(zip(entries, isa_circuits, strict=True))
    ]
    depths = [int(row["transpiled_depth"]) for row in rows]
    estimated_qpu_seconds = len(rows) * 0.55
    status = _backend_status(backend)
    ready = (
        len(physical_qubits) == N_QUBITS
        and int(args.shots) > 0
        and max(depths) <= int(args.max_depth)
        and estimated_qpu_seconds <= float(args.max_qpu_seconds)
        and status.get("available") is not False
    )
    return {
        "schema": "scpn_fim_replication_zne_readiness_v1",
        "timestamp_utc": _timestamp(),
        "experiment_id": args.experiment_id,
        "backend": args.backend,
        "backend_status": status,
        "status": "ready_for_submission" if ready else "blocked",
        "physical_qubits": list(physical_qubits),
        "shots": int(args.shots),
        "states": list(args.states),
        "depths": [int(value) for value in args.depths],
        "lambda_values": list(DEFAULT_LAMBDAS),
        "noise_scales": list(_validate_scales(args.noise_scales)),
        "replicates": int(args.replicates),
        "main_circuits": sum(1 for entry in entries if entry.block == "main"),
        "readout_calibration_circuits": sum(
            1 for entry in entries if entry.block == "readout_calibration"
        ),
        "total_circuits": len(rows),
        "estimated_qpu_seconds": estimated_qpu_seconds,
        "max_qpu_seconds": float(args.max_qpu_seconds),
        "max_depth": int(args.max_depth),
        "depth_summary": {"min": min(depths), "max": max(depths), "mean": float(mean(depths))},
        "max_two_qubit_gates": max(int(row["transpiled_two_qubit_gates"]) for row in rows),
        "metadata_rows": rows,
        "claim_boundary": (
            "Second-backend FIM replication plus local-folding ZNE stress lane. "
            "It tests whether the lambda=4 versus lambda=0 leakage/retention sign "
            "survives another IBM backend/layout and noise-scale extrapolation. "
            "It is not backend-general protection evidence."
        ),
    }


def _job_status(job: Any) -> str:
    status = job.status()
    return str(getattr(status, "name", status))


def submit_lane(args: argparse.Namespace) -> int:
    """Prepare readiness and optionally submit an IBM FIM replication/ZNE lane."""
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2
    physical_qubits = tuple(args.physical_qubits or DEFAULT_LAYOUTS.get(args.backend, ()))
    backend, _service = _load_backend(args.backend, args.instance, args.credentials_vault)
    entries = build_entries(
        states=args.states,
        depths=args.depths,
        replicates=args.replicates,
        noise_scales=args.noise_scales,
    )
    isa_circuits = _transpile_entries(backend, entries, physical_qubits)
    readiness = _readiness_payload(
        args=args,
        backend=backend,
        entries=entries,
        isa_circuits=isa_circuits,
        physical_qubits=physical_qubits,
    )
    timestamp = readiness["timestamp_utc"]
    readiness_path = (
        args.out_dir / f"fim_replication_zne_readiness_{args.backend}_{timestamp}.json"
    )
    readiness_sha = _write_json(readiness_path, readiness)
    print(f"readiness={readiness['status']}")
    print(f"readiness_json={readiness_path}")
    print(f"readiness_sha256={readiness_sha}")
    if readiness["status"] != "ready_for_submission":
        return 3
    if not args.submit:
        return 0

    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = int(args.shots)
    started = time.time()
    job = sampler.run(isa_circuits)
    job_id = str(job.job_id())
    submission = {
        "schema": "scpn_fim_replication_zne_submission_v1",
        "experiment_id": args.experiment_id,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "job_ids": [job_id],
        "job_status_at_submission": _job_status(job),
        "submission_wall_time_s": float(time.time() - started),
        "readiness_json": str(readiness_path.relative_to(REPO_ROOT)),
        "readiness_sha256": readiness_sha,
        "physical_qubits": list(physical_qubits),
        "shots": int(args.shots),
        "metadata_rows": readiness["metadata_rows"],
        "claim_boundary": readiness["claim_boundary"],
    }
    submission_path = (
        args.out_dir / f"fim_replication_zne_submission_{args.backend}_{timestamp}.json"
    )
    submission_sha = _write_json(submission_path, submission)
    print("hardware_submission=true")
    print(f"job_id={job_id}")
    print(f"submission_json={submission_path}")
    print(f"submission_sha256={submission_sha}")
    return 0


def _load_service(credentials_vault: Path, instance: str | None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token, vault_instance = _parse_vault(credentials_vault)
    service_kwargs: dict[str, str] = {"channel": "ibm_cloud"} if token else {}
    if token:
        service_kwargs["token"] = token
    selected_instance = instance or vault_instance
    if selected_instance:
        service_kwargs["instance"] = selected_instance
    return QiskitRuntimeService(**service_kwargs)


def _result_rows(job: Any, metadata_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = job.result()
    rows = []
    for index, metadata in enumerate(metadata_rows):
        rows.append(
            {
                "circuit_index": index,
                "metadata": dict(metadata),
                "counts": dict(_extract_counts(result[index])),
            }
        )
    return rows


def _total(counts: Mapping[str, int]) -> int:
    return int(sum(int(value) for value in counts.values()))


def _raw_probability(counts: Mapping[str, int], predicate) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty counts")
    return float(
        sum(int(count) for bitstring, count in counts.items() if predicate(bitstring)) / total
    )


def _raw_state_retention(counts: Mapping[str, int], target: str) -> float:
    total = _total(counts)
    if total <= 0:
        raise ValueError("empty counts")
    return float(int(counts.get(target, 0)) / total)


def _raw_magnetisation_leakage(counts: Mapping[str, int], target: str) -> float:
    target_m = _magnetisation(target)
    return _raw_probability(counts, lambda bitstring: _magnetisation(bitstring) != target_m)


def _raw_parity_leakage(counts: Mapping[str, int], target: str) -> float:
    target_p = target.count("1") % 2
    return _raw_probability(counts, lambda bitstring: bitstring.count("1") % 2 != target_p)


def _calibration_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    labels = set(computational_basis_labels(N_QUBITS))
    calibrations: dict[str, dict[str, int]] = {}
    for row in rows:
        metadata = row["metadata"]
        if metadata["block"] != "readout_calibration":
            continue
        prepared = str(metadata["observed_target_bitstring"])
        if prepared not in labels:
            raise ValueError(f"invalid calibration state {prepared}")
        calibrations[prepared] = {str(key): int(value) for key, value in row["counts"].items()}
    return calibrations


def _metric_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Any]:
    matrix = build_readout_confusion_matrix(_calibration_counts(rows), N_QUBITS)
    metrics: list[dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        if metadata["block"] != "main":
            continue
        counts = {str(key): int(value) for key, value in row["counts"].items()}
        target = str(metadata["observed_target_bitstring"])
        probabilities = mitigate_counts(counts, matrix)
        metrics.append(
            {
                **metadata,
                "shots_observed": _total(counts),
                "raw_state_retention": _raw_state_retention(counts, target),
                "raw_magnetisation_leakage": _raw_magnetisation_leakage(counts, target),
                "raw_parity_leakage": _raw_parity_leakage(counts, target),
                "mitigated_state_retention": probability_state_retention(
                    probabilities,
                    matrix.labels,
                    target,
                ),
                "mitigated_magnetisation_leakage": probability_magnetisation_leakage(
                    probabilities,
                    matrix.labels,
                    target,
                ),
                "mitigated_parity_leakage": probability_parity_leakage(
                    probabilities,
                    matrix.labels,
                    target,
                ),
            }
        )
    return metrics, matrix


def _mean_for(
    metrics: Sequence[Mapping[str, Any]],
    *,
    initial: str,
    depth: int,
    scale: int,
    lambda_fim: float,
    observable: str,
) -> float:
    selected = [
        float(row[observable])
        for row in metrics
        if row["initial_bitstring"] == initial
        and int(row["depth"]) == depth
        and int(row["zne_noise_scale"]) == scale
        and float(row["lambda_fim"]) == lambda_fim
    ]
    if not selected:
        raise ValueError("missing metric group")
    return float(mean(selected))


def _channel_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    initials = sorted({str(row["initial_bitstring"]) for row in metrics})
    depths = sorted({int(row["depth"]) for row in metrics})
    scales = sorted({int(row["zne_noise_scale"]) for row in metrics})
    observables = (
        "raw_state_retention",
        "raw_magnetisation_leakage",
        "raw_parity_leakage",
        "mitigated_state_retention",
        "mitigated_magnetisation_leakage",
        "mitigated_parity_leakage",
    )
    channels: list[dict[str, Any]] = []
    for initial in initials:
        for depth in depths:
            for observable in observables:
                lambda0 = [
                    _mean_for(
                        metrics,
                        initial=initial,
                        depth=depth,
                        scale=scale,
                        lambda_fim=0.0,
                        observable=observable,
                    )
                    for scale in scales
                ]
                lambda4 = [
                    _mean_for(
                        metrics,
                        initial=initial,
                        depth=depth,
                        scale=scale,
                        lambda_fim=4.0,
                        observable=observable,
                    )
                    for scale in scales
                ]
                deltas = [right - left for left, right in zip(lambda0, lambda4, strict=True)]
                linear = zne_extrapolate(scales, deltas, order=1)
                quadratic = zne_extrapolate(scales, deltas, order=2) if len(scales) >= 3 else None
                channels.append(
                    {
                        "initial_bitstring": initial,
                        "depth": depth,
                        "observable": observable,
                        "noise_scales": scales,
                        "lambda0_by_scale": lambda0,
                        "lambda4_by_scale": lambda4,
                        "delta_lambda4_minus_lambda0_by_scale": deltas,
                        "scale1_delta": deltas[0],
                        "linear_zne_delta": linear.zero_noise_estimate,
                        "linear_zne_fit_residual": linear.fit_residual,
                        "quadratic_zne_delta": None
                        if quadratic is None
                        else quadratic.zero_noise_estimate,
                        "quadratic_zne_fit_residual": None
                        if quadratic is None
                        else quadratic.fit_residual,
                    }
                )
    return channels


def _summarise(raw_payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics, matrix = _metric_rows(raw_payload["result_rows"])
    channels = _channel_rows(metrics)

    def _mean_abs(observable: str, key: str) -> float:
        selected = [abs(float(row[key])) for row in channels if row["observable"] == observable]
        return float(mean(selected))

    return {
        "schema": "scpn_fim_replication_zne_analysis_v1",
        "experiment_id": raw_payload["experiment_id"],
        "backend": raw_payload["backend"],
        "job_ids": raw_payload["job_ids"],
        "readout_model": {
            "n_qubits": matrix.n_qubits,
            "labels": list(matrix.labels),
            "condition_number": matrix.condition_number,
            "shots_by_prepared_state": matrix.shots_by_prepared_state,
        },
        "metric_rows": metrics,
        "channel_rows": channels,
        "mean_abs_raw_magnetisation_leakage_scale1_delta": _mean_abs(
            "raw_magnetisation_leakage",
            "scale1_delta",
        ),
        "mean_abs_raw_magnetisation_leakage_linear_zne_delta": _mean_abs(
            "raw_magnetisation_leakage",
            "linear_zne_delta",
        ),
        "mean_abs_mitigated_magnetisation_leakage_linear_zne_delta": _mean_abs(
            "mitigated_magnetisation_leakage",
            "linear_zne_delta",
        ),
        "claim_boundary": (
            "Analysis of one FIM replication/ZNE backend lane. ZNE fits use local "
            "folding scale factors 1, 3, 5 and are sensitivity tests, not proof of "
            "backend-general FIM protection."
        ),
    }


def retrieve_lane(args: argparse.Namespace) -> int:
    """Retrieve and analyse a completed FIM replication/ZNE submission."""
    submission_path = args.retrieve_submission.resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    service = _load_service(args.credentials_vault, args.instance)
    job = service.job(str(submission["job_ids"][0]))
    status = _job_status(job)
    print(f"job_status={status}")
    if status not in {"DONE", "JobStatus.DONE"}:
        print("raw_counts_available=false")
        return 2
    rows = _result_rows(job, submission["metadata_rows"])
    timestamp = _timestamp()
    raw_payload = {
        "schema": "scpn_fim_replication_zne_raw_counts_v1",
        "experiment_id": submission["experiment_id"],
        "backend": submission["backend"],
        "timestamp_utc": timestamp,
        "job_ids": submission["job_ids"],
        "submission_json": str(submission_path.relative_to(REPO_ROOT)),
        "submission_sha256": _sha256(submission_path),
        "physical_qubits": submission["physical_qubits"],
        "shots": submission["shots"],
        "result_rows": rows,
    }
    raw_path = (
        args.out_dir / f"fim_replication_zne_raw_counts_{submission['backend']}_{timestamp}.json"
    )
    raw_sha = _write_json(raw_path, raw_payload)
    analysis = _summarise(raw_payload)
    analysis["raw_counts_json"] = str(raw_path.relative_to(REPO_ROOT))
    analysis["raw_counts_sha256"] = raw_sha
    analysis_path = (
        args.out_dir / f"fim_replication_zne_analysis_{submission['backend']}_{timestamp}.json"
    )
    analysis_sha = _write_json(analysis_path, analysis)
    print("raw_counts_available=true")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.retrieve_submission is not None:
        return retrieve_lane(args)
    return submit_lane(args)


if __name__ == "__main__":
    raise SystemExit(main())
