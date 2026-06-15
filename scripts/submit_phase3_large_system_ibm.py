#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 larger-system IBM lane
"""Prepare or submit n=6/n=8 Phase 3 reduced-Pauli ZNE stress lanes."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "phase3_entanglement_tomography"
DEFAULT_CREDENTIALS_VAULT = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
DEFAULT_LAYOUTS = {
    ("ibm_fez", 6): (21, 22, 23, 24, 25, 26),
    ("ibm_fez", 8): (21, 22, 23, 24, 25, 26, 27, 28),
    ("ibm_marrakesh", 6): (2, 3, 4, 5, 6, 7),
    ("ibm_marrakesh", 8): (1, 2, 3, 4, 5, 6, 7, 8),
    ("ibm_kingston", 6): (141, 142, 143, 144, 145, 146),
    ("ibm_kingston", 8): (141, 142, 143, 144, 145, 146, 147, 148),
}
DEFAULT_SCALES = (1, 3, 5)
DEFAULT_REPETITIONS = 3
DEFAULT_SHOTS = 1024
SECONDS_PER_CIRCUIT_ESTIMATE = 0.55
SEED_TRANSPILER = 20260520
T_STEP = 0.3


@dataclass(frozen=True)
class SourceSpec:
    """One larger-system Phase 3 source circuit."""

    family: str
    label: str
    initial_bitstring: str
    depth: int
    lambda_fim: float | None


@dataclass(frozen=True)
class Phase3Entry:
    """One measured Phase 3 larger-system circuit entry."""

    block: str
    spec: SourceSpec | None
    basis_setting: str | None
    repetition: int
    noise_scale: int
    calibration_state: str | None
    circuit: QuantumCircuit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--experiment-id", default="phase3_large_system_2026-05-20")
    parser.add_argument("--n-qubits", type=int, choices=[6, 8], required=True)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--noise-scales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--physical-qubits", type=int, nargs="+")
    parser.add_argument("--max-depth", type=int, default=3500)
    parser.add_argument("--max-two-qubit-gates", type=int, default=2500)
    parser.add_argument("--max-qpu-seconds", type=float, default=3600.0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
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


def load_authenticated_backend(backend_name: str, instance: str | None, credentials_vault: Path):
    """Load a Qiskit Runtime backend from explicit or vault-backed credentials."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token, vault_instance = _parse_vault(credentials_vault)
    service_kwargs: dict[str, str] = {"channel": "ibm_cloud"} if token else {}
    if token:
        service_kwargs["token"] = token
    selected_instance = instance or vault_instance
    if selected_instance:
        service_kwargs["instance"] = selected_instance
    service = QiskitRuntimeService(**service_kwargs)
    return service.backend(backend_name)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _validate_scales(scales: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(scale) for scale in scales)
    if not resolved:
        raise ValueError("noise scales must be non-empty")
    if any(scale < 1 or scale % 2 == 0 for scale in resolved):
        raise ValueError("noise scales must be odd positive integers")
    return resolved


def _prep_bitstring(circuit: QuantumCircuit, bitstring: str) -> None:
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            circuit.x(qubit)


def _k_matrix(n_qubits: int) -> np.ndarray:
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return matrix


def _xy_source(initial: str, depth: int) -> QuantumCircuit:
    n_qubits = len(initial)
    circuit = QuantumCircuit(n_qubits)
    _prep_bitstring(circuit, initial)
    omega = np.linspace(0.8, 1.2, n_qubits)
    matrix = _k_matrix(n_qubits)
    for _ in range(depth):
        for qubit in range(n_qubits):
            circuit.rz(2.0 * omega[qubit] * T_STEP, qubit)
        for i in range(n_qubits - 1):
            theta = 2.0 * matrix[i, i + 1] * T_STEP
            circuit.rxx(theta, i, i + 1)
            circuit.ryy(theta, i, i + 1)
    return circuit


def _fim_source(initial: str, depth: int, lambda_fim: float) -> QuantumCircuit:
    n_qubits = len(initial)
    circuit = QuantumCircuit(n_qubits)
    _prep_bitstring(circuit, initial)
    omega = np.linspace(0.8, 1.2, n_qubits)
    matrix = _k_matrix(n_qubits)
    fim_theta = -4.0 * float(lambda_fim) * T_STEP / float(n_qubits)
    for _ in range(depth):
        for qubit in range(n_qubits):
            circuit.rz(2.0 * omega[qubit] * T_STEP, qubit)
        for i in range(n_qubits - 1):
            theta = 2.0 * matrix[i, i + 1] * T_STEP
            circuit.rxx(theta, i, i + 1)
            circuit.ryy(theta, i, i + 1)
        if abs(fim_theta) > 1e-15:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.rzz(fim_theta, i, j)
    return circuit


def source_specs(n_qubits: int) -> tuple[SourceSpec, ...]:
    """Return the larger-system analogue of the promoted Phase 3 source scope."""
    even_state = "0" * (n_qubits - 2) + "11"
    odd_state = "0" * (n_qubits - 1) + "1"
    return (
        SourceSpec("dla_parity", "dla_even_shallow", even_state, 6, None),
        SourceSpec("dla_parity", "dla_odd_shallow", odd_state, 6, None),
        SourceSpec("dla_parity", "dla_even_signal", even_state, 10, None),
        SourceSpec("dla_parity", "dla_odd_signal", odd_state, 10, None),
        SourceSpec("fim_pair", "fim_lambda0_reference", even_state, 4, 0.0),
        SourceSpec("fim_pair", "fim_lambda4_feedback", even_state, 4, 4.0),
    )


def phase3_observables(n_qubits: int) -> tuple[str, ...]:
    """Return left, middle, and right transverse edge settings."""
    edges = ((0, 1), (n_qubits // 2 - 1, n_qubits // 2), (n_qubits - 2, n_qubits - 1))
    rows: list[str] = []
    for edge in edges:
        for pauli in ("X", "Y"):
            label = ["I"] * n_qubits
            label[edge[0]] = pauli
            label[edge[1]] = pauli
            rows.append("".join(label))
    return tuple(dict.fromkeys(rows))


def build_source_circuit(spec: SourceSpec) -> QuantumCircuit:
    """Build the no-measurement source circuit for one larger-system source spec."""
    if spec.family == "dla_parity":
        return _xy_source(spec.initial_bitstring, spec.depth)
    if spec.family == "fim_pair":
        if spec.lambda_fim is None:
            raise ValueError("FIM source requires lambda_fim")
        return _fim_source(spec.initial_bitstring, spec.depth, spec.lambda_fim)
    raise ValueError(f"unsupported source family: {spec.family}")


def apply_measurement_basis(circuit: QuantumCircuit, basis_setting: str) -> QuantumCircuit:
    """Return a measured circuit for one reduced-Pauli basis setting."""
    if len(basis_setting) != circuit.num_qubits:
        raise ValueError("basis_setting length must match circuit width")
    measured = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    measured.compose(circuit, inplace=True)
    for qubit, basis in enumerate(basis_setting):
        if basis == "X":
            measured.h(qubit)
        elif basis == "Y":
            measured.sdg(qubit)
            measured.h(qubit)
        elif basis == "I":
            pass
        else:
            raise ValueError(f"unsupported basis label: {basis}")
    measured.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    return measured


def _readout_circuit(bitstring: str, n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits)
    _prep_bitstring(circuit, bitstring)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


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
    n_qubits: int,
    repetitions: int,
    noise_scales: Sequence[int],
) -> list[Phase3Entry]:
    """Build larger-system Phase 3 ZNE circuits plus full readout calibration."""
    scales = _validate_scales(noise_scales)
    entries: list[Phase3Entry] = []
    for spec in source_specs(n_qubits):
        source = build_source_circuit(spec)
        for basis in phase3_observables(n_qubits):
            measured = apply_measurement_basis(source, basis)
            measured.name = f"p3_large_{spec.label}_{basis}"
            for scale in scales:
                folded = locally_fold_circuit(measured, scale)
                for rep in range(repetitions):
                    entries.append(
                        Phase3Entry(
                            block="main",
                            spec=spec,
                            basis_setting=basis,
                            repetition=rep,
                            noise_scale=scale,
                            calibration_state=None,
                            circuit=folded,
                        )
                    )
    for index in range(2**n_qubits):
        observed_state = format(index, f"0{n_qubits}b")
        logical_state = observed_state[::-1]
        circuit = _readout_circuit(logical_state, n_qubits)
        circuit.name = f"p3_large_readout_{observed_state}"
        entries.append(
            Phase3Entry(
                block="readout_calibration",
                spec=None,
                basis_setting=None,
                repetition=0,
                noise_scale=1,
                calibration_state=observed_state,
                circuit=circuit,
            )
        )
    return entries


def _transpile_entries(
    backend: Any,
    entries: Sequence[Phase3Entry],
    physical_qubits: Sequence[int],
) -> list[QuantumCircuit]:
    return [
        transpile(
            entry.circuit,
            backend=backend,
            initial_layout=list(physical_qubits),
            optimization_level=1,
            seed_transpiler=SEED_TRANSPILER,
        )
        for entry in entries
    ]


def _backend_status(backend: Any) -> dict[str, Any]:
    status_method = getattr(backend, "status", None)
    if not callable(status_method):
        return {"available": None, "pending_jobs": None, "status_msg": "status unavailable"}
    status = status_method()
    return {
        "available": getattr(status, "operational", None),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "status_msg": getattr(status, "status_msg", None),
    }


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    return int(
        sum(value for key, value in circuit.count_ops().items() if key in {"cx", "cz", "ecr"})
    )


def _entry_metadata(entry: Phase3Entry, circuit: QuantumCircuit, index: int) -> dict[str, Any]:
    spec = entry.spec
    return {
        "circuit_index": index,
        "block": entry.block,
        "family": None if spec is None else spec.family,
        "label": None if spec is None else spec.label,
        "initial": None if spec is None else spec.initial_bitstring,
        "depth": None if spec is None else spec.depth,
        "lambda_fim": None if spec is None else spec.lambda_fim,
        "basis_setting": entry.basis_setting,
        "rep": entry.repetition,
        "zne_noise_scale": entry.noise_scale,
        "calibration_state": entry.calibration_state,
        "transpiled_depth": int(circuit.depth()),
        "transpiled_size": int(circuit.size()),
        "transpiled_two_qubit_gates": _two_qubit_count(circuit),
        "transpiled_ops": {str(key): int(value) for key, value in circuit.count_ops().items()},
    }


def readiness_payload(
    *,
    args: argparse.Namespace,
    backend: Any,
    entries: Sequence[Phase3Entry],
    isa_circuits: Sequence[QuantumCircuit],
    physical_qubits: Sequence[int],
) -> dict[str, Any]:
    """Build readiness metadata and gate decision for one Phase 3 larger-system lane."""
    rows = [
        _entry_metadata(entry, circuit, index)
        for index, (entry, circuit) in enumerate(zip(entries, isa_circuits, strict=True))
    ]
    depths = [int(row["transpiled_depth"]) for row in rows]
    twoq = [int(row["transpiled_two_qubit_gates"]) for row in rows]
    estimated_qpu_seconds = len(rows) * SECONDS_PER_CIRCUIT_ESTIMATE
    status = _backend_status(backend)
    ready = (
        len(physical_qubits) == args.n_qubits
        and int(args.shots) > 0
        and max(depths, default=0) <= int(args.max_depth)
        and max(twoq, default=0) <= int(args.max_two_qubit_gates)
        and estimated_qpu_seconds <= float(args.max_qpu_seconds)
        and status.get("available") is not False
    )
    return {
        "schema": "scpn_phase3_large_system_readiness_v1",
        "timestamp_utc": _timestamp(),
        "experiment_id": args.experiment_id,
        "backend": args.backend,
        "backend_status": status,
        "status": "ready_for_submission" if ready else "blocked",
        "n_qubits": int(args.n_qubits),
        "physical_qubits": list(physical_qubits),
        "shots": int(args.shots),
        "source_specs": [
            {
                "family": spec.family,
                "label": spec.label,
                "initial_bitstring": spec.initial_bitstring,
                "depth": spec.depth,
                "lambda_fim": spec.lambda_fim,
            }
            for spec in source_specs(args.n_qubits)
        ],
        "basis_settings": list(phase3_observables(args.n_qubits)),
        "noise_scales": list(_validate_scales(args.noise_scales)),
        "repetitions": int(args.repetitions),
        "main_circuits": sum(1 for entry in entries if entry.block == "main"),
        "readout_calibration_circuits": sum(
            1 for entry in entries if entry.block == "readout_calibration"
        ),
        "total_circuits": len(rows),
        "estimated_qpu_seconds": float(estimated_qpu_seconds),
        "max_qpu_seconds": float(args.max_qpu_seconds),
        "max_depth": int(args.max_depth),
        "max_two_qubit_gates": int(args.max_two_qubit_gates),
        "depth_summary": {
            "min": min(depths) if depths else None,
            "max": max(depths) if depths else None,
            "mean": float(mean(depths)) if depths else None,
        },
        "max_transpiled_two_qubit_gates": max(twoq) if twoq else None,
        "metadata_rows": rows,
        "claim_boundary": (
            f"n={args.n_qubits} larger-system Phase 3 reduced-Pauli ZNE stress lane. "
            "It tests whether transverse DLA/FIM reduced-Pauli structure persists "
            "under larger width and local noise scaling on the selected IBM backend/layout. "
            "It is not full tomography, backend-general entanglement dynamics, or "
            "quantum-advantage evidence."
        ),
    }


def _resolve_physical_qubits(args: argparse.Namespace) -> tuple[int, ...]:
    if args.physical_qubits:
        return tuple(int(qubit) for qubit in args.physical_qubits)
    return tuple(DEFAULT_LAYOUTS.get((args.backend, int(args.n_qubits)), ()))


def _job_status(job: Any) -> str:
    status = job.status()
    return str(getattr(status, "name", status))


def submit_lane(args: argparse.Namespace) -> int:
    """Prepare readiness and optionally submit one Phase 3 larger-system lane."""
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    physical_qubits = _resolve_physical_qubits(args)
    entries = build_entries(
        n_qubits=int(args.n_qubits),
        repetitions=int(args.repetitions),
        noise_scales=args.noise_scales,
    )
    isa_circuits = _transpile_entries(backend, entries, physical_qubits)
    readiness = readiness_payload(
        args=args,
        backend=backend,
        entries=entries,
        isa_circuits=isa_circuits,
        physical_qubits=physical_qubits,
    )
    timestamp = readiness["timestamp_utc"]
    readiness_path = (
        args.out_dir / f"phase3_large_system_readiness_{args.backend}_{timestamp}.json"
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
        "schema": "scpn_phase3_large_system_submission_v1",
        "experiment_id": args.experiment_id,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "n_qubits": int(args.n_qubits),
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
        args.out_dir / f"phase3_large_system_submission_{args.backend}_{timestamp}.json"
    )
    submission_sha = _write_json(submission_path, submission)
    print("hardware_submission=true")
    print(f"job_id={job_id}")
    print(f"submission_json={submission_path}")
    print(f"submission_sha256={submission_sha}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run readiness or submit one Phase 3 larger-system lane."""
    return submit_lane(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
