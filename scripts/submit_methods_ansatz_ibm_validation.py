#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IBM ansatz-validation methods lane
"""Prepare or submit a small IBM energy-validation lane for methods ansatz claims."""

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
from typing import Any

import numpy as np
from benchmark_vqe_convergence_methods import ANSATZ_FAMILIES, make_ansatz, run_vqe_trace
from qiskit import QuantumCircuit, transpile

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_hamiltonian,
)
from scpn_quantum_control.hardware.runner import _extract_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DEFAULT_CREDENTIALS_VAULT = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
DEFAULT_LAYOUT = (7, 17, 6, 8)
SEED_TRANSPILER = 20260520
QPU_SECONDS_CEILING = 140.0


@dataclass(frozen=True)
class ValidationCircuit:
    """One logical ansatz-validation or readout-calibration circuit."""

    role: str
    ansatz: str
    basis: str
    repetition: int
    calibration_state: str | None
    circuit: QuantumCircuit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or submit IBM methods ansatz-validation circuits."
    )
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--experiment-id", default="rust_vqe_methods_ansatz_validation_2026-05-20")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--maxiter", type=int, default=320)
    parser.add_argument("--max-qpu-seconds", type=float, default=QPU_SECONDS_CEILING)
    parser.add_argument("--physical-qubits", type=int, nargs="+", default=list(DEFAULT_LAYOUT))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument(
        "--wait-for-result",
        action="store_true",
        help="Wait for job completion and write raw-count analysis in the same process.",
    )
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
    credential_value, vault_instance = module.parse_vault(path)
    return credential_value, vault_instance


def load_authenticated_backend(backend_name: str, instance: str | None, credentials_vault: Path):
    """Load a Qiskit Runtime backend from explicit or vault-backed credentials."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for IBM validation") from exc
    credential_value, vault_instance = _parse_vault(credentials_vault)
    selected_instance = instance or vault_instance
    service_kwargs: dict[str, str] = {"channel": "ibm_cloud"} if credential_value else {}
    if credential_value:
        service_kwargs["token"] = credential_value
    if selected_instance:
        service_kwargs["instance"] = selected_instance
    service = QiskitRuntimeService(**service_kwargs)
    return service.backend(backend_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _sha256(path)


def pauli_terms_by_basis(n_qubits: int) -> dict[str, list[dict[str, Any]]]:
    """Group the Kuramoto-XY Hamiltonian Pauli terms into global X/Y/Z bases."""
    k_matrix = build_knm_paper27(n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    operator = knm_to_hamiltonian(k_matrix, omega)
    grouped: dict[str, list[dict[str, Any]]] = {"X": [], "Y": [], "Z": []}
    for label, coeff in operator.to_list():
        non_identity = {char for char in label if char != "I"}
        if len(non_identity) != 1:
            raise ValueError(f"term {label} is not compatible with one global basis")
        basis = non_identity.pop()
        if basis not in grouped:
            raise ValueError(f"unsupported Pauli basis in {label}")
        grouped[basis].append({"label": label, "coefficient": float(np.real(coeff))})
    return grouped


def expectation_from_counts(counts: Mapping[str, int], pauli_label: str) -> float:
    """Estimate a Qiskit-order Pauli expectation from final measured bitstrings."""
    n_qubits = len(pauli_label)
    total = 0
    weighted = 0.0
    active_qubits = [n_qubits - 1 - pos for pos, char in enumerate(pauli_label) if char != "I"]
    if not active_qubits:
        raise ValueError("pauli_label must contain a non-identity operator")
    for raw_bitstring, count in counts.items():
        bitstring = raw_bitstring.replace(" ", "")[-n_qubits:].zfill(n_qubits)
        qubit_order_bits = bitstring[::-1]
        eigenvalue = 1.0
        for qubit in active_qubits:
            eigenvalue *= 1.0 if qubit_order_bits[qubit] == "0" else -1.0
        weighted += float(count) * eigenvalue
        total += int(count)
    if total < 1:
        raise ValueError("counts must contain at least one shot")
    return weighted / total


def _probability_vector(counts: Mapping[str, int], n_qubits: int) -> np.ndarray:
    total = sum(int(value) for value in counts.values())
    if total < 1:
        raise ValueError("counts must contain at least one shot")
    vector = np.zeros(2**n_qubits, dtype=float)
    for raw_bitstring, count in counts.items():
        bitstring = raw_bitstring.replace(" ", "")[-n_qubits:].zfill(n_qubits)
        vector[int(bitstring, 2)] += int(count) / total
    return vector


def expectation_from_probabilities(
    probabilities: Sequence[float] | np.ndarray, pauli_label: str
) -> float:
    """Estimate a Qiskit-order Pauli expectation from a bitstring probability vector."""
    n_qubits = len(pauli_label)
    if len(probabilities) != 2**n_qubits:
        raise ValueError("probabilities length does not match pauli_label")
    active_qubits = [n_qubits - 1 - pos for pos, char in enumerate(pauli_label) if char != "I"]
    if not active_qubits:
        raise ValueError("pauli_label must contain a non-identity operator")
    weighted = 0.0
    for index, probability in enumerate(probabilities):
        bitstring = format(index, f"0{n_qubits}b")
        qubit_order_bits = bitstring[::-1]
        eigenvalue = 1.0
        for qubit in active_qubits:
            eigenvalue *= 1.0 if qubit_order_bits[qubit] == "0" else -1.0
        weighted += float(probability) * eigenvalue
    return float(weighted)


def energy_from_basis_counts(
    basis_counts: Mapping[str, Mapping[str, int]],
    grouped_terms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    """Reduce grouped-basis counts to a Hamiltonian energy estimate."""
    energy = 0.0
    for basis, terms in grouped_terms.items():
        counts = basis_counts[basis]
        for term in terms:
            energy += float(term["coefficient"]) * expectation_from_counts(
                counts, str(term["label"])
            )
    return float(energy)


def energy_from_basis_probabilities(
    basis_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    grouped_terms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    """Reduce grouped-basis probability vectors to a Hamiltonian energy estimate."""
    energy = 0.0
    for basis, terms in grouped_terms.items():
        probabilities = basis_probabilities[basis]
        for term in terms:
            energy += float(term["coefficient"]) * expectation_from_probabilities(
                probabilities,
                str(term["label"]),
            )
    return float(energy)


def full_readout_assignment_matrix(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    n_qubits: int,
) -> np.ndarray:
    """Build measured-given-prepared assignment matrix from full calibration rows."""
    if len(calibration_rows) != 2**n_qubits:
        raise ValueError("full calibration requires one row per computational state")
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=float)
    seen: set[int] = set()
    for row in calibration_rows:
        state = row.get("calibration_state")
        counts = row.get("counts")
        if not isinstance(state, str) or len(state) != n_qubits:
            raise ValueError("calibration_state has invalid width")
        if not isinstance(counts, Mapping):
            raise ValueError("calibration row must contain counts")
        prepared_index = int(state, 2)
        seen.add(prepared_index)
        matrix[:, prepared_index] = _probability_vector(counts, n_qubits)
    if seen != set(range(2**n_qubits)):
        raise ValueError("calibration rows do not cover all computational states")
    return matrix


def mitigate_counts_with_assignment(
    counts: Mapping[str, int],
    *,
    assignment_matrix: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Apply full assignment-matrix inversion to a measured count distribution."""
    measured = _probability_vector(counts, n_qubits)
    mitigated = np.linalg.pinv(assignment_matrix) @ measured
    return np.asarray(mitigated, dtype=float)


def _apply_basis_rotation(circuit: QuantumCircuit, basis: str, n_qubits: int) -> None:
    if basis == "X":
        for qubit in range(n_qubits):
            circuit.h(qubit)
    elif basis == "Y":
        for qubit in range(n_qubits):
            circuit.sdg(qubit)
            circuit.h(qubit)
    elif basis != "Z":
        raise ValueError(f"unsupported basis {basis}")


def _ansatz_binding(
    ansatz_name: str, n_qubits: int, reps: int, seed: int, maxiter: int
) -> dict[str, Any]:
    trace = run_vqe_trace(
        ansatz_name=ansatz_name,
        n_qubits=n_qubits,
        reps=reps,
        seed=seed,
        maxiter=maxiter,
    )
    circuit = make_ansatz(ansatz_name, n_qubits, reps)
    bound = circuit.assign_parameters(trace["optimal_params"])
    return {"trace": trace, "circuit": bound}


def build_validation_entries(
    *,
    n_qubits: int,
    reps: int,
    seed: int,
    maxiter: int,
) -> tuple[list[ValidationCircuit], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Build ansatz measurement circuits plus full computational-basis calibration."""
    grouped_terms = pauli_terms_by_basis(n_qubits)
    entries: list[ValidationCircuit] = []
    local_summaries: list[dict[str, Any]] = []
    for ansatz_name in ANSATZ_FAMILIES:
        binding = _ansatz_binding(ansatz_name, n_qubits, reps, seed, maxiter)
        local_summaries.append(binding["trace"])
        for basis in ("Z", "X", "Y"):
            circuit = QuantumCircuit(n_qubits, n_qubits)
            circuit.compose(binding["circuit"], range(n_qubits), inplace=True)
            _apply_basis_rotation(circuit, basis, n_qubits)
            circuit.measure(range(n_qubits), range(n_qubits))
            entries.append(
                ValidationCircuit(
                    role="ansatz_observable",
                    ansatz=ansatz_name,
                    basis=basis,
                    repetition=0,
                    calibration_state=None,
                    circuit=circuit,
                )
            )
    for state in range(2**n_qubits):
        state_bits = format(state, f"0{n_qubits}b")
        circuit = QuantumCircuit(n_qubits, n_qubits)
        for qubit, bit in enumerate(reversed(state_bits)):
            if bit == "1":
                circuit.x(qubit)
        circuit.measure(range(n_qubits), range(n_qubits))
        entries.append(
            ValidationCircuit(
                role="readout_calibration",
                ansatz="calibration",
                basis="Z",
                repetition=0,
                calibration_state=state_bits,
                circuit=circuit,
            )
        )
    return entries, local_summaries, grouped_terms


def _entry_metadata(entry: ValidationCircuit, circuit: QuantumCircuit) -> dict[str, Any]:
    return {
        "role": entry.role,
        "ansatz": entry.ansatz,
        "basis": entry.basis,
        "repetition": entry.repetition,
        "calibration_state": entry.calibration_state,
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_qubits": int(circuit.num_qubits),
        "n_clbits": int(circuit.num_clbits),
        "operation_counts": {str(key): int(value) for key, value in circuit.count_ops().items()},
    }


def _transpile_entries(
    backend: Any,
    entries: Sequence[ValidationCircuit],
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


def _readiness_payload(
    *,
    args: argparse.Namespace,
    backend: Any,
    entries: Sequence[ValidationCircuit],
    isa_circuits: Sequence[QuantumCircuit],
    local_summaries: Sequence[Mapping[str, Any]],
    grouped_terms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    estimated_qpu_seconds = len(entries) * 1.0
    status = _backend_status(backend)
    ready = (
        len(args.physical_qubits) == args.n_qubits
        and args.shots > 0
        and estimated_qpu_seconds <= args.max_qpu_seconds
        and status.get("available") is not False
    )
    return {
        "schema": "scpn_rust_vqe_methods_ansatz_ibm_readiness_v1",
        "experiment_id": args.experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "backend": args.backend,
        "backend_status": status,
        "status": "ready_for_submission" if ready else "blocked",
        "n_qubits": args.n_qubits,
        "logical_to_physical_layout": list(args.physical_qubits),
        "shots": args.shots,
        "qpu_seconds_ceiling": float(args.max_qpu_seconds),
        "estimated_qpu_seconds": estimated_qpu_seconds,
        "circuit_count": len(entries),
        "ansatz_families": list(ANSATZ_FAMILIES),
        "measurement_bases": ["Z", "X", "Y"],
        "readout_calibration_states": 2**args.n_qubits,
        "term_groups": grouped_terms,
        "local_vqe_summaries": list(local_summaries),
        "transpiled_circuits": [
            _entry_metadata(entry, circuit)
            for entry, circuit in zip(entries, isa_circuits, strict=True)
        ],
        "claim_boundary": (
            f"This is an n={args.n_qubits} hardware validation of ansatz-family energy ordering "
            "under one IBM backend/layout, not a convergence proof, quantum advantage "
            "claim, or backend-general ansatz superiority claim."
        ),
    }


def _extract_result_rows(
    job: Any, entries: Sequence[ValidationCircuit]
) -> tuple[str, list[dict[str, Any]]]:
    job_id = str(job.job_id())
    result = job.result()
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        pub_result = result[index]
        counts = _extract_counts(pub_result)
        rows.append(
            {
                "index": index,
                "role": entry.role,
                "ansatz": entry.ansatz,
                "basis": entry.basis,
                "repetition": entry.repetition,
                "calibration_state": entry.calibration_state,
                "counts": dict(counts),
            }
        )
    return job_id, rows


def _job_status(job: Any) -> str:
    status_method = getattr(job, "status", None)
    if not callable(status_method):
        return "unknown"
    status = status_method()
    return str(getattr(status, "name", status))


def _analyse_raw_rows(
    rows: Sequence[Mapping[str, Any]],
    grouped_terms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_ansatz: dict[str, dict[str, Mapping[str, int]]] = {}
    calibration_rows = []
    for row in rows:
        if row["role"] == "readout_calibration":
            calibration_rows.append(row)
            continue
        by_ansatz.setdefault(str(row["ansatz"]), {})[str(row["basis"])] = row["counts"]
    n_qubits = len(str(next(iter(grouped_terms["Z"]))["label"]))
    assignment_matrix = full_readout_assignment_matrix(calibration_rows, n_qubits=n_qubits)
    condition_number = float(np.linalg.cond(assignment_matrix))
    estimates = []
    for ansatz_name, basis_counts in sorted(by_ansatz.items()):
        if set(basis_counts) != {"X", "Y", "Z"}:
            raise ValueError(f"missing basis rows for {ansatz_name}")
        mitigated_probabilities = {
            basis: mitigate_counts_with_assignment(
                counts,
                assignment_matrix=assignment_matrix,
                n_qubits=n_qubits,
            )
            for basis, counts in basis_counts.items()
        }
        estimates.append(
            {
                "ansatz": ansatz_name,
                "raw_energy": energy_from_basis_counts(basis_counts, grouped_terms),
                "full_readout_mitigated_energy": energy_from_basis_probabilities(
                    mitigated_probabilities,
                    grouped_terms,
                ),
            }
        )
    return {
        "schema": "scpn_rust_vqe_methods_ansatz_ibm_analysis_v1",
        "energy_estimates": estimates,
        "readout_calibration_rows": len(calibration_rows),
        "full_readout_assignment_condition_number": condition_number,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run readiness or submit the ansatz-validation lane."""
    args = _parse_args(argv)
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    entries, local_summaries, grouped_terms = build_validation_entries(
        n_qubits=args.n_qubits,
        reps=args.reps,
        seed=args.seed,
        maxiter=args.maxiter,
    )
    isa_circuits = _transpile_entries(backend, entries, args.physical_qubits)
    readiness = _readiness_payload(
        args=args,
        backend=backend,
        entries=entries,
        isa_circuits=isa_circuits,
        local_summaries=local_summaries,
        grouped_terms=grouped_terms,
    )
    timestamp = readiness["timestamp_utc"]
    readiness_path = (
        args.out_dir / f"ansatz_ibm_validation_readiness_{args.backend}_{timestamp}.json"
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
    submission_payload = {
        "schema": "scpn_rust_vqe_methods_ansatz_ibm_submission_v1",
        "experiment_id": args.experiment_id,
        "backend": args.backend,
        "timestamp_utc": timestamp,
        "job_ids": [job_id],
        "job_status_at_submission": _job_status(job),
        "wall_time_s": float(time.time() - started),
        "readiness_json": str(readiness_path.relative_to(REPO_ROOT)),
        "readiness_sha256": readiness_sha,
        "logical_to_physical_layout": list(args.physical_qubits),
        "shots": int(args.shots),
        "term_groups": grouped_terms,
        "transpiled_circuits": readiness["transpiled_circuits"],
    }
    submission_path = (
        args.out_dir / f"ansatz_ibm_validation_submission_{args.backend}_{timestamp}.json"
    )
    submission_sha = _write_json(submission_path, submission_payload)
    print("hardware_submission=true")
    print(f"job_id={job_id}")
    print(f"submission_json={submission_path}")
    print(f"submission_sha256={submission_sha}")
    if not args.wait_for_result:
        return 0

    _, rows = _extract_result_rows(job, entries)
    raw_payload = {
        "schema": "scpn_rust_vqe_methods_ansatz_ibm_raw_counts_v1",
        "experiment_id": args.experiment_id,
        "backend": args.backend,
        "timestamp_utc": timestamp,
        "job_ids": [job_id],
        "wall_time_s": float(time.time() - started),
        "readiness_json": str(readiness_path.relative_to(REPO_ROOT)),
        "readiness_sha256": readiness_sha,
        "logical_to_physical_layout": list(args.physical_qubits),
        "shots": int(args.shots),
        "term_groups": grouped_terms,
        "circuits": rows,
    }
    raw_path = args.out_dir / f"ansatz_ibm_validation_raw_counts_{args.backend}_{timestamp}.json"
    raw_sha = _write_json(raw_path, raw_payload)
    analysis = _analyse_raw_rows(rows, grouped_terms)
    analysis["raw_counts_json"] = str(raw_path.relative_to(REPO_ROOT))
    analysis["raw_counts_sha256"] = raw_sha
    analysis_path = (
        args.out_dir / f"ansatz_ibm_validation_analysis_{args.backend}_{timestamp}.json"
    )
    analysis_sha = _write_json(analysis_path, analysis)
    print("hardware_submission=true")
    print(f"job_id={job_id}")
    print(f"raw_counts_json={raw_path}")
    print(f"raw_counts_sha256={raw_sha}")
    print(f"analysis_json={analysis_path}")
    print(f"analysis_sha256={analysis_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
