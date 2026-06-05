#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- larger-system submission-extension preparer
"""Prepare resource-gated larger-system extension lanes for submission papers.

The script never submits QPU jobs.  It produces preregistration-ready budget
artefacts that decide which larger-system lanes are credible enough to submit
later with an explicit ``--submit`` runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "large_system_submission_extensions"
DOCS_DIR = REPO_ROOT / "docs" / "campaigns"
DEFAULT_CREDENTIALS_VAULT = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
SEED_TRANSPILER = 20260520
SECONDS_PER_CIRCUIT_ESTIMATE = 0.55
METHODS_SECONDS_PER_CIRCUIT_ESTIMATE = 1.0
T_STEP = 0.3
QPU_SECONDS_CEILING = 3600.0
GPU_SECONDS_CEILING = 3600.0
MAX_FULL_READOUT_STATES_FOR_QPU = 256
MAX_TRANSPILED_DEPTH_FOR_QPU = 3500
MAX_TWO_QUBIT_GATES_FOR_QPU = 2500


@dataclass(frozen=True)
class CandidateSpec:
    """One larger-system candidate lane."""

    candidate_id: str
    paper_target: str
    n_qubits: int
    lane_type: str
    priority: int
    shots: int
    repetitions: int
    noise_scales: tuple[int, ...]
    full_readout: bool
    source_circuits: int
    observables: tuple[str, ...]
    scientific_question: str
    claim_boundary: str


@dataclass(frozen=True)
class CandidateEstimate:
    """Resource estimate and gate decision for one candidate lane."""

    candidate_id: str
    paper_target: str
    n_qubits: int
    lane_type: str
    priority: int
    status: str
    decision_reasons: tuple[str, ...]
    source_circuits: int
    observable_settings: int
    main_circuits: int
    readout_circuits: int
    total_circuits: int
    shots: int
    total_shots: int
    estimated_qpu_seconds: float
    estimated_gpu_seconds: float
    statevector_bytes: int
    dense_matrix_bytes: int
    representative_depth: int | None
    representative_size: int | None
    representative_two_qubit_gates: int | None
    max_depth_gate: int
    max_two_qubit_gate: int
    claim_boundary: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare larger-system submission-extension resource gates."
    )
    parser.add_argument("--backend", default="generic_line")
    parser.add_argument("--backend-qubits", type=int, default=127)
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--qpu-seconds-ceiling", type=float, default=QPU_SECONDS_CEILING)
    parser.add_argument("--gpu-seconds-ceiling", type=float, default=GPU_SECONDS_CEILING)
    parser.add_argument("--probe-ibm-budget", action="store_true")
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    return parser.parse_args(argv)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _line_backend(num_qubits: int) -> GenericBackendV2:
    coupling = [[index, index + 1] for index in range(num_qubits - 1)]
    coupling += [[index + 1, index] for index in range(num_qubits - 1)]
    return GenericBackendV2(
        num_qubits,
        basis_gates=["rz", "sx", "x", "ecr", "cx", "measure"],
        coupling_map=coupling,
        seed=SEED_TRANSPILER,
    )


def _parse_vault(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    phase1_path = REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py"
    spec = importlib.util.spec_from_file_location("phase1_mini_bench_ibm_kingston", phase1_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {phase1_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    token, instance = module.parse_vault(path)
    return token, instance


def load_backend_and_usage(
    backend_name: str,
    *,
    backend_qubits: int,
    instance: str | None,
    credentials_vault: Path,
    probe_ibm_budget: bool,
) -> tuple[Any, dict[str, Any] | None]:
    """Load a live IBM backend or a deterministic generic line backend."""
    if backend_name == "generic_line":
        return _line_backend(backend_qubits), None

    from qiskit_ibm_runtime import QiskitRuntimeService

    token, vault_instance = _parse_vault(credentials_vault)
    kwargs: dict[str, str] = {"channel": "ibm_cloud"} if token else {}
    if token:
        kwargs["token"] = token
    selected_instance = instance or vault_instance
    if selected_instance:
        kwargs["instance"] = selected_instance
    service = QiskitRuntimeService(**kwargs)
    usage = service.usage() if probe_ibm_budget else None
    return service.backend(backend_name), usage


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", "unknown")
    return name() if callable(name) else str(name)


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


def _xy_circuit(
    n_qubits: int, initial: str, depth: int, *, measure: bool = True
) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0)
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
    if measure:
        circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def _fim_circuit(
    n_qubits: int,
    initial: str,
    depth: int,
    lambda_fim: float,
    *,
    measure: bool = True,
) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0)
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
    if measure:
        circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def _phase3_observables(n_qubits: int) -> tuple[str, ...]:
    edges = [(0, 1), (n_qubits // 2 - 1, n_qubits // 2), (n_qubits - 2, n_qubits - 1)]
    rows: list[str] = []
    for edge in edges:
        for pauli in ("X", "Y"):
            label = ["I"] * n_qubits
            label[edge[0]] = pauli
            label[edge[1]] = pauli
            rows.append("".join(label))
    return tuple(dict.fromkeys(rows))


def _method_observables(_n_qubits: int) -> tuple[str, ...]:
    return ("global_X", "global_Y", "global_Z")


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the complete larger-system candidate set."""
    specs: list[CandidateSpec] = []
    for n_qubits, priority in ((6, 0), (8, 1)):
        phase3_obs = _phase3_observables(n_qubits)
        specs.append(
            CandidateSpec(
                candidate_id=f"phase3_reduced_pauli_n{n_qubits}",
                paper_target="submission_005_phase3_reduced_pauli_entanglement",
                n_qubits=n_qubits,
                lane_type="qpu_reduced_pauli_zne_stress",
                priority=priority,
                shots=1024,
                repetitions=3,
                noise_scales=(1, 3, 5),
                full_readout=True,
                source_circuits=6,
                observables=phase3_obs,
                scientific_question=(
                    "Do the DLA transverse reduced-Pauli deviations remain structured "
                    f"when the promoted Phase 3 mechanism test is expanded to n={n_qubits}?"
                ),
                claim_boundary=(
                    "larger-system mechanism stress test only; no quantum advantage, "
                    "backend-general entanglement dynamics, or full tomography claim"
                ),
            )
        )
        specs.append(
            CandidateSpec(
                candidate_id=f"fim_replication_zne_n{n_qubits}",
                paper_target="submission_004_scpn_fim_hamiltonian",
                n_qubits=n_qubits,
                lane_type="qpu_fim_replication_zne",
                priority=priority + 2,
                shots=1024,
                repetitions=2,
                noise_scales=(1, 3, 5),
                full_readout=True,
                source_circuits=16,
                observables=("magnetisation_leakage", "state_retention"),
                scientific_question=(
                    "Does the FIM term remain a hardware falsifier under larger n "
                    f"and controlled noise scaling at n={n_qubits}?"
                ),
                claim_boundary=(
                    "backend/circuit-specific larger-n falsification stress only; "
                    "no protection or many-body localisation claim"
                ),
            )
        )
        specs.append(
            CandidateSpec(
                candidate_id=f"methods_ansatz_energy_n{n_qubits}",
                paper_target="submission_003_rust_vqe_methods",
                n_qubits=n_qubits,
                lane_type="qpu_methods_ansatz_energy_validation",
                priority=priority + 4,
                shots=1024,
                repetitions=1,
                noise_scales=(1,),
                full_readout=True,
                source_circuits=9,
                observables=_method_observables(n_qubits),
                scientific_question=(
                    "Does the topology-informed ansatz preserve its methods advantage "
                    f"when the IBM energy-validation lane is expanded to n={n_qubits}?"
                ),
                claim_boundary=(
                    "ansatz-method validation only; no variational convergence, "
                    "advantage, or backend-general performance claim"
                ),
            )
        )
    for n_qubits, priority in ((16, 6), (20, 7)):
        specs.append(
            CandidateSpec(
                candidate_id=f"methods_gpu_scaling_n{n_qubits}",
                paper_target="submission_003_rust_vqe_methods",
                n_qubits=n_qubits,
                lane_type="gpu_classical_methods_scaling",
                priority=priority,
                shots=0,
                repetitions=1,
                noise_scales=(1,),
                full_readout=False,
                source_circuits=3,
                observables=("ansatz_construction", "transpilation", "statevector_memory"),
                scientific_question=(
                    "Can the methods paper show credible n=16/n=20 scaling without "
                    "pretending that this is QPU evidence?"
                ),
                claim_boundary=(
                    "local classical/GPU resource evidence only; no hardware execution "
                    "or quantum-advantage claim"
                ),
            )
        )
    return tuple(specs)


def _representative_circuit(spec: CandidateSpec) -> QuantumCircuit:
    if spec.candidate_id.startswith("fim_"):
        return _fim_circuit(spec.n_qubits, "0" * (spec.n_qubits - 1) + "1", 4, 4.0)
    if spec.candidate_id.startswith("methods_"):
        return _xy_circuit(spec.n_qubits, "0" * spec.n_qubits, 1)
    return _xy_circuit(spec.n_qubits, "0" * (spec.n_qubits - 1) + "1", 10)


def _two_qubit_gates(circuit: QuantumCircuit) -> int:
    return int(
        sum(count for name, count in circuit.count_ops().items() if name in {"cx", "cz", "ecr"})
    )


def _transpile_representative(spec: CandidateSpec, backend: Any) -> tuple[int, int, int]:
    circuit = _representative_circuit(spec)
    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
    )
    return int(transpiled.depth()), int(transpiled.size()), _two_qubit_gates(transpiled)


def _main_circuit_count(spec: CandidateSpec) -> int:
    if spec.lane_type == "qpu_reduced_pauli_zne_stress":
        return (
            spec.source_circuits
            * len(spec.observables)
            * spec.repetitions
            * len(spec.noise_scales)
        )
    if spec.lane_type == "qpu_fim_replication_zne":
        return spec.source_circuits * spec.repetitions * len(spec.noise_scales)
    if spec.lane_type == "qpu_methods_ansatz_energy_validation":
        return spec.source_circuits * spec.repetitions
    return 0


def _readout_circuit_count(spec: CandidateSpec) -> int:
    if not spec.full_readout:
        return 0
    return 2**spec.n_qubits


def _gpu_seconds_estimate(spec: CandidateSpec, representative_size: int | None) -> float:
    if spec.lane_type != "gpu_classical_methods_scaling":
        return 0.0
    gate_factor = float(representative_size or spec.n_qubits)
    return max(30.0, gate_factor * 2.0 * 2 ** max(0, spec.n_qubits - 16) / 16.0)


def _qpu_seconds_per_circuit(spec: CandidateSpec) -> float:
    if spec.lane_type == "qpu_methods_ansatz_energy_validation":
        return METHODS_SECONDS_PER_CIRCUIT_ESTIMATE
    return SECONDS_PER_CIRCUIT_ESTIMATE


def estimate_candidate(
    spec: CandidateSpec,
    *,
    backend: Any,
    qpu_seconds_ceiling: float,
    gpu_seconds_ceiling: float,
) -> CandidateEstimate:
    """Estimate resources and return a submit/block decision."""
    depth: int | None = None
    size: int | None = None
    twoq: int | None = None
    reasons: list[str] = []
    try:
        depth, size, twoq = _transpile_representative(spec, backend)
    except Exception as exc:
        reasons.append(f"representative transpilation failed: {type(exc).__name__}: {exc}")

    main = _main_circuit_count(spec)
    readout = _readout_circuit_count(spec)
    total = main + readout
    estimated_qpu_seconds = total * _qpu_seconds_per_circuit(spec) if spec.shots else 0.0
    estimated_gpu_seconds = _gpu_seconds_estimate(spec, size)
    statevector_bytes = 16 * 2**spec.n_qubits
    dense_matrix_bytes = 16 * (2**spec.n_qubits) ** 2

    if spec.shots and readout > MAX_FULL_READOUT_STATES_FOR_QPU:
        reasons.append(f"full readout requires {readout} calibration states")
    if spec.shots and estimated_qpu_seconds > qpu_seconds_ceiling:
        reasons.append(
            f"estimated QPU seconds {estimated_qpu_seconds:.1f} exceeds ceiling {qpu_seconds_ceiling:.1f}"
        )
    if depth is not None and depth > MAX_TRANSPILED_DEPTH_FOR_QPU:
        reasons.append(f"representative depth {depth} exceeds gate {MAX_TRANSPILED_DEPTH_FOR_QPU}")
    if twoq is not None and twoq > MAX_TWO_QUBIT_GATES_FOR_QPU:
        reasons.append(
            f"representative 2Q gates {twoq} exceeds gate {MAX_TWO_QUBIT_GATES_FOR_QPU}"
        )
    if not spec.shots and estimated_gpu_seconds > gpu_seconds_ceiling:
        reasons.append(
            f"estimated GPU seconds {estimated_gpu_seconds:.1f} exceeds ceiling {gpu_seconds_ceiling:.1f}"
        )

    if spec.lane_type.startswith("qpu_"):
        status = "ready_for_qpu_preregistration" if not reasons else "blocked_or_needs_reduction"
    else:
        status = "ready_for_gpu_execution" if not reasons else "blocked_or_needs_offload"

    return CandidateEstimate(
        candidate_id=spec.candidate_id,
        paper_target=spec.paper_target,
        n_qubits=spec.n_qubits,
        lane_type=spec.lane_type,
        priority=spec.priority,
        status=status,
        decision_reasons=tuple(reasons),
        source_circuits=spec.source_circuits,
        observable_settings=len(spec.observables),
        main_circuits=main,
        readout_circuits=readout,
        total_circuits=total,
        shots=spec.shots,
        total_shots=total * spec.shots,
        estimated_qpu_seconds=float(estimated_qpu_seconds),
        estimated_gpu_seconds=float(estimated_gpu_seconds),
        statevector_bytes=statevector_bytes,
        dense_matrix_bytes=dense_matrix_bytes,
        representative_depth=depth,
        representative_size=size,
        representative_two_qubit_gates=twoq,
        max_depth_gate=MAX_TRANSPILED_DEPTH_FOR_QPU,
        max_two_qubit_gate=MAX_TWO_QUBIT_GATES_FOR_QPU,
        claim_boundary=spec.claim_boundary,
    )


def _usage_budget_summary(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    if usage is None:
        return {
            "available": False,
            "remaining_seconds": None,
            "reason": "not probed; rerun with --probe-ibm-budget and a live IBM backend",
        }
    remaining = usage.get("usage_remaining_seconds")
    public_usage_keys = {
        "usage_allocation_seconds",
        "usage_consumed_seconds",
        "usage_limit_reached",
        "usage_limit_seconds",
        "usage_period",
        "usage_remaining_seconds",
    }
    return {
        "available": True,
        "remaining_seconds": float(remaining) if remaining is not None else None,
        "usage": {key: usage[key] for key in public_usage_keys if key in usage},
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build the complete larger-system readiness payload."""
    backend, usage = load_backend_and_usage(
        args.backend,
        backend_qubits=args.backend_qubits,
        instance=args.instance,
        credentials_vault=args.credentials_vault,
        probe_ibm_budget=args.probe_ibm_budget,
    )
    estimates = [
        estimate_candidate(
            spec,
            backend=backend,
            qpu_seconds_ceiling=float(args.qpu_seconds_ceiling),
            gpu_seconds_ceiling=float(args.gpu_seconds_ceiling),
        )
        for spec in candidate_specs()
    ]
    ready_qpu = [row for row in estimates if row.status == "ready_for_qpu_preregistration"]
    ready_gpu = [row for row in estimates if row.status == "ready_for_gpu_execution"]
    qpu_seconds_ready = sum(row.estimated_qpu_seconds for row in ready_qpu)
    usage_summary = _usage_budget_summary(usage)
    remaining_seconds = usage_summary.get("remaining_seconds")
    two_backend_seconds = 2.0 * qpu_seconds_ready
    return {
        "schema": "scpn_large_system_submission_extension_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend": _backend_name(backend),
        "backend_mode": "generic_estimator"
        if args.backend == "generic_line"
        else "live_ibm_backend",
        "hardware_submission": False,
        "submit_command_required_later": True,
        "qpu_seconds_ceiling": float(args.qpu_seconds_ceiling),
        "gpu_seconds_ceiling": float(args.gpu_seconds_ceiling),
        "seconds_per_circuit_estimate": SECONDS_PER_CIRCUIT_ESTIMATE,
        "methods_seconds_per_circuit_estimate": METHODS_SECONDS_PER_CIRCUIT_ESTIMATE,
        "usage_budget": usage_summary,
        "ready_qpu_seconds_total": float(qpu_seconds_ready),
        "ready_qpu_minutes_total": float(qpu_seconds_ready / 60.0),
        "two_backend_ready_qpu_seconds_total": float(two_backend_seconds),
        "two_backend_ready_qpu_minutes_total": float(two_backend_seconds / 60.0),
        "remaining_after_one_backend_seconds": (
            float(remaining_seconds - qpu_seconds_ready)
            if isinstance(remaining_seconds, int | float)
            else None
        ),
        "remaining_after_two_backends_seconds": (
            float(remaining_seconds - two_backend_seconds)
            if isinstance(remaining_seconds, int | float)
            else None
        ),
        "one_backend_fits_live_budget": (
            bool(remaining_seconds >= qpu_seconds_ready)
            if isinstance(remaining_seconds, int | float)
            else None
        ),
        "two_backend_fits_live_budget": (
            bool(remaining_seconds >= two_backend_seconds)
            if isinstance(remaining_seconds, int | float)
            else None
        ),
        "candidates": [asdict(row) for row in sorted(estimates, key=lambda item: item.priority)],
        "recommended_order": [
            row.candidate_id
            for row in sorted(ready_qpu + ready_gpu, key=lambda item: item.priority)
        ],
        "paper_strategy": (
            "Use n=6/n=8 QPU lanes only if live backend transpilation remains below "
            "depth and two-qubit gates. Use n=16/n=20 as local GPU/classical scaling "
            "evidence for methods, not as hardware evidence."
        ),
    }


def _format_seconds(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}s ({value / 60.0:.2f} min)"


def write_markdown(path: Path, payload: Mapping[str, Any]) -> str:
    """Write a human-readable campaign-readiness note."""
    rows = payload["candidates"]
    lines = [
        "# Larger-System Submission Extension Readiness",
        "",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Backend: `{payload['backend']}` (`{payload['backend_mode']}`)",
        "- Hardware submitted: `False`",
        f"- Ready QPU estimate, one backend: `{_format_seconds(float(payload['ready_qpu_seconds_total']))}`",
        f"- Ready QPU estimate, Fez+Marrakesh pair: `{_format_seconds(float(payload['two_backend_ready_qpu_seconds_total']))}`",
        f"- IBM usage probe: `{payload['usage_budget']['available']}`",
        f"- IBM seconds remaining: `{payload['usage_budget']['remaining_seconds']}`",
        f"- One-backend live budget fit: `{payload['one_backend_fits_live_budget']}`",
        f"- Two-backend live budget fit: `{payload['two_backend_fits_live_budget']}`",
        f"- Remaining after one backend: `{_format_seconds(float(payload['remaining_after_one_backend_seconds'])) if payload['remaining_after_one_backend_seconds'] is not None else 'n/a'}`",
        f"- Remaining after two backends: `{_format_seconds(float(payload['remaining_after_two_backends_seconds'])) if payload['remaining_after_two_backends_seconds'] is not None else 'n/a'}`",
        "",
        "| Candidate | Paper | n | Status | Circuits | QPU estimate | Representative depth | 2Q gates |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        reasons = "; ".join(row["decision_reasons"])
        status = row["status"] if not reasons else f"{row['status']} ({reasons})"
        lines.append(
            "| `{candidate_id}` | `{paper_target}` | {n_qubits} | {status} | "
            "{total_circuits} | {qpu} | {depth} | {twoq} |".format(
                candidate_id=row["candidate_id"],
                paper_target=row["paper_target"],
                n_qubits=row["n_qubits"],
                status=status,
                total_circuits=row["total_circuits"],
                qpu=_format_seconds(float(row["estimated_qpu_seconds"])),
                depth=row["representative_depth"],
                twoq=row["representative_two_qubit_gates"],
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Order",
            "",
            *[
                f"{index}. `{candidate}`"
                for index, candidate in enumerate(payload["recommended_order"], start=1)
            ],
            "",
            "## Claim Boundary",
            "",
            str(payload["paper_strategy"]),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build larger-system readiness artefacts."""
    args = _parse_args(argv)
    payload = build_payload(args)
    timestamp = _timestamp()
    json_path = args.out_dir / f"large_system_submission_extension_readiness_{timestamp}.json"
    md_path = args.docs_dir / f"large_system_submission_extension_readiness_{timestamp}.md"
    json_sha = _write_json(json_path, payload)
    md_sha = write_markdown(md_path, payload)
    print(f"readiness_json={json_path}")
    print(f"readiness_sha256={json_sha}")
    print(f"readiness_md={md_path}")
    print(f"readiness_md_sha256={md_sha}")
    print(f"ready_qpu_minutes={float(payload['ready_qpu_minutes_total']):.2f}")
    print("recommended_order=" + ",".join(payload["recommended_order"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
