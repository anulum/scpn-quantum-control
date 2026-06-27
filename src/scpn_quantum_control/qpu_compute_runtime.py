# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control - QPU compute runtime
"""Simulator runtime and JSON I/O for QPU compute contracts."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.bridge import QPUDataArtifact, read_qpu_data_artifact
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.qpu_compute_types import (
    SUPPORTED_KERNELS,
    QPUComputeRequest,
    QPUComputeResult,
    QPUFusionResult,
    QPUNodeDescriptor,
    QPUStreamDelta,
)


def deterministic_counts(probabilities: Mapping[str, float], shots: int) -> dict[str, int]:
    """Convert exact probabilities into deterministic integer counts."""
    if shots < 1:
        raise ValueError("shots must be >= 1")
    raw = {str(key): max(0.0, float(value)) * shots for key, value in probabilities.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = shots - sum(counts.values())
    if remaining > 0:
        ranked = sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)
        for key in ranked[:remaining]:
            counts[key] += 1
    return {key: value for key, value in sorted(counts.items()) if value > 0}


def make_compute_request(
    artifact: QPUDataArtifact,
    *,
    kernel: str = "sync_dla",
    backend_policy: str = "simulator_statevector",
    shots: int = 1024,
    trotter_depth: int = 2,
    time_step: float = 0.1,
    lambda_fim: float = 0.0,
    coupling_scale: float = 1.0,
    output_dir: str | None = None,
) -> QPUComputeRequest:
    """Create a compute request from a validated QPU data artifact."""
    return QPUComputeRequest(
        qpu_data_artifact_sha256=artifact.to_dict()["artifact_sha256"],
        kernel=kernel,
        backend_policy=backend_policy,
        shots=shots,
        kernel_params={
            "trotter_depth": int(trotter_depth),
            "time_step": float(time_step),
            "lambda_fim": float(lambda_fim),
            "coupling_scale": float(coupling_scale),
        },
        budget={
            "max_jobs": 0,
            "max_qpu_seconds": 0,
            "hardware_enabled": False,
        },
        circuit_limits={
            "max_qubits": artifact.n_oscillators,
            "max_depth": None,
            "max_two_qubit_gates": None,
        },
        mitigation={"mode": "none", "reason": "local exact statevector dry run"},
        output_dir=output_dir,
        metadata={
            "source_mode": artifact.source_mode,
            "domain": artifact.domain,
            "source_name": artifact.source_name,
        },
    )


def execute_simulator_request(
    artifact: QPUDataArtifact,
    request: QPUComputeRequest,
) -> QPUComputeResult:
    """Execute a request on the deterministic local statevector simulator."""
    if request.qpu_data_artifact_sha256 != artifact.to_dict()["artifact_sha256"]:
        raise ValueError("request artifact hash does not match input artifact")
    if request.backend_policy != "simulator_statevector":
        raise ValueError("execute_simulator_request only accepts simulator_statevector")

    started = time.perf_counter()
    params = request.kernel_params
    ansatz = StructuredAnsatz.from_kuramoto(
        artifact.K_nm,
        omega=artifact.omega,
        trotter_depth=int(params.get("trotter_depth", 2)),
        time_step=float(params.get("time_step", 0.1)),
        lambda_fim=float(params.get("lambda_fim", 0.0)),
        coupling_scale=float(params.get("coupling_scale", 1.0)),
    )
    circuit = ansatz.build_circuit()
    state = Statevector.from_instruction(circuit)
    counts = deterministic_counts(state.probabilities_dict(), request.shots)

    observables: dict[str, float] = {}
    classification: dict[str, str] = {}
    if request.kernel in {"sync_witness", "sync_dla"}:
        observables.update(SyncOrderParameter()(counts=counts))
        classification.update(
            {
                "sync_order": "simulated_exact_statevector",
                "sync_order_z_magnetisation": "z_magnetisation_proxy_from_counts",
                "is_xy_kuramoto_order_parameter": "claim_boundary_flag",
            }
        )
    if request.kernel in {"dla_parity", "sync_dla"}:
        dla_values = DLAParityWitness()(counts=counts)
        observables.update({key: float(value) for key, value in dla_values.items()})
        classification.update({key: "simulated_exact_statevector" for key in dla_values})

    ops = circuit.count_ops()
    elapsed = time.perf_counter() - started
    return QPUComputeResult(
        request_sha256=request.request_sha256,
        qpu_data_artifact_sha256=request.qpu_data_artifact_sha256,
        status="DONE_SIMULATED",
        backend_name="local_statevector",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel=request.kernel,
        counts=counts,
        observables=observables,
        observable_classification=classification,
        circuit_metadata={
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "operations": {str(key): int(value) for key, value in ops.items()},
            "trotter_depth": int(params.get("trotter_depth", 2)),
            "time_step": float(params.get("time_step", 0.1)),
            "lambda_fim": float(params.get("lambda_fim", 0.0)),
            "coupling_scale": float(params.get("coupling_scale", 1.0)),
        },
        mitigation=request.mitigation,
        timings={"wall_seconds": elapsed},
        simulator_seed=None,
        metadata={
            "artifact_domain": artifact.domain,
            "artifact_source_name": artifact.source_name,
            "artifact_source_mode": artifact.source_mode,
            "counts_embedded": True,
        },
    )


def write_compute_request(path: str | Path, request: QPUComputeRequest) -> None:
    """Write a compute request to JSON."""
    Path(path).write_text(request.to_json() + "\n", encoding="utf-8")


def read_compute_request(path: str | Path) -> QPUComputeRequest:
    """Read a compute request from JSON."""
    return QPUComputeRequest.from_json(Path(path).read_text(encoding="utf-8"))


def write_compute_result(path: str | Path, result: QPUComputeResult) -> None:
    """Write a compute result to JSON."""
    Path(path).write_text(result.to_json() + "\n", encoding="utf-8")


def read_compute_result(path: str | Path) -> QPUComputeResult:
    """Read a compute result from JSON."""
    return QPUComputeResult.from_json(Path(path).read_text(encoding="utf-8"))


def write_node_descriptor(path: str | Path, descriptor: QPUNodeDescriptor) -> None:
    """Write a QPU node descriptor to JSON."""
    Path(path).write_text(descriptor.to_json() + "\n", encoding="utf-8")


def read_node_descriptor(path: str | Path) -> QPUNodeDescriptor:
    """Read a QPU node descriptor from JSON."""
    return QPUNodeDescriptor.from_json(Path(path).read_text(encoding="utf-8"))


def write_stream_delta(path: str | Path, delta: QPUStreamDelta) -> None:
    """Write a QPU stream delta to JSON."""
    Path(path).write_text(delta.to_json() + "\n", encoding="utf-8")


def read_stream_delta(path: str | Path) -> QPUStreamDelta:
    """Read a QPU stream delta from JSON."""
    return QPUStreamDelta.from_json(Path(path).read_text(encoding="utf-8"))


def write_fusion_result(path: str | Path, fusion: QPUFusionResult) -> None:
    """Write a QPU fusion result to JSON."""
    Path(path).write_text(fusion.to_json() + "\n", encoding="utf-8")


def read_fusion_result(path: str | Path) -> QPUFusionResult:
    """Read a QPU fusion result from JSON."""
    return QPUFusionResult.from_json(Path(path).read_text(encoding="utf-8"))


def run_simulator_from_artifact(
    artifact_path: str | Path,
    *,
    request_out: str | Path | None = None,
    result_out: str | Path | None = None,
    kernel: str = "sync_dla",
    shots: int = 1024,
    trotter_depth: int = 2,
    time_step: float = 0.1,
    lambda_fim: float = 0.0,
    coupling_scale: float = 1.0,
    require_publication_safe: bool = True,
) -> QPUComputeResult:
    """Run the simulator compute unit from an artifact path."""
    artifact = read_qpu_data_artifact(artifact_path)
    if require_publication_safe:
        artifact.require_publication_safe()
    request = make_compute_request(
        artifact,
        kernel=kernel,
        shots=shots,
        trotter_depth=trotter_depth,
        time_step=time_step,
        lambda_fim=lambda_fim,
        coupling_scale=coupling_scale,
        output_dir=None if result_out is None else str(Path(result_out).parent),
    )
    result = execute_simulator_request(artifact, request)
    if request_out is not None:
        write_compute_request(request_out, request)
    if result_out is not None:
        write_compute_result(result_out, result)
    return result


__all__ = [
    "SUPPORTED_KERNELS",
    "deterministic_counts",
    "execute_simulator_request",
    "make_compute_request",
    "read_compute_request",
    "read_compute_result",
    "read_fusion_result",
    "read_node_descriptor",
    "read_stream_delta",
    "run_simulator_from_artifact",
    "write_compute_request",
    "write_compute_result",
    "write_fusion_result",
    "write_node_descriptor",
    "write_stream_delta",
]
