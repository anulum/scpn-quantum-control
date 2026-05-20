# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Rigetti QCS adapter for the hardware HAL
"""Rigetti pyQuil adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, cast

from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

RIGETTI_EXECUTION_MODE = "rigetti_pyquil_qcs"


def rigetti_quil_workload(
    program: str,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Quil program as a HAL workload for direct pyQuil execution."""

    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="quil",
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class RigettiQCSHALAdapter:
    """Synchronous pyQuil ``QuantumComputer`` adapter for the Rigetti QCS route."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        quantum_computer: Any | None = None,
        quantum_computer_name: str | None = None,
        quantum_computer_factory: Callable[[str], Any] | None = None,
        program_factory: Callable[[QuantumWorkload], Any] | None = None,
        shot_loop: Callable[[Any, int], Any] | None = None,
        readout_register: str = "ro",
        compile_program: bool = True,
    ) -> None:
        if profile.backend_id != "rigetti_qcs":
            raise ValueError("RigettiQCSHALAdapter requires the rigetti_qcs profile")
        if quantum_computer is None and quantum_computer_name is None:
            raise ValueError("quantum_computer or quantum_computer_name is required")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._quantum_computer = quantum_computer
        self._quantum_computer_name = quantum_computer_name or "injected"
        self._quantum_computer_factory = quantum_computer_factory
        self._program_factory = program_factory
        self._shot_loop = shot_loop
        self._readout_register = readout_register
        self._compile_program = compile_program
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        if not approval_id:
            raise PermissionError("approval_id is required for Rigetti QCS submission")
        if workload.ir_format != "quil":
            raise ValueError("Rigetti QCS direct adapter requires quil workloads")

        program = self._build_program(workload)
        executable = self._compile(program)
        raw_result = self._qc().run(executable)
        counts = _readout_counts(raw_result, self._readout_register, workload.n_qubits)
        job = QuantumJobRef(
            job_id=f"{self.backend_id}:{workload.workload_id}",
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": f"{self.backend_id}:{workload.workload_id}",
                "execution_mode": RIGETTI_EXECUTION_MODE,
                "quantum_computer": self._quantum_computer_name,
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=sum(counts.values()),
            metadata={
                "approval_id": approval_id,
                "execution_mode": RIGETTI_EXECUTION_MODE,
                "quantum_computer": self._quantum_computer_name,
                "readout_register": self._readout_register,
                "timestamp": _utc_now(),
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        return self._job(job).status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        result = self._results.get(job.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._job(job)
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _qc(self) -> Any:
        if self._quantum_computer is not None:
            return self._quantum_computer
        factory = self._quantum_computer_factory or _default_get_qc
        self._quantum_computer = factory(self._quantum_computer_name)
        return self._quantum_computer

    def _build_program(self, workload: QuantumWorkload) -> Any:
        factory = self._program_factory or _default_program_factory
        program = factory(workload)
        loop = self._shot_loop or _default_shot_loop
        return loop(program, workload.shots)

    def _compile(self, program: Any) -> Any:
        qc = self._qc()
        if not self._compile_program:
            return program
        compile_method = getattr(qc, "compile", None)
        if compile_method is None:
            raise TypeError("Rigetti quantum_computer object does not provide compile()")
        return compile_method(program)

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored


def _default_get_qc(quantum_computer_name: str) -> Any:
    try:
        pyquil = import_module("pyquil")
    except Exception as exc:
        raise RuntimeError("pyquil is required for RigettiQCSHALAdapter") from exc
    return pyquil.get_qc(quantum_computer_name)


def _default_program_factory(workload: QuantumWorkload) -> Any:
    try:
        program_cls = import_module("pyquil").Program
    except Exception as exc:
        raise RuntimeError("pyquil is required to construct Rigetti Quil programs") from exc
    return program_cls(workload.program)


def _default_shot_loop(program: Any, shots: int) -> Any:
    wrap = getattr(program, "wrap_in_numshots_loop", None)
    if wrap is None:
        raise TypeError("Rigetti program object does not provide wrap_in_numshots_loop()")
    wrapped = wrap(shots)
    return program if wrapped is None else wrapped


def _readout_counts(raw_result: Any, register: str, n_qubits: int) -> dict[str, int]:
    readout = _readout_rows(raw_result, register)
    counts: dict[str, int] = {}
    for row in readout:
        bitstring = _normalise_readout_row(row, n_qubits)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    if not counts:
        raise ValueError("Rigetti readout register did not contain any shots")
    return counts


def _readout_rows(raw_result: Any, register: str) -> Sequence[Sequence[Any]]:
    if isinstance(raw_result, Mapping):
        data = raw_result
    elif callable(getattr(raw_result, "get_register_map", None)):
        data = cast(Mapping[Any, Any], raw_result.get_register_map())
    else:
        raw_data = getattr(raw_result, "readout_data", None)
        if raw_data is None:
            raise ValueError("Rigetti result does not expose register readout data")
        data = cast(Mapping[Any, Any], raw_data)
    if not isinstance(data, Mapping):
        raise ValueError("Rigetti result does not expose register readout data")
    rows = data.get(register)
    if rows is None:
        raise ValueError(f"Rigetti result does not contain readout register: {register}")
    return cast(Sequence[Sequence[Any]], rows)


def _normalise_readout_row(row: Sequence[Any], n_qubits: int) -> str:
    bits = [int(value) for value in row]
    if len(bits) != n_qubits:
        raise ValueError("Rigetti readout width does not match workload qubit count")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Rigetti readout rows must contain binary values")
    return "".join(str(bit) for bit in bits)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "RIGETTI_EXECUTION_MODE",
    "RigettiQCSHALAdapter",
    "rigetti_quil_workload",
]
