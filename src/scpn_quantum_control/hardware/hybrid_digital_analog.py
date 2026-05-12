# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hybrid Digital-Analog Kuramoto Execution
"""Hybrid digital-analog execution plans for Kuramoto-XY workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, qasm2

from ..kuramoto_core import KuramotoProblem, build_kuramoto_problem, compile_trotter_circuit
from .analog_kuramoto import (
    AnalogKuramotoBackend,
    AnalogKuramotoPlatform,
    AnalogKuramotoProgram,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class HybridRoute(str, Enum):
    """Execution route assigned to a Kuramoto coupling term."""

    DIGITAL = "digital"
    ANALOG = "analog"


@dataclass(frozen=True)
class HybridCouplingAssignment:
    """Route decision for one upper-triangular coupling."""

    source: int
    target: int
    coefficient: float
    route: HybridRoute

    def to_payload(self) -> dict[str, float | int | str]:
        """Return a serialisable route assignment."""
        return {
            "source": self.source,
            "target": self.target,
            "coefficient": self.coefficient,
            "route": self.route.value,
        }


@dataclass(frozen=True)
class HybridCouplingPartition:
    """Analog/digital split of a validated Kuramoto coupling matrix."""

    analog_K_nm: FloatArray
    digital_K_nm: FloatArray
    assignments: tuple[HybridCouplingAssignment, ...]

    @property
    def n_couplings(self) -> int:
        """Number of non-zero couplings considered by the splitter."""
        return len(self.assignments)

    @property
    def n_analog_couplings(self) -> int:
        """Number of couplings routed to analog execution."""
        return sum(assignment.route == HybridRoute.ANALOG for assignment in self.assignments)

    @property
    def n_digital_couplings(self) -> int:
        """Number of couplings routed to digital residual execution."""
        return sum(assignment.route == HybridRoute.DIGITAL for assignment in self.assignments)

    @property
    def analog_norm(self) -> float:
        """Frobenius norm of the analog coupling block."""
        return float(np.linalg.norm(self.analog_K_nm))

    @property
    def digital_norm(self) -> float:
        """Frobenius norm of the digital residual block."""
        return float(np.linalg.norm(self.digital_K_nm))

    @property
    def analog_fraction(self) -> float:
        """Fraction of non-zero couplings routed to analog execution."""
        if not self.assignments:
            return 0.0
        return self.n_analog_couplings / len(self.assignments)

    def to_payload(self) -> dict[str, Any]:
        """Return a serialisable partition summary."""
        return {
            "n_couplings": self.n_couplings,
            "n_analog_couplings": self.n_analog_couplings,
            "n_digital_couplings": self.n_digital_couplings,
            "analog_norm": self.analog_norm,
            "digital_norm": self.digital_norm,
            "analog_fraction": self.analog_fraction,
            "assignments": [assignment.to_payload() for assignment in self.assignments],
        }


@dataclass(frozen=True)
class HybridDigitalAnalogProgram:
    """Compiled hybrid execution plan with analog and digital residual blocks."""

    platform: AnalogKuramotoPlatform
    duration: float
    digital_time: float
    partition: HybridCouplingPartition
    analog_program: AnalogKuramotoProgram
    digital_circuit: QuantumCircuit
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators represented by the hybrid plan."""
        return self.analog_program.n_oscillators

    @property
    def n_analog_couplers(self) -> int:
        """Number of analog-native couplers in the plan."""
        return self.partition.n_analog_couplings

    @property
    def n_digital_couplers(self) -> int:
        """Number of residual digital couplers in the plan."""
        return self.partition.n_digital_couplings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable hybrid programme dictionary."""
        return {
            "platform": self.platform.value,
            "duration": self.duration,
            "digital_time": self.digital_time,
            "partition": self.partition.to_payload(),
            "analog_program": self.analog_program.to_dict(),
            "payload": self.payload,
            "metadata": self.metadata,
        }


@runtime_checkable
class HybridDigitalAnalogBackendProtocol(Protocol):
    """Provider-adapter contract for hybrid digital-analog compilers."""

    name: str

    def is_available(self) -> bool:
        """True iff the compiler can run in this environment."""
        ...

    def compile(
        self,
        problem: KuramotoProblem,
        *,
        duration: float,
        digital_time: float | None = None,
        max_analog_couplers: int | None = None,
        analog_threshold: float = 0.0,
        trotter_steps: int = 8,
        trotter_order: int = 1,
    ) -> HybridDigitalAnalogProgram:
        """Compile a hybrid digital-analog execution plan."""
        ...


class HybridDigitalAnalogBackend:
    """Compiler for split Kuramoto workloads across analog and digital blocks."""

    name = "hybrid_digital_analog"

    def __init__(
        self,
        platform: AnalogKuramotoPlatform | str = AnalogKuramotoPlatform.CIRCUIT_QED,
        *,
        zero_threshold: float = 1e-12,
        c6_coefficient: float = 1.0,
    ) -> None:
        self.platform = AnalogKuramotoPlatform(platform)
        self.zero_threshold = _require_non_negative(zero_threshold, "zero_threshold")
        self.analog_backend = AnalogKuramotoBackend(
            self.platform,
            c6_coefficient=c6_coefficient,
            zero_threshold=zero_threshold,
        )

    def is_available(self) -> bool:
        """The built-in hybrid compiler has no optional runtime dependency."""
        return True

    def compile(
        self,
        problem: KuramotoProblem,
        *,
        duration: float,
        digital_time: float | None = None,
        max_analog_couplers: int | None = None,
        analog_threshold: float = 0.0,
        trotter_steps: int = 8,
        trotter_order: int = 1,
    ) -> HybridDigitalAnalogProgram:
        """Compile a Kuramoto problem into an analog block plus digital residual."""
        duration = _require_positive(duration, "duration")
        resolved_digital_time = (
            duration if digital_time is None else _require_positive(digital_time, "digital_time")
        )
        if trotter_steps < 1:
            raise ValueError("trotter_steps must be at least 1")
        if trotter_order < 1:
            raise ValueError("trotter_order must be at least 1")

        partition = partition_kuramoto_couplings(
            problem.K_nm,
            max_analog_couplers=max_analog_couplers,
            analog_threshold=analog_threshold,
            zero_threshold=self.zero_threshold,
        )
        analog_problem = build_kuramoto_problem(
            partition.analog_K_nm,
            problem.omega,
            metadata={**dict(problem.metadata), "hybrid_route": "analog_native"},
        )
        analog_program = self.analog_backend.compile(analog_problem, duration=duration)

        digital_omega = np.zeros(problem.n_oscillators, dtype=np.float64)
        digital_problem = build_kuramoto_problem(
            partition.digital_K_nm,
            digital_omega,
            metadata={**dict(problem.metadata), "hybrid_route": "digital_residual"},
        )
        digital_circuit = _compile_residual_circuit(
            digital_problem,
            time=resolved_digital_time,
            trotter_steps=trotter_steps,
            trotter_order=trotter_order,
            has_residual=partition.n_digital_couplings > 0,
        )
        digital_payload = _digital_circuit_payload(
            digital_circuit,
            digital_time=resolved_digital_time,
            trotter_steps=trotter_steps,
            trotter_order=trotter_order,
            has_residual=partition.n_digital_couplings > 0,
        )
        payload = {
            "schema": "hybrid_digital_analog_v1",
            "platform": self.platform.value,
            "duration": duration,
            "digital_time": resolved_digital_time,
            "partition": partition.to_payload(),
            "schedule": [
                {
                    "route": "analog_native",
                    "duration": duration,
                    "payload": analog_program.payload,
                },
                {
                    "route": "digital_residual",
                    "duration": resolved_digital_time,
                    "payload": digital_payload,
                },
            ],
        }
        metadata = {
            "n_oscillators": problem.n_oscillators,
            "n_analog_couplers": partition.n_analog_couplings,
            "n_digital_couplers": partition.n_digital_couplings,
            "analog_threshold": _require_non_negative(analog_threshold, "analog_threshold"),
            "zero_threshold": self.zero_threshold,
            "digital_local_detunings": "zero_residual_to_avoid_double_counting",
            "digital_circuit_depth": digital_circuit.depth(),
            "digital_circuit_size": digital_circuit.size(),
        }
        metadata.update(problem.to_metadata()["metadata"])
        return HybridDigitalAnalogProgram(
            platform=self.platform,
            duration=duration,
            digital_time=resolved_digital_time,
            partition=partition,
            analog_program=analog_program,
            digital_circuit=digital_circuit,
            payload=payload,
            metadata=metadata,
        )


def compile_hybrid_digital_analog(
    K_nm: FloatArray,
    omega: FloatArray,
    *,
    platform: AnalogKuramotoPlatform | str,
    duration: float,
    digital_time: float | None = None,
    max_analog_couplers: int | None = None,
    analog_threshold: float = 0.0,
    trotter_steps: int = 8,
    trotter_order: int = 1,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> HybridDigitalAnalogProgram:
    """Validate inputs and compile a hybrid digital-analog execution plan."""
    problem = build_kuramoto_problem(K_nm, omega, metadata=metadata or {})
    backend = HybridDigitalAnalogBackend(platform)
    return backend.compile(
        problem,
        duration=duration,
        digital_time=digital_time,
        max_analog_couplers=max_analog_couplers,
        analog_threshold=analog_threshold,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
    )


def hybrid_digital_analog_factory() -> HybridDigitalAnalogBackend:
    """Entry-point target for the built-in hybrid compiler backend."""
    return HybridDigitalAnalogBackend()


def partition_kuramoto_couplings(
    K_nm: FloatArray,
    *,
    max_analog_couplers: int | None = None,
    analog_threshold: float = 0.0,
    zero_threshold: float = 1e-12,
) -> HybridCouplingPartition:
    """Split a symmetric coupling matrix into analog and digital blocks."""
    K = _validate_coupling_matrix(K_nm)
    analog_threshold = _require_non_negative(analog_threshold, "analog_threshold")
    zero_threshold = _require_non_negative(zero_threshold, "zero_threshold")
    budget = _resolve_budget(K.shape[0], max_analog_couplers)
    analog, digital, rows, cols, route_codes = _partition_kernel(
        K,
        analog_budget=budget,
        analog_threshold=analog_threshold,
        zero_threshold=zero_threshold,
    )
    assignments = tuple(
        HybridCouplingAssignment(
            source=int(row),
            target=int(col),
            coefficient=float(K[int(row), int(col)]),
            route=HybridRoute.ANALOG if int(code) == 1 else HybridRoute.DIGITAL,
        )
        for row, col, code in zip(rows, cols, route_codes)
    )
    analog.setflags(write=False)
    digital.setflags(write=False)
    return HybridCouplingPartition(
        analog_K_nm=analog,
        digital_K_nm=digital,
        assignments=assignments,
    )


def _partition_kernel(
    K_nm: FloatArray,
    *,
    analog_budget: int,
    analog_threshold: float,
    zero_threshold: float,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray, IntArray]:
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "hybrid_coupling_partition"):
            analog, digital, rows, cols, route_codes = _engine.hybrid_coupling_partition(
                K_nm.ravel(),
                K_nm.shape[0],
                analog_budget,
                analog_threshold,
                zero_threshold,
            )
            n = K_nm.shape[0]
            return (
                np.asarray(analog, dtype=np.float64).reshape(n, n),
                np.asarray(digital, dtype=np.float64).reshape(n, n),
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
                np.asarray(route_codes, dtype=np.int64),
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return _partition_numpy(
        K_nm,
        analog_budget=analog_budget,
        analog_threshold=analog_threshold,
        zero_threshold=zero_threshold,
    )


def _partition_numpy(
    K_nm: FloatArray,
    *,
    analog_budget: int,
    analog_threshold: float,
    zero_threshold: float,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray, IntArray]:
    n = K_nm.shape[0]
    candidates: list[tuple[int, int, float]] = []
    for source in range(n):
        for target in range(source + 1, n):
            magnitude = abs(float(K_nm[source, target]))
            if magnitude > zero_threshold and magnitude >= analog_threshold:
                candidates.append((source, target, magnitude))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    selected = {(source, target) for source, target, _ in candidates[:analog_budget]}

    analog = np.zeros_like(K_nm, dtype=np.float64)
    digital = np.zeros_like(K_nm, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    route_codes: list[int] = []
    for source in range(n):
        for target in range(source + 1, n):
            coefficient = float(K_nm[source, target])
            if abs(coefficient) <= zero_threshold:
                continue
            rows.append(source)
            cols.append(target)
            if (source, target) in selected:
                analog[source, target] = analog[target, source] = coefficient
                route_codes.append(1)
            else:
                digital[source, target] = digital[target, source] = coefficient
                route_codes.append(0)
    return (
        analog,
        digital,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(route_codes, dtype=np.int64),
    )


def _compile_residual_circuit(
    problem: KuramotoProblem,
    *,
    time: float,
    trotter_steps: int,
    trotter_order: int,
    has_residual: bool,
) -> QuantumCircuit:
    if not has_residual:
        return QuantumCircuit(problem.n_oscillators, name="digital_residual_identity")
    return compile_trotter_circuit(
        problem,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
    )


def _digital_circuit_payload(
    circuit: QuantumCircuit,
    *,
    digital_time: float,
    trotter_steps: int,
    trotter_order: int,
    has_residual: bool,
) -> dict[str, Any]:
    return {
        "schema": "digital_residual_qasm2_v1",
        "has_residual": has_residual,
        "duration": digital_time,
        "trotter_steps": trotter_steps,
        "trotter_order": trotter_order,
        "n_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "qasm2": qasm2.dumps(circuit),
    }


def _validate_coupling_matrix(K_nm: FloatArray) -> FloatArray:
    K = np.array(K_nm, dtype=np.float64, copy=True)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K_nm must be a square matrix, got shape {K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("K_nm must contain only finite values")
    if not np.allclose(K, K.T, atol=1e-12, rtol=1e-12):
        raise ValueError("K_nm must be symmetric for hybrid splitting")
    np.fill_diagonal(K, 0.0)
    return K


def _resolve_budget(n: int, max_analog_couplers: int | None) -> int:
    total_edges = n * (n - 1) // 2
    if max_analog_couplers is None:
        return total_edges
    if max_analog_couplers < 0:
        raise ValueError("max_analog_couplers must be non-negative")
    return min(int(max_analog_couplers), total_edges)


def _require_positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _require_non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
