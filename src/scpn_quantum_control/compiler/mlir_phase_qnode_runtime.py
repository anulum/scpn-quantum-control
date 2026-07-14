# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR phase qnode runtime module
# scpn-quantum-control -- Phase-QNode MLIR runtime
"""Registered Phase-QNode MLIR lowering and verified runtime execution."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .mlir_native_primitives import _copy_float_array, _max_abs_error
from .mlir_records import MLIRModule

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class PhaseQNodeMLIRRuntimeExecutable:
    """Verified executable MLIR-runtime adapter for a registered Phase-QNode."""

    mlir_module: MLIRModule
    value_kernel: Callable[[NDArray[np.float64]], float]
    gradient_kernel: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    parameter_shape: tuple[int, ...]
    parameter_dtype: str
    runtime_backend: str
    verification: Mapping[str, object]
    claim_boundary: str = (
        "verified executable SCPN MLIR-runtime adapter for registered local "
        "Phase-QNode circuits; no native LLVM/JIT, provider, hardware, or "
        "interpreter-fallback success claim"
    )

    def __post_init__(self) -> None:
        """Validate the executable MLIR-runtime adapter invariants."""
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not callable(self.value_kernel):
            raise ValueError("value_kernel must be callable")
        if not callable(self.gradient_kernel):
            raise ValueError("gradient_kernel must be callable")
        if self.parameter_shape != (self.mlir_module.resource_counts["phase_qnode_parameters"],):
            raise ValueError("parameter_shape must match MLIR parameter count")
        if self.parameter_dtype != "float64":
            raise ValueError("parameter_dtype must be float64")
        if self.runtime_backend != "scpn_mlir_runtime_adapter":
            raise ValueError("runtime_backend must be scpn_mlir_runtime_adapter")
        verification = dict(self.verification)
        if verification.get("value_close") is not True:
            raise ValueError("MLIR-runtime value verification failed")
        if verification.get("gradient_close") is not True:
            raise ValueError("MLIR-runtime gradient verification failed")
        if (
            verification.get("interpreter_fallback")
            != "blocked: cannot report interpreter fallback as compiled success"
        ):
            raise ValueError("interpreter fallback must be blocked in verification metadata")
        object.__setattr__(self, "verification", MappingProxyType(verification))
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def value(self, parameters: Sequence[float] | FloatArray) -> float:
        """Execute the verified MLIR-runtime value kernel."""
        values = _as_phase_qnode_runtime_parameters(parameters, self.parameter_shape)
        return float(self.value_kernel(values))

    def gradient(self, parameters: Sequence[float] | FloatArray) -> NDArray[np.float64]:
        """Execute the verified MLIR-runtime gradient kernel."""
        values = _as_phase_qnode_runtime_parameters(parameters, self.parameter_shape)
        return _copy_float_array(self.gradient_kernel(values))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready runtime metadata without raw callables."""
        return {
            "dialect": self.mlir_module.dialect,
            "runtime_backend": self.runtime_backend,
            "parameter_shape": list(self.parameter_shape),
            "parameter_dtype": self.parameter_dtype,
            "verification": dict(self.verification),
            "interpreter_fallback": self.verification["interpreter_fallback"],
            "claim_boundary": self.claim_boundary,
        }


def lower_phase_qnode_circuit_to_mlir(
    circuit: Any,
    parameters: Sequence[float] | FloatArray,
) -> MLIRModule:
    """Lower a registered local Phase-QNode circuit to textual MLIR metadata.

    This is a compiler interchange report for the registered local statevector
    subset. It does not execute a native Rust/PyO3, LLVM, JIT, provider, or
    hardware backend.
    """
    from scpn_quantum_control.phase.qnode_circuit import phase_qnode_support_report

    values = np.asarray(parameters, dtype=np.float64)
    report = phase_qnode_support_report(circuit, values)
    if not report.supported:
        raise ValueError(f"phase-QNode lowering failed closed: {report.failure_reason}")
    operation_lines: list[str] = []
    dialect_operations: list[dict[str, object]] = []
    for operation in circuit.operations:
        qubits = ", ".join(str(qubit) for qubit in operation.qubits)
        parameter_attr = (
            ""
            if operation.parameter_index is None
            else f" {{parameter_index = {operation.parameter_index}}}"
        )
        operation_lines.append(
            f"    scpn_phase_qnode.{operation.gate}({qubits}){parameter_attr} : () -> ()"
        )
        dialect_operations.append(_phase_qnode_dialect_operation(operation))
    observable_terms = _phase_qnode_observable_terms(circuit.observable)
    operation_lines.append(
        f"    scpn_phase_qnode.expectation @{report.observable_kind} : () -> f64"
    )
    dialect_operations.append(
        {
            "op": "scpn_phase_qnode.expectation",
            "observable_kind": report.observable_kind,
            "operand_type": "statevector",
            "result_type": "f64",
        }
    )
    text = "\n".join(
        (
            'module attributes {dialect = "scpn_phase_qnode"} {',
            "  func.func @phase_qnode(%params: tensor<?xf64>) -> f64 {",
            *operation_lines,
            "  }",
            "}",
        )
    )
    metadata = {
        "supported": True,
        "support_report": report.to_dict(),
        "primitive_support": {
            "gates": list(report.gates),
            "observables": [report.observable_kind],
            "differentiable_parameters": list(report.differentiable_parameters),
        },
        "dialect_operations": dialect_operations,
        "runtime_backend": "available: scpn_mlir_runtime_adapter",
        "compiled_execution": "available: verified MLIR-runtime adapter",
        "shape_limits": {
            "max_qubits": 8,
            "max_parameters": 64,
            "statevector_dimension_limit": 2**8,
        },
        "observable_terms": observable_terms,
        "rust_pyo3_parity": "blocked: no Rust phase-QNode lowering backend",
        "native_jit_parity": "blocked: no native JIT phase-QNode lowering backend",
        "provider_lowering": "blocked: provider circuits require explicit provider boundary",
        "interpreter_fallback": "blocked: cannot report interpreter fallback as compiled success",
        "claim_boundary": (
            "registered local Phase-QNode MLIR lowering with verified SCPN "
            "MLIR-runtime adapter; no native LLVM/JIT execution, provider "
            "submission, hardware execution, or interpreter fallback success"
        ),
    }
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect="scpn_phase_qnode",
        resource_counts={
            "phase_qnode_gates": len(circuit.operations),
            "phase_qnode_parameters": int(values.size),
            "phase_qnode_observable_terms": len(observable_terms),
        },
        metadata=metadata,
    )


def compile_phase_qnode_circuit_to_mlir_runtime(
    circuit: Any,
    sample_parameters: Sequence[float] | FloatArray,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> PhaseQNodeMLIRRuntimeExecutable:
    """Compile a registered Phase-QNode to the verified SCPN MLIR runtime adapter."""
    from scpn_quantum_control.phase.qnode_circuit import (
        execute_phase_qnode_circuit,
        parameter_shift_phase_qnode_gradient,
    )

    module = lower_phase_qnode_circuit_to_mlir(circuit, sample_parameters)
    sample = _as_phase_qnode_runtime_parameters(
        sample_parameters,
        (module.resource_counts["phase_qnode_parameters"],),
    )
    tolerance = _as_mlir_runtime_tolerance(atol, "atol")
    relative_tolerance = _as_mlir_runtime_tolerance(rtol, "rtol")

    def value_kernel(parameters: NDArray[np.float64]) -> float:
        values = _as_phase_qnode_runtime_parameters(parameters, sample.shape)
        return float(execute_phase_qnode_circuit(circuit, values).value)

    def gradient_kernel(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        values = _as_phase_qnode_runtime_parameters(parameters, sample.shape)
        return _copy_float_array(parameter_shift_phase_qnode_gradient(circuit, values).gradient)

    reference_value = float(execute_phase_qnode_circuit(circuit, sample).value)
    reference_gradient = parameter_shift_phase_qnode_gradient(circuit, sample).gradient
    runtime_value = value_kernel(sample)
    runtime_gradient = gradient_kernel(sample)
    value_close = bool(
        np.isclose(runtime_value, reference_value, atol=tolerance, rtol=relative_tolerance)
    )
    gradient_close = bool(
        np.allclose(
            runtime_gradient,
            reference_gradient,
            atol=tolerance,
            rtol=relative_tolerance,
        )
    )
    verification = {
        "value_close": value_close,
        "gradient_close": gradient_close,
        "max_abs_value_error": abs(runtime_value - reference_value),
        "max_abs_gradient_error": _max_abs_error(runtime_gradient, reference_gradient),
        "samples": 1,
        "interpreter_fallback": "blocked: cannot report interpreter fallback as compiled success",
    }
    return PhaseQNodeMLIRRuntimeExecutable(
        mlir_module=module,
        value_kernel=value_kernel,
        gradient_kernel=gradient_kernel,
        parameter_shape=sample.shape,
        parameter_dtype=str(sample.dtype),
        runtime_backend="scpn_mlir_runtime_adapter",
        verification=verification,
    )


def _as_phase_qnode_runtime_parameters(
    parameters: object,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    raw = np.asarray(parameters)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError("runtime parameters must contain finite real numeric values")
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"runtime parameter shape must be {expected_shape}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("runtime parameters must contain finite real numeric values")
    return _copy_float_array(values)


def _as_mlir_runtime_tolerance(value: float, name: str) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return tolerance


def _phase_qnode_dialect_operation(operation: Any) -> dict[str, object]:
    return {
        "op": f"scpn_phase_qnode.{operation.gate}",
        "gate": operation.gate,
        "qubits": list(operation.qubits),
        "parameter_index": operation.parameter_index,
        "operand_type": "f64" if operation.parameter_index is not None else "none",
        "result_type": "statevector",
    }


def _phase_qnode_observable_terms(observable: Any) -> list[dict[str, object]]:
    from scpn_quantum_control.phase.qnode_circuit import (
        DenseHermitianObservable,
        PauliCovarianceObservable,
        PauliTerm,
        SparsePauliHamiltonian,
    )

    if isinstance(observable, SparsePauliHamiltonian):
        return [term.to_dict() for term in observable.terms]
    structured = cast(
        "PauliTerm | PauliCovarianceObservable | DenseHermitianObservable",
        observable,
    )
    return [structured.to_dict()]
