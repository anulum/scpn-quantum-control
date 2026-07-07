# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- native LLVM/JIT whole-program AD execution evidence records
"""Evidence records for executed native LLVM/JIT whole-program autodiff.

These frozen value records capture the SCPN-native llvmlite path: a recorded
whole-program AD trace compiled to native LLVM and executed through the MCJIT
engine, with its value and gradient checked against the interpreted Program AD
reference. They are distinct from the Enzyme/MLIR records in
:mod:`scpn_quantum_control.compiler.mlir_enzyme_evidence`, which evidence the
external Enzyme toolchain; this surface evidences that SCPN's own native-JIT
lowering executes compiled autodiff *beyond scalar replay* -- over the static
dense linear-algebra families (determinant, inverse, linear solve, trace) -- and
fails closed past its declared size boundaries.

The frozen records carry no execution logic. The
``build_native_whole_program_ad_execution_evidence`` builder validates and
aggregates captured per-case rows, and
``run_native_whole_program_ad_execution_evidence`` is the capture runner: it
compiles and executes a fixed battery of whole-program AD traces through the
native LLVM/JIT path and checks each against the interpreted reference.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..differentiable import program_adjoint_value_and_grad
from .mlir_whole_program_native import compile_whole_program_ad_trace_to_native_llvm_jit

NATIVE_WHOLE_PROGRAM_AD_EXECUTION_STATUSES = frozenset({"executed", "fail_closed"})

_NATIVE_SYMBOL_PATTERN = re.compile(r"@(whole_program_ad_[A-Za-z0-9_]+?)_value\b")

_NATIVE_EXECUTION_CLAIM_BOUNDARY = (
    "Bounded SCPN-native llvmlite whole-program AD execution: static dense scalar "
    "and linear-algebra (determinant, inverse, linear solve, trace) value and gradient "
    "checked against the interpreted Program AD reference within float64 tolerance; no "
    "Enzyme, provider, hardware, performance, or beyond-declared-size promotion claim."
)


@dataclass(frozen=True)
class NativeWholeProgramADExecutionCase:
    """One captured native LLVM/JIT whole-program AD execution row.

    Parameters
    ----------
    case_id:
        Required stable identifier for the execution case.
    operation_family:
        The whole-program operation family exercised by the case, for example
        ``scalar``, ``determinant``, ``inverse``, ``solve`` or ``trace``.
    operand_dimension:
        The square operand dimension (or input length for ``scalar``) the case
        compiled and executed at; must be positive.
    status:
        ``executed`` for a compiled-and-run case checked against the interpreted
        reference, or ``fail_closed`` for a case that the native lowering
        declined to compile past its declared size boundary.
    value_error:
        Absolute error between the native-JIT value and the interpreted value,
        for ``executed`` cases only.
    gradient_error:
        Absolute max error between the native-JIT gradient and the interpreted
        gradient, for ``executed`` cases only.
    runtime_seconds:
        Bounded wall-clock compile-and-execute runtime, for ``executed`` cases.
    native_symbol:
        The emitted native kernel base symbol exercised, for ``executed`` cases.
    failure_class:
        The declared fail-closed reason, required for ``fail_closed`` cases.
    claim_boundary:
        The bounded-claim wording attached to the case.
    """

    case_id: str
    operation_family: str
    operand_dimension: int
    status: str
    value_error: float | None
    gradient_error: float | None
    runtime_seconds: float | None
    native_symbol: str | None
    failure_class: str | None
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.operation_family.strip():
            raise ValueError("operation_family must be non-empty")
        if self.operand_dimension <= 0:
            raise ValueError("operand_dimension must be positive")
        if self.status not in NATIVE_WHOLE_PROGRAM_AD_EXECUTION_STATUSES:
            raise ValueError("status must be executed or fail_closed")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        if self.status == "executed":
            if self.failure_class is not None:
                raise ValueError("executed cases must not carry a failure_class")
            if not self.native_symbol or not self.native_symbol.strip():
                raise ValueError("executed cases require a native_symbol")
            for name, value in (
                ("value_error", self.value_error),
                ("gradient_error", self.gradient_error),
                ("runtime_seconds", self.runtime_seconds),
            ):
                if value is None or value < 0.0 or not np.isfinite(value):
                    raise ValueError(f"executed cases require a finite non-negative {name}")
        else:
            if not self.failure_class or not self.failure_class.strip():
                raise ValueError("fail_closed cases require a failure_class")
            if (
                self.value_error is not None
                or self.gradient_error is not None
                or self.runtime_seconds is not None
                or self.native_symbol is not None
            ):
                raise ValueError("fail_closed cases must not carry execution metrics")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready execution-case metadata."""
        return {
            "case_id": self.case_id,
            "operation_family": self.operation_family,
            "operand_dimension": self.operand_dimension,
            "status": self.status,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "native_symbol": self.native_symbol,
            "failure_class": self.failure_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class NativeWholeProgramADExecutionEvidence:
    """Aggregate captured native LLVM/JIT whole-program AD execution evidence."""

    artifact_id: str
    cases: tuple[NativeWholeProgramADExecutionCase, ...]
    beyond_scalar_executed: bool
    executed_operation_families: tuple[str, ...]
    max_value_error: float
    max_gradient_error: float
    total_runtime_seconds: float
    gradient_parity_tolerance: float
    fail_closed_boundaries: Mapping[str, int]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if not self.cases:
            raise ValueError("evidence requires at least one execution case")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        executed = tuple(case for case in self.cases if case.status == "executed")
        if not executed:
            raise ValueError("evidence requires at least one executed case")
        families = tuple(dict.fromkeys(case.operation_family for case in executed))
        if tuple(self.executed_operation_families) != families:
            raise ValueError(
                "executed_operation_families must list the executed families in order"
            )
        non_scalar_executed = any(case.operation_family != "scalar" for case in executed)
        if self.beyond_scalar_executed != non_scalar_executed:
            raise ValueError("beyond_scalar_executed must reflect an executed non-scalar family")
        for name, value in (
            ("max_value_error", self.max_value_error),
            ("max_gradient_error", self.max_gradient_error),
            ("total_runtime_seconds", self.total_runtime_seconds),
            ("gradient_parity_tolerance", self.gradient_parity_tolerance),
        ):
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        observed_value_error = max((case.value_error or 0.0) for case in executed)
        observed_gradient_error = max((case.gradient_error or 0.0) for case in executed)
        if not np.isclose(self.max_value_error, observed_value_error, rtol=0.0, atol=1e-18):
            raise ValueError("max_value_error must equal the worst executed value_error")
        if not np.isclose(self.max_gradient_error, observed_gradient_error, rtol=0.0, atol=1e-18):
            raise ValueError("max_gradient_error must equal the worst executed gradient_error")
        if observed_gradient_error > self.gradient_parity_tolerance:
            raise ValueError("executed gradient_error exceeds the declared parity tolerance")
        boundaries = dict(self.fail_closed_boundaries)
        for family, size in boundaries.items():
            if not family or size <= 0:
                raise ValueError("fail_closed_boundaries must map families to positive sizes")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready aggregate native-execution evidence."""
        return {
            "artifact_id": self.artifact_id,
            "cases": [case.to_dict() for case in self.cases],
            "beyond_scalar_executed": self.beyond_scalar_executed,
            "executed_operation_families": list(self.executed_operation_families),
            "max_value_error": self.max_value_error,
            "max_gradient_error": self.max_gradient_error,
            "total_runtime_seconds": self.total_runtime_seconds,
            "gradient_parity_tolerance": self.gradient_parity_tolerance,
            "fail_closed_boundaries": dict(self.fail_closed_boundaries),
            "claim_boundary": self.claim_boundary,
        }


def build_native_whole_program_ad_execution_evidence(
    *,
    artifact_id: str,
    cases: Sequence[NativeWholeProgramADExecutionCase],
    gradient_parity_tolerance: float,
    fail_closed_boundaries: Mapping[str, int],
    claim_boundary: str,
) -> NativeWholeProgramADExecutionEvidence:
    """Build validated native LLVM/JIT whole-program AD execution evidence.

    The aggregate value/gradient errors, executed-family list and
    beyond-scalar flag are derived from the captured ``cases`` so the record
    cannot disagree with its rows.
    """
    ordered = tuple(cases)
    executed = tuple(case for case in ordered if case.status == "executed")
    if not executed:
        raise ValueError("at least one executed case is required")
    families = tuple(dict.fromkeys(case.operation_family for case in executed))
    max_value_error = max((case.value_error or 0.0) for case in executed)
    max_gradient_error = max((case.gradient_error or 0.0) for case in executed)
    total_runtime_seconds = float(sum((case.runtime_seconds or 0.0) for case in executed))
    beyond_scalar_executed = any(case.operation_family != "scalar" for case in executed)
    return NativeWholeProgramADExecutionEvidence(
        artifact_id=artifact_id,
        cases=ordered,
        beyond_scalar_executed=beyond_scalar_executed,
        executed_operation_families=families,
        max_value_error=float(max_value_error),
        max_gradient_error=float(max_gradient_error),
        total_runtime_seconds=total_runtime_seconds,
        gradient_parity_tolerance=float(gradient_parity_tolerance),
        fail_closed_boundaries=MappingProxyType(dict(fail_closed_boundaries)),
        claim_boundary=claim_boundary,
    )


def _well_conditioned_matrix_inputs(dimension: int) -> NDArray[np.float64]:
    """Return a flattened diagonally dominant ``dimension`` square matrix."""
    base = np.arange(1.0, dimension * dimension + 1.0, dtype=np.float64)
    matrix = base.reshape(dimension, dimension) % 3.0
    matrix = matrix + (dimension + 2.0) * np.eye(dimension, dtype=np.float64)
    return np.ascontiguousarray(matrix.reshape(-1))


def _scalar_program(values: Any) -> Any:
    return values[0] * values[1] + np.sin(values[2])


def _determinant_program(dimension: int) -> Callable[[Any], Any]:
    def program(values: Any) -> Any:
        return np.linalg.det(np.reshape(values, (dimension, dimension)))

    return program


def _trace_program(dimension: int) -> Callable[[Any], Any]:
    def program(values: Any) -> Any:
        return np.trace(np.reshape(values, (dimension, dimension)))

    return program


def _inverse_program(dimension: int) -> Callable[[Any], Any]:
    def program(values: Any) -> Any:
        return np.sum(np.linalg.inv(np.reshape(values, (dimension, dimension))))

    return program


def _solve_program(dimension: int) -> Callable[[Any], Any]:
    count = dimension * dimension

    def program(values: Any) -> Any:
        matrix = np.reshape(values[:count], (dimension, dimension))
        rhs = values[count : count + dimension]
        return np.sum(np.linalg.solve(matrix, rhs))

    return program


def _capture_execution_case(
    *,
    case_id: str,
    operation_family: str,
    operand_dimension: int,
    objective: Callable[[Any], Any],
    values: NDArray[np.float64],
) -> NativeWholeProgramADExecutionCase:
    """Compile, execute and check one whole-program AD trace; return its row."""
    start = time.perf_counter()
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, values, None)
    native_value, native_gradient = kernel.value_and_grad(values)
    runtime_seconds = time.perf_counter() - start
    reference_value, reference_gradient = program_adjoint_value_and_grad(objective, values, None)
    value_error = float(abs(float(native_value) - float(reference_value)))
    gradient_error = float(
        np.max(np.abs(np.asarray(native_gradient) - np.asarray(reference_gradient)))
    )
    match = _NATIVE_SYMBOL_PATTERN.search(kernel.llvm_ir)
    native_symbol = match.group(1) if match else kernel.cache_key
    return NativeWholeProgramADExecutionCase(
        case_id=case_id,
        operation_family=operation_family,
        operand_dimension=operand_dimension,
        status="executed",
        value_error=value_error,
        gradient_error=gradient_error,
        runtime_seconds=float(runtime_seconds),
        native_symbol=native_symbol,
        failure_class=None,
        claim_boundary=_NATIVE_EXECUTION_CLAIM_BOUNDARY,
    )


def _capture_fail_closed_case(
    *,
    case_id: str,
    operation_family: str,
    operand_dimension: int,
    objective: Callable[[Any], Any],
    values: NDArray[np.float64],
) -> NativeWholeProgramADExecutionCase:
    """Attempt to compile past a declared size boundary; expect a fail-closed reason."""
    try:
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, values, None)
    except ValueError as error:
        return NativeWholeProgramADExecutionCase(
            case_id=case_id,
            operation_family=operation_family,
            operand_dimension=operand_dimension,
            status="fail_closed",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            native_symbol=None,
            failure_class=str(error),
            claim_boundary=_NATIVE_EXECUTION_CLAIM_BOUNDARY,
        )
    raise AssertionError(
        f"{operation_family} dimension {operand_dimension} compiled instead of failing closed"
    )


def run_native_whole_program_ad_execution_evidence(
    *,
    artifact_id: str = "native-whole-program-ad-execution",
    gradient_parity_tolerance: float = 1e-6,
) -> NativeWholeProgramADExecutionEvidence:
    """Capture native LLVM/JIT whole-program AD execution evidence beyond scalar replay.

    Compiles and executes a fixed battery of whole-program AD traces -- a scalar
    program plus the static dense determinant, inverse, linear-solve and trace
    families -- through the native LLVM/JIT path, checks each value and gradient
    against the interpreted Program AD reference, and records the declared
    fail-closed size boundaries for the determinant and inverse families.
    """
    executed: list[NativeWholeProgramADExecutionCase] = [
        _capture_execution_case(
            case_id="scalar_poly_3",
            operation_family="scalar",
            operand_dimension=3,
            objective=_scalar_program,
            values=np.array([1.5, -2.0, 0.7], dtype=np.float64),
        )
    ]
    for dimension in (2, 3, 4):
        executed.append(
            _capture_execution_case(
                case_id=f"determinant_{dimension}x{dimension}",
                operation_family="determinant",
                operand_dimension=dimension,
                objective=_determinant_program(dimension),
                values=_well_conditioned_matrix_inputs(dimension),
            )
        )
    for dimension in (2, 3):
        executed.append(
            _capture_execution_case(
                case_id=f"inverse_{dimension}x{dimension}",
                operation_family="inverse",
                operand_dimension=dimension,
                objective=_inverse_program(dimension),
                values=_well_conditioned_matrix_inputs(dimension),
            )
        )
    for dimension in (2, 3):
        matrix = _well_conditioned_matrix_inputs(dimension)
        rhs = np.arange(1.0, dimension + 1.0, dtype=np.float64)
        executed.append(
            _capture_execution_case(
                case_id=f"solve_{dimension}x{dimension}",
                operation_family="solve",
                operand_dimension=dimension,
                objective=_solve_program(dimension),
                values=np.concatenate([matrix, rhs]),
            )
        )
    for dimension in (2, 3):
        executed.append(
            _capture_execution_case(
                case_id=f"trace_{dimension}x{dimension}",
                operation_family="trace",
                operand_dimension=dimension,
                objective=_trace_program(dimension),
                values=_well_conditioned_matrix_inputs(dimension),
            )
        )
    fail_closed = [
        _capture_fail_closed_case(
            case_id="determinant_20x20_fail_closed",
            operation_family="determinant",
            operand_dimension=20,
            objective=_determinant_program(20),
            values=_well_conditioned_matrix_inputs(20),
        ),
        _capture_fail_closed_case(
            case_id="inverse_8x8_fail_closed",
            operation_family="inverse",
            operand_dimension=8,
            objective=_inverse_program(8),
            values=_well_conditioned_matrix_inputs(8),
        ),
    ]
    return build_native_whole_program_ad_execution_evidence(
        artifact_id=artifact_id,
        cases=tuple(executed) + tuple(fail_closed),
        gradient_parity_tolerance=gradient_parity_tolerance,
        fail_closed_boundaries={"determinant": 20, "inverse": 8, "solve": 8},
        claim_boundary=_NATIVE_EXECUTION_CLAIM_BOUNDARY,
    )
