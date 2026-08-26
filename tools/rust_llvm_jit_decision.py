# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust-JIT-decision Rust LLVM/JIT decision evidence
"""Capture the bounded evidence needed for the Rust-JIT decision Rust JIT decision.

The comparison deliberately measures the existing Rust Program-AD replay and
the existing Python-owned llvmlite JIT on the same trace and inputs.  It does
not implement a Rust JIT.  Timings are descriptive unless the caller supplies
an externally isolated environment and records that fact explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.compiler import (
    clear_native_whole_program_ad_compile_cache,
    compile_whole_program_ad_trace_to_native_llvm_jit,
)
from scpn_quantum_control.differentiable import (
    value_and_grad_program_ad_effect_ir_with_rust,
    whole_program_value_and_grad,
)

SCHEMA = "rust_llvm_jit_decision.v1"
CLAIM_BOUNDARY = (
    "BL-38 decision evidence only: functional parity and non-isolated descriptive "
    "timings for the existing bounded Rust Program-AD replay and Python-owned "
    "llvmlite JIT. No Rust LLVM/JIT implementation, isolated performance, provider, "
    "hardware, GPU, or public performance claim."
)
PARITY_TOLERANCE = 1.0e-9

FloatArray = NDArray[np.float64]
Objective = Callable[[Any], Any]


@dataclass(frozen=True)
class DecisionKernel:
    """One source-owned comparison kernel and its fixed input vector."""

    case_id: str
    family: str
    objective: Objective
    values: FloatArray


def _scalar_poly(values: Any) -> Any:
    """Return a small scalar polynomial-plus-trigonometric objective."""
    return values[0] * values[1] + np.sin(values[2])


def _scalar_mixed(values: Any) -> Any:
    """Return a branchless mixed elementary-function objective."""
    return values[0] ** 2 + np.sin(values[1]) + np.log(values[2] + 4.0) + np.sqrt(values[3] + 3.0)


def _determinant_2x2(values: Any) -> Any:
    """Return the determinant of a static 2x2 matrix."""
    return np.linalg.det(np.reshape(values, (2, 2)))


def _inverse_sum_2x2(values: Any) -> Any:
    """Return the sum of a static 2x2 inverse."""
    return np.sum(np.linalg.inv(np.reshape(values, (2, 2))))


def _solve_sum_2x2(values: Any) -> Any:
    """Return the sum of a static 2x2 linear solve."""
    matrix = np.reshape(values[:4], (2, 2))
    return np.sum(np.linalg.solve(matrix, values[4:6]))


def _trace_3x3(values: Any) -> Any:
    """Return the trace of a static 3x3 matrix."""
    return np.trace(np.reshape(values, (3, 3)))


def decision_kernels() -> tuple[DecisionKernel, ...]:
    """Return the frozen six-kernel comparison set from S38.2."""
    matrix_2 = np.array([5.0, 2.0, 0.0, 4.0], dtype=np.float64)
    matrix_3 = np.array([6.0, 2.0, 0.0, 1.0, 7.0, 2.0, 0.0, 1.0, 5.0], dtype=np.float64)
    return (
        DecisionKernel(
            "scalar_poly_3",
            "scalar",
            _scalar_poly,
            np.array([1.5, -2.0, 0.7], dtype=np.float64),
        ),
        DecisionKernel(
            "scalar_mixed_4",
            "scalar",
            _scalar_mixed,
            np.array([0.7, -0.2, 0.5, 1.0], dtype=np.float64),
        ),
        DecisionKernel("determinant_2x2", "determinant", _determinant_2x2, matrix_2),
        DecisionKernel("inverse_sum_2x2", "inverse", _inverse_sum_2x2, matrix_2),
        DecisionKernel(
            "solve_sum_2x2",
            "solve",
            _solve_sum_2x2,
            np.concatenate((matrix_2, np.array([1.0, 2.0], dtype=np.float64))),
        ),
        DecisionKernel("trace_3x3", "trace", _trace_3x3, matrix_3),
    )


def inventory_matrix() -> tuple[dict[str, object], ...]:
    """Return the source-verified current-role matrix used by S38.1."""
    return (
        {
            "surface": "branchless_scalar",
            "rust_program_ad": "bounded_value_and_gradient_replay",
            "python_mlir_llvmlite": "native_jit_value_jvp_vjp_gradient",
            "rust_jit_product_gap": False,
        },
        {
            "surface": "static_dense_det_inverse_solve_trace",
            "rust_program_ad": "bounded_value_and_gradient_replay",
            "python_mlir_llvmlite": "bounded_native_jit",
            "rust_jit_product_gap": False,
        },
        {
            "surface": "executed_branch_metadata",
            "rust_program_ad": "bounded_replay",
            "python_mlir_llvmlite": "native_jit_fail_closed",
            "rust_jit_product_gap": False,
        },
        {
            "surface": "static_array_reduction_signal_interpolation_stencil_cumulative",
            "rust_program_ad": "bounded_replay",
            "python_mlir_llvmlite": "partial_or_interpreted_fallback",
            "rust_jit_product_gap": False,
        },
        {
            "surface": "dynamic_axes_indexing_and_dynamic_metadata",
            "rust_program_ad": "fail_closed",
            "python_mlir_llvmlite": "fail_closed_or_interpreted",
            "rust_jit_product_gap": False,
        },
    )


def _max_abs_error(left: FloatArray, right: FloatArray) -> float:
    """Return the maximum absolute error between two finite vectors."""
    return float(
        np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))
    )


def _median_runtime_ns(
    callback: Callable[[], object],
    *,
    rounds: int,
    repetitions: int,
    warmups: int,
) -> int:
    """Return median nanoseconds per call for a bounded descriptive timing."""
    for _ in range(warmups):
        callback()
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            callback()
        samples.append((time.perf_counter_ns() - started) / repetitions)
    return int(round(statistics.median(samples)))


def _capture_kernel(
    kernel_spec: DecisionKernel,
    *,
    rounds: int,
    repetitions: int,
    warmups: int,
) -> dict[str, object]:
    """Capture one same-trace Rust-replay versus native-JIT comparison row."""
    traced = whole_program_value_and_grad(kernel_spec.objective, kernel_spec.values)
    if traced.program_ir is None:
        raise ValueError(f"{kernel_spec.case_id} did not emit Program AD IR")

    compile_started = time.perf_counter()
    native = compile_whole_program_ad_trace_to_native_llvm_jit(
        kernel_spec.objective,
        kernel_spec.values,
        None,
    )
    native_compile_seconds = time.perf_counter() - compile_started

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        traced.program_ir,
        kernel_spec.values,
    )
    if not rust_result.supported or rust_result.value is None:
        raise ValueError(
            f"{kernel_spec.case_id} Rust replay blocked: {rust_result.blocked_reasons}"
        )
    native_value, native_gradient = native.value_and_grad(kernel_spec.values)
    rust_value_error = abs(float(rust_result.value) - float(traced.value))
    native_value_error = abs(float(native_value) - float(traced.value))
    rust_gradient_error = _max_abs_error(rust_result.gradient, traced.gradient)
    native_gradient_error = _max_abs_error(native_gradient, traced.gradient)
    if (
        max(
            rust_value_error,
            native_value_error,
            rust_gradient_error,
            native_gradient_error,
        )
        > PARITY_TOLERANCE
    ):
        raise ValueError(f"{kernel_spec.case_id} exceeded parity tolerance")

    rust_callback = lambda: value_and_grad_program_ad_effect_ir_with_rust(  # noqa: E731
        traced.program_ir,
        kernel_spec.values,
    )
    native_callback = lambda: native.value_and_grad(kernel_spec.values)  # noqa: E731
    rust_ns = _median_runtime_ns(
        rust_callback,
        rounds=rounds,
        repetitions=repetitions,
        warmups=warmups,
    )
    native_ns = _median_runtime_ns(
        native_callback,
        rounds=rounds,
        repetitions=repetitions,
        warmups=warmups,
    )
    return {
        "case_id": kernel_spec.case_id,
        "family": kernel_spec.family,
        "parameter_count": int(kernel_spec.values.size),
        "effect_count": len(traced.program_ir.effects),
        "rust_replay_supported": True,
        "native_jit_supported": True,
        "rust_value_error": rust_value_error,
        "native_value_error": native_value_error,
        "rust_gradient_error": rust_gradient_error,
        "native_gradient_error": native_gradient_error,
        "native_compile_seconds": native_compile_seconds,
        "rust_replay_median_ns": rust_ns,
        "native_jit_median_ns": native_ns,
        "rust_replay_to_native_jit_ratio": rust_ns / native_ns,
    }


def _package_version(distribution: str) -> str | None:
    """Return installed package version when the distribution is available."""
    try:
        return version(distribution)
    except Exception:  # pragma: no cover - environment metadata is optional
        return None


def capture_decision_evidence(
    *,
    stamp: str,
    rounds: int = 5,
    repetitions: int = 200,
    warmups: int = 10,
    isolated: bool = False,
) -> dict[str, object]:
    """Capture and return the validated BL-38 decision evidence payload."""
    if not stamp.strip():
        raise ValueError("stamp must be non-empty")
    if rounds < 3 or repetitions < 1 or warmups < 0:
        raise ValueError("rounds must be >=3, repetitions >=1, and warmups >=0")
    clear_native_whole_program_ad_compile_cache()
    rows = [
        _capture_kernel(
            item,
            rounds=rounds,
            repetitions=repetitions,
            warmups=warmups,
        )
        for item in decision_kernels()
    ]
    max_error = max(
        float(cast(float, row[key]))
        for row in rows
        for key in (
            "rust_value_error",
            "native_value_error",
            "rust_gradient_error",
            "native_gradient_error",
        )
    )
    inventory = inventory_matrix()
    product_role_proven = any(bool(row["rust_jit_product_gap"]) for row in inventory)
    decision = "NO-GO"
    rationale = (
        "No user-facing path is blocked on a Rust-owned JIT: existing Rust replay already "
        "covers the bounded parity role, while the pinned Python/llvmlite path already "
        "owns native JIT execution. A second LLVM owner would add compiler and supply-chain "
        "burden without expanding BL-49 certifiable families."
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "artifact_id": f"rust-llvm-jit-decision-{stamp}",
        "stamp": stamp,
        "decision": decision,
        "rationale": rationale,
        "claim_boundary": CLAIM_BOUNDARY,
        "criteria": {
            "parity_passed": max_error <= PARITY_TOLERANCE,
            "parity_tolerance": PARITY_TOLERANCE,
            "max_observed_error": max_error,
            "isolated_performance_evidence": isolated,
            "performance_claim_made": False,
            "product_role_proven": product_role_proven,
            "parity_certificate_family_expansion": False,
            "dual_compiler_maintenance_burden": True,
            "rust_llvm_dependency_present": False,
        },
        "timing_protocol": {
            "rounds": rounds,
            "repetitions_per_round": repetitions,
            "warmups": warmups,
            "isolated": isolated,
            "interpretation": "descriptive_only_no_performance_claim",
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": _package_version("numpy"),
            "llvmlite": _package_version("llvmlite"),
            "scpn_quantum_engine": _package_version("scpn-quantum-engine"),
        },
        "inventory": list(inventory),
        "kernels": rows,
        "follow_up_boundary": (
            "Reopen only if a named user-facing path is blocked on Rust-owned native "
            "execution and a preregistered isolated spike can test >=2x benefit without "
            "weakening parity, rollback, fallback, or BL-49 coverage."
        ),
    }
    validate_decision_evidence(payload)
    digest_source = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(digest_source).hexdigest()
    return payload


def validate_decision_evidence(payload: Mapping[str, object]) -> None:
    """Fail closed when a BL-38 decision payload violates its frozen contract."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected BL-38 decision schema")
    if payload.get("decision") not in {"GO", "NO-GO", "DEFER"}:
        raise ValueError("decision must be GO, NO-GO, or DEFER")
    criteria = payload.get("criteria")
    kernels = payload.get("kernels")
    inventory = payload.get("inventory")
    if not isinstance(criteria, Mapping) or not isinstance(kernels, list):
        raise ValueError("criteria and kernels are required")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("non-empty inventory is required")
    if not kernels or len(kernels) > 10:
        raise ValueError("kernel set must contain 1..10 rows")
    if not bool(criteria.get("parity_passed")):
        raise ValueError("decision evidence requires functional parity")
    if payload.get("decision") == "GO":
        if not bool(criteria.get("product_role_proven")):
            raise ValueError("GO requires a proven product role")
        if not bool(criteria.get("isolated_performance_evidence")):
            raise ValueError("GO requires isolated performance evidence")
    if payload.get("decision") == "NO-GO" and bool(criteria.get("product_role_proven")):
        raise ValueError("NO-GO cannot claim a proven product role")
    case_ids = [row.get("case_id") for row in kernels if isinstance(row, Mapping)]
    if len(case_ids) != len(kernels) or len(set(case_ids)) != len(case_ids):
        raise ValueError("kernel case_ids must be unique non-empty mappings")


def write_decision_evidence(payload: Mapping[str, object], path: Path) -> Path:
    """Validate and atomically write one canonical JSON decision artifact."""
    validate_decision_evidence(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
