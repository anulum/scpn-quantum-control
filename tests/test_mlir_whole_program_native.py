# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Whole-Program Native Tests
"""Integration tests for whole-program AD MLIR and native LLVM lowering."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler.mlir import (
    DifferentiableMLIRCompileConfig,
    ExecutableWholeProgramADBatchResult,
    ExecutableWholeProgramADKernel,
    NativeWholeProgramADKernel,
    WholeProgramADNativeLoweringReport,
    analyse_whole_program_ad_native_lowering,
    clear_native_whole_program_ad_compile_cache,
    compile_whole_program_ad_trace_to_executable,
    compile_whole_program_ad_trace_to_mlir,
    compile_whole_program_ad_trace_to_native_llvm_jit,
    native_whole_program_ad_compile_cache_stats,
    native_whole_program_ad_linalg_support,
)
from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_value_and_grad,
    whole_program_value_and_grad,
)

FloatArray = NDArray[np.float64]


def _dense_determinant_offsets(size: int) -> FloatArray:
    """Return a deterministic non-diagonal perturbation for native determinant tests."""

    rows = np.arange(size, dtype=np.float64).reshape(size, 1) + 1.0
    cols = np.arange(size, dtype=np.float64).reshape(1, size) + 1.0
    offsets = 0.011 * np.sin(rows * (cols + 0.5)) + 0.007 * np.cos(rows + 2.0 * cols)
    np.fill_diagonal(offsets, 0.0)
    return offsets


def _dense_solve_values(size: int, *, shift: float) -> FloatArray:
    """Return deterministic nonsingular matrix and vector entries for solve tests."""

    matrix = np.diag(np.linspace(1.7 + shift, 2.5 + shift, size))
    matrix = matrix + _dense_determinant_offsets(size) * (1.0 + shift)
    rhs = np.linspace(0.25 + shift, 0.95 + shift, size)
    return cast(FloatArray, np.concatenate((matrix.reshape(-1), rhs)).astype(np.float64))


def _dense_solve_matrix_values(size: int, rhs_cols: int, *, shift: float) -> FloatArray:
    """Return deterministic nonsingular matrix and matrix RHS entries for solve tests."""

    matrix = np.diag(np.linspace(1.8 + shift, 2.6 + shift, size))
    matrix = matrix + _dense_determinant_offsets(size) * (1.0 + 0.5 * shift)
    rhs_rows = np.arange(size, dtype=np.float64).reshape(size, 1) + 1.0
    rhs_columns = np.arange(rhs_cols, dtype=np.float64).reshape(1, rhs_cols) + 1.0
    rhs = 0.17 * rhs_rows + 0.09 * rhs_columns + shift
    return cast(
        FloatArray,
        np.concatenate((matrix.reshape(-1), rhs.reshape(-1))).astype(np.float64),
    )


def test_whole_program_ad_mlir_exports_trace_and_polyglot_status() -> None:
    """Whole-program AD trace lowering should be deterministic and honest."""

    def objective(values: FloatArray) -> object:
        if values[0] > 0.0:
            return np.sin(values[0]) + values[1] ** 2
        return values[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, -0.5], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("bias")),
    )
    module = compile_whole_program_ad_trace_to_mlir(result, DifferentiableMLIRCompileConfig())
    repeat = compile_whole_program_ad_trace_to_mlir(result, DifferentiableMLIRCompileConfig())

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.resource_counts["parameters"] == 2
    assert module.resource_counts["trace_events"] == len(result.trace_events)
    assert module.metadata["polyglot_targets"]["llvm"].startswith("blocked")
    assert 'scpn.module = "whole_program_ad"' in module.text
    assert "scpn_diff.trace_event" in module.text
    assert 'execution = "python_whole_program_ad_interchange"' in module.text


def test_whole_program_ad_mlir_lowers_program_ad_effect_ir_metadata() -> None:
    """Whole-program AD MLIR lowering should expose captured Program AD IR records."""

    def objective(values: FloatArray) -> object:
        total = values[0]
        if values[1] > 0.0:
            total = total + np.sin(values[1])
        return total + values[2]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None

    module = compile_whole_program_ad_trace_to_mlir(result, DifferentiableMLIRCompileConfig())

    assert 'scpn.program_ir_format = "program_ad_effect_ir.v1"' in module.text
    assert "scpn_diff.program_ad_ssa" in module.text
    assert "scpn_diff.program_ad_effect" in module.text
    assert "scpn_diff.program_ad_control_region" in module.text
    assert "scpn_diff.program_ad_phi" in module.text
    assert module.resource_counts["program_ad_ssa_values"] == len(result.program_ir.ssa_values)
    assert module.resource_counts["program_ad_effects"] == len(result.program_ir.effects)
    assert module.resource_counts["program_ad_control_regions"] == len(
        result.program_ir.control_regions
    )
    assert module.resource_counts["program_ad_phi_nodes"] == len(result.program_ir.phi_nodes)
    assert module.metadata["program_ad_ir"]["format"] == "program_ad_effect_ir.v1"
    assert module.metadata["program_ad_ir"]["claim_boundary"] == (
        "program_ad_ir_mlir_interchange_only_no_executable_lowering"
    )
    assert module.metadata["polyglot_targets"]["rust"].startswith("blocked")


def test_whole_program_ad_trace_executable_replays_supported_scalar_ir() -> None:
    """Executable program AD trace kernels should replay gradients fail-closed."""

    def objective(values: FloatArray) -> object:
        branch = values[0] if values[0] > values[1] else values[1]
        return np.sin(values[0] * values[1]) + np.log(values[2] + 4.0) + branch * values[2]

    values = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))

    kernel = compile_whole_program_ad_trace_to_executable(objective, values, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        values,
        parameters,
    )
    value, gradient = kernel.value_and_grad(values)

    assert isinstance(kernel, ExecutableWholeProgramADKernel)
    assert kernel.backend == "program_ad_trace_replay"
    assert kernel.parameter_names == ("x", "y", "z")
    assert kernel.parameter_shape == (3,)
    assert kernel.mlir_module.resource_counts["parameters"] == 3
    assert "whole_program_ad" in kernel.mlir_module.text
    assert "branch/signature changes fail closed" in kernel.claim_boundary
    assert kernel.mlir_module.metadata["polyglot_targets"]["llvm"].startswith("blocked")
    assert value == pytest.approx(reference_value)
    assert kernel.value(values) == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.gradient(values),
        reference_gradient,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="branch signature"):
        kernel.value_and_grad(np.array([-0.25, 1.25, 0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        kernel.value_and_grad(np.array([[1.25, -0.25, 0.5]], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        kernel.value_and_grad(np.array([1.25, np.nan, 0.5], dtype=np.float64))


def test_whole_program_ad_trace_executable_batches_same_branch_rows() -> None:
    """Executable program AD trace kernels should batch same-signature rows."""

    def objective(values: FloatArray) -> object:
        branch = values[0] if values[0] > values[1] else values[1]
        return np.sin(values[0] * values[1]) + np.log(values[2] + 4.0) + branch * values[2]

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    batch = np.array(
        [
            [1.25, -0.25, 0.5],
            [1.1, -0.4, 0.75],
            [2.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))
    kernel = compile_whole_program_ad_trace_to_executable(objective, sample, parameters)

    result = kernel.batch_value_and_grad(batch)
    reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]

    assert isinstance(result, ExecutableWholeProgramADBatchResult)
    assert result.backend == "program_ad_trace_replay"
    assert result.parameter_names == ("x", "y", "z")
    assert result.mlir_sha256 == kernel.mlir_module.sha256
    assert result.row_signatures == (kernel.branch_signature,) * batch.shape[0]
    np.testing.assert_allclose(
        result.values,
        np.array([item[0] for item in reference], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.gradients,
        np.vstack([item[1] for item in reference]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(kernel.batch_value(batch), result.values)
    np.testing.assert_allclose(kernel.batch_gradient(batch), result.gradients)

    branch_drift_batch = batch.copy()
    branch_drift_batch[1] = np.array([-0.25, 1.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_value_and_grad(branch_drift_batch)
    with pytest.raises(ValueError, match="two-dimensional"):
        kernel.batch_value_and_grad(sample)
    with pytest.raises(ValueError, match="at least one row"):
        kernel.batch_value_and_grad(np.empty((0, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="parameter shape"):
        kernel.batch_value_and_grad(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        kernel.batch_value_and_grad(
            np.array([[1.25, -0.25, 0.5], [1.0, np.nan, 0.5]], dtype=np.float64)
        )


def test_whole_program_ad_trace_native_llvm_jit_executes_branchless_scalar_ir() -> None:
    """Native whole-program AD lowering should execute supported branchless traces."""

    def objective(values: FloatArray) -> object:
        return (
            np.sin(values[0] * values[1])
            + np.log(values[2] + 4.0)
            + values[0] ** 2
            - values[1] / 2.0
        )

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    batch = np.array(
        [
            [1.25, -0.25, 0.5],
            [1.1, -0.4, 0.75],
            [2.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    tangent = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))

    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        sample,
        parameters,
    )
    value, gradient = kernel.value_and_grad(sample)
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)

    assert isinstance(kernel, NativeWholeProgramADKernel)
    assert kernel.backend == "native_llvm_jit"
    assert callable(kernel.native_functions["batch_value_gradient"])
    assert callable(kernel.native_functions["batch_jvp"])
    assert callable(kernel.native_functions["batch_vjp"])
    assert kernel.verification.passed
    assert kernel.verification.gradient_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.mlir_module.resource_counts["native_whole_program_kernels"] == 1
    assert kernel.mlir_module.resource_counts["native_whole_program_batch_kernels"] == 1
    assert kernel.mlir_module.resource_counts["native_whole_program_batch_transform_kernels"] == 2
    assert kernel.mlir_module.resource_counts["native_lowerable_ops"] == len(
        kernel.lowering_report.lowerable_ops
    )
    assert kernel.mlir_module.resource_counts["native_unsupported_ops"] == 0
    assert kernel.mlir_module.metadata["native_backend"] == "native_llvm_jit"
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert isinstance(kernel.lowering_report, WholeProgramADNativeLoweringReport)
    assert kernel.lowering_report.supported is True
    assert kernel.lowering_report.unsupported_ops == ()
    assert kernel.lowering_report.operation_count == len(kernel.source_result.ir_nodes)
    assert set(kernel.lowering_report.lowerable_ops) >= {
        "parameter",
        "mul",
        "sin",
        "log",
        "pow",
        "sub",
    }
    assert kernel.mlir_module.metadata["polyglot_targets"]["llvm"].startswith("available")
    assert "scpn_diff.native_llvm_jit" in kernel.mlir_module.text
    assert "_batch_value_gradient" in kernel.llvm_ir
    assert "_batch_jvp" in kernel.llvm_ir
    assert "_batch_vjp" in kernel.llvm_ir
    assert "native LLVM/JIT execution" in kernel.claim_boundary
    assert value == pytest.approx(reference_value)
    assert kernel.value(sample) == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        kernel.gradient(sample),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert kernel.jvp(sample, tangent) == pytest.approx(float(np.dot(reference_gradient, tangent)))
    np.testing.assert_allclose(
        kernel.vjp(sample, np.array([2.0], dtype=np.float64)),
        2.0 * reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert "compiled batched native LLVM/JIT" in batch_result.claim_boundary
    batch_tangents = np.vstack([tangent, 0.5 * tangent, -tangent])
    batch_cotangents = np.array([1.0, 2.0, -0.5], dtype=np.float64)
    np.testing.assert_allclose(
        kernel.batch_jvp(batch, batch_tangents),
        np.array(
            [float(np.dot(item[1], row)) for item, row in zip(batch_reference, batch_tangents)],
            dtype=np.float64,
        ),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        kernel.batch_vjp(batch, batch_cotangents),
        np.vstack([scale * item[1] for scale, item in zip(batch_cotangents, batch_reference)]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(kernel.batch_value(batch), batch_result.values)
    np.testing.assert_allclose(kernel.batch_gradient(batch), batch_result.gradients)

    with pytest.raises(ValueError, match="tangent shape"):
        kernel.jvp(sample, np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError, match="cotangent"):
        kernel.vjp(sample, np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError, match="two-dimensional"):
        kernel.batch_value_and_grad(sample)
    with pytest.raises(ValueError, match="parameter shape"):
        kernel.batch_value_and_grad(np.ones((2, 2), dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_reuses_verified_cache() -> None:
    """Native program AD should reuse verified compile artefacts deterministically."""

    clear_native_whole_program_ad_compile_cache()
    assert native_whole_program_ad_compile_cache_stats()["entries"] == 0

    def objective(values: FloatArray) -> object:
        return np.log1p(values[0] * values[0]) + np.tanh(values[1]) + values[2] ** 3

    sample = np.array([0.125, -0.375, 0.75], dtype=np.float64)
    replay = np.array([0.2, -0.25, 0.5], dtype=np.float64)
    parameters = (Parameter("cache_x"), Parameter("cache_y"), Parameter("cache_z"))

    first = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    first_stats = native_whole_program_ad_compile_cache_stats()
    second = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    second_stats = native_whole_program_ad_compile_cache_stats()
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert first.mlir_module.metadata["native_compile_cache_key"] == first.cache_key
    assert second.mlir_module.metadata["native_compile_cache_key"] == second.cache_key
    assert first.mlir_module.metadata["native_compile_cache_hit"] is False
    assert second.mlir_module.metadata["native_compile_cache_hit"] is True
    assert first_stats["entries"] == 1
    assert cast(int, first_stats["max_size"]) >= 1
    assert first_stats["keys"] == (first.cache_key,)
    assert second_stats["entries"] == 1
    assert second_stats["keys"] == (first.cache_key,)
    assert first.mlir_module.resource_counts["native_compile_cache_hit"] == 0
    assert second.mlir_module.resource_counts["native_compile_cache_hit"] == 1
    assert first.native_functions["engine"] is second.native_functions["engine"]
    assert first.native_functions["value"] is second.native_functions["value"]
    assert first.native_functions["batch_jvp"] is second.native_functions["batch_jvp"]

    value, gradient = second.value_and_grad(replay)
    assert value == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-10, atol=1.0e-10)

    assert clear_native_whole_program_ad_compile_cache() == 1
    assert native_whole_program_ad_compile_cache_stats()["entries"] == 0
    third = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    assert third.cache_hit is False
    assert third.cache_key == first.cache_key
    assert third.native_functions["engine"] is not first.native_functions["engine"]
    assert clear_native_whole_program_ad_compile_cache() == 1


def test_whole_program_ad_native_lowering_report_blocks_unsupported_ops() -> None:
    """Native program AD lowering should report replay-supported ops that lack LLVM lowering."""

    def objective(values: FloatArray) -> object:
        matrix = np.diag(values[:20])
        return np.linalg.det(matrix) + np.sin(values[20])

    sample = np.linspace(1.1, 1.7, 21, dtype=np.float64)
    parameters = tuple(Parameter(f"x{index}") for index in range(21))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert isinstance(report, WholeProgramADNativeLoweringReport)
    assert report.supported is False
    assert report.unsupported_ops == ("linalg:det:20x20",)
    assert report.unsupported_operation_count == 1
    assert report.lowerable_operation_count == len(result.ir_nodes) - 1
    assert "unsupported native ops: linalg:det:20x20" in report.fail_closed_reason
    assert report.as_metadata()["unsupported_ops"] == report.unsupported_ops

    with pytest.raises(ValueError, match="unsupported native ops: linalg:det:20x20"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_whole_program_ad_trace_native_llvm_jit_lowers_wide_determinants() -> None:
    """Native program AD should lower helper-backed 6x6 through 19x19 determinants."""

    for size in range(6, 20):

        def objective(
            values: FloatArray,
            *,
            matrix_size: int = size,
        ) -> object:
            matrix = np.diag(values[:matrix_size])
            return (
                np.linalg.det(matrix)
                + 0.01 * values[matrix_size] * values[0]
                - np.sin(values[matrix_size - 1])
            )

        sample = np.linspace(1.1, 1.6, size + 1, dtype=np.float64)
        replay = np.linspace(1.7, 1.2, size + 1, dtype=np.float64)
        parameters = tuple(Parameter(f"x{index}") for index in range(size + 1))

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )
        det_op = f"linalg:det:{size}x{size}"

        assert report.supported is True
        assert report.unsupported_ops == ()
        assert det_op in report.lowerable_ops
        assert det_op in kernel.supported_ops
        assert f"scpn_det{size}_fl_value_partials" in kernel.llvm_ir
        assert f"%det{size}_helper_matrix_" in kernel.llvm_ir
        assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
        assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        batch = np.vstack(
            [
                sample,
                replay,
                np.linspace(1.25, 1.75, size + 1, dtype=np.float64),
            ]
        )
        batch_reference = [
            program_adjoint_value_and_grad(objective, row, parameters) for row in batch
        ]
        batch_result = kernel.batch_value_and_grad(batch)
        np.testing.assert_allclose(
            batch_result.values,
            np.array([item[0] for item in batch_reference], dtype=np.float64),
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            batch_result.gradients,
            np.vstack([item[1] for item in batch_reference]),
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_native_linalg_support_contract_reports_dense_det_boundary() -> None:
    """Native linalg support contracts should expose exact fail-closed determinant limits."""

    support = native_whole_program_ad_linalg_support()

    assert scpn.native_whole_program_ad_linalg_support is native_whole_program_ad_linalg_support
    assert compiler_mlir.native_whole_program_ad_linalg_support is (
        native_whole_program_ad_linalg_support
    )
    assert support["determinant_expression_sizes"] == (2, 3, 4, 5)
    assert support["determinant_helper_sizes"] == tuple(range(6, 20))
    assert support["determinant_static_dense_sizes"] == tuple(range(2, 20))
    assert support["determinant_fail_closed_from"] == 20
    assert support["determinant_derivative"] == "exact_forward_partials"
    assert support["determinant_policy"] == "static_dense_native_or_fail_closed"
    assert support["inverse_sizes"] == tuple(range(2, 8))
    assert support["inverse_fail_closed_from"] == 8
    assert support["solve_sizes"] == tuple(range(2, 8))
    assert support["solve_matrix_sizes"] == tuple(range(2, 8))
    assert support["solve_matrix_max_rhs_columns"] == 4
    assert support["solve_rhs_policy"] == "static_vector_or_matrix_rhs"
    assert support["solve_fail_closed_from"] == 8
    assert support["quotient_linalg_helper_sizes"] == (5, 6, 7)
    assert support["quotient_linalg_factorisation_sizes"] == (5, 6, 7)
    assert support["quotient_linalg_reuse_policy"] == "shared_factorisation_per_static_matrix"
    assert support["quotient_linalg_unsuitable_from"] == 8
    assert support["quotient_linalg_unsuitable_reason"] == (
        "full_output_inverse_and_matrix_rhs_solve_require_larger_factorisation_helper"
    )
    assert support["unsupported_policy"] == "fail_closed_report_before_compile"


def test_whole_program_ad_trace_native_llvm_jit_lowers_dense_wide_determinants() -> None:
    """Native wide determinant helpers should match replay AD on non-diagonal matrices."""

    for size in (7, 9, 11, 13, 15, 16, 17, 18, 19):
        offsets = _dense_determinant_offsets(size)

        def objective(
            values: FloatArray,
            *,
            matrix_size: int = size,
            dense_offsets: FloatArray = offsets,
        ) -> object:
            matrix = np.diag(values[:matrix_size]) + dense_offsets
            return np.linalg.det(matrix) + 0.005 * values[0] * values[matrix_size - 1]

        sample = np.linspace(1.25, 1.75, size, dtype=np.float64)
        replay = np.linspace(1.8, 1.3, size, dtype=np.float64)
        parameters = tuple(Parameter(f"dense_x{index}") for index in range(size))
        det_op = f"linalg:det:{size}x{size}"

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )

        assert report.supported is True
        assert det_op in report.lowerable_ops
        assert report.unsupported_ops == ()
        assert f"scpn_det{size}_fl_value_partials" in kernel.llvm_ir
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_trace_native_llvm_jit_lowers_5x5_determinant() -> None:
    """Native program AD should lower scalar 5x5 determinant nodes."""

    def objective(values: FloatArray) -> object:
        matrix = values[0:25].reshape((5, 5))
        return np.linalg.det(matrix) + 0.0625 * values[25] * values[0] - np.cos(values[24])

    sample = np.array(
        [
            1.2,
            0.1,
            -0.2,
            0.3,
            0.0,
            0.2,
            1.4,
            0.5,
            -0.1,
            0.3,
            0.2,
            -0.3,
            1.1,
            0.4,
            -0.2,
            -0.1,
            0.2,
            0.3,
            1.3,
            0.4,
            0.1,
            -0.2,
            0.5,
            -0.3,
            1.5,
            0.75,
        ],
        dtype=np.float64,
    )
    replay = np.array(
        [
            1.5,
            -0.2,
            0.4,
            -0.1,
            0.2,
            0.3,
            1.1,
            -0.5,
            0.2,
            -0.4,
            -0.4,
            0.6,
            1.6,
            -0.3,
            0.5,
            0.2,
            -0.1,
            0.5,
            1.25,
            -0.2,
            0.4,
            0.1,
            -0.3,
            0.6,
            1.4,
            -0.4,
        ],
        dtype=np.float64,
    )
    parameters = tuple(Parameter(f"a{index}") for index in range(25)) + (Parameter("scale"),)

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:5x5" in report.lowerable_ops
    assert "linalg:det:5x5" in kernel.supported_ops
    assert "det5_value" in kernel.llvm_ir
    assert "det5_cofactor_44" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [
                1.1,
                0.2,
                -0.3,
                0.4,
                -0.2,
                -0.2,
                1.3,
                0.1,
                -0.5,
                0.3,
                0.3,
                -0.4,
                1.7,
                0.2,
                -0.1,
                0.5,
                -0.1,
                0.2,
                1.2,
                0.4,
                -0.3,
                0.6,
                0.1,
                -0.2,
                1.6,
                0.25,
            ],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_4x4_determinant() -> None:
    """Native program AD should lower scalar 4x4 determinant nodes."""

    def objective(values: FloatArray) -> object:
        matrix = values[0:16].reshape((4, 4))
        return np.linalg.det(matrix) + 0.125 * values[16] * values[0] - np.sin(values[15])

    sample = np.array(
        [
            1.2,
            0.1,
            -0.2,
            0.3,
            0.0,
            1.4,
            0.5,
            -0.1,
            0.2,
            -0.3,
            1.1,
            0.4,
            -0.1,
            0.2,
            0.3,
            1.3,
            0.75,
        ],
        dtype=np.float64,
    )
    replay = np.array(
        [
            1.5,
            -0.2,
            0.4,
            -0.1,
            0.3,
            1.1,
            -0.5,
            0.2,
            -0.4,
            0.6,
            1.6,
            -0.3,
            0.2,
            -0.1,
            0.5,
            1.25,
            -0.4,
        ],
        dtype=np.float64,
    )
    parameters = tuple(Parameter(f"a{index}") for index in range(16)) + (Parameter("scale"),)

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:4x4" in report.lowerable_ops
    assert "linalg:det:4x4" in kernel.supported_ops
    assert "det4_value" in kernel.llvm_ir
    assert "det4_cofactor_33" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [
                1.1,
                0.2,
                -0.3,
                0.4,
                -0.2,
                1.3,
                0.1,
                -0.5,
                0.3,
                -0.4,
                1.7,
                0.2,
                0.5,
                -0.1,
                0.2,
                1.2,
                0.25,
            ],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_3x3_determinant() -> None:
    """Native program AD should lower scalar 3x3 determinant nodes."""

    def objective(values: FloatArray) -> object:
        matrix = values[0:9].reshape((3, 3))
        return np.linalg.det(matrix) + 0.25 * values[9] * values[0] - np.cos(values[8])

    sample = np.array(
        [1.2, 0.1, -0.2, 0.3, 1.5, 0.4, -0.1, 0.2, 1.1, 0.75],
        dtype=np.float64,
    )
    replay = np.array(
        [1.0, -0.3, 0.4, 0.2, 1.25, -0.5, 0.6, 0.1, 1.4, -0.2],
        dtype=np.float64,
    )
    parameters = (
        Parameter("a00"),
        Parameter("a01"),
        Parameter("a02"),
        Parameter("a10"),
        Parameter("a11"),
        Parameter("a12"),
        Parameter("a20"),
        Parameter("a21"),
        Parameter("a22"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:3x3" in report.lowerable_ops
    assert "linalg:det:3x3" in kernel.supported_ops
    assert "det3_cofactor_00" in kernel.llvm_ir
    assert "det3_cofactor_22" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [1.4, 0.2, -0.1, -0.3, 1.1, 0.5, 0.4, -0.2, 1.6, 0.3],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_2x2_inverse_and_solve() -> None:
    """Native program AD should lower scalar 2x2 inverse and solve nodes."""

    def objective(values: FloatArray) -> object:
        matrix = values[0:4].reshape((2, 2))
        rhs = values[4:6]
        return (
            np.linalg.inv(matrix).sum()
            + 0.5 * np.linalg.solve(matrix, rhs).sum()
            + values[6] * values[0]
            - np.cos(values[5])
        )

    sample = np.array([1.0, 0.2, 0.3, 1.5, 0.4, -0.2, 0.75], dtype=np.float64)
    replay = np.array([1.5, -0.4, 0.6, 2.0, -0.25, 0.7, -0.5], dtype=np.float64)
    singular = np.array([1.0, 2.0, 0.5, 1.0, 0.4, -0.2, 0.75], dtype=np.float64)
    parameters = (
        Parameter("a00"),
        Parameter("a01"),
        Parameter("a10"),
        Parameter("a11"),
        Parameter("rhs0"),
        Parameter("rhs1"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {
        "linalg:inv:2x2:0:0",
        "linalg:inv:2x2:0:1",
        "linalg:inv:2x2:1:0",
        "linalg:inv:2x2:1:1",
        "linalg:solve:2x2:rhs:2:0",
        "linalg:solve:2x2:rhs:2:1",
    }.issubset(report.lowerable_ops)
    assert "inv2_det" in kernel.llvm_ir
    assert "solve2_det" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [2.0, 0.1, -0.2, 1.25, 0.5, -0.35, 0.2],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        kernel.gradient(singular)


def test_whole_program_ad_trace_native_llvm_jit_lowers_2x2_product_linalg_ops() -> None:
    """Native program AD should lower scalar 2x2 matrix_power and multi_dot nodes."""

    def objective(values: FloatArray) -> object:
        left = values[0:4].reshape((2, 2))
        right = values[4:8].reshape((2, 2))
        return (
            np.linalg.matrix_power(left, 2).sum()
            + 0.5 * np.linalg.multi_dot((left, right)).sum()
            + values[8] * values[0]
            - np.sin(values[7])
        )

    sample = np.array([1.0, 0.2, 0.3, 1.5, 0.4, -0.2, 0.6, 0.9, 0.75], dtype=np.float64)
    replay = np.array([1.5, -0.4, 0.6, 2.0, -0.25, 0.7, -0.5, 1.25, -0.2], dtype=np.float64)
    parameters = (
        Parameter("left00"),
        Parameter("left01"),
        Parameter("left10"),
        Parameter("left11"),
        Parameter("right00"),
        Parameter("right01"),
        Parameter("right10"),
        Parameter("right11"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {
        "linalg:matrix_power:2x2:power:2:0:0",
        "linalg:matrix_power:2x2:power:2:0:1",
        "linalg:matrix_power:2x2:power:2:1:0",
        "linalg:matrix_power:2x2:power:2:1:1",
        "linalg:multi_dot:2x2__2x2:out:2x2:0",
        "linalg:multi_dot:2x2__2x2:out:2x2:1",
        "linalg:multi_dot:2x2__2x2:out:2x2:2",
        "linalg:multi_dot:2x2__2x2:out:2x2:3",
    }.issubset(report.lowerable_ops)
    assert "matrix_power2_first" in kernel.llvm_ir
    assert "multi_dot2_first" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [2.0, 0.1, -0.2, 1.25, 0.5, -0.35, 0.2, 1.1, 0.4],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_native_lowering_blocks_unverified_static_linalg_backends() -> None:
    """Wider static linalg kernels must not promote to native or Rust by metadata alone."""

    def objective(values: FloatArray) -> object:
        matrix = values[0:9].reshape((3, 3))
        left = values[9:15].reshape((2, 3))
        middle = values[15:27].reshape((3, 4))
        right = values[27:35].reshape((4, 2))
        return (
            np.linalg.matrix_power(matrix, 3).sum()
            + np.linalg.multi_dot((left, middle, right)).sum()
        )

    sample = np.array(
        [
            1.5,
            -0.25,
            0.5,
            0.0,
            2.0,
            0.75,
            0.25,
            -0.5,
            1.25,
            1.0,
            -0.5,
            0.75,
            0.25,
            1.5,
            -1.0,
            1.25,
            -0.5,
            0.0,
            0.5,
            0.75,
            1.0,
            -0.25,
            0.25,
            0.0,
            0.5,
            1.5,
            -0.75,
            1.0,
            -0.25,
            0.5,
            1.25,
            -0.75,
            0.0,
            0.25,
            0.5,
        ],
        dtype=np.float64,
    )
    parameters = tuple(Parameter(f"theta_{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert report.supported is False
    assert any(op.startswith("linalg:matrix_power:3x3:power:3:") for op in report.unsupported_ops)
    assert any(
        op.startswith("linalg:multi_dot:2x3__3x4__4x2:out:2x2:") for op in report.unsupported_ops
    )
    assert "native LLVM/JIT and Rust static linalg lowering blocked" in report.fail_closed_reason
    assert "verified executable kernels" in report.fail_closed_reason
    with pytest.raises(ValueError, match="static linalg lowering blocked.*verified executable"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_whole_program_ad_trace_native_llvm_jit_lowers_2x2_linalg_scalar_ops() -> None:
    """Native program AD should lower scalar 2x2 det and trace linalg nodes."""

    def objective(values: FloatArray) -> object:
        matrix = values[0:4].reshape((2, 2))
        return (
            np.linalg.det(matrix)
            + 0.25 * np.trace(matrix)
            + values[4] * values[0]
            - np.sin(values[3])
        )

    sample = np.array([1.0, 0.2, 0.3, 1.5, 0.75], dtype=np.float64)
    replay = np.array([1.5, -0.4, 0.6, 2.0, -0.25], dtype=np.float64)
    parameters = (
        Parameter("a00"),
        Parameter("a01"),
        Parameter("a10"),
        Parameter("a11"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:2x2" in report.lowerable_ops
    assert "linalg:trace:2x2:offset:0" in report.lowerable_ops
    assert "linalg:det:2x2" in kernel.supported_ops
    assert "linalg:trace:2x2:offset:0" in kernel.supported_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "det2_diag" in kernel.llvm_ir
    assert "det2_offdiag" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [2.0, 0.1, -0.2, 1.25, 0.5],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_inverse_ops() -> None:
    """Native program AD should lower bounded static dense inverse nodes."""

    for size in (3, 4, 5, 6):
        weights = np.linspace(0.15, 0.85, size * size, dtype=np.float64).reshape(size, size)

        def objective(
            values: FloatArray,
            *,
            matrix_size: int = size,
            inverse_weights: FloatArray = weights,
        ) -> object:
            matrix = values[: matrix_size * matrix_size].reshape((matrix_size, matrix_size))
            inverse = np.linalg.inv(matrix)
            weighted_inverse = sum(
                inverse_weights[row, col] * inverse[row, col]
                for row in range(matrix_size)
                for col in range(matrix_size)
            )
            return weighted_inverse + 0.015 * values[0] * values[-1] - np.cos(inverse[-1, 0])

        sample = _dense_solve_values(size, shift=0.0)[: size * size]
        replay = _dense_solve_values(size, shift=0.35)[: size * size]
        parameters = tuple(Parameter(f"inv{size}_x{index}") for index in range(sample.size))

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )
        expected_ops = {
            f"linalg:inv:{size}x{size}:{row}:{col}" for row in range(size) for col in range(size)
        }

        assert report.supported is True
        assert report.unsupported_ops == ()
        assert expected_ops.issubset(report.lowerable_ops)
        assert expected_ops.issubset(kernel.supported_ops)
        assert f"inv{size}_" in kernel.llvm_ir
        if size >= 5:
            assert f"inv{size}_shared_" in kernel.llvm_ir
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        batch = np.vstack(
            [
                sample,
                replay,
                _dense_solve_values(size, shift=0.15)[: size * size],
            ]
        )
        batch_reference = [
            program_adjoint_value_and_grad(objective, row, parameters) for row in batch
        ]
        batch_result = kernel.batch_value_and_grad(batch)
        np.testing.assert_allclose(
            batch_result.values,
            np.array([item[0] for item in batch_reference], dtype=np.float64),
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            batch_result.gradients,
            np.vstack([item[1] for item in batch_reference]),
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_native_lowering_report_blocks_wider_inverse_ops() -> None:
    """Native program AD inverse support should fail closed beyond the bounded helper range."""

    def objective(values: FloatArray) -> object:
        matrix = values[:64].reshape((8, 8))
        return np.linalg.inv(matrix).sum()

    sample = _dense_solve_values(8, shift=0.0)[:64]
    parameters = tuple(Parameter(f"inv8_x{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert report.supported is False
    assert "linalg:inv:8x8:0:0" in report.unsupported_ops
    assert "unsupported native ops: linalg:inv:8x8:0:0" in report.fail_closed_reason
    with pytest.raises(ValueError, match="unsupported native ops: linalg:inv:8x8:0:0"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_solve_vector_ops() -> None:
    """Native program AD should lower bounded static dense vector solve nodes."""

    for size in (3, 4, 5, 6):
        weights = np.linspace(0.2, 0.8, size, dtype=np.float64)

        def objective(
            values: FloatArray,
            *,
            matrix_size: int = size,
            solution_weights: FloatArray = weights,
        ) -> object:
            matrix = values[: matrix_size * matrix_size].reshape((matrix_size, matrix_size))
            rhs = values[matrix_size * matrix_size : matrix_size * matrix_size + matrix_size]
            solution = np.linalg.solve(matrix, rhs)
            weighted_solution = sum(
                solution_weights[index] * solution[index] for index in range(matrix_size)
            )
            return weighted_solution + 0.015 * values[0] * values[-1] - np.sin(solution[-1])

        sample = _dense_solve_values(size, shift=0.0)
        replay = _dense_solve_values(size, shift=0.35)
        parameters = tuple(Parameter(f"solve{size}_x{index}") for index in range(sample.size))

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )
        expected_ops = {f"linalg:solve:{size}x{size}:rhs:{size}:{row}" for row in range(size)}

        assert report.supported is True
        assert report.unsupported_ops == ()
        assert expected_ops.issubset(report.lowerable_ops)
        assert expected_ops.issubset(kernel.supported_ops)
        assert f"solve{size}_" in kernel.llvm_ir
        if size >= 5:
            assert f"solve{size}_shared_" in kernel.llvm_ir
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        batch = np.vstack(
            [
                sample,
                replay,
                _dense_solve_values(size, shift=0.15),
            ]
        )
        batch_reference = [
            program_adjoint_value_and_grad(objective, row, parameters) for row in batch
        ]
        batch_result = kernel.batch_value_and_grad(batch)
        np.testing.assert_allclose(
            batch_result.values,
            np.array([item[0] for item in batch_reference], dtype=np.float64),
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            batch_result.gradients,
            np.vstack([item[1] for item in batch_reference]),
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_solve_matrix_ops() -> None:
    """Native program AD should lower bounded static dense matrix-RHS solve nodes."""

    for size, rhs_cols in ((2, 2), (3, 2), (4, 3), (5, 2), (6, 2)):
        weights = np.linspace(0.12, 0.72, size * rhs_cols, dtype=np.float64).reshape(
            size,
            rhs_cols,
        )

        def objective(
            values: FloatArray,
            *,
            matrix_size: int = size,
            column_count: int = rhs_cols,
            solution_weights: FloatArray = weights,
        ) -> object:
            matrix_end = matrix_size * matrix_size
            matrix = values[:matrix_end].reshape((matrix_size, matrix_size))
            rhs = values[matrix_end : matrix_end + matrix_size * column_count].reshape(
                (matrix_size, column_count)
            )
            solution = np.linalg.solve(matrix, rhs)
            weighted_solution = sum(
                solution_weights[row, col] * solution[row, col]
                for row in range(matrix_size)
                for col in range(column_count)
            )
            return weighted_solution + 0.015 * values[0] * values[-1] - np.sin(solution[-1, -1])

        sample = _dense_solve_matrix_values(size, rhs_cols, shift=0.0)
        replay = _dense_solve_matrix_values(size, rhs_cols, shift=0.3)
        parameters = tuple(
            Parameter(f"solve{size}m{rhs_cols}_x{index}") for index in range(sample.size)
        )

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )
        expected_ops = {
            f"linalg:solve:{size}x{size}:rhs:{size}x{rhs_cols}:{row}:{col}"
            for row in range(size)
            for col in range(rhs_cols)
        }

        assert report.supported is True
        assert report.unsupported_ops == ()
        assert expected_ops.issubset(report.lowerable_ops)
        assert expected_ops.issubset(kernel.supported_ops)
        assert f"solve{size}m{rhs_cols}_" in kernel.llvm_ir
        if size >= 5:
            assert f"solve{size}m{rhs_cols}_shared_" in kernel.llvm_ir
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        batch = np.vstack(
            [
                sample,
                replay,
                _dense_solve_matrix_values(size, rhs_cols, shift=0.15),
            ]
        )
        batch_reference = [
            program_adjoint_value_and_grad(objective, row, parameters) for row in batch
        ]
        batch_result = kernel.batch_value_and_grad(batch)
        np.testing.assert_allclose(
            batch_result.values,
            np.array([item[0] for item in batch_reference], dtype=np.float64),
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            batch_result.gradients,
            np.vstack([item[1] for item in batch_reference]),
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_native_lowering_report_blocks_wider_solve_matrix_ops() -> None:
    """Native program AD matrix-RHS solve should fail closed beyond the bounded range."""

    def objective(values: FloatArray) -> object:
        matrix = values[:64].reshape((8, 8))
        rhs = values[64:80].reshape((8, 2))
        return np.linalg.solve(matrix, rhs).sum()

    sample = _dense_solve_matrix_values(8, 2, shift=0.0)
    parameters = tuple(Parameter(f"solve8m2_x{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert report.supported is False
    assert "linalg:solve:8x8:rhs:8x2:0:0" in report.unsupported_ops
    assert "unsupported native ops: linalg:solve:8x8:rhs:8x2:0:0" in report.fail_closed_reason
    with pytest.raises(ValueError, match="unsupported native ops: linalg:solve:8x8:rhs:8x2:0:0"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_whole_program_ad_native_lowering_report_blocks_wider_solve_ops() -> None:
    """Native program AD solve support should fail closed beyond the bounded helper range."""

    def objective(values: FloatArray) -> object:
        matrix = values[:64].reshape((8, 8))
        rhs = values[64:72]
        return np.linalg.solve(matrix, rhs).sum()

    sample = _dense_solve_values(8, shift=0.0)
    parameters = tuple(Parameter(f"solve8_x{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert report.supported is False
    assert "linalg:solve:8x8:rhs:8:0" in report.unsupported_ops
    assert "unsupported native ops: linalg:solve:8x8:rhs:8:0" in report.fail_closed_reason
    with pytest.raises(ValueError, match="unsupported native ops: linalg:solve:8x8:rhs:8:0"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_trace_ops() -> None:
    """Native program AD should lower static square and rectangular trace nodes."""

    def objective(values: FloatArray) -> object:
        rectangle = values[0:20].reshape((4, 5))
        square = values[20:45].reshape((5, 5))
        return (
            np.trace(rectangle, offset=1)
            + 0.5 * np.trace(rectangle, offset=-1)
            + 0.25 * np.trace(square)
            + values[45] * values[0]
            - np.cos(values[44])
        )

    sample = np.linspace(-0.6, 0.9, 46, dtype=np.float64)
    replay = np.linspace(0.8, -0.7, 46, dtype=np.float64)
    parameters = tuple(Parameter(f"x{index}") for index in range(46))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {
        "linalg:trace:4x5:offset:1",
        "linalg:trace:4x5:offset:-1",
        "linalg:trace:5x5:offset:0",
    }.issubset(report.lowerable_ops)
    assert {
        "linalg:trace:4x5:offset:1",
        "linalg:trace:4x5:offset:-1",
        "linalg:trace:5x5:offset:0",
    }.issubset(kernel.supported_ops)
    assert "trace_" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.vstack(
        [
            sample,
            replay,
            np.linspace(-0.25, 1.25, 46, dtype=np.float64),
        ]
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_diagonal_ops() -> None:
    """Native program AD should lower static diagonal gather/scatter nodes."""

    def objective(values: FloatArray) -> object:
        rectangle = values[0:12].reshape((3, 4))
        diagonal = values[12:16]
        flattened = values[16:20]
        return (
            np.diag(rectangle, k=1).sum()
            + 0.5 * np.diag(diagonal, k=-1).sum()
            + 0.25 * np.diagflat(flattened, k=2).sum()
            + values[20] * values[0]
            - np.sin(values[19])
        )

    sample = np.linspace(-0.8, 0.6, 21, dtype=np.float64)
    replay = np.linspace(0.7, -0.9, 21, dtype=np.float64)
    parameters = tuple(Parameter(f"x{index}") for index in range(21))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    expected_ops = {
        "linalg:diag:3x4:offset:1:extract:0",
        "linalg:diag:3x4:offset:1:extract:1",
        "linalg:diag:3x4:offset:1:extract:2",
        "linalg:diag:4:offset:-1:construct:0",
        "linalg:diag:4:offset:-1:construct:1",
        "linalg:diag:4:offset:-1:construct:2",
        "linalg:diag:4:offset:-1:construct:3",
        "linalg:diagflat:4:offset:2:construct:0",
        "linalg:diagflat:4:offset:2:construct:1",
        "linalg:diagflat:4:offset:2:construct:2",
        "linalg:diagflat:4:offset:2:construct:3",
    }
    assert report.supported is True
    assert report.unsupported_ops == ()
    assert expected_ops.issubset(report.lowerable_ops)
    assert expected_ops.issubset(kernel.supported_ops)
    assert "diag_" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.vstack(
        [
            sample,
            replay,
            np.linspace(-0.25, 1.25, 21, dtype=np.float64),
        ]
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_scalar_where() -> None:
    """Native program AD should lower scalar np.where selection traces."""

    def objective(values: FloatArray) -> object:
        selected = np.where(values[0] > values[1], values[0:1] ** 2, values[1:2] ** 2)
        return selected.sum() + values[2] * values[0]

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    replay = np.array([-0.5, 1.0, 2.0], dtype=np.float64)
    parameters = (Parameter("left"), Parameter("right"), Parameter("scale"))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "where" in report.lowerable_ops
    assert "where" in kernel.supported_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "select i1" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            [1.25, -0.25, 0.5],
            [-0.5, 1.0, 2.0],
            [2.0, -1.0, -0.25],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises(ValueError, match="non-differentiable at equality"):
        kernel.gradient(np.array([1.0, 1.0, 0.5], dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_lowers_scalar_clip() -> None:
    """Native program AD should lower strict scalar np.clip selection traces."""

    def objective(values: FloatArray) -> object:
        clipped = np.clip(values[0:1], values[1:2], values[2:3])
        return clipped.sum() + values[3] * values[0]

    sample = np.array([0.25, -0.5, 1.5, 2.0], dtype=np.float64)
    lower_replay = np.array([-1.0, -0.5, 1.5, 2.0], dtype=np.float64)
    upper_replay = np.array([2.0, -0.5, 1.5, -0.25], dtype=np.float64)
    parameters = (
        Parameter("value"),
        Parameter("lower"),
        Parameter("upper"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    lower_value, lower_gradient = program_adjoint_value_and_grad(
        objective,
        lower_replay,
        parameters,
    )
    upper_value, upper_gradient = program_adjoint_value_and_grad(
        objective,
        upper_replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "clip" in report.lowerable_ops
    assert "clip" in kernel.supported_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "fcmp olt" in kernel.llvm_ir
    assert "fcmp ogt" in kernel.llvm_ir
    assert "select i1" in kernel.llvm_ir
    assert kernel.value(lower_replay) == pytest.approx(lower_value)
    np.testing.assert_allclose(
        kernel.gradient(lower_replay),
        lower_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert kernel.value(upper_replay) == pytest.approx(upper_value)
    np.testing.assert_allclose(
        kernel.gradient(upper_replay),
        upper_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.vstack([sample, lower_replay, upper_replay])
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises(ValueError, match="clipping boundary"):
        kernel.gradient(np.array([-0.5, -0.5, 1.5, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="lower bound"):
        kernel.value(np.array([0.0, 2.0, 1.0, 2.0], dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_lowers_strict_selection_ops() -> None:
    """Native program AD should lower strict no-tie maximum/minimum selection ops."""

    def objective(values: FloatArray) -> object:
        return (
            np.maximum(values[0], values[1])
            + np.minimum(values[2], values[3])
            + values[4] * values[0]
        )

    sample = np.array([1.25, -0.25, 0.5, 1.5, 2.0], dtype=np.float64)
    replay = np.array([-0.5, 1.0, 2.0, -1.0, 0.25], dtype=np.float64)
    parameters = (
        Parameter("left"),
        Parameter("right"),
        Parameter("lower"),
        Parameter("upper"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {"maximum", "minimum"}.issubset(report.lowerable_ops)
    assert kernel.lowering_report.lowerable_ops == report.lowerable_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "maximum" in kernel.supported_ops
    assert "minimum" in kernel.supported_ops
    assert "fcmp ogt" in kernel.llvm_ir
    assert "fcmp olt" in kernel.llvm_ir
    assert "select i1" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises(ValueError, match="non-differentiable at equal inputs"):
        kernel.gradient(np.array([1.0, 1.0, 0.5, 1.5, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="non-differentiable at equal inputs"):
        kernel.batch_value_and_grad(
            np.array(
                [
                    [1.25, -0.25, 0.5, 1.5, 2.0],
                    [1.0, 1.0, 0.5, 1.5, 2.0],
                ],
                dtype=np.float64,
            )
        )


def test_whole_program_ad_trace_native_llvm_jit_executes_stable_branch_path() -> None:
    """Native LLVM/JIT program AD should execute stable branch traces and reject drift."""

    def objective(values: FloatArray) -> object:
        if values[0] > values[1]:
            return np.sin(values[0] * values[1]) + values[2]
        return np.cos(values[0] - values[1]) - values[2]

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    same_branch = np.array(
        [
            [1.25, -0.25, 0.5],
            [1.1, -0.4, 0.75],
            [2.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))

    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
        objective,
        sample,
        parameters,
    )
    ref_value, ref_gradient = program_adjoint_value_and_grad(
        objective,
        sample,
        parameters,
    )
    value, gradient = kernel.value_and_grad(sample)

    assert kernel.backend == "native_llvm_jit"
    assert any(item.startswith("branch:") for item in kernel.supported_ops)
    assert "stable executed branch signatures" in kernel.claim_boundary
    assert "stable executed branch signatures" in kernel.mlir_module.metadata["claim_boundary"]
    assert value == pytest.approx(ref_value)
    assert kernel.value(sample) == pytest.approx(ref_value)
    np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-10, atol=1e-10)

    tangent = np.array([0.25, -0.5, 1.5], dtype=np.float64)
    assert kernel.jvp(sample, tangent) == pytest.approx(float(np.dot(ref_gradient, tangent)))
    np.testing.assert_allclose(
        kernel.vjp(sample, np.array([2.0], dtype=np.float64)),
        2.0 * ref_gradient,
        rtol=1e-10,
        atol=1e-10,
    )

    batch_result = kernel.batch_value_and_grad(same_branch)
    batch_reference = [
        program_adjoint_value_and_grad(objective, row, parameters) for row in same_branch
    ]
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1e-10,
        atol=1e-10,
    )

    branch_drift = np.array([-0.25, 1.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.value(branch_drift)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.gradient(branch_drift)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.jvp(branch_drift, tangent)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.vjp(branch_drift, np.array([1.0], dtype=np.float64))

    drift_batch = same_branch.copy()
    drift_batch[1] = branch_drift
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_value_and_grad(drift_batch)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_jvp(drift_batch, np.vstack([tangent, tangent, tangent]))
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_vjp(drift_batch, np.ones(3, dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_lowers_elementary_ops() -> None:
    """Native LLVM/JIT program AD should lower adjoint-supported scalar primitives."""

    def objective(values: FloatArray) -> object:
        return (
            np.tan(values[0])
            + np.tanh(values[1])
            + np.expm1(values[0] * values[1])
            + np.log1p(values[2])
            + np.arcsin(values[0])
            - np.arccos(values[1])
            + np.reciprocal(values[3])
            + np.square(values[2])
            + np.abs(values[1])
        )

    sample = np.array([0.2, -0.3, 0.4, 1.5], dtype=np.float64)
    parameters = (
        Parameter("angle"),
        Parameter("offset"),
        Parameter("positive"),
        Parameter("denominator"),
    )
    tangent = np.array([0.5, -1.0, 1.5, -0.25], dtype=np.float64)

    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
        objective,
        sample,
        parameters,
    )
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        sample,
        parameters,
    )
    value, gradient = kernel.value_and_grad(sample)

    assert kernel.backend == "native_llvm_jit"
    assert kernel.mlir_module.resource_counts["native_supported_elementary_ops"] >= 9
    assert "expanded elementary ops" in kernel.mlir_module.metadata["claim_boundary"]
    for op in (
        "tan",
        "tanh",
        "expm1",
        "log1p",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
    ):
        assert op in kernel.supported_ops
    assert value == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-10, atol=1.0e-10)
    assert kernel.jvp(sample, tangent) == pytest.approx(float(np.dot(reference_gradient, tangent)))
    np.testing.assert_allclose(
        kernel.vjp(sample, np.array([1.75], dtype=np.float64)),
        1.75 * reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    abs_boundary = sample.copy()
    abs_boundary[1] = 0.0
    with pytest.raises(ValueError, match="output must be finite"):
        kernel.gradient(abs_boundary)
