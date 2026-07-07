# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- symmetric 2x2 native LLVM/JIT AD compilation for the MLIR surface
"""Native LLVM/JIT autodiff compilation for symmetric 2x2 primitives.

Emits the LLVM IR and builds the executable native-JIT kernels, lowering rules and
primitive transforms that evaluate each symmetric 2x2 primitive together with its forward
and reverse derivatives. It depends only on the shared executable-kernel core, the
native lowering primitives, the MLIR record types and the differentiable kernel
contracts, so it stays a leaf of the compiler package.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
)
from .mlir_executable_kernel import (
    ExecutableCompilerADKernel,
    _verify_executable_ad_kernel,
    compile_custom_derivative_rule_to_mlir,
    make_executable_ad_kernel_batching_rule,
)
from .mlir_native_primitives import (
    _as_finite_vector,
    _compile_native_llvm_jit_functions,
    _escape_mlir_string,
    _load_llvmlite_binding,
    _safe_llvm_symbol,
)
from .mlir_records import CompilerADExecutableConfig

FloatArray: TypeAlias = NDArray[np.float64]


def _compile_symmetric_2x2_cholesky_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "symmetric_2x2_cholesky"',
            '; source = "native_symmetric_2x2_cholesky_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 3",
            "; input_layout = upper_triangle",
            "; output_layout = lower_triangle",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a11ptr = getelementptr double, double* %values, i64 2",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %l00 = call double @llvm.sqrt.f64(double %a00)",
            "  %l10 = fdiv double %a01, %l00",
            "  %l10_sq = fmul double %l10, %l10",
            "  %schur = fsub double %a11, %l10_sq",
            "  %l11 = call double @llvm.sqrt.f64(double %schur)",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  store double %l00, double* %out0",
            "  store double %l10, double* %out1",
            "  store double %l11, double* %out2",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [3 x double]",
            "  %cotangent0 = getelementptr [3 x double], [3 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [3 x double], [3 x double]* %cotangent, i64 0, i64 1",
            "  %cotangent2 = getelementptr [3 x double], [3 x double]* %cotangent, i64 0, i64 2",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            "  store double 1.0, double* %cotangent2",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %l00_jvp = call double @llvm.sqrt.f64(double %a00_jvp)",
            "  %l10_jvp = fdiv double %a01_jvp, %l00_jvp",
            "  %l10_sq_jvp = fmul double %l10_jvp, %l10_jvp",
            "  %schur_jvp = fsub double %a11_jvp, %l10_sq_jvp",
            "  %l11_jvp = call double @llvm.sqrt.f64(double %schur_jvp)",
            "  %two_l00_jvp = fmul double 2.0, %l00_jvp",
            "  %tangent_l00 = fdiv double %t00_jvp, %two_l00_jvp",
            "  %t01_over_l00 = fdiv double %t01_jvp, %l00_jvp",
            "  %l10_tangent_l00 = fmul double %l10_jvp, %tangent_l00",
            "  %l10_tangent_l00_over_l00 = fdiv double %l10_tangent_l00, %l00_jvp",
            "  %tangent_l10 = fsub double %t01_over_l00, %l10_tangent_l00_over_l00",
            "  %two_l10_jvp = fmul double 2.0, %l10_jvp",
            "  %schur_tangent_term = fmul double %two_l10_jvp, %tangent_l10",
            "  %schur_tangent = fsub double %t11_jvp, %schur_tangent_term",
            "  %two_l11_jvp = fmul double 2.0, %l11_jvp",
            "  %tangent_l11 = fdiv double %schur_tangent, %two_l11_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  %out_jvp2 = getelementptr double, double* %out, i64 2",
            "  store double %tangent_l00, double* %out_jvp0",
            "  store double %tangent_l10, double* %out_jvp1",
            "  store double %tangent_l11, double* %out_jvp2",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent1ptr = getelementptr double, double* %cotangent, i64 1",
            "  %cotangent2ptr = getelementptr double, double* %cotangent, i64 2",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %cotangent1 = load double, double* %cotangent1ptr",
            "  %cotangent2 = load double, double* %cotangent2ptr",
            "  %l00_vjp = call double @llvm.sqrt.f64(double %a00_vjp)",
            "  %l10_vjp = fdiv double %a01_vjp, %l00_vjp",
            "  %l10_sq_vjp = fmul double %l10_vjp, %l10_vjp",
            "  %schur_vjp = fsub double %a11_vjp, %l10_sq_vjp",
            "  %l11_vjp = call double @llvm.sqrt.f64(double %schur_vjp)",
            "  %two_l11_vjp = fmul double 2.0, %l11_vjp",
            "  %adjoint_schur = fdiv double %cotangent2, %two_l11_vjp",
            "  %two_l10_vjp = fmul double 2.0, %l10_vjp",
            "  %l10_schur_adjoint = fmul double %two_l10_vjp, %adjoint_schur",
            "  %adjoint_l10 = fsub double %cotangent1, %l10_schur_adjoint",
            "  %l00_sq_vjp = fmul double %l00_vjp, %l00_vjp",
            "  %adjoint_l10_a01 = fmul double %adjoint_l10, %a01_vjp",
            "  %adjoint_l10_a01_over_l00_sq = fdiv double %adjoint_l10_a01, %l00_sq_vjp",
            "  %adjoint_l00 = fsub double %cotangent0, %adjoint_l10_a01_over_l00_sq",
            "  %two_l00_vjp = fmul double 2.0, %l00_vjp",
            "  %out0_value = fdiv double %adjoint_l00, %two_l00_vjp",
            "  %out1_value = fdiv double %adjoint_l10, %l00_vjp",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  store double %out0_value, double* %out0",
            "  store double %out1_value, double* %out1",
            "  store double %adjoint_schur, double* %out2",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_symmetric_2x2_eigenvalues_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "symmetric_2x2_eigenvalues"',
            '; source = "native_symmetric_2x2_eigenvalues_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 3",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a11ptr = getelementptr double, double* %values, i64 2",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %trace = fadd double %a00, %a11",
            "  %centre = fmul double 5.0e-1, %trace",
            "  %delta = fsub double %a00, %a11",
            "  %half_delta = fmul double 5.0e-1, %delta",
            "  %half_delta_square = fmul double %half_delta, %half_delta",
            "  %offdiag_square = fmul double %a01, %a01",
            "  %radius_square = fadd double %half_delta_square, %offdiag_square",
            "  %radius = call double @llvm.sqrt.f64(double %radius_square)",
            "  %lower = fsub double %centre, %radius",
            "  %upper = fadd double %centre, %radius",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %lower, double* %out0",
            "  store double %upper, double* %out1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [2 x double]",
            "  %cotangent0 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 1",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %trace_tangent_jvp = fadd double %t00_jvp, %t11_jvp",
            "  %centre_tangent_jvp = fmul double 5.0e-1, %trace_tangent_jvp",
            "  %delta_jvp = fsub double %a00_jvp, %a11_jvp",
            "  %half_delta_jvp = fmul double 5.0e-1, %delta_jvp",
            "  %tangent_delta_jvp = fsub double %t00_jvp, %t11_jvp",
            "  %tangent_half_delta_jvp = fmul double 5.0e-1, %tangent_delta_jvp",
            "  %half_delta_square_jvp = fmul double %half_delta_jvp, %half_delta_jvp",
            "  %offdiag_square_jvp = fmul double %a01_jvp, %a01_jvp",
            "  %radius_square_jvp = fadd double %half_delta_square_jvp, %offdiag_square_jvp",
            "  %radius_jvp = call double @llvm.sqrt.f64(double %radius_square_jvp)",
            "  %radius_term0_jvp = fmul double %half_delta_jvp, %tangent_half_delta_jvp",
            "  %radius_term1_jvp = fmul double %a01_jvp, %t01_jvp",
            "  %radius_tangent_num_jvp = fadd double %radius_term0_jvp, %radius_term1_jvp",
            "  %radius_tangent_jvp = fdiv double %radius_tangent_num_jvp, %radius_jvp",
            "  %lower_jvp = fsub double %centre_tangent_jvp, %radius_tangent_jvp",
            "  %upper_jvp = fadd double %centre_tangent_jvp, %radius_tangent_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  store double %lower_jvp, double* %out_jvp0",
            "  store double %upper_jvp, double* %out_jvp1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %lower_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %upper_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %lower_cotangent_vjp = load double, double* %lower_cotangent_ptr_vjp",
            "  %upper_cotangent_vjp = load double, double* %upper_cotangent_ptr_vjp",
            "  %delta_vjp = fsub double %a00_vjp, %a11_vjp",
            "  %half_delta_vjp = fmul double 5.0e-1, %delta_vjp",
            "  %half_delta_square_vjp = fmul double %half_delta_vjp, %half_delta_vjp",
            "  %offdiag_square_vjp = fmul double %a01_vjp, %a01_vjp",
            "  %radius_square_vjp = fadd double %half_delta_square_vjp, %offdiag_square_vjp",
            "  %radius_vjp = call double @llvm.sqrt.f64(double %radius_square_vjp)",
            "  %two_radius_vjp = fmul double 2.0, %radius_vjp",
            "  %half_term_vjp = fdiv double %half_delta_vjp, %two_radius_vjp",
            "  %offdiag_term_vjp = fdiv double %a01_vjp, %radius_vjp",
            "  %lower_a00_factor = fsub double 5.0e-1, %half_term_vjp",
            "  %upper_a00_factor = fadd double 5.0e-1, %half_term_vjp",
            "  %lower_a11_factor = fadd double 5.0e-1, %half_term_vjp",
            "  %upper_a11_factor = fsub double 5.0e-1, %half_term_vjp",
            "  %adj_a00_lower = fmul double %lower_cotangent_vjp, %lower_a00_factor",
            "  %adj_a00_upper = fmul double %upper_cotangent_vjp, %upper_a00_factor",
            "  %adj_a00 = fadd double %adj_a00_lower, %adj_a00_upper",
            "  %cotangent_diff = fsub double %upper_cotangent_vjp, %lower_cotangent_vjp",
            "  %adj_a01 = fmul double %cotangent_diff, %offdiag_term_vjp",
            "  %adj_a11_lower = fmul double %lower_cotangent_vjp, %lower_a11_factor",
            "  %adj_a11_upper = fmul double %upper_cotangent_vjp, %upper_a11_factor",
            "  %adj_a11 = fadd double %adj_a11_lower, %adj_a11_upper",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  store double %adj_a00, double* %out_vjp0",
            "  store double %adj_a01, double* %out_vjp1",
            "  store double %adj_a11, double* %out_vjp2",
            "  ret void",
            "}",
            "",
        ]
    )


def _as_native_symmetric_2x2_cholesky_values(
    label: str,
    values: Sequence[float] | FloatArray,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 3:
        raise ValueError(
            "native symmetric 2x2 Cholesky LLVM/JIT kernel requires upper-triangle values"
        )
    a00, a01, a11 = (float(item) for item in checked_values)
    schur = a11 - (a01 * a01) / a00 if a00 > 0.0 else float("nan")
    if not np.isfinite(schur) or a00 <= 1.0e-24 or schur <= 1.0e-24:
        raise ValueError(
            f"native symmetric 2x2 Cholesky LLVM/JIT kernel requires positive definite {label}"
        )
    return checked_values


def _call_native_symmetric_2x2_cholesky_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_symmetric_2x2_cholesky_values("values", values)
    if output_size != 3:
        raise ValueError("native symmetric 2x2 Cholesky LLVM/JIT output_size must be three")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_symmetric_2x2_cholesky_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_symmetric_2x2_cholesky_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_vector.size != 3:
        raise ValueError(
            f"native symmetric 2x2 Cholesky LLVM/JIT kernel requires three {label} values"
        )
    if output_size != 3:
        raise ValueError("native symmetric 2x2 Cholesky LLVM/JIT output_size must be three")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_symmetric_2x2_eigenvalues_values(
    label: str,
    values: Sequence[float] | FloatArray,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 3:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT kernel requires upper-triangle values"
        )
    half_delta = 0.5 * (checked_values[0] - checked_values[2])
    radius_square = half_delta * half_delta + checked_values[1] * checked_values[1]
    if not np.isfinite(radius_square) or float(radius_square) <= 1.0e-24:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT kernel requires distinct eigenvalues"
        )
    return checked_values


def _call_native_symmetric_2x2_eigenvalues_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_symmetric_2x2_eigenvalues_values("values", values)
    if output_size not in {2, 3}:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT output_size must be two or three"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_symmetric_2x2_eigenvalues_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_symmetric_2x2_eigenvalues_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 3 if label == "tangent" else 2
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {2, 3}:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT output_size must be two or three"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile SPD symmetric 2x2 Cholesky value/JVP/VJP kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native symmetric 2x2 Cholesky AD requires backend='native_llvm_jit'")
    values = _as_native_symmetric_2x2_cholesky_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_symmetric_2x2_cholesky_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_symmetric_2x2_cholesky_unary(
            native_functions["value"],
            raw_values,
            3,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_symmetric_2x2_cholesky_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            3,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_symmetric_2x2_cholesky_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            3,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_symmetric_2x2_cholesky_unary(
            native_functions["gradient"],
            values,
            3,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(3, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT symmetric 2x2 Cholesky sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT SPD symmetric 2x2 Cholesky value/JVP/VJP "
            "kernel with sum-output gradient provenance; public gradient remains "
            "scalar-output fail-closed and non-SPD matrices remain fail-closed"
        ),
    )


def make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for SPD symmetric 2x2 Cholesky native LLVM/JIT kernels."""
    captured_values = (
        None
        if sample_values is None
        else _as_native_symmetric_2x2_cholesky_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | FloatArray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | FloatArray | None = None,
        sample_cotangent: Sequence[float] | FloatArray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native symmetric 2x2 Cholesky lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT SPD 2x2 Cholesky contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native symmetric 2x2 Cholesky primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_symmetric_2x2_cholesky_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:cholesky;dimension:2;layout:upper_triangle"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_symmetric_2x2_cholesky",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT SPD symmetric 2x2 Cholesky JVP"
            ),
            "rust": (
                "available: Rust PyO3 SPD symmetric 2x2 Cholesky value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine symmetric_2x2_cholesky "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "symmetric_2x2_cholesky_value,symmetric_2x2_cholesky_jvp,"
                "symmetric_2x2_cholesky_vjp,symmetric_2x2_cholesky_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
            "jit": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT SPD symmetric 2x2 Cholesky value/JVP/VJP"
            ),
            "static_derivative_factory": "native_symmetric_2x2_cholesky_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "non_spd_symmetric_2x2_cholesky",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (3,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="symmetric_2x2_spd_cholesky_real_domain",
        effect="pure",
    )


def compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile distinct symmetric 2x2 eigenvalue value/JVP/VJP kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native symmetric 2x2 eigenvalue AD requires backend='native_llvm_jit'")
    values = _as_native_symmetric_2x2_eigenvalues_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_symmetric_2x2_eigenvalues_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_symmetric_2x2_eigenvalues_unary(
            native_functions["value"],
            raw_values,
            2,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_symmetric_2x2_eigenvalues_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            2,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_symmetric_2x2_eigenvalues_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            3,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_symmetric_2x2_eigenvalues_unary(
            native_functions["gradient"],
            values,
            3,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(2, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT symmetric 2x2 eigenvalue sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT distinct symmetric 2x2 eigenvalues value/JVP/VJP "
            "kernel with sum-output gradient provenance; public gradient remains "
            "scalar-output fail-closed and repeated eigenvalues remain fail-closed"
        ),
    )


def make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for distinct symmetric 2x2 eigenvalue native LLVM/JIT kernels."""
    captured_values = (
        None
        if sample_values is None
        else _as_native_symmetric_2x2_eigenvalues_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | FloatArray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | FloatArray | None = None,
        sample_cotangent: Sequence[float] | FloatArray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native symmetric 2x2 eigenvalue lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT distinct symmetric eigvalsh contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native symmetric 2x2 eigenvalue primitive transform requires "
            "backend='native_llvm_jit'"
        )
    values = _as_native_symmetric_2x2_eigenvalues_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:eigvalsh;dimension:2;layout:upper_triangle"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_symmetric_2x2_eigenvalues",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT distinct symmetric 2x2 eigenvalue JVP"
            ),
            "rust": (
                "available: Rust PyO3 distinct symmetric 2x2 eigenvalue "
                "value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine symmetric_2x2_eigenvalues "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "symmetric_2x2_eigenvalues_value,symmetric_2x2_eigenvalues_jvp,"
                "symmetric_2x2_eigenvalues_vjp,symmetric_2x2_eigenvalues_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
            "jit": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT distinct symmetric 2x2 eigenvalue value/JVP/VJP"
            ),
            "static_derivative_factory": "native_symmetric_2x2_eigenvalues_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "repeated_symmetric_2x2_eigenvalue",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (2,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="distinct_symmetric_2x2_eigenvalues_real_domain",
        effect="pure",
    )
