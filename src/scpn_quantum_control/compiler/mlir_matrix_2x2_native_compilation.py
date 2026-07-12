# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR matrix 2x2 native compilation module
# scpn-quantum-control -- 2x2 dense linear-algebra native LLVM/JIT AD compilation
"""Native LLVM/JIT autodiff compilation for closed-form 2x2 dense linear algebra.

Emits the LLVM IR and builds the executable native-JIT kernels, lowering rules and
primitive transforms for the closed-form 2x2 determinant, inverse, linear solve,
eigenvalues and eigensystem primitives together with their forward and reverse
derivatives. It depends only on the shared executable-kernel core, the native
lowering primitives, the MLIR record types and the differentiable contracts, so it
stays a leaf of the compiler package.
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


def _compile_matrix_2x2_determinant_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_determinant"',
            '; source = "native_matrix_2x2_determinant_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a10ptr = getelementptr double, double* %values, i64 2",
            "  %a11ptr = getelementptr double, double* %values, i64 3",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a10 = load double, double* %a10ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %main_diag = fmul double %a00, %a11",
            "  %off_diag = fmul double %a01, %a10",
            "  %det = fsub double %main_diag, %off_diag",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %det, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr_gradient = getelementptr double, double* %values, i64 0",
            "  %a01ptr_gradient = getelementptr double, double* %values, i64 1",
            "  %a10ptr_gradient = getelementptr double, double* %values, i64 2",
            "  %a11ptr_gradient = getelementptr double, double* %values, i64 3",
            "  %a00_gradient = load double, double* %a00ptr_gradient",
            "  %a01_gradient = load double, double* %a01ptr_gradient",
            "  %a10_gradient = load double, double* %a10ptr_gradient",
            "  %a11_gradient = load double, double* %a11ptr_gradient",
            "  %neg_a10_gradient = fsub double 0.0, %a10_gradient",
            "  %neg_a01_gradient = fsub double 0.0, %a01_gradient",
            "  %out_gradient0 = getelementptr double, double* %out, i64 0",
            "  %out_gradient1 = getelementptr double, double* %out, i64 1",
            "  %out_gradient2 = getelementptr double, double* %out, i64 2",
            "  %out_gradient3 = getelementptr double, double* %out, i64 3",
            "  store double %a11_gradient, double* %out_gradient0",
            "  store double %neg_a10_gradient, double* %out_gradient1",
            "  store double %neg_a01_gradient, double* %out_gradient2",
            "  store double %a00_gradient, double* %out_gradient3",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 3",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t10ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a10_jvp = load double, double* %a10ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t10_jvp = load double, double* %t10ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %term0_jvp = fmul double %t00_jvp, %a11_jvp",
            "  %term1_jvp = fmul double %a00_jvp, %t11_jvp",
            "  %term2_jvp = fmul double %t01_jvp, %a10_jvp",
            "  %term3_jvp = fmul double %a01_jvp, %t10_jvp",
            "  %sum0_jvp = fadd double %term0_jvp, %term1_jvp",
            "  %sum1_jvp = fsub double %sum0_jvp, %term2_jvp",
            "  %sum2_jvp = fsub double %sum1_jvp, %term3_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  store double %sum2_jvp, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 3",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a10_vjp = load double, double* %a10ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %neg_a10_vjp = fsub double 0.0, %a10_vjp",
            "  %neg_a01_vjp = fsub double 0.0, %a01_vjp",
            "  %scaled_vjp0 = fmul double %cotangent0, %a11_vjp",
            "  %scaled_vjp1 = fmul double %cotangent0, %neg_a10_vjp",
            "  %scaled_vjp2 = fmul double %cotangent0, %neg_a01_vjp",
            "  %scaled_vjp3 = fmul double %cotangent0, %a00_vjp",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %scaled_vjp0, double* %out_vjp0",
            "  store double %scaled_vjp1, double* %out_vjp1",
            "  store double %scaled_vjp2, double* %out_vjp2",
            "  store double %scaled_vjp3, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_inverse_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_inverse"',
            '; source = "native_matrix_2x2_inverse_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a10ptr = getelementptr double, double* %values, i64 2",
            "  %a11ptr = getelementptr double, double* %values, i64 3",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a10 = load double, double* %a10ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %main_diag = fmul double %a00, %a11",
            "  %off_diag = fmul double %a01, %a10",
            "  %det = fsub double %main_diag, %off_diag",
            "  %neg_a01 = fsub double 0.0, %a01",
            "  %neg_a10 = fsub double 0.0, %a10",
            "  %inv00 = fdiv double %a11, %det",
            "  %inv01 = fdiv double %neg_a01, %det",
            "  %inv10 = fdiv double %neg_a10, %det",
            "  %inv11 = fdiv double %a00, %det",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  %out3 = getelementptr double, double* %out, i64 3",
            "  store double %inv00, double* %out0",
            "  store double %inv01, double* %out1",
            "  store double %inv10, double* %out2",
            "  store double %inv11, double* %out3",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [4 x double]",
            "  %cotangent0 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 1",
            "  %cotangent2 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 2",
            "  %cotangent3 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 3",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            "  store double 1.0, double* %cotangent2",
            "  store double 1.0, double* %cotangent3",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 3",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t10ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a10_jvp = load double, double* %a10ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t10_jvp = load double, double* %t10ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %main_diag_jvp = fmul double %a00_jvp, %a11_jvp",
            "  %off_diag_jvp = fmul double %a01_jvp, %a10_jvp",
            "  %det_jvp = fsub double %main_diag_jvp, %off_diag_jvp",
            "  %det2_jvp = fmul double %det_jvp, %det_jvp",
            "  %term_detdot0 = fmul double %t00_jvp, %a11_jvp",
            "  %term_detdot1 = fmul double %a00_jvp, %t11_jvp",
            "  %term_detdot2 = fmul double %t01_jvp, %a10_jvp",
            "  %term_detdot3 = fmul double %a01_jvp, %t10_jvp",
            "  %detdot_sum0 = fadd double %term_detdot0, %term_detdot1",
            "  %detdot_sum1 = fsub double %detdot_sum0, %term_detdot2",
            "  %detdot = fsub double %detdot_sum1, %term_detdot3",
            "  %num00_left = fmul double %t11_jvp, %det_jvp",
            "  %num00_right = fmul double %a11_jvp, %detdot",
            "  %num00 = fsub double %num00_left, %num00_right",
            "  %neg_t01_jvp = fsub double 0.0, %t01_jvp",
            "  %num01_left = fmul double %neg_t01_jvp, %det_jvp",
            "  %num01_right = fmul double %a01_jvp, %detdot",
            "  %num01 = fadd double %num01_left, %num01_right",
            "  %neg_t10_jvp = fsub double 0.0, %t10_jvp",
            "  %num10_left = fmul double %neg_t10_jvp, %det_jvp",
            "  %num10_right = fmul double %a10_jvp, %detdot",
            "  %num10 = fadd double %num10_left, %num10_right",
            "  %num11_left = fmul double %t00_jvp, %det_jvp",
            "  %num11_right = fmul double %a00_jvp, %detdot",
            "  %num11 = fsub double %num11_left, %num11_right",
            "  %jvp00 = fdiv double %num00, %det2_jvp",
            "  %jvp01 = fdiv double %num01, %det2_jvp",
            "  %jvp10 = fdiv double %num10, %det2_jvp",
            "  %jvp11 = fdiv double %num11, %det2_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  %out_jvp2 = getelementptr double, double* %out, i64 2",
            "  %out_jvp3 = getelementptr double, double* %out, i64 3",
            "  store double %jvp00, double* %out_jvp0",
            "  store double %jvp01, double* %out_jvp1",
            "  store double %jvp10, double* %out_jvp2",
            "  store double %jvp11, double* %out_jvp3",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 3",
            "  %g00ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %g01ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %g10ptr_vjp = getelementptr double, double* %cotangent, i64 2",
            "  %g11ptr_vjp = getelementptr double, double* %cotangent, i64 3",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a10_vjp = load double, double* %a10ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %g00_vjp = load double, double* %g00ptr_vjp",
            "  %g01_vjp = load double, double* %g01ptr_vjp",
            "  %g10_vjp = load double, double* %g10ptr_vjp",
            "  %g11_vjp = load double, double* %g11ptr_vjp",
            "  %main_diag_vjp = fmul double %a00_vjp, %a11_vjp",
            "  %off_diag_vjp = fmul double %a01_vjp, %a10_vjp",
            "  %det_vjp = fsub double %main_diag_vjp, %off_diag_vjp",
            "  %neg_a01_vjp = fsub double 0.0, %a01_vjp",
            "  %neg_a10_vjp = fsub double 0.0, %a10_vjp",
            "  %y00 = fdiv double %a11_vjp, %det_vjp",
            "  %y01 = fdiv double %neg_a01_vjp, %det_vjp",
            "  %y10 = fdiv double %neg_a10_vjp, %det_vjp",
            "  %y11 = fdiv double %a00_vjp, %det_vjp",
            "  %m00_left = fmul double %y00, %g00_vjp",
            "  %m00_right = fmul double %y10, %g10_vjp",
            "  %m00 = fadd double %m00_left, %m00_right",
            "  %m01_left = fmul double %y00, %g01_vjp",
            "  %m01_right = fmul double %y10, %g11_vjp",
            "  %m01 = fadd double %m01_left, %m01_right",
            "  %m10_left = fmul double %y01, %g00_vjp",
            "  %m10_right = fmul double %y11, %g10_vjp",
            "  %m10 = fadd double %m10_left, %m10_right",
            "  %m11_left = fmul double %y01, %g01_vjp",
            "  %m11_right = fmul double %y11, %g11_vjp",
            "  %m11 = fadd double %m11_left, %m11_right",
            "  %h00_left = fmul double %m00, %y00",
            "  %h00_right = fmul double %m01, %y01",
            "  %h00_sum = fadd double %h00_left, %h00_right",
            "  %h00 = fsub double 0.0, %h00_sum",
            "  %h01_left = fmul double %m00, %y10",
            "  %h01_right = fmul double %m01, %y11",
            "  %h01_sum = fadd double %h01_left, %h01_right",
            "  %h01 = fsub double 0.0, %h01_sum",
            "  %h10_left = fmul double %m10, %y00",
            "  %h10_right = fmul double %m11, %y01",
            "  %h10_sum = fadd double %h10_left, %h10_right",
            "  %h10 = fsub double 0.0, %h10_sum",
            "  %h11_left = fmul double %m10, %y10",
            "  %h11_right = fmul double %m11, %y11",
            "  %h11_sum = fadd double %h11_left, %h11_right",
            "  %h11 = fsub double 0.0, %h11_sum",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %h00, double* %out_vjp0",
            "  store double %h01, double* %out_vjp1",
            "  store double %h10, double* %out_vjp2",
            "  store double %h11, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_solve_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_solve"',
            '; source = "native_matrix_2x2_solve_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 6",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a10ptr = getelementptr double, double* %values, i64 2",
            "  %a11ptr = getelementptr double, double* %values, i64 3",
            "  %b0ptr = getelementptr double, double* %values, i64 4",
            "  %b1ptr = getelementptr double, double* %values, i64 5",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a10 = load double, double* %a10ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %b0 = load double, double* %b0ptr",
            "  %b1 = load double, double* %b1ptr",
            "  %main_diag = fmul double %a00, %a11",
            "  %off_diag = fmul double %a01, %a10",
            "  %det = fsub double %main_diag, %off_diag",
            "  %num0_left = fmul double %a11, %b0",
            "  %num0_right = fmul double %a01, %b1",
            "  %num0 = fsub double %num0_left, %num0_right",
            "  %num1_left = fmul double %a00, %b1",
            "  %num1_right = fmul double %a10, %b0",
            "  %num1 = fsub double %num1_left, %num1_right",
            "  %x0 = fdiv double %num0, %det",
            "  %x1 = fdiv double %num1, %det",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %x0, double* %out0",
            "  store double %x1, double* %out1",
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
            "  %a10ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 3",
            "  %b0ptr_jvp = getelementptr double, double* %values, i64 4",
            "  %b1ptr_jvp = getelementptr double, double* %values, i64 5",
            "  %da00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %da01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %da10ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %da11ptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %db0ptr_jvp = getelementptr double, double* %tangent, i64 4",
            "  %db1ptr_jvp = getelementptr double, double* %tangent, i64 5",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a10_jvp = load double, double* %a10ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %b0_jvp = load double, double* %b0ptr_jvp",
            "  %b1_jvp = load double, double* %b1ptr_jvp",
            "  %da00_jvp = load double, double* %da00ptr_jvp",
            "  %da01_jvp = load double, double* %da01ptr_jvp",
            "  %da10_jvp = load double, double* %da10ptr_jvp",
            "  %da11_jvp = load double, double* %da11ptr_jvp",
            "  %db0_jvp = load double, double* %db0ptr_jvp",
            "  %db1_jvp = load double, double* %db1ptr_jvp",
            "  %main_diag_jvp = fmul double %a00_jvp, %a11_jvp",
            "  %off_diag_jvp = fmul double %a01_jvp, %a10_jvp",
            "  %det_jvp = fsub double %main_diag_jvp, %off_diag_jvp",
            "  %num0_left_jvp = fmul double %a11_jvp, %b0_jvp",
            "  %num0_right_jvp = fmul double %a01_jvp, %b1_jvp",
            "  %num0_jvp = fsub double %num0_left_jvp, %num0_right_jvp",
            "  %num1_left_jvp = fmul double %a00_jvp, %b1_jvp",
            "  %num1_right_jvp = fmul double %a10_jvp, %b0_jvp",
            "  %num1_jvp = fsub double %num1_left_jvp, %num1_right_jvp",
            "  %x0_jvp = fdiv double %num0_jvp, %det_jvp",
            "  %x1_jvp = fdiv double %num1_jvp, %det_jvp",
            "  %dax0_left = fmul double %da00_jvp, %x0_jvp",
            "  %dax0_right = fmul double %da01_jvp, %x1_jvp",
            "  %dax0 = fadd double %dax0_left, %dax0_right",
            "  %dax1_left = fmul double %da10_jvp, %x0_jvp",
            "  %dax1_right = fmul double %da11_jvp, %x1_jvp",
            "  %dax1 = fadd double %dax1_left, %dax1_right",
            "  %r0 = fsub double %db0_jvp, %dax0",
            "  %r1 = fsub double %db1_jvp, %dax1",
            "  %dx0_left = fmul double %a11_jvp, %r0",
            "  %dx0_right = fmul double %a01_jvp, %r1",
            "  %dx0_num = fsub double %dx0_left, %dx0_right",
            "  %dx1_left = fmul double %a00_jvp, %r1",
            "  %dx1_right = fmul double %a10_jvp, %r0",
            "  %dx1_num = fsub double %dx1_left, %dx1_right",
            "  %dx0 = fdiv double %dx0_num, %det_jvp",
            "  %dx1 = fdiv double %dx1_num, %det_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  store double %dx0, double* %out_jvp0",
            "  store double %dx1, double* %out_jvp1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 3",
            "  %b0ptr_vjp = getelementptr double, double* %values, i64 4",
            "  %b1ptr_vjp = getelementptr double, double* %values, i64 5",
            "  %c0ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %c1ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a10_vjp = load double, double* %a10ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %b0_vjp = load double, double* %b0ptr_vjp",
            "  %b1_vjp = load double, double* %b1ptr_vjp",
            "  %c0_vjp = load double, double* %c0ptr_vjp",
            "  %c1_vjp = load double, double* %c1ptr_vjp",
            "  %main_diag_vjp = fmul double %a00_vjp, %a11_vjp",
            "  %off_diag_vjp = fmul double %a01_vjp, %a10_vjp",
            "  %det_vjp = fsub double %main_diag_vjp, %off_diag_vjp",
            "  %num0_left_vjp = fmul double %a11_vjp, %b0_vjp",
            "  %num0_right_vjp = fmul double %a01_vjp, %b1_vjp",
            "  %num0_vjp = fsub double %num0_left_vjp, %num0_right_vjp",
            "  %num1_left_vjp = fmul double %a00_vjp, %b1_vjp",
            "  %num1_right_vjp = fmul double %a10_vjp, %b0_vjp",
            "  %num1_vjp = fsub double %num1_left_vjp, %num1_right_vjp",
            "  %x0_vjp = fdiv double %num0_vjp, %det_vjp",
            "  %x1_vjp = fdiv double %num1_vjp, %det_vjp",
            "  %p0_left = fmul double %a11_vjp, %c0_vjp",
            "  %p0_right = fmul double %a10_vjp, %c1_vjp",
            "  %p0_num = fsub double %p0_left, %p0_right",
            "  %p1_left = fmul double %a00_vjp, %c1_vjp",
            "  %p1_right = fmul double %a01_vjp, %c0_vjp",
            "  %p1_num = fsub double %p1_left, %p1_right",
            "  %p0 = fdiv double %p0_num, %det_vjp",
            "  %p1 = fdiv double %p1_num, %det_vjp",
            "  %adj_a00_raw = fmul double %p0, %x0_vjp",
            "  %adj_a01_raw = fmul double %p0, %x1_vjp",
            "  %adj_a10_raw = fmul double %p1, %x0_vjp",
            "  %adj_a11_raw = fmul double %p1, %x1_vjp",
            "  %adj_a00 = fsub double 0.0, %adj_a00_raw",
            "  %adj_a01 = fsub double 0.0, %adj_a01_raw",
            "  %adj_a10 = fsub double 0.0, %adj_a10_raw",
            "  %adj_a11 = fsub double 0.0, %adj_a11_raw",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  %out_vjp4 = getelementptr double, double* %out, i64 4",
            "  %out_vjp5 = getelementptr double, double* %out, i64 5",
            "  store double %adj_a00, double* %out_vjp0",
            "  store double %adj_a01, double* %out_vjp1",
            "  store double %adj_a10, double* %out_vjp2",
            "  store double %adj_a11, double* %out_vjp3",
            "  store double %p0, double* %out_vjp4",
            "  store double %p1, double* %out_vjp5",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_eigenvalues_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_eigenvalues"',
            '; source = "native_matrix_2x2_eigenvalues_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %aptr = getelementptr double, double* %values, i64 0",
            "  %bptr = getelementptr double, double* %values, i64 1",
            "  %cptr = getelementptr double, double* %values, i64 2",
            "  %dptr = getelementptr double, double* %values, i64 3",
            "  %a = load double, double* %aptr",
            "  %b = load double, double* %bptr",
            "  %c = load double, double* %cptr",
            "  %d = load double, double* %dptr",
            "  %trace = fadd double %a, %d",
            "  %delta = fsub double %a, %d",
            "  %delta_square = fmul double %delta, %delta",
            "  %bc = fmul double %b, %c",
            "  %four_bc = fmul double 4.0, %bc",
            "  %discriminant = fadd double %delta_square, %four_bc",
            "  %root = call double @llvm.sqrt.f64(double %discriminant)",
            "  %lower_num = fsub double %trace, %root",
            "  %upper_num = fadd double %trace, %root",
            "  %lower = fmul double 5.0e-1, %lower_num",
            "  %upper = fmul double 5.0e-1, %upper_num",
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
            "  %aptr_jvp = getelementptr double, double* %values, i64 0",
            "  %bptr_jvp = getelementptr double, double* %values, i64 1",
            "  %cptr_jvp = getelementptr double, double* %values, i64 2",
            "  %dptr_jvp = getelementptr double, double* %values, i64 3",
            "  %taptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %tbptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %tcptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %tdptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a_jvp = load double, double* %aptr_jvp",
            "  %b_jvp = load double, double* %bptr_jvp",
            "  %c_jvp = load double, double* %cptr_jvp",
            "  %d_jvp = load double, double* %dptr_jvp",
            "  %ta_jvp = load double, double* %taptr_jvp",
            "  %tb_jvp = load double, double* %tbptr_jvp",
            "  %tc_jvp = load double, double* %tcptr_jvp",
            "  %td_jvp = load double, double* %tdptr_jvp",
            "  %trace_tangent_jvp = fadd double %ta_jvp, %td_jvp",
            "  %delta_jvp = fsub double %a_jvp, %d_jvp",
            "  %delta_tangent_jvp = fsub double %ta_jvp, %td_jvp",
            "  %delta_square_jvp = fmul double %delta_jvp, %delta_jvp",
            "  %bc_jvp = fmul double %b_jvp, %c_jvp",
            "  %four_bc_jvp = fmul double 4.0, %bc_jvp",
            "  %discriminant_jvp = fadd double %delta_square_jvp, %four_bc_jvp",
            "  %root_jvp = call double @llvm.sqrt.f64(double %discriminant_jvp)",
            "  %disc_tangent_delta = fmul double 2.0, %delta_jvp",
            "  %disc_tangent_delta_scaled = fmul double %disc_tangent_delta, %delta_tangent_jvp",
            "  %tb_c = fmul double %tb_jvp, %c_jvp",
            "  %b_tc = fmul double %b_jvp, %tc_jvp",
            "  %offdiag_tangent_sum = fadd double %tb_c, %b_tc",
            "  %offdiag_disc_tangent = fmul double 4.0, %offdiag_tangent_sum",
            "  %disc_tangent = fadd double %disc_tangent_delta_scaled, %offdiag_disc_tangent",
            "  %two_root_jvp = fmul double 2.0, %root_jvp",
            "  %root_tangent = fdiv double %disc_tangent, %two_root_jvp",
            "  %lower_tangent_num = fsub double %trace_tangent_jvp, %root_tangent",
            "  %upper_tangent_num = fadd double %trace_tangent_jvp, %root_tangent",
            "  %lower_tangent = fmul double 5.0e-1, %lower_tangent_num",
            "  %upper_tangent = fmul double 5.0e-1, %upper_tangent_num",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  store double %lower_tangent, double* %out_jvp0",
            "  store double %upper_tangent, double* %out_jvp1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %aptr_vjp = getelementptr double, double* %values, i64 0",
            "  %bptr_vjp = getelementptr double, double* %values, i64 1",
            "  %cptr_vjp = getelementptr double, double* %values, i64 2",
            "  %dptr_vjp = getelementptr double, double* %values, i64 3",
            "  %lower_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %upper_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %a_vjp = load double, double* %aptr_vjp",
            "  %b_vjp = load double, double* %bptr_vjp",
            "  %c_vjp = load double, double* %cptr_vjp",
            "  %d_vjp = load double, double* %dptr_vjp",
            "  %lower_cotangent_vjp = load double, double* %lower_cotangent_ptr_vjp",
            "  %upper_cotangent_vjp = load double, double* %upper_cotangent_ptr_vjp",
            "  %delta_vjp = fsub double %a_vjp, %d_vjp",
            "  %delta_square_vjp = fmul double %delta_vjp, %delta_vjp",
            "  %bc_vjp = fmul double %b_vjp, %c_vjp",
            "  %four_bc_vjp = fmul double 4.0, %bc_vjp",
            "  %discriminant_vjp = fadd double %delta_square_vjp, %four_bc_vjp",
            "  %root_vjp = call double @llvm.sqrt.f64(double %discriminant_vjp)",
            "  %cotangent_sum = fadd double %lower_cotangent_vjp, %upper_cotangent_vjp",
            "  %alpha = fmul double 5.0e-1, %cotangent_sum",
            "  %cotangent_diff = fsub double %upper_cotangent_vjp, %lower_cotangent_vjp",
            "  %four_root_vjp = fmul double 4.0, %root_vjp",
            "  %beta = fdiv double %cotangent_diff, %four_root_vjp",
            "  %two_delta = fmul double 2.0, %delta_vjp",
            "  %a_disc_term = fmul double %two_delta, %beta",
            "  %adj_a = fadd double %alpha, %a_disc_term",
            "  %adj_d = fsub double %alpha, %a_disc_term",
            "  %four_c = fmul double 4.0, %c_vjp",
            "  %four_b = fmul double 4.0, %b_vjp",
            "  %adj_b = fmul double %four_c, %beta",
            "  %adj_c = fmul double %four_b, %beta",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %adj_a, double* %out_vjp0",
            "  store double %adj_b, double* %out_vjp1",
            "  store double %adj_c, double* %out_vjp2",
            "  store double %adj_d, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_eigensystem_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_eigensystem"',
            '; source = "native_matrix_2x2_eigensystem_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            "; output_count = 6",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %aptr = getelementptr double, double* %values, i64 0",
            "  %bptr = getelementptr double, double* %values, i64 1",
            "  %cptr = getelementptr double, double* %values, i64 2",
            "  %dptr = getelementptr double, double* %values, i64 3",
            "  %a = load double, double* %aptr",
            "  %b = load double, double* %bptr",
            "  %c = load double, double* %cptr",
            "  %d = load double, double* %dptr",
            "  %trace = fadd double %a, %d",
            "  %delta = fsub double %a, %d",
            "  %delta_square = fmul double %delta, %delta",
            "  %bc = fmul double %b, %c",
            "  %four_bc = fmul double 4.0, %bc",
            "  %discriminant = fadd double %delta_square, %four_bc",
            "  %root = call double @llvm.sqrt.f64(double %discriminant)",
            "  %lower_num = fsub double %trace, %root",
            "  %upper_num = fadd double %trace, %root",
            "  %lower = fmul double 5.0e-1, %lower_num",
            "  %upper = fmul double 5.0e-1, %upper_num",
            "  %neg_delta = fsub double 0.0, %delta",
            "  %q_lower_num = fsub double %neg_delta, %root",
            "  %q_upper_num = fadd double %neg_delta, %root",
            "  %q_lower = fmul double 5.0e-1, %q_lower_num",
            "  %q_upper = fmul double 5.0e-1, %q_upper_num",
            "  %b_square_lower = fmul double %b, %b",
            "  %q_lower_square = fmul double %q_lower, %q_lower",
            "  %norm_lower_square = fadd double %b_square_lower, %q_lower_square",
            "  %norm_lower = call double @llvm.sqrt.f64(double %norm_lower_square)",
            "  %b_square_upper = fmul double %b, %b",
            "  %q_upper_square = fmul double %q_upper, %q_upper",
            "  %norm_upper_square = fadd double %b_square_upper, %q_upper_square",
            "  %norm_upper = call double @llvm.sqrt.f64(double %norm_upper_square)",
            "  %v_lower0 = fdiv double %b, %norm_lower",
            "  %v_lower1 = fdiv double %q_lower, %norm_lower",
            "  %v_upper0 = fdiv double %b, %norm_upper",
            "  %v_upper1 = fdiv double %q_upper, %norm_upper",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  %out3 = getelementptr double, double* %out, i64 3",
            "  %out4 = getelementptr double, double* %out, i64 4",
            "  %out5 = getelementptr double, double* %out, i64 5",
            "  store double %lower, double* %out0",
            "  store double %upper, double* %out1",
            "  store double %v_lower0, double* %out2",
            "  store double %v_upper0, double* %out3",
            "  store double %v_lower1, double* %out4",
            "  store double %v_upper1, double* %out5",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [6 x double]",
            "  %cotangent0 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 1",
            "  %cotangent2 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 2",
            "  %cotangent3 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 3",
            "  %cotangent4 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 4",
            "  %cotangent5 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 5",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            "  store double 1.0, double* %cotangent2",
            "  store double 1.0, double* %cotangent3",
            "  store double 1.0, double* %cotangent4",
            "  store double 1.0, double* %cotangent5",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %aptr_jvp = getelementptr double, double* %values, i64 0",
            "  %bptr_jvp = getelementptr double, double* %values, i64 1",
            "  %cptr_jvp = getelementptr double, double* %values, i64 2",
            "  %dptr_jvp = getelementptr double, double* %values, i64 3",
            "  %taptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %tbptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %tcptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %tdptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a_jvp = load double, double* %aptr_jvp",
            "  %b_jvp = load double, double* %bptr_jvp",
            "  %c_jvp = load double, double* %cptr_jvp",
            "  %d_jvp = load double, double* %dptr_jvp",
            "  %ta_jvp = load double, double* %taptr_jvp",
            "  %tb_jvp = load double, double* %tbptr_jvp",
            "  %tc_jvp = load double, double* %tcptr_jvp",
            "  %td_jvp = load double, double* %tdptr_jvp",
            "  %trace_tangent_jvp = fadd double %ta_jvp, %td_jvp",
            "  %delta_jvp = fsub double %a_jvp, %d_jvp",
            "  %delta_tangent_jvp = fsub double %ta_jvp, %td_jvp",
            "  %delta_square_jvp = fmul double %delta_jvp, %delta_jvp",
            "  %bc_jvp = fmul double %b_jvp, %c_jvp",
            "  %four_bc_jvp = fmul double 4.0, %bc_jvp",
            "  %discriminant_jvp = fadd double %delta_square_jvp, %four_bc_jvp",
            "  %root_jvp = call double @llvm.sqrt.f64(double %discriminant_jvp)",
            "  %disc_tangent_delta = fmul double 2.0, %delta_jvp",
            "  %disc_tangent_delta_scaled = fmul double %disc_tangent_delta, %delta_tangent_jvp",
            "  %tb_c = fmul double %tb_jvp, %c_jvp",
            "  %b_tc = fmul double %b_jvp, %tc_jvp",
            "  %offdiag_tangent_sum = fadd double %tb_c, %b_tc",
            "  %offdiag_disc_tangent = fmul double 4.0, %offdiag_tangent_sum",
            "  %disc_tangent = fadd double %disc_tangent_delta_scaled, %offdiag_disc_tangent",
            "  %two_root_jvp = fmul double 2.0, %root_jvp",
            "  %root_tangent = fdiv double %disc_tangent, %two_root_jvp",
            "  %lower_tangent_num = fsub double %trace_tangent_jvp, %root_tangent",
            "  %upper_tangent_num = fadd double %trace_tangent_jvp, %root_tangent",
            "  %lower_tangent = fmul double 5.0e-1, %lower_tangent_num",
            "  %upper_tangent = fmul double 5.0e-1, %upper_tangent_num",
            "  %neg_delta_jvp = fsub double 0.0, %delta_jvp",
            "  %q_lower_num_jvp = fsub double %neg_delta_jvp, %root_jvp",
            "  %q_upper_num_jvp = fadd double %neg_delta_jvp, %root_jvp",
            "  %q_lower_jvp = fmul double 5.0e-1, %q_lower_num_jvp",
            "  %q_upper_jvp = fmul double 5.0e-1, %q_upper_num_jvp",
            "  %q_lower_tangent = fsub double %lower_tangent, %ta_jvp",
            "  %q_upper_tangent = fsub double %upper_tangent, %ta_jvp",
            "  %b_square_lower_jvp = fmul double %b_jvp, %b_jvp",
            "  %q_lower_square_jvp = fmul double %q_lower_jvp, %q_lower_jvp",
            "  %norm_lower_square_jvp = fadd double %b_square_lower_jvp, %q_lower_square_jvp",
            "  %norm_lower_jvp = call double @llvm.sqrt.f64(double %norm_lower_square_jvp)",
            "  %b_square_upper_jvp = fmul double %b_jvp, %b_jvp",
            "  %q_upper_square_jvp = fmul double %q_upper_jvp, %q_upper_jvp",
            "  %norm_upper_square_jvp = fadd double %b_square_upper_jvp, %q_upper_square_jvp",
            "  %norm_upper_jvp = call double @llvm.sqrt.f64(double %norm_upper_square_jvp)",
            "  %v_lower0_jvp = fdiv double %b_jvp, %norm_lower_jvp",
            "  %v_lower1_jvp = fdiv double %q_lower_jvp, %norm_lower_jvp",
            "  %v_upper0_jvp = fdiv double %b_jvp, %norm_upper_jvp",
            "  %v_upper1_jvp = fdiv double %q_upper_jvp, %norm_upper_jvp",
            "  %vl_dot_term0 = fmul double %v_lower0_jvp, %tb_jvp",
            "  %vl_dot_term1 = fmul double %v_lower1_jvp, %q_lower_tangent",
            "  %vl_dot = fadd double %vl_dot_term0, %vl_dot_term1",
            "  %vl_proj0 = fmul double %v_lower0_jvp, %vl_dot",
            "  %vl_proj1 = fmul double %v_lower1_jvp, %vl_dot",
            "  %vl_raw0 = fsub double %tb_jvp, %vl_proj0",
            "  %vl_raw1 = fsub double %q_lower_tangent, %vl_proj1",
            "  %vl_tangent0 = fdiv double %vl_raw0, %norm_lower_jvp",
            "  %vl_tangent1 = fdiv double %vl_raw1, %norm_lower_jvp",
            "  %vu_dot_term0 = fmul double %v_upper0_jvp, %tb_jvp",
            "  %vu_dot_term1 = fmul double %v_upper1_jvp, %q_upper_tangent",
            "  %vu_dot = fadd double %vu_dot_term0, %vu_dot_term1",
            "  %vu_proj0 = fmul double %v_upper0_jvp, %vu_dot",
            "  %vu_proj1 = fmul double %v_upper1_jvp, %vu_dot",
            "  %vu_raw0 = fsub double %tb_jvp, %vu_proj0",
            "  %vu_raw1 = fsub double %q_upper_tangent, %vu_proj1",
            "  %vu_tangent0 = fdiv double %vu_raw0, %norm_upper_jvp",
            "  %vu_tangent1 = fdiv double %vu_raw1, %norm_upper_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  %out_jvp2 = getelementptr double, double* %out, i64 2",
            "  %out_jvp3 = getelementptr double, double* %out, i64 3",
            "  %out_jvp4 = getelementptr double, double* %out, i64 4",
            "  %out_jvp5 = getelementptr double, double* %out, i64 5",
            "  store double %lower_tangent, double* %out_jvp0",
            "  store double %upper_tangent, double* %out_jvp1",
            "  store double %vl_tangent0, double* %out_jvp2",
            "  store double %vu_tangent0, double* %out_jvp3",
            "  store double %vl_tangent1, double* %out_jvp4",
            "  store double %vu_tangent1, double* %out_jvp5",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %aptr_vjp = getelementptr double, double* %values, i64 0",
            "  %bptr_vjp = getelementptr double, double* %values, i64 1",
            "  %cptr_vjp = getelementptr double, double* %values, i64 2",
            "  %dptr_vjp = getelementptr double, double* %values, i64 3",
            "  %clptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %cuptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %cvl0ptr_vjp = getelementptr double, double* %cotangent, i64 2",
            "  %cvu0ptr_vjp = getelementptr double, double* %cotangent, i64 3",
            "  %cvl1ptr_vjp = getelementptr double, double* %cotangent, i64 4",
            "  %cvu1ptr_vjp = getelementptr double, double* %cotangent, i64 5",
            "  %a_vjp = load double, double* %aptr_vjp",
            "  %b_vjp = load double, double* %bptr_vjp",
            "  %c_vjp = load double, double* %cptr_vjp",
            "  %d_vjp = load double, double* %dptr_vjp",
            "  %cl_vjp = load double, double* %clptr_vjp",
            "  %cu_vjp = load double, double* %cuptr_vjp",
            "  %cvl0_vjp = load double, double* %cvl0ptr_vjp",
            "  %cvu0_vjp = load double, double* %cvu0ptr_vjp",
            "  %cvl1_vjp = load double, double* %cvl1ptr_vjp",
            "  %cvu1_vjp = load double, double* %cvu1ptr_vjp",
            "  %delta_vjp = fsub double %a_vjp, %d_vjp",
            "  %delta_square_vjp = fmul double %delta_vjp, %delta_vjp",
            "  %bc_vjp = fmul double %b_vjp, %c_vjp",
            "  %four_bc_vjp = fmul double 4.0, %bc_vjp",
            "  %discriminant_vjp = fadd double %delta_square_vjp, %four_bc_vjp",
            "  %root_vjp = call double @llvm.sqrt.f64(double %discriminant_vjp)",
            "  %neg_delta_vjp = fsub double 0.0, %delta_vjp",
            "  %q_lower_num_vjp = fsub double %neg_delta_vjp, %root_vjp",
            "  %q_upper_num_vjp = fadd double %neg_delta_vjp, %root_vjp",
            "  %q_lower_vjp = fmul double 5.0e-1, %q_lower_num_vjp",
            "  %q_upper_vjp = fmul double 5.0e-1, %q_upper_num_vjp",
            "  %b_square_lower_vjp = fmul double %b_vjp, %b_vjp",
            "  %q_lower_square_vjp = fmul double %q_lower_vjp, %q_lower_vjp",
            "  %norm_lower_square_vjp = fadd double %b_square_lower_vjp, %q_lower_square_vjp",
            "  %norm_lower_vjp = call double @llvm.sqrt.f64(double %norm_lower_square_vjp)",
            "  %b_square_upper_vjp = fmul double %b_vjp, %b_vjp",
            "  %q_upper_square_vjp = fmul double %q_upper_vjp, %q_upper_vjp",
            "  %norm_upper_square_vjp = fadd double %b_square_upper_vjp, %q_upper_square_vjp",
            "  %norm_upper_vjp = call double @llvm.sqrt.f64(double %norm_upper_square_vjp)",
            "  %vl0_vjp = fdiv double %b_vjp, %norm_lower_vjp",
            "  %vl1_vjp = fdiv double %q_lower_vjp, %norm_lower_vjp",
            "  %vu0_vjp = fdiv double %b_vjp, %norm_upper_vjp",
            "  %vu1_vjp = fdiv double %q_upper_vjp, %norm_upper_vjp",
            "  %vl_eta_dot0 = fmul double %vl0_vjp, %cvl0_vjp",
            "  %vl_eta_dot1 = fmul double %vl1_vjp, %cvl1_vjp",
            "  %vl_eta_dot = fadd double %vl_eta_dot0, %vl_eta_dot1",
            "  %vl_eta_proj0 = fmul double %vl0_vjp, %vl_eta_dot",
            "  %vl_eta_proj1 = fmul double %vl1_vjp, %vl_eta_dot",
            "  %vl_gu0_num = fsub double %cvl0_vjp, %vl_eta_proj0",
            "  %vl_gu1_num = fsub double %cvl1_vjp, %vl_eta_proj1",
            "  %vl_gu0 = fdiv double %vl_gu0_num, %norm_lower_vjp",
            "  %vl_gu1 = fdiv double %vl_gu1_num, %norm_lower_vjp",
            "  %vu_eta_dot0 = fmul double %vu0_vjp, %cvu0_vjp",
            "  %vu_eta_dot1 = fmul double %vu1_vjp, %cvu1_vjp",
            "  %vu_eta_dot = fadd double %vu_eta_dot0, %vu_eta_dot1",
            "  %vu_eta_proj0 = fmul double %vu0_vjp, %vu_eta_dot",
            "  %vu_eta_proj1 = fmul double %vu1_vjp, %vu_eta_dot",
            "  %vu_gu0_num = fsub double %cvu0_vjp, %vu_eta_proj0",
            "  %vu_gu1_num = fsub double %cvu1_vjp, %vu_eta_proj1",
            "  %vu_gu0 = fdiv double %vu_gu0_num, %norm_upper_vjp",
            "  %vu_gu1 = fdiv double %vu_gu1_num, %norm_upper_vjp",
            "  %glambda_lower = fadd double %cl_vjp, %vl_gu1",
            "  %glambda_upper = fadd double %cu_vjp, %vu_gu1",
            "  %cotangent_sum = fadd double %glambda_lower, %glambda_upper",
            "  %alpha = fmul double 5.0e-1, %cotangent_sum",
            "  %cotangent_diff = fsub double %glambda_upper, %glambda_lower",
            "  %four_root_vjp = fmul double 4.0, %root_vjp",
            "  %beta = fdiv double %cotangent_diff, %four_root_vjp",
            "  %two_delta = fmul double 2.0, %delta_vjp",
            "  %a_disc_term = fmul double %two_delta, %beta",
            "  %adj_a_eig = fadd double %alpha, %a_disc_term",
            "  %adj_d_eig = fsub double %alpha, %a_disc_term",
            "  %four_c = fmul double 4.0, %c_vjp",
            "  %four_b = fmul double 4.0, %b_vjp",
            "  %adj_b_eig = fmul double %four_c, %beta",
            "  %adj_c_eig = fmul double %four_b, %beta",
            "  %gu1_sum = fadd double %vl_gu1, %vu_gu1",
            "  %adj_a_chart = fsub double %adj_a_eig, %gu1_sum",
            "  %gu0_sum = fadd double %vl_gu0, %vu_gu0",
            "  %adj_b_chart = fadd double %adj_b_eig, %gu0_sum",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %adj_a_chart, double* %out_vjp0",
            "  store double %adj_b_chart, double* %out_vjp1",
            "  store double %adj_c_eig, double* %out_vjp2",
            "  store double %adj_d_eig, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _call_native_matrix_2x2_determinant_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 4:
        raise ValueError("native 2x2 determinant LLVM/JIT kernel requires four matrix values")
    if output_size not in {1, 4}:
        raise ValueError("native 2x2 determinant LLVM/JIT output_size must be one or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_determinant_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    output_size: int,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 4:
        raise ValueError("native 2x2 determinant LLVM/JIT kernel requires four matrix values")
    expected_vector_size = 4 if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native 2x2 determinant LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, 4}:
        raise ValueError("native 2x2 determinant LLVM/JIT output_size must be one or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_inverse_values(
    label: str,
    values: Sequence[float] | FloatArray,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 4:
        raise ValueError("native 2x2 inverse LLVM/JIT kernel requires four matrix values")
    determinant = checked_values[0] * checked_values[3] - checked_values[1] * checked_values[2]
    if not np.isfinite(determinant) or abs(float(determinant)) <= 1.0e-12:
        raise ValueError("native 2x2 inverse LLVM/JIT kernel requires a nonsingular matrix")
    return checked_values


def _call_native_matrix_2x2_inverse_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_inverse_values("values", values)
    if output_size != 4:
        raise ValueError("native 2x2 inverse LLVM/JIT output_size must be four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_inverse_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_inverse_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_vector.size != 4:
        raise ValueError(f"native 2x2 inverse LLVM/JIT kernel requires four {label} value(s)")
    output = np.zeros(4, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_solve_values(
    label: str,
    values: Sequence[float] | FloatArray,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 6:
        raise ValueError(
            "native 2x2 solve LLVM/JIT kernel requires four matrix and two vector values"
        )
    determinant = checked_values[0] * checked_values[3] - checked_values[1] * checked_values[2]
    if not np.isfinite(determinant) or abs(float(determinant)) <= 1.0e-12:
        raise ValueError("native 2x2 solve LLVM/JIT kernel requires a nonsingular matrix")
    return checked_values


def _call_native_matrix_2x2_solve_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_solve_values("values", values)
    if output_size not in {2, 6}:
        raise ValueError("native 2x2 solve LLVM/JIT output_size must be two or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_solve_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_solve_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 6 if label == "tangent" else 2
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native 2x2 solve LLVM/JIT kernel requires {expected_vector_size} {label} value(s)"
        )
    if output_size not in {2, 6}:
        raise ValueError("native 2x2 solve LLVM/JIT output_size must be two or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_eigenvalues_values(
    label: str,
    values: Sequence[float] | FloatArray,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 4:
        raise ValueError(
            "native matrix 2x2 eigenvalue LLVM/JIT kernel requires row-major matrix values"
        )
    a00, a01, a10, a11 = checked_values
    discriminant = (a00 - a11) * (a00 - a11) + 4.0 * a01 * a10
    if not np.isfinite(discriminant) or float(discriminant) <= 1.0e-24:
        raise ValueError(
            "native matrix 2x2 eigenvalue LLVM/JIT kernel requires real distinct eigenvalues"
        )
    return checked_values


def _call_native_matrix_2x2_eigenvalues_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_eigenvalues_values("values", values)
    if output_size not in {2, 4}:
        raise ValueError("native matrix 2x2 eigenvalue LLVM/JIT output_size must be two or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_eigenvalues_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_eigenvalues_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 4 if label == "tangent" else 2
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native matrix 2x2 eigenvalue LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {2, 4}:
        raise ValueError("native matrix 2x2 eigenvalue LLVM/JIT output_size must be two or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_eigensystem_values(
    label: str,
    values: Sequence[float] | FloatArray,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_eigenvalues_values(label, values)
    if abs(float(checked_values[1])) <= 1.0e-12:
        raise ValueError(
            "native matrix 2x2 eigensystem LLVM/JIT kernel requires a non-zero "
            "upper off-diagonal eigenvector chart"
        )
    return checked_values


def _call_native_matrix_2x2_eigensystem_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_eigensystem_values("values", values)
    if output_size not in {4, 6}:
        raise ValueError("native matrix 2x2 eigensystem LLVM/JIT output_size must be four or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_eigensystem_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    output_size: int,
) -> FloatArray:
    checked_values = _as_native_matrix_2x2_eigensystem_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 4 if label == "tangent" else 6
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native matrix 2x2 eigensystem LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {4, 6}:
        raise ValueError("native matrix 2x2 eigensystem LLVM/JIT output_size must be four or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile exact 2x2 determinant value/JVP/VJP/gradient kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 determinant AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 4:
        raise ValueError("native 2x2 determinant AD requires four sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_determinant_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_determinant_unary(
            native_functions["value"],
            raw_values,
            1,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_determinant_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            1,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_determinant_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            4,
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
        native_gradient = _call_native_matrix_2x2_determinant_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT 2x2 determinant gradient verification failed")
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
            "verified native LLVM MCJIT 2x2 determinant value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for exact 2x2 determinant native LLVM/JIT kernels."""
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
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
            raise ValueError("native 2x2 determinant lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_determinant_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT 2x2 determinant contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native 2x2 determinant primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 4:
        raise ValueError("native 2x2 determinant primitive transform requires four sample values")
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:determinant;dimension:2;layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_determinant",
            "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 determinant JVP",
            "rust": "available: Rust PyO3 2x2 determinant value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_determinant "
                "value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_2x2_determinant_value,matrix_2x2_determinant_jvp,"
                "matrix_2x2_determinant_vjp,matrix_2x2_determinant_gradient"
            ),
            "llvm": "available: native LLVM MCJIT 2x2 determinant AD kernel",
            "jit": "available: native LLVM MCJIT 2x2 determinant AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT 2x2 determinant value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_2x2_determinant_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_polynomial_matrix_2x2_determinant",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="polynomial_matrix_2x2_determinant_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile exact nonsingular 2x2 inverse value/JVP/VJP kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 inverse AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_inverse_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_inverse_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_inverse_unary(
            native_functions["value"],
            raw_values,
            4,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_inverse_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_inverse_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
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
        native_sum_gradient = _call_native_matrix_2x2_inverse_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(4, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT 2x2 inverse sum-gradient provenance verification failed"
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
            "verified native LLVM MCJIT 2x2 inverse value/JVP/VJP kernel with "
            "sum-output gradient provenance; public gradient remains scalar-output "
            "fail-closed and singular matrices remain fail-closed"
        ),
    )


def make_matrix_2x2_inverse_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for exact nonsingular 2x2 inverse native LLVM/JIT kernels."""
    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_inverse_values("sample_values", sample_values)
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
            raise ValueError("native 2x2 inverse lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_inverse_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT nonsingular 2x2 inverse contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native 2x2 inverse primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_matrix_2x2_inverse_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:inverse;dimension:2;layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_2x2_inverse_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_inverse",
            "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 inverse JVP",
            "rust": (
                "available: Rust PyO3 nonsingular 2x2 inverse value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_inverse "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_2x2_inverse_value,matrix_2x2_inverse_jvp,"
                "matrix_2x2_inverse_vjp,matrix_2x2_inverse_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT 2x2 inverse AD kernel",
            "jit": "available: native LLVM MCJIT 2x2 inverse AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT 2x2 inverse value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_2x2_inverse_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "singular_matrix_2x2_inverse",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (4,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="nonsingular_matrix_2x2_inverse_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_solve_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile exact nonsingular 2x2 linear-solve value/JVP/VJP kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 solve AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_solve_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_solve_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_solve_unary(
            native_functions["value"],
            raw_values,
            2,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_solve_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            2,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_solve_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            6,
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
        native_sum_gradient = _call_native_matrix_2x2_solve_unary(
            native_functions["gradient"],
            values,
            6,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(2, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT 2x2 solve sum-gradient provenance verification failed"
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
            "verified native LLVM MCJIT 2x2 solve value/JVP/VJP kernel with "
            "sum-output gradient provenance; public gradient remains scalar-output "
            "fail-closed and singular matrices remain fail-closed"
        ),
    )


def make_matrix_2x2_solve_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for exact nonsingular 2x2 solve native LLVM/JIT kernels."""
    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_solve_values("sample_values", sample_values)
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
            raise ValueError("native 2x2 solve lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_solve_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_solve_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT nonsingular 2x2 solve contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 solve primitive transform requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_solve_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_solve_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:solve;dimension:2;layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_2x2_solve_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_solve",
            "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 solve JVP",
            "rust": (
                "available: Rust PyO3 nonsingular 2x2 solve value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_solve value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_2x2_solve_value,matrix_2x2_solve_jvp,"
                "matrix_2x2_solve_vjp,matrix_2x2_solve_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT 2x2 solve AD kernel",
            "jit": "available: native LLVM MCJIT 2x2 solve AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": "verified: native LLVM MCJIT 2x2 solve value/JVP/VJP",
            "static_derivative_factory": "native_matrix_2x2_solve_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "singular_matrix_2x2_solve",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (2,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="nonsingular_matrix_2x2_solve_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile real-simple nonsymmetric 2x2 eigenvalue value/JVP/VJP kernels."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix 2x2 eigenvalue AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_eigenvalues_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_eigenvalues_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_eigenvalues_unary(
            native_functions["value"],
            raw_values,
            2,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_eigenvalues_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            2,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_eigenvalues_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            4,
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
        native_sum_gradient = _call_native_matrix_2x2_eigenvalues_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(2, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT matrix 2x2 eigenvalue sum-gradient provenance verification failed"
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
            "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalues "
            "value/JVP/VJP kernel with sum-output gradient provenance; public "
            "gradient remains scalar-output fail-closed and complex or repeated "
            "eigenvalues remain fail-closed"
        ),
    )


def make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for real-simple nonsymmetric 2x2 eigenvalue kernels."""
    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_eigenvalues_values("sample_values", sample_values)
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
            raise ValueError("native matrix 2x2 eigenvalue lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT eigenvalue contract.

    The contract is intentionally narrow: row-major real nonsymmetric 2x2
    matrices whose spectra are real and distinct. Complex spectra and repeated
    eigenvalues remain fail-closed.
    """
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix 2x2 eigenvalue primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_matrix_2x2_eigenvalues_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_eigenvalues",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigenvalue JVP"
            ),
            "rust": (
                "available: Rust PyO3 real-simple nonsymmetric 2x2 eigenvalue "
                "value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_eigenvalues "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": (
                "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple"
            ),
            "rust_backend_functions": (
                "matrix_2x2_eigenvalues_value,matrix_2x2_eigenvalues_jvp,"
                "matrix_2x2_eigenvalues_vjp,matrix_2x2_eigenvalues_sum_gradient"
            ),
            "llvm": (
                "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel"
            ),
            "jit": (
                "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel"
            ),
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_2x2_eigenvalues_llvm_jit",
            "static_signature": (
                "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple"
            ),
            "nondifferentiable_boundary": "nonreal_or_repeated_matrix_2x2_eigenvalue",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (2,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="real_simple_matrix_2x2_eigenvalues_domain",
        effect="pure",
    )


def compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP kernels."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix 2x2 eigensystem AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_eigensystem_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_eigensystem_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_eigensystem_unary(
            native_functions["value"],
            raw_values,
            6,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_eigensystem_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            6,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_2x2_eigensystem_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            4,
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
        native_sum_gradient = _call_native_matrix_2x2_eigensystem_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(6, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT matrix 2x2 eigensystem sum-gradient provenance verification failed"
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
            "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem "
            "value/JVP/VJP kernel with sum-output gradient provenance; complex "
            "spectra, repeated eigenvalues, and zero upper off-diagonal eigenvector "
            "charts remain fail-closed"
        ),
    )


def make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for real-simple nonsymmetric 2x2 eigensystem kernels."""
    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_eigensystem_values("sample_values", sample_values)
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
            raise ValueError("native matrix 2x2 eigensystem lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT eigensystem contract.

    The contract is intentionally narrow: row-major real nonsymmetric 2x2
    matrices whose spectra are real and distinct and whose eigenvector chart
    uses a non-zero upper off-diagonal entry. All other eigensystem domains
    remain fail-closed.
    """
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix 2x2 eigensystem primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_matrix_2x2_eigensystem_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_eigensystem",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigensystem JVP"
            ),
            "rust": (
                "available: Rust PyO3 real-simple nonsymmetric 2x2 eigensystem "
                "value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_eigensystem "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": (
                "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
            ),
            "rust_backend_functions": (
                "matrix_2x2_eigensystem_value,matrix_2x2_eigensystem_jvp,"
                "matrix_2x2_eigensystem_vjp,matrix_2x2_eigensystem_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
            "jit": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_2x2_eigensystem_llvm_jit",
            "static_signature": (
                "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
            ),
            "nondifferentiable_boundary": (
                "nonreal_repeated_or_zero_upper_chart_matrix_2x2_eigensystem"
            ),
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (6,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="real_simple_upper_chart_matrix_2x2_eigensystem_domain",
        effect="pure",
    )
