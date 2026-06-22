# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- matrix native LLVM/JIT AD compilation for the MLIR surface
"""Native LLVM/JIT autodiff compilation for matrix primitives.

Emits the LLVM IR and builds the executable native-JIT kernels, lowering rules and
primitive transforms that evaluate each matrix primitive together with its forward
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


def _validate_matrix_quadratic_form_dimension(dimension: int | np.integer) -> int:
    checked = int(dimension)
    if checked < 1:
        raise ValueError("native matrix quadratic form dimension must be positive")
    return checked


def _matrix_quadratic_form_value_count(dimension: int) -> int:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    return checked_dimension * checked_dimension + checked_dimension


def _matrix_quadratic_form_matrix_index(dimension: int, row: int, column: int) -> int:
    return row * dimension + column


def _matrix_quadratic_form_vector_index(dimension: int, index: int) -> int:
    return dimension * dimension + index


def _compile_matrix_quadratic_form_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_quadratic_form"',
        '; source = "native_matrix_quadratic_form_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_value_sum = "0.0"
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_value{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %xptr_left_value{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %xptr_right_value{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_value{term} = load double, double* %aptr_value{term}",
                    f"  %x_left_value{term} = load double, double* %xptr_left_value{term}",
                    f"  %x_right_value{term} = load double, double* %xptr_right_value{term}",
                    f"  %left_value{term} = fmul double %a_value{term}, %x_left_value{term}",
                    f"  %term_value{term} = fmul double %left_value{term}, %x_right_value{term}",
                    f"  %sum_value{term} = fadd double {previous_value_sum}, %term_value{term}",
                ]
            )
            previous_value_sum = f"%sum_value{term}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_value_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_matrix_left{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %xptr_matrix_right{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %x_matrix_left{term} = load double, double* %xptr_matrix_left{term}",
                    f"  %x_matrix_right{term} = load double, double* %xptr_matrix_right{term}",
                    f"  %grad_matrix{term} = fmul double %x_matrix_left{term}, %x_matrix_right{term}",
                    f"  %out_matrix{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %grad_matrix{term}, double* %out_matrix{term}",
                ]
            )
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        previous_column_sum = "0.0"
        for column in range(checked_dimension):
            row_matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_matrix_index = _matrix_quadratic_form_matrix_index(
                checked_dimension, column, row
            )
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %a_row_ptr_grad{term} = getelementptr double, double* %values, i64 {row_matrix_index}",
                    f"  %a_col_ptr_grad{term} = getelementptr double, double* %values, i64 {column_matrix_index}",
                    f"  %x_ptr_grad{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_row_grad{term} = load double, double* %a_row_ptr_grad{term}",
                    f"  %a_col_grad{term} = load double, double* %a_col_ptr_grad{term}",
                    f"  %x_grad{term} = load double, double* %x_ptr_grad{term}",
                    f"  %row_term_grad{term} = fmul double %a_row_grad{term}, %x_grad{term}",
                    f"  %column_term_grad{term} = fmul double %a_col_grad{term}, %x_grad{term}",
                    f"  %row_sum_grad{term} = fadd double {previous_row_sum}, %row_term_grad{term}",
                    f"  %column_sum_grad{term} = fadd double {previous_column_sum}, %column_term_grad{term}",
                ]
            )
            previous_row_sum = f"%row_sum_grad{term}"
            previous_column_sum = f"%column_sum_grad{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        lines.extend(
            [
                f"  %grad_vector{row} = fadd double {previous_row_sum}, {previous_column_sum}",
                f"  %out_vector{row} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double %grad_vector{row}, double* %out_vector{row}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    lines.extend(
        [
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %da_ptr_jvp{term} = getelementptr double, double* %tangent, i64 {matrix_index}",
                    f"  %x_left_ptr_jvp{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %x_right_ptr_jvp{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %da_jvp{term} = load double, double* %da_ptr_jvp{term}",
                    f"  %x_left_jvp{term} = load double, double* %x_left_ptr_jvp{term}",
                    f"  %x_right_jvp{term} = load double, double* %x_right_ptr_jvp{term}",
                    f"  %matrix_left_jvp{term} = fmul double %da_jvp{term}, %x_left_jvp{term}",
                    f"  %matrix_term_jvp{term} = fmul double %matrix_left_jvp{term}, %x_right_jvp{term}",
                    f"  %matrix_sum_jvp{term} = fadd double {previous_jvp_sum}, %matrix_term_jvp{term}",
                ]
            )
            previous_jvp_sum = f"%matrix_sum_jvp{term}"
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        previous_column_sum = "0.0"
        for column in range(checked_dimension):
            row_matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_matrix_index = _matrix_quadratic_form_matrix_index(
                checked_dimension, column, row
            )
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %a_row_ptr_jvp{term} = getelementptr double, double* %values, i64 {row_matrix_index}",
                    f"  %a_col_ptr_jvp{term} = getelementptr double, double* %values, i64 {column_matrix_index}",
                    f"  %x_ptr_jvp{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_row_jvp{term} = load double, double* %a_row_ptr_jvp{term}",
                    f"  %a_col_jvp{term} = load double, double* %a_col_ptr_jvp{term}",
                    f"  %x_grad_jvp{term} = load double, double* %x_ptr_jvp{term}",
                    f"  %row_term_jvp{term} = fmul double %a_row_jvp{term}, %x_grad_jvp{term}",
                    f"  %column_term_jvp{term} = fmul double %a_col_jvp{term}, %x_grad_jvp{term}",
                    f"  %row_sum_vector_jvp{term} = fadd double {previous_row_sum}, %row_term_jvp{term}",
                    f"  %column_sum_vector_jvp{term} = fadd double {previous_column_sum}, %column_term_jvp{term}",
                ]
            )
            previous_row_sum = f"%row_sum_vector_jvp{term}"
            previous_column_sum = f"%column_sum_vector_jvp{term}"
        vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        lines.extend(
            [
                f"  %grad_vector_jvp{row} = fadd double {previous_row_sum}, {previous_column_sum}",
                f"  %dx_ptr_jvp{row} = getelementptr double, double* %tangent, i64 {vector_index}",
                f"  %dx_jvp{row} = load double, double* %dx_ptr_jvp{row}",
                f"  %vector_term_jvp{row} = fmul double %grad_vector_jvp{row}, %dx_jvp{row}",
                f"  %vector_sum_jvp{row} = fadd double {previous_jvp_sum}, %vector_term_jvp{row}",
            ]
        )
        previous_jvp_sum = f"%vector_sum_jvp{row}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_vjp_left{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %xptr_vjp_right{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %x_vjp_left{term} = load double, double* %xptr_vjp_left{term}",
                    f"  %x_vjp_right{term} = load double, double* %xptr_vjp_right{term}",
                    f"  %grad_matrix_vjp{term} = fmul double %x_vjp_left{term}, %x_vjp_right{term}",
                    f"  %vjp_matrix{term} = fmul double %grad_matrix_vjp{term}, %cotangent0",
                    f"  %out_matrix_vjp{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %vjp_matrix{term}, double* %out_matrix_vjp{term}",
                ]
            )
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        previous_column_sum = "0.0"
        for column in range(checked_dimension):
            row_matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_matrix_index = _matrix_quadratic_form_matrix_index(
                checked_dimension, column, row
            )
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %a_row_ptr_vjp{term} = getelementptr double, double* %values, i64 {row_matrix_index}",
                    f"  %a_col_ptr_vjp{term} = getelementptr double, double* %values, i64 {column_matrix_index}",
                    f"  %x_ptr_vjp{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_row_vjp{term} = load double, double* %a_row_ptr_vjp{term}",
                    f"  %a_col_vjp{term} = load double, double* %a_col_ptr_vjp{term}",
                    f"  %x_grad_vjp{term} = load double, double* %x_ptr_vjp{term}",
                    f"  %row_term_vjp{term} = fmul double %a_row_vjp{term}, %x_grad_vjp{term}",
                    f"  %column_term_vjp{term} = fmul double %a_col_vjp{term}, %x_grad_vjp{term}",
                    f"  %row_sum_vjp{term} = fadd double {previous_row_sum}, %row_term_vjp{term}",
                    f"  %column_sum_vjp{term} = fadd double {previous_column_sum}, %column_term_vjp{term}",
                ]
            )
            previous_row_sum = f"%row_sum_vjp{term}"
            previous_column_sum = f"%column_sum_vjp{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        lines.extend(
            [
                f"  %grad_vector_vjp{row} = fadd double {previous_row_sum}, {previous_column_sum}",
                f"  %vjp_vector{row} = fmul double %grad_vector_vjp{row}, %cotangent0",
                f"  %out_vector_vjp{row} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double %vjp_vector{row}, double* %out_vector_vjp{row}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_vector_product_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    value_count = _matrix_quadratic_form_value_count(checked_dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_vector_product"',
        '; source = "native_matrix_vector_product_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {value_count}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_value{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %xptr_value{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %a_value{term} = load double, double* %aptr_value{term}",
                    f"  %x_value{term} = load double, double* %xptr_value{term}",
                    f"  %prod_value{term} = fmul double %a_value{term}, %x_value{term}",
                    f"  %sum_value{term} = fadd double {previous_row_sum}, %prod_value{term}",
                ]
            )
            previous_row_sum = f"%sum_value{term}"
        lines.extend(
            [
                f"  %out_value{row} = getelementptr double, double* %out, i64 {row}",
                f"  store double {previous_row_sum}, double* %out_value{row}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_gradient{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %x_gradient{term} = load double, double* %xptr_gradient{term}",
                    f"  %out_matrix_gradient{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %x_gradient{term}, double* %out_matrix_gradient{term}",
                ]
            )
    for column in range(checked_dimension):
        previous_column_sum = "0.0"
        for row in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_gradient{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %a_gradient{term} = load double, double* %aptr_gradient{term}",
                    f"  %sum_gradient{term} = fadd double {previous_column_sum}, %a_gradient{term}",
                ]
            )
            previous_column_sum = f"%sum_gradient{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
        lines.extend(
            [
                f"  %out_vector_gradient{column} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double {previous_column_sum}, double* %out_vector_gradient{column}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        previous_jvp_sum = "0.0"
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_jvp{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %xptr_jvp{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %taptr_jvp{term} = getelementptr double, double* %tangent, i64 {matrix_index}",
                    f"  %txptr_jvp{term} = getelementptr double, double* %tangent, i64 {vector_index}",
                    f"  %a_jvp{term} = load double, double* %aptr_jvp{term}",
                    f"  %x_jvp{term} = load double, double* %xptr_jvp{term}",
                    f"  %ta_jvp{term} = load double, double* %taptr_jvp{term}",
                    f"  %tx_jvp{term} = load double, double* %txptr_jvp{term}",
                    f"  %left_jvp{term} = fmul double %ta_jvp{term}, %x_jvp{term}",
                    f"  %right_jvp{term} = fmul double %a_jvp{term}, %tx_jvp{term}",
                    f"  %term_jvp{term} = fadd double %left_jvp{term}, %right_jvp{term}",
                    f"  %sum_jvp{term} = fadd double {previous_jvp_sum}, %term_jvp{term}",
                ]
            )
            previous_jvp_sum = f"%sum_jvp{term}"
        lines.extend(
            [
                f"  %out_jvp{row} = getelementptr double, double* %out, i64 {row}",
                f"  store double {previous_jvp_sum}, double* %out_jvp{row}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        cotangent_ptr = f"%cotangent_ptr{row}"
        cotangent_value = f"%cotangent{row}"
        lines.extend(
            [
                f"  {cotangent_ptr} = getelementptr double, double* %cotangent, i64 {row}",
                f"  {cotangent_value} = load double, double* {cotangent_ptr}",
            ]
        )
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_vjp{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %x_vjp{term} = load double, double* %xptr_vjp{term}",
                    f"  %matrix_grad_vjp{term} = fmul double {cotangent_value}, %x_vjp{term}",
                    f"  %out_matrix_vjp{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %matrix_grad_vjp{term}, double* %out_matrix_vjp{term}",
                ]
            )
    for column in range(checked_dimension):
        previous_vector_sum = "0.0"
        for row in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_vjp{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %cotangent_ptr_vjp{term} = getelementptr double, double* %cotangent, i64 {row}",
                    f"  %a_vjp{term} = load double, double* %aptr_vjp{term}",
                    f"  %cotangent_vjp{term} = load double, double* %cotangent_ptr_vjp{term}",
                    f"  %vector_term_vjp{term} = fmul double %a_vjp{term}, %cotangent_vjp{term}",
                    f"  %vector_sum_vjp{term} = fadd double {previous_vector_sum}, %vector_term_vjp{term}",
                ]
            )
            previous_vector_sum = f"%vector_sum_vjp{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
        lines.extend(
            [
                f"  %out_vector_vjp{column} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double {previous_vector_sum}, double* %out_vector_vjp{column}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_matrix_product_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    value_count = 2 * matrix_size
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_matrix_product"',
        '; source = "native_matrix_matrix_product_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {value_count}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            previous_sum = "0.0"
            for inner in range(checked_dimension):
                left_index = row * checked_dimension + inner
                right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{column}_{inner}"
                lines.extend(
                    [
                        f"  %leftptr_value{term} = getelementptr double, double* %values, i64 {left_index}",
                        f"  %rightptr_value{term} = getelementptr double, double* %values, i64 {right_index}",
                        f"  %left_value{term} = load double, double* %leftptr_value{term}",
                        f"  %right_value{term} = load double, double* %rightptr_value{term}",
                        f"  %prod_value{term} = fmul double %left_value{term}, %right_value{term}",
                        f"  %sum_value{term} = fadd double {previous_sum}, %prod_value{term}",
                    ]
                )
                previous_sum = f"%sum_value{term}"
            output_index = row * checked_dimension + column
            lines.extend(
                [
                    f"  %out_value{row}_{column} = getelementptr double, double* %out, i64 {output_index}",
                    f"  store double {previous_sum}, double* %out_value{row}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for inner in range(checked_dimension):
            previous_left_sum = "0.0"
            for column in range(checked_dimension):
                left_right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{inner}_{column}"
                lines.extend(
                    [
                        f"  %rightptr_gradient{term} = getelementptr double, double* %values, i64 {left_right_index}",
                        f"  %right_gradient{term} = load double, double* %rightptr_gradient{term}",
                        f"  %left_sum_gradient{term} = fadd double {previous_left_sum}, %right_gradient{term}",
                    ]
                )
                previous_left_sum = f"%left_sum_gradient{term}"
            left_output_index = row * checked_dimension + inner
            lines.extend(
                [
                    f"  %out_left_gradient{row}_{inner} = getelementptr double, double* %out, i64 {left_output_index}",
                    f"  store double {previous_left_sum}, double* %out_left_gradient{row}_{inner}",
                ]
            )
    for inner in range(checked_dimension):
        for column in range(checked_dimension):
            previous_right_sum = "0.0"
            for row in range(checked_dimension):
                right_left_index = row * checked_dimension + inner
                term = f"{inner}_{column}_{row}"
                lines.extend(
                    [
                        f"  %leftptr_gradient{term} = getelementptr double, double* %values, i64 {right_left_index}",
                        f"  %left_gradient{term} = load double, double* %leftptr_gradient{term}",
                        f"  %right_sum_gradient{term} = fadd double {previous_right_sum}, %left_gradient{term}",
                    ]
                )
                previous_right_sum = f"%right_sum_gradient{term}"
            right_output_index = matrix_size + inner * checked_dimension + column
            lines.extend(
                [
                    f"  %out_right_gradient{inner}_{column} = getelementptr double, double* %out, i64 {right_output_index}",
                    f"  store double {previous_right_sum}, double* %out_right_gradient{inner}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            previous_jvp_sum = "0.0"
            for inner in range(checked_dimension):
                left_index = row * checked_dimension + inner
                right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{column}_{inner}"
                lines.extend(
                    [
                        f"  %leftptr_jvp{term} = getelementptr double, double* %values, i64 {left_index}",
                        f"  %rightptr_jvp{term} = getelementptr double, double* %values, i64 {right_index}",
                        f"  %tleftptr_jvp{term} = getelementptr double, double* %tangent, i64 {left_index}",
                        f"  %trightptr_jvp{term} = getelementptr double, double* %tangent, i64 {right_index}",
                        f"  %left_jvp{term} = load double, double* %leftptr_jvp{term}",
                        f"  %right_jvp{term} = load double, double* %rightptr_jvp{term}",
                        f"  %tleft_jvp{term} = load double, double* %tleftptr_jvp{term}",
                        f"  %tright_jvp{term} = load double, double* %trightptr_jvp{term}",
                        f"  %left_term_jvp{term} = fmul double %tleft_jvp{term}, %right_jvp{term}",
                        f"  %right_term_jvp{term} = fmul double %left_jvp{term}, %tright_jvp{term}",
                        f"  %term_jvp{term} = fadd double %left_term_jvp{term}, %right_term_jvp{term}",
                        f"  %sum_jvp{term} = fadd double {previous_jvp_sum}, %term_jvp{term}",
                    ]
                )
                previous_jvp_sum = f"%sum_jvp{term}"
            output_index = row * checked_dimension + column
            lines.extend(
                [
                    f"  %out_jvp{row}_{column} = getelementptr double, double* %out, i64 {output_index}",
                    f"  store double {previous_jvp_sum}, double* %out_jvp{row}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for inner in range(checked_dimension):
            previous_left_sum = "0.0"
            for column in range(checked_dimension):
                cotangent_index = row * checked_dimension + column
                right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{inner}_{column}"
                lines.extend(
                    [
                        f"  %cotangent_left_ptr_vjp{term} = getelementptr double, double* %cotangent, i64 {cotangent_index}",
                        f"  %rightptr_vjp{term} = getelementptr double, double* %values, i64 {right_index}",
                        f"  %cotangent_left_vjp{term} = load double, double* %cotangent_left_ptr_vjp{term}",
                        f"  %right_vjp{term} = load double, double* %rightptr_vjp{term}",
                        f"  %left_term_vjp{term} = fmul double %cotangent_left_vjp{term}, %right_vjp{term}",
                        f"  %left_sum_vjp{term} = fadd double {previous_left_sum}, %left_term_vjp{term}",
                    ]
                )
                previous_left_sum = f"%left_sum_vjp{term}"
            left_output_index = row * checked_dimension + inner
            lines.extend(
                [
                    f"  %out_left_vjp{row}_{inner} = getelementptr double, double* %out, i64 {left_output_index}",
                    f"  store double {previous_left_sum}, double* %out_left_vjp{row}_{inner}",
                ]
            )
    for inner in range(checked_dimension):
        for column in range(checked_dimension):
            previous_right_sum = "0.0"
            for row in range(checked_dimension):
                left_index = row * checked_dimension + inner
                cotangent_index = row * checked_dimension + column
                term = f"{inner}_{column}_{row}"
                lines.extend(
                    [
                        f"  %leftptr_vjp{term} = getelementptr double, double* %values, i64 {left_index}",
                        f"  %cotangent_right_ptr_vjp{term} = getelementptr double, double* %cotangent, i64 {cotangent_index}",
                        f"  %left_vjp{term} = load double, double* %leftptr_vjp{term}",
                        f"  %cotangent_right_vjp{term} = load double, double* %cotangent_right_ptr_vjp{term}",
                        f"  %right_term_vjp{term} = fmul double %left_vjp{term}, %cotangent_right_vjp{term}",
                        f"  %right_sum_vjp{term} = fadd double {previous_right_sum}, %right_term_vjp{term}",
                    ]
                )
                previous_right_sum = f"%right_sum_vjp{term}"
            right_output_index = matrix_size + inner * checked_dimension + column
            lines.extend(
                [
                    f"  %out_right_vjp{inner}_{column} = getelementptr double, double* %out, i64 {right_output_index}",
                    f"  store double {previous_right_sum}, double* %out_right_vjp{inner}_{column}",
                ]
            )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_trace_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_trace"',
        '; source = "native_matrix_trace_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {matrix_size}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_sum = "0.0"
    for index in range(checked_dimension):
        diagonal_index = index * checked_dimension + index
        lines.extend(
            [
                f"  %diagptr_value{index} = getelementptr double, double* %values, i64 {diagonal_index}",
                f"  %diag_value{index} = load double, double* %diagptr_value{index}",
                f"  %sum_value{index} = fadd double {previous_sum}, %diag_value{index}",
            ]
        )
        previous_sum = f"%sum_value{index}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            matrix_index = row * checked_dimension + column
            value = "1.0" if row == column else "0.0"
            lines.extend(
                [
                    f"  %out_gradient{row}_{column} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double {value}, double* %out_gradient{row}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for index in range(checked_dimension):
        diagonal_index = index * checked_dimension + index
        lines.extend(
            [
                f"  %diagptr_jvp{index} = getelementptr double, double* %tangent, i64 {diagonal_index}",
                f"  %diag_jvp{index} = load double, double* %diagptr_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %diag_jvp{index}",
            ]
        )
        previous_jvp_sum = f"%sum_jvp{index}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            matrix_index = row * checked_dimension + column
            value = "%cotangent0" if row == column else "0.0"
            lines.extend(
                [
                    f"  %out_vjp{row}_{column} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double {value}, double* %out_vjp{row}_{column}",
                ]
            )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_frobenius_norm_squared_native_llvm_ir(
    rule_name: str,
    dimension: int,
) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_frobenius_norm_squared"',
        '; source = "native_matrix_frobenius_norm_squared_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {matrix_size}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_value_sum = "0.0"
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_value{index} = getelementptr double, double* %values, i64 {index}",
                f"  %value_value{index} = load double, double* %valueptr_value{index}",
                f"  %square_value{index} = fmul double %value_value{index}, %value_value{index}",
                f"  %sum_value{index} = fadd double {previous_value_sum}, %square_value{index}",
            ]
        )
        previous_value_sum = f"%sum_value{index}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_value_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_gradient{index} = getelementptr double, double* %values, i64 {index}",
                f"  %value_gradient{index} = load double, double* %valueptr_gradient{index}",
                f"  %gradient{index} = fmul double 2.0, %value_gradient{index}",
                f"  %out_gradient{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %gradient{index}, double* %out_gradient{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_jvp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %tangentptr_jvp{index} = getelementptr double, double* %tangent, i64 {index}",
                f"  %value_jvp{index} = load double, double* %valueptr_jvp{index}",
                f"  %tangent_jvp{index} = load double, double* %tangentptr_jvp{index}",
                f"  %product_jvp{index} = fmul double %value_jvp{index}, %tangent_jvp{index}",
                f"  %scaled_jvp{index} = fmul double 2.0, %product_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %scaled_jvp{index}",
            ]
        )
        previous_jvp_sum = f"%sum_jvp{index}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_vjp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %value_vjp{index} = load double, double* %valueptr_vjp{index}",
                f"  %gradient_vjp{index} = fmul double 2.0, %value_vjp{index}",
                f"  %scaled_vjp{index} = fmul double %cotangent0, %gradient_vjp{index}",
                f"  %out_vjp{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %scaled_vjp{index}, double* %out_vjp{index}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


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


def _call_native_matrix_quadratic_form_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    if output_size not in {1, expected_value_count}:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT output_size must be one or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_quadratic_form_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    expected_vector_size = expected_value_count if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix quadratic form LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, expected_value_count}:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT output_size must be one or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_vector_product_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    if output_size not in {checked_dimension, expected_value_count}:
        raise ValueError(
            "native matrix-vector product LLVM/JIT output_size must be dimension or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_vector_product_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    expected_vector_size = expected_value_count if label == "tangent" else checked_dimension
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix-vector product LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {checked_dimension, expected_value_count}:
        raise ValueError(
            "native matrix-vector product LLVM/JIT output_size must be dimension or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_matrix_product_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT kernel requires "
            "2 * dimension * dimension values"
        )
    if output_size not in {matrix_size, expected_value_count}:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT output_size must be matrix-sized or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_matrix_product_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT kernel requires "
            "2 * dimension * dimension values"
        )
    expected_vector_size = expected_value_count if label == "tangent" else matrix_size
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix-matrix product LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {matrix_size, expected_value_count}:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT output_size must be matrix-sized or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_trace_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix trace LLVM/JIT kernel requires dimension * dimension values"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError("native matrix trace LLVM/JIT output_size must be one or matrix-sized")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_trace_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix trace LLVM/JIT kernel requires dimension * dimension values"
        )
    expected_vector_size = matrix_size if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix trace LLVM/JIT kernel requires {expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError("native matrix trace LLVM/JIT output_size must be one or matrix-sized")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_frobenius_norm_squared_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT kernel requires dimension * dimension values"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT output_size must be one or matrix-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_frobenius_norm_squared_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT kernel requires dimension * dimension values"
        )
    expected_vector_size = matrix_size if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT output_size must be one or matrix-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


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


def compile_matrix_vector_product_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix-vector value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix-vector product AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product AD requires dimension * dimension + dimension "
            "sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_vector_product_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_vector_product_unary(
            native_functions["value"], raw_values, checked_dimension, checked_dimension
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_vector_product_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            checked_dimension,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_vector_product_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            expected_value_count,
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
        native_gradient = _call_native_matrix_vector_product_unary(
            native_functions["gradient"], values, checked_dimension, expected_value_count
        )
        reference_gradient = vjp_kernel(values, np.ones(checked_dimension, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix-vector product gradient verification failed")
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
            "verified native LLVM MCJIT matrix-vector product value/JVP/VJP kernel; "
            "gradient() remains fail-closed for vector-output kernels"
        ),
    )


def make_matrix_vector_product_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix-vector native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
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
            raise ValueError("native matrix-vector product lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_vector_product_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_vector_product_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT matrix-vector contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix-vector product primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product primitive transform requires "
            "dimension * dimension + dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_vector_product_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:matvec;dimension:{checked_dimension};layout:matrix_then_vector"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_vector_product_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_vector_product",
            "mlir_runtime_verification": "verified: native LLVM/JIT matrix-vector JVP",
            "rust": (
                "available: Rust PyO3 matrix-vector product value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_vector_product "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_vector_product_value,matrix_vector_product_jvp,"
                "matrix_vector_product_vjp,matrix_vector_product_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix-vector AD kernel",
            "jit": "available: native LLVM MCJIT matrix-vector AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix-vector value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_vector_product_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_vector_product",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (checked_dimension,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_vector_product_real_domain",
        effect="pure",
    )


def compile_matrix_matrix_product_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix-matrix value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix-matrix product AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product AD requires 2 * dimension * dimension sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_matrix_product_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_matrix_product_unary(
            native_functions["value"], raw_values, checked_dimension, matrix_size
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_matrix_product_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            matrix_size,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_matrix_product_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            expected_value_count,
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
        native_gradient = _call_native_matrix_matrix_product_unary(
            native_functions["gradient"], values, checked_dimension, expected_value_count
        )
        reference_gradient = vjp_kernel(values, np.ones(matrix_size, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix-matrix product gradient verification failed")
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
            "verified native LLVM MCJIT matrix-matrix product value/JVP/VJP kernel; "
            "gradient() remains fail-closed for matrix-output kernels"
        ),
    )


def make_matrix_matrix_product_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix-matrix native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
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
            raise ValueError("native matrix-matrix product lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_matrix_product_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_matrix_product_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT matrix-matrix contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix-matrix product primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product primitive transform requires "
            "2 * dimension * dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_matrix_product_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:matmul;dimension:{checked_dimension};layout:left_then_right"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_matrix_product_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_matrix_product",
            "mlir_runtime_verification": "verified: native LLVM/JIT matrix-matrix JVP",
            "rust": (
                "available: Rust PyO3 matrix-matrix product value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_matrix_product "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_matrix_product_value,matrix_matrix_product_jvp,"
                "matrix_matrix_product_vjp,matrix_matrix_product_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix-matrix AD kernel",
            "jit": "available: native LLVM MCJIT matrix-matrix AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix-matrix value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_matrix_product_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_matrix_product",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (checked_dimension, checked_dimension),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_matrix_product_real_domain",
        effect="pure",
    )


def compile_matrix_trace_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix trace value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix trace AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError("native matrix trace AD requires dimension * dimension sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_trace_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_trace_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_trace_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_trace_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            matrix_size,
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
        native_gradient = _call_native_matrix_trace_unary(
            native_functions["gradient"], values, checked_dimension, matrix_size
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix trace gradient verification failed")
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
            "verified native LLVM MCJIT matrix trace value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_trace_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix trace native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
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
            raise ValueError("native matrix trace lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_trace_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_trace_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT matrix-trace contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix trace primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError(
            "native matrix trace primitive transform requires dimension * dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_trace_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:trace;dimension:{checked_dimension};layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_trace_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_trace",
            "mlir_runtime_verification": "verified: native LLVM/JIT matrix trace JVP",
            "rust": "available: Rust PyO3 matrix trace value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_trace value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_trace_value,matrix_trace_jvp,matrix_trace_vjp,matrix_trace_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix trace AD kernel",
            "jit": "available: native LLVM MCJIT matrix trace AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix trace value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_trace_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_trace",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_trace_real_domain",
        effect="pure",
    )


def compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix Frobenius-squared value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix Frobenius-squared AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared AD requires dimension * dimension sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_frobenius_norm_squared_native_llvm_ir(
        rule.name,
        checked_dimension,
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_frobenius_norm_squared_unary(
            native_functions["value"],
            raw_values,
            checked_dimension,
            1,
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_frobenius_norm_squared_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_frobenius_norm_squared_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            matrix_size,
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
        native_gradient = _call_native_matrix_frobenius_norm_squared_unary(
            native_functions["gradient"],
            values,
            checked_dimension,
            matrix_size,
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT matrix Frobenius-squared gradient verification failed"
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
            "verified native LLVM MCJIT matrix Frobenius-squared value/JVP/VJP/gradient "
            "kernel; unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix Frobenius-squared native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
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
            raise ValueError("native matrix Frobenius-squared lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT Frobenius-squared contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix Frobenius-squared primitive transform requires "
            "backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared primitive transform requires dimension * "
            "dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = (
        f"primitive:frobenius_norm_squared;dimension:{checked_dimension};layout:row_major"
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_frobenius_norm_squared",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT matrix Frobenius-squared JVP"
            ),
            "rust": (
                "available: Rust PyO3 matrix Frobenius-squared value/JVP/VJP/gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_frobenius_norm_squared "
                "value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_frobenius_norm_squared_value,matrix_frobenius_norm_squared_jvp,"
                "matrix_frobenius_norm_squared_vjp,matrix_frobenius_norm_squared_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
            "jit": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix Frobenius-squared value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_frobenius_norm_squared_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_frobenius_norm_squared",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_frobenius_norm_squared_real_domain",
        effect="pure",
    )


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


def compile_matrix_quadratic_form_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix quadratic-form value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix quadratic form AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form AD requires dimension * dimension + dimension "
            "sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_quadratic_form_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_matrix_quadratic_form_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_matrix_quadratic_form_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_matrix_quadratic_form_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            expected_value_count,
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
        native_gradient = _call_native_matrix_quadratic_form_unary(
            native_functions["gradient"], values, checked_dimension, expected_value_count
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix quadratic form gradient verification failed")
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
            "verified native LLVM MCJIT matrix quadratic form "
            "value/JVP/VJP/gradient kernel; unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_quadratic_form_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix quadratic-form native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
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
            raise ValueError("native matrix quadratic form lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_quadratic_form_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_quadratic_form_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT quadratic-form contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix quadratic form primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form primitive transform requires dimension * "
            "dimension + dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_quadratic_form_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = (
        f"primitive:quadratic_form;dimension:{checked_dimension};layout:matrix_then_vector"
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_quadratic_form_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_quadratic_form",
            "mlir_runtime_verification": ("verified: native LLVM/JIT matrix quadratic form JVP"),
            "rust": "available: Rust PyO3 matrix quadratic form value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_quadratic_form value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_quadratic_form_value,matrix_quadratic_form_jvp,"
                "matrix_quadratic_form_vjp,matrix_quadratic_form_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix quadratic form AD kernel",
            "jit": "available: native LLVM MCJIT matrix quadratic form AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix quadratic form value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_quadratic_form_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_quadratic_form",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_quadratic_form_real_domain",
        effect="pure",
    )
