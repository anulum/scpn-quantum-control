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
