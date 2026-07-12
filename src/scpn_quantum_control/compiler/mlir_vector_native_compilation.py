# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR vector native compilation module
# scpn-quantum-control -- vector native LLVM/JIT AD compilation for the MLIR surface
"""Native LLVM/JIT autodiff compilation for vector primitives.

Emits the LLVM IR and builds the executable native-JIT kernels, lowering rules and
primitive transforms that evaluate each vector primitive together with its forward
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


def _validate_vector_dot_dimension(dimension: int | np.integer) -> int:
    checked = int(dimension)
    if checked < 1:
        raise ValueError("native vector dot dimension must be positive")
    return checked


def _compile_vector_dot_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "dot"',
        '; source = "native_vector_dot_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_sum = "0.0"
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %x{index} = load double, double* %xptr{index}",
                f"  %y{index} = load double, double* %yptr{index}",
                f"  %prod{index} = fmul double %x{index}, %y{index}",
                f"  %sum{index} = fadd double {previous_sum}, %prod{index}",
            ]
        )
        previous_sum = f"%sum{index}"
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
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %x{index} = load double, double* %xptr{index}",
                f"  %y{index} = load double, double* %yptr{index}",
                f"  %outxptr{index} = getelementptr double, double* %out, i64 {index}",
                f"  %outyptr{index} = getelementptr double, double* %out, i64 {right_index}",
                f"  store double %y{index}, double* %outxptr{index}",
                f"  store double %x{index}, double* %outyptr{index}",
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
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr_jvp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr_jvp{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %txptr{index} = getelementptr double, double* %tangent, i64 {index}",
                f"  %typtr{index} = getelementptr double, double* %tangent, i64 {right_index}",
                f"  %x_jvp{index} = load double, double* %xptr_jvp{index}",
                f"  %y_jvp{index} = load double, double* %yptr_jvp{index}",
                f"  %tx{index} = load double, double* %txptr{index}",
                f"  %ty{index} = load double, double* %typtr{index}",
                f"  %left_jvp{index} = fmul double %y_jvp{index}, %tx{index}",
                f"  %right_jvp{index} = fmul double %x_jvp{index}, %ty{index}",
                f"  %term_jvp{index} = fadd double %left_jvp{index}, %right_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %term_jvp{index}",
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
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr_vjp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr_vjp{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %x_vjp{index} = load double, double* %xptr_vjp{index}",
                f"  %y_vjp{index} = load double, double* %yptr_vjp{index}",
                f"  %vjp_x{index} = fmul double %y_vjp{index}, %cotangent0",
                f"  %vjp_y{index} = fmul double %x_vjp{index}, %cotangent0",
                f"  %outxptr_vjp{index} = getelementptr double, double* %out, i64 {index}",
                f"  %outyptr_vjp{index} = getelementptr double, double* %out, i64 {right_index}",
                f"  store double %vjp_x{index}, double* %outxptr_vjp{index}",
                f"  store double %vjp_y{index}, double* %outyptr_vjp{index}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_vector_squared_norm_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "squared_norm"',
        '; source = "native_vector_squared_norm_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_sum = "0.0"
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr{index} = getelementptr double, double* %values, i64 {index}",
                f"  %x{index} = load double, double* %xptr{index}",
                f"  %square{index} = fmul double %x{index}, %x{index}",
                f"  %sum{index} = fadd double {previous_sum}, %square{index}",
            ]
        )
        previous_sum = f"%sum{index}"
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
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr_gradient{index} = getelementptr double, double* %values, i64 {index}",
                f"  %x_gradient{index} = load double, double* %xptr_gradient{index}",
                f"  %grad{index} = fmul double 2.0, %x_gradient{index}",
                f"  %out_gradient{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %grad{index}, double* %out_gradient{index}",
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
        lines.extend(
            [
                f"  %xptr_jvp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %tptr{index} = getelementptr double, double* %tangent, i64 {index}",
                f"  %x_jvp{index} = load double, double* %xptr_jvp{index}",
                f"  %t{index} = load double, double* %tptr{index}",
                f"  %prod_jvp{index} = fmul double %x_jvp{index}, %t{index}",
                f"  %term_jvp{index} = fmul double 2.0, %prod_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %term_jvp{index}",
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
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr_vjp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %x_vjp{index} = load double, double* %xptr_vjp{index}",
                f"  %scaled_vjp{index} = fmul double 2.0, %x_vjp{index}",
                f"  %vjp{index} = fmul double %scaled_vjp{index}, %cotangent0",
                f"  %outptr_vjp{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %vjp{index}, double* %outptr_vjp{index}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _call_native_vector_dot_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 2 * checked_dimension:
        raise ValueError("native vector dot LLVM/JIT kernel requires 2 * dimension values")
    if output_size not in {1, 2 * checked_dimension}:
        raise ValueError("native vector dot LLVM/JIT output_size must be one or 2 * dimension")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_dot_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 2 * checked_dimension:
        raise ValueError("native vector dot LLVM/JIT kernel requires 2 * dimension values")
    expected_vector_size = 2 * checked_dimension if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native vector dot LLVM/JIT kernel requires {expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, 2 * checked_dimension}:
        raise ValueError("native vector dot LLVM/JIT output_size must be one or 2 * dimension")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_squared_norm_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != checked_dimension:
        raise ValueError("native vector squared norm LLVM/JIT kernel requires dimension values")
    if output_size not in {1, checked_dimension}:
        raise ValueError(
            "native vector squared norm LLVM/JIT output_size must be one or dimension"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_squared_norm_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
    dimension: int,
    output_size: int,
) -> FloatArray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != checked_dimension:
        raise ValueError("native vector squared norm LLVM/JIT kernel requires dimension values")
    expected_vector_size = checked_dimension if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native vector squared norm LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, checked_dimension}:
        raise ValueError(
            "native vector squared norm LLVM/JIT output_size must be one or dimension"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def compile_vector_dot_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile vector dot-product value/JVP/VJP/gradient kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native vector dot AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 2 * checked_dimension:
        raise ValueError("native vector dot AD requires exactly 2 * dimension sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_vector_dot_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_vector_dot_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_vector_dot_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_vector_dot_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            2 * checked_dimension,
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
        native_gradient = _call_native_vector_dot_unary(
            native_functions["gradient"], values, checked_dimension, 2 * checked_dimension
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT vector dot gradient verification failed")
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
            "verified native LLVM MCJIT vector dot value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_vector_dot_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for vector dot-product native LLVM/JIT kernels."""
    checked_dimension = _validate_vector_dot_dimension(dimension)
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
            raise ValueError("native vector dot lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_vector_dot_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_vector_dot_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT vector dot contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native vector dot primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 2 * checked_dimension:
        raise ValueError(
            "native vector dot primitive transform requires exactly 2 * dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_vector_dot_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:dot;dimension:{checked_dimension};layout:x_then_y"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_vector_dot_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_vector_dot",
            "mlir_runtime_verification": "verified: native LLVM/JIT vector dot JVP",
            "rust": "available: Rust PyO3 vector dot value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine vector_dot value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "vector_dot_value,vector_dot_jvp,vector_dot_vjp,vector_dot_gradient"
            ),
            "llvm": "available: native LLVM MCJIT vector dot AD kernel",
            "jit": "available: native LLVM MCJIT vector dot AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT vector dot value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_vector_dot_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_bilinear_vector_dot",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_bilinear_real_domain",
        effect="pure",
    )


def compile_vector_squared_norm_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile vector squared-norm value/JVP/VJP/gradient kernels to LLVM MCJIT."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native vector squared norm AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != checked_dimension:
        raise ValueError("native vector squared norm AD requires exactly dimension sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_vector_squared_norm_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_vector_squared_norm_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_vector_squared_norm_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_vector_squared_norm_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            checked_dimension,
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
        native_gradient = _call_native_vector_squared_norm_unary(
            native_functions["gradient"], values, checked_dimension, checked_dimension
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT vector squared norm gradient verification failed")
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
            "verified native LLVM MCJIT vector squared norm value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_vector_squared_norm_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for vector squared-norm native LLVM/JIT kernels."""
    checked_dimension = _validate_vector_dot_dimension(dimension)
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
            raise ValueError("native vector squared norm lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_vector_squared_norm_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_vector_squared_norm_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT squared-norm contract."""
    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native vector squared norm primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != checked_dimension:
        raise ValueError(
            "native vector squared norm primitive transform requires exactly dimension "
            "sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_vector_squared_norm_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:squared_norm;dimension:{checked_dimension}"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_vector_squared_norm_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_vector_squared_norm",
            "mlir_runtime_verification": "verified: native LLVM/JIT vector squared norm JVP",
            "rust": "available: Rust PyO3 vector squared norm value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine vector_squared_norm value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "vector_squared_norm_value,vector_squared_norm_jvp,"
                "vector_squared_norm_vjp,vector_squared_norm_gradient"
            ),
            "llvm": "available: native LLVM MCJIT vector squared norm AD kernel",
            "jit": "available: native LLVM MCJIT vector squared norm AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT vector squared norm value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_vector_squared_norm_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_vector_squared_norm",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_vector_squared_norm_real_domain",
        effect="pure",
    )
