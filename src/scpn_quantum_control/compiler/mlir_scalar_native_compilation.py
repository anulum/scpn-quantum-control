# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- scalar native LLVM/JIT AD compilation for the MLIR surface
"""Native LLVM/JIT autodiff compilation for scalar primitives.

Emits the LLVM IR and builds the executable native-JIT kernels, lowering rules and
primitive transforms that evaluate each scalar primitive together with its forward
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
)
from .mlir_executable_kernel import (
    ExecutableCompilerADKernel,
    _verify_executable_ad_kernel,
    compile_custom_derivative_rule_to_mlir,
)
from .mlir_native_primitives import (
    _as_finite_vector,
    _compile_native_llvm_jit_functions,
    _escape_mlir_string,
    _fmt_float,
    _load_llvmlite_binding,
    _safe_llvm_symbol,
)
from .mlir_records import CompilerADExecutableConfig

FloatArray: TypeAlias = NDArray[np.float64]


def _compile_scalar_quadratic_native_llvm_ir(
    rule_name: str,
    quadratic: float,
    linear: float,
    constant: float,
) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    doubled_quadratic = 2.0 * quadratic
    quadratic_literal = _fmt_llvm_double(quadratic)
    linear_literal = _fmt_llvm_double(linear)
    constant_literal = _fmt_llvm_double(constant)
    doubled_quadratic_literal = _fmt_llvm_double(doubled_quadratic)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; source = "native_scalar_quadratic_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            "  %x2 = fmul double %x, %x",
            f"  %ax2 = fmul double {quadratic_literal}, %x2",
            f"  %bx = fmul double {linear_literal}, %x",
            "  %sum = fadd double %ax2, %bx",
            f"  %value = fadd double %sum, {constant_literal}",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %value, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            f"  %ax = fmul double {doubled_quadratic_literal}, %x",
            f"  %grad = fadd double %ax, {linear_literal}",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %grad, double* %out0",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_jvp(double* %values, "
                "double* %tangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            f"  %ax = fmul double {doubled_quadratic_literal}, %x",
            f"  %grad = fadd double %ax, {linear_literal}",
            "  %tangent0ptr = getelementptr double, double* %tangent, i64 0",
            "  %tangent0 = load double, double* %tangent0ptr",
            "  %jvp = fmul double %grad, %tangent0",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %jvp, double* %out0",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_vjp(double* %values, "
                "double* %cotangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            f"  %ax = fmul double {doubled_quadratic_literal}, %x",
            f"  %grad = fadd double %ax, {linear_literal}",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %vjp = fmul double %grad, %cotangent0",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %vjp, double* %out0",
            "  ret void",
            "}",
            "",
        ]
    )


def _fmt_llvm_double(value: float) -> str:
    text = _fmt_float(float(value))
    if "." not in text and "e" not in text.lower():
        return f"{text}.0"
    return text


def _scalar_unary_native_intrinsics(primitive: str) -> tuple[str, ...]:
    if primitive == "sin":
        return ("sin", "cos")
    if primitive == "cos":
        return ("sin", "cos")
    if primitive == "exp":
        return ("exp",)
    raise ValueError("native scalar unary LLVM/JIT primitive must be one of sin, cos, exp")


def _scalar_unary_native_value_lines(primitive: str) -> tuple[str, ...]:
    if primitive == "sin":
        return ("%value = call double @llvm.sin.f64(double %x)",)
    if primitive == "cos":
        return ("%value = call double @llvm.cos.f64(double %x)",)
    if primitive == "exp":
        return ("%value = call double @llvm.exp.f64(double %x)",)
    raise ValueError("native scalar unary LLVM/JIT primitive must be one of sin, cos, exp")


def _scalar_unary_native_gradient_lines(primitive: str) -> tuple[str, ...]:
    if primitive == "sin":
        return ("%grad = call double @llvm.cos.f64(double %x)",)
    if primitive == "cos":
        return (
            "%sin = call double @llvm.sin.f64(double %x)",
            "%grad = fsub double -0.0, %sin",
        )
    if primitive == "exp":
        return ("%grad = call double @llvm.exp.f64(double %x)",)
    raise ValueError("native scalar unary LLVM/JIT primitive must be one of sin, cos, exp")


def _compile_scalar_unary_elementwise_native_llvm_ir(
    rule_name: str,
    primitive: str,
) -> str:
    checked_primitive = primitive.strip().lower()
    intrinsics = _scalar_unary_native_intrinsics(checked_primitive)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        f'; primitive = "{_escape_mlir_string(checked_primitive)}"',
        '; source = "native_scalar_unary_elementwise_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
    ]
    for intrinsic in intrinsics:
        lines.append(f"declare double @llvm.{intrinsic}.f64(double)")
    lines.extend(
        [
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
        ]
    )
    lines.extend(f"  {line}" for line in _scalar_unary_native_value_lines(checked_primitive))
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %value, double* %out0",
            "  ret void",
            "}",
            "",
        ]
    )
    for function_name, operand_name, result_name in (
        ("gradient", None, "grad"),
        ("jvp", "tangent", "jvp"),
        ("vjp", "cotangent", "vjp"),
    ):
        if operand_name is None:
            lines.append(
                f"define void @{base_symbol}_{function_name}(double* %values, double* %out) {{"
            )
        else:
            lines.append(
                f"define void @{base_symbol}_{function_name}(double* %values, "
                f"double* %{operand_name}, double* %out) {{"
            )
        lines.extend(
            [
                "entry:",
                "  %xptr = getelementptr double, double* %values, i64 0",
                "  %x = load double, double* %xptr",
            ]
        )
        lines.extend(
            f"  {line}" for line in _scalar_unary_native_gradient_lines(checked_primitive)
        )
        if operand_name is not None:
            lines.extend(
                [
                    f"  %{operand_name}0ptr = getelementptr double, double* %{operand_name}, i64 0",
                    f"  %{operand_name}0 = load double, double* %{operand_name}0ptr",
                    f"  %{result_name} = fmul double %grad, %{operand_name}0",
                ]
            )
        lines.extend(
            [
                "  %out0 = getelementptr double, double* %out, i64 0",
                f"  store double %{result_name}, double* %out0",
                "  ret void",
                "}",
                "",
            ]
        )
    return "\n".join(lines)


def _scalar_binary_native_value_line(primitive: str) -> str:
    if primitive == "add":
        return "%value = fadd double %x, %y"
    if primitive == "subtract":
        return "%value = fsub double %x, %y"
    if primitive == "multiply":
        return "%value = fmul double %x, %y"
    raise ValueError(
        "native scalar binary LLVM/JIT primitive must be one of add, subtract, multiply"
    )


def _scalar_binary_native_gradient_lines(primitive: str) -> tuple[str, ...]:
    if primitive == "add":
        return ("%grad_x = fadd double 1.0, 0.0", "%grad_y = fadd double 1.0, 0.0")
    if primitive == "subtract":
        return ("%grad_x = fadd double 1.0, 0.0", "%grad_y = fsub double -0.0, 1.0")
    if primitive == "multiply":
        return ("%grad_x = fadd double %y, 0.0", "%grad_y = fadd double %x, 0.0")
    raise ValueError(
        "native scalar binary LLVM/JIT primitive must be one of add, subtract, multiply"
    )


def _compile_scalar_binary_elementwise_native_llvm_ir(
    rule_name: str,
    primitive: str,
) -> str:
    checked_primitive = primitive.strip().lower()
    value_line = _scalar_binary_native_value_line(checked_primitive)
    gradient_lines = _scalar_binary_native_gradient_lines(checked_primitive)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        f'; primitive = "{_escape_mlir_string(checked_primitive)}"',
        '; source = "native_scalar_binary_elementwise_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
        "  %xptr = getelementptr double, double* %values, i64 0",
        "  %yptr = getelementptr double, double* %values, i64 1",
        "  %x = load double, double* %xptr",
        "  %y = load double, double* %yptr",
        f"  {value_line}",
        "  %out0 = getelementptr double, double* %out, i64 0",
        "  store double %value, double* %out0",
        "  ret void",
        "}",
        "",
        f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
        "entry:",
        "  %xptr = getelementptr double, double* %values, i64 0",
        "  %yptr = getelementptr double, double* %values, i64 1",
        "  %x = load double, double* %xptr",
        "  %y = load double, double* %yptr",
    ]
    lines.extend(f"  {line}" for line in gradient_lines)
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %grad_x, double* %out0",
            "  store double %grad_y, double* %out1",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_jvp(double* %values, "
                "double* %tangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %yptr = getelementptr double, double* %values, i64 1",
            "  %x = load double, double* %xptr",
            "  %y = load double, double* %yptr",
        ]
    )
    lines.extend(f"  {line}" for line in gradient_lines)
    lines.extend(
        [
            "  %tangent0ptr = getelementptr double, double* %tangent, i64 0",
            "  %tangent1ptr = getelementptr double, double* %tangent, i64 1",
            "  %tangent0 = load double, double* %tangent0ptr",
            "  %tangent1 = load double, double* %tangent1ptr",
            "  %jvp_x = fmul double %grad_x, %tangent0",
            "  %jvp_y = fmul double %grad_y, %tangent1",
            "  %jvp = fadd double %jvp_x, %jvp_y",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %jvp, double* %out0",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_vjp(double* %values, "
                "double* %cotangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %yptr = getelementptr double, double* %values, i64 1",
            "  %x = load double, double* %xptr",
            "  %y = load double, double* %yptr",
        ]
    )
    lines.extend(f"  {line}" for line in gradient_lines)
    lines.extend(
        [
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %vjp_x = fmul double %grad_x, %cotangent0",
            "  %vjp_y = fmul double %grad_y, %cotangent0",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %vjp_x, double* %out0",
            "  store double %vjp_y, double* %out1",
            "  ret void",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _call_native_scalar_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 1:
        raise ValueError("native scalar LLVM/JIT kernel requires one value")
    output = np.zeros(1, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_scalar_binary(
    function: Callable[[Any, Any, Any], None],
    values: FloatArray,
    tangent_or_cotangent: FloatArray,
    label: str,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 1:
        raise ValueError("native scalar LLVM/JIT kernel requires one value")
    if checked_vector.size != 1:
        raise ValueError(f"native scalar LLVM/JIT kernel requires one {label} value")
    output = np.zeros(1, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_scalar_pair_unary(
    function: Callable[[Any, Any], None],
    values: FloatArray,
    output_size: int,
) -> FloatArray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 2:
        raise ValueError("native scalar binary LLVM/JIT kernel requires two values")
    if output_size not in {1, 2}:
        raise ValueError("native scalar binary LLVM/JIT output_size must be one or two")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_scalar_pair_binary(
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
    if checked_values.size != 2:
        raise ValueError("native scalar binary LLVM/JIT kernel requires two values")
    expected_vector_size = 2 if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native scalar binary LLVM/JIT kernel requires {expected_vector_size} "
            f"{label} value(s)"
        )
    if output_size not in {1, 2}:
        raise ValueError("native scalar binary LLVM/JIT output_size must be one or two")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def compile_scalar_quadratic_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    quadratic: float,
    linear: float,
    constant: float,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile scalar quadratic value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native scalar quadratic AD requires backend='native_llvm_jit'")
    coefficients = np.asarray([quadratic, linear, constant], dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("quadratic, linear, and constant coefficients must be finite")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 1:
        raise ValueError("native scalar quadratic AD requires exactly one sample value")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_scalar_quadratic_native_llvm_ir(
        rule.name,
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_scalar_unary(native_functions["value"], raw_values)

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_scalar_binary(
            native_functions["jvp"], raw_values, raw_tangent, "tangent"
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_scalar_binary(
            native_functions["vjp"], raw_values, raw_cotangent, "cotangent"
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
        native_gradient = _call_native_scalar_unary(native_functions["gradient"], values)
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT gradient kernel verification failed")
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
            "verified native LLVM MCJIT scalar quadratic value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_scalar_quadratic_native_llvm_jit_lowering_rule(
    *,
    quadratic: float,
    linear: float,
    constant: float,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for scalar quadratic native LLVM/JIT AD kernels."""

    coefficients = np.asarray([quadratic, linear, constant], dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("quadratic, linear, and constant coefficients must be finite")
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
            raise ValueError("native scalar quadratic lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_scalar_quadratic_ad_to_native_llvm_jit(
            rule,
            quadratic=float(coefficients[0]),
            linear=float(coefficients[1]),
            constant=float(coefficients[2]),
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    primitive: str,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile scalar unary elementwise value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_primitive = primitive.strip().lower()
    _scalar_unary_native_intrinsics(checked_primitive)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native scalar unary AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 1:
        raise ValueError("native scalar unary AD requires exactly one sample value")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_scalar_unary_elementwise_native_llvm_ir(
        rule.name,
        checked_primitive,
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_scalar_unary(native_functions["value"], raw_values)

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_scalar_binary(
            native_functions["jvp"], raw_values, raw_tangent, "tangent"
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_scalar_binary(
            native_functions["vjp"], raw_values, raw_cotangent, "cotangent"
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
        native_gradient = _call_native_scalar_unary(native_functions["gradient"], values)
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT scalar unary gradient verification failed")
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
            "verified native LLVM MCJIT scalar unary value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_scalar_unary_elementwise_native_llvm_jit_lowering_rule(
    *,
    primitive: str,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for scalar unary elementwise native LLVM/JIT kernels."""

    checked_primitive = primitive.strip().lower()
    _scalar_unary_native_intrinsics(checked_primitive)
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
            raise ValueError("native scalar unary lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
            rule,
            primitive=checked_primitive,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    primitive: str,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile scalar binary elementwise value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_primitive = primitive.strip().lower()
    _scalar_binary_native_value_line(checked_primitive)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native scalar binary AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 2:
        raise ValueError("native scalar binary AD requires exactly two sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_scalar_binary_elementwise_native_llvm_ir(
        rule.name,
        checked_primitive,
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _call_native_scalar_pair_unary(native_functions["value"], raw_values, 1)

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        return _call_native_scalar_pair_binary(
            native_functions["jvp"], raw_values, raw_tangent, "tangent", 1
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        return _call_native_scalar_pair_binary(
            native_functions["vjp"], raw_values, raw_cotangent, "cotangent", 2
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
        native_gradient = _call_native_scalar_pair_unary(native_functions["gradient"], values, 2)
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT scalar binary gradient verification failed")
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
            "verified native LLVM MCJIT scalar binary value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_scalar_binary_elementwise_native_llvm_jit_lowering_rule(
    *,
    primitive: str,
    sample_values: Sequence[float] | FloatArray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for scalar binary elementwise native LLVM/JIT kernels."""

    checked_primitive = primitive.strip().lower()
    _scalar_binary_native_value_line(checked_primitive)
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
            raise ValueError("native scalar binary lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
            rule,
            primitive=checked_primitive,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule
