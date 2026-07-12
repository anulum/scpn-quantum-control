# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR executable kernel module
# scpn-quantum-control -- executable compiler-AD kernel core for the MLIR surface
"""Executable compiler-AD kernel: batching, verification and custom-derivative lowering.

This module holds the shared executable-kernel infrastructure used by both the
matrix/vector native-JIT compilers and the facade-level compile entry points: the
``ExecutableCompilerADKernel`` wrapper around compiled value/JVP/VJP/gradient
callables, its batching rule and per-axis batch helpers, the kernel verifier that
checks an emitted kernel against the interpreted reference, the scalar-gradient
LLVM IR builder, and the generic custom-derivative-rule to MLIR text lowering.

It depends only on the shared native lowering primitives, the MLIR record types and
the differentiable kernel contracts, so it stays a leaf of the compiler package and
never imports the ``mlir`` facade or the matrix-compilation module back.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    CustomDerivativeRule,
    PrimitiveBatchingRule,
    value_and_custom_jacobian,
)
from .mlir_native_primitives import (
    _as_finite_vector,
    _escape_mlir_string,
    _fmt_bool,
    _fmt_float,
    _max_abs_error,
    _safe_llvm_symbol,
)
from .mlir_records import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    DifferentiableMLIRCompileConfig,
    MLIRModule,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ExecutableCompilerADKernel:
    """Executable compiler-backed primitive AD kernel with MLIR provenance."""

    rule_name: str
    backend: str
    mlir_module: MLIRModule
    value_kernel: Callable[[FloatArray], FloatArray]
    jvp_kernel: Callable[[FloatArray, FloatArray], FloatArray] | None
    vjp_kernel: Callable[[FloatArray, FloatArray], FloatArray] | None
    verification: CompilerADKernelVerification
    llvm_gradient_ir: str | None = None
    claim_boundary: str = (
        "verified executable MLIR-runtime primitive AD kernel; "
        "native LLVM/JIT code generation remains fail-closed"
    )

    def __post_init__(self) -> None:
        if not self.rule_name:
            raise ValueError("rule_name must be non-empty")
        if self.backend not in {"mlir_runtime", "native_llvm_jit"}:
            raise ValueError("backend must be 'mlir_runtime' or 'native_llvm_jit'")
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not callable(self.value_kernel):
            raise ValueError("value_kernel must be callable")
        if self.jvp_kernel is not None and not callable(self.jvp_kernel):
            raise ValueError("jvp_kernel must be callable")
        if self.vjp_kernel is not None and not callable(self.vjp_kernel):
            raise ValueError("vjp_kernel must be callable")
        if not isinstance(self.verification, CompilerADKernelVerification):
            raise ValueError("verification must be CompilerADKernelVerification")
        if not self.verification.passed:
            raise ValueError("executable compiler AD kernel verification failed")
        if self.llvm_gradient_ir is not None and not self.llvm_gradient_ir.strip():
            raise ValueError("llvm_gradient_ir must be non-empty or None")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def value(self, values: FloatArray) -> FloatArray:
        """Execute the compiled value kernel."""
        return self.value_kernel(values)

    def jvp(self, values: FloatArray, tangent: FloatArray) -> FloatArray:
        """Execute the compiled JVP kernel."""
        if self.jvp_kernel is None:
            raise ValueError(f"kernel {self.rule_name} has no JVP rule")
        return self.jvp_kernel(values, tangent)

    def vjp(self, values: FloatArray, cotangent: FloatArray) -> FloatArray:
        """Execute the compiled VJP kernel."""
        if self.vjp_kernel is None:
            raise ValueError(f"kernel {self.rule_name} has no VJP rule")
        return self.vjp_kernel(values, cotangent)

    def gradient(self, values: FloatArray) -> FloatArray:
        """Execute the compiled scalar-output gradient kernel."""
        if self.vjp_kernel is None:
            raise ValueError(f"kernel {self.rule_name} has no VJP rule")
        checked_values = _as_finite_vector("values", values)
        output = self.value_kernel(checked_values)
        if output.size != 1:
            raise ValueError(f"kernel {self.rule_name} gradient requires scalar output")
        return self.vjp_kernel(checked_values, np.ones(1, dtype=np.float64))


def make_executable_ad_kernel_batching_rule(
    kernel: ExecutableCompilerADKernel,
    *,
    method: str = "auto",
) -> PrimitiveBatchingRule:
    """Create a primitive-specific batching rule backed by an executable AD kernel.

    ``method="auto"`` dispatches one-argument calls to ``value`` and two-argument
    calls to ``jvp`` or ``vjp`` by matching the second slice against the input and
    output dimensions. If those dimensions are equal, callers must request
    ``method="jvp"`` or ``method="vjp"`` explicitly so transform nesting remains
    fail-closed rather than guessing.
    """
    if not isinstance(kernel, ExecutableCompilerADKernel):
        raise ValueError("kernel must be an ExecutableCompilerADKernel")
    if method not in {"auto", "value", "jvp", "vjp", "gradient"}:
        raise ValueError("method must be 'auto', 'value', 'jvp', 'vjp', or 'gradient'")

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function
        batched_args, batch_size = _prepare_executable_kernel_batch_args(args, axes)
        outputs = []
        for item in range(batch_size):
            call_args = tuple(
                _slice_executable_kernel_batch_arg(arg, axis, item) for arg, axis in batched_args
            )
            outputs.append(_execute_kernel_batch_slice(kernel, method, call_args))
        return _stack_executable_kernel_batch_outputs(outputs, out_axes)

    return batching_rule


def _prepare_executable_kernel_batch_args(
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
) -> tuple[tuple[tuple[object, int | None], ...], int]:
    if not args:
        raise ValueError("executable AD kernel batching requires at least one argument")
    if len(args) != len(axes):
        raise ValueError("executable AD kernel batching axes must match argument count")
    batched: list[tuple[object, int | None]] = []
    batch_size: int | None = None
    for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
        if axis is None:
            batched.append((arg, None))
            continue
        if not isinstance(axis, int):
            raise ValueError("executable AD kernel batching axes must be integers or None")
        array = _as_executable_kernel_batch_array(f"argument {index}", arg)
        axis_index = _normalise_executable_kernel_batch_axis(
            f"axes[{index}]",
            axis,
            array.ndim,
        )
        size = int(array.shape[axis_index])
        if size <= 0:
            raise ValueError("executable AD kernel batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("executable AD kernel batching axes must have the same length")
        batched.append((array, axis_index))
    if batch_size is None:
        raise ValueError("executable AD kernel batching requires at least one mapped axis")
    return tuple(batched), batch_size


def _slice_executable_kernel_batch_arg(arg: object, axis: int | None, item: int) -> object:
    if axis is None:
        return arg
    return np.take(cast(FloatArray, arg), item, axis=axis)


def _as_executable_kernel_batch_array(name: str, value: object) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "O", "S", "U"}:
        raise ValueError(f"executable AD kernel batching {name} must be numeric")
    # Checked before ascontiguousarray, which silently promotes 0-d input to 1-d.
    if raw.ndim == 0:
        raise ValueError(f"executable AD kernel batching {name} cannot map over a scalar")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"executable AD kernel batching {name} must contain only finite values")
    return array


def _normalise_executable_kernel_batch_axis(name: str, axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis


def _execute_kernel_batch_slice(
    kernel: ExecutableCompilerADKernel,
    method: str,
    args: tuple[object, ...],
) -> FloatArray:
    if method == "value":
        if len(args) != 1:
            raise ValueError("executable AD kernel value batching requires one argument")
        return kernel.value(_as_finite_vector("values", args[0]))
    if method == "gradient":
        if len(args) != 1:
            raise ValueError("executable AD kernel gradient batching requires one argument")
        return kernel.gradient(_as_finite_vector("values", args[0]))
    if method == "jvp":
        if len(args) != 2:
            raise ValueError("executable AD kernel JVP batching requires values and tangent")
        return kernel.jvp(
            _as_finite_vector("values", args[0]),
            _as_finite_vector("tangent", args[1]),
        )
    if method == "vjp":
        if len(args) != 2:
            raise ValueError("executable AD kernel VJP batching requires values and cotangent")
        return kernel.vjp(
            _as_finite_vector("values", args[0]),
            _as_finite_vector("cotangent", args[1]),
        )
    if len(args) == 1:
        return kernel.value(_as_finite_vector("values", args[0]))
    if len(args) != 2:
        raise ValueError("automatic executable AD kernel batching supports one or two arguments")
    values = _as_finite_vector("values", args[0])
    tangent_or_cotangent = _as_finite_vector("tangent_or_cotangent", args[1])
    output_size = int(kernel.value(values).size)
    input_size = int(values.size)
    jvp_matches = kernel.jvp_kernel is not None and tangent_or_cotangent.size == input_size
    vjp_matches = kernel.vjp_kernel is not None and tangent_or_cotangent.size == output_size
    if jvp_matches and vjp_matches:
        raise ValueError(
            "ambiguous executable AD kernel batching method; specify method='jvp' or method='vjp'"
        )
    if jvp_matches:
        return kernel.jvp(values, tangent_or_cotangent)
    if vjp_matches:
        return kernel.vjp(values, tangent_or_cotangent)
    raise ValueError(
        "automatic executable AD kernel batching could not match the second argument "
        "to tangent or cotangent dimensions"
    )


def _stack_executable_kernel_batch_outputs(
    outputs: Sequence[FloatArray],
    out_axes: int,
) -> FloatArray:
    if not outputs:
        raise ValueError("executable AD kernel batching outputs must be non-empty")
    arrays = [np.asarray(output, dtype=np.float64) for output in outputs]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("executable AD kernel batching outputs must have consistent shapes")
    result_rank = arrays[0].ndim + 1
    axis = out_axes
    if axis < 0:
        axis += result_rank
    if axis < 0 or axis >= result_rank:
        raise ValueError("executable AD kernel batching out_axes is out of bounds")
    return np.stack(arrays, axis=axis)


def compile_custom_derivative_rule_to_mlir(
    rule: CustomDerivativeRule,
    values: FloatArray,
    config: DifferentiableMLIRCompileConfig | None = None,
) -> MLIRModule:
    """Lower an exact custom derivative rule to deterministic MLIR-style text.

    This emits an auditable differentiable-primitive interchange artefact with
    value and Jacobian shape metadata. When numeric payloads are enabled, the
    current value and exact custom Jacobian are embedded as deterministic
    attributes. The function deliberately does not claim executable LLVM or JIT
    code generation.
    """
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("differentiable MLIR lowering requires a CustomDerivativeRule")
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    jacobian_result = value_and_custom_jacobian(rule, values)
    parameter_count = jacobian_result.jacobian.shape[1]
    output_count = jacobian_result.value.size
    lines = [
        f'module attributes {{scpn.module = "differentiable_primitive", '
        f'scpn.dialect = "{compile_config.dialect}", '
        f'scpn.rule = "{_escape_mlir_string(rule.name)}", '
        f"scpn.n_parameters = {parameter_count}, "
        f"scpn.n_outputs = {output_count}}} {{",
        "  func.func @main() {",
    ]
    for index, (name, trainable) in enumerate(
        zip(jacobian_result.parameter_names, jacobian_result.trainable, strict=True)
    ):
        lines.append(
            "    scpn_diff.parameter "
            f'%p{index} {{name = "{_escape_mlir_string(name)}", trainable = {_fmt_bool(trainable)}}}'
        )
    if compile_config.include_numeric_payload:
        for index, value in enumerate(jacobian_result.value):
            lines.append(f"    scpn_diff.value %{index} {{value = {_fmt_float(float(value))}}}")
        for row in range(output_count):
            for column in range(parameter_count):
                value = float(jacobian_result.jacobian[row, column])
                if abs(value) > 1.0e-15:
                    lines.append(
                        "    scpn_diff.jacobian "
                        f"{{row = {row}, col = {column}, value = {_fmt_float(value)}}}"
                    )
    lines.append(
        "    scpn_diff.custom_rule "
        f"{{jvp = {_fmt_bool(rule.jvp_rule is not None)}, "
        f"vjp = {_fmt_bool(rule.vjp_rule is not None)}, "
        'execution = "interchange_only"}}'
    )
    lines.append("    return")
    lines.append("  }")
    if compile_config.include_metadata:
        metadata = {
            "method": jacobian_result.method,
            "parameter_names": list(jacobian_result.parameter_names),
            "trainable": list(jacobian_result.trainable),
            "target": compile_config.target,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=compile_config.dialect,
        resource_counts={
            "parameters": parameter_count,
            "outputs": output_count,
            "jacobian_nnz": int(np.count_nonzero(jacobian_result.jacobian)),
            "trainable_parameters": int(sum(jacobian_result.trainable)),
        },
        metadata={
            "claim_boundary": "textual differentiable MLIR-style IR export; no executable LLVM or JIT lowering",
            "rule": rule.name,
            "target": compile_config.target,
            "sha256_source": "module.text",
        },
    )


def _verify_executable_ad_kernel(
    rule: CustomDerivativeRule,
    values: FloatArray,
    value_kernel: Callable[[FloatArray], FloatArray],
    jvp_kernel: Callable[[FloatArray, FloatArray], FloatArray] | None,
    vjp_kernel: Callable[[FloatArray, FloatArray], FloatArray] | None,
    config: CompilerADExecutableConfig,
    *,
    sample_tangent: Sequence[float] | FloatArray | None,
    sample_cotangent: Sequence[float] | FloatArray | None,
) -> CompilerADKernelVerification:
    if not config.verify:
        return CompilerADKernelVerification(
            value_close=True,
            jvp_close=None,
            vjp_close=None,
            max_abs_error=0.0,
            samples=1,
            gradient_close=None,
        )
    errors: list[float] = []
    expected_value = _as_finite_vector("rule value", rule.value_fn(values))
    kernel_value = value_kernel(values)
    value_close = bool(
        np.allclose(kernel_value, expected_value, atol=config.atol, rtol=config.rtol)
    )
    errors.append(_max_abs_error(kernel_value, expected_value))
    jvp_close: bool | None = None
    if rule.jvp_rule is not None and jvp_kernel is not None:
        tangent = (
            np.ones_like(values)
            if sample_tangent is None
            else _as_finite_vector("sample_tangent", sample_tangent)
        )
        if tangent.shape != values.shape:
            raise ValueError("sample_tangent shape must match sample_values shape")
        expected_jvp = _as_finite_vector("rule JVP", rule.jvp_rule(values, tangent))
        kernel_jvp = jvp_kernel(values, tangent)
        jvp_close = bool(np.allclose(kernel_jvp, expected_jvp, atol=config.atol, rtol=config.rtol))
        errors.append(_max_abs_error(kernel_jvp, expected_jvp))
    vjp_close: bool | None = None
    gradient_close: bool | None = None
    if rule.vjp_rule is not None and vjp_kernel is not None:
        cotangent = (
            np.ones_like(expected_value)
            if sample_cotangent is None
            else _as_finite_vector("sample_cotangent", sample_cotangent)
        )
        if cotangent.shape != expected_value.shape:
            raise ValueError("sample_cotangent shape must match value output shape")
        expected_vjp = _as_finite_vector("rule VJP", rule.vjp_rule(values, cotangent))
        kernel_vjp = vjp_kernel(values, cotangent)
        vjp_close = bool(np.allclose(kernel_vjp, expected_vjp, atol=config.atol, rtol=config.rtol))
        errors.append(_max_abs_error(kernel_vjp, expected_vjp))
        if expected_value.size == 1:
            unit_cotangent = np.ones(1, dtype=np.float64)
            expected_gradient = _as_finite_vector(
                "rule scalar gradient", rule.vjp_rule(values, unit_cotangent)
            )
            kernel_gradient = vjp_kernel(values, unit_cotangent)
            gradient_close = bool(
                np.allclose(kernel_gradient, expected_gradient, atol=config.atol, rtol=config.rtol)
            )
            errors.append(_max_abs_error(kernel_gradient, expected_gradient))
    verification = CompilerADKernelVerification(
        value_close=value_close,
        jvp_close=jvp_close,
        vjp_close=vjp_close,
        max_abs_error=max(errors),
        samples=1,
        gradient_close=gradient_close,
    )
    if not verification.passed:
        raise ValueError("executable compiler AD kernel verification failed")
    return verification


def _compile_scalar_gradient_llvm_ir(
    rule: CustomDerivativeRule,
    values: FloatArray,
    vjp_kernel: Callable[[FloatArray, FloatArray], FloatArray],
) -> str:
    value = _as_finite_vector("rule value", rule.value_fn(values))
    if value.size != 1:
        return ""
    gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
    function_name = _safe_llvm_symbol(f"{rule.name}_gradient")
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule.name)}"',
        '; source = "verified_mlir_runtime_vjp_cotangent_one"',
        '; execution = "mlir_runtime_gradient_adapter"',
        '; native_llvm_jit = "blocked_until_native_codegen_backend_exists"',
        f"define void @{function_name}(double* %out) {{",
        "entry:",
    ]
    for index, component in enumerate(gradient):
        lines.append(f"  %slot{index} = getelementptr double, double* %out, i64 {index}")
        lines.append(f"  store double {_fmt_float(float(component))}, double* %slot{index}")
    lines.append("  ret void")
    lines.append("}")
    return "\n".join(lines) + "\n"
