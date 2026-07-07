# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- executable compiler-AD kernel core tests
"""Module-specific tests for the executable compiler-AD kernel core.

The suite drives ``ExecutableCompilerADKernel`` and its batching rule through
plain verified Python kernels, pins every fail-closed admission boundary, and
checks the custom-derivative MLIR lowering, the kernel verifier, and the
scalar-gradient LLVM IR builder against analytic references.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.compiler.mlir_executable_kernel import (
    ExecutableCompilerADKernel,
    FloatArray,
    _compile_scalar_gradient_llvm_ir,
    _normalise_executable_kernel_batch_axis,
    _stack_executable_kernel_batch_outputs,
    _verify_executable_ad_kernel,
    compile_custom_derivative_rule_to_mlir,
    make_executable_ad_kernel_batching_rule,
)
from scpn_quantum_control.compiler.mlir_records import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    DifferentiableMLIRCompileConfig,
    MLIRModule,
)
from scpn_quantum_control.program_ad_registry import CustomDerivativeRule

_SCALAR_MATRIX = np.array([[2.0, -3.0]], dtype=np.float64)
_WIDE_MATRIX = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
_SQUARE_MATRIX = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)


def _mlir_module() -> MLIRModule:
    """Return a minimal valid MLIR module record for kernel provenance."""
    text = "module {}\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect="scpn_diff",
        resource_counts={},
    )


def _passed_verification() -> CompilerADKernelVerification:
    """Return a fully passed kernel verification record."""
    return CompilerADKernelVerification(
        value_close=True,
        jvp_close=True,
        vjp_close=True,
        max_abs_error=0.0,
        samples=1,
        gradient_close=True,
    )


def _linear_kernel(
    matrix: FloatArray,
    *,
    with_jvp: bool = True,
    with_vjp: bool = True,
) -> ExecutableCompilerADKernel:
    """Return a verified executable kernel around the linear map ``matrix @ x``."""

    def value_kernel(values: FloatArray) -> FloatArray:
        return matrix @ values

    def jvp_kernel(values: FloatArray, tangent: FloatArray) -> FloatArray:
        del values
        return matrix @ tangent

    def vjp_kernel(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        del values
        return matrix.T @ cotangent

    return ExecutableCompilerADKernel(
        rule_name="linear_map",
        backend="mlir_runtime",
        mlir_module=_mlir_module(),
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if with_jvp else None,
        vjp_kernel=vjp_kernel if with_vjp else None,
        verification=_passed_verification(),
    )


def _product_rule() -> CustomDerivativeRule:
    """Return the two-parameter product rule ``f(a, b) = [a * b]``."""

    def value_fn(values: FloatArray) -> FloatArray:
        return np.array([values[0] * values[1]], dtype=np.float64)

    def jvp_rule(values: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array(
            [tangent[0] * values[1] + values[0] * tangent[1]],
            dtype=np.float64,
        )

    def vjp_rule(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array(
            [cotangent[0] * values[1], cotangent[0] * values[0]],
            dtype=np.float64,
        )

    return CustomDerivativeRule(
        name="product_pair",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a", "b"),
        trainable=(True, True),
    )


def test_kernel_contract_rejects_malformed_fields() -> None:
    """The executable-kernel contract rejects every malformed constructor field."""
    valid: dict[str, Any] = {
        "rule_name": "linear_map",
        "backend": "mlir_runtime",
        "mlir_module": _mlir_module(),
        "value_kernel": lambda values: values,
        "jvp_kernel": None,
        "vjp_kernel": None,
        "verification": _passed_verification(),
    }
    failed = CompilerADKernelVerification(
        value_close=False,
        jvp_close=None,
        vjp_close=None,
        max_abs_error=1.0,
        samples=1,
    )
    for overrides, message in (
        ({"rule_name": ""}, "rule_name must be non-empty"),
        ({"backend": "cuda"}, "backend must be 'mlir_runtime' or 'native_llvm_jit'"),
        ({"mlir_module": "module {}"}, "mlir_module must be an MLIRModule"),
        ({"value_kernel": 7}, "value_kernel must be callable"),
        ({"jvp_kernel": 7}, "jvp_kernel must be callable"),
        ({"vjp_kernel": 7}, "vjp_kernel must be callable"),
        ({"verification": {"passed": True}}, "verification must be CompilerADKernelVerification"),
        ({"verification": failed}, "kernel verification failed"),
        ({"llvm_gradient_ir": "  "}, "llvm_gradient_ir must be non-empty or None"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ):
        broken = dict(valid)
        broken.update(overrides)
        with pytest.raises(ValueError, match=message):
            ExecutableCompilerADKernel(**broken)


def test_kernel_executes_value_jvp_vjp_and_gradient() -> None:
    """Kernel call surfaces execute the compiled callables with exact results."""
    kernel = _linear_kernel(_SCALAR_MATRIX)
    values = np.array([1.5, 2.0], dtype=np.float64)
    tangent = np.array([1.0, -1.0], dtype=np.float64)
    cotangent = np.array([2.0], dtype=np.float64)

    np.testing.assert_allclose(kernel.value(values), _SCALAR_MATRIX @ values)
    np.testing.assert_allclose(kernel.jvp(values, tangent), _SCALAR_MATRIX @ tangent)
    np.testing.assert_allclose(kernel.vjp(values, cotangent), _SCALAR_MATRIX.T @ cotangent)
    np.testing.assert_allclose(kernel.gradient(values), _SCALAR_MATRIX[0])


def test_kernel_fails_closed_without_derivative_callables() -> None:
    """Missing JVP/VJP callables raise typed errors instead of guessing."""
    kernel = _linear_kernel(_SCALAR_MATRIX, with_jvp=False, with_vjp=False)
    values = np.array([1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="has no JVP rule"):
        kernel.jvp(values, values)
    with pytest.raises(ValueError, match="has no VJP rule"):
        kernel.vjp(values, np.ones(1, dtype=np.float64))
    with pytest.raises(ValueError, match="has no VJP rule"):
        kernel.gradient(values)


def test_kernel_gradient_requires_scalar_output() -> None:
    """The gradient adapter fails closed on non-scalar kernel outputs."""
    kernel = _linear_kernel(_WIDE_MATRIX)
    with pytest.raises(ValueError, match="gradient requires scalar output"):
        kernel.gradient(np.array([1.0, 2.0], dtype=np.float64))


def test_batching_rule_factory_rejects_bad_inputs() -> None:
    """The batching-rule factory admits only executable kernels and known methods."""
    with pytest.raises(ValueError, match="must be an ExecutableCompilerADKernel"):
        make_executable_ad_kernel_batching_rule(cast("Any", object()))
    with pytest.raises(ValueError, match="method must be"):
        make_executable_ad_kernel_batching_rule(_linear_kernel(_SCALAR_MATRIX), method="hessian")


def _run_batching(
    kernel: ExecutableCompilerADKernel,
    method: str,
    args: tuple[object, ...],
    axes: tuple[Any, ...],
    out_axes: int = 0,
) -> FloatArray:
    """Invoke a batching rule and return the stacked float output."""
    rule = make_executable_ad_kernel_batching_rule(kernel, method=method)
    return cast(
        "FloatArray",
        rule(lambda *call_args: None, args, cast("tuple[int | None, ...]", axes), out_axes),
    )


def test_batching_rule_batches_value_gradient_jvp_and_vjp() -> None:
    """Explicit batching methods reproduce per-row kernel results."""
    kernel = _linear_kernel(_SCALAR_MATRIX)
    batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    tangents = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    cotangents = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    values_out = _run_batching(kernel, "value", (batch,), (0,))
    np.testing.assert_allclose(values_out, batch @ _SCALAR_MATRIX.T)

    gradient_out = _run_batching(kernel, "gradient", (batch,), (0,))
    np.testing.assert_allclose(gradient_out, np.tile(_SCALAR_MATRIX[0], (3, 1)))

    jvp_out = _run_batching(kernel, "jvp", (batch, tangents), (0, 0))
    np.testing.assert_allclose(jvp_out, tangents @ _SCALAR_MATRIX.T)

    vjp_out = _run_batching(kernel, "vjp", (batch, cotangents), (0, 0))
    np.testing.assert_allclose(vjp_out, cotangents @ _SCALAR_MATRIX)


def test_batching_rule_supports_unmapped_arguments_and_negative_axes() -> None:
    """Unmapped arguments broadcast per slice and negative axes normalise."""
    kernel = _linear_kernel(_SCALAR_MATRIX)
    values = np.array([1.0, 2.0], dtype=np.float64)
    tangents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    jvp_out = _run_batching(kernel, "jvp", (values, tangents), (None, -2))
    np.testing.assert_allclose(jvp_out, tangents @ _SCALAR_MATRIX.T)

    stacked_last = _run_batching(kernel, "value", (tangents,), (0,), out_axes=-1)
    np.testing.assert_allclose(stacked_last, (tangents @ _SCALAR_MATRIX.T).T)


def test_batching_rule_enforces_method_argument_counts() -> None:
    """Each explicit batching method fails closed on the wrong argument count."""
    kernel = _linear_kernel(_SCALAR_MATRIX)
    batch = np.ones((2, 2), dtype=np.float64)
    for method, args, axes, message in (
        ("value", (batch, batch), (0, 0), "value batching requires one argument"),
        ("gradient", (batch, batch), (0, 0), "gradient batching requires one argument"),
        ("jvp", (batch,), (0,), "JVP batching requires values and tangent"),
        ("vjp", (batch,), (0,), "VJP batching requires values and cotangent"),
        (
            "auto",
            (batch, batch, batch),
            (0, 0, 0),
            "supports one or two arguments",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _run_batching(kernel, method, args, axes)


def test_batching_rule_auto_dispatch_matches_dimensions() -> None:
    """Automatic dispatch selects value, JVP or VJP from slice dimensions."""
    kernel = _linear_kernel(_WIDE_MATRIX)
    batch = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tangents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cotangents = np.ones((2, 3), dtype=np.float64)

    np.testing.assert_allclose(
        _run_batching(kernel, "auto", (batch,), (0,)),
        batch @ _WIDE_MATRIX.T,
    )
    np.testing.assert_allclose(
        _run_batching(kernel, "auto", (batch, tangents), (0, 0)),
        tangents @ _WIDE_MATRIX.T,
    )
    np.testing.assert_allclose(
        _run_batching(kernel, "auto", (batch, cotangents), (0, 0)),
        cotangents @ _WIDE_MATRIX,
    )


def test_batching_rule_auto_dispatch_fails_closed_on_ambiguity() -> None:
    """Square kernels with both derivative callables refuse automatic dispatch."""
    kernel = _linear_kernel(_SQUARE_MATRIX)
    batch = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="ambiguous executable AD kernel batching method"):
        _run_batching(kernel, "auto", (batch, batch), (0, 0))


def test_batching_rule_auto_dispatch_uses_the_only_available_kernel() -> None:
    """A square kernel with a single derivative callable dispatches unambiguously."""
    jvp_only = _linear_kernel(_SQUARE_MATRIX, with_vjp=False)
    vjp_only = _linear_kernel(_SQUARE_MATRIX, with_jvp=False)
    batch = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    seconds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    np.testing.assert_allclose(
        _run_batching(jvp_only, "auto", (batch, seconds), (0, 0)),
        seconds @ _SQUARE_MATRIX.T,
    )
    np.testing.assert_allclose(
        _run_batching(vjp_only, "auto", (batch, seconds), (0, 0)),
        seconds @ _SQUARE_MATRIX,
    )


def test_batching_rule_auto_dispatch_rejects_unmatchable_second_argument() -> None:
    """A second slice matching neither dimension fails closed with guidance."""
    kernel = _linear_kernel(_WIDE_MATRIX)
    batch = np.ones((2, 2), dtype=np.float64)
    mismatched = np.ones((2, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="could not match the second argument"):
        _run_batching(kernel, "auto", (batch, mismatched), (0, 0))


def test_batching_rule_admission_rejects_malformed_arguments() -> None:
    """Batch-argument admission pins every fail-closed boundary."""
    kernel = _linear_kernel(_SCALAR_MATRIX)
    batch = np.ones((2, 2), dtype=np.float64)
    for args, axes, message in (
        ((), (), "requires at least one argument"),
        ((batch,), (0, 0), "axes must match argument count"),
        ((batch,), (cast("int", "0"),), "axes must be integers or None"),
        ((batch,), (None,), "requires at least one mapped axis"),
        ((np.ones((0, 2)),), (0,), "axes must be non-empty"),
        ((batch, np.ones((3, 2))), (0, 0), "axes must have the same length"),
        ((batch,), (2,), "out of bounds for argument rank"),
        ((np.float64(1.0),), (0,), "cannot map over a scalar"),
        ((np.array([True, False]),), (0,), "must be numeric"),
        ((np.array([[1.0, np.inf], [1.0, 1.0]]),), (0,), "only finite values"),
    ):
        with pytest.raises(ValueError, match=message):
            _run_batching(kernel, "value", args, axes)


def test_batching_rule_rejects_out_of_bounds_output_axis() -> None:
    """Output stacking fails closed when out_axes exceeds the result rank."""
    kernel = _linear_kernel(_SCALAR_MATRIX)
    batch = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="out_axes is out of bounds"):
        _run_batching(kernel, "value", (batch,), (0,), out_axes=5)


def test_batch_output_stacking_guards_unreachable_inconsistencies() -> None:
    """The output stacker rejects empty and shape-inconsistent slices."""
    with pytest.raises(ValueError, match="outputs must be non-empty"):
        _stack_executable_kernel_batch_outputs((), 0)
    with pytest.raises(ValueError, match="consistent shapes"):
        _stack_executable_kernel_batch_outputs(
            (np.ones(2, dtype=np.float64), np.ones(3, dtype=np.float64)),
            0,
        )


def test_batch_axis_normaliser_guards_scalar_rank() -> None:
    """The axis normaliser keeps its rank-zero guard for future callers."""
    with pytest.raises(ValueError, match="cannot map over a scalar"):
        _normalise_executable_kernel_batch_axis("axes[0]", 0, 0)


def test_compile_custom_derivative_rule_requires_rule() -> None:
    """The MLIR lowering admits only CustomDerivativeRule instances."""
    with pytest.raises(ValueError, match="requires a CustomDerivativeRule"):
        compile_custom_derivative_rule_to_mlir(
            cast("Any", object()),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_compile_custom_derivative_rule_emits_payload_and_metadata() -> None:
    """Default lowering embeds parameters, payload, Jacobian and metadata."""
    values = np.array([2.0, 3.0], dtype=np.float64)
    module = compile_custom_derivative_rule_to_mlir(_product_rule(), values)

    assert module.dialect == "scpn_diff"
    assert module.sha256 == hashlib.sha256(module.text.encode("utf-8")).hexdigest()
    assert 'scpn.rule = "product_pair"' in module.text
    assert 'scpn_diff.parameter %p0 {name = "a", trainable = true}' in module.text
    assert 'scpn_diff.parameter %p1 {name = "b", trainable = true}' in module.text
    assert "scpn_diff.value %0 {value = 6}" in module.text
    assert "scpn_diff.jacobian {row = 0, col = 0, value = 3}" in module.text
    assert "scpn_diff.jacobian {row = 0, col = 1, value = 2}" in module.text
    assert 'execution = "interchange_only"' in module.text
    assert "scpn.metadata" in module.text
    assert module.resource_counts["parameters"] == 2
    assert module.resource_counts["outputs"] == 1
    assert module.resource_counts["jacobian_nnz"] == 2
    assert module.resource_counts["trainable_parameters"] == 2
    assert module.metadata["rule"] == "product_pair"
    assert "no executable LLVM or JIT lowering" in str(module.metadata["claim_boundary"])


def test_compile_custom_derivative_rule_honours_config_switches() -> None:
    """Payload and metadata switches remove the corresponding sections."""
    values = np.array([2.0, 3.0], dtype=np.float64)
    module = compile_custom_derivative_rule_to_mlir(
        _product_rule(),
        values,
        DifferentiableMLIRCompileConfig(include_numeric_payload=False, include_metadata=False),
    )
    assert "scpn_diff.value" not in module.text
    assert "scpn_diff.jacobian" not in module.text
    assert "scpn.metadata" not in module.text


def test_compile_custom_derivative_rule_filters_near_zero_jacobian_entries() -> None:
    """Near-zero Jacobian entries stay out of the text but count as non-zero."""
    values = np.array([1.0e-16, 3.0], dtype=np.float64)
    module = compile_custom_derivative_rule_to_mlir(_product_rule(), values)
    assert "scpn_diff.jacobian {row = 0, col = 0, value = 3}" in module.text
    assert "col = 1" not in module.text
    assert module.resource_counts["jacobian_nnz"] == 2


def test_verifier_returns_trivial_evidence_when_disabled() -> None:
    """Disabling verification returns a passed record without executing kernels."""

    def broken_kernel(values: FloatArray) -> FloatArray:
        raise AssertionError("value kernel must not run when verify is disabled")

    verification = _verify_executable_ad_kernel(
        _product_rule(),
        np.array([2.0, 3.0], dtype=np.float64),
        broken_kernel,
        None,
        None,
        CompilerADExecutableConfig(verify=False),
        sample_tangent=None,
        sample_cotangent=None,
    )
    assert verification.passed
    assert verification.jvp_close is None
    assert verification.vjp_close is None
    assert verification.gradient_close is None
    assert verification.max_abs_error == 0.0


def test_verifier_confirms_matching_kernels_including_scalar_gradient() -> None:
    """Matching kernels verify value, JVP, VJP and the scalar gradient."""
    rule = _product_rule()

    def value_kernel(values: FloatArray) -> FloatArray:
        return np.array([values[0] * values[1]], dtype=np.float64)

    def jvp_kernel(values: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array(
            [tangent[0] * values[1] + values[0] * tangent[1]],
            dtype=np.float64,
        )

    def vjp_kernel(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array(
            [cotangent[0] * values[1], cotangent[0] * values[0]],
            dtype=np.float64,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        np.array([2.0, 3.0], dtype=np.float64),
        value_kernel,
        jvp_kernel,
        vjp_kernel,
        CompilerADExecutableConfig(),
        sample_tangent=np.array([1.0, -1.0], dtype=np.float64),
        sample_cotangent=np.array([2.0], dtype=np.float64),
    )
    assert verification.passed
    assert verification.value_close is True
    assert verification.jvp_close is True
    assert verification.vjp_close is True
    assert verification.gradient_close is True
    assert verification.max_abs_error == 0.0


def test_verifier_rejects_mismatched_probe_shapes() -> None:
    """Sample tangent and cotangent shapes must match the rule geometry."""
    rule = _product_rule()

    def value_kernel(values: FloatArray) -> FloatArray:
        return np.array([values[0] * values[1]], dtype=np.float64)

    def jvp_kernel(values: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array(
            [tangent[0] * values[1] + values[0] * tangent[1]],
            dtype=np.float64,
        )

    def vjp_kernel(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array(
            [cotangent[0] * values[1], cotangent[0] * values[0]],
            dtype=np.float64,
        )

    values = np.array([2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="sample_tangent shape must match"):
        _verify_executable_ad_kernel(
            rule,
            values,
            value_kernel,
            jvp_kernel,
            vjp_kernel,
            CompilerADExecutableConfig(),
            sample_tangent=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            sample_cotangent=None,
        )
    with pytest.raises(ValueError, match="sample_cotangent shape must match"):
        _verify_executable_ad_kernel(
            rule,
            values,
            value_kernel,
            None,
            vjp_kernel,
            CompilerADExecutableConfig(),
            sample_tangent=None,
            sample_cotangent=np.array([1.0, 2.0], dtype=np.float64),
        )


def test_verifier_fails_closed_on_divergent_kernels() -> None:
    """A kernel that disagrees with the rule reference cannot pass verification."""

    def wrong_value_kernel(values: FloatArray) -> FloatArray:
        return np.array([values[0] * values[1] + 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="kernel verification failed"):
        _verify_executable_ad_kernel(
            _product_rule(),
            np.array([2.0, 3.0], dtype=np.float64),
            wrong_value_kernel,
            None,
            None,
            CompilerADExecutableConfig(),
            sample_tangent=None,
            sample_cotangent=None,
        )


def test_verifier_skips_gradient_probe_for_vector_outputs() -> None:
    """Vector-valued rules verify VJP without a scalar-gradient probe."""

    def value_fn(values: FloatArray) -> FloatArray:
        return np.array([values[0], 2.0 * values[1]], dtype=np.float64)

    def vjp_rule(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        del values
        return np.array([cotangent[0], 2.0 * cotangent[1]], dtype=np.float64)

    rule = CustomDerivativeRule(name="diagonal_scale", value_fn=value_fn, vjp_rule=vjp_rule)
    verification = _verify_executable_ad_kernel(
        rule,
        np.array([1.0, 2.0], dtype=np.float64),
        value_fn,
        None,
        vjp_rule,
        CompilerADExecutableConfig(),
        sample_tangent=None,
        sample_cotangent=None,
    )
    assert verification.passed
    assert verification.jvp_close is None
    assert verification.vjp_close is True
    assert verification.gradient_close is None


def test_scalar_gradient_llvm_ir_builder_emits_bounded_kernel() -> None:
    """The scalar-gradient IR builder emits a bounded store-only kernel."""
    rule = _product_rule()

    def vjp_kernel(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array(
            [cotangent[0] * values[1], cotangent[0] * values[0]],
            dtype=np.float64,
        )

    ir = _compile_scalar_gradient_llvm_ir(
        rule,
        np.array([2.0, 3.0], dtype=np.float64),
        vjp_kernel,
    )
    assert "define void @product_pair_gradient(double* %out)" in ir
    assert "store double 3, double* %slot0" in ir
    assert "store double 2, double* %slot1" in ir
    assert 'native_llvm_jit = "blocked_until_native_codegen_backend_exists"' in ir


def test_scalar_gradient_llvm_ir_builder_declines_vector_outputs() -> None:
    """Vector-valued rules produce no scalar-gradient IR."""

    def value_fn(values: FloatArray) -> FloatArray:
        return np.array([values[0], values[1]], dtype=np.float64)

    def vjp_rule(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        del values
        return cotangent

    rule = CustomDerivativeRule(name="identity_pair", value_fn=value_fn, vjp_rule=vjp_rule)
    ir = _compile_scalar_gradient_llvm_ir(
        rule,
        np.array([1.0, 2.0], dtype=np.float64),
        vjp_rule,
    )
    assert ir == ""
