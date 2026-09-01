# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR matrix-native public validation tests
"""Exercise fail-closed dimension-generic matrix LLVM/JIT contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler import mlir_matrix_native_compilation as matrix_native
from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    ExecutableCompilerADKernel,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRule,
    PrimitiveTransformRule,
)

FloatArray = NDArray[np.float64]
VectorRule = Callable[[FloatArray], FloatArray]
DerivativeRule = Callable[[FloatArray, FloatArray], FloatArray]
Compiler = Callable[..., ExecutableCompilerADKernel]
LoweringFactory = Callable[..., Callable[..., ExecutableCompilerADKernel]]
TransformFactory = Callable[..., PrimitiveTransformRule]
DerivativeMode = Literal["both", "jvp", "vjp"]


def _quadratic_value(values: FloatArray) -> FloatArray:
    """Evaluate the 2x2 quadratic-form fixture."""
    matrix = values[:4].reshape(2, 2)
    vector = values[4:]
    return np.asarray([vector @ matrix @ vector], dtype=np.float64)


def _quadratic_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 quadratic-form JVP fixture."""
    matrix = values[:4].reshape(2, 2)
    vector = values[4:]
    matrix_tangent = tangent[:4].reshape(2, 2)
    vector_tangent = tangent[4:]
    return np.asarray(
        [
            vector @ matrix_tangent @ vector
            + vector_tangent @ matrix @ vector
            + vector @ matrix @ vector_tangent
        ],
        dtype=np.float64,
    )


def _quadratic_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 quadratic-form VJP fixture."""
    matrix = values[:4].reshape(2, 2)
    vector = values[4:]
    gradient = np.concatenate([np.outer(vector, vector).reshape(-1), (matrix + matrix.T) @ vector])
    return np.asarray(cotangent[0] * gradient, dtype=np.float64)


def _matrix_vector_value(values: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-vector fixture."""
    return np.asarray(values[:4].reshape(2, 2) @ values[4:], dtype=np.float64)


def _matrix_vector_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-vector JVP fixture."""
    matrix = values[:4].reshape(2, 2)
    vector = values[4:]
    matrix_tangent = tangent[:4].reshape(2, 2)
    vector_tangent = tangent[4:]
    return np.asarray(matrix_tangent @ vector + matrix @ vector_tangent, dtype=np.float64)


def _matrix_vector_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-vector VJP fixture."""
    matrix = values[:4].reshape(2, 2)
    vector = values[4:]
    return np.asarray(
        np.concatenate([np.outer(cotangent, vector).reshape(-1), matrix.T @ cotangent]),
        dtype=np.float64,
    )


def _matrix_matrix_value(values: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-matrix fixture."""
    left = values[:4].reshape(2, 2)
    right = values[4:].reshape(2, 2)
    return np.asarray((left @ right).reshape(-1), dtype=np.float64)


def _matrix_matrix_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-matrix JVP fixture."""
    left = values[:4].reshape(2, 2)
    right = values[4:].reshape(2, 2)
    left_tangent = tangent[:4].reshape(2, 2)
    right_tangent = tangent[4:].reshape(2, 2)
    return np.asarray(
        (left_tangent @ right + left @ right_tangent).reshape(-1),
        dtype=np.float64,
    )


def _matrix_matrix_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-matrix VJP fixture."""
    left = values[:4].reshape(2, 2)
    right = values[4:].reshape(2, 2)
    cotangent_matrix = cotangent.reshape(2, 2)
    return np.asarray(
        np.concatenate(
            [
                (cotangent_matrix @ right.T).reshape(-1),
                (left.T @ cotangent_matrix).reshape(-1),
            ]
        ),
        dtype=np.float64,
    )


def _trace_value(values: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-trace fixture."""
    return np.asarray([np.trace(values.reshape(2, 2))], dtype=np.float64)


def _trace_jvp(_values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-trace JVP fixture."""
    return np.asarray([np.trace(tangent.reshape(2, 2))], dtype=np.float64)


def _trace_vjp(_values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 matrix-trace VJP fixture."""
    return np.asarray(cotangent[0] * np.eye(2).reshape(-1), dtype=np.float64)


def _frobenius_value(values: FloatArray) -> FloatArray:
    """Evaluate the 2x2 Frobenius-squared fixture."""
    return np.asarray([values @ values], dtype=np.float64)


def _frobenius_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 Frobenius-squared JVP fixture."""
    return np.asarray([2.0 * values @ tangent], dtype=np.float64)


def _frobenius_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the 2x2 Frobenius-squared VJP fixture."""
    return np.asarray(2.0 * cotangent[0] * values, dtype=np.float64)


@dataclass(frozen=True)
class MatrixNativeCase:
    """One public dimension-generic native matrix primitive contract."""

    name: str
    values: tuple[float, ...]
    output_size: int
    value_rule: VectorRule
    jvp_rule: DerivativeRule
    vjp_rule: DerivativeRule
    compiler: Compiler
    lowering_factory: LoweringFactory
    transform_factory: TransformFactory

    @property
    def sample_values(self) -> FloatArray:
        """Return a fresh contiguous sample input."""
        return np.asarray(self.values, dtype=np.float64)

    @property
    def sample_tangent(self) -> FloatArray:
        """Return a non-zero tangent aligned with the sample input."""
        return np.linspace(0.1, 0.1 * len(self.values), len(self.values), dtype=np.float64)

    @property
    def sample_cotangent(self) -> FloatArray:
        """Return a non-zero cotangent aligned with the primitive output."""
        return np.linspace(0.25, 0.25 * self.output_size, self.output_size, dtype=np.float64)

    @property
    def identity(self) -> str:
        """Return the public primitive identity for this fixture."""
        return f"scpn.compiler_ad.native:{self.name}@1"

    def derivative_rule(self, mode: DerivativeMode = "both") -> CustomDerivativeRule:
        """Build a complete or deliberately one-sided derivative contract."""
        jvp = self.jvp_rule if mode in {"both", "jvp"} else None
        vjp = self.vjp_rule if mode in {"both", "vjp"} else None
        return CustomDerivativeRule(
            name=f"native_{self.name}_{mode}_validation_rule",
            value_fn=self.value_rule,
            jvp_rule=jvp,
            vjp_rule=vjp,
            parameter_names=tuple(f"x{index}" for index in range(len(self.values))),
            trainable=(True,) * len(self.values),
        )


CASES = (
    MatrixNativeCase(
        "matrix_quadratic_form",
        (2.0, -1.0, 0.5, 3.0, 1.5, -2.0),
        1,
        _quadratic_value,
        _quadratic_jvp,
        _quadratic_vjp,
        compiler_mlir.compile_matrix_quadratic_form_ad_to_native_llvm_jit,
        compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_lowering_rule,
        compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_primitive_transform,
    ),
    MatrixNativeCase(
        "matrix_vector_product",
        (2.0, -1.0, 0.5, 3.0, 1.5, -2.0),
        2,
        _matrix_vector_value,
        _matrix_vector_jvp,
        _matrix_vector_vjp,
        compiler_mlir.compile_matrix_vector_product_ad_to_native_llvm_jit,
        compiler_mlir.make_matrix_vector_product_native_llvm_jit_lowering_rule,
        compiler_mlir.make_matrix_vector_product_native_llvm_jit_primitive_transform,
    ),
    MatrixNativeCase(
        "matrix_matrix_product",
        (2.0, -1.0, 0.5, 3.0, 1.5, -2.0, 0.25, 4.0),
        4,
        _matrix_matrix_value,
        _matrix_matrix_jvp,
        _matrix_matrix_vjp,
        compiler_mlir.compile_matrix_matrix_product_ad_to_native_llvm_jit,
        compiler_mlir.make_matrix_matrix_product_native_llvm_jit_lowering_rule,
        compiler_mlir.make_matrix_matrix_product_native_llvm_jit_primitive_transform,
    ),
    MatrixNativeCase(
        "matrix_trace",
        (2.0, -1.0, 0.5, 3.0),
        1,
        _trace_value,
        _trace_jvp,
        _trace_vjp,
        compiler_mlir.compile_matrix_trace_ad_to_native_llvm_jit,
        compiler_mlir.make_matrix_trace_native_llvm_jit_lowering_rule,
        compiler_mlir.make_matrix_trace_native_llvm_jit_primitive_transform,
    ),
    MatrixNativeCase(
        "matrix_frobenius_norm_squared",
        (2.0, -1.0, 0.5, 3.0),
        1,
        _frobenius_value,
        _frobenius_jvp,
        _frobenius_vjp,
        compiler_mlir.compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit,
        compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule,
        compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_matrix_native_compile_rejects_invalid_public_contracts(case: MatrixNativeCase) -> None:
    """Reject invalid rules, dimensions, backends, and sample widths."""
    rule = case.derivative_rule()
    values = case.sample_values

    with pytest.raises(ValueError, match="rule must be a CustomDerivativeRule"):
        case.compiler(cast(CustomDerivativeRule, object()), dimension=2, sample_values=values)
    with pytest.raises(ValueError, match="dimension must be positive"):
        case.compiler(rule, dimension=0, sample_values=values)
    with pytest.raises(ValueError, match="backend='native_llvm_jit'"):
        case.compiler(
            rule,
            dimension=2,
            sample_values=values,
            config=CompilerADExecutableConfig(),
        )
    with pytest.raises(ValueError, match="sample values"):
        case.compiler(rule, dimension=2, sample_values=values[:-1])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_matrix_native_transform_rejects_invalid_public_contracts(
    case: MatrixNativeCase,
) -> None:
    """Reject malformed transform inputs and execute the default config path."""
    rule = case.derivative_rule()
    values = case.sample_values

    with pytest.raises(ValueError, match="rule must be a CustomDerivativeRule"):
        case.transform_factory(
            case.identity,
            cast(CustomDerivativeRule, object()),
            dimension=2,
            sample_values=values,
        )
    with pytest.raises(ValueError, match="backend='native_llvm_jit'"):
        case.transform_factory(
            case.identity,
            rule,
            dimension=2,
            sample_values=values,
            config=CompilerADExecutableConfig(),
        )
    with pytest.raises(ValueError, match="sample values"):
        case.transform_factory(
            case.identity,
            rule,
            dimension=2,
            sample_values=values[:-1],
        )

    transform = case.transform_factory(
        case.identity,
        rule,
        dimension=2,
        sample_values=values,
    )
    assert transform.identity.key == case.identity
    assert transform.lowering_metadata is not None
    assert transform.lowering_metadata["native_backend"] == "native_llvm_jit"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_matrix_native_lowering_requires_samples_and_accepts_runtime_override(
    case: MatrixNativeCase,
) -> None:
    """Require a sample source and honor explicit runtime derivative samples."""
    rule = case.derivative_rule()
    values = case.sample_values
    tangent = case.sample_tangent
    cotangent = case.sample_cotangent
    lowering = case.lowering_factory(dimension=2)

    with pytest.raises(ValueError, match="lowering requires sample_values"):
        lowering(rule)

    runtime_kernel = lowering(
        rule,
        values,
        CompilerADExecutableConfig(backend="native_llvm_jit"),
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    np.testing.assert_allclose(runtime_kernel.value(values), case.value_rule(values))

    captured_lowering = case.lowering_factory(
        dimension=2,
        sample_values=values,
        config=CompilerADExecutableConfig(backend="native_llvm_jit"),
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    captured_kernel = captured_lowering(rule)
    np.testing.assert_allclose(captured_kernel.value(values), case.value_rule(values))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_matrix_native_kernel_rejects_malformed_runtime_vectors(case: MatrixNativeCase) -> None:
    """Reject malformed values, tangents, and cotangents at public kernels."""
    values = case.sample_values
    tangent = case.sample_tangent
    cotangent = case.sample_cotangent
    kernel = case.compiler(
        case.derivative_rule(),
        dimension=2,
        sample_values=values,
        config=CompilerADExecutableConfig(backend="native_llvm_jit"),
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    short_values = values[:-1]
    short_tangent = tangent[:-1]
    wrong_cotangent = np.zeros(case.output_size + 1, dtype=np.float64)

    with pytest.raises(ValueError, match="kernel requires"):
        kernel.value(short_values)
    with pytest.raises(ValueError, match="kernel requires"):
        kernel.jvp(short_values, tangent)
    with pytest.raises(ValueError, match="tangent value"):
        kernel.jvp(values, short_tangent)
    with pytest.raises(ValueError, match="kernel requires"):
        kernel.vjp(short_values, cotangent)
    with pytest.raises(ValueError, match="cotangent value"):
        kernel.vjp(values, wrong_cotangent)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_matrix_native_compile_accepts_single_derivative_directions(
    case: MatrixNativeCase,
) -> None:
    """Compile real JVP-only and VJP-only kernels without fabricating parity."""
    values = case.sample_values
    tangent = case.sample_tangent
    cotangent = case.sample_cotangent

    jvp_kernel = case.compiler(
        case.derivative_rule("jvp"),
        dimension=2,
        sample_values=values,
    )
    np.testing.assert_allclose(jvp_kernel.jvp(values, tangent), case.jvp_rule(values, tangent))
    assert jvp_kernel.vjp_kernel is None
    with pytest.raises(ValueError, match="has no VJP rule"):
        jvp_kernel.vjp(values, cotangent)

    vjp_kernel = case.compiler(
        case.derivative_rule("vjp"),
        dimension=2,
        sample_values=values,
    )
    np.testing.assert_allclose(
        vjp_kernel.vjp(values, cotangent),
        case.vjp_rule(values, cotangent),
    )
    assert vjp_kernel.jvp_kernel is None
    with pytest.raises(ValueError, match="has no JVP rule"):
        vjp_kernel.jvp(values, tangent)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_matrix_native_compile_rejects_gradient_verification_mismatch(
    case: MatrixNativeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an injected native gradient fault through the public compiler."""
    compile_native = cast(
        Callable[[str, str], Mapping[str, Any]],
        vars(matrix_native)["_compile_native_llvm_jit_functions"],
    )

    def compile_with_fault(llvm_ir: str, base_symbol: str) -> dict[str, Any]:
        functions = dict(compile_native(llvm_ir, base_symbol))
        native_gradient = functions["gradient"]

        def faulty_gradient(values: Any, output: Any) -> None:
            native_gradient(values, output)
            output[0] = float(output[0]) + 1.0

        functions["gradient"] = faulty_gradient
        return functions

    monkeypatch.setattr(
        matrix_native,
        "_compile_native_llvm_jit_functions",
        compile_with_fault,
    )
    with pytest.raises(ValueError, match="gradient verification failed"):
        case.compiler(
            case.derivative_rule(),
            dimension=2,
            sample_values=case.sample_values,
            config=CompilerADExecutableConfig(backend="native_llvm_jit", verify=False),
        )
