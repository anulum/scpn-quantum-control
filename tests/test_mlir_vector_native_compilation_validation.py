# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR vector-native public validation tests
"""Exercise fail-closed dimension-generic vector LLVM/JIT contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler import mlir_vector_native_compilation as vector_native
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


def _dot_value(values: FloatArray) -> FloatArray:
    """Evaluate the two-dimensional dot-product fixture."""
    return np.asarray([values[:2] @ values[2:]], dtype=np.float64)


def _dot_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the two-dimensional dot-product JVP fixture."""
    return np.asarray(
        [values[2:] @ tangent[:2] + values[:2] @ tangent[2:]],
        dtype=np.float64,
    )


def _dot_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the two-dimensional dot-product VJP fixture."""
    return np.asarray(
        np.concatenate([values[2:], values[:2]]) * cotangent[0],
        dtype=np.float64,
    )


def _squared_norm_value(values: FloatArray) -> FloatArray:
    """Evaluate the squared-norm fixture."""
    return np.asarray([values @ values], dtype=np.float64)


def _squared_norm_jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
    """Evaluate the squared-norm JVP fixture."""
    return np.asarray([2.0 * values @ tangent], dtype=np.float64)


def _squared_norm_vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
    """Evaluate the squared-norm VJP fixture."""
    return np.asarray(2.0 * values * cotangent[0], dtype=np.float64)


@dataclass(frozen=True)
class VectorNativeCase:
    """One public dimension-generic native vector primitive contract."""

    name: str
    values: tuple[float, ...]
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
    def identity(self) -> str:
        """Return the public primitive identity for this fixture."""
        return f"scpn.compiler_ad.native:{self.name}@1"

    def derivative_rule(self, *, with_vjp: bool = True) -> CustomDerivativeRule:
        """Build a complete or deliberately JVP-only derivative contract."""
        return CustomDerivativeRule(
            name=f"native_{self.name}_{'full' if with_vjp else 'jvp'}_validation_rule",
            value_fn=self.value_rule,
            jvp_rule=self.jvp_rule,
            vjp_rule=self.vjp_rule if with_vjp else None,
        )


CASES = (
    VectorNativeCase(
        "vector_dot",
        (1.0, 2.0, -3.0, 4.0),
        _dot_value,
        _dot_jvp,
        _dot_vjp,
        compiler_mlir.compile_vector_dot_ad_to_native_llvm_jit,
        compiler_mlir.make_vector_dot_native_llvm_jit_lowering_rule,
        compiler_mlir.make_vector_dot_native_llvm_jit_primitive_transform,
    ),
    VectorNativeCase(
        "vector_squared_norm",
        (1.5, -2.0),
        _squared_norm_value,
        _squared_norm_jvp,
        _squared_norm_vjp,
        compiler_mlir.compile_vector_squared_norm_ad_to_native_llvm_jit,
        compiler_mlir.make_vector_squared_norm_native_llvm_jit_lowering_rule,
        compiler_mlir.make_vector_squared_norm_native_llvm_jit_primitive_transform,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vector_native_compile_rejects_invalid_public_contracts(case: VectorNativeCase) -> None:
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
def test_vector_native_transform_rejects_invalid_public_contracts(
    case: VectorNativeCase,
) -> None:
    """Reject malformed transform rules, backends, and sample widths."""
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


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vector_native_lowering_requires_or_uses_captured_samples(
    case: VectorNativeCase,
) -> None:
    """Require a sample source and compile from explicitly captured samples."""
    rule = case.derivative_rule()
    values = case.sample_values
    lowering = case.lowering_factory(dimension=2)

    with pytest.raises(ValueError, match="lowering requires sample_values"):
        lowering(rule)

    captured_lowering = case.lowering_factory(
        dimension=2,
        sample_values=values,
        config=CompilerADExecutableConfig(backend="native_llvm_jit"),
        sample_tangent=case.sample_tangent,
        sample_cotangent=np.ones(1, dtype=np.float64),
    )
    kernel = captured_lowering(rule)
    np.testing.assert_allclose(kernel.value(values), case.value_rule(values))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vector_native_kernel_rejects_malformed_runtime_vectors(case: VectorNativeCase) -> None:
    """Reject malformed values, tangents, and cotangents at public kernels."""
    values = case.sample_values
    tangent = case.sample_tangent
    kernel = case.compiler(
        case.derivative_rule(),
        dimension=2,
        sample_values=values,
        sample_tangent=tangent,
        sample_cotangent=np.ones(1, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="kernel requires"):
        kernel.value(values[:-1])
    with pytest.raises(ValueError, match="kernel requires"):
        kernel.jvp(values[:-1], tangent)
    with pytest.raises(ValueError, match="tangent value"):
        kernel.jvp(values, tangent[:-1])
    with pytest.raises(ValueError, match="cotangent value"):
        kernel.vjp(values, np.ones(2, dtype=np.float64))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vector_native_compile_accepts_jvp_only_rules(case: VectorNativeCase) -> None:
    """Compile a real JVP-only kernel without fabricating reverse-mode parity."""
    values = case.sample_values
    tangent = case.sample_tangent
    kernel = case.compiler(
        case.derivative_rule(with_vjp=False),
        dimension=2,
        sample_values=values,
        sample_tangent=tangent,
    )

    np.testing.assert_allclose(kernel.jvp(values, tangent), case.jvp_rule(values, tangent))
    assert kernel.vjp_kernel is None


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vector_native_compile_rejects_gradient_verification_mismatch(
    case: VectorNativeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an injected native gradient mismatch through the public compiler."""
    monkeypatch.setattr(np, "allclose", lambda *_args, **_kwargs: False)

    with pytest.raises(ValueError, match="gradient verification failed"):
        case.compiler(
            case.derivative_rule(),
            dimension=2,
            sample_values=case.sample_values,
            config=CompilerADExecutableConfig(backend="native_llvm_jit", verify=False),
        )


@dataclass(frozen=True)
class OutputGuardCase:
    """One unreachable-through-closure native output-width invariant."""

    helper_name: str
    values: tuple[float, ...]
    vector: tuple[float, ...] | None
    label: str | None
    invalid_output_size: int


OUTPUT_GUARD_CASES = (
    OutputGuardCase("_call_native_vector_dot_unary", CASES[0].values, None, None, 2),
    OutputGuardCase(
        "_call_native_vector_dot_binary", CASES[0].values, (0.1, 0.2, 0.3, 0.4), "tangent", 2
    ),
    OutputGuardCase("_call_native_vector_squared_norm_unary", CASES[1].values, None, None, 3),
    OutputGuardCase(
        "_call_native_vector_squared_norm_binary", CASES[1].values, (0.1, 0.2), "tangent", 3
    ),
)


@pytest.mark.parametrize("case", OUTPUT_GUARD_CASES, ids=lambda case: case.helper_name)
def test_native_output_adapters_reject_impossible_closure_widths(case: OutputGuardCase) -> None:
    """Reject output widths that public compiler closures never generate."""
    helper = cast(Callable[..., FloatArray], getattr(vector_native, case.helper_name))
    values = np.asarray(case.values, dtype=np.float64)

    def no_op(*_args: object) -> None:
        """Represent a native function that must not run after validation fails."""

    with pytest.raises(ValueError, match="output_size"):
        if case.vector is None:
            helper(no_op, values, 2, case.invalid_output_size)
        else:
            helper(
                no_op,
                values,
                np.asarray(case.vector, dtype=np.float64),
                case.label,
                2,
                case.invalid_output_size,
            )
