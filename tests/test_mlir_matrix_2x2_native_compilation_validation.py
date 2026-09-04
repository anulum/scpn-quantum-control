# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR matrix 2x2 native validation tests
"""Exercise fail-closed invariants of native 2x2 LLVM/JIT adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.compiler.mlir as compiler_mlir
from scpn_quantum_control.compiler import mlir_matrix_2x2_native_compilation as matrix_2x2
from scpn_quantum_control.compiler.mlir import CompilerADExecutableConfig
from scpn_quantum_control.differentiable import CustomDerivativeRule

FloatArray = NDArray[np.float64]
Compiler = Callable[..., object]


@dataclass(frozen=True)
class OutputGuardCase:
    """One unreachable-through-closure native output-width invariant."""

    helper_name: str
    values: tuple[float, ...]
    vector: tuple[float, ...] | None
    label: str | None
    invalid_output_size: int
    message: str


OUTPUT_GUARD_CASES = (
    OutputGuardCase(
        "_call_native_matrix_2x2_determinant_unary",
        (2.0, -1.0, 0.5, 3.0),
        None,
        None,
        2,
        "output_size must be one or four",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_determinant_binary",
        (2.0, -1.0, 0.5, 3.0),
        (0.1, 0.2, 0.3, 0.4),
        "tangent",
        2,
        "output_size must be one or four",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_inverse_unary",
        (2.0, -1.0, 0.5, 3.0),
        None,
        None,
        2,
        "output_size must be four",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_solve_unary",
        (2.0, -1.0, 0.5, 3.0, 1.0, 2.0),
        None,
        None,
        4,
        "output_size must be two or six",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_solve_binary",
        (2.0, -1.0, 0.5, 3.0, 1.0, 2.0),
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        "tangent",
        4,
        "output_size must be two or six",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_eigenvalues_unary",
        (3.0, 1.0, 0.5, 1.0),
        None,
        None,
        3,
        "output_size must be two or four",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_eigenvalues_binary",
        (3.0, 1.0, 0.5, 1.0),
        (0.1, 0.2, 0.3, 0.4),
        "tangent",
        3,
        "output_size must be two or four",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_eigensystem_unary",
        (3.0, 1.0, 0.5, 1.0),
        None,
        None,
        5,
        "output_size must be four or six",
    ),
    OutputGuardCase(
        "_call_native_matrix_2x2_eigensystem_binary",
        (3.0, 1.0, 0.5, 1.0),
        (0.1, 0.2, 0.3, 0.4),
        "tangent",
        5,
        "output_size must be four or six",
    ),
)


@pytest.mark.parametrize("case", OUTPUT_GUARD_CASES, ids=lambda case: case.helper_name)
def test_native_output_adapters_reject_impossible_closure_widths(case: OutputGuardCase) -> None:
    """Reject output widths that public compiler closures never generate."""
    helper = cast(Callable[..., FloatArray], getattr(matrix_2x2, case.helper_name))
    values = np.asarray(case.values, dtype=np.float64)

    def no_op(*_args: object) -> None:
        """Represent a native function that must not run after validation fails."""

    with pytest.raises(ValueError, match=case.message):
        if case.vector is None:
            helper(no_op, values, case.invalid_output_size)
        else:
            helper(
                no_op,
                values,
                np.asarray(case.vector, dtype=np.float64),
                case.label,
                case.invalid_output_size,
            )


def _dummy_rule(name: str, input_size: int) -> CustomDerivativeRule:
    """Build a VJP-bearing rule for pre-execution failure paths."""
    return CustomDerivativeRule(
        name=name,
        value_fn=lambda _values: np.zeros(1, dtype=np.float64),
        vjp_rule=lambda _values, _cotangent: np.zeros(input_size, dtype=np.float64),
    )


def test_inverse_and_determinant_transform_reject_invalid_matrix_samples() -> None:
    """Reject singular inverse samples and malformed determinant transform samples."""
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        compiler_mlir.compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
            _dummy_rule("singular_inverse_validation", 4),
            sample_values=np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="requires four sample values"):
        compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform(
            "scpn.compiler_ad.native:invalid_determinant_width@1",
            _dummy_rule("invalid_determinant_width", 4),
            sample_values=np.ones(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class GradientGuardCase:
    """One public compiler native-gradient provenance guard."""

    name: str
    compiler: Compiler
    values: tuple[float, ...]
    cotangent_size: int
    message: str


GRADIENT_GUARD_CASES = (
    GradientGuardCase(
        "determinant",
        compiler_mlir.compile_matrix_2x2_determinant_ad_to_native_llvm_jit,
        (2.0, -1.0, 0.5, 3.0),
        1,
        "determinant gradient verification failed",
    ),
    GradientGuardCase(
        "inverse",
        compiler_mlir.compile_matrix_2x2_inverse_ad_to_native_llvm_jit,
        (2.0, -1.0, 0.5, 3.0),
        4,
        "inverse sum-gradient provenance verification failed",
    ),
    GradientGuardCase(
        "solve",
        compiler_mlir.compile_matrix_2x2_solve_ad_to_native_llvm_jit,
        (2.0, -1.0, 0.5, 3.0, 1.0, 2.0),
        2,
        "solve sum-gradient provenance verification failed",
    ),
    GradientGuardCase(
        "eigenvalues",
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit,
        (3.0, 1.0, 0.5, 1.0),
        2,
        "eigenvalue sum-gradient provenance verification failed",
    ),
    GradientGuardCase(
        "eigensystem",
        compiler_mlir.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit,
        (3.0, 1.0, 0.5, 1.0),
        6,
        "eigensystem sum-gradient provenance verification failed",
    ),
)


@pytest.mark.parametrize("case", GRADIENT_GUARD_CASES, ids=lambda case: case.name)
def test_public_compilers_reject_native_gradient_provenance_mismatch(
    case: GradientGuardCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when a native sum-gradient disagrees with its rule."""
    monkeypatch.setattr(np, "allclose", lambda *_args, **_kwargs: False)
    values = np.asarray(case.values, dtype=np.float64)
    rule = CustomDerivativeRule(
        name=f"native_matrix_2x2_{case.name}_gradient_guard",
        value_fn=lambda _values: np.zeros(case.cotangent_size, dtype=np.float64),
        vjp_rule=lambda _values, _cotangent: np.zeros(values.size, dtype=np.float64),
    )

    with pytest.raises(ValueError, match=case.message):
        case.compiler(
            rule,
            sample_values=values,
            config=CompilerADExecutableConfig(backend="native_llvm_jit", verify=False),
        )
