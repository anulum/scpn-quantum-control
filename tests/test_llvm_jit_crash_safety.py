# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — LLVM JIT crash safety tests
# scpn-quantum-control -- LLVM/JIT crash-safety gate tests
"""Crash-safety tests for native whole-program AD LLVM/JIT lowering."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from scpn_quantum_control.compiler import (
    analyse_whole_program_ad_native_lowering,
    compile_whole_program_ad_trace_to_native_llvm_jit,
    native_whole_program_ad_linalg_support,
)
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad

FloatVector = npt.NDArray[np.float64]


def test_native_llvm_jit_reports_unsupported_wide_determinant_before_compile() -> None:
    """Unsupported wide linalg must fail before native LLVM/JIT compilation."""

    def objective(values: FloatVector) -> object:
        matrix = np.diag(values[:20])
        return np.linalg.det(matrix) + np.sin(values[20])

    sample = np.linspace(1.1, 1.7, 21, dtype=np.float64)
    parameters = tuple(Parameter(f"theta_{index}") for index in range(sample.size))
    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert report.supported is False
    assert report.unsupported_ops == ("linalg:det:20x20",)
    assert report.unsupported_operation_count == 1
    assert "unsupported native ops: linalg:det:20x20" in report.fail_closed_reason
    with pytest.raises(ValueError, match="unsupported native ops: linalg:det:20x20"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_native_llvm_jit_rejects_nondifferentiable_selection_boundary() -> None:
    """Strict selection kernels must reject replay inputs on nondifferentiable ties."""

    def objective(values: FloatVector) -> object:
        return (
            np.maximum(values[0], values[1])
            + np.minimum(values[2], values[3])
            + values[4] * values[0]
        )

    sample = np.array([1.25, -0.25, 0.5, 1.5, 2.0], dtype=np.float64)
    parameters = (
        Parameter("left"),
        Parameter("right"),
        Parameter("lower"),
        Parameter("upper"),
        Parameter("scale"),
    )
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)

    with pytest.raises(ValueError, match="non-differentiable at equal inputs"):
        kernel.gradient(np.array([1.0, 1.0, 0.5, 1.5, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="non-differentiable at equal inputs"):
        kernel.batch_value_and_grad(
            np.array(
                [
                    [1.25, -0.25, 0.5, 1.5, 2.0],
                    [1.0, 1.0, 0.5, 1.5, 2.0],
                ],
                dtype=np.float64,
            )
        )


def test_native_llvm_jit_support_metadata_declares_fail_closed_boundaries() -> None:
    """Published support metadata must declare native linalg crash-safety cutoffs."""

    support = native_whole_program_ad_linalg_support()

    assert support["determinant_fail_closed_from"] == 20
    assert support["determinant_policy"] == "static_dense_native_or_fail_closed"
    assert support["inverse_fail_closed_from"] == 8
    assert support["solve_fail_closed_from"] == 8
    assert support["unsupported_policy"] == "fail_closed_report_before_compile"
