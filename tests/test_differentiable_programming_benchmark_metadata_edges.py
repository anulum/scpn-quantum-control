# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable programming benchmark metadata edge tests
"""Metadata and optional-backend edge tests for differentiable-programming benchmarks."""

from __future__ import annotations

import numpy as np
import pytest
from _differentiable_programming_benchmark_edge_helpers import (
    _benchmark_row,
    _gradient,
    _quantum_row,
)

from scpn_quantum_control.benchmarks import differentiable_programming as dp
from scpn_quantum_control.benchmarks.differentiable_programming import (
    DifferentiableProgrammingBenchmarkResult,
    QuantumGradientBenchmarkResult,
)


def test_benchmark_result_validation_rejects_empty_blocked_reason() -> None:
    """Differentiable benchmark rows should reject malformed blocker metadata."""

    row = _benchmark_row()
    assert row.passed is True
    with pytest.raises(ValueError, match="blocked_reasons"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="diagnostic",
            value=1.0,
            gradient=_gradient(2),
            analytic_gradient=_gradient(2),
            max_abs_gradient_error=0.0,
            adjoint_supported=False,
            max_abs_adjoint_error=None,
            claim_boundary="diagnostic",
            blocked_reasons=("",),
        )


def test_quantum_gradient_validation_rejects_remaining_malformed_fields() -> None:
    """Quantum-gradient rows should reject every remaining malformed metadata field."""

    gradient = _gradient(2)
    with pytest.raises(ValueError, match="category"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient.copy(),
            analytic_gradient=gradient.copy(),
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="value"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=np.nan,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient.copy(),
            analytic_gradient=gradient.copy(),
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="finite-difference"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient.copy(),
            analytic_gradient=gradient.copy(),
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=np.inf,
            verification_passed=True,
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient.copy(),
            analytic_gradient=gradient.copy(),
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=8,
            claim_boundary="",
        )


@pytest.mark.parametrize(
    ("torch_available", "jax_available", "expected_optional"),
    (
        (False, False, ()),
        (True, False, ("torch_state", "torch_transform", "torch_compile")),
        (False, True, ("jax_native", "jax_pytree", "jax_sharding")),
    ),
)
def test_quantum_gradient_suite_optional_backend_gates(
    monkeypatch: pytest.MonkeyPatch,
    torch_available: bool,
    jax_available: bool,
    expected_optional: tuple[str, ...],
) -> None:
    """Quantum-gradient optional rows should be gated independently by backend."""

    monkeypatch.setattr(dp, "is_phase_torch_available", lambda: torch_available)
    monkeypatch.setattr(dp, "is_phase_jax_available", lambda: jax_available)
    monkeypatch.setattr(
        dp,
        "_torch_registered_phase_qnode_statevector_case",
        lambda: _quantum_row("torch_state"),
    )
    monkeypatch.setattr(
        dp,
        "_torch_registered_phase_qnode_func_transform_case",
        lambda: _quantum_row("torch_transform"),
    )
    monkeypatch.setattr(
        dp,
        "_torch_registered_phase_qnode_compile_case",
        lambda: _quantum_row("torch_compile"),
    )
    monkeypatch.setattr(
        dp,
        "_jax_registered_phase_qnode_native_transform_case",
        lambda: _quantum_row("jax_native"),
    )
    monkeypatch.setattr(
        dp,
        "_jax_registered_phase_qnode_pytree_transform_case",
        lambda: _quantum_row("jax_pytree"),
    )
    monkeypatch.setattr(
        dp,
        "_jax_registered_phase_qnode_sharding_transform_case",
        lambda: _quantum_row("jax_sharding"),
    )

    rows = dp.run_quantum_gradient_benchmark_suite()

    assert tuple(row.case_id for row in rows[3:]) == expected_optional


def test_structured_numeric_interpolation_fails_closed_outside_grid() -> None:
    """Structured numeric directional checks should reject out-of-grid samples."""

    source = np.zeros(16, dtype=np.float64)
    direction = np.zeros(16, dtype=np.float64)
    source[13:16] = np.array([-0.5, 0.5, 1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="interpolation samples"):
        dp._structured_numeric_value_and_directional(source, direction)
