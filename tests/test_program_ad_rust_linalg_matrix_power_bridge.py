# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Rust Program AD matrix-power linalg bridge tests
"""Focused Rust bridge tests for Program AD matrix-power replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)

_POSITIVE_POWER_WEIGHTS = np.array([[0.5, -1.0], [0.75, 1.25]], dtype=np.float64)
_NEGATIVE_POWER_WEIGHTS = np.array([[1.25, -0.5], [0.2, 0.75]], dtype=np.float64)
_ZERO_POWER_WEIGHTS = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


def _weighted_matrix_power_objective(values: Any) -> Any:
    """Return a scalar objective that exercises positive, negative, and zero powers."""

    matrix = np.reshape(values, (2, 2))
    return (
        np.sum(np.linalg.matrix_power(matrix, 3) * _POSITIVE_POWER_WEIGHTS)
        + np.sum(np.linalg.matrix_power(matrix, -2) * _NEGATIVE_POWER_WEIGHTS)
        + np.sum(np.linalg.matrix_power(matrix, 0) * _ZERO_POWER_WEIGHTS)
    )


def test_rust_bridge_replays_program_ad_matrix_power_nodes() -> None:
    """Rust Program AD replay should match Python adjoints for static matrix powers."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    sample = np.array([1.5, 0.4, 0.2, 1.1], dtype=np.float64)
    result = whole_program_value_and_grad(
        _weighted_matrix_power_objective,
        sample,
        parameters=tuple(Parameter(f"x{index}") for index in range(sample.size)),
    )

    assert result.program_ir is not None
    matrix_power_ops = [
        effect.operation
        for effect in result.program_ir.effects
        if effect.operation is not None and effect.operation.startswith("linalg:matrix_power:")
    ]
    assert {
        "linalg:matrix_power:2x2:power:3:0:0",
        "linalg:matrix_power:2x2:power:-2:0:0",
        "linalg:matrix_power:2x2:power:0:0:0",
    } <= set(matrix_power_ops)

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, rtol=1.0e-12, atol=1.0e-12)
    assert "static_linalg_primitives" in rust_result.claim_boundary
