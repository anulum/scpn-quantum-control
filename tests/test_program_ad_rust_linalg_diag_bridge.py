# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust linalg diag bridge tests
# scpn-quantum-control -- Rust Program AD diag linalg bridge tests
"""Focused Rust bridge tests for Program AD diag gather/scatter replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)

_CONSTRUCT_WEIGHTS = np.array(
    [
        [0.2, 1.5, -0.4, 0.7],
        [-0.6, 0.3, 2.0, -1.25],
        [0.5, -0.8, 0.9, 1.1],
        [1.75, -0.2, 0.6, -0.5],
    ],
    dtype=np.float64,
)
_EXTRACT_WEIGHTS = np.array([1.25, -0.5], dtype=np.float64)


def _weighted_diag_construct_objective(values: Any) -> Any:
    """Return a scalar weighted offset-diagonal construction objective."""

    return np.sum(np.diag(values, k=1) * _CONSTRUCT_WEIGHTS)


def _weighted_diag_extract_objective(values: Any) -> Any:
    """Return a scalar weighted offset-diagonal extraction objective."""

    matrix = np.reshape(values, (3, 2))
    return np.sum(np.diag(matrix, k=-1) * _EXTRACT_WEIGHTS)


def _rust_parity_probe(
    objective: Any,
    values: NDArray[np.float64],
    expected_ops: set[str],
) -> None:
    """Assert bit-tight Rust replay parity for one diag objective."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    assert expected_ops <= {effect.operation for effect in result.program_ir.effects}

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, rtol=1.0e-12, atol=1.0e-12)
    assert "static_linalg_primitives" in rust_result.claim_boundary


def test_rust_bridge_replays_program_ad_diag_construct_nodes() -> None:
    """Rust Program AD replay should match Python adjoints for vector-to-matrix diag."""

    _rust_parity_probe(
        _weighted_diag_construct_objective,
        np.array([1.5, -2.0, 0.75], dtype=np.float64),
        {
            "linalg:diag:3:offset:1:construct:0",
            "linalg:diag:3:offset:1:construct:1",
            "linalg:diag:3:offset:1:construct:2",
        },
    )


def test_rust_bridge_replays_program_ad_diag_extract_nodes() -> None:
    """Rust Program AD replay should match Python adjoints for matrix-to-vector diag."""

    _rust_parity_probe(
        _weighted_diag_extract_objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        {
            "linalg:diag:3x2:offset:-1:extract:0",
            "linalg:diag:3x2:offset:-1:extract:1",
        },
    )
