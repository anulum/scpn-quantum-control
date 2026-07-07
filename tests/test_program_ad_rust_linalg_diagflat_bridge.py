# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Rust Program AD diagflat linalg bridge tests
"""Focused Rust bridge tests for Program AD diagflat replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)

_MAIN_DIAGONAL_WEIGHTS = np.array(
    [[0.4, -0.7, 0.2], [0.9, -0.2, 0.5], [-0.3, 0.6, 0.3]],
    dtype=np.float64,
)
_OFFSET_DIAGONAL_WEIGHTS = np.array(
    [[0.5, 1.5, -0.4], [0.25, -0.6, 2.0], [0.8, -1.1, 0.35]],
    dtype=np.float64,
)
_MATRIX_SOURCE_WEIGHTS = np.arange(1.0, 17.0, dtype=np.float64).reshape(4, 4) / 8.0


def _weighted_main_diagonal_objective(values: Any) -> Any:
    """Return a scalar weighted main-diagonal construction for a 3-vector."""

    return np.sum(np.diagflat(values) * _MAIN_DIAGONAL_WEIGHTS)


def _weighted_offset_diagonal_objective(values: Any) -> Any:
    """Return a scalar weighted super-diagonal construction for a 2-vector."""

    return np.sum(np.diagflat(values, k=1) * _OFFSET_DIAGONAL_WEIGHTS)


def _weighted_matrix_source_objective(values: Any) -> Any:
    """Return a scalar weighted diagonal construction for a flattened 2x2 source."""

    matrix = np.reshape(values, (2, 2))
    return np.sum(np.diagflat(matrix) * _MATRIX_SOURCE_WEIGHTS)


def _rust_parity_probe(
    objective: Any,
    values: NDArray[np.float64],
    expected_ops: set[str],
) -> None:
    """Assert bit-tight Rust replay parity for one diagflat objective."""

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
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)


def test_rust_bridge_replays_program_ad_main_diagonal_diagflat() -> None:
    """Rust Program AD replay should match Python adjoints for np.diagflat."""

    _rust_parity_probe(
        _weighted_main_diagonal_objective,
        np.array([1.5, -2.0, 0.75], dtype=np.float64),
        {
            "linalg:diagflat:3:offset:0:construct:0",
            "linalg:diagflat:3:offset:0:construct:1",
            "linalg:diagflat:3:offset:0:construct:2",
        },
    )


def test_rust_bridge_replays_program_ad_offset_diagonal_diagflat() -> None:
    """Rust Program AD replay should match Python adjoints for offset diagonals."""

    _rust_parity_probe(
        _weighted_offset_diagonal_objective,
        np.array([0.5, -1.25], dtype=np.float64),
        {
            "linalg:diagflat:2:offset:1:construct:0",
            "linalg:diagflat:2:offset:1:construct:1",
        },
    )


def test_rust_bridge_replays_program_ad_matrix_source_diagflat() -> None:
    """Rust Program AD replay should flatten matrix sources exactly like NumPy."""

    _rust_parity_probe(
        _weighted_matrix_source_objective,
        np.array([0.5, -1.25, 2.0, 3.5], dtype=np.float64),
        {
            "linalg:diagflat:2x2:offset:0:construct:0",
            "linalg:diagflat:2x2:offset:0:construct:1",
            "linalg:diagflat:2x2:offset:0:construct:2",
            "linalg:diagflat:2x2:offset:0:construct:3",
        },
    )
