# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD Rust linalg-array bridge tests
"""Tests for Rust Program AD replay of compact linalg-array primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import program_adjoint_value_and_grad
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)


def _multi_dot_weighted_objective(values: Any) -> Any:
    """Return a compact matrix-chain objective with matrix and scalar outputs."""

    left = np.reshape(values[:4], (2, 2))
    middle = np.reshape(values[4:8], (2, 2))
    right = np.reshape(values[8:12], (2, 2))
    vector_left = values[12:14]
    vector_right = values[14:16]
    matrix_weights = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float64)
    scalar_weight = 0.75
    return np.sum(
        np.linalg.multi_dot((left, middle, right)) * matrix_weights
    ) + scalar_weight * np.linalg.multi_dot((vector_left, middle, vector_right))


def _multi_dot_sample() -> NDArray[np.float64]:
    """Return a nonsingular static matrix-chain sample for Rust replay tests."""

    return np.array(
        [
            1.0,
            2.0,
            3.0,
            5.0,
            0.5,
            -1.0,
            2.0,
            1.5,
            2.0,
            0.25,
            -0.5,
            3.0,
            1.25,
            -0.75,
            0.5,
            2.5,
        ],
        dtype=np.float64,
    )


def test_rust_bridge_replays_program_ad_multi_dot_array_nodes() -> None:
    """The PyO3 bridge should replay compact multi_dot array nodes end to end."""

    pytest.importorskip("scpn_quantum_engine")

    sample = _multi_dot_sample()
    result = whole_program_value_and_grad(
        _multi_dot_weighted_objective,
        sample,
        parameters=tuple(Parameter(f"m{index}") for index in range(sample.size)),
    )
    assert result.program_ir is not None
    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:multi_dot:")] == [
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:0",
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:1",
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:2",
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:3",
        "linalg:multi_dot:2__2x2__2:out:scalar",
    ]

    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is True, rust.blocked_reasons
    _, reference = program_adjoint_value_and_grad(_multi_dot_weighted_objective, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-12, atol=1.0e-12)
    assert "static_linalg_primitives_value_and_gradient" in rust.claim_boundary
