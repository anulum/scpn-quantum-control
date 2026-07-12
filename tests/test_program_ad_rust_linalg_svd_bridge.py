# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust linalg svd bridge tests
# scpn-quantum-control -- Program AD Rust SVD bridge tests
"""Tests for Rust Program AD replay of compact singular-value primitives."""

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


def _svdvals_3x2_sample() -> NDArray[np.float64]:
    """Return a full-rank rectangular sample with distinct singular values."""

    return np.array([3.0, 0.5, -1.0, 2.0, 0.25, -0.75], dtype=np.float64)


def _svdvals_2x3_sample() -> NDArray[np.float64]:
    """Return a wide full-rank sample with distinct singular values."""

    return np.array([2.0, -0.5, 1.25, 0.75, 3.0, -1.5], dtype=np.float64)


def _svdvals_3x2_weighted_objective(values: Any) -> Any:
    """Return a scalar objective over static 3x2 singular values."""

    matrix = np.reshape(values, (3, 2))
    weights = np.array([0.7, -1.2], dtype=np.float64)
    return np.sum(np.linalg.svd(matrix, compute_uv=False) * weights)


def _svdvals_2x3_weighted_objective(values: Any) -> Any:
    """Return a scalar objective over static 2x3 singular values."""

    matrix = np.reshape(values, (2, 3))
    weights = np.array([-0.4, 1.1], dtype=np.float64)
    return np.sum(np.linalg.svd(matrix, compute_uv=False) * weights)


@pytest.mark.parametrize(
    ("sample", "objective", "shape_label"),
    (
        (_svdvals_3x2_sample(), _svdvals_3x2_weighted_objective, "3x2"),
        (_svdvals_2x3_sample(), _svdvals_2x3_weighted_objective, "2x3"),
    ),
)
def test_rust_bridge_replays_program_ad_rectangular_svdvals_nodes(
    sample: NDArray[np.float64],
    objective: Any,
    shape_label: str,
) -> None:
    """The PyO3 bridge should replay rectangular svdvals nodes end to end."""

    pytest.importorskip("scpn_quantum_engine")

    result = whole_program_value_and_grad(
        objective,
        sample,
        parameters=tuple(Parameter(f"v{index}") for index in range(sample.size)),
    )
    assert result.program_ir is not None
    svd_nodes = [node.op for node in result.ir_nodes if node.op.startswith("linalg:svdvals:")]
    assert svd_nodes == [
        f"linalg:svdvals:{shape_label}:0",
        f"linalg:svdvals:{shape_label}:1",
    ]

    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is True, rust.blocked_reasons
    _, reference = program_adjoint_value_and_grad(objective, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-12, atol=1.0e-12)
    assert "static_linalg_primitives" in rust.claim_boundary
    assert "value_and_gradient" in rust.claim_boundary
