# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust linalg eig bridge tests
# scpn-quantum-control -- Program AD Rust eig bridge tests
"""Tests for Rust Program AD replay of compact eig primitives."""

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


def _eig_sample() -> NDArray[np.float64]:
    """Return a real-simple non-symmetric 2x2 matrix sample."""

    return np.array([2.0, 0.3, 0.1, 3.0], dtype=np.float64)


def _eig_weighted_objective(values: Any) -> Any:
    """Return a scalar objective over real-simple eig outputs."""

    matrix = np.reshape(values, (2, 2))
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalue_weights = np.array([0.75, -1.25], dtype=np.float64)
    eigenvector_weights = np.array([[0.2, -0.4], [0.6, 0.1]], dtype=np.float64)
    return np.sum(eigenvalues * eigenvalue_weights) + np.sum(eigenvectors * eigenvector_weights)


def test_rust_bridge_replays_program_ad_real_simple_eig_nodes() -> None:
    """The PyO3 bridge should replay real-simple eig nodes end to end."""

    pytest.importorskip("scpn_quantum_engine")

    sample = _eig_sample()
    result = whole_program_value_and_grad(
        _eig_weighted_objective,
        sample,
        parameters=tuple(Parameter(f"v{index}") for index in range(sample.size)),
    )
    assert result.program_ir is not None
    eig_nodes = [node.op for node in result.ir_nodes if node.op.startswith("linalg:eig:")]
    assert eig_nodes == [
        "linalg:eig:eigenvalue:2x2:0",
        "linalg:eig:eigenvalue:2x2:1",
        "linalg:eig:eigenvector:2x2:0:0",
        "linalg:eig:eigenvector:2x2:1:0",
        "linalg:eig:eigenvector:2x2:0:1",
        "linalg:eig:eigenvector:2x2:1:1",
    ]

    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is True, rust.blocked_reasons
    _, reference = program_adjoint_value_and_grad(_eig_weighted_objective, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-12, atol=1.0e-12)
    assert "static_linalg_primitives" in rust.claim_boundary
    assert "value_and_gradient" in rust.claim_boundary
