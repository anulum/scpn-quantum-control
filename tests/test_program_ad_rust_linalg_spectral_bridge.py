# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Rust Program AD spectral linalg bridge tests
"""Focused Rust bridge tests for Program AD spectral linalg replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)

_EIGVALSH_WEIGHTS = np.array([0.75, -1.25], dtype=np.float64)
_EIGVALS_WEIGHTS = np.array([0.75, -1.25], dtype=np.float64)
_EIGH_EIGENVALUE_WEIGHTS = np.array([0.75, -1.25], dtype=np.float64)
_EIGH_EIGENVECTOR_WEIGHTS = np.array([[0.2, -0.4], [0.6, 0.1]], dtype=np.float64)
_SVDVALS_WEIGHTS = np.array([0.5, -1.3], dtype=np.float64)
_PINV_WEIGHTS = np.array(
    [[0.4, -0.2, 0.3], [0.1, -0.5, 0.25]],
    dtype=np.float64,
)


def _weighted_eigvalsh_objective(values: Any) -> Any:
    """Return a scalar weighted spectrum for a 2x2 symmetric matrix."""

    matrix = np.reshape(values, (2, 2))
    return np.sum(np.linalg.eigvalsh(matrix) * _EIGVALSH_WEIGHTS)


def _weighted_eigvals_objective(values: Any) -> Any:
    """Return a scalar weighted spectrum for a real-simple 2x2 matrix."""

    matrix = np.reshape(values, (2, 2))
    return np.sum(np.linalg.eigvals(matrix) * _EIGVALS_WEIGHTS)


def _weighted_eigh_objective(values: Any) -> Any:
    """Return a scalar weighted eigensystem for a 2x2 symmetric matrix."""

    raw = np.reshape(values, (2, 2))
    matrix = 0.5 * (raw + raw.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return np.sum(eigenvalues * _EIGH_EIGENVALUE_WEIGHTS) + np.sum(
        eigenvectors * _EIGH_EIGENVECTOR_WEIGHTS
    )


def _weighted_svdvals_objective(values: Any) -> Any:
    """Return a scalar weighted singular-value spectrum for a 2x2 matrix."""

    matrix = np.reshape(values, (2, 2))
    return np.sum(np.linalg.svd(matrix, compute_uv=False) * _SVDVALS_WEIGHTS)


def _weighted_pinv_objective(values: Any) -> Any:
    """Return a scalar weighted pseudoinverse for a full-rank 3x2 matrix."""

    matrix = np.reshape(values, (3, 2))
    return np.sum(np.linalg.pinv(matrix, rcond=0.0) * _PINV_WEIGHTS)


def test_rust_bridge_replays_program_ad_2x2_eigvalsh_value_and_gradient() -> None:
    """Rust Program AD replay should match Python spectral adjoints for eigvalsh."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([2.0, 0.35, 0.35, 3.0], dtype=np.float64)

    result = whole_program_value_and_grad(
        _weighted_eigvalsh_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    assert {"linalg:eigvalsh:0", "linalg:eigvalsh:1"} <= {
        effect.operation for effect in result.program_ir.effects
    }

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)


def test_rust_bridge_replays_program_ad_2x2_eigh_value_and_gradient() -> None:
    """Rust Program AD replay should match Python eigensystem adjoints for eigh."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([2.0, 0.35, -0.2, 3.0], dtype=np.float64)

    result = whole_program_value_and_grad(
        _weighted_eigh_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    assert {
        "linalg:eigh:eigenvalue:2x2:L:0",
        "linalg:eigh:eigenvalue:2x2:L:1",
        "linalg:eigh:eigenvector:2x2:L:0:0",
        "linalg:eigh:eigenvector:2x2:L:1:0",
        "linalg:eigh:eigenvector:2x2:L:0:1",
        "linalg:eigh:eigenvector:2x2:L:1:1",
    } <= {effect.operation for effect in result.program_ir.effects}

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)


def test_rust_bridge_replays_program_ad_2x2_svdvals_value_and_gradient() -> None:
    """Rust Program AD replay should match Python singular-value adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([2.0, 0.3, -0.2, 1.1], dtype=np.float64)

    result = whole_program_value_and_grad(
        _weighted_svdvals_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    assert {"linalg:svdvals:2x2:0", "linalg:svdvals:2x2:1"} <= {
        effect.operation for effect in result.program_ir.effects
    }

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)


def test_rust_bridge_replays_program_ad_3x2_pinv_value_and_gradient() -> None:
    """Rust Program AD replay should match Python full-rank pinv adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([2.0, 0.2, 0.3, 1.4, 0.5, -0.7], dtype=np.float64)

    result = whole_program_value_and_grad(
        _weighted_pinv_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    pinv_operations = {effect.operation for effect in result.program_ir.effects}
    assert {
        "linalg:pinv:3x2:0:0:0",
        "linalg:pinv:3x2:0:0:1",
        "linalg:pinv:3x2:0:0:2",
        "linalg:pinv:3x2:0:1:0",
        "linalg:pinv:3x2:0:1:1",
        "linalg:pinv:3x2:0:1:2",
    } <= pinv_operations

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)


@pytest.mark.parametrize(
    "values",
    (
        np.array([2.0, 0.4, 0.15, 3.0], dtype=np.float64),
        np.array([3.0, 0.4, 0.15, 2.0], dtype=np.float64),
        np.array([2.0, 0.0, 1.0, 3.0], dtype=np.float64),
    ),
)
def test_rust_bridge_replays_program_ad_2x2_eigvals_value_and_gradient(
    values: NDArray[np.float64],
) -> None:
    """Rust Program AD replay should match Python spectral adjoints for eigvals."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    result = whole_program_value_and_grad(
        _weighted_eigvals_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    assert {"linalg:eigvals:2x2:0", "linalg:eigvals:2x2:1"} <= {
        effect.operation for effect in result.program_ir.effects
    }

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
