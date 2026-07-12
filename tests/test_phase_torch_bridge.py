# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch Phase Bridge Tests
"""Availability and fail-closed facade tests for the public Torch bridge."""

from __future__ import annotations

import numpy as np
import pytest
from _phase_torch_bridge_test_helpers import _objective

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    is_phase_torch_available,
    run_torch_compile_compatibility_audit,
    run_torch_func_compatibility_audit,
    run_torch_module_wrapper_audit,
    torch_autograd_qnn_value_and_grad,
    torch_bounded_qnn_layer,
    torch_bounded_qnn_module,
    torch_bounded_qnn_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


def test_torch_bridge_fails_closed_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when the optional PyTorch dependency cannot be imported."""

    def missing_torch() -> object:
        """Raise the deterministic optional-dependency failure for PyTorch."""
        raise ImportError("torch blocked")

    monkeypatch.setattr(torch_bridge, "_load_torch", missing_torch)
    assert not is_phase_torch_available()
    with pytest.raises(ImportError, match="torch blocked"):
        torch_parameter_shift_value_and_grad(_objective, np.array([0.2, -0.4], dtype=float))
    with pytest.raises(ImportError, match="torch blocked"):
        torch_bounded_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        torch_autograd_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        run_torch_func_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.2], dtype=float),
            params_batch=np.array([[0.2]], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        run_torch_compile_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        torch_bounded_qnn_module(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        torch_bounded_qnn_layer(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        run_torch_module_wrapper_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
