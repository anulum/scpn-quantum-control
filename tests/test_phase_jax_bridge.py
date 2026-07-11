# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Bridge Availability Tests
"""Tests for optional JAX availability and import failure boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import (
    _objective,
)

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    check_jax_parameter_shift_agreement,
    is_phase_jax_available,
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    jax_parameter_shift_value_and_grad,
)


def test_phase_jax_bridge_fails_closed_when_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise ImportError("blocked")

    monkeypatch.setattr(jax_bridge, "_load_jax", unavailable)

    assert not is_phase_jax_available()
    with pytest.raises(ImportError, match="blocked"):
        jax_parameter_shift_value_and_grad(_objective, np.array([0.2, -0.4], dtype=float))
    with pytest.raises(ImportError, match="blocked"):
        check_jax_parameter_shift_agreement(
            _objective,
            lambda values: values,
            np.array([0.2, -0.4], dtype=float),
        )
    with pytest.raises(ImportError, match="blocked"):
        jax_native_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="blocked"):
        jax_custom_vjp_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )


def test_phase_jax_bridge_import_runtime_failure_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JAX import-time compatibility errors should fail closed as unavailable."""

    def incompatible_jax() -> tuple[object, object]:
        raise AttributeError("numpy dtype surface is incompatible with this JAX build")

    monkeypatch.setattr(jax_bridge, "_load_jax", incompatible_jax)

    assert not is_phase_jax_available()
