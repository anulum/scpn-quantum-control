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

from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)

_EIGVALSH_WEIGHTS = np.array([0.75, -1.25], dtype=np.float64)


def _weighted_eigvalsh_objective(values: Any) -> Any:
    """Return a scalar weighted spectrum for a 2x2 symmetric matrix."""

    matrix = np.reshape(values, (2, 2))
    return np.sum(np.linalg.eigvalsh(matrix) * _EIGVALSH_WEIGHTS)


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
