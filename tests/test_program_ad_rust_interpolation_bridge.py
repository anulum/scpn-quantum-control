# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust interpolation bridge tests
# scpn-quantum-control -- Program AD Rust interpolation bridge tests
"""Tests for compact interpolation Program AD replay through the Rust bridge."""

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

_INTERP_GRID = np.array([-2.0, -0.25, 1.25, 3.5], dtype=np.float64)


def _interpolation_weighted_objective(values: Any) -> Any:
    """Return an interpolation objective covering interior and boundary samples."""

    samples = values[:4]
    fp_values = values[4:8]
    default_boundary_weights = values[8:12]
    static_boundary_weights = values[12:16]

    default_boundary = np.interp(samples, _INTERP_GRID, fp_values)
    static_boundary = np.interp(
        samples,
        _INTERP_GRID,
        fp_values,
        left=-1.75,
        right=2.25,
    )
    return 0.31 * np.sum(default_boundary * default_boundary_weights) - 0.47 * np.sum(
        static_boundary * static_boundary_weights
    )


def _interpolation_sample() -> NDArray[np.float64]:
    """Return finite samples, interpolation values, and objective weights."""

    return np.array(
        [
            -2.5,
            -1.1,
            0.4,
            4.2,
            1.25,
            -0.75,
            2.0,
            0.5,
            0.2,
            -0.4,
            0.6,
            -0.8,
            -0.5,
            0.25,
            1.25,
            -0.75,
        ],
        dtype=np.float64,
    )


def test_rust_bridge_replays_compact_interpolation_program_ad_nodes() -> None:
    """The Rust bridge should replay compact interpolation value+gradient IR."""

    pytest.importorskip("scpn_quantum_engine")

    sample = _interpolation_sample()
    result = whole_program_value_and_grad(
        _interpolation_weighted_objective,
        sample,
        parameters=tuple(Parameter(f"i{index}") for index in range(sample.size)),
    )
    assert result.program_ir is not None
    interpolation_ops = [
        node.op for node in result.ir_nodes if node.op.startswith("interpolation:interp:")
    ]
    assert interpolation_ops == [
        *(
            "interpolation:interp:samples:4:grid:-2,-0.25,1.25,3.5:"
            f"left:none:right:none:out:{index}"
            for index in range(4)
        ),
        *(
            "interpolation:interp:samples:4:grid:-2,-0.25,1.25,3.5:"
            f"left:-1.75:right:2.25:out:{index}"
            for index in range(4)
        ),
    ]

    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is True, rust.blocked_reasons
    assert rust.value == pytest.approx(result.value, abs=1.0e-12)
    _, reference = program_adjoint_value_and_grad(_interpolation_weighted_objective, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-12, atol=1.0e-12)
    assert "static_interpolation_primitives" in rust.claim_boundary
