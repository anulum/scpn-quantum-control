# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust cumulative bridge tests
# scpn-quantum-control -- Program AD Rust cumulative bridge tests
"""Tests for compact cumulative Program AD replay through the Rust bridge."""

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


def _cumulative_weighted_objective(values: Any) -> Any:
    """Return a compact cumulative objective with flat and static-axis scans."""

    matrix = np.reshape(values[:6], (2, 3))
    cumsum_axis_weights = np.reshape(values[6:12], (2, 3))
    cumprod_axis_weights = np.reshape(values[12:18], (2, 3))
    diff_axis_weights = np.reshape(values[18:20], (2, 1))
    cumsum_flat_weights = values[20:26]

    return (
        np.sum(np.cumsum(matrix, axis=1) * cumsum_axis_weights)
        + 0.125 * np.sum(np.cumprod(matrix, axis=1) * cumprod_axis_weights)
        + np.sum(np.diff(matrix, n=2, axis=1) * diff_axis_weights)
        + 0.05 * np.sum(np.cumsum(matrix) * cumsum_flat_weights)
    )


def _cumulative_sample() -> NDArray[np.float64]:
    """Return a nonzero sample covering scan, product-scan, and difference rules."""

    return np.array(
        [
            1.25,
            -0.75,
            2.0,
            0.5,
            1.5,
            -1.25,
            0.1,
            -0.2,
            0.3,
            0.4,
            -0.5,
            0.6,
            1.0,
            0.75,
            -0.25,
            0.5,
            -1.5,
            2.0,
            1.75,
            -0.5,
            0.2,
            -0.1,
            0.4,
            -0.3,
            0.6,
            -0.7,
        ],
        dtype=np.float64,
    )


def test_rust_bridge_replays_compact_cumulative_program_ad_nodes() -> None:
    """The Rust bridge should replay compact cumulative value+gradient IR."""

    pytest.importorskip("scpn_quantum_engine")

    sample = _cumulative_sample()
    result = whole_program_value_and_grad(
        _cumulative_weighted_objective,
        sample,
        parameters=tuple(Parameter(f"c{index}") for index in range(sample.size)),
    )
    assert result.program_ir is not None
    cumulative_ops = [
        node.op for node in result.ir_nodes if node.op.startswith(("cumsum:", "cumprod:", "diff:"))
    ]
    assert cumulative_ops == [
        "cumsum:shape:2x3:axis:1:out:0",
        "cumsum:shape:2x3:axis:1:out:1",
        "cumsum:shape:2x3:axis:1:out:2",
        "cumsum:shape:2x3:axis:1:out:3",
        "cumsum:shape:2x3:axis:1:out:4",
        "cumsum:shape:2x3:axis:1:out:5",
        "cumprod:shape:2x3:axis:1:out:0",
        "cumprod:shape:2x3:axis:1:out:1",
        "cumprod:shape:2x3:axis:1:out:2",
        "cumprod:shape:2x3:axis:1:out:3",
        "cumprod:shape:2x3:axis:1:out:4",
        "cumprod:shape:2x3:axis:1:out:5",
        "diff:shape:2x3:n:2:axis:1:out:0",
        "diff:shape:2x3:n:2:axis:1:out:1",
        "cumsum:shape:2x3:axis:flat:out:0",
        "cumsum:shape:2x3:axis:flat:out:1",
        "cumsum:shape:2x3:axis:flat:out:2",
        "cumsum:shape:2x3:axis:flat:out:3",
        "cumsum:shape:2x3:axis:flat:out:4",
        "cumsum:shape:2x3:axis:flat:out:5",
    ]

    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is True, rust.blocked_reasons
    assert rust.value == pytest.approx(result.value, abs=1.0e-12)
    _, reference = program_adjoint_value_and_grad(_cumulative_weighted_objective, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-12, atol=1.0e-12)
    assert "static_cumulative_primitives_value_and_gradient" in rust.claim_boundary
