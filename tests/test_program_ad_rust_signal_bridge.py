# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust signal bridge tests
# scpn-quantum-control -- Program AD Rust signal bridge tests
"""Tests for compact signal Program AD replay through the Rust bridge."""

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


def _signal_weighted_objective(values: Any) -> Any:
    """Return a compact signal objective covering convolution and correlation modes."""

    signal = values[:4]
    kernel = values[4:7]
    convolve_full_weights = values[7:13]
    convolve_same_weights = values[13:17]
    convolve_valid_weights = values[17:19]
    correlate_full_weights = values[19:25]
    correlate_same_weights = values[25:29]
    correlate_valid_weights = values[29:31]

    return (
        0.17 * np.sum(np.convolve(signal, kernel, mode="full") * convolve_full_weights)
        - 0.23 * np.sum(np.convolve(signal, kernel, mode="same") * convolve_same_weights)
        + 0.31 * np.sum(np.convolve(signal, kernel, mode="valid") * convolve_valid_weights)
        + 0.19 * np.sum(np.correlate(signal, kernel, mode="full") * correlate_full_weights)
        - 0.29 * np.sum(np.correlate(signal, kernel, mode="same") * correlate_same_weights)
        + 0.37 * np.sum(np.correlate(signal, kernel, mode="valid") * correlate_valid_weights)
    )


def _signal_sample() -> NDArray[np.float64]:
    """Return a finite sample covering dynamic signal, kernel, and weight parameters."""

    return np.array(
        [
            1.25,
            -0.75,
            2.0,
            0.5,
            -1.5,
            0.75,
            1.25,
            0.2,
            -0.4,
            0.6,
            -0.8,
            1.0,
            -1.2,
            -0.5,
            0.25,
            1.25,
            -0.75,
            1.4,
            -0.9,
            -0.15,
            0.35,
            -0.55,
            0.75,
            -0.95,
            1.15,
            -0.6,
            0.4,
            -0.2,
            0.8,
            0.25,
            -0.85,
        ],
        dtype=np.float64,
    )


def test_rust_bridge_replays_compact_signal_program_ad_nodes() -> None:
    """The Rust bridge should replay compact signal value+gradient IR."""

    pytest.importorskip("scpn_quantum_engine")

    sample = _signal_sample()
    result = whole_program_value_and_grad(
        _signal_weighted_objective,
        sample,
        parameters=tuple(Parameter(f"s{index}") for index in range(sample.size)),
    )
    assert result.program_ir is not None
    signal_ops = [node.op for node in result.ir_nodes if node.op.startswith("signal:")]
    assert signal_ops == [
        *(f"signal:convolve:left:4:right:3:mode:full:out:{index}" for index in range(6)),
        *(f"signal:convolve:left:4:right:3:mode:same:out:{index}" for index in range(4)),
        *(f"signal:convolve:left:4:right:3:mode:valid:out:{index}" for index in range(2)),
        *(f"signal:correlate:left:4:right:3:mode:full:out:{index}" for index in range(6)),
        *(f"signal:correlate:left:4:right:3:mode:same:out:{index}" for index in range(4)),
        *(f"signal:correlate:left:4:right:3:mode:valid:out:{index}" for index in range(2)),
    ]

    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is True, rust.blocked_reasons
    assert rust.value == pytest.approx(result.value, abs=1.0e-12)
    _, reference = program_adjoint_value_and_grad(_signal_weighted_objective, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-12, atol=1.0e-12)
    assert "static_signal_primitives" in rust.claim_boundary
