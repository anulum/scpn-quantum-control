# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — full-program journeys across the admitted whole-program AD routes
"""One user program goes through every admitted route with the same identity and numbers.

The program composes an elementwise polynomial, a supported linear solve and a
scalar objective. Oracles: the analytic chain rule written out here, the Python
replay record of the same execution, and the bounded Rust effect-IR replay of
the same serialised program. A composition that fails inside library code is
refused with the stage located in the user's source; an exception the user's
own code raises propagates unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    WholeProgramStageRefusal,
    whole_program_value_and_grad,
)
from scpn_quantum_control.whole_program_ad_api import (
    WholeProgramReplayRecord,
    capture_whole_program_replay,
    replay_whole_program,
)

_A = np.array([[3.0, 1.0], [1.0, 2.0]])
_C = np.array([0.5, -1.0])
_POINT = np.array([0.7, -0.4])


def _journey(p: Any) -> Any:
    """Polynomial, then a linear solve, then a scalar objective."""
    b = p * p + 2.0 * p + _C
    x = np.linalg.solve(_A, b)
    return np.sum(x * x) + 3.0 * np.sum(x)


def _mixed(p: Any) -> Any:
    """Build an array from traced scalars, which the solve cannot take."""
    b = np.array([p[0] * p[0] + p[1], p[0] * p[1] - 1.0])
    x = np.linalg.solve(_A, b)
    return x[0] * x[0] + 3.0 * x[1]


def _analytic(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """Return the value and the chain-rule gradient ``(db/dp)^T A^-T (2x + 3)``."""
    b = p * p + 2.0 * p + _C
    x = np.linalg.solve(_A, b)
    outer = np.linalg.solve(_A.T, 2.0 * x + 3.0)
    return float(np.sum(x * x) + 3.0 * np.sum(x)), np.diag(2.0 * p + 2.0).T @ outer


def test_whole_program_backend_journeys_01_composed_program_matches_the_chain_rule() -> None:
    """The whole-program value and gradient equal the hand-written chain rule."""
    for point in (_POINT, np.array([-1.3, 0.25]), np.array([2.0, 1.5])):
        result = whole_program_value_and_grad(_journey, point)
        value, gradient = _analytic(point)
        assert result.value == pytest.approx(value, rel=1e-13, abs=1e-13)
        np.testing.assert_allclose(result.gradient, gradient, rtol=1e-12, atol=1e-12)
        assert result.program_ir is not None and result.method == "whole_program_ad"


def test_whole_program_backend_journeys_02_frozen_program_replays_across_admitted_routes() -> None:
    """Python replay and Rust effect-IR replay reproduce the saved hash and numbers."""
    result = whole_program_value_and_grad(_journey, _POINT)
    record = capture_whole_program_replay(result)
    assert result.program_ir is not None
    serialised = result.program_ir.serialization
    assert record.program_ir_sha256 == hashlib.sha256(serialised.encode()).hexdigest()
    # The record survives a JSON round trip unchanged, and the Python route replays it.
    restored = WholeProgramReplayRecord.from_json(record.to_json())
    assert restored == record
    report = replay_whole_program(restored, _journey)
    assert report.record.program_ir_sha256 == record.program_ir_sha256
    # The Rust route consumes the same serialised program.
    engine = pytest.importorskip("scpn_quantum_engine")
    assert engine is not None
    from scpn_quantum_control.program_ad_rust_bridge import (
        value_and_grad_program_ad_effect_ir_with_rust,
    )

    rust = value_and_grad_program_ad_effect_ir_with_rust(serialised, _POINT)
    assert rust.supported and rust.blocked_reasons == ()
    assert rust.supported_effect_count == rust.effect_count == len(result.program_ir.effects)
    assert rust.value == pytest.approx(record.value, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(rust.gradient, record.gradient, rtol=1e-12, atol=1e-12)
    assert (
        "rust" in result.polyglot_targets and "effect-IR replay" in result.polyglot_targets["rust"]
    )


def test_whole_program_backend_journeys_03_unsupported_mixed_composition_is_located() -> None:
    """An array built from traced scalars refuses at the solve line, naming the operation."""
    with pytest.raises(WholeProgramStageRefusal) as refused:
        whole_program_value_and_grad(_mixed, _POINT)
    stage = refused.value
    assert stage.filename == __file__
    assert stage.statement == "x = np.linalg.solve(_A, b)"
    assert stage.operation.endswith(":solve")
    assert "no value or derivative is issued" in str(stage)
    assert isinstance(stage.__cause__, TypeError)
    # The admitted program still runs afterwards: nothing was left behind by the refusal.
    assert whole_program_value_and_grad(_journey, _POINT).value == pytest.approx(
        _analytic(_POINT)[0]
    )


def test_package_refusals_are_not_relocated() -> None:
    """A refusal raised inside the package keeps its own type and message."""

    def scalar_output(p: Any) -> Any:
        """Return a vector instead of a scalar."""
        return p * 2.0

    with pytest.raises(ValueError, match="must return a whole-program AD scalar") as raised:
        whole_program_value_and_grad(scalar_output, _POINT)
    assert not isinstance(raised.value, WholeProgramStageRefusal)


def test_user_raised_exceptions_propagate_unchanged() -> None:
    """An exception the user's own code raises keeps its type; only library failures are located."""

    def stop() -> None:
        """Raise as user code may do deliberately."""
        raise RuntimeError("stopped by the user", 7)

    def stop_midway(p: Any) -> Any:
        """Stop after a traced product through the user's own helper."""
        product = p[0] * p[1]
        stop()
        return product

    with pytest.raises(RuntimeError, match="stopped by the user") as raised:
        whole_program_value_and_grad(stop_midway, _POINT)
    assert type(raised.value) is RuntimeError and raised.value.args[1] == 7
    # The admitted program still runs afterwards.
    assert whole_program_value_and_grad(_journey, _POINT).value == pytest.approx(
        _analytic(_POINT)[0]
    )
    record = json.loads(
        capture_whole_program_replay(whole_program_value_and_grad(_journey, _POINT)).to_json()
    )
    assert record["record"]["program_ir_sha256"] and record["digest"]
