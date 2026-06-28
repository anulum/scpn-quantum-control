# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable programming benchmark Rust edge tests
"""Rust replay edge tests for differentiable-programming benchmarks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _differentiable_programming_benchmark_edge_helpers import (
    _fake_whole_program,
    _program_ir,
    _whole_program_result,
)

from scpn_quantum_control.benchmarks import differentiable_programming as dp


def test_program_ad_rust_scalar_interpreter_fails_closed_and_checks_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust scalar replay benchmark should fail closed on unsupported or divergent replay."""

    valid_result = _whole_program_result(
        program_ir=_program_ir(
            effects=(SimpleNamespace(kind="primitive", ordering=0),),
            control_regions=(SimpleNamespace(kind="runtime_branch", entered=True),),
            phi_nodes=(
                SimpleNamespace(
                    target="phi:runtime_branch:0",
                    selected="executed_true",
                    control_region=0,
                    incoming=("true", "false"),
                ),
            ),
        ),
        gradient=np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float64),
    )
    monkeypatch.setattr(dp, "whole_program_value_and_grad", _fake_whole_program(valid_result))

    monkeypatch.setattr(
        dp,
        "interpret_program_ad_effect_ir_with_rust",
        lambda _ir, _values: SimpleNamespace(
            supported=False,
            value=None,
            blocked_reasons=("native extension unavailable",),
            supported_effect_count=0,
        ),
    )
    blocked = dp._program_ad_rust_scalar_interpreter_case()
    assert blocked.blocked_reasons == ("native extension unavailable",)

    monkeypatch.setattr(
        dp,
        "interpret_program_ad_effect_ir_with_rust",
        lambda _ir, _values: SimpleNamespace(
            supported=True,
            value=valid_result.value,
            blocked_reasons=(),
            supported_effect_count=1,
        ),
    )
    monkeypatch.setattr(
        dp,
        "value_and_grad_program_ad_effect_ir_with_rust",
        lambda _ir, _values: SimpleNamespace(
            supported=False,
            value=None,
            gradient=None,
            blocked_reasons=("value-grad unavailable",),
            supported_effect_count=0,
        ),
    )
    blocked_value_grad = dp._program_ad_rust_scalar_interpreter_case()
    assert blocked_value_grad.blocked_reasons == ("value-grad unavailable",)

    monkeypatch.setattr(
        dp,
        "interpret_program_ad_effect_ir_with_rust",
        lambda _ir, _values: SimpleNamespace(
            supported=True,
            value=valid_result.value + 1.0,
            blocked_reasons=(),
            supported_effect_count=1,
        ),
    )
    with pytest.raises(ValueError, match="value diverged"):
        dp._program_ad_rust_scalar_interpreter_case()

    monkeypatch.setattr(
        dp,
        "interpret_program_ad_effect_ir_with_rust",
        lambda _ir, _values: SimpleNamespace(
            supported=True,
            value=valid_result.value,
            blocked_reasons=(),
            supported_effect_count=0,
        ),
    )
    with pytest.raises(ValueError, match="execute every effect"):
        dp._program_ad_rust_scalar_interpreter_case()

    monkeypatch.setattr(
        dp,
        "interpret_program_ad_effect_ir_with_rust",
        lambda _ir, _values: SimpleNamespace(
            supported=True,
            value=valid_result.value,
            blocked_reasons=(),
            supported_effect_count=1,
        ),
    )
    for replay_result, match in (
        (
            SimpleNamespace(
                supported=True,
                value=valid_result.value + 1.0,
                gradient=valid_result.gradient,
                blocked_reasons=(),
                supported_effect_count=1,
            ),
            "value diverged",
        ),
        (
            SimpleNamespace(
                supported=True,
                value=valid_result.value,
                gradient=valid_result.gradient,
                blocked_reasons=(),
                supported_effect_count=0,
            ),
            "missed effects",
        ),
        (
            SimpleNamespace(
                supported=True,
                value=valid_result.value,
                gradient=valid_result.gradient + 1.0,
                blocked_reasons=(),
                supported_effect_count=1,
            ),
            "gradient diverged",
        ),
    ):
        monkeypatch.setattr(
            dp,
            "value_and_grad_program_ad_effect_ir_with_rust",
            lambda _ir, _values, replay_result=replay_result: replay_result,
        )
        with pytest.raises(ValueError, match=match):
            dp._program_ad_rust_scalar_interpreter_case()


def test_program_ad_rust_scalar_interpreter_rejects_invalid_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust scalar replay benchmark should reject malformed emitted IR contracts."""

    for result, match in (
        (_whole_program_result(program_ir=None), "requires program IR"),
        (
            _whole_program_result(program_ir=_program_ir(control_regions=(), phi_nodes=())),
            "runtime branch metadata",
        ),
        (
            _whole_program_result(
                program_ir=_program_ir(
                    alias_edges=(SimpleNamespace(source="a", target="b", kind="alias"),),
                ),
            ),
            "must not emit alias edges",
        ),
    ):
        monkeypatch.setattr(dp, "whole_program_value_and_grad", _fake_whole_program(result))
        with pytest.raises(ValueError, match=match):
            dp._program_ad_rust_scalar_interpreter_case()
