# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Program AD Rust bridge wrappers
"""Tests for the extracted Program AD Rust bridge wrappers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control.program_ad_rust_bridge import (
    RustProgramADInterpreterResult,
    RustProgramADValueAndGradientResult,
    interpret_program_ad_effect_ir_with_rust,
    value_and_grad_program_ad_effect_ir_with_rust,
)


@dataclass(frozen=True)
class _IR:
    serialization: str


def _install_fake_engine(monkeypatch: pytest.MonkeyPatch, engine: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", engine)


def test_value_and_gradient_bridge_is_shared_by_module_and_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct and facade imports should call the same extracted Rust bridge."""

    calls: list[tuple[str, list[float]]] = []
    fake_engine = ModuleType("scpn_quantum_engine")

    def replay(serialization: str, inputs: list[float]) -> str:
        calls.append((serialization, inputs))
        return json.dumps(
            {
                "supported": True,
                "value": 3.5,
                "gradient": [1.0, -2.0],
                "parameter_targets": ["%0", "%1"],
                "effect_count": 4,
                "supported_effect_count": 4,
                "blocked_reasons": [],
                "claim_boundary": "bounded_rust_program_ad_ir_scalar_primitives_value_and_gradient_executed_branch_no_alias_no_llvm_jit",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = replay
    _install_fake_engine(monkeypatch, fake_engine)
    program_ir = _IR(serialization="program-ir-json")

    direct = value_and_grad_program_ad_effect_ir_with_rust(
        program_ir,
        np.array([0.25, -0.5], dtype=np.float64),
    )
    facade = differentiable_facade.value_and_grad_program_ad_effect_ir_with_rust(
        program_ir,
        [0.25, -0.5],
    )

    assert isinstance(direct, RustProgramADValueAndGradientResult)
    assert isinstance(facade, RustProgramADValueAndGradientResult)
    assert calls == [
        ("program-ir-json", [0.25, -0.5]),
        ("program-ir-json", [0.25, -0.5]),
    ]
    assert direct.supported == facade.supported
    assert direct.value == facade.value
    assert direct.parameter_targets == facade.parameter_targets
    assert direct.effect_count == facade.effect_count
    assert direct.supported_effect_count == facade.supported_effect_count
    assert direct.blocked_reasons == facade.blocked_reasons
    assert direct.claim_boundary == facade.claim_boundary
    np.testing.assert_allclose(direct.gradient, np.array([1.0, -2.0], dtype=np.float64))
    np.testing.assert_allclose(direct.gradient, facade.gradient)


def test_forward_interpreter_bridge_normalises_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward interpreter wrapper should parse Rust JSON into a typed result."""

    fake_engine = ModuleType("scpn_quantum_engine")

    def replay(serialization: str, inputs: list[float]) -> str:
        assert serialization == "program-ir-json"
        assert inputs == [2.0]
        return json.dumps(
            {
                "supported": True,
                "value": 9.0,
                "effect_count": 3,
                "supported_effect_count": 3,
                "blocked_reasons": [],
                "claim_boundary": "bounded_rust_program_ad_ir_scalar_primitives_executed_branch_no_alias_no_llvm_jit",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_forward = replay
    _install_fake_engine(monkeypatch, fake_engine)

    result = interpret_program_ad_effect_ir_with_rust("program-ir-json", [2])

    assert isinstance(result, RustProgramADInterpreterResult)
    assert result.supported is True
    assert result.value == pytest.approx(9.0)
    assert result.supported_effect_count == 3
    assert "no_llvm_jit" in result.claim_boundary


def test_bridge_fails_closed_when_native_extension_or_export_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional native extension failures should stay structured and fail-closed."""

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    missing_extension = value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])
    assert missing_extension.supported is False
    assert missing_extension.gradient.size == 0
    assert missing_extension.blocked_reasons == (
        "scpn_quantum_engine native extension is not built",
    )

    fake_engine = ModuleType("scpn_quantum_engine")
    _install_fake_engine(monkeypatch, fake_engine)
    missing_export = value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])
    assert missing_export.supported is False
    assert missing_export.blocked_reasons == (
        "scpn_quantum_engine native extension lacks Program AD value+gradient replay",
    )

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    missing_forward_extension = interpret_program_ad_effect_ir_with_rust("{}", [0.0])
    assert missing_forward_extension.supported is False
    assert missing_forward_extension.blocked_reasons == (
        "scpn_quantum_engine native extension is not built",
    )


def test_bridge_rejects_malformed_inputs_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bridge validation should reject unsafe coercions and malformed Rust payloads."""

    with pytest.raises(ValueError, match="must contain real numeric scalars"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [True])
    with pytest.raises(ValueError, match="must be a non-empty string"):
        interpret_program_ad_effect_ir_with_rust("", [0.0])

    fake_engine = ModuleType("scpn_quantum_engine")

    def invalid_json(_serialization: str, _inputs: list[float]) -> str:
        return "{"

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_forward = invalid_json
    _install_fake_engine(monkeypatch, fake_engine)
    with pytest.raises(ValueError, match="returned invalid JSON"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def non_json_object(_serialization: str, _inputs: list[float]) -> str:
        return "[]"

    engine_exports.program_ad_effect_ir_interpret_forward = non_json_object
    with pytest.raises(ValueError, match="payload must be a JSON object"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def non_text_forward(_serialization: str, _inputs: list[float]) -> int:
        return 1

    engine_exports.program_ad_effect_ir_interpret_forward = non_text_forward
    with pytest.raises(ValueError, match="must return JSON text"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def malformed_value_and_gradient(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": True,
                "value": 1.0,
                "gradient": "bad",
                "parameter_targets": [],
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": [],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = malformed_value_and_gradient
    with pytest.raises(ValueError, match="gradient must be a JSON list"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])

    def non_text_value_and_gradient(_serialization: str, _inputs: list[float]) -> int:
        return 1

    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = non_text_value_and_gradient
    with pytest.raises(ValueError, match="must return JSON text"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])

    with pytest.raises(ValueError, match="must be one-dimensional"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", cast(Any, [[0.0]]))
    with pytest.raises(ValueError, match="must contain finite values"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [float("nan")])
    with pytest.raises(ValueError, match="must be a rectangular numeric array"):
        value_and_grad_program_ad_effect_ir_with_rust(
            "{}",
            cast(Any, [[1.0], [1.0, 2.0]]),
        )

    def bad_forward_count(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": False,
                "value": None,
                "effect_count": True,
                "supported_effect_count": 0,
                "blocked_reasons": ["blocked"],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_effect_ir_interpret_forward = bad_forward_count
    with pytest.raises(ValueError, match="effect_count must be an integer"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def bad_forward_boundary(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 0,
                "blocked_reasons": ["blocked"],
                "claim_boundary": 3,
            }
        )

    engine_exports.program_ad_effect_ir_interpret_forward = bad_forward_boundary
    with pytest.raises(ValueError, match="claim_boundary must be a string"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def bad_value_and_gradient_targets(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": False,
                "value": None,
                "gradient": [],
                "parameter_targets": "bad",
                "effect_count": 1,
                "supported_effect_count": 0,
                "blocked_reasons": ["blocked"],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = (
        bad_value_and_gradient_targets
    )
    with pytest.raises(ValueError, match="parameter target must be a list"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "supported": "yes",
                "value": None,
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported flag must be boolean",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": -1,
                "supported_effect_count": 0,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "counts must be non-negative",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 2,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported count exceeds effect count",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("",),
                "claim_boundary": "bounded",
            },
            "blocked reasons must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported state is inconsistent",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "",
            },
            "claim boundary must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": float("inf"),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "value must be finite",
        ),
    ],
)
def test_interpreter_result_rejects_invalid_states(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Interpreter result container should reject inconsistent states."""

    with pytest.raises(ValueError, match=match):
        RustProgramADInterpreterResult(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "supported": "yes",
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported flag must be boolean",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([[0.0]], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "gradient must be one-dimensional",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "target count must match gradient",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": -1,
                "supported_effect_count": 0,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "counts must be non-negative",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 1,
                "supported_effect_count": 2,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "supported count exceeds effect count",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "targets must be non-empty",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("",),
                "claim_boundary": "bounded",
            },
            "blocked reasons must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": None,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("%0",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported state is inconsistent",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "",
            },
            "claim boundary must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": np.array([1.0], dtype=np.float64),
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("%0",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "value must be a real numeric scalar",
        ),
        (
            {
                "supported": True,
                "value": True,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("%0",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "value must be a real numeric scalar",
        ),
    ],
)
def test_value_and_gradient_result_rejects_invalid_states(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Value+gradient result container should reject inconsistent states."""

    with pytest.raises(ValueError, match=match):
        RustProgramADValueAndGradientResult(**kwargs)  # type: ignore[arg-type]
