# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust bridge contracts tests
"""Validation and immutable-result contract tests for the Program AD Rust bridge."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from _program_ad_rust_bridge_test_fixtures import (
    _install_fake_engine,
)

from scpn_quantum_control.program_ad_rust_bridge import (
    RustProgramADInterpreterResult,
    RustProgramADValueAndGradientResult,
    interpret_program_ad_effect_ir_with_rust,
    mirror_program_ad_registry_metadata_with_rust,
    value_and_grad_program_ad_effect_ir_with_rust,
)


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

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    missing_mirror_extension = mirror_program_ad_registry_metadata_with_rust()
    assert missing_mirror_extension.supported is False
    assert missing_mirror_extension.primitive_count == 118
    assert missing_mirror_extension.blocked_reasons == (
        "scpn_quantum_engine native extension is not built",
    )

    fake_engine = ModuleType("scpn_quantum_engine")
    _install_fake_engine(monkeypatch, fake_engine)
    missing_mirror_export = mirror_program_ad_registry_metadata_with_rust()
    assert missing_mirror_export.supported is False
    assert missing_mirror_export.blocked_reasons == (
        "scpn_quantum_engine native extension lacks Program AD registry metadata mirror",
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

    def malformed_registry_mirror(_snapshot: str) -> str:
        return json.dumps(
            {
                "supported": True,
                "primitive_count": 118,
                "covered_primitives": 118,
                "family_counts": {"elementwise": 24},
                "facet_counts": [],
                "executable_operation_count": 1,
                "executable_operations": ["sin"],
                "blocked_reasons": [],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_registry_metadata_mirror = malformed_registry_mirror
    with pytest.raises(ValueError, match="facet_counts must be a JSON object"):
        mirror_program_ad_registry_metadata_with_rust()


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
