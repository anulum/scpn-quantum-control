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
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
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
                "claim_boundary": "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit",
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
                "claim_boundary": "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_executed_branch_view_alias_only_no_llvm_jit",
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


def test_rust_program_ad_value_and_gradient_replay_matches_python_trace() -> None:
    """Rust Program AD scalar replay should match the emitted Python trace."""

    pytest.importorskip("scpn_quantum_engine")
    values = np.array([0.4, -0.2], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y = trace_values
        return x * x + 2.0 * y + np.sin(x)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y")),
    )
    assert result.program_ir is not None

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert isinstance(rust_result, RustProgramADValueAndGradientResult)
    assert rust_result.supported is True
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert len(rust_result.parameter_targets) == values.size
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_executed_branch_trace() -> None:
    """Rust Program AD replay should preserve executed scalar branch semantics."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([0.4, -0.2], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y = trace_values
        return (x * x if x > y else y * y) + 2.0 * y + np.sin(x)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y")),
    )
    assert result.program_ir is not None
    assert result.program_ir.control_regions
    assert result.program_ir.phi_nodes
    assert not result.program_ir.alias_edges
    assert "runtime_branch" in {region.kind for region in result.program_ir.control_regions}, (
        result.program_ir.control_regions
    )
    assert '"kind":"runtime_branch"' in result.program_ir.serialization
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_scalar_primitive_family_trace() -> None:
    """Rust Program AD replay should preserve emitted scalar primitive-family traces."""

    pytest.importorskip("scpn_quantum_engine")
    values = np.array([0.4, -0.2, 0.25, 0.1], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y, z, w = trace_values
        return (
            np.sqrt(x + 2.0)
            + np.tanh(y)
            + np.log1p(z)
            + np.expm1(w)
            + np.reciprocal(x + 3.0)
            + np.arcsin(0.2 * y)
            + np.arccos(0.1 * z)
            + abs(w + 1.0)
        )

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(
            Parameter("x"),
            Parameter("y"),
            Parameter("z"),
            Parameter("w"),
        ),
    )
    assert result.program_ir is not None
    assert not result.program_ir.alias_edges
    assert not result.program_ir.control_regions
    operations = {effect.operation for effect in result.program_ir.effects}
    assert {
        "sqrt",
        "tanh",
        "log1p",
        "expm1",
        "reciprocal",
        "arcsin",
        "arccos",
        "abs",
    } <= operations

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
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


def _objective_reshape_sumsq(values: Any) -> Any:
    return np.sum(np.reshape(values, (2, 2)) ** 2)


def _objective_mutation(values: Any) -> Any:
    work = values * 1.0
    work[0] = work[0] * work[1]
    return np.sum(work)


def test_bridge_replays_inert_view_alias_program_with_real_engine() -> None:
    """With the real engine, a reshape view-alias program replays bit-exact."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_reshape_sumsq, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine does not yet replay view aliases: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_reshape_sumsq, sample)
    np.testing.assert_array_equal(np.asarray(rust.gradient), reference)
    assert rust.claim_boundary.endswith("view_alias_only_no_llvm_jit")


def test_bridge_fails_closed_on_mutation_alias_with_real_engine() -> None:
    """A mutation-aliasing program stays outside the bounded Rust replay."""

    pytest.importorskip("scpn_quantum_engine")

    sample = np.array([2.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_mutation, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is False
    assert any("non-view alias" in reason for reason in rust.blocked_reasons)


def _objective_trace_2x2(values: Any) -> Any:
    return np.trace(np.reshape(values, (2, 2)))


def _objective_det_2x2(values: Any) -> Any:
    return np.linalg.det(np.reshape(values, (2, 2)))


def test_bridge_replays_linalg_trace_with_real_engine() -> None:
    """With the real engine, a 2x2 trace program replays bit-exact."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_trace_2x2, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:trace replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_trace_2x2, sample)
    np.testing.assert_array_equal(np.asarray(rust.gradient), reference)


def test_bridge_replays_linalg_det_2x2_with_real_engine() -> None:
    """With the real engine, a 2x2 determinant program replays within float64 tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_det_2x2, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:det replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_2x2, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)
    assert rust.claim_boundary.endswith(
        "scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    )


def _objective_inv_2x2_sum(values: Any) -> Any:
    return np.sum(np.linalg.inv(np.reshape(values, (2, 2))))


def _objective_solve_2x2_sum(values: Any) -> Any:
    return np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:]))


def _objective_solve_2x2_indexed(values: Any) -> Any:
    return np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:])[0]


def test_bridge_replays_linalg_inverse_with_real_engine() -> None:
    """With the real engine, a reduced 2x2 inverse program replays within tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_inv_2x2_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:inv replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_2x2_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_replays_linalg_solve_with_real_engine() -> None:
    """With the real engine, a reduced 2x2 linear solve replays within tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0, 1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_solve_2x2_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:solve replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_solve_2x2_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_fails_closed_on_indexed_multi_output_linalg_with_real_engine() -> None:
    """A bare indexed solve component stays outside the bounded Rust replay."""

    pytest.importorskip("scpn_quantum_engine")

    sample = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 6.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_solve_2x2_indexed, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is False
    assert any("indexed multi-output linalg" in reason for reason in rust.blocked_reasons)


def _objective_det_3x3(values: Any) -> Any:
    return np.linalg.det(np.reshape(values, (3, 3)))


def test_bridge_replays_linalg_det_3x3_with_real_engine() -> None:
    """With the real engine, a 3x3 determinant program replays within float64 tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_det_3x3, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:det:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_3x3, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)
