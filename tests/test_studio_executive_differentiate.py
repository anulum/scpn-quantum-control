# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive differentiate handler tests
"""Tests for the differentiate executive action handler."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scpn_quantum_engine", reason="Rust engine (pyo3) not installed")
pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio.executive import (  # noqa: E402
    ExecutiveRequest,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from scpn_quantum_control.studio.executive_differentiate import (  # noqa: E402
    DIFFERENTIATE_VERB,
    DifferentiateActionHandler,
    _as_float,
    _normalise_program,
    _safe_slug,
    build_effect_ir,
    default_registry,
)


def _program(operations: list[dict[str, Any]], output: str, inputs: Any = None) -> dict[str, Any]:
    return {
        "inputs": inputs if inputs is not None else [["x", 3.0], ["y", 5.0]],
        "operations": operations,
        "output": output,
    }


_BILINEAR = _program([{"op": "mul", "inputs": ["x", "y"], "into": "f"}], "f")


def _request(program: dict[str, Any], *, backend: str | None = None) -> ExecutiveRequest:
    return ExecutiveRequest(
        verb=DIFFERENTIATE_VERB, action_id="unit-diff", parameters=program, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end through the spine
# --------------------------------------------------------------------------- #
def test_bilinear_gradient_is_the_swapped_inputs() -> None:
    record = run_action(_request(_BILINEAR), registry=default_registry())
    assert record.result.status == "succeeded"
    assert record.result.outputs["value"] == pytest.approx(15.0)
    assert record.result.outputs["gradient"] == pytest.approx([5.0, 3.0])
    assert record.result.outputs["verified"] is True


def test_quadratic_plus_linear_gradient() -> None:
    program = _program(
        [
            {"op": "mul", "inputs": ["x", "x"], "into": "x2"},
            {"op": "mul", "inputs": ["y", "2.0"], "into": "y2"},
            {"op": "add", "inputs": ["x2", "y2"], "into": "f"},
        ],
        "f",
    )
    record = run_action(_request(program), registry=default_registry())
    assert record.result.outputs["value"] == pytest.approx(19.0)
    assert record.result.outputs["gradient"] == pytest.approx([6.0, 2.0])
    assert record.result.outputs["verified"] is True


def test_generated_script_embeds_sealed_values_and_compiles() -> None:
    record = run_action(_request(_BILINEAR), registry=default_registry())
    assert record.script is not None
    compile(record.script.source, record.script.filename, "exec")
    assert "EXPECTED_VALUE = 15.0" in record.script.source
    assert "EXPECTED_GRADIENT = [5.0, 3.0]" in record.script.source
    assert record.script.entrypoint.endswith(".py")


def test_default_registry_registers_differentiate() -> None:
    assert default_registry().verbs() == ("differentiate",)


def test_preview_plan_uses_default_backend_and_read_only_contract() -> None:
    plan = preview_action(_request(_BILINEAR), registry=default_registry())
    assert plan.backend == "python"
    assert plan.requires_approval is False
    assert len(plan.steps) == 4


def test_explicit_rust_backend_is_accepted() -> None:
    plan = preview_action(_request(_BILINEAR, backend="rust"), registry=default_registry())
    assert plan.backend == "rust"


def test_unknown_backend_is_rejected() -> None:
    handler = DifferentiateActionHandler()
    contract = resolve_verb_contract(DIFFERENTIATE_VERB)
    with pytest.raises(ValueError, match="is not declared for the differentiate verb"):
        handler.plan(_request(_BILINEAR, backend="cuda"), contract)


# --------------------------------------------------------------------------- #
# build_effect_ir
# --------------------------------------------------------------------------- #
def test_build_effect_ir_targets_and_values() -> None:
    program = _normalise_program(_BILINEAR)
    ir, targets, values = build_effect_ir(program)
    assert targets == ("%0", "%1")
    assert values == [3.0, 5.0]
    assert '"format":"program_ad_effect_ir.v1"' in ir


# --------------------------------------------------------------------------- #
# _as_float
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "3", None, float("inf"), float("nan")])
def test_as_float_rejects_non_finite_or_non_numbers(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_float("v", bad)


def test_as_float_accepts_int_and_float() -> None:
    assert _as_float("v", 3) == 3.0
    assert _as_float("v", 2.5) == 2.5


# --------------------------------------------------------------------------- #
# _safe_slug
# --------------------------------------------------------------------------- #
def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("demo-Action.1") == "demo_Action_1"
    assert _safe_slug("***") == "action"


# --------------------------------------------------------------------------- #
# _normalise_program error branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "parameters",
    [
        {"inputs": "x", "operations": [], "output": "f"},
        {"inputs": [["x", 1.0]], "operations": "nope", "output": "f"},
        {"inputs": [["x", 1.0]], "operations": [{"op": "mul", "inputs": ["x", "x"], "into": "f"}]},
        {
            "inputs": [],
            "operations": [{"op": "mul", "inputs": ["x", "x"], "into": "f"}],
            "output": "f",
        },
        {"inputs": [["x", 1.0]], "operations": [], "output": "f"},
        {
            "inputs": [["x", 1.0, 9.0]],
            "operations": [{"op": "add", "inputs": ["x", "x"], "into": "f"}],
            "output": "f",
        },
        {
            "inputs": [[1, 1.0]],
            "operations": [{"op": "add", "inputs": ["x", "x"], "into": "f"}],
            "output": "f",
        },
        {
            "inputs": [["x", 1.0], ["x", 2.0]],
            "operations": [{"op": "add", "inputs": ["x", "x"], "into": "f"}],
            "output": "f",
        },
        {
            "inputs": [["x", 1.0]],
            "operations": [{"op": "mul", "inputs": ["x", "x"], "into": "g"}],
            "output": "f",
        },
    ],
)
def test_normalise_program_rejects_invalid(parameters: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _normalise_program(parameters)


# --------------------------------------------------------------------------- #
# _normalise_operation / _resolve_reference error branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "operation",
    [
        "not-a-mapping",
        {"op": "sub", "inputs": ["x", "x"], "into": "f"},
        {"op": "mul", "inputs": ["x", "x"], "into": ""},
        {"op": "mul", "inputs": ["x", "x"], "into": "x"},
        {"op": "mul", "inputs": ["x"], "into": "f"},
        {"op": "mul", "inputs": ["x", ""], "into": "f"},
        {"op": "mul", "inputs": ["x", "unknown"], "into": "f"},
    ],
)
def test_normalise_operation_rejects_invalid(operation: Any) -> None:
    parameters = {"inputs": [["x", 1.0]], "operations": [operation], "output": "f"}
    with pytest.raises(ValueError):
        _normalise_program(parameters)


def test_numeric_literal_reference_is_accepted() -> None:
    program = _program(
        [{"op": "mul", "inputs": ["x", "2.0"], "into": "f"}], "f", inputs=[["x", 4.0]]
    )
    record = run_action(_request(program), registry=default_registry())
    assert record.result.outputs["value"] == pytest.approx(8.0)
    assert record.result.outputs["gradient"] == pytest.approx([2.0])


def test_execute_fails_closed_on_unsupported_engine_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scpn_quantum_engine as engine

    monkeypatch.setattr(
        engine,
        "program_ad_effect_ir_interpret_value_and_gradient",
        lambda ir, inputs: '{"supported": false}',
    )
    record = run_action(_request(_BILINEAR), registry=default_registry())
    assert record.result.status == "failed"
    assert "not a supported bounded rational replay" in (record.result.error or "")
