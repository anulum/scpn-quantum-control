# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD reverse adjoint replay tests
"""Tests for whole-program AD reverse adjoint replay and public APIs."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control import program_ad_adjoint as adjoint_module
from scpn_quantum_control.differentiable import (
    Parameter,
    ProgramADAdjointResult,
    ProgramADAdjointStep,
    ProgramADAliasEdge,
    ProgramADAliasEffectAnalysis,
    ProgramADAliasSet,
    ProgramADControlRegion,
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADPhiNode,
    ProgramADSSAValue,
    ProgramADStaticAliasLatticeComponent,
    ProgramADStaticAliasLatticeReport,
    TraceADArray,
    WholeProgramADResult,
    WholeProgramBytecodeBasicBlock,
    WholeProgramBytecodeInstruction,
    WholeProgramCompilerFrontendReport,
    WholeProgramIRNode,
    WholeProgramSemanticsReport,
    WholeProgramSourceBytecodeLineMap,
    WholeProgramSourceIRFeature,
    WholeProgramSourceRegion,
    WholeProgramSymbolScopeEntry,
    WholeProgramTraceEvent,
    WholeProgramUnsupportedSemanticDiagnostic,
    analyze_program_ad_alias_effects,
    compile_whole_program_frontend,
    parse_program_ad_effect_ir,
    program_ad_static_alias_lattice_report,
    program_adjoint_grad,
    program_adjoint_gradient,
    program_adjoint_result,
    program_adjoint_value_and_grad,
    whole_program_grad,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _dict_payload(value: object) -> dict[str, Any]:
    """Return a JSON-like dictionary payload for dynamic result assertions."""

    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _list_payload(value: object) -> list[Any]:
    """Return a JSON-like list payload for dynamic result assertions."""

    assert isinstance(value, list)
    return value


def _minimal_whole_program_result(
    adjoint_result: ProgramADAdjointResult | None,
) -> WholeProgramADResult:
    """Build a minimal whole-program result for adjoint accessor tests."""

    return WholeProgramADResult(
        value=1.0,
        gradient=np.array([1.0], dtype=np.float64),
        method="whole_program_ad",
        step=0.0,
        evaluations=1,
        parameter_names=("x",),
        trainable=(True,),
        trace_events=(),
        source=None,
        control_flow_observed=False,
        numpy_observed=False,
        polyglot_targets={"python": "local"},
        claim_boundary="test-only whole-program result",
        adjoint_result=adjoint_result,
    )


def test_program_adjoint_input_token_helpers_resolve_literals_and_ir_values() -> None:
    """Program adjoint token helpers should resolve SSA and literal input tokens."""

    node = WholeProgramIRNode(
        index=0,
        op="parameter",
        inputs=("theta",),
        value=2.5,
        tangent=np.array([1.0], dtype=np.float64),
    )
    node_by_name = {"%0": node}

    assert adjoint_module._program_adjoint_is_ir_value("%0") is True
    assert adjoint_module._program_adjoint_is_ir_value("%x") is False
    assert adjoint_module._program_adjoint_is_ir_value("0.5") is False
    assert vars(differentiable)["_program_adjoint_is_ir_value"] is (
        adjoint_module._program_adjoint_is_ir_value
    )
    assert vars(differentiable)["_program_adjoint_input_value"] is (
        adjoint_module._program_adjoint_input_value
    )
    assert adjoint_module._program_adjoint_input_value("%0", node_by_name) == pytest.approx(2.5)
    assert adjoint_module._program_adjoint_input_value("3.25", node_by_name) == pytest.approx(3.25)
    assert adjoint_module._program_adjoint_input_value(
        "np.float64(-1.5)", node_by_name
    ) == pytest.approx(-1.5)


def test_program_adjoint_accessors_fail_closed_for_invalid_or_unsupported_results() -> None:
    """Extracted adjoint accessors should reject invalid or unsupported inputs."""

    unsupported_adjoint = ProgramADAdjointResult(
        gradient=np.array([0.0], dtype=np.float64),
        supported=False,
        unsupported_ops=("unsupported_op",),
        method="program_adjoint_ir_generation",
        claim_boundary="unsupported scalar replay",
    )

    with pytest.raises(ValueError, match="WholeProgramADResult"):
        adjoint_module.program_adjoint_result(object())
    with pytest.raises(ValueError, match="does not contain adjoint"):
        adjoint_module.program_adjoint_result(_minimal_whole_program_result(None))
    with pytest.raises(ValueError, match="unsupported for ops: unsupported_op"):
        adjoint_module.program_adjoint_gradient(_minimal_whole_program_result(unsupported_adjoint))


def test_program_adjoint_input_token_helpers_fail_closed_for_missing_or_bad_tokens() -> None:
    """Program adjoint token helpers should reject missing SSA and malformed literals."""

    with pytest.raises(ValueError, match="missing from IR"):
        adjoint_module._program_adjoint_input_value("%2", {})
    with pytest.raises(ValueError, match="not numeric"):
        adjoint_module._program_adjoint_input_value("not-a-number", {})


def test_whole_program_grad_respects_trainable_mask() -> None:
    """Whole-program gradients should preserve frozen parameters."""

    gradient = whole_program_grad(
        lambda values: values[0] ** 2 + values[1] ** 2,
        np.array([2.0, 3.0], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("frozen", trainable=False)),
        trace=False,
    )

    _assert_allclose(gradient, [4.0, 0.0], rtol=1.0e-6, atol=1.0e-6)


def test_whole_program_ad_is_exported_from_package_root() -> None:
    """Whole-program AD should be stable as a package-root API."""

    import scpn_quantum_control as scpn

    assert ProgramADAdjointResult is adjoint_module.ProgramADAdjointResult
    assert ProgramADAdjointStep is adjoint_module.ProgramADAdjointStep
    assert scpn.TraceADArray is TraceADArray
    assert scpn.ProgramADAdjointResult is ProgramADAdjointResult
    assert scpn.ProgramADAdjointStep is ProgramADAdjointStep
    assert scpn.ProgramADAliasEdge is ProgramADAliasEdge
    assert scpn.ProgramADAliasEffectAnalysis is ProgramADAliasEffectAnalysis
    assert scpn.ProgramADAliasSet is ProgramADAliasSet
    assert scpn.ProgramADControlRegion is ProgramADControlRegion
    assert scpn.ProgramADEffect is ProgramADEffect
    assert scpn.ProgramADEffectIR is ProgramADEffectIR
    assert scpn.ProgramADPhiNode is ProgramADPhiNode
    assert scpn.ProgramADSSAValue is ProgramADSSAValue
    assert scpn.ProgramADStaticAliasLatticeComponent is ProgramADStaticAliasLatticeComponent
    assert scpn.ProgramADStaticAliasLatticeReport is ProgramADStaticAliasLatticeReport
    assert scpn.WholeProgramADResult is WholeProgramADResult
    assert scpn.WholeProgramBytecodeBasicBlock is WholeProgramBytecodeBasicBlock
    assert scpn.WholeProgramBytecodeInstruction is WholeProgramBytecodeInstruction
    assert scpn.WholeProgramCompilerFrontendReport is WholeProgramCompilerFrontendReport
    assert scpn.WholeProgramTraceEvent is WholeProgramTraceEvent
    assert scpn.WholeProgramSourceIRFeature is WholeProgramSourceIRFeature
    assert scpn.WholeProgramSourceBytecodeLineMap is WholeProgramSourceBytecodeLineMap
    assert scpn.WholeProgramSourceRegion is WholeProgramSourceRegion
    assert scpn.WholeProgramSymbolScopeEntry is WholeProgramSymbolScopeEntry
    assert scpn.WholeProgramSemanticsReport is WholeProgramSemanticsReport
    assert (
        scpn.WholeProgramUnsupportedSemanticDiagnostic is WholeProgramUnsupportedSemanticDiagnostic
    )
    assert scpn.program_adjoint_grad is program_adjoint_grad
    assert differentiable.program_adjoint_gradient is adjoint_module.program_adjoint_gradient
    assert differentiable.program_adjoint_result is adjoint_module.program_adjoint_result
    assert scpn.program_adjoint_gradient is adjoint_module.program_adjoint_gradient
    assert scpn.program_adjoint_result is adjoint_module.program_adjoint_result
    assert scpn.program_adjoint_value_and_grad is program_adjoint_value_and_grad
    assert scpn.analyze_program_ad_alias_effects is analyze_program_ad_alias_effects
    assert scpn.program_ad_static_alias_lattice_report is program_ad_static_alias_lattice_report
    assert scpn.parse_program_ad_effect_ir is parse_program_ad_effect_ir
    assert scpn.whole_program_grad is whole_program_grad
    assert scpn.whole_program_value_and_grad is whole_program_value_and_grad
    assert scpn.compile_whole_program_frontend is compile_whole_program_frontend


def test_program_adjoint_value_and_grad_api_returns_reverse_replay_gradient() -> None:
    """First-class program adjoint API should return reverse replay gradients."""

    def objective(values: Any) -> object:
        x, y, z = values
        return np.sin(x * y) + np.log(z + 4.0) + np.sqrt(x + 3.0) + y**2

    values = np.array([0.75, -1.25, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
        trace=False,
    )
    value, reverse_gradient = program_adjoint_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
        trace=False,
    )

    assert value == pytest.approx(result.value)
    assert result.adjoint_result is not None
    _assert_allclose(reverse_gradient, result.adjoint_result.gradient, atol=1.0e-12)
    _assert_allclose(program_adjoint_grad(objective, values, trace=False), result.gradient)


def test_program_adjoint_grad_api_respects_trainable_mask() -> None:
    """Program adjoint gradient API should preserve frozen-parameter semantics."""

    gradient = program_adjoint_grad(
        lambda values: values[0] ** 2 + values[1] ** 2 + values[0] * values[1],
        np.array([2.0, 3.0], dtype=np.float64),
        parameters=(Parameter("trainable"), Parameter("frozen", trainable=False)),
        trace=False,
    )

    _assert_allclose(gradient, [7.0, 0.0], atol=1.0e-12)


def test_program_adjoint_replay_matches_forward_program_ad_for_supported_ir() -> None:
    """Reverse-mode program adjoint replay should match forward Program AD on supported IR."""

    def objective(values: Any) -> object:
        x, y, z = values
        branch = x if x > y else y
        return (
            np.sin(x)
            + np.cos(y)
            + np.exp(x - y)
            + np.log(z + 3.0)
            + np.sqrt(z + 4.0)
            + np.tanh(x * z)
            + (x**2.0)
            + (2.0**y)
            + branch
        )

    values = np.array([1.25, -0.4, 0.75], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )
    adjoint = program_adjoint_result(result)

    assert adjoint.supported is True
    assert adjoint.unsupported_ops == ()
    assert adjoint.method == "program_adjoint_ir_generation"
    assert adjoint.adjoint_step_count == len(result.ir_nodes)
    assert adjoint.adjoint_steps
    assert all(isinstance(step, ProgramADAdjointStep) for step in adjoint.adjoint_steps)
    assert adjoint.adjoint_steps[0].primal_value == f"%{len(result.ir_nodes) - 1}"
    assert adjoint.adjoint_steps[0].supported is True
    assert adjoint.adjoint_steps[-1].operation == "parameter"
    branch_steps = tuple(
        step for step in adjoint.adjoint_steps if step.operation.startswith("branch:")
    )
    assert len(branch_steps) == 1
    branch_step = branch_steps[0]
    assert branch_step.control_region is not None
    assert branch_step.control_region_kind == "runtime_branch"
    assert branch_step.control_region_entered is True
    assert branch_step.phi_node is not None
    assert branch_step.phi_selected == "executed_true"
    effect_ordering = tuple(
        step.effect_ordering for step in adjoint.adjoint_steps if step.effect_ordering is not None
    )
    assert effect_ordering == tuple(sorted(effect_ordering, reverse=True))
    payload = _dict_payload(adjoint.to_dict())
    payload_steps = _list_payload(payload["adjoint_steps"])
    first_step = _dict_payload(payload_steps[0])
    assert payload["adjoint_step_count"] == len(result.ir_nodes)
    assert first_step["primal_effect"] is not None
    assert first_step["effect_kind"] == "pure"
    assert first_step["effect_version"] >= 0
    assert first_step["effect_ordering"] == len(result.ir_nodes) - 1
    assert len(_list_payload(first_step["contribution_inputs"])) == len(
        _list_payload(first_step["contribution_scales"])
    )
    assert len(_list_payload(first_step["contribution_inputs"])) == len(
        _list_payload(first_step["contribution_cotangents"])
    )
    payload_branch = next(
        _dict_payload(step)
        for step in payload_steps
        if str(_dict_payload(step)["operation"]).startswith("branch:")
    )
    assert payload_branch["control_region"] == branch_step.control_region
    assert payload_branch["control_region_kind"] == "runtime_branch"
    assert payload_branch["control_region_entered"] is True
    assert payload_branch["phi_node"] == branch_step.phi_node
    assert payload_branch["phi_selected"] == "executed_true"
    _assert_allclose(adjoint.gradient, result.gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), result.gradient, rtol=1.0e-12, atol=1.0e-12)


def test_program_adjoint_generation_steps_record_local_pullback_flow() -> None:
    """Generated adjoint steps should expose finite local pullback cotangent flow."""

    def objective(values: Any) -> object:
        x, y = values
        return (x * y) + (x + x)

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, 3.0], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("y")),
    )
    adjoint = program_adjoint_result(result)

    mul_step = next(step for step in adjoint.adjoint_steps if step.operation == "mul")
    assert mul_step.contribution_inputs == ("%0", "%1")
    assert mul_step.incoming_cotangent == pytest.approx(1.0)
    _assert_allclose(mul_step.contribution_scales, [3.0, 2.0], atol=1.0e-12)
    _assert_allclose(mul_step.contribution_cotangents, [3.0, 2.0], atol=1.0e-12)

    repeated_add_step = next(
        step
        for step in adjoint.adjoint_steps
        if step.operation == "add" and step.input_values == ("%0", "%0")
    )
    assert repeated_add_step.contribution_inputs == ("%0",)
    assert repeated_add_step.incoming_cotangent == pytest.approx(1.0)
    _assert_allclose(repeated_add_step.contribution_scales, [2.0], atol=1.0e-12)
    _assert_allclose(repeated_add_step.contribution_cotangents, [2.0], atol=1.0e-12)

    parameter_cotangents = {
        step.input_values[0]: step.incoming_cotangent
        for step in adjoint.adjoint_steps
        if step.operation == "parameter"
    }
    assert parameter_cotangents == pytest.approx({"x": 5.0, "y": 2.0})

    payload_steps = {
        tuple(_list_payload(step["input_values"])): step
        for step in _list_payload(_dict_payload(adjoint.to_dict())["adjoint_steps"])
        if _dict_payload(step)["operation"] == "add"
    }
    assert _dict_payload(payload_steps[("%0", "%0")])["contribution_scales"] == [2.0]
    assert _dict_payload(payload_steps[("%0", "%0")])["contribution_cotangents"] == [2.0]


def test_program_adjoint_replay_supports_static_setitem_effects() -> None:
    """Reverse-mode program adjoints should replay static setitem dataflow exactly."""

    def objective(values: Any) -> object:
        work = values.copy()
        work[0] = values[1] * values[1]
        return work[0] + values[0]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 1.5], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("y")),
    )

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    assert result.adjoint_result.unsupported_ops == ()
    assert result.program_ir is not None
    assert result.adjoint_result.replay_node_count == len(result.ir_nodes)
    assert result.adjoint_result.replay_effect_count == len(result.program_ir.effects)
    assert result.adjoint_result.replay_control_region_count == len(
        result.program_ir.control_regions
    )
    assert result.adjoint_result.replay_phi_node_count == len(result.program_ir.phi_nodes)
    assert result.adjoint_result.replay_ir_format == "program_ad_effect_ir.v1"
    assert result.adjoint_result.adjoint_step_count == len(result.ir_nodes)
    assert all(step.supported for step in result.adjoint_result.adjoint_steps)
    _assert_allclose(result.gradient, [1.0, 3.0], atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_adjoint_result_validation_paths() -> None:
    """Program adjoint result metadata should reject malformed reverse-mode outputs."""

    def valid_step(**overrides: object) -> ProgramADAdjointStep:
        payload: dict[str, object] = {
            "index": 0,
            "primal_value": "%0",
            "primal_effect": 0,
            "effect_kind": "pure",
            "effect_version": 0,
            "effect_ordering": 0,
            "operation": "parameter",
            "input_values": ("x",),
            "contribution_inputs": (),
            "supported": True,
        }
        payload.update(overrides)
        step = cast(Any, ProgramADAdjointStep)(**payload)
        assert isinstance(step, ProgramADAdjointStep)
        return step

    result = ProgramADAdjointResult(
        gradient=np.array([1.0], dtype=np.float64),
        supported=True,
        unsupported_ops=(),
        method="program_adjoint_ir_generation",
        claim_boundary="supported scalar replay",
        adjoint_steps=(valid_step(),),
    )

    result_payload = _dict_payload(result.to_dict())
    result_steps = _list_payload(result_payload["adjoint_steps"])
    first_step = _dict_payload(result_steps[0])
    no_effect_step = ProgramADAdjointStep(
        index=0,
        primal_value="%0",
        primal_effect=None,
        operation="branch:%0:True",
        input_values=(),
        contribution_inputs=(),
        supported=True,
    )
    assert result.supported is True
    assert no_effect_step.primal_effect is None
    assert no_effect_step.effect_kind is None
    assert result.replay_ir_format == "program_ad_effect_ir.v1"
    assert result.adjoint_step_count == 1
    assert first_step["operation"] == "parameter"
    assert first_step["effect_kind"] == "pure"
    assert first_step["effect_version"] == 0
    assert first_step["effect_ordering"] == 0
    assert first_step["control_region"] is None
    assert first_step["control_region_kind"] is None
    assert first_step["control_region_entered"] is None
    assert first_step["phi_node"] is None
    assert first_step["phi_selected"] is None
    assert first_step["incoming_cotangent"] == 0.0
    assert first_step["contribution_scales"] == []
    assert first_step["contribution_cotangents"] == []

    with pytest.raises(ValueError, match="one-dimensional"):
        ProgramADAdjointResult(
            gradient=np.array([[1.0]], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="finite values"):
        ProgramADAdjointResult(
            gradient=np.array([float("nan")], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="supported must be a boolean"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=cast(Any, "yes"),
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="unsupported_ops"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=False,
            unsupported_ops=cast(tuple[str, ...], ("",)),
            method="program_adjoint_replay",
            claim_boundary="unsupported scalar replay",
        )
    with pytest.raises(ValueError, match="cannot be supported"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=("mutation:setitem",),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="method"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="",
        )
    with pytest.raises(ValueError, match="replay_node_count"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
            replay_node_count=-1,
        )
    with pytest.raises(ValueError, match="replay_ir_format"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
            replay_ir_format="",
        )
    with pytest.raises(ValueError, match="adjoint_steps"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_ir_generation",
            claim_boundary="supported scalar replay",
            adjoint_steps=cast(tuple[ProgramADAdjointStep, ...], (object(),)),
        )
    with pytest.raises(ValueError, match="densely indexed"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_ir_generation",
            claim_boundary="supported scalar replay",
            adjoint_steps=(valid_step(index=1),),
        )
    with pytest.raises(ValueError, match="unsupported steps"):
        ProgramADAdjointResult(
            gradient=np.array([0.0], dtype=np.float64),
            supported=False,
            unsupported_ops=("unsupported_op",),
            method="program_adjoint_ir_generation",
            claim_boundary="unsupported scalar replay",
            adjoint_steps=(
                valid_step(
                    operation="different_op",
                    input_values=(),
                    supported=False,
                    unsupported_reason="no rule",
                ),
            ),
        )
    with pytest.raises(ValueError, match="supported program AD adjoint cannot"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_ir_generation",
            claim_boundary="supported scalar replay",
            adjoint_steps=(
                valid_step(
                    operation="unsupported_op",
                    input_values=(),
                    supported=False,
                    unsupported_reason="no rule",
                ),
            ),
        )
    with pytest.raises(ValueError, match="index"):
        valid_step(index=-1)
    with pytest.raises(ValueError, match="primal_value"):
        valid_step(primal_value="")
    with pytest.raises(ValueError, match="primal_effect"):
        valid_step(primal_effect=-1)
    with pytest.raises(ValueError, match="supported program AD adjoint step"):
        valid_step(unsupported_reason="not allowed")
    with pytest.raises(ValueError, match="supported must be a boolean"):
        valid_step(supported=cast(Any, "yes"))
    with pytest.raises(ValueError, match="unsupported_reason"):
        valid_step(
            operation="unsupported_op",
            input_values=(),
            supported=False,
            unsupported_reason="",
        )
    with pytest.raises(ValueError, match="effect metadata requires"):
        valid_step(primal_effect=None)
    with pytest.raises(ValueError, match="effect_kind"):
        valid_step(effect_kind=None)
    with pytest.raises(ValueError, match="effect_version"):
        valid_step(effect_version=-1)
    with pytest.raises(ValueError, match="effect_ordering"):
        valid_step(effect_ordering=cast(Any, True))
    with pytest.raises(ValueError, match="control metadata"):
        valid_step(control_region_kind="runtime_branch")
    with pytest.raises(ValueError, match="control_region must be non-negative"):
        valid_step(
            control_region=-1,
            control_region_kind="runtime_branch",
            control_region_entered=True,
        )
    with pytest.raises(ValueError, match="control_region_kind"):
        valid_step(
            control_region=0,
            control_region_kind="",
            control_region_entered=True,
        )
    with pytest.raises(ValueError, match="control_region_entered"):
        valid_step(
            control_region=0,
            control_region_kind="runtime_branch",
            control_region_entered=None,
        )
    with pytest.raises(ValueError, match="phi metadata requires a phi_node"):
        valid_step(phi_selected="executed_true")
    with pytest.raises(ValueError, match="phi metadata requires control metadata"):
        valid_step(phi_node=0, phi_selected="executed_true")
    with pytest.raises(ValueError, match="phi_node"):
        valid_step(
            control_region=0,
            control_region_kind="runtime_branch",
            control_region_entered=True,
            phi_node=-1,
            phi_selected="executed_true",
        )
    with pytest.raises(ValueError, match="phi_selected"):
        valid_step(
            control_region=0,
            control_region_kind="runtime_branch",
            control_region_entered=True,
            phi_node=0,
            phi_selected="",
        )
    with pytest.raises(ValueError, match="operation"):
        valid_step(operation="")
    with pytest.raises(ValueError, match="input_values"):
        valid_step(input_values=("",))
    with pytest.raises(ValueError, match="contribution_inputs entries"):
        valid_step(contribution_inputs=("",))
    with pytest.raises(ValueError, match="sorted and unique"):
        valid_step(contribution_inputs=("%1", "%0"), contribution_scales=(1.0, 1.0))
    with pytest.raises(ValueError, match="contribution_scales length"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            contribution_scales=(),
        )
    with pytest.raises(ValueError, match="incoming_cotangent must be finite"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            incoming_cotangent=float("inf"),
            contribution_scales=(1.0,),
            contribution_cotangents=(1.0,),
        )
    with pytest.raises(ValueError, match="incoming_cotangent must be a finite float"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            incoming_cotangent=cast(Any, True),
            contribution_scales=(1.0,),
            contribution_cotangents=(1.0,),
        )
    with pytest.raises(ValueError, match="contribution_scales entries must be finite"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            contribution_scales=(float("nan"),),
        )
    with pytest.raises(ValueError, match="contribution_scales entries must be finite floats"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            contribution_scales=(cast(Any, "bad"),),
        )
    with pytest.raises(ValueError, match="contribution_cotangents length"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            contribution_scales=(1.0,),
            contribution_cotangents=(),
        )
    with pytest.raises(ValueError, match="contribution_cotangents entries must be finite"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            contribution_scales=(1.0,),
            contribution_cotangents=(float("nan"),),
        )
    with pytest.raises(ValueError, match="contribution_cotangents entries must be finite floats"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            contribution_scales=(1.0,),
            contribution_cotangents=(cast(Any, "bad"),),
        )
    with pytest.raises(ValueError, match="contribution_cotangents must match"):
        valid_step(
            operation="add",
            input_values=("%0", "%1"),
            contribution_inputs=("%0",),
            incoming_cotangent=2.0,
            contribution_scales=(3.0,),
            contribution_cotangents=(5.0,),
        )
