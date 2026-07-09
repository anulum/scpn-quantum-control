# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Whole-Program Emitter Tests
"""Emitter-contract tests for whole-program native LLVM IR generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pytest

from scpn_quantum_control.compiler import mlir_whole_program_emitter as emitter
from scpn_quantum_control.differentiable import WholeProgramADResult, WholeProgramIRNode

NodeOperation: TypeAlias = Callable[[list[str], WholeProgramADResult, WholeProgramIRNode], None]


def _node(
    index: int,
    op: str,
    inputs: Sequence[str],
    *,
    value: float = 1.0,
) -> WholeProgramIRNode:
    """Return a minimal IR node for emitter tests."""

    return WholeProgramIRNode(
        index=index,
        op=op,
        inputs=tuple(inputs),
        value=value,
        tangent=np.array([0.0, 0.0], dtype=np.float64),
    )


def _result(
    nodes: Sequence[WholeProgramIRNode],
    *,
    trainable: tuple[bool, ...] = (True, True),
) -> WholeProgramADResult:
    """Return a minimal validated whole-program AD result."""

    gradient = np.array(
        [float(index + 1) if flag else 0.0 for index, flag in enumerate(trainable)],
        dtype=np.float64,
    )
    return WholeProgramADResult(
        value=1.0,
        gradient=gradient,
        method="whole_program_ad",
        step=0.0,
        evaluations=1,
        parameter_names=tuple(f"x{index}" for index in range(len(trainable))),
        trainable=trainable,
        trace_events=(),
        source=None,
        control_flow_observed=False,
        numpy_observed=True,
        polyglot_targets={"python": "available"},
        claim_boundary="test emitter contract",
        ir_nodes=tuple(nodes),
    )


def _operation_lines(
    result: WholeProgramADResult,
    node: WholeProgramIRNode,
) -> list[str]:
    """Emit one operation through the production emitter dispatcher."""

    lines: list[str] = []
    emitter._emit_whole_program_native_operation(
        lines,
        result,
        node,
        emitter._WholeProgramNativeEmissionState(),
    )
    return lines


def _tokens(count: int) -> tuple[str, ...]:
    """Return deterministic scalar-token inputs for emitter contracts."""

    return tuple(str(1.0 + float(index) / 10.0) for index in range(count))


def test_native_computation_emits_parameters_constants_branches_and_unary_variants() -> None:
    """Whole-kernel computation emission should handle scalar leaf IR records."""

    nodes = (
        _node(0, "parameter", ("x0",), value=0.25),
        _node(1, "parameter", ("x1",), value=1.25),
        _node(2, "constant", ("1.5",), value=1.5),
        _node(3, "branch:test:True", (), value=2.0),
        _node(4, "exp", ("%0",), value=float(np.exp(0.25))),
        _node(5, "sqrt", ("%1",), value=float(np.sqrt(1.25))),
        _node(6, "negative", ("%2",), value=-1.5),
    )
    lines, final_value, final_derivatives = emitter._emit_whole_program_native_computation(
        _result(nodes),
        values_pointer="%values",
    )

    text = "\n".join(lines)
    assert final_value == "%n6"
    assert final_derivatives == ("%d6_0", "%d6_1")
    assert "call double @llvm.exp.f64" in text
    assert "call double @llvm.sqrt.f64" in text
    assert "%n6 = fsub double 0.00000000000000000e+00" in text


@pytest.mark.parametrize(
    ("nodes", "message"),
    (
        ((_node(0, "parameter", ("missing",)),), "parameter node is malformed"),
        ((_node(0, "branch:test:True", ("%0",)),), "branch node must be signature-only"),
        ((), "requires IR nodes"),
    ),
)
def test_native_computation_rejects_malformed_leaf_ir(
    nodes: tuple[WholeProgramIRNode, ...],
    message: str,
) -> None:
    """Computation emission should fail closed on malformed structural nodes."""

    with pytest.raises(ValueError, match=message):
        emitter._emit_whole_program_native_computation(
            _result(nodes),
            values_pointer="%values",
        )


@pytest.mark.parametrize(
    ("op", "inputs", "message"),
    (
        ("sin", (), "expects one input"),
        ("linalg:det:2x2", _tokens(3), "expects four matrix inputs"),
        ("linalg:det:3x3", _tokens(8), "expects nine matrix inputs"),
        ("linalg:det:4x4", _tokens(15), "expects sixteen matrix inputs"),
        ("linalg:det:5x5", _tokens(24), "expects twenty-five matrix inputs"),
        ("linalg:det:6x6", _tokens(35), "expects 36 matrix inputs"),
        ("linalg:trace:2x2:offset:0", _tokens(1), "expects 2 diagonal inputs"),
        ("linalg:diag:2:offset:0:construct:0", (), "expects 1 diagonal input"),
        ("linalg:matrix_power:2x2:power:2:0:0", _tokens(3), "expects four inputs"),
        (
            "linalg:multi_dot:2x2__2x2:out:2x2:0",
            _tokens(7),
            "expects eight inputs",
        ),
        ("linalg:inv:2x2:0:0", _tokens(3), "expects four matrix inputs"),
        ("linalg:inv:3x3:0:0", _tokens(8), "expects matrix inputs"),
        ("linalg:solve:2x2:rhs:2:0", _tokens(5), "expects matrix and rhs inputs"),
        ("linalg:solve:3x3:rhs:3:0", _tokens(11), "expects matrix and rhs inputs"),
        (
            "linalg:solve:3x3:rhs:3x2:0:0",
            _tokens(14),
            "expects matrix and rhs matrix inputs",
        ),
        ("where", ("%0", "%1"), "where expects predicate"),
        ("clip", ("%0", "%1"), "clip expects value"),
        ("unsupported:op", ("%0",), "does not support op"),
        ("add", ("%0",), "expects two inputs"),
        ("pow", ("%0", "%1"), "requires constant exponent"),
    ),
)
def test_operation_dispatcher_rejects_invalid_contracts(
    op: str,
    inputs: tuple[str, ...],
    message: str,
) -> None:
    """The operation dispatcher should reject malformed op-specific contracts."""

    with pytest.raises(ValueError, match=message):
        _operation_lines(_result((_node(0, "parameter", ("x0",)),)), _node(4, op, inputs))


@pytest.mark.parametrize(
    ("op", "inputs", "message"),
    (
        (
            "linalg:matrix_power:2x2:power:2:2:2",
            _tokens(4),
            "does not support op",
        ),
        (
            "linalg:multi_dot:2x2__2x2:out:2x2:4",
            _tokens(8),
            "does not support op",
        ),
        ("linalg:inv:2x2:2:2", _tokens(4), "does not support op"),
        ("linalg:solve:2x2:rhs:2:2", _tokens(6), "does not support op"),
    ),
)
def test_linalg_dispatcher_rejects_unknown_output_slots(
    op: str,
    inputs: tuple[str, ...],
    message: str,
) -> None:
    """Compact linalg op families should reject unknown output coordinates."""

    with pytest.raises(ValueError, match=message):
        _operation_lines(_result((_node(0, "parameter", ("x0",)),)), _node(8, op, inputs))


def test_dispatcher_exercises_faddeev_leverrier_wide_det_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wide-det dispatcher should call the non-helper Faddeev-LeVerrier route when selected."""

    calls: list[tuple[int, str]] = []

    def fake_faddeev(
        lines: list[str],
        result: WholeProgramADResult,
        node: WholeProgramIRNode,
        value_name: str,
        inputs: Sequence[str],
        *,
        size: int,
        prefix: str,
    ) -> None:
        calls.append((size, prefix))
        lines.append(f"  {value_name} = fadd double 0.00000000000000000e+00, {float(size):.1f}")
        for derivative_index in range(int(result.gradient.size)):
            lines.append(
                f"  {emitter._whole_program_native_derivative_name(node.index, derivative_index)} = "
                "fadd double 0.00000000000000000e+00, 0.00000000000000000e+00"
            )

    monkeypatch.setattr(emitter, "_WHOLE_PROGRAM_NATIVE_LOOP_HELPER_DET_SIZES", frozenset())
    monkeypatch.setattr(emitter, "_emit_whole_program_native_det_faddeev_leverrier", fake_faddeev)

    lines = _operation_lines(
        _result((_node(0, "parameter", ("x0",)),), trainable=(True,)),
        _node(9, "linalg:det:6x6", _tokens(36)),
    )

    assert calls == [(6, "det6")]
    assert any("%n9 = fadd" in line for line in lines)


def test_linalg_helper_emitters_cover_cached_inverse_and_solve_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared helper emitters should generate reusable determinant, inverse, and solve code."""

    monkeypatch.setattr(emitter, "_WHOLE_PROGRAM_NATIVE_FACTORISATION_HELPER_SIZES", frozenset())
    matrix_tokens = ("%0", "%1", *_tokens(23))
    rhs_tokens = ("%0", "%1", *_tokens(3))
    result = _result(
        (
            _node(0, "parameter", ("x0",)),
            _node(1, "parameter", ("x1",)),
        )
    )
    state = emitter._WholeProgramNativeEmissionState()
    lines: list[str] = []

    helper = emitter._emit_whole_program_native_inverse_helper(
        lines,
        matrix_tokens,
        size=5,
        prefix="inv5",
        emission_state=state,
    )
    repeated = emitter._emit_whole_program_native_inverse_helper(
        lines,
        matrix_tokens,
        size=5,
        prefix="inv5",
        emission_state=state,
    )
    emitter._emit_whole_program_native_inverse_fixed(
        lines,
        result,
        _node(20, "linalg:inv:5x5:0:1", matrix_tokens),
        "%n20",
        matrix_tokens,
        size=5,
        output_row=0,
        output_col=1,
        prefix="inv5",
        emission_state=state,
    )
    emitter._emit_whole_program_native_solve_fixed(
        lines,
        result,
        _node(21, "linalg:solve:5x5:rhs:5:2", (*matrix_tokens, *rhs_tokens)),
        "%n21",
        (*matrix_tokens, *rhs_tokens),
        size=5,
        output_row=2,
        prefix="solve5",
        emission_state=state,
    )

    text = "\n".join(lines)
    assert repeated is helper
    assert helper.determinant == "%inv5_shared_0_det"
    assert "call void @scpn_det5_fl_value_partials" in text
    assert "%n20 = fadd double" in text
    assert "%n21 = fadd double" in text
    assert "solve5_shared_0_solution_2" in text


def test_factorisation_helper_paths_and_guardrails() -> None:
    """Factorisation-backed helpers should reject invalid shapes and emit pivoting IR."""

    result = _result((_node(0, "parameter", ("x0",)),), trainable=(True,))
    state = emitter._WholeProgramNativeEmissionState()
    matrix_tokens = ("%0", *_tokens(24))
    rhs_tokens = ("%0", *_tokens(4))
    lines: list[str] = []

    with pytest.raises(ValueError, match="unsupported size"):
        emitter._emit_whole_program_native_inverse_helper(
            lines,
            matrix_tokens,
            size=4,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="unsupported size"):
        emitter._emit_whole_program_native_inverse_factorisation_helper(
            lines,
            matrix_tokens,
            size=4,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="full square matrix"):
        emitter._emit_whole_program_native_inverse_factorisation_helper(
            lines,
            matrix_tokens[:-1],
            size=5,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="one RHS column"):
        emitter._emit_whole_program_native_solve_helper(
            lines,
            matrix_tokens,
            rhs_tokens[:-1],
            size=5,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="unsupported size"):
        emitter._emit_whole_program_native_inverse_fixed(
            lines,
            result,
            _node(30, "linalg:inv:8x8:0:0", matrix_tokens),
            "%n30",
            matrix_tokens,
            size=8,
            output_row=0,
            output_col=0,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="outside the matrix"):
        emitter._emit_whole_program_native_inverse_fixed(
            lines,
            result,
            _node(31, "linalg:inv:5x5:5:0", matrix_tokens),
            "%n31",
            matrix_tokens,
            size=5,
            output_row=5,
            output_col=0,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="unsupported size"):
        emitter._emit_whole_program_native_solve_fixed(
            lines,
            result,
            _node(32, "linalg:solve:8x8:rhs:8:0", (*matrix_tokens, *rhs_tokens)),
            "%n32",
            (*matrix_tokens, *rhs_tokens),
            size=8,
            output_row=0,
            prefix="bad",
            emission_state=state,
        )
    with pytest.raises(ValueError, match="outside the solution"):
        emitter._emit_whole_program_native_solve_fixed(
            lines,
            result,
            _node(33, "linalg:solve:5x5:rhs:5:5", (*matrix_tokens, *rhs_tokens)),
            "%n33",
            (*matrix_tokens, *rhs_tokens),
            size=5,
            output_row=5,
            prefix="bad",
            emission_state=state,
        )

    emitter._emit_whole_program_native_inverse_fixed(
        lines,
        result,
        _node(34, "linalg:inv:5x5:1:2", matrix_tokens),
        "%n34",
        matrix_tokens,
        size=5,
        output_row=1,
        output_col=2,
        prefix="inv5f",
        emission_state=state,
    )
    emitter._emit_whole_program_native_solve_fixed(
        lines,
        result,
        _node(35, "linalg:solve:5x5:rhs:5:3", (*matrix_tokens, *rhs_tokens)),
        "%n35",
        (*matrix_tokens, *rhs_tokens),
        size=5,
        output_row=3,
        prefix="solve5f",
        emission_state=state,
    )

    text = "\n".join(lines)
    assert "candidate_is_better" in text
    assert "%n34 = fadd double" in text
    assert "%n35 = fadd double" in text


def test_determinant_derivative_helpers_and_faddeev_paths() -> None:
    """Determinant emitters should cover recursive, helper-backed, and FL derivative code."""

    lines: list[str] = []
    matrix3 = (
        ("%0", "1.0", "2.0"),
        ("3.0", "%1", "4.0"),
        ("5.0", "6.0", "7.0"),
    )
    derivative3 = (
        ("1.0", "0.0", "0.0"),
        ("0.0", "0.0", "0.0"),
        ("0.0", "0.0", "%d1_0"),
    )
    det3, derivatives3 = emitter._emit_whole_program_native_det_with_derivatives(
        lines,
        matrix3,
        (derivative3,),
        prefix="det3",
    )
    matrix5 = tuple(tuple(_tokens(25)[row * 5 + col] for col in range(5)) for row in range(5))
    zero = emitter._fmt_llvm_float(0.0)
    derivative5 = tuple(
        tuple("1.0" if row == col == 0 else zero for col in range(5)) for row in range(5)
    )
    det5, derivatives5 = emitter._emit_whole_program_native_det_with_derivatives(
        lines,
        matrix5,
        (derivative5,),
        prefix="det5",
    )
    result = _result((_node(0, "parameter", ("x0",)),), trainable=(True,))
    emitter._emit_whole_program_native_det_faddeev_leverrier(
        lines,
        result,
        _node(40, "linalg:det:3x3", _tokens(9)),
        "%n40",
        _tokens(9),
        size=3,
        prefix="fl3",
    )
    emitter._emit_whole_program_native_det_faddeev_leverrier(
        lines,
        result,
        _node(41, "linalg:det:4x4", _tokens(16)),
        "%n41",
        _tokens(16),
        size=4,
        prefix="fl4",
    )

    text = "\n".join(lines)
    assert det3.startswith("%det3")
    assert derivatives3 == ("%det3_d0",)
    assert det5 == "%det5_helper_det"
    assert derivatives5 == ("%det5_d0",)
    assert "call void @scpn_det5_fl_value_partials" in text
    assert "%n40 = fsub double" in text
    assert "%n41 = fadd double" in text

    with pytest.raises(ValueError, match="requires a square matrix"):
        emitter._emit_whole_program_native_det_with_derivatives(
            lines,
            (("1.0", "2.0"),),
            (),
            prefix="bad",
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        emitter._emit_whole_program_native_det_with_derivatives(
            lines,
            matrix3,
            ((("1.0",),),),
            prefix="bad",
        )
    with pytest.raises(ValueError, match="unsupported size"):
        emitter._emit_whole_program_native_det_helper_with_derivatives(
            lines,
            matrix3,
            (derivative3,),
            prefix="bad",
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        emitter._emit_whole_program_native_det_helper_with_derivatives(
            lines,
            (
                ("1.0", "2.0", "3.0", "4.0", "5.0"),
                ("1.0", "2.0", "3.0", "4.0", "5.0"),
                ("1.0", "2.0", "3.0", "4.0", "5.0"),
                ("1.0", "2.0", "3.0", "4.0", "5.0"),
                ("1.0",),
            ),
            (),
            prefix="bad",
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        emitter._emit_whole_program_native_det_helper_with_derivatives(
            lines,
            matrix5,
            ((("1.0",),),),
            prefix="bad",
        )
    with pytest.raises(ValueError, match="unsupported size"):
        emitter._emit_whole_program_native_det_loop_helper_call(
            lines,
            result,
            _node(42, "linalg:det:4x4", _tokens(16)),
            "%n42",
            _tokens(16),
            size=4,
            prefix="bad",
        )


def test_spec_parsers_predicates_operands_and_formatters_fail_closed() -> None:
    """Spec parsers and token helpers should accept only supported native contracts."""

    assert emitter._whole_program_native_where_branch_op("%0:gt:%1") is None
    assert emitter._whole_program_native_where_branch_op("left:truth:2") is None
    assert emitter._whole_program_native_where_branch_op("left:truth:1") == "branch:left:True"
    assert emitter._whole_program_native_where_branch_op("left:truth:0") == "branch:left:False"
    assert emitter._whole_program_native_signature_inputs(_node(0, "add", ("%0", "%1"))) == (
        "%0",
        "%1",
    )
    assert emitter._whole_program_native_signature_inputs(
        _node(1, "where", ("%0:gt:%1:truth:1", "%2", "%3"))
    ) == ("%0:gt:%1", "%2", "%3")
    assert emitter._whole_program_native_det_loop_helper_symbol(6) == "scpn_det6_fl_value_partials"
    assert emitter._whole_program_native_wide_det_size("not-det") is None
    assert emitter._whole_program_native_wide_det_size("linalg:det:6x") is None
    assert emitter._whole_program_native_inverse_spec("linalg:inv:3x3:1:2") == (3, 1, 2)
    assert emitter._whole_program_native_inverse_spec("linalg:inv:axb:0:0") is None
    assert emitter._whole_program_native_solve_vector_spec("linalg:solve:3x3:rhs:3:2") == (3, 2)
    assert emitter._whole_program_native_solve_vector_spec("linalg:solve:axb:rhs:3:0") is None
    assert emitter._whole_program_native_solve_matrix_spec("linalg:solve:3x3:rhs:3x2:2:1") == (
        3,
        2,
        2,
        1,
    )
    assert emitter._whole_program_native_solve_matrix_spec("linalg:solve:3x3:rhs:3xb:0:0") is None
    assert emitter._whole_program_native_trace_input_count("not-trace") is None
    assert emitter._whole_program_native_trace_input_count("linalg:trace:2x:offset:0") is None
    assert emitter._whole_program_native_trace_input_count("linalg:trace:0x2:offset:0") is None
    assert emitter._whole_program_native_trace_input_count("linalg:trace:2x2:offset:5") is None
    assert (
        emitter._whole_program_native_diag_input_count("linalg:sum:2:offset:0:construct:0") is None
    )
    assert (
        emitter._whole_program_native_diag_input_count("linalg:diag:2:bad:0:construct:0") is None
    )
    assert (
        emitter._whole_program_native_diag_input_count("linalg:diag:a:offset:0:construct:0")
        is None
    )
    assert (
        emitter._whole_program_native_diag_input_count("linalg:diag:0:offset:0:construct:0")
        is None
    )
    assert emitter._whole_program_native_diag_input_count("linalg:diag:2:offset:0:bad:0") is None
    assert (
        emitter._whole_program_native_diag_input_count("linalg:diagflat:2:offset:0:extract:0")
        is None
    )
    assert (
        emitter._whole_program_native_diag_input_count("linalg:diag:2:offset:0:construct:-1")
        is None
    )

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            emitter,
            "_WHOLE_PROGRAM_NATIVE_FACTORISATION_HELPER_SIZES",
            frozenset(),
        )
        assert emitter._whole_program_native_det_derivative_helper_size("linalg:inv:5x5:0:0") == 5
        assert (
            emitter._whole_program_native_det_derivative_helper_size("linalg:solve:5x5:rhs:5:0")
            == 5
        )
        assert (
            emitter._whole_program_native_det_derivative_helper_size(
                "linalg:solve:5x5:rhs:5x2:0:0"
            )
            == 5
        )
    finally:
        monkeypatch.undo()

    predicate_lines: list[str] = []
    assert (
        emitter._emit_whole_program_native_where_predicate(
            predicate_lines,
            7,
            "%0:ne:2.0:truth:1",
        )
        == "%where_pred_7"
    )
    assert "fcmp one" in predicate_lines[-1]
    with pytest.raises(ValueError, match="predicate is malformed"):
        emitter._emit_whole_program_native_where_predicate([], 1, "bad:truth:1")
    with pytest.raises(ValueError, match="predicate op bad is unsupported"):
        emitter._emit_whole_program_native_where_predicate([], 1, "%0:bad:%1:truth:1")
    with pytest.raises(ValueError, match="requires recorded truth"):
        emitter._whole_program_native_where_predicate_body("%0:gt:%1")
    with pytest.raises(ValueError, match="truth must be 0 or 1"):
        emitter._whole_program_native_where_predicate_body("%0:gt:%1:truth:2")
    with pytest.raises(ValueError, match="cannot lower operand"):
        emitter._whole_program_native_operand("not-a-number")
    with pytest.raises(ValueError, match="cannot lower derivative operand"):
        emitter._whole_program_native_derivative_operand("not-a-number", 0)
    with pytest.raises(ValueError, match="numeric constants must be finite"):
        emitter._fmt_llvm_float(float("inf"))
    with pytest.raises(ValueError, match="integer constants must be positive"):
        emitter._fmt_llvm_int(0)


def test_structural_derivative_operand_folds_parameter_seeds() -> None:
    """Structural derivatives should fold direct parameter seeds when possible."""

    parameter = _node(0, "parameter", ("x0",))
    frozen = _node(1, "parameter", ("x1",))
    out_of_order = _node(5, "parameter", ("x0",))
    non_parameter = _node(2, "constant", ("1.0",), value=1.0)
    missing_name = _node(3, "parameter", ("missing",))
    result = _result(
        (parameter, frozen, out_of_order, non_parameter, missing_name),
        trainable=(True, False),
    )

    assert emitter._whole_program_native_structural_derivative_operand(result, "1.0", 0) == (
        "0.00000000000000000e+00"
    )
    assert emitter._whole_program_native_structural_derivative_operand(result, "%0", 0) == (
        "1.00000000000000000e+00"
    )
    assert emitter._whole_program_native_structural_derivative_operand(result, "%1", 1) == (
        "0.00000000000000000e+00"
    )
    assert emitter._whole_program_native_structural_derivative_operand(result, "%5", 0) == (
        "1.00000000000000000e+00"
    )
    assert emitter._whole_program_native_structural_derivative_operand(result, "%9", 0) == "%d9_0"
    assert emitter._whole_program_native_structural_derivative_operand(result, "%2", 0) == "%d2_0"
    assert emitter._whole_program_native_structural_derivative_operand(result, "%3", 0) == "%d3_0"


def test_sum_and_batch_emitters_cover_identity_and_accumulation_paths() -> None:
    """Batch and sum helpers should emit deterministic LLVM loops and sum identities."""

    result = _result((_node(0, "parameter", ("x0",)), _node(1, "parameter", ("x1",))))
    jvp = emitter._emit_whole_program_native_batch_jvp(
        result,
        "kernel",
        ("  %n0 = fadd double 0.00000000000000000e+00, 1.00000000000000000e+00",),
        ("%d0_0", "%d0_1"),
    )
    vjp = emitter._emit_whole_program_native_batch_vjp(
        result,
        "kernel",
        ("  %n0 = fadd double 0.00000000000000000e+00, 1.00000000000000000e+00",),
        ("%d0_0", "%d0_1"),
    )
    lines: list[str] = []
    with pytest.raises(ValueError, match="requires at least one input"):
        emitter._emit_whole_program_native_sum(
            [],
            result,
            _node(3, "trace", ()),
            "%n3",
            (),
            "trace",
        )
    emitter._emit_whole_program_native_sum(
        lines,
        result,
        _node(4, "trace", ("1.0",)),
        "%n4",
        ("1.0",),
        "trace",
    )
    emitter._emit_whole_program_native_sum_operands(lines, (), "%empty")
    emitter._emit_whole_program_native_sum_operands(lines, ("%a",), "%one")
    emitter._emit_whole_program_native_sum_operands(lines, ("%a", "%b", "%c"), "%many")

    assert "define void @kernel_batch_jvp" in "\n".join(jvp)
    assert "define void @kernel_batch_vjp" in "\n".join(vjp)
    assert any("%empty = fadd double" in line for line in lines)
    assert any("%one = fadd double %a" in line for line in lines)
    assert any("%many = fadd double" in line for line in lines)
