# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program trace runtime tests
"""Tests for whole-program AD trace-runtime metadata builders."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable import TraceADScalar
from scpn_quantum_control.whole_program_frontend import (
    WholeProgramBytecodeInstruction,
    WholeProgramSourceIRFeature,
)
from scpn_quantum_control.whole_program_trace_runtime import (
    _trace_whole_program_objective,
)
from scpn_quantum_control.whole_program_trace_runtime import (
    _WholeProgramTraceContext as ExtractedWholeProgramTraceContext,
)


def _context() -> ExtractedWholeProgramTraceContext:
    """Return a trace context bound to the facade-owned scalar wrapper."""

    return ExtractedWholeProgramTraceContext(2, scalar_factory=TraceADScalar)


def test_trace_context_facade_identity_and_factory_binding() -> None:
    """The facade should expose the extracted context and require a scalar factory."""

    assert vars(differentiable)["_WholeProgramTraceContext"] is ExtractedWholeProgramTraceContext
    context = ExtractedWholeProgramTraceContext(1)
    with pytest.raises(RuntimeError, match="no scalar factory"):
        context.make("parameter", ("theta",), 1.0, np.array([1.0], dtype=np.float64))

    context.bind_scalar_factory(TraceADScalar)
    scalar = context.make("parameter", ("theta",), 1.0, np.array([1.0], dtype=np.float64))

    assert scalar.name == "%0"
    assert scalar.context is context
    assert context.nodes[0].op == "parameter"
    assert context.effects[0].kind == "parameter"
    assert context.ssa_values[0].effect == 0


def test_trace_context_emits_effect_alias_and_branch_metadata() -> None:
    """Scalar emission should preserve effect, mutation, primitive, and branch evidence."""

    context = _context()
    parameter = context.make("parameter", ("theta",), 2.0, np.array([1.0, 0.0]))
    mutation = context.make("mutation:list_setitem", (parameter.name,), 3.0, np.array([1.0, 0.0]))
    context.make("branch:test:False", (), 0.0, np.array([0.0, 0.0]))
    primitive = context.make("sin", (parameter.name,), 0.5, np.array([0.25, 0.0]))
    pure = context.make("add", (primitive.name, mutation.name), 3.5, np.array([1.25, 0.0]))

    assert [node.op for node in context.nodes] == [
        "parameter",
        "mutation:list_setitem",
        "branch:test:False",
        "sin",
        "add",
    ]
    assert [effect.kind for effect in context.effects] == [
        "parameter",
        "mutation",
        "control_branch",
        "primitive",
        "pure",
    ]
    assert context.alias_edges[0].kind == "mutation_version"
    assert context.alias_edges[0].source == parameter.name
    assert context.control_regions[0].entered is False
    assert context.phi_nodes[0].selected == "executed_false"
    np.testing.assert_allclose(pure.tangent, np.array([1.25, 0.0]))


def test_trace_context_records_array_view_aliases() -> None:
    """Array-view alias metadata should validate source indices and trace ownership."""

    context = _context()
    first = context.make("parameter", ("x",), 1.0, np.array([1.0, 0.0]))
    second = context.make("parameter", ("y",), 2.0, np.array([0.0, 1.0]))

    context.record_array_view_aliases("reshape", (0, None), (first, second))

    assert [(edge.source, edge.target, edge.kind) for edge in context.alias_edges] == [
        ("%array[0]", "view:reshape:0[0]", "view_alias"),
        ("view:reshape:0[0]", first.name, "view_alias"),
        ("view:reshape:0[1]", second.name, "view_alias"),
    ]
    with pytest.raises(ValueError, match="source and item counts"):
        context.record_array_view_aliases("bad", (0,), (first, second))
    with pytest.raises(ValueError, match="non-negative"):
        context.record_array_view_aliases("bad", (-1,), (first,))
    other_context = _context()
    foreign = other_context.make("parameter", ("z",), 4.0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="different trace"):
        context.record_array_view_aliases("bad", (0,), (foreign,))


def test_trace_context_builds_program_effect_ir_from_source_features() -> None:
    """Program IR assembly should merge runtime evidence with source-level features."""

    context = _context()
    context.make("parameter", ("theta",), 1.0, np.array([1.0, 0.0]))
    features = (
        WholeProgramSourceIRFeature("local_rebinding_alias", "a->b", 10),
        WholeProgramSourceIRFeature("array_alias", "buffer", 11),
        WholeProgramSourceIRFeature("runtime_branch", "theta > 0", 12),
        WholeProgramSourceIRFeature("for_loop", "for item in items", 13),
    )
    bytecode = (
        WholeProgramBytecodeInstruction(
            offset=2,
            opname="LOAD_FAST",
            argrepr="theta",
            line_number=10,
        ),
    )

    program_ir = context.program_ir(source_ir_features=features, bytecode_instructions=bytecode)
    payload = json.loads(program_ir.serialization)

    assert payload["format"] == "program_ad_effect_ir.v1"
    assert payload["bytecode_offsets"] == [2]
    assert [edge.kind for edge in program_ir.alias_edges] == [
        "local_rebinding_alias",
        "array_alias",
    ]
    assert [region.kind for region in program_ir.control_regions] == [
        "source_runtime_branch",
        "source_for_loop",
    ]
    assert program_ir.phi_nodes[0].incoming == ("executed_path", "non_executed_path")
    assert program_ir.phi_nodes[1].incoming == ("loop_entry", "loop_backedge")


def test_trace_context_rejects_malformed_explicit_alias_feature() -> None:
    """Explicit source alias features should encode a source and target."""

    context = _context()
    with pytest.raises(ValueError, match="must encode source->target"):
        context.program_ir(
            source_ir_features=(
                WholeProgramSourceIRFeature("control_path_alias", "missing-target", 4),
            ),
            bytecode_instructions=(),
        )


def test_trace_whole_program_objective_captures_source_lines_and_restores_tracer() -> None:
    """The source-line tracer should capture objective lines and restore the previous hook."""

    values = np.array([2.0, 3.0], dtype=np.float64)

    def objective(parameters: NDArray[np.float64]) -> float:
        parameters[0] = 99.0
        total = parameters[0] + parameters[1]
        return float(total)

    previous_trace = sys.gettrace()
    events = _trace_whole_program_objective(objective, values)

    assert sys.gettrace() is previous_trace
    assert values.tolist() == [2.0, 3.0]
    assert {event.function_name for event in events} == {"objective"}
    assert any("total = parameters[0] + parameters[1]" in event.source for event in events)
    assert all(event.line_number > 0 for event in events)


def test_trace_whole_program_objective_deduplicates_manual_tracer_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed trace callback should record each source line only once."""

    captured: dict[str, object] = {}

    def fake_settrace(tracer: object) -> None:
        captured["tracer"] = tracer

    monkeypatch.setattr(sys, "settrace", fake_settrace)

    def objective(values: NDArray[np.float64]) -> float:
        tracer = captured["tracer"]
        assert callable(tracer)
        frame = sys._getframe()
        tracer(frame, "call", None)
        for _ in range(2):
            tracer(frame, "line", None)
        return float(values[0] + values[1])

    events = _trace_whole_program_objective(objective, np.array([1.0, 2.0]))

    assert len(events) == 1
    assert events[0].function_name == "objective"
    assert events[0].source.strip() == 'tracer(frame, "line", None)'


def test_trace_whole_program_objective_handles_code_less_callable() -> None:
    """Callables without a direct ``__code__`` attribute should not be traced."""

    class Objective:
        def __call__(self, values: NDArray[np.float64]) -> float:
            return float(values.sum())

    assert _trace_whole_program_objective(Objective(), np.array([1.0], dtype=np.float64)) == ()


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (True, "real numeric scalar"),
        (complex(1.0, 0.0), "real numeric scalar"),
        (float("nan"), "finite"),
    ],
)
def test_trace_whole_program_objective_rejects_invalid_scalar_results(
    raw: object, match: str
) -> None:
    """Trace replay should keep finite real scalar objective contracts."""

    def objective(values: NDArray[np.float64]) -> object:
        del values
        return raw

    with pytest.raises(ValueError, match=match):
        _trace_whole_program_objective(objective, np.array([1.0], dtype=np.float64))
