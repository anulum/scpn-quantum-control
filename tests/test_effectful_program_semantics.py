# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — effectful program semantic qualification
"""Public runtime qualification for control flow, scatter, effects and replay."""

from __future__ import annotations

import json
from copy import copy
from dataclasses import replace
from threading import Event
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control import (
    TraceADArray,
    TraceADScalar,
    whole_program_value_and_grad,
)
from scpn_quantum_control.differentiable import (
    parse_program_ad_effect_ir,
    program_adjoint_gradient,
    program_adjoint_replay_gradient,
)
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)


@pytest.mark.parametrize(
    "kind", ["external_callback", "ambient_rng", "nondifferentiable", "unregistered_effect"]
)
@pytest.mark.parametrize("consumer", ["replacement", "copy", "replay", "native"])
def test_unknown_imported_effect_cannot_acquire_an_existing_derivative(
    kind: str, consumer: str
) -> None:
    """Imported external effects cannot inherit an ordinary arithmetic pullback.

    Parameters
    ----------
    kind
        Unsupported external, random, nondifferentiable or unregistered effect.
    consumer
        Public result construction, gradient copying, Python replay or Rust FFI.

    """

    def objective(values: TraceADArray) -> object:
        return values[0] * values[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.program_ir is not None
    original = result.program_ir.serialization
    payload: object = json.loads(original)
    assert isinstance(payload, dict)
    effects: object = payload["effects"]
    assert isinstance(effects, list)
    row: object = effects[-1]
    assert isinstance(row, dict)
    effect_index = row["index"]
    row["kind"] = kind
    imported = parse_program_ad_effect_ir(json.dumps(payload, sort_keys=True))
    if consumer == "native":
        native = value_and_grad_program_ad_effect_ir_with_rust(imported, np.array([2.0]))
        assert not native.supported
        assert native.gradient.size == 0
        assert native.parameter_targets == ()
        assert native.value is None
        assert any(
            kind in reason and str(effect_index) in reason for reason in native.blocked_reasons
        )
    elif consumer == "replacement":
        with pytest.raises(ValueError, match=rf"effect {effect_index}.*{kind}"):
            replace(result, program_ir=imported)
    else:
        corrupted = copy(result)
        object.__setattr__(corrupted, "program_ir", imported)
        getter = (
            program_adjoint_gradient if consumer == "copy" else program_adjoint_replay_gradient
        )
        with pytest.raises(ValueError, match=rf"effect {effect_index}.*{kind}"):
            getter(corrupted)
    assert result.program_ir.serialization == original
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])
    native_original = value_and_grad_program_ad_effect_ir_with_rust(
        result.program_ir, np.array([2.0])
    )
    assert native_original.supported, native_original.blocked_reasons
    assert native_original.value == 4.0
    np.testing.assert_array_equal(native_original.gradient, [4.0])


@pytest.mark.parametrize("trace", [False, True])
def test_effectful_program_semantics_01(trace: bool) -> None:
    """Duplicate-index scatter accumulates values and cotangents sequentially.

    Parameters
    ----------
    trace
        Whether the public execution also collects source-line evidence.

    """

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        np.add.at(working, [0, 0, 2], values)
        return cast(TraceADArray, working**2).sum()

    inputs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    reference = inputs.copy()
    np.add.at(reference, [0, 0, 2], inputs)
    result = whole_program_value_and_grad(objective, inputs, trace=trace)

    assert result.value == pytest.approx(float(np.sum(reference**2)), abs=1.0e-12)
    # w=(2*x+y,y,2*z), so d(sum(w**2))=(4*w0,2*w0+2*y,4*w2).
    expected = np.array([16.0, 12.0, 24.0], dtype=np.float64)
    np.testing.assert_allclose(result.gradient, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result), expected, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_array_equal(inputs, [1.0, 2.0, 3.0])
    assert result.program_ir is not None
    assert any(effect.kind == "mutation" for effect in result.program_ir.effects)


@pytest.mark.parametrize(
    ("x", "expected_value", "expected_gradient"),
    [(2.0, 7.0, (4.0, 1.0)), (-2.0, 3.0, (3.0, 6.0))],
)
def test_effectful_program_semantics_02(
    x: float, expected_value: float, expected_gradient: tuple[float, float]
) -> None:
    """Both active branches agree with analytic derivatives and native IR replay.

    Parameters
    ----------
    x
        Positive or negative first parameter selecting a smooth branch interior.
    expected_value
        Independently evaluated piecewise-polynomial value at the fixture.
    expected_gradient
        Analytic local derivative of the selected branch, in parameter order.

    """

    def objective(values: TraceADArray) -> object:
        first, second = values
        return first * first + second if first > 0.0 else 3.0 * first + second * second

    inputs = np.array([x, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(objective, inputs)
    assert result.value == pytest.approx(expected_value, abs=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result), expected_gradient, rtol=0.0, atol=1.0e-12
    )
    assert result.program_ir is not None
    native = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, inputs)
    assert native.supported, native.blocked_reasons
    assert native.value == pytest.approx(expected_value, abs=1.0e-12)
    np.testing.assert_allclose(native.gradient, expected_gradient, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("trace", [False, True])
def test_effectful_program_semantics_03(trace: bool) -> None:
    """Unregistered callbacks refuse before invoking external side effects.

    Parameters
    ----------
    trace
        Whether the source-line trace would otherwise invoke the callback twice.

    """
    calls: list[str] = []

    def external_callback() -> float:
        calls.append("called")
        return 7.0

    def objective(values: TraceADArray) -> object:
        return values[0] ** 2 + external_callback()

    with pytest.raises(ValueError, match="callback|external") as refusal:
        whole_program_value_and_grad(objective, np.array([2.0]), trace=trace)
    assert "line=" in str(refusal.value)
    assert calls == []


def test_effectful_program_semantics_04() -> None:
    """Captured coefficient mutation invalidates replay without changing its tape."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    result = whole_program_value_and_grad(objective, np.array([3.0]), trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert result.program_ir is not None
    serialization = result.program_ir.serialization
    state[0] = 5.0

    with pytest.raises(ValueError, match="captured.*(changed|state)|state.*changed"):
        program_adjoint_replay_gradient(result)
    assert result.program_ir.serialization == serialization
    np.testing.assert_array_equal(result.gradient, [2.0])
    assert state == [5.0]


@pytest.mark.parametrize("trace", [False, True])
def test_captured_mutation_is_refused_before_execution(trace: bool) -> None:
    """Captured indexed writes refuse with a location and leave prior state intact.

    Parameters
    ----------
    trace
        Whether a second execution would otherwise repeat the captured mutation.

    """
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        state[0] = state[0] + 1.0
        return values[0] * state[0]

    with pytest.raises(ValueError, match="captured|mutation") as refusal:
        whole_program_value_and_grad(objective, np.array([3.0]), trace=trace)
    assert "line=" in str(refusal.value)
    assert state == [2.0]


def test_ambient_rng_is_refused_without_advancing_random_state() -> None:
    """Ambient NumPy randomness cannot become an unrecorded constant effect."""

    def objective(values: TraceADArray) -> object:
        return values[0] ** 2 + np.random.random()

    before = np.random.get_state(legacy=True)
    assert isinstance(before, tuple)
    try:
        with pytest.raises(ValueError, match="random|rng") as refusal:
            whole_program_value_and_grad(objective, np.array([2.0]))
        assert "line=" in str(refusal.value)
        after = np.random.get_state(legacy=True)
        assert isinstance(after, tuple)
        assert after[0] == before[0]
        np.testing.assert_array_equal(after[1], before[1])
        assert after[2:] == before[2:]
    finally:
        np.random.set_state(before)


def test_duplicate_gather_accumulates_reverse_contributions_in_native_ir() -> None:
    """Repeated gather indices contribute twice to their original parameter."""

    def objective(values: TraceADArray) -> object:
        gathered: object = np.take(values, [2, 0, 2])
        return cast(TraceADArray, gathered * np.array([1.5, -2.0, 0.25])).sum()

    inputs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(objective, inputs)
    assert result.value == 3.25
    np.testing.assert_array_equal(result.gradient, [-2.0, 0.0, 1.75])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), result.gradient)
    assert result.program_ir is not None
    native = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, inputs)
    assert native.supported, native.blocked_reasons
    assert native.value == result.value
    np.testing.assert_array_equal(native.gradient, result.gradient)


def test_loop_carried_state_preserves_primal_and_local_reverse_derivative() -> None:
    """The captured bounded loop agrees with an independent expanded polynomial."""

    def objective(values: TraceADArray) -> object:
        total = values[0]
        for index in range(1, 4):
            total = total * values[1] + index * values[0]
        return total

    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]))
    # x*(y**3+y**2+2*y+3), differentiated at x=2 and y=3.
    assert result.value == 90.0
    np.testing.assert_array_equal(result.gradient, [45.0, 70.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), result.gradient)
    assert result.program_ir is not None
    assert any(edge.kind == "loop_carried_state" for edge in result.program_ir.alias_edges)


def test_dynamic_shape_refuses_with_an_explicit_boundary() -> None:
    """Parameter-dependent integer shape conversion cannot silently detach AD."""

    def objective(values: TraceADArray) -> object:
        count = int(cast(TraceADScalar, values[0]))
        return np.sum(np.zeros(count)) + values[0]

    with pytest.raises(ValueError, match="integer|int|shape|unsupported"):
        whole_program_value_and_grad(objective, np.array([2.0]))


def test_cancellation_refuses_before_external_callback_invocation() -> None:
    """An already cancelled scope leaves callback state and the caller input intact."""
    calls: list[str] = []

    def external_callback() -> float:
        calls.append("called")
        return 7.0

    def objective(values: TraceADArray) -> object:
        return values[0] ** 2 + external_callback()

    cancelled = Event()
    cancelled.set()
    inputs = np.array([2.0])
    with pytest.raises(RuntimeError, match="cancel"):
        whole_program_value_and_grad(objective, inputs, cancelled=cancelled)
    assert calls == []
    np.testing.assert_array_equal(inputs, [2.0])
