# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD tape correspondence tests
"""Keep runtime primal, typed IR, raw evidence and derivatives together."""

from __future__ import annotations

import sys
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import replace
from threading import Event
from time import monotonic
from types import FrameType

import numpy as np
import pytest

from scpn_quantum_control import TraceADArray, whole_program_value_and_grad
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import (
    program_adjoint_gradient,
    program_adjoint_replay_gradient,
    program_adjoint_result,
)
from scpn_quantum_control.differentiable_parameter_contracts import Parameter
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
)
from scpn_quantum_control.program_ad_effect_ir import ProgramADEffectIR
from scpn_quantum_control.program_ad_rust_bridge import (
    interpret_program_ad_effect_ir_with_rust,
    value_and_grad_program_ad_effect_ir_with_rust,
)
from scpn_quantum_control.whole_program_ad_result import WholeProgramADResult, WholeProgramIRNode


def _square(values: TraceADArray) -> object:
    return values[0] * values[0]


def _twice(values: TraceADArray) -> object:
    return values[0] + values[0]


def _replacement(
    result: WholeProgramADResult, part: str
) -> float | ProgramADEffectIR | tuple[WholeProgramIRNode, ...]:
    other = whole_program_value_and_grad(_twice, [2.0], trace=False)
    assert result.program_ir is not None and other.program_ir is not None
    if part == "program_ir":
        return other.program_ir
    if part == "typed_effects":
        return replace(result.program_ir, effects=other.program_ir.effects)
    if part == "raw_serialization":
        return replace(result.program_ir, serialization=other.program_ir.serialization)
    if part == "value":
        return 99.0
    return (*result.ir_nodes[:-1], replace(result.ir_nodes[-1], value=99.0))


@pytest.mark.parametrize(
    "part", ["program_ir", "typed_effects", "raw_serialization", "value", "ir_nodes"]
)
def test_runtime_replacement_refuses_mixed_primal_ir_and_derivative(part: str) -> None:
    """A valid foreign IR cannot inherit the original objective's pullback.

    Parameters
    ----------
    part
        Valid foreign IR, one changed IR projection, primal or node metadata.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    candidate = _replacement(result, part)
    field = "program_ir" if part in {"typed_effects", "raw_serialization"} else part
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured derivative tape"):
        if field == "program_ir":
            assert isinstance(candidate, ProgramADEffectIR)
            replace(result, program_ir=candidate)
        elif field == "value":
            assert isinstance(candidate, float)
            replace(result, value=candidate)
        else:
            assert isinstance(candidate, tuple)
            replace(result, ir_nodes=candidate)
    assert active_reserved_bytes() == baseline
    assert result.value == 4.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])


@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
@pytest.mark.parametrize(
    "part", ["program_ir", "typed_effects", "raw_serialization", "value", "ir_nodes"]
)
def test_post_construction_corruption_refuses_and_restoration_recovers(
    access: str, part: str
) -> None:
    """All public derivative getters observe corruption of the bound artifact.

    Parameters
    ----------
    access
        Public metadata, gradient-copy or executable-replay getter.
    part
        IR, one changed IR projection, primal or captured node metadata.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    candidate = _replacement(result, part)
    field = "program_ir" if part in {"typed_effects", "raw_serialization"} else part
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    original = getattr(result, field)
    assert result.program_ir is not None and result.adjoint_result is not None
    raw = result.program_ir.serialization
    adjoint = result.adjoint_result.to_dict()
    baseline = active_reserved_bytes()
    object.__setattr__(result, field, candidate)
    try:
        with pytest.raises(ValueError, match="captured derivative tape"):
            getters[access](result)
        assert active_reserved_bytes() == baseline
    finally:
        object.__setattr__(result, field, original)
    assert result.program_ir.serialization == raw
    assert result.adjoint_result.to_dict() == adjoint
    np.testing.assert_array_equal(program_adjoint_gradient(result), [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])


@pytest.mark.parametrize("part", ["typed_effects", "raw_serialization"])
def test_native_typed_ir_refuses_a_different_serialized_computation(part: str) -> None:
    """Actual FFI cannot execute raw arithmetic that differs from its typed IR.

    Parameters
    ----------
    part
        Typed operation rows or raw serialization replaced by another objective.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.program_ir is not None
    imported = _replacement(result, part)
    assert isinstance(imported, ProgramADEffectIR)
    with pytest.raises(ValueError, match="typed rows.*serialization"):
        value_and_grad_program_ad_effect_ir_with_rust(imported, np.array([2.0]))
    for original in (result.program_ir, result.program_ir.serialization):
        native = value_and_grad_program_ad_effect_ir_with_rust(original, np.array([2.0]))
        assert native.supported, native.blocked_reasons
        assert native.value == 4.0
        np.testing.assert_array_equal(native.gradient, [4.0])


@pytest.mark.parametrize("copying", [copy, deepcopy, replace])
def test_ordinary_copies_preserve_the_same_computation(
    copying: Callable[[WholeProgramADResult], WholeProgramADResult],
) -> None:
    """Binding is based on numerical content rather than Python object identity.

    Parameters
    ----------
    copying
        Standard shallow copy, deep copy or unchanged dataclass replacement.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    copied = copying(result)
    assert copied is not result
    assert copied.value == 4.0
    np.testing.assert_array_equal(program_adjoint_gradient(copied), [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(copied), [4.0])


@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
@pytest.mark.parametrize("storage", ["forward_gradient", "node_tangent", "effect_operation"])
def test_mutable_storage_cannot_change_an_existing_derivative(access: str, storage: str) -> None:
    """Finite buffer and row mutations refuse rather than acquiring a new tape.

    Parameters
    ----------
    access
        Metadata retrieval, gradient copying or Python replay.
    storage
        Forward gradient, node tangent or an emitted effect operation.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.program_ir is not None
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    original_raw = result.program_ir.serialization
    baseline = active_reserved_bytes()
    if storage == "effect_operation":
        row = result.program_ir.effects[-1]
        operation = row.operation
        object.__setattr__(row, "operation", "add")
        try:
            with pytest.raises(ValueError, match="captured derivative tape"):
                getters[access](result)
        finally:
            object.__setattr__(row, "operation", operation)
    else:
        buffer = result.gradient if storage == "forward_gradient" else result.ir_nodes[-1].tangent
        original = buffer.copy()
        buffer[0] += 1.0
        try:
            with pytest.raises(ValueError, match="captured derivative tape"):
                getters[access](result)
        finally:
            buffer[:] = original
    assert active_reserved_bytes() == baseline
    assert result.program_ir.serialization == original_raw
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])


def test_missing_tape_binding_refuses_and_restoration_recovers() -> None:
    """Removing the private content companion cannot detach a live derivative."""
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.captured_state is not None
    original = result.captured_state.tape_digest
    object.__setattr__(result.captured_state, "tape_digest", None)
    try:
        for getter in (
            program_adjoint_result,
            program_adjoint_gradient,
            program_adjoint_replay_gradient,
        ):
            with pytest.raises(ValueError, match="captured derivative tape"):
                getter(result)
    finally:
        object.__setattr__(result.captured_state, "tape_digest", original)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])


@pytest.mark.parametrize("corruption", ["opaque", "array_dtype", "nested_inputs"])
def test_unrepresentable_runtime_tape_refuses_without_invoking_object_protocols(
    corruption: str,
) -> None:
    """A malformed stored tape cannot dispatch arbitrary serialisation hooks.

    Parameters
    ----------
    corruption
        Opaque primal, wrong numeric storage or deeply nested input metadata.

    """
    called: list[str] = []

    class Opaque:
        def __repr__(self) -> str:
            called.append("repr")
            return "opaque"

    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    baseline = active_reserved_bytes()
    target: object = result
    field = "value"
    candidate: object = Opaque()
    if corruption == "array_dtype":
        target = result.ir_nodes[-1]
        field = "tangent"
        candidate = np.array([4], dtype=np.int64)
    elif corruption == "nested_inputs":
        target = result.ir_nodes[-1]
        field = "inputs"
        candidate = "theta_0"
        for _ in range(70):
            candidate = (candidate,)
    original = getattr(target, field)
    object.__setattr__(target, field, candidate)
    try:
        with pytest.raises(ValueError, match="captured derivative tape"):
            program_adjoint_result(result)
        assert not called
        assert active_reserved_bytes() == baseline
    finally:
        object.__setattr__(target, field, original)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])


@pytest.mark.parametrize("access", ["copy", "replay"])
def test_tape_validation_keeps_the_public_resource_policy(access: str) -> None:
    """Content revalidation inherits memory, cancellation and deadline admission.

    Parameters
    ----------
    access
        Gradient-copy or Python-replay entry point carrying the caller policy.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    getter = program_adjoint_gradient if access == "copy" else program_adjoint_replay_gradient
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        getter(result, max_execution_gib=1.0 / 1024**3)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ExecutionCancelledError):
        getter(result, cancelled=cancelled)
    with pytest.raises(TimeoutError):
        getter(result, deadline_monotonic=monotonic() - 1.0)
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(getter(result), [4.0])


def test_native_forward_keeps_typed_wire_correspondence_and_recovers() -> None:
    """The actual forward interpreter enforces the same typed/raw admission."""
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.program_ir is not None
    imported = _replacement(result, "raw_serialization")
    assert isinstance(imported, ProgramADEffectIR)
    with pytest.raises(ValueError, match="typed rows.*serialization"):
        interpret_program_ad_effect_ir_with_rust(imported, np.array([2.0]))
    original = interpret_program_ad_effect_ir_with_rust(result.program_ir, np.array([2.0]))
    assert original.supported, original.blocked_reasons
    assert original.value == 4.0


def test_unicode_parameter_names_keep_their_exact_tape_content() -> None:
    """UTF-8 width and surrogate metadata survive ordinary copies and replay."""
    name = "xéẞ𐀀\udc00" + "é" * 4096
    result = whole_program_value_and_grad(
        _square, [2.0], parameters=[Parameter(name)], trace=False
    )
    assert result.parameter_names == (name,)
    assert result.value == 4.0
    copied = deepcopy(result)
    assert copied.parameter_names == (name,)
    np.testing.assert_array_equal(program_adjoint_gradient(copied), [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(copied), [4.0])


def test_capture_finalisation_refuses_removed_binding_without_exposing_a_result() -> None:
    """A real finalisation-entry fault cannot expose an unbound runtime tape."""
    previous = sys.getprofile()
    observed = False
    baseline = active_reserved_bytes()

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal observed
        del argument
        if event == "call" and frame.f_code.co_name == "_bind_program_tape":
            result = frame.f_locals["result"]
            assert isinstance(result, WholeProgramADResult)
            observed = True
            object.__setattr__(result, "captured_state", None)

    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="captured derivative tape"):
            whole_program_value_and_grad(_square, [2.0], trace=False)
    finally:
        sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])


def test_standalone_historical_tape_remains_executable_without_a_live_binding() -> None:
    """The original unbound record contract survives runtime-companion checks."""
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.adjoint_result is not None
    original_wire = result.adjoint_result.to_dict()
    standalone = replace(
        result,
        captured_state=None,
        adjoint_result=replace(result.adjoint_result, captured_state=None),
    )
    assert standalone.captured_state is None
    assert program_adjoint_result(standalone).to_dict() == original_wire
    np.testing.assert_array_equal(program_adjoint_gradient(standalone), [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(standalone), [4.0])


@pytest.mark.parametrize("metadata", ["missing", "unsupported"])
def test_bound_result_cannot_replace_its_adjoint_admission(metadata: str) -> None:
    """Diagnostic classification does not relax construction of a captured tape.

    Parameters
    ----------
    metadata
        Removed or unsupported replacement for the bound adjoint metadata.

    """
    result = whole_program_value_and_grad(_square, [2.0], trace=False)
    assert result.adjoint_result is not None
    candidate = (
        None
        if metadata == "missing"
        else replace(result.adjoint_result, supported=False, unsupported_ops=("unsupported_op",))
    )
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured derivative tape.*adjoint metadata"):
        replace(result, adjoint_result=candidate)
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(program_adjoint_gradient(result), [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])
