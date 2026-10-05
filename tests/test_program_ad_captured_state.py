# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — captured program state tests
"""Exercise captured-state validity through public differentiation and replay."""

from __future__ import annotations

import gc
import sys
import weakref
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from types import BuiltinFunctionType, CodeType, FrameType, FunctionType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import TraceADArray, WholeProgramADResult, whole_program_value_and_grad
from scpn_quantum_control.dense_budget import GIB, DenseAllocationError
from scpn_quantum_control.differentiable import (
    program_adjoint_gradient,
    program_adjoint_replay_gradient,
    program_adjoint_result,
)
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
)

_GLOBAL_COEFFICIENT = 2.0


def _capture_lifetime_fixture() -> tuple[
    WholeProgramADResult,
    weakref.ReferenceType[Callable[[TraceADArray], object]],
    weakref.ReferenceType[NDArray[np.float64]],
]:
    """Return a real derivative result and weak observers of its dependencies."""
    coefficient = np.array([2.0], dtype=np.float64)

    def objective(values: TraceADArray) -> object:
        return values[0] * coefficient[0]

    return (
        whole_program_value_and_grad(objective, [3.0], trace=False),
        weakref.ref(objective),
        weakref.ref(coefficient),
    )


@pytest.mark.parametrize("copying", ["shallow", "deep", "replace"])
def test_result_lifetime_retains_and_releases_callable_capture(copying: str) -> None:
    """Copied results retain live dependencies until the last result is released.

    Parameters
    ----------
    copying
        Public shallow-copy, deep-copy or dataclass replacement operation.

    """
    baseline = active_reserved_bytes()
    result, objective_ref, coefficient_ref = _capture_lifetime_fixture()
    copied = (
        copy(result)
        if copying == "shallow"
        else deepcopy(result)
        if copying == "deep"
        else replace(result)
    )
    del result
    gc.collect()
    assert objective_ref() is not None
    coefficient = coefficient_ref()
    assert coefficient is not None
    np.testing.assert_array_equal(program_adjoint_replay_gradient(copied), [2.0])
    coefficient[0] = 5.0
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(copied)
    coefficient[0] = 2.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(copied), [2.0])
    del coefficient, copied
    gc.collect()
    assert objective_ref() is None
    assert coefficient_ref() is None
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("phase", ["capture", "replay", "during_replay"])
def test_observed_concurrent_capture_change_refuses_and_recapture_recovers(phase: str) -> None:
    """A real thread's changed coefficient cannot reuse an observed old binding.

    Parameters
    ----------
    phase
        Mutation at objective entry, before replay, or after replay's initial check.

    """
    coefficient = np.array([2.0], dtype=np.float64)
    inputs = np.array([3.0], dtype=np.float64)

    def objective(values: TraceADArray) -> object:
        return values[0] * coefficient[0]

    result = whole_program_value_and_grad(objective, inputs, trace=False)
    baseline = active_reserved_bytes()
    requested, changed = Event(), Event()

    def mutate() -> None:
        """Change the coefficient after the caller reaches the requested boundary."""
        if requested.wait(timeout=5.0):
            coefficient[0] = 5.0
            changed.set()

    worker = Thread(target=mutate)
    observed: list[bool] = []
    code = objective.__code__ if phase == "capture" else program_adjoint_replay_gradient.__code__

    def observe(frame: FrameType, event: str, argument: object) -> None:
        """Synchronize a real worker with the publicly invoked execution path."""
        selected = (
            frame.f_code.co_name == "_program_adjoint_execute_steps"
            and frame.f_code.co_filename == program_adjoint_replay_gradient.__code__.co_filename
            if phase == "during_replay"
            else frame.f_code is code
        )
        if event == "call" and selected and not observed:
            observed.append(True)
            requested.set()
            assert changed.wait(timeout=5.0)

    previous = sys.getprofile()
    refusal: ValueError | None = None
    worker.start()
    try:
        sys.setprofile(observe)
        if phase == "capture":
            whole_program_value_and_grad(objective, inputs, trace=False)
        else:
            program_adjoint_replay_gradient(result)
    except ValueError as error:
        refusal = error
    finally:
        sys.setprofile(previous)
        requested.set()
        worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert observed and changed.is_set()
    assert coefficient[0] == 5.0
    assert refusal is not None, "observed changed coefficient returned an old derivative"
    assert "captured program state changed" in str(refusal)
    np.testing.assert_array_equal(inputs, [3.0])
    assert active_reserved_bytes() == baseline
    current = whole_program_value_and_grad(objective, inputs, trace=False)
    assert current.value == 15.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(current), [5.0])
    coefficient[0] = 2.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("replacement", ["remove", "other_objective"])
def test_result_replacement_cannot_discard_its_live_state_binding(replacement: str) -> None:
    """Replacing a runtime result refuses removed or unrelated state bindings.

    Parameters
    ----------
    replacement
        Removal of the binding or substitution from another captured objective.

    """
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    def unrelated(values: TraceADArray) -> object:
        return values[0] * 5.0

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    other = whole_program_value_and_grad(unrelated, [3.0], trace=False)
    candidate = None if replacement == "remove" else other.captured_state
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured state binding"):
        replace(result, captured_state=candidate)
    assert active_reserved_bytes() == baseline
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    state[0] = 5.0
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(result)


def test_result_refuses_a_state_binding_of_another_type() -> None:
    """A result accepts the private captured state binding or none, nothing shaped like it."""

    def objective(values: TraceADArray) -> object:
        return values[0] * 2.0

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    baseline = active_reserved_bytes()
    with pytest.raises(
        ValueError, match="captured_state must be a captured program state binding or None"
    ):
        replace(result, captured_state=object())  # type: ignore[arg-type]
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])


@pytest.mark.parametrize("copying", ["shallow", "deep", "replace"])
def test_copied_result_keeps_live_binding_and_recovers_after_restoration(copying: str) -> None:
    """Ordinary copies preserve their live callable rather than cloning captures.

    Parameters
    ----------
    copying
        The standard shallow-copy, deep-copy or dataclass replacement operation.

    """
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    copied = (
        copy(result)
        if copying == "shallow"
        else deepcopy(result)
        if copying == "deep"
        else replace(result)
    )
    baseline = active_reserved_bytes()
    assert copied is not result
    assert copied.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(copied), [2.0])
    state[0] = 5.0
    for getter in (
        program_adjoint_result,
        program_adjoint_gradient,
        program_adjoint_replay_gradient,
    ):
        with pytest.raises(ValueError, match="captured program state changed"):
            getter(copied)
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(copied.gradient, [2.0])
    state[0] = 2.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(copied), [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])


@pytest.mark.parametrize("record", ["whole_program", "adjoint"])
@pytest.mark.parametrize("replacement", ["remove", "other_objective"])
@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
def test_corrupted_result_binding_refuses_derivative_access_and_recovers(
    record: str, replacement: str, access: str
) -> None:
    """Post-construction corruption cannot turn a bound tape into a raw record.

    Parameters
    ----------
    record
        Enclosing whole-program record or its attached adjoint metadata.
    replacement
        Binding removal or substitution from another objective.
    access
        Public metadata, gradient-copy or executable-replay entry point.

    """
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    def unrelated(values: TraceADArray) -> object:
        return values[0] * 5.0

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    other = whole_program_value_and_grad(unrelated, [3.0], trace=False)
    assert result.adjoint_result is not None
    assert result.program_ir is not None
    original_ir = result.program_ir.serialization
    original_adjoint = result.adjoint_result.to_dict()
    target = result if record == "whole_program" else result.adjoint_result
    original_binding = target.captured_state
    candidate = None if replacement == "remove" else other.captured_state
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    getter = getters[access]
    baseline = active_reserved_bytes()
    object.__setattr__(target, "captured_state", candidate)
    try:
        with pytest.raises(ValueError, match="captured state binding"):
            getter(result)
        assert active_reserved_bytes() == baseline
        assert result.program_ir.serialization == original_ir
        assert result.adjoint_result.to_dict() == original_adjoint
        np.testing.assert_array_equal(result.gradient, [2.0])
    finally:
        object.__setattr__(target, "captured_state", original_binding)
    assert program_adjoint_result(result).supported
    np.testing.assert_array_equal(program_adjoint_gradient(result), [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])


@pytest.mark.parametrize(
    "protocol",
    [
        "constructor",
        "doc_descriptor",
        "dict_descriptor",
        "annotation_storage",
        "first_line_storage",
        "static_attribute_storage",
        "static_attribute_item",
        "inherited_container",
        "foreign_dictionary_descriptor",
    ],
)
def test_local_container_protocols_refuse_before_object_creation(protocol: str) -> None:
    """Unregistered constructors and descriptors cannot masquerade as containers.

    Parameters
    ----------
    protocol
        Constructor, descriptor, inheritance or malformed passive metadata.

    """
    calls: list[str] = []

    def callback(instance: object) -> None:
        del instance
        calls.append("called")

    namespace: dict[str, object] = {"__annotations__": {"value": float}}
    if protocol == "constructor":
        namespace["__init__"] = callback
    elif protocol == "doc_descriptor":
        namespace["__doc__"] = property(callback)
    elif protocol == "dict_descriptor":
        namespace["__dict__"] = property(callback)
    elif protocol == "annotation_storage":
        namespace["__annotations__"] = []
    elif protocol == "first_line_storage":
        namespace["__firstlineno__"] = "1"
    elif protocol == "static_attribute_storage":
        namespace["__static_attributes__"] = ["value"]
    elif protocol == "static_attribute_item":
        namespace["__static_attributes__"] = ("value", 1)
    elif protocol == "foreign_dictionary_descriptor":
        Other = type("Other", (object,), {})
        namespace["__dict__"] = vars(Other)["__dict__"]
    bases = (list,) if protocol == "inherited_container" else (object,)
    Scratch = type("Scratch", bases, namespace)

    def objective(values: TraceADArray) -> object:
        scratch = Scratch()
        return values[0] * (2.0 if scratch is not None else 5.0)

    with pytest.raises(ValueError, match="external callback"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []


@pytest.mark.parametrize("metadata", ["first_line", "static_attributes", "both"])
def test_passive_class_metadata_preserves_local_numeric_storage(metadata: str) -> None:
    """Valid native class metadata keeps local derivative-carrying attributes.

    Parameters
    ----------
    metadata
        First-line metadata, static attribute names or both supported records.

    """

    class Scratch:
        value: float

    if metadata in {"first_line", "both"}:
        type.__setattr__(Scratch, "__firstlineno__", 1)
    if metadata in {"static_attributes", "both"}:
        type.__setattr__(Scratch, "__static_attributes__", ("value",))

    def objective(values: TraceADArray) -> object:
        scratch = Scratch()
        scratch.value = cast(float, values[0])
        return scratch.value * 2.0

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("mutation", ["annotations", "constructor"])
def test_local_container_class_rebinding_invalidates_its_captured_derivative(
    mutation: str,
) -> None:
    """Passive class metadata remains bound without executing changed protocols.

    Parameters
    ----------
    mutation
        Rebinding of the class annotation or installation of a constructor hook.

    """
    calls: list[str] = []

    class Scratch:
        value: float

    def callback(instance: object) -> None:
        del instance
        calls.append("called")

    def objective(values: TraceADArray) -> object:
        scratch = Scratch()
        scratch.value = cast(float, values[0])
        return scratch.value * 2.0

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    if mutation == "annotations":
        Scratch.__annotations__["value"] = int
    else:
        type.__setattr__(Scratch, "__init__", callback)
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(result)
    assert calls == []
    np.testing.assert_array_equal(result.gradient, [2.0])


def test_local_instance_cannot_mutate_its_captured_class_annotations() -> None:
    """Class storage reached through a local instance retains captured ownership."""

    class Scratch:
        value: float

    previous = Scratch.__annotations__.copy()

    def objective(values: TraceADArray) -> object:
        scratch = Scratch()
        scratch.__annotations__["value"] = int
        return values[0] * 2.0

    with pytest.raises(ValueError, match="captured mutation"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert Scratch.__annotations__ == previous


@pytest.mark.parametrize("view", ["flip", "squeeze", "atleast_3d"])
def test_numeric_capture_view_retains_ownership_before_indexed_write(view: str) -> None:
    """Known NumPy views cannot hide captured storage behind a local assignment.

    Parameters
    ----------
    view
        Anchored NumPy view operation applied to captured numeric storage.

    """
    state = np.full((1, 1, 1), 2.0)
    operations: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
        "flip": np.flip,
        "squeeze": np.squeeze,
        "atleast_3d": np.atleast_3d,
    }
    operation = operations[view]

    def objective(values: TraceADArray) -> object:
        alias = operation(state)
        alias.flat[0] = 5.0
        return values[0] * state.flat[0]

    with pytest.raises(ValueError, match="captured mutation"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, np.full((1, 1, 1), 2.0))


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("storage", ["list", "dict", "array", "strided-array", "nested"])
def test_coefficient_storage_change_invalidates_public_derivative_access(
    trace: bool, storage: str
) -> None:
    """Mutable numeric captures preserve their tape and refuse stale derivatives.

    Parameters
    ----------
    trace
        Whether source tracing adds a second execution.
    storage
        Exact numeric coefficient storage used by a source-visible helper.

    """
    coefficients = [2.0]
    mapping = {"coefficient": 2.0}
    array = np.array([2.0, 99.0, 2.0])[::2] if storage == "strided-array" else np.array([2.0])
    nested = {"weights": [2.0]}

    def coefficient() -> float:
        if storage == "list":
            return coefficients[0]
        if storage == "dict":
            return mapping["coefficient"]
        if storage in {"array", "strided-array"}:
            return float(array[0])
        return nested["weights"][0]

    def objective(values: TraceADArray) -> object:
        return values[0] * coefficient()

    result = whole_program_value_and_grad(objective, [3.0], trace=trace)
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert result.program_ir is not None
    serialization = result.program_ir.serialization
    if storage == "list":
        coefficients[0] = 5.0
    elif storage == "dict":
        mapping["coefficient"] = 5.0
    elif storage in {"array", "strided-array"}:
        array[0] = 5.0
    else:
        nested["weights"][0] = 5.0
    for access in (
        program_adjoint_result,
        program_adjoint_gradient,
        program_adjoint_replay_gradient,
    ):
        with pytest.raises(ValueError, match="captured program state changed"):
            access(result)
    assert result.program_ir.serialization == serialization
    np.testing.assert_array_equal(result.gradient, [2.0])
    refreshed = whole_program_value_and_grad(objective, [3.0], trace=trace)
    assert refreshed.value == 15.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(refreshed), [5.0])


@pytest.mark.parametrize("keyword", [False, True])
def test_default_rebinding_invalidates_replay(keyword: bool) -> None:
    """Changing positional or keyword defaults invalidates an existing tape.

    Parameters
    ----------
    keyword
        Whether the coefficient is a keyword-only default.

    """

    def positional(values: TraceADArray, coefficient: float = 2.0) -> object:
        return values[0] * coefficient

    def named(values: TraceADArray, *, coefficient: float = 2.0) -> object:
        return values[0] * coefficient

    objective = named if keyword else positional
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    if keyword:
        cast(FunctionType, objective).__kwdefaults__ = {"coefficient": 5.0}
    else:
        cast(FunctionType, objective).__defaults__ = (5.0,)
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(result)
    np.testing.assert_array_equal(result.gradient, [2.0])


def test_global_rebinding_invalidates_replay_and_recapture_recovers() -> None:
    """A global coefficient is checked at replay and restored after the test."""
    global _GLOBAL_COEFFICIENT

    def objective(values: TraceADArray) -> object:
        return values[0] * _GLOBAL_COEFFICIENT

    previous = _GLOBAL_COEFFICIENT
    try:
        _GLOBAL_COEFFICIENT = 2.0
        result = whole_program_value_and_grad(objective, [3.0], trace=False)
        _GLOBAL_COEFFICIENT = 5.0
        with pytest.raises(ValueError, match="captured program state changed"):
            program_adjoint_replay_gradient(result)
        fresh = whole_program_value_and_grad(objective, [3.0], trace=False)
        np.testing.assert_array_equal(program_adjoint_replay_gradient(fresh), [5.0])
    finally:
        _GLOBAL_COEFFICIENT = previous


@pytest.mark.parametrize("helper", ["function", "lambda", "nested_function"])
def test_unregistered_nested_callable_refuses_before_objective_execution(helper: str) -> None:
    """Local callback definitions retain their located pre-execution refusal.

    Parameters
    ----------
    helper
        Source-visible local callable whose body reads the global coefficient.

    """

    def local_function(values: TraceADArray) -> object:
        def coefficient() -> float:
            return _GLOBAL_COEFFICIENT

        return values[0] * coefficient()

    def local_lambda(values: TraceADArray) -> object:
        return values[0] * (lambda: _GLOBAL_COEFFICIENT)()

    def nested_function(values: TraceADArray) -> object:
        def coefficient() -> float:
            def inner() -> float:
                return _GLOBAL_COEFFICIENT

            return inner()

        return values[0] * coefficient()

    objectives = {
        "function": local_function,
        "lambda": local_lambda,
        "nested_function": nested_function,
    }
    objective = objectives[helper]
    baseline = active_reserved_bytes()
    calls = 0

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal calls
        del argument
        if event == "call" and frame.f_code is cast(FunctionType, objective).__code__:
            calls += 1

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="external_callback.*line=.*absolute_line="):
            whole_program_value_and_grad(objective, [3.0], trace=False)
    finally:
        sys.setprofile(previous_profile)
    assert calls == 0
    assert active_reserved_bytes() == baseline


def test_unregistered_nested_numeric_callback_retains_located_refusal() -> None:
    """A numeric local callback cannot silently acquire a derivative contract."""

    def objective(values: TraceADArray) -> object:
        def local_sine() -> object:
            return np.sin(values[0])

        return local_sine()

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="external_callback.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [0.25], trace=False)
    assert active_reserved_bytes() == baseline


def test_objective_code_replacement_invalidates_replay() -> None:
    """Replacing callable code cannot make its old derivative current."""

    def objective(values: TraceADArray) -> object:
        return values[0] * 2.0

    def replacement(values: TraceADArray) -> object:
        return values[0] * 5.0

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    cast(FunctionType, objective).__code__ = cast(FunctionType, replacement).__code__
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(result)


def test_captured_class_coefficient_read_retains_located_refusal() -> None:
    """Unsupported class-attribute reads retain the object-attribute boundary."""

    class Coefficients:
        value = 2.0

    def objective(values: TraceADArray) -> object:
        return values[0] * Coefficients.value

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="object_attribute.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert Coefficients.value == 2.0
    assert active_reserved_bytes() == baseline


def test_captured_class_descriptor_refuses_without_implicit_callback() -> None:
    """Reading an unregistered descriptor refuses before its callback executes."""
    calls: list[str] = []

    class CoefficientDescriptor:
        def __get__(self, instance: object, owner: object) -> float:
            calls.append("get")
            return 2.0

    class Coefficients:
        value = CoefficientDescriptor()

    def objective(values: TraceADArray) -> object:
        return values[0] * Coefficients.value

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="object_attribute.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


def test_cyclic_numeric_capture_has_a_bounded_stable_binding() -> None:
    """A self-referential list is checked without recursive expansion."""
    state: list[object] = [2.0]
    state.append(state)

    def objective(values: TraceADArray) -> object:
        return values[0] * cast(float, state[0])

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    state[0] = 5.0
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(result)


def test_opaque_capture_refuses_without_using_its_object_protocols() -> None:
    """State inspection never calls opaque representation or iteration methods."""
    calls: list[str] = []

    class Opaque:
        def __repr__(self) -> str:
            calls.append("repr")
            raise AssertionError("opaque representation was called")

        def __iter__(self) -> object:
            calls.append("iter")
            raise AssertionError("opaque iteration was called")

    opaque = Opaque()

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if opaque is not None else 5.0)

    with pytest.raises(ValueError, match="captured program state contains unsupported storage"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []


def test_numeric_module_callable_rebinding_invalidates_replay() -> None:
    """Replacing a referenced NumPy intrinsic is visible to the live binding."""

    def objective(values: TraceADArray) -> object:
        return np.sum(values)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    namespace = cast(dict[str, object], vars(np))
    previous = namespace["sum"]
    try:
        namespace["sum"] = np.prod
        with pytest.raises(ValueError, match="captured program state changed"):
            program_adjoint_replay_gradient(result)
    finally:
        namespace["sum"] = previous


def test_helper_module_dependencies_are_checked_in_each_function_scope() -> None:
    """Distinct helper call sites retain every referenced module intrinsic."""

    def helper(values: TraceADArray) -> object:
        return np.sin(values[0])

    def objective(values: TraceADArray) -> object:
        return np.cos(values[0]) + helper(values)

    result = whole_program_value_and_grad(objective, [0.25], trace=False)
    expected = float(np.cos(0.25) - np.sin(0.25))
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result), [expected], rtol=0.0, atol=1e-12
    )
    namespace = cast(dict[str, object], vars(np))
    previous = namespace["sin"]
    try:
        namespace["sin"] = np.cos
        with pytest.raises(ValueError, match="captured program state changed"):
            program_adjoint_replay_gradient(result)
    finally:
        namespace["sin"] = previous


def test_opaque_capture_does_not_invoke_metaclass_hash_or_equality() -> None:
    """Exact storage admission uses identity comparisons for untrusted types."""
    calls: list[str] = []

    class ProtocolMeta(type):
        def __hash__(cls) -> int:
            calls.append("hash")
            raise AssertionError("opaque metaclass hash was called")

        def __eq__(cls, other: object) -> bool:
            calls.append("equality")
            raise AssertionError("opaque metaclass equality was called")

    class Opaque(metaclass=ProtocolMeta):
        def __getattribute__(self, name: str) -> object:
            calls.append(name)
            raise AssertionError("opaque attribute protocol was called")

    opaque = Opaque()

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if opaque is not None else 5.0)

    with pytest.raises(ValueError, match="captured program state contains unsupported storage"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        (None, True),
        (True, False),
        (-123456789012345678901234567890, 7),
        (2.0, 5.0),
        (2.0 + 1.0j, 5.0 + 1.0j),
        ("mode\udc00", "other-mode"),
        (b"mode\x00", b"other-mode"),
        (np.bool_(True), np.bool_(False)),
        (np.float32(2.0), np.float32(5.0)),
        (np.complex128(2.0 + 1.0j), np.complex128(5.0 + 1.0j)),
    ],
)
def test_static_branch_metadata_change_invalidates_replay(
    original: object, changed: object
) -> None:
    """Numeric and identifier captures retain the selected branch's provenance.

    Parameters
    ----------
    original
        Exact scalar or identifier selecting the first branch.
    changed
        Different scalar or identifier selecting the second branch.

    """
    state = [original]

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state[0] == original else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    state[0] = changed
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_replay_gradient(result)
    fresh = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert fresh.value == 15.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(fresh), [5.0])


@pytest.mark.parametrize("access", ["copy", "replay"])
@pytest.mark.parametrize("policy", ["cancelled", "expired"])
def test_adjoint_lifecycle_is_checked_before_stale_capture(access: str, policy: str) -> None:
    """Stopped adjoint access refuses before inspecting a stale captured value.

    Parameters
    ----------
    access
        Whether the attached gradient is copied or replayed.
    policy
        Pre-existing cancellation or elapsed monotonic deadline.

    """
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    state[0] = 5.0
    getter = program_adjoint_gradient if access == "copy" else program_adjoint_replay_gradient
    baseline = active_reserved_bytes()
    if policy == "cancelled":
        cancelled = Event()
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            getter(result, cancelled=cancelled)
    else:
        with pytest.raises(TimeoutError, match="deadline"):
            getter(result, deadline_monotonic=monotonic() - 1.0)
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(result.gradient, [2.0])
    state[0] = 2.0
    np.testing.assert_array_equal(getter(result), [2.0])


def _large_capture(storage: str) -> object:
    if storage == "array":
        return np.full(512 * 1024, 2.0)
    if storage == "string":
        return "𐀀" * (512 * 1024)
    return 1 << (2 * 1024**2 * 8 - 1)


@pytest.mark.parametrize("storage", ["array", "string", "integer"])
def test_forward_capture_copy_obeys_execution_memory_cap(storage: str) -> None:
    """Fingerprint byte copies are admitted before numerical execution.

    Parameters
    ----------
    storage
        Numeric array, UTF-8 string or integer requiring a large snapshot copy.

    """
    state = _large_capture(storage)

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(objective, [3.0], trace=False, max_execution_gib=1.0 / 1024)
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        objective, [3.0], trace=False, max_execution_gib=8.0 / 1024
    )
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "text",
    ["x" * (512 * 1024), "aé𐀀\udc00" * (64 * 1024)],
    ids=["ascii", "mixed_unicode"],
)
def test_utf8_capture_uses_encoded_byte_size_under_sufficient_cap(text: str) -> None:
    """String admission retains a sufficient budget for the exact UTF-8 payload.

    The cap of 2.25 MiB lies between two measured totals for the mixed-width
    text. Charging the exact 655,360 encoded bytes needs about 2.10 MB in all
    (2,099,958 bytes under Python 3.11, slightly less under 3.12; the total
    also grows with the size of this module, whose source is charged).
    Charging four bytes for each of the 262,144 characters would need 393,216
    bytes more, about 2.49 MB, and exceed the cap. A cap of exactly 2 MiB left
    a few kilobytes of margin and failed under Python 3.11.

    Parameters
    ----------
    text
        Large ASCII or mixed-width Unicode including an unpaired surrogate.

    """
    assert len(text.encode("utf-8", errors="surrogatepass")) < 1024**2

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if text is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(
        objective, [3.0], trace=False, max_execution_gib=2.25 / 1024
    )
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("character", ["a", "é", "\u0800", "𐀀"])
def test_capture_string_bound_counts_utf8_bytes_instead_of_worst_case_width(
    character: str,
) -> None:
    """The snapshot accepts each encoding width within the actual byte limit.

    Parameters
    ----------
    character
        A Unicode code point encoded in one, two, three or four UTF-8 bytes.

    """
    text = character * (2 * 1024**2 + 1)
    encoded_size = len(text.encode("utf-8"))
    baseline = active_reserved_bytes()

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if text is not None else 5.0)

    if encoded_size > 8 * 1024**2:
        with pytest.raises(
            ValueError, match="captured program state contains unsupported storage"
        ):
            whole_program_value_and_grad(objective, [3.0], trace=False)
    else:
        result = whole_program_value_and_grad(objective, [3.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(program_adjoint_result(result).gradient, [2.0])
        np.testing.assert_array_equal(program_adjoint_gradient(result), [2.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "storage",
    [
        "list",
        "tuple",
        "dict",
        "depth",
        "nodes",
        "integer",
        "integer_payload",
        "string",
        "bytes",
        "array",
    ],
)
def test_snapshot_limits_refuse_with_no_reservation_leak_and_recapture_recovers(
    storage: str,
) -> None:
    """Oversized captures retain a bounded refusal and permit a smaller recapture.

    Parameters
    ----------
    storage
        Container count, aggregate depth, visited node count or byte payload limit.

    """
    state: object
    if storage == "list":
        state = [2.0] * 4097
    elif storage == "tuple":
        state = (2.0,) * 4097
    elif storage == "dict":
        state = {str(index): 2.0 for index in range(4097)}
    elif storage == "depth":
        state = 2.0
        for _ in range(65):
            state = [state]
    elif storage == "nodes":
        state = [[2.0] for _ in range(2048)]
    elif storage == "integer":
        state = 1 << (8 * 1024**2 * 8)
    elif storage == "integer_payload":
        state = 1 << (8 * 1024**2 * 8 - 1)
    elif storage == "string":
        state = "a" * (8 * 1024**2 + 1)
    elif storage == "bytes":
        state = b"a" * (8 * 1024**2 + 1)
    else:
        state = np.zeros(1024**2 + 1, dtype=np.float64)

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured program state contains unsupported storage"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert active_reserved_bytes() == baseline
    state = [2.0]
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("access", ["copy", "replay"])
@pytest.mark.parametrize("storage", ["array", "string", "integer"])
def test_adjoint_capture_copy_obeys_requested_memory_cap(access: str, storage: str) -> None:
    """Adjoint policy includes revalidation copies of the bound captured state.

    Parameters
    ----------
    access
        Whether the attached gradient is copied or replayed.
    storage
        Large supported capture whose fingerprint must be checked again.

    """
    state = _large_capture(storage)

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    getter = program_adjoint_gradient if access == "copy" else program_adjoint_replay_gradient
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        getter(result, max_execution_gib=128 * 1024 / GIB)
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(getter(result, max_execution_gib=8.0 / 1024), [2.0])
    assert active_reserved_bytes() == baseline


def test_direct_adjoint_result_inherits_parent_capture_budget() -> None:
    """The metadata getter cannot copy captured bytes outside an active owner."""
    state = np.full(512 * 1024, 2.0)

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    baseline = active_reserved_bytes()
    with reserve_execution_memory(plan, max_gib=1.0 / 1024):
        with pytest.raises(DenseAllocationError):
            program_adjoint_result(result)
        assert active_reserved_bytes() == baseline + 16
    assert active_reserved_bytes() == baseline
    assert program_adjoint_result(result).supported


def test_direct_adjoint_result_inherits_parent_cancellation() -> None:
    """Metadata revalidation observes an already cancelled enclosing scope."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * state[0]

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    cancelled = Event()
    baseline = active_reserved_bytes()
    with reserve_execution_memory(plan, cancelled=cancelled):
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            program_adjoint_result(result)
        assert active_reserved_bytes() == baseline + 16
        cancelled.clear()
    assert active_reserved_bytes() == baseline
    assert program_adjoint_result(result).supported


@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
@pytest.mark.parametrize("storage", ["array", "string", "integer"])
def test_capture_copy_charge_and_cancellation_are_observed_during_access(
    access: str, storage: str
) -> None:
    """Actual snapshot copies own their charge and observe cancellation on return.

    Parameters
    ----------
    access
        Metadata retrieval, attached gradient copying or executable replay.
    storage
        Numeric array, UTF-8 string or integer requiring a large byte copy.

    """
    state = _large_capture(storage)
    payload_size = (
        cast(NDArray[np.float64], state).nbytes
        if storage == "array"
        else len(cast(str, state).encode("utf-8"))
        if storage == "string"
        else (cast(int, state).bit_length() + 8) // 8
    )
    method = "tobytes" if storage == "array" else "encode" if storage == "string" else "to_bytes"

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    getter = getters[access]
    cancelled = Event()
    observed = False
    peak = 0

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal observed, peak
        del frame
        if (
            event == "c_return"
            and isinstance(argument, BuiltinFunctionType)
            and argument.__self__ is state
            and argument.__name__ == method
        ):
            observed = True
            peak = active_reserved_bytes()
            cancelled.set()

    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    previous = sys.getprofile()
    with reserve_execution_memory(plan, cancelled=cancelled):
        sys.setprofile(profile)
        try:
            with pytest.raises(ExecutionCancelledError):
                getter(result)
        finally:
            sys.setprofile(previous)
        assert observed
        assert peak >= baseline + payload_size
        assert active_reserved_bytes() == baseline + 16
        cancelled.clear()
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(result.gradient, [2.0])
    assert program_adjoint_result(result).supported


def test_numpy_python_intrinsic_code_identity_refuses_changes_and_recovers() -> None:
    """A referenced genuine NumPy Python callable retains its code identity."""

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if np.isscalar is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    intrinsic = cast(FunctionType, np.isscalar)
    original = intrinsic.__code__
    try:
        intrinsic.__code__ = original.replace(co_filename="changed-numpy-intrinsic.py")
        with pytest.raises(ValueError, match="captured program state changed"):
            program_adjoint_result(result)
    finally:
        intrinsic.__code__ = original
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_empty_captured_cell_refuses_derivative_access_and_restoration_recovers() -> None:
    """An emptied closure cannot silently remove its captured-state binding."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    closure = cast(FunctionType, objective).__closure__
    assert closure is not None and len(closure) == 1
    cell = closure[0]
    saved = cell.cell_contents
    del cell.cell_contents
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_result(result)
    cell.cell_contents = saved
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("metadata", ["bytecode", "names"])
def test_oversized_callable_metadata_refuses_access_and_recapture_recovers(metadata: str) -> None:
    """Callable storage limits hold independently of ordinary numeric captures.

    Parameters
    ----------
    metadata
        Oversized compiled bytecode or an excessive global-name table.

    """

    def dependency() -> float:
        return 2.0

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    original = dependency.__code__
    if metadata == "bytecode":
        source = "def oversized():\n" + "    coefficient\n" * 40000 + "    return 2.0\n"
        compiled = compile(source, __file__, "exec")
        code = next(value for value in compiled.co_consts if type(value) is CodeType)
        dependency.__code__ = code
    else:
        dependency.__code__ = original.replace(co_names=tuple(f"name_{i}" for i in range(4097)))
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_result(result)
    assert active_reserved_bytes() == baseline
    dependency.__code__ = original
    recovered = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert recovered.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(recovered), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("namespace", ["global_protocol", "global_key", "global_size", "builtins"])
def test_callable_namespace_protocols_refuse_without_implicit_lookup(namespace: str) -> None:
    """Captured callable namespaces admit plain bounded string-key storage.

    Parameters
    ----------
    namespace
        Invalid globals protocol, key, size or builtins container.

    """
    calls: list[str] = []

    class Namespace(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            """Record a forbidden caller dictionary protocol."""
            calls.append(key)
            raise AssertionError("unexpected namespace protocol")

    def original_dependency() -> float:
        return 2.0

    mapping: dict[str, object] = {"__builtins__": {}}
    if namespace == "global_protocol":
        mapping = Namespace(mapping)
    elif namespace == "global_key":
        cast(dict[object, object], mapping)[1] = 2.0
    elif namespace == "global_size":
        mapping.update({f"key_{i}": 2.0 for i in range(4097)})
    else:
        mapping["__builtins__"] = []
    dependency = FunctionType(original_dependency.__code__, mapping)

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="unsupported storage"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


def test_absent_dependency_global_retains_binding_when_the_namespace_changes() -> None:
    """An unexecuted dependency's absent global stays observable on rebinding."""
    source = "def dependency():\n    return coefficient\n"
    compiled = compile(source, __file__, "exec")
    code = next(value for value in compiled.co_consts if type(value) is CodeType)
    mapping: dict[str, object] = {"__builtins__": {}}
    dependency = FunctionType(code, mapping)

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    mapping["coefficient"] = 5.0
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_result(result)
    del mapping["coefficient"]
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline
