# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — objective effect admission tests
"""Qualify no-execution effect findings through their public compiler contract."""

from __future__ import annotations

import ast
import builtins
import inspect
import math
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType, FunctionType, ModuleType
from typing import Any, TypedDict, Unpack, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import TraceADArray, TraceADScalar, whole_program_value_and_grad
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient, vmap
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects

_UNTYPED_VMAP = cast(Callable[..., Any], vmap)


class _NumpyOutput(TypedDict):
    """One native output buffer forwarded through keyword-call syntax."""

    out: NDArray[np.float64]


def _find(objective: Callable[..., object]) -> tuple[str, ...]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    findings = find_objective_effects(objective, tree)
    assert all(getattr(finding.node, "lineno", 0) > 0 for finding in findings)
    return tuple(finding.semantic for finding in findings)


def _details(objective: Callable[..., object]) -> tuple[str, ...]:
    """Return the detail of every finding, in source order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    return tuple(finding.detail for finding in find_objective_effects(objective, tree))


def test_supported_numpy_stencil_keeps_its_independent_derivative() -> None:
    """Anchored ``np.gradient`` admission retains the existing stencil runtime."""
    weights = np.array([1.0, 2.0, 3.0])

    def objective(values: TraceADArray) -> object:
        return np.sum(np.gradient(values, edge_order=2) * weights)

    result = whole_program_value_and_grad(objective, [1.0, 2.0, 4.0], trace=False)
    assert result.value == 11.0
    np.testing.assert_array_equal(result.gradient, [-1.0, -4.0, 5.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [-1.0, -4.0, 5.0])


def test_numeric_class_named_numpy_alias_retains_object_attribute_refusal() -> None:
    """A conventional module alias cannot admit unsupported class storage."""

    class Coefficients:
        value = 2.0

    np = Coefficients

    def objective(values: TraceADArray) -> object:
        return values[0] * np.value

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="object_attribute.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert Coefficients.value == 2.0
    assert active_reserved_bytes() == baseline


def test_numpy_alias_descriptor_refuses_before_implicit_callback() -> None:
    """A nonmodule alias cannot run a descriptor during numeric capture."""
    calls: list[str] = []

    class CoefficientDescriptor:
        def __get__(self, instance: object, owner: object) -> float:
            calls.append("get")
            return 2.0

    class Coefficients:
        value = CoefficientDescriptor()

    np = Coefficients

    def objective(values: TraceADArray) -> object:
        return values[0] * np.value

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="object_attribute.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("constructor", ["float64", "dtype"])
def test_real_numpy_class_metadata_keeps_supported_derivative(constructor: str) -> None:
    """Actual anchored NumPy types preserve their immutable metadata reads.

    Parameters
    ----------
    constructor
        Real NumPy class referenced by the source-visible objective.

    """

    def float_name(values: TraceADArray) -> object:
        return values[0] * (2.0 if np.float64.__name__ == "float64" else 5.0)

    def dtype_name(values: TraceADArray) -> object:
        return values[0] * (2.0 if np.dtype.__name__ == "dtype" else 5.0)

    objective = float_name if constructor == "float64" else dtype_name
    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_stencil_name_rebinding_cannot_admit_a_callback_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named NumPy call retains its actual identity and refuses captured writes.

    Parameters
    ----------
    monkeypatch
        Test-local restoration of the real NumPy module's gradient attribute.

    """
    ledger: list[str] = []

    def callback(values: TraceADArray) -> object:
        ledger.append("called")
        return values

    def objective(values: TraceADArray) -> object:
        return np.sum(np.gradient(values))

    monkeypatch.setattr(np, "gradient", callback)
    with pytest.raises(ValueError, match="external_callback"):
        whole_program_value_and_grad(objective, [1.0, 2.0, 4.0], trace=False)
    assert ledger == []


def test_callback_inspection_leaves_external_ledger_untouched() -> None:
    """A source-visible callback containing a captured write refuses at its caller."""
    ledger: list[str] = []

    def callback() -> float:
        ledger.append("called")
        return 7.0

    def objective(values: TraceADArray) -> object:
        return values[0] ** 2 + callback()

    assert "external_callback" in _find(objective)
    assert ledger == []


def test_captured_index_mutation_and_alias_are_detected() -> None:
    """Alias resolution identifies a captured write without modifying its storage."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        alias = state
        alias[0] = 5.0
        return values[0] * state[0]

    assert "captured_mutation" in _find(objective)
    assert state == [2.0]


def test_helper_parameter_retains_captured_storage_provenance() -> None:
    """Passing a captured array to a helper does not turn its write into local state."""
    state = np.array([2.0])

    def write(storage: object) -> None:
        cast(list[float], storage)[0] = 5.0

    def objective(values: TraceADArray) -> object:
        write(state)
        return values[0] * state[0]

    assert "captured_mutation" in _find(objective)
    np.testing.assert_array_equal(state, [2.0])


def test_ambient_numpy_rng_has_no_state_advance() -> None:
    """Identity-resolved NumPy randomness refuses before a draw is made."""

    def objective(values: TraceADArray) -> object:
        return values[0] + np.random.random()

    before = np.random.get_state(legacy=True)
    assert "ambient_rng" in _find(objective)
    after = np.random.get_state(legacy=True)
    assert isinstance(before, tuple) and isinstance(after, tuple)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_active_integer_conversion_is_located_statically() -> None:
    """A derivative-carrying scalar cannot become unrecorded integer shape metadata."""

    def objective(values: TraceADArray) -> object:
        count = int(cast(float, values[0]))
        return np.sum(np.zeros(count)) + values[0]

    assert "dynamic_integer" in _find(objective)


def test_pure_helper_closure_and_static_metadata_remain_eligible() -> None:
    """Recursive source inspection preserves a numeric helper and static range."""
    coefficient = [2.0]

    def helper(value: TraceADScalar | TraceADArray) -> object:
        return value * coefficient[0]

    def objective(values: TraceADArray) -> object:
        total = helper(values[0])
        for index in range(1, 3):
            total = total + index * values[0]
        return total

    assert _find(objective) == ()
    assert coefficient == [2.0]
    result = whole_program_value_and_grad(objective, [2.0, 3.0])
    assert result.value == 10.0
    np.testing.assert_array_equal(result.gradient, [5.0, 0.0])


def test_local_list_mutation_and_duplicate_scatter_remain_eligible() -> None:
    """Owned copies and a local list retain the existing trace mutation path."""

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        np.add.at(working, [0, 0], values)
        items = [values[0]]
        alias = items
        alias.append(values[1])
        return working.sum() + sum(items)

    assert _find(objective) == ()


def test_numpy_spelling_does_not_admit_an_unregistered_callable() -> None:
    """Module attribute inspection uses identity rather than an allowed root name."""
    calls: list[str] = []
    substitute = ModuleType("numpy")

    def callback(value: object) -> object:
        calls.append("called")
        return value

    vars(substitute)["sin"] = callback
    np_alias = substitute

    def objective(values: TraceADArray) -> object:
        return np_alias.sin(values[0])

    assert "external_callback" in _find(objective)
    assert calls == []


def test_opaque_attribute_callback_is_not_resolved_by_protocol() -> None:
    """Static inspection never invokes a callable object's attribute hooks."""
    ledger: list[str] = []

    class Opaque:
        def __getattribute__(self, name: str) -> object:
            ledger.append(name)
            raise AssertionError("attribute protocol ran")

    opaque = Opaque()

    def objective(values: TraceADArray) -> object:
        return cast(Callable[[object], object], opaque.callback)(values[0])

    assert "external_callback" in _find(objective)
    assert ledger == []


def test_repeat_inspection_has_identical_source_findings() -> None:
    """Preliminary and bytecode-binding passes can reuse deterministic findings."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        state[0] = 4.0
        return values[0]

    assert _find(objective) == _find(objective) == ("captured_mutation",)
    assert state == [2.0]


def test_branch_alias_merge_cannot_hide_captured_storage() -> None:
    """Both branch destinations participate in a subsequent mutation decision."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        if values[0] > 0:
            alias = state
        else:
            alias = [3.0]
        alias[0] = 5.0
        return values[0]

    assert "captured_mutation" in _find(objective)
    assert state == [2.0]


def test_helper_returning_capture_cannot_hide_mutation_origin() -> None:
    """A helper result alias retains the origin of its captured container."""
    state = [2.0]

    def helper() -> list[float]:
        return state

    def objective(values: TraceADArray) -> object:
        alias = helper()
        alias[0] = 5.0
        return values[0]

    assert "captured_mutation" in _find(objective)
    assert state == [2.0]


def test_inplace_alias_assignment_cannot_mutate_capture() -> None:
    """List in-place augmentation mutates its captured origin despite local binding."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        alias = state
        alias += [5.0]
        return values[0]

    assert "captured_mutation" in _find(objective)
    assert state == [2.0]


def test_parameter_loop_item_remains_active_for_integer_conversion() -> None:
    """Iteration carries parameter dependence into the integer boundary."""

    def objective(values: TraceADArray) -> object:
        for value in values:
            count = int(cast(float, value))
            return values[0] + count
        return values[0]

    assert "dynamic_integer" in _find(objective)


def test_captured_reshape_view_keeps_external_write_origin() -> None:
    """A reshape view aliases captured storage while copy creates owned storage."""
    state = np.array([2.0])

    def objective(values: TraceADArray) -> object:
        alias = state.reshape(1)
        alias[0] = 5.0
        return values[0]

    assert "captured_mutation" in _find(objective)
    np.testing.assert_array_equal(state, [2.0])


def test_active_augmented_scalar_cannot_become_shape_metadata() -> None:
    """Parameter dependence survives local accumulator augmentation."""

    def objective(values: TraceADArray) -> object:
        total = 0.0
        total += cast(float, values[0])
        count = int(total)
        return values[0] + count

    assert "dynamic_integer" in _find(objective)


def test_parameter_comprehension_cannot_hide_integer_conversion() -> None:
    """Comprehension targets retain active parameter provenance."""

    def objective(values: TraceADArray) -> object:
        counts = [int(cast(float, value)) for value in values]
        return values[0] + sum(counts)

    assert "dynamic_integer" in _find(objective)


def test_loop_carried_capture_is_considered_before_second_write() -> None:
    """A later alias assignment propagates to the next iteration's write."""
    state = [2.0]

    def objective(values: TraceADArray) -> object:
        alias = [0.0]
        for _ in range(2):
            alias[0] = 5.0
            alias = state
        return values[0]

    assert "captured_mutation" in _find(objective)
    assert state == [2.0]


def test_active_comprehension_result_keeps_shape_dependence() -> None:
    """Aggregate results retain the active element's integer boundary."""

    def objective(values: TraceADArray) -> object:
        entries = [value for value in values]
        count = int(cast(float, entries[0]))
        return values[0] + count

    assert "dynamic_integer" in _find(objective)


def test_builtin_namespace_rebinding_does_not_admit_external_callback() -> None:
    """The callable's real builtins namespace owns an intrinsic's identity."""
    calls: list[str] = []

    def callback(value: object) -> object:
        calls.append("called")
        return value

    def objective(values: TraceADArray) -> object:
        return abs(cast(TraceADScalar, values[0]))

    namespace = dict(cast(FunctionType, objective).__globals__)
    namespace["__builtins__"] = {"abs": callback}
    rebound = FunctionType(cast(FunctionType, objective).__code__, namespace)
    assert "external_callback" in _find(rebound)
    with pytest.raises(ValueError, match="external.*callback") as refusal:
        whole_program_value_and_grad(rebound, [2.0])
    assert "line=" in str(refusal.value)
    assert calls == []


def test_direct_async_objective_retains_existing_frontend_category() -> None:
    """Root asynchronous syntax keeps its original located unsupported category."""

    async def objective(values: TraceADArray) -> object:
        return values[0]

    assert _find(objective) == ("async_function",)
    from scpn_quantum_control.differentiable import compile_whole_program_frontend

    report = compile_whole_program_frontend(objective)
    assert not report.frontend_ready
    assert report.semantics_report.unsupported_python_semantics == ("async_function",)


@pytest.mark.parametrize(
    ("ufunc", "point", "expected"),
    [
        (np.reciprocal, 2.0, -0.25),
        (np.log1p, 0.5, 2.0 / 3.0),
        (np.expm1, 0.2, math.exp(0.2)),
        (np.tan, 0.25, 1.0 / math.cos(0.25) ** 2),
        (np.arcsin, 0.5, 2.0 / math.sqrt(3.0)),
        (np.arccos, 0.5, -2.0 / math.sqrt(3.0)),
    ],
)
def test_registered_unary_alias_in_lambda_keeps_exact_gradient(
    ufunc: np.ufunc, point: float, expected: float
) -> None:
    """Captured genuine ufunc aliases retain analytic derivatives in lambdas."""
    result = whole_program_value_and_grad(lambda values: np.sum(ufunc(values)), [point])
    np.testing.assert_allclose(result.gradient, [expected], rtol=1.0e-12, atol=1.0e-12)


def test_source_visible_lambda_helper_preserves_captured_coefficient() -> None:
    """A pure lambda helper propagates its parameter and closure into the runtime."""

    def helper_for(coefficient: list[float]) -> Callable[[TraceADScalar], object]:
        return lambda value: value * coefficient[0]

    helper = helper_for([2.0])

    def objective(values: TraceADArray) -> object:
        return helper(cast(TraceADScalar, values[0])) + values[1] ** 2

    result = whole_program_value_and_grad(objective, [3.0, 4.0])
    assert result.value == pytest.approx(22.0)
    np.testing.assert_allclose(result.gradient, [2.0, 8.0], rtol=1.0e-12, atol=1.0e-12)


def test_lambda_callback_refuses_before_external_write() -> None:
    """Inspecting a root lambda still finds a helper's captured ledger write."""
    calls: list[str] = []

    def callback() -> float:
        calls.append("called")
        return 7.0

    with pytest.raises(ValueError, match="external.*callback") as refusal:
        whole_program_value_and_grad(lambda values: values[0] + callback(), [2.0])
    assert "line=" in str(refusal.value)
    assert calls == []


def test_lambda_scatter_refuses_before_captured_write() -> None:
    """A lambda's registered scatter identity cannot mutate captured storage."""
    state = np.array([2.0])
    with pytest.raises(ValueError, match="captured_mutation"):
        whole_program_value_and_grad(lambda values: np.add.at(state, [0], [values[0]]), [2.0])
    np.testing.assert_array_equal(state, [2.0])


def test_lambda_rng_refuses_without_state_advance() -> None:
    """Lambda admission refuses ambient draws before the first random value."""
    before = np.random.get_state(legacy=True)
    with pytest.raises(ValueError, match="ambient_rng"):
        whole_program_value_and_grad(lambda values: values[0] + np.random.random(), [2.0])
    after = np.random.get_state(legacy=True)
    assert isinstance(before, tuple) and isinstance(after, tuple)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_lambda_helper_return_keeps_captured_alias_origin() -> None:
    """A lambda result alias keeps its captured storage write refusal."""
    state = [2.0]

    def helper_for() -> Callable[[], list[float]]:
        return lambda: state

    helper = helper_for()

    def objective(values: TraceADArray) -> object:
        alias = helper()
        alias[0] = 5.0
        return values[0]

    with pytest.raises(ValueError, match="captured_mutation"):
        whole_program_value_and_grad(objective, [2.0])
    assert state == [2.0]


def test_lambda_helper_preserves_active_integer_refusal() -> None:
    """Active helper arguments cannot become integer metadata inside a lambda."""

    def helper_for() -> Callable[[TraceADScalar], int]:
        return lambda value: int(cast(float, value))

    helper = helper_for()

    def objective(values: TraceADArray) -> object:
        return values[0] + helper(cast(TraceADScalar, values[0]))

    with pytest.raises(ValueError, match="dynamic_integer"):
        whole_program_value_and_grad(objective, [2.0])


def test_ambiguous_lambda_source_cannot_borrow_a_pure_sibling() -> None:
    """Multiple source lambdas refuse until the actual callable can be identified."""
    calls: list[str] = []

    def callback() -> float:
        calls.append("called")
        return 7.0

    objectives = (lambda values: np.sum(values**2), lambda values: values[0] + callback())
    with pytest.raises(ValueError, match="source definition is unavailable"):
        whole_program_value_and_grad(objectives[1], [2.0])
    assert calls == []


def test_numpy_append_creates_local_storage_with_the_native_derivative() -> None:
    """A frozen NumPy allocator cannot be confused with a captured list method."""

    def objective(values: TraceADArray) -> object:
        appended = cast(TraceADArray, np.append(values, values))
        return cast(TraceADArray, appended**2).sum()

    inputs = np.array([1.0, 2.0])
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == 10.0
    np.testing.assert_array_equal(result.gradient, [4.0, 8.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0, 8.0])
    np.testing.assert_array_equal(inputs, [1.0, 2.0])


def test_scatter_alias_has_stable_identity_through_loop_inspection() -> None:
    """Repeated attribute reads bind the same native scatter operation."""

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        for index in range(2):
            scatter = cast(Callable[[object, object, object], None], np.add.at)
            scatter(working, [index], values[index])
        return cast(TraceADArray, working**2).sum()

    result = whole_program_value_and_grad(objective, [1.0, 2.0], trace=False)
    assert result.value == 20.0
    np.testing.assert_array_equal(result.gradient, [8.0, 16.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [8.0, 16.0])


def test_helper_parameter_keeps_the_native_scatter_callable_and_destination() -> None:
    """A source-visible helper retains both callable and destination provenance."""

    def apply(scatter: Callable[[object, object, object], None], storage: TraceADArray) -> None:
        scatter(storage, [0], storage[1])

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        apply(cast(Callable[[object, object, object], None], np.add.at), working)
        return cast(TraceADArray, working**2).sum()

    result = whole_program_value_and_grad(objective, [1.0, 2.0], trace=False)
    assert result.value == 13.0
    np.testing.assert_array_equal(result.gradient, [6.0, 10.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [6.0, 10.0])


def test_scatter_alias_cannot_mutate_captured_destination() -> None:
    """Native method admission retains refusal before captured storage changes."""
    state = np.array([2.0])

    def objective(values: TraceADArray) -> object:
        scatter = cast(Callable[[object, object, object], None], np.add.at)
        scatter(state, [0], values[0])
        return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured_mutation.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [2.0])
    assert active_reserved_bytes() == baseline


def test_numpy_reshape_and_array_reductions_preserve_active_storage() -> None:
    """A NumPy function result remains an owned trace receiver for reductions."""

    def objective(values: TraceADArray) -> object:
        vector = cast(TraceADArray, np.reshape(values, (2,)))
        return vector.var() + vector.std() + vector.max() - vector.min()

    result = whole_program_value_and_grad(objective, [1.0, 3.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [-2.5, 2.5])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [-2.5, 2.5])


def test_frozen_numpy_linear_solve_keeps_its_analytic_pullback() -> None:
    """Nested module identity resolves a supported native linear-algebra route."""

    def objective(values: TraceADArray) -> object:
        matrix = cast(TraceADArray, np.reshape(values, (1, 1)))
        return np.sum(np.linalg.solve(matrix, np.array([2.0])))

    result = whole_program_value_and_grad(objective, [4.0], trace=False)
    assert result.value == 0.5
    np.testing.assert_array_equal(result.gradient, [-0.125])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [-0.125])


@pytest.mark.parametrize("view", ["broadcast", "ravel", "transpose", "diagonal", "split"])
def test_captured_numpy_view_cannot_hide_external_storage(view: str) -> None:
    """Supported view identities retain the captured source of a later write.

    Parameters
    ----------
    view
        Actual NumPy view operation whose result aliases captured storage.

    """
    state = np.array([[2.0]])

    def objective(values: TraceADArray) -> object:
        if view == "broadcast":
            alias = np.broadcast_to(state, (1, 1))
        elif view == "ravel":
            alias = np.ravel(state)
        elif view == "transpose":
            alias = np.transpose(state)
        elif view == "diagonal":
            alias = np.diagonal(state)
        else:
            alias = np.split(state, 1)[0]
        alias[0] = 5.0
        return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured_mutation.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [[2.0]])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("route", ["ufunc_keyword", "ufunc_positional", "reduction", "method"])
def test_numpy_output_storage_refuses_before_captured_buffer_changes(route: str) -> None:
    """Native output operands cannot mutate captured storage during derivative capture.

    Parameters
    ----------
    route
        Public NumPy output argument route used by the objective.

    """
    state = np.array([7.0])
    if route == "ufunc_keyword":

        def objective(values: TraceADArray) -> object:
            np.add(1.0, 1.0, out=state)
            return values[0]

    elif route == "ufunc_positional":

        def objective(values: TraceADArray) -> object:
            np.add(1.0, 1.0, state)
            return values[0]

    elif route == "reduction":

        def objective(values: TraceADArray) -> object:
            np.sum(np.array([1.0, 1.0]), out=state, keepdims=True)
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            np.array([1.0, 1.0]).sum(out=state, keepdims=True)
            return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError) as refusal:
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert "captured_mutation" in str(refusal.value)
    assert "line=" in str(refusal.value) and "absolute_line=" in str(refusal.value)
    assert active_reserved_bytes() == baseline


def test_numpy_local_output_storage_keeps_its_analytic_derivative() -> None:
    """An objective-owned native output buffer remains eligible for real AD execution."""

    def objective(values: TraceADArray) -> object:
        working = np.zeros(1)
        np.add(1.0, 2.0, out=working)
        return values[0] * working[0]

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "route",
    [
        "tuple_alias",
        "tuple_copy",
        "list_index",
        "mapping_index",
        "mapping_copy",
        "mapping_keyword",
        "helper_kwargs",
        "branches",
    ],
)
def test_output_container_alias_preserves_captured_buffer_provenance(route: str) -> None:
    """Owned containers and helper keyword dictionaries cannot hide external output buffers.

    Parameters
    ----------
    route
        Source-visible container or helper route retaining the same captured array.

    """
    state = np.array([7.0])

    def write(**options: Unpack[_NumpyOutput]) -> None:
        np.add(1.0, 1.0, **options)

    if route == "tuple_alias":

        def objective(values: TraceADArray) -> object:
            outputs = (state,)
            np.add(1.0, 1.0, out=outputs)
            return values[0]

    elif route == "tuple_copy":

        def objective(values: TraceADArray) -> object:
            outputs = cast(tuple[NDArray[np.float64]], tuple([state]))
            np.add(1.0, 1.0, out=outputs)
            return values[0]

    elif route == "list_index":

        def objective(values: TraceADArray) -> object:
            outputs = [state]
            np.add(1.0, 1.0, out=outputs[0])
            return values[0]

    elif route == "mapping_index":

        def objective(values: TraceADArray) -> object:
            options = {"out": state}
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "mapping_keyword":

        def objective(values: TraceADArray) -> object:
            options: _NumpyOutput = {"out": state}
            np.add(1.0, 1.0, **options)
            return values[0]

    elif route == "mapping_copy":

        def objective(values: TraceADArray) -> object:
            options = dict({"out": state})
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "helper_kwargs":

        def objective(values: TraceADArray) -> object:
            write(out=state)
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            if values[0] > 0:
                outputs = [state]
            else:
                outputs = [np.zeros(1)]
            np.add(1.0, 1.0, out=outputs[0])
            return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError) as refusal:
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert "captured_mutation" in str(refusal.value)
    assert "line=" in str(refusal.value)
    assert active_reserved_bytes() == baseline


def test_helper_keyword_output_keeps_objective_owned_storage_and_replay() -> None:
    """Expanded local keyword storage preserves a source-visible helper's valid derivative."""

    def write(**options: Unpack[_NumpyOutput]) -> None:
        np.add(1.0, 2.0, **options)

    def objective(values: TraceADArray) -> object:
        working = np.zeros(1)
        options: _NumpyOutput = {"out": working}
        write(**options)
        return values[0] * working[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])


@pytest.mark.parametrize("route", ["array_without_copy", "keyword_view"])
def test_numpy_aliasing_constructor_refuses_before_captured_write(route: str) -> None:
    """Copy policy and keyword view inputs preserve the source array's storage origin.

    Parameters
    ----------
    route
        Aliasing array construction or keyword view operation.

    """
    state = np.array([7.0])
    if route == "array_without_copy":

        def objective(values: TraceADArray) -> object:
            alias = np.array(state, copy=False)
            alias[0] = 2.0
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            alias = np.transpose(a=state)
            alias[0] = 2.0
            return values[0]

    with pytest.raises(ValueError) as refusal:
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert "captured_mutation" in str(refusal.value)


def test_captured_view_shape_metadata_does_not_transfer_array_ownership() -> None:
    """An immutable captured shape cannot turn an owned trace view into external storage."""
    shape = (2,)

    def objective(values: TraceADArray) -> object:
        working = cast(TraceADArray, np.reshape(values, shape))
        scatter = cast(Callable[[object, object, object], None], np.add.at)
        scatter(working, [0], working[1])
        return cast(TraceADArray, working**2).sum()

    inputs = np.array([1.0, 2.0])
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == 13.0
    np.testing.assert_array_equal(result.gradient, [6.0, 10.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [6.0, 10.0])
    np.testing.assert_array_equal(inputs, [1.0, 2.0])


def test_captured_numpy_array_copy_owns_its_output_buffer() -> None:
    """An explicit native array copy can change without mutating its captured source."""
    state = np.array([7.0])

    def objective(values: TraceADArray) -> object:
        working = np.array(state, copy=True)
        np.add(1.0, 2.0, out=working)
        return values[0] * working[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])


def test_unknown_callable_identity_refuses_without_equality_or_execution() -> None:
    """Compiler inspection cannot invoke an unknown callable's comparison protocol."""
    calls: list[str] = []

    class OpaqueCallable:
        def __eq__(self, other: object) -> bool:
            calls.append("equality")
            raise AssertionError("opaque callable equality was invoked")

        def __call__(self) -> float:
            calls.append("call")
            return 7.0

    opaque = OpaqueCallable()

    def objective(values: TraceADArray) -> object:
        return values[0] + opaque()

    assert "external_callback" in _find(objective)
    assert calls == []


@pytest.mark.parametrize(
    "route",
    ["ufunc", "reduction", "method", "helper", "scatter", "view", "array", "binding"],
)
def test_positional_expansion_preserves_captured_storage_before_execution(route: str) -> None:
    """Expanded call operands cannot hide a native write to captured array storage.

    Parameters
    ----------
    route
        Native output, view, scatter or source-visible helper call route.

    """
    state = np.array([7.0])
    invoke = cast(Callable[..., object], np.add)
    reduce = cast(Callable[..., object], np.sum)
    scatter = cast(Callable[[object, object, object], None], np.add.at)

    def write(*arguments: object) -> None:
        invoke(*arguments)

    def write_buffer(buffer: NDArray[np.float64]) -> None:
        buffer[0] = 2.0

    if route == "ufunc":

        def objective(values: TraceADArray) -> object:
            arguments = (1.0, 1.0, state)
            invoke(*arguments)
            return values[0]

    elif route == "reduction":

        def objective(values: TraceADArray) -> object:
            arguments = (np.array([1.0, 1.0]), None, None, state)
            reduce(*arguments, keepdims=True)
            return values[0]

    elif route == "method":

        def objective(values: TraceADArray) -> object:
            arguments = (None, None, state)
            np.array([1.0, 1.0]).sum(*arguments, keepdims=True)
            return values[0]

    elif route == "helper":

        def objective(values: TraceADArray) -> object:
            arguments = (1.0, 1.0, state)
            write(*arguments)
            return values[0]

    elif route == "scatter":

        def objective(values: TraceADArray) -> object:
            arguments = (state, [0], 2.0)
            scatter(*arguments)
            return values[0]

    elif route == "view":

        def objective(values: TraceADArray) -> object:
            alias = np.transpose(*(state,))
            alias[0] = 2.0
            return values[0]

    elif route == "array":

        def objective(values: TraceADArray) -> object:
            alias = np.array(*(state,), copy=False)
            alias[0] = 2.0
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            arguments = (state,)
            write_buffer(*arguments)
            return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError) as refusal:
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert "captured_mutation" in str(refusal.value)
    assert "line=" in str(refusal.value)
    assert active_reserved_bytes() == baseline


def test_variadic_helper_keeps_owned_mutation_and_analytic_replay() -> None:
    """Local array provenance survives variadic binding and nested buffer projection."""

    def write(*buffers: NDArray[np.float64]) -> None:
        buffers[0][0] = 3.0

    def objective(values: TraceADArray) -> object:
        working = np.zeros(1)
        write(*(working,))
        return values[0] * working[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])


def test_unknown_arity_trace_expansion_keeps_existing_product_derivative() -> None:
    """A trace array expanded into helper varargs retains the original active domain."""

    def product(*items: TraceADScalar) -> object:
        return items[0] * items[1]

    def objective(values: TraceADArray) -> object:
        return product(*values)

    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0, 2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0, 2.0])


@pytest.mark.parametrize(
    "route", ["duplicate_expansions", "duplicate_explicit", "opaque", "oversized", "owned"]
)
def test_helper_keyword_expansion_admits_owned_storage_and_refuses_ambiguous_binding(
    route: str,
) -> None:
    """Callback keyword binding preserves local storage or refuses before caller writes.

    Parameters
    ----------
    route
        Duplicate, opaque, oversized or supported locally owned keyword mapping.

    """
    state = np.array([7.0])
    protocols: list[str] = []

    class Options(dict[str, NDArray[np.float64]]):
        def __len__(self) -> int:
            """Record unwanted dictionary-subclass size inspection."""
            protocols.append("length")
            return super().__len__()

        def __iter__(self) -> Iterator[str]:
            """Record unwanted dictionary-subclass key inspection."""
            protocols.append("iteration")
            return super().__iter__()

    def helper(*, output: NDArray[np.float64]) -> float:
        np.add(1.0, 2.0, out=output)
        return float(output[0])

    invoke = cast(Callable[..., float], helper)
    if route == "duplicate_expansions":

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(**{"output": state}, **{"output": np.zeros(1)})

    elif route == "duplicate_explicit":

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(output=state, **{"output": np.zeros(1)})

    elif route == "opaque":
        opaque_options = Options(output=state)

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(**opaque_options)

    elif route == "oversized":
        oversized_options = {f"output_{index}": state for index in range(4097)}

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(**oversized_options)

    else:

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(**{"output": np.zeros(1)})

    baseline = active_reserved_bytes()
    if route == "owned":
        result = whole_program_value_and_grad(objective, [2.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    else:
        detail = (
            "duplicate callback keyword binding is unsupported"
            if route.startswith("duplicate_")
            else "keyword expansion requires known plain mapping storage"
        )
        with pytest.raises(ValueError, match=f"{detail}.*line="):
            whole_program_value_and_grad(objective, [2.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert protocols == []
    assert active_reserved_bytes() == baseline


def test_variadic_active_integer_conversion_remains_located_before_execution() -> None:
    """Helper varargs preserve active dependence in source-visible integer conversion."""

    def count(*items: TraceADScalar) -> int:
        return int(items[0])

    def objective(values: TraceADArray) -> object:
        return count(*values) * values[0]

    assert "dynamic_integer" in _find(objective)
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="dynamic_integer.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "route",
    [
        "literal_star",
        "multiple_stars",
        "captured_tuple",
        "unknown_prefix",
        "mapping_shallow_copy",
        "list_shallow_copy",
        "loop_item",
        "comprehension_item",
        "mapping_get",
        "mapping_assignment",
        "list_assignment",
        "mapping_update",
        "mapping_alias",
        "nested_alias",
        "helper_update",
        "branch_update",
    ],
)
def test_container_projection_retains_external_buffer_origin(route: str) -> None:
    """Iteration, shallow copies and container writes retain the pointed-to array identity.

    Parameters
    ----------
    route
        Source-visible projection or container update preceding the native write.

    """
    state = np.array([7.0])
    invoke = cast(Callable[..., object], np.add)
    captured_arguments = (1.0, 1.0, state)

    def replace_output(options: dict[str, NDArray[np.float64]]) -> None:
        options["out"] = state

    if route == "literal_star":

        def objective(values: TraceADArray) -> object:
            prefix = (1.0, 1.0)
            arguments = (*prefix, state)
            invoke(*arguments)
            return values[0]

    elif route == "multiple_stars":

        def objective(values: TraceADArray) -> object:
            invoke(*(1.0,), *(1.0, state))
            return values[0]

    elif route == "captured_tuple":

        def objective(values: TraceADArray) -> object:
            invoke(*captured_arguments)
            return values[0]

    elif route == "unknown_prefix":

        def objective(values: TraceADArray) -> object:
            invoke(*range(2), state)
            return values[0]

    elif route == "mapping_shallow_copy":

        def objective(values: TraceADArray) -> object:
            options = {"out": state}.copy()
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "list_shallow_copy":

        def objective(values: TraceADArray) -> object:
            outputs = [state].copy()
            np.add(1.0, 1.0, out=outputs[0])
            return values[0]

    elif route == "loop_item":

        def objective(values: TraceADArray) -> object:
            for buffer in [state]:
                np.add(1.0, 1.0, out=buffer)
            return values[0]

    elif route == "comprehension_item":

        def objective(values: TraceADArray) -> object:
            outputs = [buffer for buffer in (state,)]
            np.add(1.0, 1.0, out=outputs[0])
            return values[0]

    elif route == "mapping_get":

        def objective(values: TraceADArray) -> object:
            options = {"out": state, "other": np.zeros(1)}
            key = "out" if values[0] > 0 else "other"
            np.add(1.0, 1.0, out=options.get(key))
            return values[0]

    elif route == "mapping_assignment":

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            options["out"] = state
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "list_assignment":

        def objective(values: TraceADArray) -> object:
            outputs = [np.zeros(1)]
            outputs[0] = state
            np.add(1.0, 1.0, out=outputs[0])
            return values[0]

    elif route == "mapping_update":

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            options.update(out=state)
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "mapping_alias":

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            alias = options
            alias["out"] = state
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "nested_alias":

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            nested = (options,)
            nested[0]["out"] = state
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    elif route == "helper_update":

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            replace_output(options)
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            if values[0] > 0:
                options["out"] = state
            else:
                options["out"] = np.zeros(1)
            np.add(1.0, 1.0, out=options["out"])
            return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError) as refusal:
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert "captured_mutation" in str(refusal.value)
    assert "line=" in str(refusal.value)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("route", ["append", "loop", "comprehension", "native_expansion"])
def test_container_projection_keeps_owned_storage_and_replay(route: str) -> None:
    """Container projection preserves a supported local buffer's numerical contribution.

    Parameters
    ----------
    route
        Owned container operation or exact native numeric sequence expansion.

    """
    coefficients = np.array([1.0, 2.0])
    invoke = cast(Callable[..., object], np.add)
    if route == "append":

        def objective(values: TraceADArray) -> object:
            outputs: list[NDArray[np.float64]] = []
            outputs.append(np.zeros(1))
            np.add(1.0, 2.0, out=outputs[0])
            return values[0] * outputs[0][0]

    elif route == "loop":

        def objective(values: TraceADArray) -> object:
            working = np.zeros(1)
            for buffer in (working,):
                np.add(1.0, 2.0, out=buffer)
            return values[0] * working[0]

    elif route == "comprehension":

        def objective(values: TraceADArray) -> object:
            working = np.zeros(1)
            outputs = [buffer for buffer in (working,)]
            np.add(1.0, 2.0, out=outputs[0])
            return values[0] * working[0]

    else:

        def objective(values: TraceADArray) -> object:
            coefficient = cast(float, invoke(*coefficients))
            return values[0] * coefficient

    inputs = np.array([2.0])
    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(inputs, [2.0])
    np.testing.assert_array_equal(coefficients, [1.0, 2.0])
    assert active_reserved_bytes() == baseline


def test_replaced_external_container_element_keeps_owned_output_domain() -> None:
    """A sequential overwrite removes an unused external element from the output path."""
    state = np.array([7.0])

    def objective(values: TraceADArray) -> object:
        options = {"out": state}
        alias = options
        alias["out"] = np.zeros(1)
        np.add(1.0, 2.0, out=options["out"])
        return values[0] * options["out"][0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])


@pytest.mark.parametrize("captured", [True, False])
def test_mapping_update_pairs_preserve_existing_storage_domain(captured: bool) -> None:
    """Plain key/value pairs retain original mapping-update support and buffer ownership.

    Parameters
    ----------
    captured
        Whether the incoming output buffer belongs to the caller or the objective.

    """
    state = np.array([7.0])
    if captured:

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            options.update([("out", state)])
            np.add(1.0, 2.0, out=options["out"])
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            options.update([("out", np.zeros(1))])
            np.add(1.0, 2.0, out=options["out"])
            return values[0] * options["out"][0]

    baseline = active_reserved_bytes()
    if captured:
        assert "captured_mutation" in _find(objective)
        with pytest.raises(ValueError, match="captured_mutation.*line="):
            whole_program_value_and_grad(objective, [2.0], trace=False)
    else:
        result = whole_program_value_and_grad(objective, [2.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("storage", ["short_pair", "opaque_key", "large_pairs"])
def test_mapping_pair_admission_refuses_unknown_bindings_before_writes(storage: str) -> None:
    """Malformed or unbounded pairs cannot hide an external output buffer.

    Parameters
    ----------
    storage
        Incomplete pair, foreign key with a hash protocol or oversized pair list.

    """
    state = np.array([7.0])
    calls: list[str] = []

    class OpaqueKey:
        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("foreign mapping key hash was called")

    pairs: object
    if storage == "short_pair":
        pairs = [("out", state), ("missing",)]
    elif storage == "opaque_key":
        pairs = [("out", state), (OpaqueKey(), np.zeros(1))]
    else:
        pairs = [("out", state)] * 4097

    def objective(values: TraceADArray) -> object:
        options = {"out": np.zeros(1)}
        options.update(cast(list[tuple[str, NDArray[np.float64]]], pairs))
        np.add(1.0, 2.0, out=options["out"])
        return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="mapping update requires known plain storage.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("captured", [True, False])
@pytest.mark.parametrize("position", ["prefix", "middle", "suffix"])
def test_starred_unpacking_retains_each_selected_buffers_storage_domain(
    captured: bool, position: str
) -> None:
    """Starred lists retain external writes and real owned-buffer differentiation.

    Parameters
    ----------
    captured
        Whether the selected output belongs to the caller or this objective.
    position
        Position of the starred target in an ordinary unpacking assignment.

    """
    state = np.array([7.0])
    if captured:

        def obtain_buffer() -> NDArray[np.float64]:
            return state

    else:

        def obtain_buffer() -> NDArray[np.float64]:
            return np.zeros(1)

    def objective(values: TraceADArray) -> object:
        if position == "prefix":
            *buffers, _last = [np.zeros(1), obtain_buffer(), np.zeros(1)]
            output = buffers[1]
        elif position == "middle":
            _first, *buffers, _last = [np.zeros(1), obtain_buffer(), np.zeros(1)]
            output = buffers[0]
        else:
            _first, *buffers = [np.zeros(1), obtain_buffer(), np.zeros(1)]
            output = buffers[0]
        np.add(1.0, 2.0, out=output)
        return values[0] * output[0]

    baseline = active_reserved_bytes()
    if captured:
        with pytest.raises(ValueError, match="captured_mutation.*line="):
            whole_program_value_and_grad(objective, [2.0], trace=False)
    else:
        result = whole_program_value_and_grad(objective, [2.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


def test_extended_slice_cardinality_refuses_before_an_output_write() -> None:
    """A mismatched stepped slice has an authored refusal before subsequent effects."""
    state = np.array([7.0])

    def objective(values: TraceADArray) -> object:
        buffers = [np.zeros(1), np.zeros(1), np.zeros(1)]
        buffers[::2] = [state]
        np.add(1.0, 2.0, out=buffers[0])
        return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="slice update cardinality is unsupported.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("captured", [True, False])
@pytest.mark.parametrize(
    "route",
    [
        "mapping_lookup",
        "mapping_assignment",
        "list_assignment",
        "dynamic_slice",
        "variadic_slice",
        "variadic_extend",
        "variadic_extend_tail",
        "variadic_augment",
        "variadic_augment_tail",
        "variadic_insert",
        "while_else",
        "for_else",
        "filtered_comprehension",
        "delete_binding",
    ],
)
def test_container_selection_and_control_flow_preserve_output_ownership(
    tmp_path: Path, captured: bool, route: str
) -> None:
    """Container joins retain caller writes and real owned-buffer native derivatives.

    Parameters
    ----------
    tmp_path
        Owned directory for each actual source-visible objective module.
    captured
        Whether the selected output buffer belongs to the caller.
    route
        Container selection, structural update or control-flow projection.

    """
    bodies = {
        "mapping_lookup": (
            "selection = 0 if values[0] > 0 else 1\n"
            "buffers = {0: obtain_buffer(), 1: np.zeros(1)}\n"
            "output = buffers[selection]\n"
        ),
        "mapping_assignment": (
            "selection = 0 if values[0] > 0 else 1\n"
            "buffers = {0: np.zeros(1), 1: np.zeros(1)}\n"
            "buffers[selection] = obtain_buffer()\n"
            "output = buffers[selection]\n"
        ),
        "list_assignment": (
            "selection = 0 if values[0] > 0 else 1\n"
            "buffers = [np.zeros(1), np.zeros(1)]\n"
            "buffers[selection] = obtain_buffer()\n"
            "output = buffers[selection]\n"
        ),
        "dynamic_slice": (
            "selection = 0 if values[0] > 0 else 1\n"
            "buffers = [np.zeros(1), np.zeros(1)]\n"
            "buffers[selection:] = [obtain_buffer()]\n"
            "output = buffers[0]\n"
        ),
        "variadic_slice": (
            "buffers = [*values]\nbuffers[1:] = [*values, obtain_buffer()]\noutput = buffers[-1]\n"
        ),
        "variadic_extend": (
            "buffers = [*values]\nbuffers.extend([obtain_buffer()])\noutput = buffers[-1]\n"
        ),
        "variadic_extend_tail": (
            "buffers = [*values]\n"
            "buffers.extend([*values, obtain_buffer()])\n"
            "output = buffers[-1]\n"
        ),
        "variadic_augment": (
            "buffers = [*values]\nbuffers += [obtain_buffer()]\noutput = buffers[-1]\n"
        ),
        "variadic_augment_tail": (
            "buffers = [*values]\nbuffers += [*values, obtain_buffer()]\noutput = buffers[-1]\n"
        ),
        "variadic_insert": (
            "buffers = [*values]\n"
            "buffers.insert(len(buffers), obtain_buffer())\n"
            "output = buffers[-1]\n"
        ),
        "while_else": (
            "output = obtain_buffer()\n"
            "index = 0\n"
            "while index < 2:\n"
            "    index += 1\n"
            "else:\n"
            "    np.add(1.0, 2.0, out=output)\n"
        ),
        "for_else": (
            "output = np.zeros(1)\n"
            "for item in (obtain_buffer(),):\n"
            "    output = item\n"
            "else:\n"
            "    np.add(1.0, 2.0, out=output)\n"
        ),
        "filtered_comprehension": (
            "buffers = [item for item in (obtain_buffer(),) if item is not None]\n"
            "output = buffers[0]\n"
        ),
        "delete_binding": ("output = obtain_buffer()\nunused = output\ndel unused\n"),
    }
    state = np.array([7.0])
    accessor = "return state" if captured else "return np.zeros(1)"
    body = bodies[route] + "np.add(1.0, 2.0, out=output)\nreturn values[0] * output[0]\n"
    source = (
        "import numpy as np\n\n"
        f"def obtain_buffer():\n    {accessor}\n\n"
        "def objective(values):\n" + textwrap.indent(body, "    ")
    )
    path = tmp_path / "storage_objective.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins), "state": state}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    inputs = np.array([2.0])
    baseline = active_reserved_bytes()
    if captured:
        assert "captured_mutation" in _find(objective)
        with pytest.raises(ValueError, match="captured_mutation.*line=.*absolute_line="):
            whole_program_value_and_grad(objective, inputs, trace=False)
    elif route == "filtered_comprehension":
        assert _find(objective) == ()
        with pytest.raises(ValueError, match="semantic=filtered_comprehension.*line="):
            whole_program_value_and_grad(objective, inputs, trace=False)
    else:
        assert _find(objective) == ()
        result = whole_program_value_and_grad(objective, inputs, trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(inputs, [2.0])
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    ("route", "semantic", "detail"),
    [
        ("active_mapping_key", "captured_mutation", "NumPy output storage"),
        (
            "unknown_mapping_expansion",
            "external_callback",
            "keyword expansion requires known plain mapping storage",
        ),
        (
            "unknown_mapping_constructor",
            "external_callback",
            "keyword expansion requires known plain mapping storage",
        ),
        (
            "sort_keywords",
            "external_callback",
            "keyword expansion requires known plain mapping storage",
        ),
        ("sort_opaque_key", "external_callback", "sort key callback identity"),
        (
            "update_keywords",
            "external_callback",
            "keyword expansion requires known plain mapping storage",
        ),
        (
            "ufunc_keywords",
            "external_callback",
            "keyword expansion requires known plain mapping storage",
        ),
        ("variadic_assignment", "captured_mutation", "NumPy output storage"),
    ],
)
def test_dynamic_mapping_and_keyword_refusals_preserve_state_and_recover(
    tmp_path: Path, route: str, semantic: str, detail: str
) -> None:
    """Unknown mapping or call bindings refuse before writes and foreign protocols.

    Parameters
    ----------
    tmp_path
        Owned location of the genuine loaded objective module.
    route
        Mapping projection, expanded keywords, opaque sort key or variadic update.
    semantic
        Authored unsupported-effect category required at the public boundary.
    detail
        Stable deliberately authored refusal detail expected for this route.

    """
    bodies = {
        "active_mapping_key": "buffers = {values[0]: state}\nnp.add(1.0, 2.0, out=buffers[values[0]])\n",
        "unknown_mapping_expansion": "buffers = {**values, 'out': state}\n_unused = (*buffers, *values)\nnp.add(1.0, 2.0, **buffers)\n",
        "unknown_mapping_constructor": "buffers = dict(values, out=state)\nnp.add(1.0, 2.0, **buffers)\n",
        "sort_keywords": "buffers = [1.0, 2.0]\nbuffers.sort(**values)\n",
        "sort_opaque_key": "buffers = [1.0, 2.0]\nbuffers.sort(key=opaque)\n",
        "update_keywords": "buffers = {}\nbuffers.update(**values)\n",
        "ufunc_keywords": "np.add(1.0, 2.0, **values)\n",
        "variadic_assignment": "buffers = [*values, state]\nselection = 0 if values[0] > 0 else 1\nbuffers[selection] = state\nnp.add(1.0, 2.0, out=buffers[-1])\n",
    }
    calls: list[str] = []

    class OpaqueKey:
        def __eq__(self, other: object) -> bool:
            calls.append("equality")
            raise AssertionError("opaque sort key equality was invoked")

        def __call__(self, argument: object) -> float:
            calls.append("call")
            return 0.0

    state = np.array([7.0])
    source = "import numpy as np\n\ndef objective(values):\n" + textwrap.indent(
        bodies[route] + "return values[0]\n", "    "
    )
    path = tmp_path / "mapping_binding.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {
        "__builtins__": vars(builtins),
        "state": state,
        "opaque": OpaqueKey(),
    }
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    findings = find_objective_effects(objective, ast.parse(source))
    assert any(item.semantic == semantic and detail in item.detail for item in findings)
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    with pytest.raises(ValueError, match=semantic + ".*line=.*absolute_line="):
        whole_program_value_and_grad(objective, inputs, trace=False)
    np.testing.assert_array_equal(state, [7.0])
    np.testing.assert_array_equal(inputs, [2.0])
    assert calls == []
    assert active_reserved_bytes() == baseline
    restored = "import numpy as np\n\ndef objective(values):\n    return values[0] * 3.0\n"
    path.write_text(restored, encoding="utf-8")
    exec(compile(restored, str(path), "exec"), namespace)
    result = whole_program_value_and_grad(
        cast(FunctionType, namespace["objective"]), inputs, trace=False
    )
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    assert calls == []
    np.testing.assert_array_equal(state, [7.0])
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "route", ["annotation_only", "empty_helper", "constant_array_fill", "owned_attribute_delete"]
)
def test_owned_array_and_helper_statements_preserve_native_derivatives(
    tmp_path: Path, route: str
) -> None:
    """Owned array writes and effect-free source statements retain native derivatives.

    Parameters
    ----------
    tmp_path
        Actual loaded module path for the selected ordinary Python statements.
    route
        Local annotation, empty helper, native array fill or owned attribute deletion.

    """
    bodies = {
        "annotation_only": "coefficient: float\nreturn values[0] * 3.0\n",
        "empty_helper": "helper()\nreturn values[0] * 3.0\n",
        "constant_array_fill": "output = np.zeros(1)\noutput.fill(3.0)\nreturn values[0] * output[0]\n",
        "owned_attribute_delete": "record = Storage()\nrecord.value = 3.0\ndel record.value\nreturn values[0] * 3.0\n",
    }
    source = (
        "import numpy as np\n\nclass Storage:\n    pass\n\ndef helper():\n    return\n\n"
        "def objective(values):\n" + textwrap.indent(bodies[route], "    ")
    )
    path = tmp_path / "owned_statements.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    inputs = np.array([2.0])
    baseline = active_reserved_bytes()
    assert objective(inputs.copy()) == 6.0
    assert find_objective_effects(objective, ast.parse(source)) == ()
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    ("route", "semantic", "detail"),
    [
        ("external_delete", "captured_mutation", "captured mutation through an alias"),
        ("external_attribute_aug", "captured_mutation", "captured mutation through an alias"),
        ("unregistered_at", "external_callback", "external callback identity"),
        ("native_method_alias", "external_callback", "external callback identity"),
        ("mismatched_unpack", "captured_mutation", "NumPy output storage"),
        ("variadic_output", "captured_mutation", "NumPy output storage"),
    ],
)
def test_external_delete_and_output_bindings_refuse_before_objective_entry(
    tmp_path: Path, route: str, semantic: str, detail: str
) -> None:
    """External writes and uncertain output storage refuse before objective execution.

    Parameters
    ----------
    tmp_path
        Owned source-visible module path used for refusal and recapture recovery.
    route
        Actual deletion, attribute update, native method or output binding.
    semantic
        Authored effect category required before numerical evaluation.
    detail
        Deliberately authored detail for the selected storage or callable boundary.

    """
    bodies = {
        "external_delete": "del state[0]\n",
        "external_attribute_aug": "state.shape += (1,)\n",
        "unregistered_at": "state.at(0)\n",
        "native_method_alias": "native_total()\n",
        "mismatched_unpack": "left, right, extra = [state, np.zeros(1)]\nnp.add(1.0, 2.0, out=left)\n",
        "variadic_output": "buffers = (*values, state)\nnp.add(1.0, 2.0, out=buffers)\n",
    }
    source = "import numpy as np\n\ndef objective(values):\n" + textwrap.indent(
        bodies[route] + "return values[0] * 3.0\n", "    "
    )
    state = np.array([7.0])
    namespace: dict[str, object] = {
        "__builtins__": vars(builtins),
        "state": state,
        "native_total": state.sum,
    }
    path = tmp_path / "external_statements.py"
    path.write_text(source, encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    findings = find_objective_effects(objective, ast.parse(source))
    assert any(item.semantic == semantic and detail in item.detail for item in findings)
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    previous = sys.getprofile()
    calls: list[str] = []

    def observe(frame: FrameType, event: str, argument: object) -> None:
        if event == "call" and frame.f_code is objective.__code__:
            calls.append("objective")

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match=semantic + ".*line=.*absolute_line="):
            whole_program_value_and_grad(objective, inputs, trace=False)
    finally:
        sys.setprofile(previous)
    assert calls == []
    assert state.shape == (1,)
    np.testing.assert_array_equal(state, [7.0])
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline
    restored = "def objective(values):\n    return values[0] * 3.0\n"
    path.write_text(restored, encoding="utf-8")
    exec(compile(restored, str(path), "exec"), namespace)
    result = whole_program_value_and_grad(
        cast(FunctionType, namespace["objective"]), inputs, trace=False
    )
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    assert state.shape == (1,)
    np.testing.assert_array_equal(state, [7.0])
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline
    assert sys.getprofile() is previous


@pytest.mark.parametrize("binding", ["global", "nonlocal"])
def test_declared_external_assignment_refuses_before_binding_changes(
    tmp_path: Path, binding: str
) -> None:
    """Explicit global and nonlocal writes retain located pre-execution refusal.

    Parameters
    ----------
    tmp_path
        Actual source-visible module and closure factory path.
    binding
        External binding declaration selected by the objective.

    """
    if binding == "global":
        source = (
            "state = 7.0\n"
            "def objective(values):\n    global state\n    state = 8.0\n    return values[0]\n"
            "def read_state():\n    return state\n"
        )
    else:
        source = (
            "def build():\n    state = 7.0\n"
            "    def objective(values):\n        nonlocal state\n        state = 8.0\n        return values[0]\n"
            "    def read_state():\n        return state\n"
            "    return objective, read_state\n"
            "objective, read_state = build()\n"
        )
    path = tmp_path / "declared_binding.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    read_state = cast(Callable[[], float], namespace["read_state"])
    assert "captured_mutation" in _find(objective)
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    with pytest.raises(ValueError, match="captured_mutation.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, inputs, trace=False)
    assert read_state() == 7.0
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline


def test_callable_instance_cannot_supply_function_effect_identity() -> None:
    """A callable object refuses source identity without executing its protocols."""
    calls: list[str] = []

    class Objective:
        def __call__(self, values: TraceADArray) -> object:
            calls.append("objective")
            return values[0] * 3.0

    objective = Objective()
    tree = ast.parse(textwrap.dedent(inspect.getsource(Objective.__call__)))
    findings = find_objective_effects(objective, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback requires source-visible function identity"
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("view", ["ravel", "broadcast_arrays"])
def test_unknown_arity_view_cannot_borrow_an_external_output_buffer(
    tmp_path: Path, view: str
) -> None:
    """Unknown positional view inputs retain output refusal before native execution.

    Parameters
    ----------
    tmp_path
        Owned source-visible module path for the selected NumPy view.
    view
        Single-source view or multi-source view with unknown expansion arity.

    """
    state = np.array([7.0])
    source = (
        "import numpy as np\n\ndef objective(values):\n"
        f"    output = np.{view}(*values, out=state)\n"
        "    return values[0] * output[0]\n"
    )
    path = tmp_path / "expanded_view.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins), "state": state}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    findings = find_objective_effects(objective, ast.parse(source))
    assert any(item.semantic == "captured_mutation" for item in findings)
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    with pytest.raises(ValueError, match="captured_mutation.*line=.*absolute_line="):
        whole_program_value_and_grad(objective, inputs, trace=False)
    np.testing.assert_array_equal(state, [7.0])
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("call", ["append()", "insert(0)"])
def test_mixed_native_container_missing_operands_refuse_before_execution(
    tmp_path: Path, call: str
) -> None:
    """Missing native operands refuse even when a branch joins different containers.

    Parameters
    ----------
    tmp_path
        Actual source-visible objective module used for binding inspection.
    call
        Native mutation lacking its required incoming operand.

    """
    source = (
        "def objective(values):\n"
        "    storage = [1.0, 2.0] if values[0] > 0 else {'one': 1.0}\n"
        f"    storage.{call}\n"
        "    return values[0] * 3.0\n"
    )
    path = tmp_path / "mixed_missing_operands.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    findings = find_objective_effects(objective, ast.parse(source))
    assert any(
        item.semantic == "external_callback"
        and item.detail == "local container call signature is unsupported"
        for item in findings
    )
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    with pytest.raises(ValueError, match="local container call signature is unsupported.*line="):
        whole_program_value_and_grad(objective, inputs, trace=False)
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "container", ["np.zeros(1)", "(1.0, 2.0)", "(item for item in (1.0, 2.0))"]
)
def test_local_nonmutable_deletion_refuses_before_numerical_execution(
    tmp_path: Path, container: str
) -> None:
    """Unsupported native deletion retains a located refusal before evaluation.

    Parameters
    ----------
    tmp_path
        Actual loaded objective source path for the selected native storage.
    container
        NumPy array, tuple or generator whose native elements cannot be deleted.

    """
    source = (
        "import numpy as np\n\ndef objective(values):\n"
        f"    storage = {container}\n"
        "    del storage[0]\n"
        "    return values[0] * 3.0\n"
    )
    path = tmp_path / "nonmutable_deletion.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    findings = find_objective_effects(objective, ast.parse(source))
    assert any(
        item.semantic == "external_callback"
        and item.detail == "local deletion requires known mutable container storage"
        for item in findings
    )
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    calls: list[str] = []
    previous = sys.getprofile()

    def observe(frame: FrameType, event: str, argument: object) -> None:
        if event == "call" and frame.f_code is objective.__code__:
            calls.append("objective")

    try:
        sys.setprofile(observe)
        with pytest.raises(
            ValueError,
            match="local deletion requires known mutable container storage.*line=.*absolute_line=",
        ):
            whole_program_value_and_grad(objective, inputs, trace=False)
    finally:
        sys.setprofile(previous)
    assert calls == []
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline
    assert sys.getprofile() is previous


@pytest.mark.parametrize("value", [2.0, -2.0])
def test_mixed_native_container_copy_preserves_each_branch_derivative(
    tmp_path: Path, value: float
) -> None:
    """Both native copy methods retain supported local derivatives after a branch join.

    Parameters
    ----------
    tmp_path
        Actual source-visible module path for the joined native storage.
    value
        Positive list branch or negative dictionary branch input.

    """
    source = (
        "def objective(values):\n"
        "    storage = [3.0] if values[0] > 0 else {0: 3.0}\n"
        "    copied = storage.copy()\n"
        "    return values[0] * copied[0]\n"
    )
    path = tmp_path / "mixed_copy.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    inputs = np.array([value])
    baseline = active_reserved_bytes()
    assert objective(inputs.copy()) == value * 3.0
    assert find_objective_effects(objective, ast.parse(source)) == ()
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == value * 3.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(inputs, [value])
    assert active_reserved_bytes() == baseline


def test_batching_transform_over_a_source_visible_helper_keeps_its_derivative() -> None:
    """The package's own ``vmap`` over an inspected helper is admitted and differentiates exactly.

    The helper is inspected like a directly called helper. A type alias used
    in ``cast`` is type metadata and is not captured program state.
    """
    alias = NDArray[np.float64]

    def row_loss(row: TraceADArray) -> object:
        return row[0] ** 2 + np.sin(row[1])

    def objective(values: TraceADArray) -> object:
        rows = values.reshape((2, 2))
        return np.sum(cast(alias, vmap(row_loss)(rows)))

    def bound_first(values: TraceADArray) -> object:
        mapped = vmap(row_loss)
        return np.sum(cast(Any, mapped(values.reshape((2, 2)))))

    inputs = [0.5, 0.25, -1.2, 0.75]
    expected = [1.0, math.cos(0.25), -2.4, math.cos(0.75)]
    baseline = active_reserved_bytes()
    for candidate in (objective, bound_first):
        assert _find(candidate) == ()
        result = whole_program_value_and_grad(candidate, inputs, trace=False)
        assert result.value == pytest.approx(
            0.25 + math.sin(0.25) + 1.44 + math.sin(0.75), abs=1e-14
        )
        np.testing.assert_allclose(result.gradient, expected, rtol=0.0, atol=1e-14)
    assert active_reserved_bytes() == baseline


def test_batching_transform_keeps_the_refusals_of_the_mapped_helper() -> None:
    """A mapped helper that writes captured storage is refused at the mapping call."""
    state = np.array([2.0])

    def write(row: TraceADArray) -> object:
        cast(list[float], state)[0] = 5.0
        return row[0]

    def objective(values: TraceADArray) -> object:
        return np.sum(cast(Any, vmap(write)(values.reshape((2, 1)))))

    assert "captured_mutation" in _find(objective)
    np.testing.assert_array_equal(state, [2.0])


@pytest.mark.parametrize("form", ["axes", "native", "two_functions", "starred"])
def test_batching_transform_contract_covers_one_function_over_the_leading_axis(
    form: str,
) -> None:
    """Axis options, a native callable and several functions, written out or starred, are refused.

    Parameters
    ----------
    form
        The call shape outside the effect contract of the transform.

    """
    helpers: tuple[Callable[..., object], ...] = (np.sin, np.cos)

    def row_loss(row: TraceADArray) -> object:
        return row[0]

    if form == "axes":

        def objective(values: TraceADArray) -> object:
            return np.sum(cast(Any, vmap(row_loss, in_axes=0)(values.reshape((2, 1)))))

    elif form == "native":

        def objective(values: TraceADArray) -> object:
            return np.sum(cast(Any, vmap(np.sin)(values)))

    elif form == "two_functions":

        def objective(values: TraceADArray) -> object:
            return np.sum(_UNTYPED_VMAP(row_loss, row_loss)(values.reshape((2, 1))))

    else:

        def objective(values: TraceADArray) -> object:
            return np.sum(_UNTYPED_VMAP(*helpers)(values))

    assert set(_find(objective)) == {"external_callback"}
    assert (
        "batching transform effect contract covers one source-visible function "
        "mapped over the leading axis"
    ) in _details(objective)


@pytest.mark.parametrize("form", ["keyword", "two_batches", "starred"])
def test_mapped_helper_is_called_with_one_positional_batch(form: str) -> None:
    """A mapped helper called with a keyword, two batches or a batch of unknown arity is refused.

    Parameters
    ----------
    form
        The call shape outside the effect contract of the mapped helper.

    """

    def row_loss(row: TraceADArray) -> object:
        return row[0]

    if form == "keyword":

        def objective(values: TraceADArray) -> object:
            return np.sum(cast(Any, vmap(row_loss)(row=values.reshape((2, 1)))))

    elif form == "two_batches":

        def objective(values: TraceADArray) -> object:
            return np.sum(
                cast(Any, vmap(row_loss)(values.reshape((2, 1)), values.reshape((2, 1))))
            )

    else:

        def objective(values: TraceADArray) -> object:
            return np.sum(cast(Any, vmap(row_loss)(*values.reshape((1, 2, 1)))))

    assert _find(objective) == ("external_callback",)
    assert _details(objective) == ("mapped callback is called with one positional batch only",)
