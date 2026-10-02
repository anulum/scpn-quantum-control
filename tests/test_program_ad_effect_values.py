# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared effect storage contracts
"""Qualify abstract storage through actual public capture and replay behavior."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import TraceADArray, whole_program_value_and_grad
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects


@pytest.mark.parametrize("route", ["expression", "statement"])
def test_selected_container_alias_retains_all_possible_write_targets(route: str) -> None:
    """A selected local alias cannot hide a later write into the caller's buffer.

    Parameters
    ----------
    route
        Conditional-expression or branch-statement container selection.

    """
    state = np.array([7.0])
    if route == "expression":

        def objective(values: TraceADArray) -> object:
            first = {"out": np.zeros(1)}
            second = {"out": np.zeros(1)}
            alias = first if values[0] > 0 else second
            alias["out"] = state
            np.add(1.0, 1.0, out=first["out"])
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            first = {"out": np.zeros(1)}
            second = {"out": np.zeros(1)}
            if values[0] > 0:
                alias = first
            else:
                alias = second
            alias["out"] = state
            np.add(1.0, 1.0, out=first["out"])
            return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError) as refusal:
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert "captured_mutation" in str(refusal.value)
    assert "line=" in str(refusal.value)
    assert active_reserved_bytes() == baseline


def test_shallow_outer_copy_keeps_the_nested_container_storage_identity() -> None:
    """Copying an outer list preserves shared storage in its nested dictionary."""
    state = np.array([7.0])

    def objective(values: TraceADArray) -> object:
        child = {"out": np.zeros(1)}
        outer = [child]
        copied = outer.copy()
        copied[0]["out"] = state
        np.add(1.0, 1.0, out=child["out"])
        return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured_mutation.*line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("captured", [True, False])
def test_helper_returned_container_keeps_buffer_ownership(captured: bool) -> None:
    """A returned helper container preserves the origin of its native output buffer.

    Parameters
    ----------
    captured
        Whether the helper receives caller-owned or objective-owned storage.

    """
    state = np.array([7.0])

    def package(buffer: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        return {"out": buffer}

    if captured:

        def objective(values: TraceADArray) -> object:
            options = package(state)
            np.add(1.0, 2.0, out=options["out"])
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            options = package(np.zeros(1))
            np.add(1.0, 2.0, out=options["out"])
            return values[0] * options["out"][0]

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


@pytest.mark.parametrize("captured", [True, False])
def test_finite_cyclic_container_projection_preserves_buffer_origin(captured: bool) -> None:
    """Finite self-referential projections preserve native buffer ownership.

    Parameters
    ----------
    captured
        Whether the selected buffer belongs to the caller or the objective.

    """
    state = np.array([7.0])
    if captured:

        def objective(values: TraceADArray) -> object:
            options: dict[str, object] = {"out": np.zeros(1)}
            options["self"] = options
            options["out"] = state
            nested = cast("dict[str, object]", options["self"])
            np.add(1.0, 2.0, out=cast("NDArray[np.float64]", nested["out"]))
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            working = np.zeros(1)
            options: dict[str, object] = {"out": working}
            options["self"] = options
            nested = cast("dict[str, object]", options["self"])
            np.add(1.0, 2.0, out=cast("NDArray[np.float64]", nested["out"]))
            return values[0] * working[0]

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


def test_conditional_value_keeps_active_integer_dependence() -> None:
    """Joining conditional values preserves the active dependence of their predicate."""

    def objective(values: TraceADArray) -> object:
        count = 1 if values[0] > 0 else 2
        return values[0] * int(count)

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    findings = find_objective_effects(objective, tree)
    assert any(finding.semantic == "dynamic_integer" for finding in findings)
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="dynamic_integer.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("captured", [True, False])
def test_branch_join_preserves_repeated_nested_container_aliases(captured: bool) -> None:
    """A branch join preserves repeated aliases of the same nested container.

    Parameters
    ----------
    captured
        Whether one branch selects a caller-owned output buffer.

    """
    state = np.array([7.0])
    if captured:

        def objective(values: TraceADArray) -> object:
            child = {"out": np.zeros(1)}
            parent = {"left": child, "right": child}
            if values[0] > 0:
                parent["left"]["out"] = state
            else:
                child["out"] = np.zeros(1)
            np.add(1.0, 2.0, out=parent["right"]["out"])
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            working = np.zeros(1)
            child = {"out": working}
            parent = {"left": child, "right": child}
            if values[0] > 0:
                parent["left"]["out"] = working
            else:
                child["out"] = working
            np.add(1.0, 2.0, out=parent["right"]["out"])
            return values[0] * working[0]

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


@pytest.mark.parametrize(
    "route",
    [
        "insert",
        "reverse",
        "pop_result",
        "pop_shift",
        "mapping_pop",
        "delete",
        "inplace_extend",
        "remove",
        "destructure",
        "slice",
    ],
)
def test_structural_container_mutation_cannot_hide_external_output(route: str) -> None:
    """Structural container updates retain the caller's output buffer identity.

    Parameters
    ----------
    route
        In-place sequence update, returned element, mapping projection or binding.

    """
    state = np.array([7.0])
    if route == "insert":

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1)]
            items.insert(0, state)
            np.add(1.0, 1.0, out=items[0])
            return values[0]

    elif route == "reverse":

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1), state]
            items.reverse()
            np.add(1.0, 1.0, out=items[0])
            return values[0]

    elif route == "pop_result":

        def objective(values: TraceADArray) -> object:
            items = [state]
            output = items.pop()
            np.add(1.0, 1.0, out=output)
            return values[0]

    elif route == "pop_shift":

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1), state]
            items.pop(0)
            np.add(1.0, 1.0, out=items[0])
            return values[0]

    elif route == "mapping_pop":

        def objective(values: TraceADArray) -> object:
            options = {"out": state}
            output = options.pop("out")
            np.add(1.0, 1.0, out=output)
            return values[0]

    elif route == "delete":

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1), state]
            del items[0]
            np.add(1.0, 1.0, out=items[0])
            return values[0]

    elif route == "inplace_extend":

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1)]
            alias = items
            items += [state]
            np.add(1.0, 1.0, out=alias[-1])
            return values[0]

    elif route == "remove":

        def objective(values: TraceADArray) -> object:
            items: list[object] = [None, state]
            items.remove(None)
            np.add(1.0, 1.0, out=cast("NDArray[np.float64]", items[0]))
            return values[0]

    elif route == "destructure":

        def objective(values: TraceADArray) -> object:
            (output,) = [state]
            np.add(1.0, 1.0, out=output)
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1)]
            items[:] = [state]
            np.add(1.0, 1.0, out=items[0])
            return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured_mutation.*line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("route", ["clear", "pop_then_replace", "destructure", "slice"])
def test_structural_local_storage_preserves_primal_derivative_and_replay(route: str) -> None:
    """Owned structural updates preserve the supported native output domain.

    Parameters
    ----------
    route
        Structural update that leaves a known objective-owned output array.

    """
    state = np.array([7.0])
    if route == "clear":

        def objective(values: TraceADArray) -> object:
            items = [state]
            items.clear()
            items.append(np.zeros(1))
            np.add(1.0, 2.0, out=items[0])
            return values[0] * items[0][0]

    elif route == "pop_then_replace":

        def objective(values: TraceADArray) -> object:
            items = [state, np.zeros(1)]
            items.pop(0)
            np.add(1.0, 2.0, out=items[0])
            return values[0] * items[0][0]

    elif route == "destructure":

        def objective(values: TraceADArray) -> object:
            (output,) = [np.zeros(1)]
            np.add(1.0, 2.0, out=output)
            return values[0] * output[0]

    else:

        def objective(values: TraceADArray) -> object:
            items = [state]
            items[:] = [np.zeros(1)]
            np.add(1.0, 2.0, out=items[0])
            return values[0] * items[0][0]

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("mutable", [True, False])
def test_sequence_augmentation_preserves_mutable_and_immutable_aliases(mutable: bool) -> None:
    """List extension shares storage while tuple concatenation creates new storage.

    Parameters
    ----------
    mutable
        Whether the augmented sequence is a mutable list or immutable tuple.

    """
    state = np.array([7.0])
    if mutable:

        def objective(values: TraceADArray) -> object:
            items = [np.zeros(1)]
            alias = items
            items += [np.zeros(1)]
            np.add(1.0, 2.0, out=alias[-1])
            return values[0] * alias[-1][0]

    else:

        def objective(values: TraceADArray) -> object:
            items: tuple[NDArray[np.float64], ...] = (np.zeros(1),)
            alias = items
            items += (state,)
            np.add(1.0, 2.0, out=alias[0])
            return values[0] * alias[0][0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])


def test_nested_augmented_array_write_refuses_before_external_buffer_changes() -> None:
    """Augmenting a local container's borrowed array still writes external storage."""
    state = np.array([7.0])

    def objective(values: TraceADArray) -> object:
        options = {"out": state}
        options["out"] += 1.0
        return values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured_mutation.*line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


def test_sort_key_callback_refuses_before_writing_external_ledger() -> None:
    """A local list cannot conceal the external effects of its sort callback."""
    ledger: list[float] = []

    def key(value: float) -> float:
        ledger.append(value)
        return value

    def objective(values: TraceADArray) -> object:
        items = [2.0, 1.0]
        items.sort(key=key)
        return values[0] * items[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="external_callback.*line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert ledger == []
    assert active_reserved_bytes() == baseline


def test_sort_with_frozen_pure_key_preserves_analytic_replay() -> None:
    """A frozen builtin sort key preserves the original objective-owned domain."""

    def objective(values: TraceADArray) -> object:
        items = [-2.0, 1.0]
        items.sort(key=abs)
        return values[0] * items[0]

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 3.0
    np.testing.assert_array_equal(result.gradient, [1.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [1.0])


@pytest.mark.parametrize("route", ["loop", "comprehension", "expanded_helper"])
def test_mapping_iteration_uses_keys_without_borrowing_value_dependence(route: str) -> None:
    """Mapping keys remain passive even when their values depend on active inputs.

    Parameters
    ----------
    route
        Direct iteration, comprehension or positional expansion of mapping keys.

    """
    if route == "loop":

        def objective(values: TraceADArray) -> object:
            options = {"1": values[0], "2": values[0] * 2.0}
            scale = 0
            for name in options:
                scale += int(name)
            return values[0] * scale

    elif route == "comprehension":

        def objective(values: TraceADArray) -> object:
            options = {"1": values[0], "2": values[0] * 2.0}
            scale = sum([int(name) for name in options])
            return values[0] * scale

    else:

        def coefficient(first: str, second: str) -> float:
            return float(int(first) + int(second))

        def objective(values: TraceADArray) -> object:
            options = {"1": values[0], "2": values[0] * 2.0}
            return values[0] * coefficient(*options)

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])


@pytest.mark.parametrize("captured", [True, False])
def test_mapping_pop_expanded_arity_preserves_output_storage(captured: bool) -> None:
    """Expanded optional pop defaults preserve owned and borrowed output identity.

    Parameters
    ----------
    captured
        Whether the selected dictionary buffer belongs to the caller.

    """
    state = np.array([7.0])
    if captured:

        def objective(values: TraceADArray) -> object:
            options = {"out": state}
            arguments = ("out",) if values[0] > 0 else ("out", np.zeros(1))
            output = options.pop(*arguments)
            np.add(1.0, 2.0, out=output)
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            options = {"out": np.zeros(1)}
            arguments = ("out",) if values[0] > 0 else ("out", np.zeros(1))
            output = options.pop(*arguments)
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


@pytest.mark.parametrize(
    "route", ["insert", "pop", "mapping_pop", "slice", "delete", "remove", "mapping_default"]
)
@pytest.mark.parametrize("captured", [True, False])
def test_unknown_structural_selection_retains_possible_output_buffers(
    route: str, captured: bool
) -> None:
    """Conditional selection preserves borrowed refusal and owned analytic replay.

    Parameters
    ----------
    route
        Structural operation whose position, key or bound depends on an active branch.
    captured
        Whether a possible selected native output belongs to the caller.

    """
    state = np.array([7.0])
    if captured:

        def obtain_buffer() -> NDArray[np.float64]:
            return state

    else:

        def obtain_buffer() -> NDArray[np.float64]:
            return np.zeros(1)

    if route == "insert":

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            items = [np.zeros(1)]
            position = 0 if values[0] > 0 else 1
            items.insert(position, buffer)
            np.add(1.0, 2.0, out=items[0])
            return values[0] * items[0][0]

    elif route == "pop":

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            items = [buffer, np.zeros(1)]
            position = 0 if values[0] > 0 else 1
            output = items.pop(position)
            np.add(1.0, 2.0, out=output)
            return values[0] * output[0]

    elif route == "mapping_pop":

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            options = {"out": buffer, "safe": np.zeros(1)}
            key = "out" if values[0] > 0 else "safe"
            output = options.pop(key)
            np.add(1.0, 2.0, out=output)
            return values[0] * output[0]

    elif route == "slice":

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            items = [np.zeros(1), np.zeros(1)]
            position = 0 if values[0] > 0 else 1
            items[position:] = [buffer]
            np.add(1.0, 2.0, out=items[0])
            return values[0] * items[0][0]

    elif route == "delete":

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            items = [np.zeros(1), buffer]
            position = 0 if values[0] > 0 else 1
            del items[position]
            np.add(1.0, 2.0, out=items[0])
            return values[0] * items[0][0]

    elif route == "remove":

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            items: list[object] = [None, buffer]
            selected = None if values[0] > 0 else 1
            items.remove(selected)
            output = cast("NDArray[np.float64]", items[0])
            np.add(1.0, 2.0, out=output)
            return values[0] * output[0]

    else:

        def objective(values: TraceADArray) -> object:
            buffer = obtain_buffer()
            options = {"safe": np.zeros(1)}
            key = "missing" if values[0] > 0 else "safe"
            output = options.pop(key, buffer)
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


@pytest.mark.parametrize("captured", [True, False])
def test_integer_mapping_keys_use_key_identity_instead_of_element_position(captured: bool) -> None:
    """Integer mapping lookup preserves the buffer bound to that key.

    Parameters
    ----------
    captured
        Whether key zero selects a borrowed buffer rather than an owned one.

    """
    state = np.array([7.0])
    if captured:

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: state}
            np.add(1.0, 2.0, out=options[0])
            return values[0]

    else:

        def objective(values: TraceADArray) -> object:
            options = {1: state, 0: np.zeros(1)}
            np.add(1.0, 2.0, out=options[0])
            return values[0] * options[0][0]

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


def test_integer_mapping_iteration_keeps_passive_keys_with_active_values() -> None:
    """Integer keys are independent of their dictionary's active values."""

    def objective(values: TraceADArray) -> object:
        options = {1: values[0], 2: values[0] * 2.0}
        scale = 0
        for key in options:
            scale += int(key)
        return values[0] * scale

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])


def test_captured_mapping_expansion_keeps_passive_native_keys() -> None:
    """A captured plain mapping expands its passive keys without transferring buffers."""
    options = {"1": np.array([7.0]), "2": np.array([8.0])}

    def coefficient(first: str, second: str) -> float:
        return float(int(first) + int(second))

    def objective(values: TraceADArray) -> object:
        return values[0] * coefficient(*options)

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(options["1"], [7.0])
    np.testing.assert_array_equal(options["2"], [8.0])


@pytest.mark.parametrize(
    "route", ["get", "assignment", "update", "pairs", "pop", "copy", "expand"]
)
@pytest.mark.parametrize("captured", [True, False])
def test_integer_mapping_operations_preserve_selected_storage(route: str, captured: bool) -> None:
    """Dictionary operations preserve integer-key buffer identity before writes.

    Parameters
    ----------
    route
        Lookup, replacement, update, removal, copy or dictionary expansion.
    captured
        Whether the selected buffer belongs to the caller.

    """
    state = np.array([7.0])
    if captured:

        def obtain_buffer() -> NDArray[np.float64]:
            return state

    else:

        def obtain_buffer() -> NDArray[np.float64]:
            return np.zeros(1)

    if route == "get":

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: obtain_buffer()}
            output = options.get(0)
            np.add(1.0, 2.0, out=output)
            return values[0] * options[0][0]

    elif route == "assignment":

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: np.zeros(1)}
            options[0] = obtain_buffer()
            np.add(1.0, 2.0, out=options[0])
            return values[0] * options[0][0]

    elif route == "update":

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: np.zeros(1)}
            options.update({0: obtain_buffer()})
            np.add(1.0, 2.0, out=options[0])
            return values[0] * options[0][0]

    elif route == "pairs":

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: np.zeros(1)}
            options.update([(0, obtain_buffer())])
            np.add(1.0, 2.0, out=options[0])
            return values[0] * options[0][0]

    elif route == "pop":

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: obtain_buffer()}
            output = options.pop(0)
            np.add(1.0, 2.0, out=output)
            return values[0] * output[0]

    elif route == "copy":

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: obtain_buffer()}
            copied = dict(options)
            np.add(1.0, 2.0, out=copied[0])
            return values[0] * copied[0][0]

    else:

        def objective(values: TraceADArray) -> object:
            options = {1: np.zeros(1), 0: obtain_buffer()}
            copied = {**options}
            np.add(1.0, 2.0, out=copied[0])
            return values[0] * copied[0][0]

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


@pytest.mark.parametrize("route", ["list", "tuple"])
def test_mapping_sequence_constructors_iterate_keys(route: str) -> None:
    """Sequence construction from a dictionary keeps keys independent of active values.

    Parameters
    ----------
    route
        Mutable or immutable sequence constructor.

    """
    if route == "list":

        def objective(values: TraceADArray) -> object:
            options = {1: values[0], 2: values[0] * 2.0}
            keys = list(options)
            return values[0] * (int(keys[0]) + int(keys[1]))

    else:

        def objective(values: TraceADArray) -> object:
            options = {1: values[0], 2: values[0] * 2.0}
            keys = tuple(options)
            return values[0] * (int(keys[0]) + int(keys[1]))

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])


def test_integer_mapping_keyword_expansion_refuses_before_callback_execution() -> None:
    """Non-string dictionary keys remain invalid for actual callback keyword binding."""
    state = np.array([7.0])

    def callback(**kwargs: object) -> float:
        state[0] = 9.0
        return 3.0

    def objective(values: TraceADArray) -> object:
        return values[0] * callback(**cast(dict[str, object], {0: 3.0}))

    with pytest.raises(ValueError, match="external_callback.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])


@pytest.mark.parametrize("captured", [True, False])
def test_integer_mapping_deletion_retains_the_other_keys_buffer(captured: bool) -> None:
    """Deleting an integer key leaves the other key's buffer provenance intact.

    Parameters
    ----------
    captured
        Whether the remaining buffer belongs to the caller.

    """
    state = np.array([7.0])
    if captured:

        def obtain_buffer() -> NDArray[np.float64]:
            return state

    else:

        def obtain_buffer() -> NDArray[np.float64]:
            return np.zeros(1)

    def objective(values: TraceADArray) -> object:
        options = {1: np.zeros(1), 0: obtain_buffer()}
        del options[1]
        np.add(1.0, 2.0, out=options[0])
        return values[0] * options[0][0]

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
