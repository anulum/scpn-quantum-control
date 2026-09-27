# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD array indexing module
"""Public padding/insertion derivative admission and independent numerical oracles."""

import sys
from threading import Event
from types import FrameType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import ArrayLike

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import primitive_contract_for
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_delete_derivative_rule,
    program_ad_array_getitem_derivative_rule,
    program_ad_array_insert_derivative_rule,
    program_ad_array_pad_derivative_rule,
    program_ad_array_take_along_axis_derivative_rule,
    program_ad_array_take_derivative_rule,
)


def test_public_pad_rule_refuses_large_layout_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real environment cap refuses layout buffers before large constant padding."""
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            program_ad_array_pad_derivative_rule((2,), (10_000_000, 10_000_000))
        assert active_reserved_bytes() == baseline
        rule = program_ad_array_pad_derivative_rule((2,), (1, 2), constant_values=7.0)
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        values = np.array([2.0, 3.0])
        np.testing.assert_array_equal(rule.value_fn(values), np.array([7.0, 2.0, 3.0, 7.0, 7.0]))
        np.testing.assert_array_equal(
            rule.jvp_rule(values, np.array([4.0, 5.0])), np.array([0.0, 4.0, 5.0, 0.0, 0.0])
        )
        np.testing.assert_array_equal(
            rule.vjp_rule(values, np.arange(1.0, 6.0)), np.array([2.0, 3.0])
        )
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("selector", [1, slice(None)])
def test_public_insert_rule_refuses_layout_and_recovers(
    selector: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source-sized insertion layout is refused before its arange/zero buffers exist."""
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            program_ad_array_insert_derivative_rule((10_000_000,), selector, 7.0)
        assert active_reserved_bytes() == baseline
        rule = program_ad_array_insert_derivative_rule((2,), 1, 7.0)
        np.testing.assert_array_equal(
            rule.value_fn(np.array([2.0, 3.0])), np.array([2.0, 7.0, 3.0])
        )
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "selector,expected",
    [
        (1, [1.0, 7.0, 2.0, 3.0, 8.0, 4.0]),
        ((1,), [1.0, 7.0, 8.0, 2.0, 3.0, 7.0, 8.0, 4.0]),
    ],
)
def test_public_insert_rule_preserves_scalar_and_vector_axis_semantics(
    selector: object, expected: list[float]
) -> None:
    """Scalar and length-one index vectors keep NumPy's different column broadcasting."""
    rule = program_ad_array_insert_derivative_rule((2, 2), selector, (7.0, 8.0), axis=1)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(rule.value_fn(values), np.array(expected))
    assert rule.vjp_rule is not None
    np.testing.assert_array_equal(rule.vjp_rule(values, np.ones(len(expected))), np.ones(4))


@pytest.mark.parametrize("operation", ["pad", "insert"])
@pytest.mark.parametrize("transform", ["value", "jvp", "vjp"])
def test_public_direct_rule_readmits_execution_after_budget_change(
    operation: str, transform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached layout does not bypass the live budget on any direct transform."""
    rule = (
        program_ad_array_pad_derivative_rule((2,), (1, 2), constant_values=7.0)
        if operation == "pad"
        else program_ad_array_insert_derivative_rule((2,), 1, (7.0, 8.0, 9.0))
    )
    values = np.array([2.0, 3.0])
    tangent = np.array([4.0, 5.0])
    cotangent = np.arange(1.0, 6.0)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None

    def callback() -> ArrayLike:
        if transform == "value":
            return rule.value_fn(values)
        if transform == "jvp":
            assert rule.jvp_rule is not None
            return rule.jvp_rule(values, tangent)
        assert rule.vjp_rule is not None
        return rule.vjp_rule(values, cotangent)

    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.000000001")
        with pytest.raises(DenseAllocationError):
            callback()
        assert active_reserved_bytes() == baseline
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        expected = {
            ("pad", "value"): [7.0, 2.0, 3.0, 7.0, 7.0],
            ("insert", "value"): [2.0, 7.0, 8.0, 9.0, 3.0],
            ("pad", "jvp"): [0.0, 4.0, 5.0, 0.0, 0.0],
            ("insert", "jvp"): [4.0, 0.0, 0.0, 0.0, 5.0],
            ("pad", "vjp"): [2.0, 3.0],
            ("insert", "vjp"): [1.0, 5.0],
        }[operation, transform]
        np.testing.assert_array_equal(callback(), np.array(expected))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("operation", ["pad", "insert"])
def test_public_direct_rule_bad_operand_releases_execution_charge(
    operation: str,
) -> None:
    """Malformed tangent/cotangent sizes do not leak the admitted transform scope."""
    rule = (
        program_ad_array_pad_derivative_rule((2,), (1, 2), constant_values=7.0)
        if operation == "pad"
        else program_ad_array_insert_derivative_rule((2,), 1, (7.0, 8.0, 9.0))
    )
    values = np.array([2.0, 3.0])
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="tangent with 2 values"):
        rule.jvp_rule(values, np.ones(3))
    assert active_reserved_bytes() == baseline
    with pytest.raises(ValueError, match="cotangent with 5 values"):
        rule.vjp_rule(values, np.ones(4))
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(rule.vjp_rule(values, np.ones(5)), np.ones(2))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("operation", ["pad", "insert"])
def test_public_direct_rule_observes_owner_cancellation_and_recovers(operation: str) -> None:
    """Direct callbacks inherit the live caller's cooperative cancellation scope."""
    rule = (
        program_ad_array_pad_derivative_rule((2,), (1, 2), constant_values=7.0)
        if operation == "pad"
        else program_ad_array_insert_derivative_rule((2,), 1, (7.0, 8.0, 9.0))
    )
    values = np.array([2.0, 3.0])
    cancelled = Event()
    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("caller_values", "forward", (2,), "float64"),))
    with reserve_execution_memory(plan, cancelled=cancelled):
        owned = active_reserved_bytes()
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            rule.value_fn(values)
        assert active_reserved_bytes() == owned
        cancelled.clear()
        expected = [7.0, 2.0, 3.0, 7.0, 7.0] if operation == "pad" else [2.0, 7.0, 8.0, 9.0, 3.0]
        np.testing.assert_array_equal(rule.value_fn(values), np.array(expected))
        assert active_reserved_bytes() == owned
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("along_axis", [False, True])
def test_public_take_rule_refuses_large_output_layout_and_recovers(
    along_axis: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real zero-stride index view cannot bypass output-size admission."""
    indices = np.broadcast_to(np.array(0, dtype=np.int64), (10_000_000,))
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            if along_axis:
                program_ad_array_take_along_axis_derivative_rule(
                    (1, 2), indices.reshape((-1, 1)), axis=1
                )
            else:
                program_ad_array_take_derivative_rule((2,), indices)
        assert active_reserved_bytes() == baseline
        rule = (
            program_ad_array_take_along_axis_derivative_rule((1, 2), ((1, 0, 1),), axis=1)
            if along_axis
            else program_ad_array_take_derivative_rule((2,), (1, 0, 1))
        )
        values = np.array([2.0, 3.0])
        np.testing.assert_array_equal(rule.value_fn(values), np.array([3.0, 2.0, 3.0]))
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        np.testing.assert_array_equal(rule.jvp_rule(values, np.array([4.0, 5.0])), [5.0, 4.0, 5.0])
        np.testing.assert_array_equal(rule.vjp_rule(values, np.array([1.0, 2.0, 3.0])), [2.0, 4.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("operation", ["getitem", "take", "take_along_axis", "delete"])
@pytest.mark.parametrize("transform", ["value", "jvp", "vjp"])
def test_public_gather_scatter_rule_readmits_transform_storage(
    operation: str,
    transform: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every existing direct gather/scatter facade observes a new live cap."""
    if operation == "getitem":
        rule = program_ad_array_getitem_derivative_rule((2,), [1, 0, 1])
    elif operation == "take":
        rule = program_ad_array_take_derivative_rule((2,), (1, 0, 1))
    elif operation == "take_along_axis":
        rule = program_ad_array_take_along_axis_derivative_rule((1, 2), ((1, 0, 1),), axis=-1)
    else:
        rule = program_ad_array_delete_derivative_rule((4,), 1)
    values = np.array([2.0, 3.0, 4.0, 5.0] if operation == "delete" else [2.0, 3.0])
    tangent = np.array([6.0, 7.0, 8.0, 9.0] if operation == "delete" else [4.0, 5.0])
    cotangent = np.array([1.0, 2.0, 3.0])

    def callback() -> ArrayLike:
        if transform == "value":
            return rule.value_fn(values)
        if transform == "jvp":
            assert rule.jvp_rule is not None
            return rule.jvp_rule(values, tangent)
        assert rule.vjp_rule is not None
        return rule.vjp_rule(values, cotangent)

    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.000000001")
        with pytest.raises(DenseAllocationError):
            callback()
        assert active_reserved_bytes() == baseline
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        expected = {
            "value": [2.0, 4.0, 5.0] if operation == "delete" else [3.0, 2.0, 3.0],
            "jvp": [6.0, 8.0, 9.0] if operation == "delete" else [5.0, 4.0, 5.0],
            "vjp": [1.0, 0.0, 2.0, 3.0] if operation == "delete" else [2.0, 4.0],
        }[transform]
        np.testing.assert_array_equal(callback(), np.array(expected))
    assert active_reserved_bytes() == baseline


def test_public_take_along_empty_output_admits_coordinate_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty output still declares NumPy's non-axis arange coordinate workspace."""
    indices = np.empty((0, 1), dtype=np.int64)
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            program_ad_array_take_along_axis_derivative_rule((0, 10_000_000), indices, axis=0)
        assert active_reserved_bytes() == baseline
        rule = program_ad_array_take_along_axis_derivative_rule((0, 3), indices, axis=0)
        np.testing.assert_array_equal(rule.value_fn(np.empty(0)), np.empty(0))
    assert active_reserved_bytes() == baseline


def test_public_take_along_rank_error_preserves_ledger_and_retry() -> None:
    """Incompatible static rank fails before layout without poisoning another rule."""
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="shape-compatible with source rank"):
        program_ad_array_take_along_axis_derivative_rule((1, 2), (0, 1), axis=1)
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_take_along_axis_derivative_rule((1, 2), ((1, 0),), axis=1)
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0, 3.0])), np.array([3.0, 2.0]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("selector_kind", ["scalar", "slice", "boolean"])
def test_public_delete_layout_refuses_large_source_and_recovers(
    selector_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source/output/mask workspace admission precedes actual deletion layout."""
    selector = (
        1
        if selector_kind == "scalar"
        else slice(1, None, 2)
        if selector_kind == "slice"
        else np.broadcast_to(np.array(False), (10_000_000,))
    )
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            program_ad_array_delete_derivative_rule((10_000_000,), selector)
        assert active_reserved_bytes() == baseline
        selector = (
            1
            if selector_kind == "scalar"
            else slice(1, None, 2)
            if selector_kind == "slice"
            else np.array([False, True, False, True])
        )
        rule = program_ad_array_delete_derivative_rule((4,), selector)
        values = np.array([2.0, 3.0, 4.0, 5.0])
        expected = [2.0, 4.0, 5.0] if selector_kind == "scalar" else [2.0, 4.0]
        np.testing.assert_array_equal(rule.value_fn(values), expected)
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        np.testing.assert_array_equal(rule.jvp_rule(values, values), expected)
        np.testing.assert_array_equal(
            rule.vjp_rule(values, np.ones(len(expected))),
            [1.0, 0.0, 1.0, 1.0] if selector_kind == "scalar" else [1.0, 0.0, 1.0, 0.0],
        )
    assert active_reserved_bytes() == baseline


def test_public_delete_bad_boolean_mask_releases_layout_and_recovers() -> None:
    """A real NumPy mask-size error disposes its layout scope before retry."""
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="in-bounds deletion selectors"):
        program_ad_array_delete_derivative_rule((4,), np.array([True]))
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_delete_derivative_rule((4,), np.array([False, True, False, True]))
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0, 3.0, 4.0, 5.0])), [2.0, 4.0])
    assert active_reserved_bytes() == baseline


def test_public_delete_single_integer_array_does_not_require_large_axis_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy's single-index optimization preserves a tiny empty-source deletion."""
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        rule = program_ad_array_delete_derivative_rule((0, 10_000_000), np.array([1]), axis=1)
        np.testing.assert_array_equal(rule.value_fn(np.empty(0)), np.empty(0))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "selector,expected",
    [
        ((slice(None), Ellipsis, slice(1, None, 2)), [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
        (
            (np.array([[0], [1]]), slice(None), np.array([1, 3])),
            [1, 5, 9, 3, 7, 11, 13, 17, 21, 15, 19, 23],
        ),
        ((slice(None), np.array([[0], [2]]), np.array([1, 3])), [1, 3, 9, 11, 13, 15, 21, 23]),
        (np.array([[True, False, False], [False, True, False]]), [0, 1, 2, 3, 16, 17, 18, 19]),
        ((None, Ellipsis, np.array([1, 3])), [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
        (np.array(1), list(range(12, 24))),
        ((1, 2, 3), [23]),
        ((slice(0, 0),), []),
    ],
)
def test_public_getitem_layout_preserves_basic_and_advanced_oracles(
    selector: object,
    expected: list[int],
) -> None:
    """Hand-enumerated flat slots validate selection order and scatter multiplicity."""
    rule = program_ad_array_getitem_derivative_rule((2, 3, 4), selector)
    values = np.arange(24.0)
    np.testing.assert_array_equal(rule.value_fn(values), np.array(expected, dtype=np.float64))
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    np.testing.assert_array_equal(rule.jvp_rule(values, values + 10.0), np.array(expected) + 10.0)
    expected_gradient = np.zeros(24)
    for slot in expected:
        expected_gradient[slot] += 1.0
    np.testing.assert_array_equal(rule.vjp_rule(values, np.ones(len(expected))), expected_gradient)


@pytest.mark.parametrize("advanced", [False, True])
def test_public_getitem_layout_refuses_oversize_before_source_or_selector_copy(
    advanced: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real large source metadata and zero-stride selector views refuse before copying."""
    selector = (
        np.broadcast_to(np.array(0, dtype=np.int64), (10_000_000,)) if advanced else slice(None)
    )
    source_shape = (2,) if advanced else (10_000_000,)
    baseline = active_reserved_bytes()
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            program_ad_array_getitem_derivative_rule(source_shape, selector)
        assert active_reserved_bytes() == baseline
        rule = program_ad_array_getitem_derivative_rule((2,), np.array([1, 0, 1]))
        np.testing.assert_array_equal(rule.value_fn(np.array([2.0, 3.0])), [3.0, 2.0, 3.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "selector",
    [
        (Ellipsis, Ellipsis),
        (0, 0, 0),
        np.array([True, False, True]),
        (np.array([0, 1]), np.array([0, 1, 2])),
    ],
)
def test_public_getitem_invalid_metadata_releases_snapshot_and_recovers(selector: object) -> None:
    """Bad rank, ellipsis, mask and broadcast cases leave no snapshot charge."""
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="requires in-bounds indices"):
        program_ad_array_getitem_derivative_rule((2, 2), selector)
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_getitem_derivative_rule((2, 2), (1, slice(None)))
    np.testing.assert_array_equal(rule.value_fn(np.arange(4.0)), [2.0, 3.0])
    assert active_reserved_bytes() == baseline


def test_public_getitem_snapshot_remains_charged_during_real_boolean_count() -> None:
    """A production NumPy profile boundary observes ownership and caller-mask mutation."""
    mask = np.array([False, True, False, True, False])
    observations: list[int] = []
    baseline = active_reserved_bytes()

    def profile(frame: FrameType, event: str, argument: object) -> None:
        del argument
        if (
            event == "call"
            and frame.f_code.co_name == "count_nonzero"
            and str(frame.f_globals.get("__name__", "")).startswith("numpy.")
        ):
            observations.append(active_reserved_bytes())
            mask[:] = True

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        rule = program_ad_array_getitem_derivative_rule((5,), mask)
    finally:
        sys.setprofile(previous_profile)
    assert len(observations) == 1
    assert observations[0] - baseline >= mask.nbytes
    np.testing.assert_array_equal(mask, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(rule.value_fn(np.arange(5.0)), np.array([1.0, 3.0]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "operation", ["getitem", "take", "take_along_axis", "delete", "pad", "insert"]
)
@pytest.mark.parametrize(
    "source_shape",
    [(sys.maxsize + 1,), (sys.maxsize // 8 + 1,), (0, sys.maxsize + 1), (1 << 32, 1 << 32)],
)
def test_public_index_rule_refuses_unaddressable_dimensions_before_layout(
    operation: str,
    source_shape: tuple[int, ...],
) -> None:
    """All public factories reject dimension and int64-product overflow without allocation."""
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError, match="native addressability"):
        if operation == "getitem":
            program_ad_array_getitem_derivative_rule(source_shape, slice(None))
        elif operation == "take":
            program_ad_array_take_derivative_rule(source_shape, (0,))
        elif operation == "take_along_axis":
            program_ad_array_take_along_axis_derivative_rule(source_shape, (0,), axis=0)
        elif operation == "delete":
            program_ad_array_delete_derivative_rule(source_shape, 0)
        elif operation == "pad":
            program_ad_array_pad_derivative_rule(source_shape, 0)
        else:
            program_ad_array_insert_derivative_rule(source_shape, 0, 7.0)
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_getitem_derivative_rule((2,), slice(None))
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0, 3.0])), [2.0, 3.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "source_shape,selector,values", [((), (), [7.0]), ((0, 3), slice(None), [])]
)
def test_public_getitem_native_size_preserves_scalar_and_zero_extent(
    source_shape: tuple[int, ...],
    selector: object,
    values: list[float],
) -> None:
    """Native bounds retain rank-zero and ordinary zero-extent source semantics."""
    baseline = active_reserved_bytes()
    rule = program_ad_array_getitem_derivative_rule(source_shape, selector)
    np.testing.assert_array_equal(rule.value_fn(np.array(values)), np.array(values))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", ["direct", "contract"])
def test_public_take_along_axis_rejects_actual_out_of_bounds_selection(surface: str) -> None:
    """Both public owners refuse NumPy selection faults and release all declarations."""
    baseline = active_reserved_bytes()
    if surface == "direct":
        with pytest.raises(ValueError, match="in-bounds"):
            program_ad_array_take_along_axis_derivative_rule((1, 2), ((5,),), axis=1)
    else:
        contract = primitive_contract_for("scpn.program_ad.array:take_along_axis")
        assert contract.shape_rule is not None
        with pytest.raises(ValueError, match="in bounds"):
            contract.shape_rule((np.zeros((1, 2)), np.array([[5]]), 1))
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_take_along_axis_derivative_rule((1, 2), ((1,),), axis=1)
    np.testing.assert_array_equal(rule.value_fn(np.array([3.0, 7.0])), [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "shape,selector,values,axis,diagnostic",
    [
        ((2,), [[1]], 7.0, None, "one-dimensional"),
        ((2,), 99, 7.0, None, "in-bounds"),
        ((1, 2), (0, 1), np.zeros((3, 2)), 1, "incompatible"),
        ((2,), (99, 100), 7.0, None, "compatible with the source"),
    ],
)
def test_public_insert_rejects_malformed_selection_and_broadcast(
    shape: tuple[int, ...], selector: object, values: object, axis: int | None, diagnostic: str
) -> None:
    """Real invalid selectors and broadcast metadata never retain layout charges."""
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match=diagnostic):
        program_ad_array_insert_derivative_rule(shape, selector, values, axis=axis)
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_insert_derivative_rule((2,), 1, 7.0)
    np.testing.assert_array_equal(rule.value_fn(np.array([3.0, 5.0])), [3.0, 7.0, 5.0])
    assert active_reserved_bytes() == baseline


def test_public_rule_rejects_wrong_sized_array_protocol() -> None:
    """Post-conversion size validation also covers genuine ndarray subclasses."""
    baseline = active_reserved_bytes()

    class ArraySubclass(np.ndarray[Any, Any]):
        """Exercise public array conversion without the exact ndarray fast path."""

        pass

    rule = program_ad_array_getitem_derivative_rule((2,), slice(None))
    with pytest.raises(ValueError, match="2 values"):
        rule.value_fn(np.arange(3.0).view(ArraySubclass))
    np.testing.assert_array_equal(rule.value_fn(np.array([3.0, 5.0])), [3.0, 5.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("surface", ["direct", "contract"])
def test_public_getitem_refuses_tampered_admitted_selector(surface: str) -> None:
    """A corrupted selector cannot publish output larger than the admitted snapshot."""
    baseline = active_reserved_bytes()
    mask = np.array([False, True, False])
    tampered: list[int] = []

    def profile(frame: FrameType, event: str, argument: object) -> None:
        if (
            event == "return"
            and frame.f_code.co_name == "_program_ad_array_getitem_layout_plan"
            and argument is not None
        ):
            _, size, prepared = cast(
                tuple[ExecutionMemoryPlan, int, np.ndarray[Any, Any]], argument
            )
            prepared.setflags(write=True)
            prepared[:] = True
            tampered.append(size)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="in-bounds"):
            if surface == "direct":
                program_ad_array_getitem_derivative_rule((3,), mask)
            else:
                contract = primitive_contract_for("scpn.program_ad.array:getitem")
                assert contract.shape_rule is not None
                contract.shape_rule((np.zeros(3), mask))
    finally:
        sys.setprofile(previous)
    assert tampered == [1]
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_getitem_derivative_rule((3,), mask)
    np.testing.assert_array_equal(rule.value_fn(np.array([3.0, 5.0, 7.0])), [5.0])
    assert active_reserved_bytes() == baseline


def test_public_pad_refuses_constants_changed_during_real_layout() -> None:
    """Malformed constant metadata at the NumPy boundary fails without retaining charges."""
    constants = np.array([7.0, 8.0])
    baseline = active_reserved_bytes()
    observed: list[int] = []

    def profile(frame: FrameType, event: str, argument: object) -> None:
        del argument
        if event == "call" and frame.f_code.co_name == "pad" and not observed:
            observed.append(active_reserved_bytes())
            constants.resize((3,), refcheck=False)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="compatible with the source rank"):
            program_ad_array_pad_derivative_rule((2,), (1, 1), constant_values=constants)
    finally:
        sys.setprofile(previous)
    assert observed and observed[0] > baseline
    assert active_reserved_bytes() == baseline
    rule = program_ad_array_pad_derivative_rule((2,), (1, 1), constant_values=(7.0, 8.0))
    np.testing.assert_array_equal(rule.value_fn(np.array([3.0, 5.0])), [7.0, 3.0, 5.0, 8.0])
    assert active_reserved_bytes() == baseline
