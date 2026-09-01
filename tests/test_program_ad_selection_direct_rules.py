# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD selection direct rules tests
# scpn-quantum-control -- Program AD selection direct derivative rules
"""Tests for Program AD selection direct derivative-rule factories."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.program_ad_selection_primitives as selection_primitives
from scpn_quantum_control.differentiable import (
    PrimitiveIdentity,
    custom_derivative_rule_for,
    program_ad_selection_clip_derivative_rule,
    program_ad_selection_where_derivative_rule,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy closeness across dynamically typed direct-rule payloads."""
    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_program_ad_selection_factories_remain_facade_compatible() -> None:
    """Selection factories should re-export the extracted module implementations."""
    assert (
        program_ad_selection_where_derivative_rule
        is selection_primitives.program_ad_selection_where_derivative_rule
    )
    assert (
        program_ad_selection_clip_derivative_rule
        is selection_primitives.program_ad_selection_clip_derivative_rule
    )


def test_program_ad_selection_static_derivative_factories() -> None:
    """Static where and clip factories should expose exact branch/clip adjoints."""
    where_rule = program_ad_selection_where_derivative_rule(
        np.array([True, False, True]), (3,), ()
    )
    assert where_rule.name == "program_ad_selection_where_3_by_scalar_static_direct_rule"
    assert where_rule.jvp_rule is not None
    assert where_rule.vjp_rule is not None
    where_jvp_rule = where_rule.jvp_rule
    where_vjp_rule = where_rule.vjp_rule
    where_values = np.array([1.0, -2.0, 0.5, 0.25], dtype=np.float64)
    where_tangent = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    where_cotangent = np.array([1.5, -2.0, 0.75], dtype=np.float64)
    _assert_allclose(where_rule.value_fn(where_values), [1.0, 0.25, 0.5])
    _assert_allclose(where_jvp_rule(where_values, where_tangent), [0.1, 0.4, 0.3])
    _assert_allclose(
        where_vjp_rule(where_values, where_cotangent),
        [1.5, 0.0, 0.75, -2.0],
    )

    clip_rule = program_ad_selection_clip_derivative_rule((3,), lower_shape=(), upper_shape=(3,))
    assert clip_rule.name == "program_ad_selection_clip_3_bounds_scalar_by_3_direct_rule"
    assert clip_rule.jvp_rule is not None
    assert clip_rule.vjp_rule is not None
    clip_jvp_rule = clip_rule.jvp_rule
    clip_vjp_rule = clip_rule.vjp_rule
    clip_values = np.array([-2.0, 0.25, 2.0, -1.0, 1.0, 1.0, 1.5], dtype=np.float64)
    clip_tangent = np.array([0.2, -0.3, 0.5, 0.75, 0.1, 0.2, 0.3], dtype=np.float64)
    clip_cotangent = np.array([1.5, -2.0, 0.75], dtype=np.float64)
    _assert_allclose(clip_rule.value_fn(clip_values), [-1.0, 0.25, 1.5])
    _assert_allclose(clip_jvp_rule(clip_values, clip_tangent), [0.75, -0.3, 0.3])
    _assert_allclose(
        clip_vjp_rule(clip_values, clip_cotangent),
        [0.0, -2.0, 0.0, 1.5, 0.0, 0.0, 0.75],
    )

    with pytest.raises(ValueError, match="clipping boundary"):
        clip_jvp_rule(
            np.array([-1.0, 0.25, 2.0, -1.0, 1.0, 1.0, 1.5], dtype=np.float64),
            clip_tangent,
        )


def test_program_ad_selection_where_factory_covers_broadcast_adjoint_contracts() -> None:
    """Where factories should reduce broadcast adjoints back to each branch shape."""
    condition = np.array([[True, False, True], [False, True, False]])
    rule = program_ad_selection_where_derivative_rule(condition, (2, 1), (1, 3))
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    values = np.array([1.0, 2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, 0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    cotangent = np.arange(1.0, 7.0, dtype=np.float64)

    _assert_allclose(rule.value_fn(values), [1.0, 0.5, 1.0, -1.0, 2.0, 3.0])
    _assert_allclose(rule.jvp_rule(values, tangent), [0.1, -0.4, 0.1, 0.3, 0.2, 0.5])
    _assert_allclose(rule.vjp_rule(values, cotangent), [4.0, 5.0, 4.0, 2.0, 6.0])

    rank_broadcast_rule = program_ad_selection_where_derivative_rule(
        np.ones((2, 3), dtype=np.bool_),
        (1,),
        (2, 3),
    )
    assert rank_broadcast_rule.vjp_rule is not None
    _assert_allclose(
        rank_broadcast_rule.vjp_rule(
            np.arange(7.0, dtype=np.float64),
            cotangent,
        ),
        [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    ("condition", "true_shape", "false_shape", "message"),
    [
        (True, (-1,), (), "non-negative dimensions"),
        (True, (2,), (3,), "broadcast-compatible"),
        (np.array([1, 0]), (2,), (2,), "boolean condition"),
        (np.array([True]), (2,), (2,), "scalar or output-shaped"),
    ],
)
def test_program_ad_selection_where_factory_rejects_invalid_static_signatures(
    condition: object,
    true_shape: tuple[int, ...],
    false_shape: tuple[int, ...],
    message: str,
) -> None:
    """Where factories should reject malformed shape and predicate signatures."""
    with pytest.raises(ValueError, match=message):
        program_ad_selection_where_derivative_rule(condition, true_shape, false_shape)


def test_program_ad_selection_where_factory_rejects_malformed_flat_payloads() -> None:
    """Where direct rules should reject malformed values, tangents, and cotangents."""
    rule = program_ad_selection_where_derivative_rule(np.array([True, False]), (2,), ())
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    valid_values = np.array([1.0, 2.0, -1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="flattened true branch"):
        rule.value_fn(np.array([1.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="real numeric scalars"):
        rule.value_fn(np.array([1.0, 2.0, 3.0j]))
    with pytest.raises(ValueError, match="flattened true branch"):
        rule.jvp_rule(valid_values, np.array([0.1, 0.2], dtype=np.float64))
    with pytest.raises(ValueError, match="cotangent shape"):
        rule.vjp_rule(valid_values, np.array([1.0], dtype=np.float64))


@pytest.mark.parametrize(
    ("source_shape", "lower_shape", "upper_shape", "message"),
    [
        ((-1,), (), (), "non-negative dimensions"),
        ((2, 2), (3,), (), "bounds broadcastable"),
        ((1,), (2,), (), "bounds broadcastable"),
    ],
)
def test_program_ad_selection_clip_factory_rejects_invalid_static_signatures(
    source_shape: tuple[int, ...],
    lower_shape: tuple[int, ...],
    upper_shape: tuple[int, ...],
    message: str,
) -> None:
    """Clip factories should reject negative or non-source-preserving shapes."""
    with pytest.raises(ValueError, match=message):
        program_ad_selection_clip_derivative_rule(
            source_shape,
            lower_shape=lower_shape,
            upper_shape=upper_shape,
        )


def test_program_ad_selection_clip_factory_rejects_malformed_payloads_and_domains() -> None:
    """Clip rules should fail closed on malformed vectors, bounds, and derivative kinks."""
    rule = program_ad_selection_clip_derivative_rule((2,), lower_shape=(), upper_shape=())
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    valid_values = np.array([-0.5, 0.5, -1.0, 1.0], dtype=np.float64)
    valid_tangent = np.arange(1.0, 5.0, dtype=np.float64)

    with pytest.raises(ValueError, match="flattened source"):
        rule.value_fn(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    with pytest.raises(ValueError, match="lower bound"):
        rule.value_fn(np.array([0.0, 1.0, 2.0, -2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="flattened source"):
        rule.jvp_rule(valid_values, np.array([1.0, 2.0, 3.0], dtype=np.float64))
    with pytest.raises(ValueError, match="cotangent shape"):
        rule.vjp_rule(valid_values, np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="clipping boundary"):
        rule.vjp_rule(
            np.array([-1.0, 0.5, -1.0, 1.0], dtype=np.float64),
            np.ones(2, dtype=np.float64),
        )
    _assert_allclose(rule.jvp_rule(valid_values, valid_tangent), [1.0, 2.0])


def test_program_ad_selection_generic_registry_rules_refuse_missing_static_signatures() -> None:
    """Generic registry rules should require operator interception or a static factory."""
    values = np.array([1.0, 2.0], dtype=np.float64)
    for name in ("where", "clip", "sort", "select", "piecewise", "choose", "compress", "extract"):
        rule = custom_derivative_rule_for(
            PrimitiveIdentity("scpn.program_ad.selection", name, "1")
        )
        assert rule.jvp_rule is not None
        with pytest.raises(ValueError, match="static derivative factories"):
            rule.value_fn(values)
        with pytest.raises(ValueError, match="static derivative factories"):
            rule.jvp_rule(values, values)
