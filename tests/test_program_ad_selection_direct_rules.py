# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD selection direct derivative rules
"""Tests for Program AD selection direct derivative-rule factories."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    program_ad_selection_clip_derivative_rule,
    program_ad_selection_where_derivative_rule,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy closeness across dynamically typed direct-rule payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


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
