# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD fail-closed boundary tests
"""Tests for Program AD nondifferentiable fail-closed boundary handling."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.differentiable import whole_program_value_and_grad


def test_program_ad_advanced_indexing_fails_closed() -> None:
    """Program AD array indexing should reject dynamic advanced index selectors."""

    with pytest.raises(ValueError, match="static integer or boolean"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, 2))[[values[0], values[1]]]),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="static integer or boolean"):
        whole_program_value_and_grad(
            lambda values: np.sum(values[np.array([values[0] > 0.0, True], dtype=object)]),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="static integer indices"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.take_along_axis(
                    np.reshape(values, (2, 2)),
                    np.array([[values[0], values[1]]], dtype=object),
                    axis=1,
                )
            ),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_index_selection_primitives_fail_closed() -> None:
    """Index-valued selection should require an explicit nondifferentiable policy."""

    with pytest.raises(
        ValueError, match="registered nondifferentiable integer selection primitives"
    ):
        whole_program_value_and_grad(
            lambda values: np.argmax(values),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(
        ValueError, match="registered nondifferentiable integer selection primitives"
    ):
        whole_program_value_and_grad(
            lambda values: np.reshape(values, (2, 2)).argmin(axis=1)[0],
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(
        ValueError, match="registered nondifferentiable integer selection primitives"
    ):
        whole_program_value_and_grad(
            lambda values: np.argsort(values, kind="stable")[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_extreme_reductions_fail_closed_on_ties() -> None:
    """Program AD max/min reductions should reject nondifferentiable tied selectors."""

    with pytest.raises(ValueError, match="np.max.*ties"):
        whole_program_value_and_grad(
            lambda values: np.max(values),
            np.array([2.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="np.min.*ties"):
        whole_program_value_and_grad(
            lambda values: np.min(np.reshape(values, (2, 2)), axis=1)[0],
            np.array([1.0, 1.0, 3.0, 4.0], dtype=np.float64),
        )


def test_whole_program_ad_selection_primitives_fail_closed_at_nondifferentiable_boundaries() -> (
    None
):
    """Selection primitives should reject boundary points with ambiguous derivatives."""

    with pytest.raises(ValueError, match="maximum is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.maximum(values[0], values[1]),
            np.array([1.0, 1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="minimum is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.minimum(values[0], values[1]),
            np.array([1.0, 1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="np.clip is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.clip(values, -0.5, 0.5)),
            np.array([-0.5, 0.25], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="np.clip is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.clip(values, -0.5, 0.5)),
            np.array([0.25, 0.5], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ordering predicate is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: values[0] if values[0] > values[1] else values[1],
            np.array([1.0, 1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ordering predicate is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.where(values >= 0.0, values, -values)),
            np.array([0.0, 1.0], dtype=np.float64),
        )


def test_whole_program_ad_abs_fails_closed_at_zero_cusp() -> None:
    """Python and NumPy absolute-value syntax should share the zero-cusp policy."""

    result = whole_program_value_and_grad(
        lambda values: abs(values[0]) + np.abs(values[1]),
        np.array([-2.0, 3.0], dtype=np.float64),
    )
    assert result.value == pytest.approx(5.0)
    np.testing.assert_allclose(result.gradient, np.array([-1.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="absolute value is non-differentiable at zero"):
        whole_program_value_and_grad(
            lambda values: abs(values[0]),
            np.array([0.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="absolute value is non-differentiable at zero"):
        whole_program_value_and_grad(
            lambda values: np.abs(values[0]),
            np.array([0.0], dtype=np.float64),
        )
