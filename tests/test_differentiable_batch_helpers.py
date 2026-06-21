# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable batch helper tests
"""Regression tests for extracted differentiable batch helper contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable_batch_helpers import (
    _as_batch_parameter_array,
    _as_batch_vector_array,
    _as_parameter_shift_sample_tensor,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy closeness across helper arrays."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_facade_reuses_extracted_batch_helpers() -> None:
    """The differentiable facade should reuse the extracted helper objects."""

    facade_symbols = vars(differentiable)
    assert facade_symbols["_as_parameter_shift_sample_tensor"] is _as_parameter_shift_sample_tensor
    assert facade_symbols["_as_batch_parameter_array"] is _as_batch_parameter_array
    assert facade_symbols["_as_batch_vector_array"] is _as_batch_vector_array


def test_parameter_shift_sample_tensor_accepts_single_and_multi_term_shapes() -> None:
    """Parameter-shift samples should normalize one-term vectors and preserve matrices."""

    _assert_allclose(
        _as_parameter_shift_sample_tensor("samples", [1.0, 2.0], term_count=1),
        [[1.0, 2.0]],
    )
    _assert_allclose(
        _as_parameter_shift_sample_tensor(
            "samples",
            [[1.0, 2.0], [3.0, 4.0]],
            term_count=2,
        ),
        [[1.0, 2.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize(
    ("values", "term_count", "message"),
    [
        (1.0, 1, "shape"),
        ([[1.0, 2.0]], 2, "first dimension"),
        (np.empty((1, 0), dtype=np.float64), 1, "at least one parameter column"),
        ([[np.inf]], 1, "finite values"),
        (cast(Any, [["bad"]]), 1, "real numeric"),
    ],
)
def test_parameter_shift_sample_tensor_rejects_invalid_shapes(
    values: object,
    term_count: int,
    message: str,
) -> None:
    """Parameter-shift samples should fail closed on malformed tensors."""

    with pytest.raises(ValueError, match=message):
        _as_parameter_shift_sample_tensor("samples", cast(Any, values), term_count=term_count)


def test_batch_parameter_array_accepts_two_dimensional_parameter_batches() -> None:
    """Parameter batches should be finite and match the requested width."""

    _assert_allclose(
        _as_batch_parameter_array("parameters", [[1.0, 2.0], [3.0, 4.0]], 2),
        [[1.0, 2.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0, 2.0], "two-dimensional batch"),
        (np.empty((0, 2), dtype=np.float64), "at least one row"),
        ([[1.0, 2.0]], "parameter length"),
        ([[np.nan]], "finite values"),
        (cast(Any, [["bad"]]), "real numeric"),
    ],
)
def test_batch_parameter_array_rejects_invalid_batches(values: object, message: str) -> None:
    """Parameter batches should fail closed on invalid batch contracts."""

    with pytest.raises(ValueError, match=message):
        _as_batch_parameter_array("parameters", cast(Any, values), 1)


def test_batch_vector_array_accepts_two_dimensional_vector_batches() -> None:
    """Cotangent batches should be finite and match the requested vector width."""

    _assert_allclose(
        _as_batch_vector_array("cotangents", [[1.0, 2.0], [3.0, 4.0]], 2),
        [[1.0, 2.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0, 2.0], "two-dimensional batch"),
        (np.empty((0, 2), dtype=np.float64), "at least one row"),
        ([[1.0, 2.0]], "vector length"),
        ([[np.nan]], "finite values"),
        (cast(Any, [["bad"]]), "real numeric"),
    ],
)
def test_batch_vector_array_rejects_invalid_batches(values: object, message: str) -> None:
    """Cotangent batches should fail closed on invalid vector-batch contracts."""

    with pytest.raises(ValueError, match=message):
        _as_batch_vector_array("cotangents", cast(Any, values), 1)
