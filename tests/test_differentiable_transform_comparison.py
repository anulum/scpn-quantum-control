# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Numerical transform comparison contracts
"""Exercise pure AD comparison and original audit supplier semantics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import scpn_quantum_control.differentiable_transform_algebra as algebra
from scpn_quantum_control.differentiable_finite_difference import jacfwd
from scpn_quantum_control.differentiable_transform_comparison import (
    as_finite_flat_array,
    maximum_absolute_residual,
)


@pytest.mark.parametrize("shape", [(2, 3), (6,), (0,)])
def test_finite_comparison_values_have_independent_flat_storage(shape: tuple[int, ...]) -> None:
    """Preserve C-order values and storage independence for noncontiguous inputs.

    Parameters
    ----------
    shape
        Supplied observation shape, including empty values.

    """
    size = int(np.prod(shape))
    original = np.arange(2 * size, dtype=np.float64)[::2].reshape(shape)
    values = as_finite_flat_array(original)
    assert values.dtype == np.dtype(np.float64)
    assert values.shape == (size,)
    np.testing.assert_array_equal(values, original.reshape(-1))
    assert not np.shares_memory(values, original)
    original[...] = -1.0
    np.testing.assert_array_equal(values, np.arange(2 * size, dtype=np.float64)[::2])


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_comparison_values_refuse_admission(nonfinite: float) -> None:
    """Reject invalid observations before residual or success construction.

    Parameters
    ----------
    nonfinite
        Invalid scalar in an otherwise finite observation.

    """
    with pytest.raises(ValueError, match="transform algebra values must be finite"):
        as_finite_flat_array([[0.2, nonfinite]])


def test_comparison_accepts_scalar_and_integer_observations() -> None:
    """Promote admitted observations to the original real float64 comparison domain."""
    np.testing.assert_array_equal(as_finite_flat_array(3.0), [3.0])
    result = as_finite_flat_array(np.array([[1, -2]], dtype=np.int32))
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result, [1.0, -2.0])


def test_residual_preserves_observation_shapes_and_maximum_error() -> None:
    """Measure exact errors and refuse to promote empty or incompatible observations."""
    lhs = np.array([[0.5, -0.75], [2.0, 0.0]], dtype=np.float64)
    rhs = np.array([[0.0, 0.5], [2.25, 0.0]], dtype=np.float64)
    assert maximum_absolute_residual(lhs, rhs) == 1.25
    assert maximum_absolute_residual(lhs, lhs) == 0.0
    assert maximum_absolute_residual(lhs, rhs.ravel()) == float("inf")
    empty = np.array([], dtype=np.float64)
    assert maximum_absolute_residual(empty, empty) == float("inf")


def test_public_audit_reads_cast_after_real_array_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep live source reads after a real derivative's array protocol executes.

    Parameters
    ----------
    monkeypatch
        Scoped source cast replacement performed by a derivative observation.

    """
    original_forward = jacfwd
    assert vars(algebra)["jacfwd"] is original_forward
    events: list[str] = []

    def current_cast(kind: object, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Observe the original nominal array alias and preserve the actual values."""
        assert kind is algebra.FloatArray
        events.append("cast")
        return values

    class RebindingObservation:
        """A genuine array-protocol observation around actual derivative values."""

        def __init__(self, values: NDArray[np.float64]) -> None:
            """Retain the derivative's numerical values without changing them."""
            self.values = values

        def __array__(
            self, dtype: np.dtype[np.float64] | None = None, copy: bool | None = None
        ) -> NDArray[np.float64]:
            """Convert the actual derivative and replace the current source cast."""
            events.append("convert")
            monkeypatch.setitem(vars(algebra), "cast", current_cast)
            return np.array(self.values, dtype=dtype, copy=copy)

    def forward(
        objective: Callable[[NDArray[np.float64]], ArrayLike], values: ArrayLike
    ) -> ArrayLike:
        """Execute the real forward Jacobian and expose its real array conversion."""
        return RebindingObservation(original_forward(objective, values))

    monkeypatch.setitem(vars(algebra), "jacfwd", forward)
    result = algebra.run_transform_algebra_audit()
    assert result.passed
    assert events[0:2] == ["convert", "cast"]


def test_public_audit_retains_native_missing_numpy_error_after_array_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the original missing global when a real observation withdraws it.

    Parameters
    ----------
    monkeypatch
        Scoped removal performed at the real array conversion boundary.

    """
    original_forward = jacfwd
    assert vars(algebra)["jacfwd"] is original_forward

    class WithdrawingObservation:
        """Actual derivative values whose array conversion withdraws a supplier."""

        def __init__(self, values: NDArray[np.float64]) -> None:
            """Retain the real derivative observations."""
            self.values = values

        def __array__(
            self, dtype: np.dtype[np.float64] | None = None, copy: bool | None = None
        ) -> NDArray[np.float64]:
            """Return actual values and withdraw the original NumPy supplier."""
            monkeypatch.delitem(vars(algebra), "np")
            return np.array(self.values, dtype=dtype, copy=copy)

    def forward(
        objective: Callable[[NDArray[np.float64]], ArrayLike], values: ArrayLike
    ) -> ArrayLike:
        """Execute the real Jacobian before exposing its array boundary."""
        return WithdrawingObservation(original_forward(objective, values))

    monkeypatch.setitem(vars(algebra), "jacfwd", forward)
    with pytest.raises(NameError) as raised:
        algebra.run_transform_algebra_audit()
    assert raised.value.name == "np"
    assert raised.value.__context__ is None
    assert not raised.value.__suppress_context__
