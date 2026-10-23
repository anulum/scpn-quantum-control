# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Numerical differentiation input admission tests
"""Exercise real arrays, scalar boundaries and live caller reads through public admission."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.differentiable_tensor_admission import (
    AdmissionPrimitives,
    require_nonnegative_tolerance,
    require_real_parameter_vector,
    require_real_scalar,
)


def _primitives(reads: list[str] | None = None) -> AdmissionPrimitives:
    def numpy() -> object:
        if reads is not None:
            reads.append("numpy")
        return np

    def scalar() -> type[float]:
        if reads is not None:
            reads.append("float")
        return float

    def invalid() -> type[ValueError]:
        if reads is not None:
            reads.append("error")
        return ValueError

    return AdmissionPrimitives(numpy=numpy, scalar=scalar, invalid=invalid)


@pytest.mark.parametrize(
    ("values", "width", "expected"),
    [
        ([0.2, -0.4], None, (0.2, -0.4)),
        (np.array([1, 2], dtype=np.int32), 2, (1.0, 2.0)),
        (np.arange(6, dtype=np.float64)[::2], 3, (0.0, 2.0, 4.0)),
        ([], None, ()),
        ([], 0, ()),
    ],
)
def test_public_parameter_admission_preserves_real_values(
    values: object, width: int | None, expected: tuple[float, ...]
) -> None:
    """Admit real contiguous, strided, converted and empty parameter vectors.

    Parameters
    ----------
    values
        Actual array-like parameters.
    width
        Original caller's declared width or an unconstrained width.
    expected
        Expected numerical values after real dtype conversion.

    """
    actual = require_real_parameter_vector(
        "parameters", values, width=width, primitives=_primitives()
    )
    assert actual.ndim == 1 and actual.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(actual, expected)


def test_public_admission_keeps_copy_policy_with_the_caller() -> None:
    """An already admitted real array retains its original backing storage."""
    original = np.array([0.2, -0.4], dtype=np.float64)
    actual = require_real_parameter_vector("values", original, primitives=_primitives())
    assert np.shares_memory(actual, original)
    np.testing.assert_array_equal(original, [0.2, -0.4])


@pytest.mark.parametrize(
    ("values", "width", "message"),
    [
        (0.2, None, "values must be a one-dimensional array"),
        ([[0.2, -0.4]], None, "values must be a one-dimensional array"),
        ([0.2, -0.4], 1, "values must have shape (1,), got (2,)"),
        ([], 1, "values must have shape (1,), got (0,)"),
        ([0.2, float("nan")], None, "values must contain only finite values"),
        ([float("inf")], None, "values must contain only finite values"),
        ([float("-inf")], None, "values must contain only finite values"),
    ],
)
def test_public_parameter_admission_rejects_invalid_input(
    values: object, width: int | None, message: str
) -> None:
    """Reject actual rank, width and non-finite numerical failures.

    Parameters
    ----------
    values
        Invalid real input.
    width
        Required caller width.
    message
        The unchanged caller-facing error text.

    """
    with pytest.raises(ValueError) as caught:
        require_real_parameter_vector("values", values, width=width, primitives=_primitives())
    assert str(caught.value) == message


@pytest.mark.parametrize("value", [0.25, np.array(0.25), np.float32(0.25), -0.0])
def test_public_scalar_admission_accepts_finite_values(value: object) -> None:
    """Convert actual scalar types without accepting singleton vectors.

    Parameters
    ----------
    value
        Finite scalar or zero-dimensional array.

    """
    actual = require_real_scalar("loss", value, primitives=_primitives())
    assert isinstance(actual, float)
    assert actual == float(np.asarray(value))
    assert math.copysign(1.0, actual) == math.copysign(1.0, float(np.asarray(value)))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([0.25], "loss must be a scalar"),
        ([], "loss must be a scalar"),
        (float("nan"), "loss must be finite"),
        (float("inf"), "loss must be finite"),
        (float("-inf"), "loss must be finite"),
    ],
)
def test_public_scalar_admission_rejects_rank_and_finiteness(value: object, message: str) -> None:
    """Preserve separate scalar-shape and scalar-finiteness errors.

    Parameters
    ----------
    value
        Actual invalid scalar input.
    message
        Required error message.

    """
    with pytest.raises(ValueError) as caught:
        require_real_scalar("loss", value, primitives=_primitives())
    assert str(caught.value) == message


@pytest.mark.parametrize("negative_first", [False, True])
@pytest.mark.parametrize("value", [0.0, -0.0, 0.25])
def test_public_tolerance_admission_preserves_signed_zero(
    negative_first: bool, value: float
) -> None:
    """Admit finite non-negative tolerances in both legacy evaluation orders.

    Parameters
    ----------
    negative_first
        The caller's original short-circuit order.
    value
        Valid real tolerance.

    """
    actual = require_nonnegative_tolerance(
        value,
        error_text="invalid tolerance",
        negative_first=negative_first,
        primitives=_primitives(),
    )
    assert actual == value
    assert math.copysign(1.0, actual) == math.copysign(1.0, value)


@pytest.mark.parametrize("negative_first", [False, True])
@pytest.mark.parametrize("value", [-0.25, float("-inf"), float("nan"), float("inf")])
def test_public_tolerance_admission_rejects_invalid_values(
    negative_first: bool, value: float
) -> None:
    """Reject actual negative and non-finite tolerances with caller-owned text.

    Parameters
    ----------
    negative_first
        The caller's original short-circuit order.
    value
        Invalid real tolerance.

    """
    with pytest.raises(ValueError) as caught:
        require_nonnegative_tolerance(
            value,
            error_text="original tolerance refusal",
            negative_first=negative_first,
            primitives=_primitives(),
        )
    assert str(caught.value) == "original tolerance refusal"


@pytest.mark.parametrize("negative_first", [False, True])
def test_negative_tolerance_keeps_original_finiteness_lookup(
    negative_first: bool,
) -> None:
    """A negativity-first rejection must retain its skipped namespace lookup.

    Parameters
    ----------
    negative_first
        The real short-circuit choice used by JAX or Torch/TensorFlow adapters.

    """
    reads: list[str] = []
    with pytest.raises(ValueError):
        require_nonnegative_tolerance(
            -1.0,
            error_text="original refusal",
            negative_first=negative_first,
            primitives=_primitives(reads),
        )
    assert reads == (["float", "error"] if negative_first else ["float", "numpy", "error"])


def test_public_vector_reads_conversion_and_finiteness_namespaces_late() -> None:
    """Real array admission resolves each numerical lookup without cached globals."""
    reads: list[str] = []
    actual = require_real_parameter_vector("values", [0.2, -0.4], primitives=_primitives(reads))
    np.testing.assert_array_equal(actual, [0.2, -0.4])
    assert reads == ["numpy", "float", "numpy", "numpy"]


def test_scalar_conversion_observes_constructor_rebinding_during_array_admission() -> None:
    """Real NumPy conversion can change the constructor used by the next expression."""
    constructors: list[type[float]] = [float]

    class ConvertedScalar(float):
        """The scalar constructor selected during actual array conversion."""

    class ScalarInput:
        """A real scalar protocol input that changes the active constructor."""

        def __float__(self) -> float:
            """Convert the input and replace the subsequent scalar constructor."""
            constructors[0] = ConvertedScalar
            return 0.25

    actual = require_real_scalar(
        "loss",
        ScalarInput(),
        primitives=AdmissionPrimitives(
            numpy=lambda: np, scalar=lambda: constructors[0], invalid=lambda: ValueError
        ),
    )
    assert type(actual) is ConvertedScalar
    assert actual == 0.25


def test_vector_failure_observes_error_rebinding_during_real_conversion() -> None:
    """The rejection uses the error constructor selected by actual NumPy input conversion."""
    constructors: list[type[ValueError]] = [ValueError]

    class AdmissionError(ValueError):
        """The error selected while converting a non-finite numerical input."""

    class NonFiniteInput:
        """A scalar protocol input with a conversion-time binding change."""

        def __float__(self) -> float:
            """Produce a real infinity and change the subsequent error constructor."""
            constructors[0] = AdmissionError
            return float("inf")

    with pytest.raises(AdmissionError, match="values must contain only finite values"):
        require_real_parameter_vector(
            "values",
            [NonFiniteInput()],
            primitives=AdmissionPrimitives(
                numpy=lambda: np, scalar=lambda: float, invalid=lambda: constructors[0]
            ),
        )
