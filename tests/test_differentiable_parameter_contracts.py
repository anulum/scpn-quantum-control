# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable parameter contracts tests
# scpn-quantum-control -- differentiable parameter contract tests
"""Tests for differentiable parameter metadata and shift-rule contracts."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control import differentiable_parameter_contracts as contracts


def test_parameter_contracts_are_facade_and_package_root_compatible() -> None:
    """Extracted parameter contracts should preserve existing import identities."""

    assert differentiable.Parameter is contracts.Parameter
    assert differentiable.ParameterBounds is contracts.ParameterBounds
    assert differentiable.ParameterShiftRule is contracts.ParameterShiftRule
    assert differentiable.multi_frequency_parameter_shift_rule is (
        contracts.multi_frequency_parameter_shift_rule
    )
    assert scpn.Parameter is contracts.Parameter
    assert scpn.ParameterBounds is contracts.ParameterBounds
    assert scpn.ParameterShiftRule is contracts.ParameterShiftRule


def test_real_numeric_validation_accepts_arrays_scalars_and_indices() -> None:
    """Validation helpers should accept finite real arrays, scalars, and index vectors."""

    np.testing.assert_allclose(
        contracts._as_real_numeric_array("values", [1, 2.5]),
        np.array([1.0, 2.5], dtype=np.float64),
    )
    assert contracts._as_real_scalar("value", np.array(2.0)) == pytest.approx(2.0)
    np.testing.assert_array_equal(
        contracts._as_index_vector("indices", np.array([0, 2], dtype=np.uint64)),
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_allclose(
        contracts._as_parameter_array([1.0, 2.0]),
        np.array([1.0, 2.0], dtype=np.float64),
    )


def test_real_numeric_validation_rejects_implicit_or_malformed_inputs() -> None:
    """Validation helpers should fail closed for non-real and non-vector inputs."""

    with pytest.raises(ValueError, match="rectangular"):
        contracts._as_real_numeric_array("values", [[1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="real numeric scalars"):
        contracts._as_real_numeric_array("values", [True])
    with pytest.raises(ValueError, match="real numeric scalars"):
        contracts._as_real_numeric_array("values", [1.0 + 0.0j])
    with pytest.raises(ValueError, match="real numeric scalars"):
        contracts._as_real_numeric_array(
            "values",
            np.array(["2026-06-21"], dtype="datetime64[D]"),
        )
    with pytest.raises(ValueError, match="real numeric scalar"):
        contracts._as_real_scalar("value", True)
    with pytest.raises(ValueError, match="real numeric scalar"):
        contracts._as_real_scalar("value", [1.0])
    with pytest.raises(ValueError, match="real numeric scalar"):
        contracts._as_real_scalar("value", "1.0")
    with pytest.raises(ValueError, match="finite"):
        contracts._as_real_scalar("value", math.inf)
    with pytest.raises(ValueError, match="integer indices"):
        contracts._as_index_vector("indices", [0.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        contracts._as_index_vector("indices", [[0, 1]])
    with pytest.raises(ValueError, match="non-negative"):
        contracts._as_index_vector("indices", [-1])
    with pytest.raises(ValueError, match="one-dimensional"):
        contracts._as_parameter_array([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        contracts._as_parameter_array([math.nan])


def test_parameter_and_bounds_contracts_validate_metadata() -> None:
    """Parameter metadata and bound intervals should reject malformed records."""

    assert contracts.Parameter("theta").trainable is True
    frozen = contracts.Parameter("phi", trainable=False)
    assert frozen.name == "phi"
    bounded = contracts.ParameterBounds(lower=-1, upper=1, periodic=True)
    assert bounded.lower == pytest.approx(-1.0)
    assert bounded.upper == pytest.approx(1.0)

    with pytest.raises(ValueError, match="non-empty"):
        contracts.Parameter("")
    with pytest.raises(ValueError, match="boolean"):
        contracts.Parameter("theta", trainable=cast(Any, np.bool_(True)))
    with pytest.raises(ValueError, match="periodic flag"):
        contracts.ParameterBounds(periodic=cast(Any, "yes"))
    with pytest.raises(ValueError, match="less than or equal"):
        contracts.ParameterBounds(lower=2.0, upper=1.0)
    with pytest.raises(ValueError, match="finite lower and upper"):
        contracts.ParameterBounds(lower=0.0, periodic=True)
    with pytest.raises(ValueError, match="lower < upper"):
        contracts.ParameterBounds(lower=1.0, upper=1.0, periodic=True)


def test_parameter_shift_rule_normalises_single_and_multi_term_rules() -> None:
    """Parameter-shift rules should freeze canonical single- and multi-term metadata."""

    single = contracts.ParameterShiftRule()
    assert single.shift == pytest.approx(math.pi / 2.0)
    assert single.coefficient == pytest.approx(0.5)
    assert single.terms == ((single.shift, single.coefficient),)
    assert single.is_single_term is True

    multi = contracts.ParameterShiftRule(
        shifts=(0.1, 0.2),
        coefficients=(1.0, -0.25),
        frequencies=(1.0, 2.0),
    )
    assert multi.shift == pytest.approx(0.1)
    assert multi.coefficient == pytest.approx(1.0)
    assert multi.terms == ((0.1, 1.0), (0.2, -0.25))
    assert multi.frequencies == (1.0, 2.0)
    assert multi.is_single_term is False


def test_parameter_shift_rule_rejects_malformed_terms() -> None:
    """Parameter-shift rule construction should fail closed for invalid terms."""

    with pytest.raises(ValueError, match="provided together"):
        contracts.ParameterShiftRule(shifts=(0.1,))
    with pytest.raises(ValueError, match="positive"):
        contracts.ParameterShiftRule(shift=0.0)
    with pytest.raises(ValueError, match="coefficient must be a real numeric scalar"):
        contracts.ParameterShiftRule(coefficient=cast(Any, "0.5"))
    with pytest.raises(ValueError, match="matching shapes"):
        contracts.ParameterShiftRule(shifts=(0.1, 0.2), coefficients=(1.0,))
    with pytest.raises(ValueError, match="at least one term"):
        contracts.ParameterShiftRule(shifts=(), coefficients=())
    with pytest.raises(ValueError, match="positive values"):
        contracts.ParameterShiftRule(shifts=(0.0,), coefficients=(1.0,))
    with pytest.raises(ValueError, match="at least one value"):
        contracts.ParameterShiftRule(frequencies=())
    with pytest.raises(ValueError, match="positive values"):
        contracts.ParameterShiftRule(frequencies=(0.0,))
    with pytest.raises(ValueError, match="unique"):
        contracts.ParameterShiftRule(frequencies=(1.0, 1.0))


def test_multi_frequency_parameter_shift_rule_solves_exact_coefficients() -> None:
    """Multi-frequency shift rules should solve finite exact coefficient systems."""

    rule = contracts.multi_frequency_parameter_shift_rule([1.0, 2.0, 3.0])
    theta = 0.23
    terms = rule.terms
    expected = math.cos(theta) - 0.4 * math.sin(2.0 * theta) + 0.15 * math.cos(3.0 * theta)
    shifted_sum = 0.0
    for shift, coefficient in terms:
        plus = (
            math.sin(theta + shift)
            + 0.2 * math.cos(2.0 * (theta + shift))
            + 0.05 * math.sin(3.0 * (theta + shift))
        )
        minus = (
            math.sin(theta - shift)
            + 0.2 * math.cos(2.0 * (theta - shift))
            + 0.05 * math.sin(3.0 * (theta - shift))
        )
        shifted_sum += coefficient * (plus - minus)

    assert rule.frequencies == (1.0, 2.0, 3.0)
    assert shifted_sum == pytest.approx(expected, abs=1.0e-12)


def test_multi_frequency_parameter_shift_rule_rejects_bad_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-frequency rule construction should reject invalid and ill-conditioned systems."""

    with pytest.raises(ValueError, match="at least one value"):
        contracts.multi_frequency_parameter_shift_rule([])
    with pytest.raises(ValueError, match="positive values"):
        contracts.multi_frequency_parameter_shift_rule([0.0, 1.0])
    with pytest.raises(ValueError, match="unique"):
        contracts.multi_frequency_parameter_shift_rule([1.0, 1.0])
    with pytest.raises(ValueError, match="greater than one"):
        contracts.multi_frequency_parameter_shift_rule([1.0], max_condition=1.0)
    with pytest.raises(ValueError, match="same length"):
        contracts.multi_frequency_parameter_shift_rule([1.0, 2.0], shifts=[0.1])
    with pytest.raises(ValueError, match="positive values"):
        contracts.multi_frequency_parameter_shift_rule([1.0], shifts=[0.0])
    with pytest.raises(ValueError, match="well-conditioned"):
        contracts._default_multi_frequency_shifts(np.array([1.0]), max_condition=0.0)

    with monkeypatch.context() as patched:
        patched.setattr(np, "sin", lambda values: np.zeros_like(values, dtype=np.float64))
        with pytest.raises(ValueError, match="ill-conditioned"):
            contracts.multi_frequency_parameter_shift_rule([1.0], shifts=[1.0])

    with pytest.raises(ValueError, match="ill-conditioned"):
        contracts.multi_frequency_parameter_shift_rule([1.0, 2.0], shifts=[0.1, 0.1])

    with monkeypatch.context() as patched:
        patched.setattr(np.linalg, "cond", lambda _matrix: math.inf)
        with pytest.raises(ValueError, match="ill-conditioned"):
            contracts.multi_frequency_parameter_shift_rule([1.0], shifts=[math.pi / 4.0])

    def nonfinite_solve(_matrix: object, _rhs: object) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.array([math.inf], dtype=np.float64)

    with monkeypatch.context() as patched:
        patched.setattr(np.linalg, "solve", nonfinite_solve)
        with pytest.raises(ValueError, match="coefficients"):
            contracts.multi_frequency_parameter_shift_rule([1.0], shifts=[math.pi / 4.0])
