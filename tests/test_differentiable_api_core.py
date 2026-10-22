# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Public numerical API owner contracts
"""Exercise original and lower public numerical API routes with coupled references."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control import differentiable_api_core as core
from scpn_quantum_control import differentiable_api_core_algorithms as algorithms

_API = importlib.import_module("scpn_quantum_control.differentiable_api")
_ROOT = importlib.import_module("scpn_quantum_control")


@pytest.mark.parametrize("operation", ["value", "gradient", "jacobian", "hessian"])
@pytest.mark.parametrize("entry", ["core", "legacy", "root", "dispatch"])
def test_public_numerical_routes_preserve_coupled_values_and_owned_arrays(
    operation: str, entry: str
) -> None:
    """Compare actual public routes with independent value and derivative references.

    Parameters
    ----------
    operation
        Complete scalar or vector numerical operation.
    entry
        Owning AD leaf, original facade, package root or original dispatcher.

    """
    values = np.array([-0.37, 0.24], dtype=np.float64)
    original = values.copy()
    difference = values[0] - values[1]

    def scalar(x: Any) -> Any:
        """Evaluate the actual coupled smooth scalar objective."""
        return np.sin(x[0]) + 0.5 * np.cos(x[1]) + 0.25 * np.sin(x[0] - x[1])

    def vector(x: Any) -> Any:
        """Evaluate the actual nonsymmetric vector objective."""
        return np.array([np.sin(x[0]) + np.cos(x[1]), np.sin(x[0] - x[1]) + x[0]])

    objective: Callable[[Any], Any] = vector if operation == "jacobian" else scalar
    if entry == "dispatch":
        result = _API.differentiable_api(
            operation, objective=objective, values=values, method="finite_difference"
        )
    else:
        namespace = {"core": core, "legacy": _API, "root": _ROOT}[entry]
        result = getattr(namespace, "differentiable_" + operation)(
            objective, values, method="finite_difference"
        )
    assert result.operation == operation and result.supported
    expected_value = float(scalar(values))
    if operation != "jacobian":
        assert result.value == pytest.approx(expected_value, abs=1e-12)
    if operation == "gradient":
        actual = result.gradient
        expected = np.array(
            [
                np.cos(values[0]) + 0.25 * np.cos(difference),
                -0.5 * np.sin(values[1]) - 0.25 * np.cos(difference),
            ]
        )
    elif operation == "jacobian":
        actual = result.jacobian
        expected = np.array(
            [
                [np.cos(values[0]), -np.sin(values[1])],
                [np.cos(difference) + 1.0, -np.cos(difference)],
            ]
        )
    elif operation == "hessian":
        actual = result.hessian
        expected = np.array(
            [
                [-np.sin(values[0]) - 0.25 * np.sin(difference), 0.25 * np.sin(difference)],
                [0.25 * np.sin(difference), -0.5 * np.cos(values[1]) - 0.25 * np.sin(difference)],
            ]
        )
    else:
        assert result.gradient is None and result.jacobian is None and result.hessian is None
        np.testing.assert_array_equal(values, original)
        return
    assert actual is not None and actual.dtype == np.float64 and actual.flags.writeable
    assert not np.shares_memory(actual, values)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("operation", ["value", "gradient", "jacobian", "hessian"])
def test_executed_legacy_route_retains_original_function_and_default_identities(
    operation: str,
) -> None:
    """Keep real original entry points instead of replacing them with aliases.

    Parameters
    ----------
    operation
        Original numerical function whose actual route and metadata are checked.

    """
    function = getattr(_API, "differentiable_" + operation)
    assert getattr(_ROOT, function.__name__) is function
    assert function.__globals__ is vars(_API)
    assert function.__module__ == "scpn_quantum_control.differentiable_api"
    assert function.__qualname__ == "differentiable_" + operation
    assert function.__code__.co_freevars == ()
    defaults = function.__kwdefaults__
    assert defaults is not None
    assert defaults == (
        {"method": "parameter_shift", "step": None}
        if operation in {"value", "gradient"}
        else {"method": "finite_difference", "step": 1e-6 if operation == "jacobian" else 1e-4}
    )
    for namespace in (core, algorithms):
        parameters = inspect.signature(
            getattr(namespace, "differentiable_" + operation)
        ).parameters
        assert parameters["method"].default == defaults["method"]
        assert parameters["step"].default == defaults["step"]
    assert core.__all__ == [
        "differentiable_value",
        "differentiable_gradient",
        "differentiable_jacobian",
        "differentiable_hessian",
    ]
    assert list(inspect.signature(function).parameters) == [
        "objective",
        "values",
        "method",
        "step",
    ]
    result = function(
        lambda x: np.array([x[0] ** 2]) if operation == "jacobian" else x[0] ** 2,
        [0.2],
        method="finite_difference",
    )
    assert result.supported and result.operation == operation
