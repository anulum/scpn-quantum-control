# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Live numerical API binding contracts
"""Observe real objective-time rebinding without replacing numerical providers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np
import pytest

from scpn_quantum_control.differentiable_api_contracts import UnifiedDifferentiableAPIResult

_Item = TypeVar("_Item")


@pytest.mark.parametrize("operation", ["value", "gradient", "jacobian", "hessian"])
@pytest.mark.parametrize("namespace", ["legacy", "core"])
def test_real_objective_rebindings_keep_original_constructor_and_primitive_order(
    operation: str, namespace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute actual AD while its objective changes the caller's live suppliers.

    Parameters
    ----------
    operation
        Scalar or vector route whose native result is converted into an envelope.
    namespace
        Original facade or independently usable pure AD module.
    monkeypatch
        Scoped observers that return actual scalar/list/dtype implementations.

    """
    module = importlib.import_module(
        "scpn_quantum_control.differentiable_api" + ("_core" if namespace == "core" else "")
    )
    events: list[str] = []

    class FirstResult(UnifiedDifferentiableAPIResult):
        """Actual result subtype selected by the objective before argument evaluation."""

    class LaterResult(UnifiedDifferentiableAPIResult):
        """Actual result subtype installed after the constructor was already read."""

    class ObservedFloat(float):
        """Delegate scalar conversion while recording its actual lookup stage."""

        def __new__(cls, value: float) -> ObservedFloat:
            """Convert the real scalar and change only the later constructor binding."""
            events.append("scalar")
            monkeypatch.setattr(module, "UnifiedDifferentiableAPIResult", LaterResult)
            return super().__new__(cls, value)

    class ObservedNumPy:
        """Expose the actual NumPy dtype through an observed property lookup."""

        @property
        def float64(self) -> type[np.float64]:
            """Return the real dtype when the original copy operation reads it."""
            events.append("dtype")
            return np.float64

    def later_list(values: Iterable[_Item]) -> list[_Item]:
        """Copy actual trainable metadata with the caller's second list supplier."""
        events.append("trainable")
        return list(values)

    def first_list(values: Iterable[_Item]) -> list[_Item]:
        """Copy actual names and install the list supplier read for the next field."""
        events.append("names")
        monkeypatch.setattr(module, "list", later_list, raising=False)
        return list(values)

    armed = False

    def objective(values: Any) -> Any:
        """Evaluate the real objective after installing scoped live suppliers once."""
        nonlocal armed
        if not armed:
            armed = True
            monkeypatch.setattr(module, "UnifiedDifferentiableAPIResult", FirstResult)
            monkeypatch.setattr(module, "float", ObservedFloat, raising=False)
            monkeypatch.setattr(module, "list", first_list, raising=False)
            monkeypatch.setattr(module, "np", ObservedNumPy())
        return (
            np.array([values[0] ** 2 + values[1], values[0] * values[1]])
            if operation == "jacobian"
            else values[0] ** 2 + values[0] * values[1] + values[1] ** 2
        )

    result = getattr(module, "differentiable_" + operation)(
        objective, [0.2, -0.3], method="finite_difference"
    )
    assert type(result) is FirstResult
    expected = [] if operation == "jacobian" else ["scalar"]
    if operation != "value":
        expected.append("dtype")
    assert events == expected + ["names", "trainable"]
    assert result.payload["parameter_names"] == ["theta_0", "theta_1"]
    assert result.payload["trainable"] == [True, True]


@pytest.mark.parametrize("missing", ["objective", "values"])
def test_original_public_dispatch_preserves_live_error_constructor(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep caller-defined error identities on real missing-input dispatch routes.

    Parameters
    ----------
    missing
        Original required numerical input deliberately omitted.
    monkeypatch
        Scoped change to the original facade's error constructor.

    """
    module = importlib.import_module("scpn_quantum_control.differentiable_api")

    class DeclaredError(ValueError):
        """Caller-defined subtype retaining the actual refusal text."""

    monkeypatch.setattr(module, "ValueError", DeclaredError, raising=False)
    message = (
        "objective is required for this differentiable operation"
        if missing == "objective"
        else "values are required for this differentiable operation"
    )
    with pytest.raises(DeclaredError, match="^" + message + "$"):
        module.differentiable_api(
            "gradient",
            objective=None if missing == "objective" else lambda x: x[0] ** 2,
            values=None if missing == "values" else [0.2],
        )
