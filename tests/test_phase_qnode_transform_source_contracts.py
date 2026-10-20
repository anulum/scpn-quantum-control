# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Scalar QNode source contracts
"""Preserve original scalar QNode callable, result and live dependency contracts."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.phase import param_shift
from scpn_quantum_control.phase import qnode_transforms as qnode


def _objective(values: NDArray[np.float64]) -> float:
    """Return an exact smooth scalar observable for the public transform route."""
    return float(np.sum(np.cos(values)))


@pytest.mark.parametrize(
    "name",
    (
        "PhaseQNodeComplexDerivativeContract",
        "PhaseQNodeTransformResult",
        "PhaseQNodeTransformReadinessSuiteResult",
    ),
)
def test_scalar_qnode_result_methods_keep_original_defining_globals(name: str) -> None:
    """Preserve natural class and method ownership used by result consumers.

    Parameters
    ----------
    name
        Original result class name exported by the scalar transform module.

    """
    definition = getattr(qnode, name)
    assert definition.__module__ == qnode.__name__
    pickle.dumps(definition, protocol=4)
    for member_name, descriptor in vars(definition).items():
        if member_name == "to_dict" or isinstance(descriptor, property):
            method = descriptor.fget if isinstance(descriptor, property) else descriptor
            assert inspect.isfunction(method)
            assert method.__module__ == qnode.__name__
            assert method.__globals__ is vars(qnode)


def test_scalar_qnode_imported_derivatives_keep_original_identities() -> None:
    """Keep the actual imported functions and their protected serialisation paths."""
    for name in (
        "parameter_shift_gradient",
        "parameter_shift_hessian",
        "value_and_parameter_shift_grad",
    ):
        imported = vars(qnode)[name]
        original = vars(param_shift)[name]
        assert imported is original
        assert pickle.dumps(imported, protocol=4) == pickle.dumps(original, protocol=4)


def test_scalar_qnode_gradient_reads_the_current_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the original metadata keyword to the actual replacement supplier."""
    monkeypatch.setitem(vars(qnode), "parameter_shift_gradient", np.sum)
    with pytest.raises(TypeError, match="parameters"):
        qnode.execute_phase_qnode_transform("grad", _objective, np.array([0.2, -0.3]))


def test_scalar_qnode_result_constructor_is_read_after_objective_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the original nominal result constructor after real objective probes."""

    def objective(values: NDArray[np.float64]) -> float:
        """Observe a smooth scalar and replace the current nominal constructor."""
        monkeypatch.setitem(
            vars(qnode), "PhaseQNodeTransformResult", qnode.PhaseQNodeComplexDerivativeContract
        )
        return _objective(values)

    with pytest.raises(TypeError, match="transform"):
        qnode.execute_phase_qnode_transform("grad", objective, np.array([0.2, -0.3]))


def test_scalar_qnode_missing_derivative_keeps_native_error_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a withdrawn dependency with its native missing-name metadata."""
    monkeypatch.delitem(vars(qnode), "parameter_shift_gradient")
    with pytest.raises(NameError) as raised:
        qnode.execute_phase_qnode_transform("grad", _objective, np.array([0.2, -0.3]))
    assert raised.value.name == "parameter_shift_gradient"
    assert raised.value.__context__ is None
    assert not raised.value.__suppress_context__


def test_scalar_qnode_keeps_native_cached_builtin_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve numerical admission through the original functions' cached builtins."""
    function = qnode.execute_phase_qnode_transform
    assert inspect.isfunction(function)
    builtins = function.__getattribute__("__builtins__")
    monkeypatch.setitem(builtins, "np", vars(qnode)["np"])
    monkeypatch.delitem(vars(qnode), "np")
    monkeypatch.setitem(vars(qnode), "__builtins__", {})
    values = np.array([0.2, -0.3], dtype=np.float64)
    result = qnode.execute_phase_qnode_transform("grad", _objective, values)
    assert result.gradient is not None
    np.testing.assert_allclose(result.gradient, -np.sin(values), atol=1e-14)


def test_scalar_qnode_readiness_preserves_the_original_objective_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the actual readiness callable in its original source and lifetime."""
    original = qnode.execute_phase_qnode_transform
    observations: list[str] = []

    def consume(
        transform: str | tuple[str, ...],
        objective: Callable[[NDArray[np.float64]], float],
        params: ArrayLike,
        *,
        tangent: ArrayLike | None = None,
        cotangent: ArrayLike | float | None = None,
        backend: str = "statevector",
        shots: int | None = None,
    ) -> qnode.PhaseQNodeTransformResult:
        """Inspect the real source callable and execute the real public consumer."""
        assert inspect.isfunction(objective)
        assert objective.__module__ == qnode.__name__
        assert objective.__qualname__ == (
            "run_phase_qnode_transform_readiness_suite.<locals>.objective"
        )
        assert objective.__globals__ is vars(qnode)
        assert objective.__closure__ is None
        observations.append(objective.__qualname__)
        return original(
            transform,
            objective,
            params,
            tangent=tangent,
            cotangent=cotangent,
            backend=backend,
            shots=shots,
        )

    monkeypatch.setitem(vars(qnode), "execute_phase_qnode_transform", consume)
    result = qnode.run_phase_qnode_transform_readiness_suite()
    assert result.passed
    assert len(observations) == 10
