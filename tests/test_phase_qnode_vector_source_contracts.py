# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Vector QNode source contracts
"""Preserve defining namespaces and live suppliers of public vector transforms."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.differentiable import Parameter, ParameterShiftRule
from scpn_quantum_control.phase import param_shift
from scpn_quantum_control.phase import qnode_vector_transforms as qnode


def _objective(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return independent analytic vector observables for genuine derivatives."""
    return np.array([np.cos(values[0]), np.sin(values[1])], dtype=np.float64)


@pytest.mark.parametrize(
    "definition",
    (qnode.PhaseQNodeVectorTransformResult, qnode.PhaseQNodeVectorTransformReadinessSuiteResult),
)
def test_vector_result_methods_keep_the_original_defining_globals(definition: type) -> None:
    """Preserve class identities and genuine property and serializer namespaces.

    Parameters
    ----------
    definition
        Public record class consumed by vector transform callers.

    """
    assert definition.__module__ == qnode.__name__
    pickle.dumps(definition, protocol=4)
    for name, descriptor in vars(definition).items():
        if name == "to_dict" or isinstance(descriptor, property):
            method = descriptor.fget if isinstance(descriptor, property) else descriptor
            assert inspect.isfunction(method)
            assert method.__module__ == qnode.__name__
            assert method.__globals__ is vars(qnode)


def test_vector_transforms_retain_original_imported_derivative_identities() -> None:
    """Preserve the original nominal gradient and Hessian serialization routes."""
    for name in ("parameter_shift_gradient", "parameter_shift_hessian"):
        original = vars(param_shift)[name]
        imported = vars(qnode)[name]
        assert imported is original
        assert pickle.dumps(imported, protocol=4) == pickle.dumps(original, protocol=4)


@pytest.mark.parametrize("derivative", ["gradient", "hessian"])
def test_vector_derivatives_receive_the_original_live_component_closures(
    monkeypatch: pytest.MonkeyPatch, derivative: str
) -> None:
    """Inspect retained callbacks while delegating all work to the real derivative.

    Parameters
    ----------
    monkeypatch
        Scoped replacement of the original imported derivative supplier.
    derivative
        Public vector Jacobian or Hessian route whose callbacks remain observable.

    """
    original = getattr(qnode, f"parameter_shift_{derivative}")
    callbacks: list[Callable[[NDArray[np.float64]], float]] = []
    outer = (
        "_evaluate_vector_jacobian"
        if derivative == "gradient"
        else "execute_phase_qnode_vector_hessian"
    )

    def consume(
        objective: Callable[[NDArray[np.float64]], float],
        params: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
    ) -> NDArray[np.float64]:
        """Observe the actual per-component callable and run its real supplier."""
        assert inspect.isfunction(objective)
        assert objective.__globals__ is vars(qnode)
        assert objective.__module__ == qnode.__name__
        assert objective.__qualname__ == outer + ".<locals>.scalar_component"
        assert objective.__code__.co_freevars == ("objective", "vector_value")
        assert objective.__defaults__ is None
        assert objective.__kwdefaults__ == {"index": len(callbacks)}
        callbacks.append(objective)
        result: NDArray[np.float64] = original(objective, params, parameters=parameters, rule=rule)
        return result

    monkeypatch.setitem(vars(qnode), f"parameter_shift_{derivative}", consume)
    params = np.array([0.2, -0.3], dtype=np.float64)
    if derivative == "gradient":
        result = qnode.execute_phase_qnode_vector_jacobian("jacfwd", _objective, params)
        assert result.jacobian is not None
        np.testing.assert_allclose(
            result.jacobian, np.diag([-np.sin(params[0]), np.cos(params[1])]), atol=1e-14
        )
    else:
        result = qnode.execute_phase_qnode_vector_hessian(_objective, params)
        assert result.hessian_tensor is not None
        np.testing.assert_allclose(
            result.hessian_tensor,
            [[[-np.cos(params[0]), 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, -np.sin(params[1])]]],
            atol=1e-14,
        )
    assert len(callbacks) == 2
    # A caller may retain a derivative callback and use it after the outer call.
    assert callbacks[0](params) == pytest.approx(np.cos(params[0]))
    assert callbacks[1](params) == pytest.approx(np.sin(params[1]))


def test_vector_result_constructor_is_read_after_objective_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honor a nominal result supplier rebound by a real objective probe.

    Parameters
    ----------
    monkeypatch
        Scoped replacement made during the public derivative's actual execution.

    """

    def objective(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate observables and replace the current result constructor."""
        monkeypatch.setitem(
            vars(qnode),
            "PhaseQNodeVectorTransformResult",
            qnode.PhaseQNodeVectorTransformReadinessSuiteResult,
        )
        return _objective(values)

    with pytest.raises(TypeError, match="transform"):
        qnode.execute_phase_qnode_vector_jacobian("jacfwd", objective, [0.2, -0.3])


def test_vector_missing_derivative_preserves_native_name_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the native error fields when a current dependency is withdrawn.

    Parameters
    ----------
    monkeypatch
        Scoped removal of the original module's imported derivative.

    """
    monkeypatch.delitem(vars(qnode), "parameter_shift_gradient")
    with pytest.raises(NameError) as raised:
        qnode.execute_phase_qnode_vector_jacobian("jacrev", _objective, [0.2, -0.3])
    assert raised.value.name == "parameter_shift_gradient"
    assert raised.value.__context__ is None
    assert not raised.value.__suppress_context__


def test_vector_admission_uses_the_original_cached_builtin_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep native builtin fallback rather than a replacement namespace view.

    Parameters
    ----------
    monkeypatch
        Scoped globals and cached builtins changes, restored after the call.

    """
    function = qnode.execute_phase_qnode_vector_jacobian
    assert inspect.isfunction(function)
    builtins = function.__getattribute__("__builtins__")
    monkeypatch.setitem(builtins, "np", vars(qnode)["np"])
    monkeypatch.delitem(vars(qnode), "np")
    # Newly created component closures still need their native float builtin.
    monkeypatch.setitem(vars(qnode), "__builtins__", {"float": float})
    result = function("jacfwd", _objective, [0.2, -0.3])
    assert result.supported
    assert result.jacobian is not None
    np.testing.assert_allclose(result.jacobian, np.diag([-np.sin(0.2), np.cos(-0.3)]), atol=1e-14)


def test_vector_readiness_keeps_original_nested_objective_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consume actual source-local fixtures through the real readiness routes.

    Parameters
    ----------
    monkeypatch
        Scoped public executor observers that delegate to the original executors.

    """
    vector_executor = qnode.execute_phase_qnode_vector_jacobian
    scalar_executor = qnode.execute_phase_qnode_vmap_grad
    observed: list[str] = []

    def observe(objective: Callable[..., object], name: str) -> None:
        """Inspect the actual fixture without changing its implementation."""
        assert inspect.isfunction(objective)
        assert objective.__module__ == qnode.__name__
        assert objective.__globals__ is vars(qnode)
        assert objective.__qualname__ == (
            "run_phase_qnode_vector_transform_readiness_suite.<locals>." + name
        )
        assert objective.__closure__ is None
        observed.append(name)

    def vector_consumer(
        transform: str,
        objective: Callable[[NDArray[np.float64]], ArrayLike],
        params: ArrayLike,
        *,
        backend: str = "statevector",
        adapter: str = "native",
        shots: int | None = None,
    ) -> qnode.PhaseQNodeVectorTransformResult:
        """Observe vector fixtures and execute their actual public transform."""
        observe(objective, "vector_objective")
        return vector_executor(
            transform, objective, params, backend=backend, adapter=adapter, shots=shots
        )

    def scalar_consumer(
        objective: Callable[[NDArray[np.float64]], float],
        batched_params: ArrayLike,
        *,
        backend: str = "statevector",
        shots: int | None = None,
    ) -> qnode.PhaseQNodeVectorTransformResult:
        """Observe scalar fixtures and execute their actual manual batch route."""
        observe(objective, "scalar_objective")
        return scalar_executor(objective, batched_params, backend=backend, shots=shots)

    monkeypatch.setitem(vars(qnode), "execute_phase_qnode_vector_jacobian", vector_consumer)
    monkeypatch.setitem(vars(qnode), "execute_phase_qnode_vmap_grad", scalar_consumer)
    assert qnode.run_phase_qnode_vector_transform_readiness_suite().passed
    assert observed.count("vector_objective") == 4
    assert observed.count("scalar_objective") == 2
