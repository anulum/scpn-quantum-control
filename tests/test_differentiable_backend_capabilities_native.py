# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real derivative backend capability integration
"""Exercise real CPU SDK capability admission through original adapter routes."""

from __future__ import annotations

import importlib
import pickle
from types import ModuleType

import numpy as np
import pytest
from _pickle_identity import IdentityUnpickler

from scpn_quantum_control.differentiable_backend_capabilities import (
    BackendCapabilityPrimitives,
    require_jax_host_callback,
    resolve_torch_autograd_function,
    resolve_torch_autograd_grad,
)
from scpn_quantum_control.phase import (
    jax_bridge,
    jax_compatibility,
    jax_gradients,
    torch_autograd_function,
    torch_bridge,
    torch_gradients,
)

pytestmark = pytest.mark.framework_imports


def _primitives() -> BackendCapabilityPrimitives:
    return BackendCapabilityPrimitives(
        attribute=lambda: getattr,
        predicate=lambda: callable,
        presence=lambda: hasattr,
        invalid=lambda: RuntimeError,
    )


def test_real_torch_callable_identity_and_custom_backward() -> None:
    """Both original public adapters execute the actual native backward rule."""
    torch = importlib.import_module("torch")
    assert resolve_torch_autograd_function(torch, primitives=_primitives()) is (
        torch.autograd.Function
    )
    assert resolve_torch_autograd_grad(torch, primitives=_primitives()) is torch.autograd.grad
    features = np.array([[0.1, 0.2], [0.4, -0.2], [-0.1, 0.6]])
    labels = np.array([0.0, 1.0, 0.0])
    values = np.array([0.3, -0.4])
    result = torch_bridge.torch_autograd_qnn_value_and_grad(features, labels, values)
    assert result.passed
    np.testing.assert_allclose(
        np.asarray(result.torch_gradient), result.parameter_shift_gradient, atol=1e-12
    )
    params = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss = torch_autograd_function.torch_autograd_function_qnn_loss(features, labels, params)
    loss.backward()
    np.testing.assert_allclose(
        np.asarray(params.grad), result.parameter_shift_gradient, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(loss.detach()), result.loss, atol=1e-12)


@pytest.mark.parametrize("capability", ["autograd", "Function", "apply", "grad"])
def test_original_torch_route_rejects_missing_real_sdk_capability(
    monkeypatch: pytest.MonkeyPatch, capability: str
) -> None:
    """Remove only the named real SDK capability at its admission boundary.

    Parameters
    ----------
    monkeypatch
        Restore the actual SDK before the next native execution.
    capability
        Original lookup or callability branch to exercise.

    """
    torch = importlib.import_module("torch")
    owner = (
        torch
        if capability == "autograd"
        else torch.autograd.Function
        if capability == "apply"
        else torch.autograd
    )
    message = (
        "PyTorch module does not expose torch.autograd.grad"
        if capability == "grad"
        else "PyTorch module does not expose torch.autograd.Function.apply"
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(owner, capability, None)
        with pytest.raises(RuntimeError) as caught:
            torch_bridge.torch_autograd_qnn_value_and_grad([[0.1]], [0.0], [0.3])
    assert str(caught.value) == message


def test_real_jax_jitted_host_callback_and_custom_vjp() -> None:
    """The original JAX entries retain real staged callbacks and native VJPs."""
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    with jax.enable_x64():
        callback = jax_bridge.jax_parameter_shift_value_and_grad(
            lambda values: float(np.cos(values[0])), [0.3], jit=True
        )
        assert callback.jitted and callback.host_callback
        np.testing.assert_allclose(callback.gradient, [-np.sin(0.3)], atol=1e-12)
        result = jax_bridge.jax_custom_vjp_qnn_value_and_grad(
            [[0.1, 0.2], [0.4, -0.2]], [0.0, 1.0], [0.3, -0.4], jit=True
        )
        assert result.jitted and result.passed
        np.testing.assert_allclose(result.gradient, result.parameter_shift_gradient, atol=1e-12)
        assert jnp.asarray(result.gradient).shape == (2,)


@pytest.mark.parametrize(
    ("capability", "message"),
    [
        ("jit", "JAX JIT is unavailable in the active JAX module"),
        ("pure_callback", "JAX pure_callback is required for JIT-wrapped host gradients"),
        (
            "ShapeDtypeStruct",
            "JAX ShapeDtypeStruct is required for JIT-wrapped host gradients",
        ),
    ],
)
def test_jax_callback_admission_rejects_missing_real_sdk_attribute(
    monkeypatch: pytest.MonkeyPatch, capability: str, message: str
) -> None:
    """The public capability operation preserves the original presence checks.

    Parameters
    ----------
    monkeypatch
        Restore the actual JAX namespace immediately after refusal.
    capability
        Missing runtime attribute.
    message
        Original refusal associated with that first missing attribute.

    """
    jax = importlib.import_module("jax")
    with monkeypatch.context() as scoped:
        scoped.delattr(jax, capability)
        with pytest.raises(RuntimeError) as caught:
            require_jax_host_callback(jax, primitives=_primitives())
    assert str(caught.value) == message


@pytest.mark.parametrize("capability", ["custom_vjp", "value_and_grad"])
def test_original_jax_route_rejects_noncallable_real_sdk_attribute(
    monkeypatch: pytest.MonkeyPatch, capability: str
) -> None:
    """The original public custom-VJP entry fails before constructing the model.

    Parameters
    ----------
    monkeypatch
        Restore the actual JAX attribute after the explicit fault boundary.
    capability
        Native derivative capability made unavailable.

    """
    jax = importlib.import_module("jax")
    with monkeypatch.context() as scoped:
        scoped.setattr(jax, capability, None)
        with pytest.raises(RuntimeError) as caught:
            jax_bridge.jax_custom_vjp_qnn_value_and_grad([[0.1]], [0.0], [0.3])
    assert str(caught.value) == (
        f"JAX {capability} is required for the bounded-QNN custom VJP route"
    )


def test_original_torch_route_reads_rebound_lookup_and_error_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native public execution observes lookup rebinding between SDK reads.

    Parameters
    ----------
    monkeypatch
        Restore the legacy module's globals after execution and explicit refusal.

    """
    reads: list[str] = []

    def later(value: object, name: str, default: object) -> object:
        reads.append("later:" + name)
        return getattr(value, name, default)

    def first(value: object, name: str, default: object) -> object:
        reads.append("first:" + name)
        if name == "autograd":
            monkeypatch.setattr(torch_gradients, "getattr", later, raising=False)
        return getattr(value, name, default)

    monkeypatch.setattr(torch_gradients, "getattr", first, raising=False)
    result = torch_bridge.torch_autograd_qnn_value_and_grad([[0.1]], [0.0], [0.3])
    assert result.passed
    assert reads[:3] == ["first:autograd", "later:Function", "later:apply"]

    class CapabilityError(RuntimeError):
        """Error selected while the original module resolves a missing capability."""

    def unavailable(value: object, name: str, default: object) -> object:
        if name == "Function":
            monkeypatch.setattr(torch_gradients, "RuntimeError", CapabilityError, raising=False)
            return None
        return getattr(value, name, default)

    monkeypatch.setattr(torch_gradients, "getattr", unavailable)
    with pytest.raises(CapabilityError) as caught:
        torch_bridge.torch_autograd_qnn_value_and_grad([[0.1]], [0.0], [0.3])
    assert str(caught.value) == "PyTorch module does not expose torch.autograd.Function.apply"


def test_original_jax_routes_read_rebound_presence_and_callable_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real staged and custom-VJP routes retain their original primitive reads.

    Parameters
    ----------
    monkeypatch
        Restore the original JAX adapter namespace after its live-read probes.

    """
    jax = importlib.import_module("jax")
    reads: list[str] = []

    def later_presence(value: object, name: str) -> bool:
        reads.append("later:" + name)
        return hasattr(value, name)

    def first_presence(value: object, name: str) -> bool:
        reads.append("first:" + name)
        monkeypatch.setattr(jax_gradients, "hasattr", later_presence, raising=False)
        return hasattr(value, name)

    def later_callable(value: object) -> bool:
        reads.append("later:callable")
        return callable(value)

    def first_callable(value: object) -> bool:
        reads.append("first:callable")
        monkeypatch.setattr(jax_gradients, "callable", later_callable, raising=False)
        return callable(value)

    monkeypatch.setattr(jax_gradients, "hasattr", first_presence, raising=False)
    monkeypatch.setattr(jax_gradients, "callable", first_callable, raising=False)
    with jax.enable_x64():
        result = jax_bridge.jax_parameter_shift_value_and_grad(
            lambda values: float(np.cos(values[0])), [0.3], jit=True
        )
        np.testing.assert_allclose(result.gradient, [-np.sin(0.3)], atol=1e-12)
        assert reads[:3] == ["first:jit", "later:pure_callback", "later:ShapeDtypeStruct"]
        result_vjp = jax_bridge.jax_custom_vjp_qnn_value_and_grad([[0.1]], [0.0], [0.3])
        assert result_vjp.passed
    assert reads[3:] == ["first:callable", "later:callable"]


@pytest.mark.parametrize(
    ("module", "name", "consumer"),
    [
        (torch_gradients, "_torch_autograd_function", torch_bridge),
        (torch_gradients, "_torch_autograd_grad", torch_bridge),
        (torch_autograd_function, "_torch_autograd_function", torch_autograd_function),
        (jax_gradients, "_require_jax_custom_vjp_support", jax_compatibility),
    ],
)
def test_original_definitions_keep_namespace_pickle_and_imported_identity(
    module: ModuleType, name: str, consumer: ModuleType
) -> None:
    """Legacy compatibility includes imported private helpers' natural identity.

    Parameters
    ----------
    module
        Original defining module.
    name
        Retained natural function definition.
    consumer
        Existing importing consumer whose alias must remain the same function.

    """
    function = getattr(module, name)
    assert function.__module__ == module.__name__
    assert function.__qualname__ == name
    assert function.__globals__ is vars(module)
    expected = {(module.__name__, name): function}
    assert IdentityUnpickler(pickle.dumps(function), expected).load() is function
    assert getattr(consumer, name) is function
