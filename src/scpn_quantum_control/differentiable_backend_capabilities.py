# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native derivative backend capabilities
"""Admit native derivative capabilities without executing or selecting a model."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

BackendAttributeLookup: TypeAlias = Callable[[object, str, object], object]
"""SDK attribute reader, retaining descriptor behavior and the absence sentinel."""


@dataclass(frozen=True)
class BackendCapabilityPrimitives:
    """Read the caller's active Python primitives at each admission operation.

    Parameters
    ----------
    attribute
        Resolve the active attribute reader before each lookup.
    predicate
        Resolve the active callable predicate before each callability test.
    presence
        Resolve the active attribute-presence predicate before each JAX check.
    invalid
        Resolve the active runtime-error constructor after a capability fails.

    """

    attribute: Callable[[], BackendAttributeLookup]
    predicate: Callable[[], Callable[[object], bool]]
    presence: Callable[[], Callable[[object, str], bool]]
    invalid: Callable[[], type[RuntimeError]]


def require_jax_host_callback(
    namespace: object, *, primitives: BackendCapabilityPrimitives
) -> None:
    """Require the JAX capabilities used by a staged host callback.

    Parameters
    ----------
    namespace
        Actual JAX namespace supplied by the caller's optional-runtime loader.
    primitives
        Live Python operations from the original entry point.

    Raises
    ------
    RuntimeError
        The first missing capability, in JIT, callback, shape-description order.

    """
    if not primitives.presence()(namespace, "jit"):
        raise primitives.invalid()("JAX JIT is unavailable in the active JAX module")
    if not primitives.presence()(namespace, "pure_callback"):
        raise primitives.invalid()("JAX pure_callback is required for JIT-wrapped host gradients")
    if not primitives.presence()(namespace, "ShapeDtypeStruct"):
        raise primitives.invalid()(
            "JAX ShapeDtypeStruct is required for JIT-wrapped host gradients"
        )


def require_jax_custom_vjp(namespace: object, *, primitives: BackendCapabilityPrimitives) -> None:
    """Require callable JAX custom-VJP registration and value/gradient execution.

    Parameters
    ----------
    namespace
        Actual JAX namespace. Admission does not invoke either capability.
    primitives
        Live attribute, callable and refusal operations.

    Raises
    ------
    RuntimeError
        Registration or value/gradient execution is absent or not callable.
        Existing adapter refusal messages are retained.

    """
    custom_vjp = primitives.attribute()(namespace, "custom_vjp", None)
    if not primitives.predicate()(custom_vjp):
        raise primitives.invalid()(
            "JAX custom_vjp is required for the bounded-QNN custom VJP route"
        )
    value_and_grad = primitives.attribute()(namespace, "value_and_grad", None)
    if not primitives.predicate()(value_and_grad):
        raise primitives.invalid()(
            "JAX value_and_grad is required for the bounded-QNN custom VJP route"
        )


def resolve_torch_autograd_function(
    namespace: object, *, primitives: BackendCapabilityPrimitives
) -> object:
    """Return the native autograd Function after checking its apply capability.

    Parameters
    ----------
    namespace
        Actual PyTorch namespace. All three original nested lookups are made.
    primitives
        Live lookup, callable and refusal operations.

    Returns
    -------
    object
        The exact Function class read from the SDK, without wrapping it.

    Raises
    ------
    RuntimeError
        Autograd or Function is absent, or apply is not callable.

    """
    autograd = primitives.attribute()(namespace, "autograd", None)
    function = primitives.attribute()(autograd, "Function", None)
    apply = primitives.attribute()(function, "apply", None)
    if autograd is None or function is None or not primitives.predicate()(apply):
        raise primitives.invalid()("PyTorch module does not expose torch.autograd.Function.apply")
    return function


def resolve_torch_autograd_grad(
    namespace: object, *, primitives: BackendCapabilityPrimitives
) -> object:
    """Return the native gradient callable without changing its binding.

    Parameters
    ----------
    namespace
        Actual PyTorch namespace used by the caller's recording context.
    primitives
        Live lookup, callable and refusal operations.

    Returns
    -------
    object
        The exact native grad callable; admission never invokes it.

    Raises
    ------
    RuntimeError
        The autograd grad capability is absent or not callable.

    """
    autograd = primitives.attribute()(namespace, "autograd", None)
    grad = primitives.attribute()(autograd, "grad", None)
    if not primitives.predicate()(grad):
        raise primitives.invalid()("PyTorch module does not expose torch.autograd.grad")
    return grad


def torch_runtime_available() -> bool:
    """Return whether the optional PyTorch runtime can be imported.

    Returns
    -------
    bool
        ``False`` when importing ``torch`` raises ``ImportError``.

    """
    try:
        importlib.import_module("torch")
    except ImportError:
        return False
    return True


def jax_runtime_available() -> bool:
    """Return whether the optional JAX runtime and ``jax.numpy`` can be imported.

    Returns
    -------
    bool
        ``False`` when either import raises ``AttributeError``, ``ImportError``,
        ``RuntimeError`` or ``ValueError``, the refusals a misconfigured JAX
        installation produces at import time.

    """
    try:
        importlib.import_module("jax")
        importlib.import_module("jax.numpy")
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return False
    return True


def tensorflow_runtime_available() -> bool:
    """Return whether the optional TensorFlow runtime can be imported.

    Returns
    -------
    bool
        ``False`` when importing ``tensorflow`` raises ``ImportError``.

    """
    try:
        importlib.import_module("tensorflow")
    except ImportError:
        return False
    return True


__all__ = [
    "BackendAttributeLookup",
    "BackendCapabilityPrimitives",
    "jax_runtime_available",
    "require_jax_custom_vjp",
    "require_jax_host_callback",
    "resolve_torch_autograd_function",
    "resolve_torch_autograd_grad",
    "tensorflow_runtime_available",
    "torch_runtime_available",
]
