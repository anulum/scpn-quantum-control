# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native tensor construction
"""Construct native tensors without selecting a model or differentiation route."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast


class TensorAttributeLookup(Protocol):
    """Read one optional SDK attribute without evaluating the returned callable."""

    def __call__(self, value: object, name: str, default: object, /) -> object:
        """Return the attribute, or the supplied default when it is absent."""
        ...


class NativeTensorFactory(Protocol):
    """Construct a tensor with either the SDK default or an explicit dtype."""

    def __call__(self, values: object, *, dtype: object = ...) -> object:
        """Convert values while retaining the SDK's native allocation semantics."""
        ...


@dataclass(frozen=True)
class TensorConstructionPrimitives:
    """Read the caller's attribute, capability and refusal primitives lazily.

    Parameters
    ----------
    attribute
        Read the active attribute lookup before each SDK capability lookup.
    predicate
        Read the active callable predicate before each capability test.
    invalid
        Read the active runtime-error constructor only when a capability fails.

    """

    attribute: Callable[[], TensorAttributeLookup]
    predicate: Callable[[], Callable[[object], bool]]
    invalid: Callable[[], type[RuntimeError]]


def construct_torch_tensor(
    namespace: object,
    values: object,
    *,
    primitives: TensorConstructionPrimitives,
) -> object:
    """Convert values using PyTorch's preferred factory and available float64.

    Parameters
    ----------
    namespace
        Actual PyTorch SDK namespace. Prefer ``as_tensor`` over ``tensor``.
    values
        Values accepted by the selected native factory.
    primitives
        Live capability and refusal operations from the owning entry point.

    Returns
    -------
    object
        Native tensor, including any storage sharing chosen by the SDK.

    Raises
    ------
    RuntimeError
        Neither factory is callable. Native factory exceptions propagate.

    """
    dtype = primitives.attribute()(namespace, "float64", None)
    as_tensor = primitives.attribute()(namespace, "as_tensor", None)
    if primitives.predicate()(as_tensor):
        if dtype is None:
            return cast(NativeTensorFactory, as_tensor)(values)
        return cast(NativeTensorFactory, as_tensor)(values, dtype=dtype)
    tensor = primitives.attribute()(namespace, "tensor", None)
    if primitives.predicate()(tensor):
        if dtype is None:
            return cast(NativeTensorFactory, tensor)(values)
        return cast(NativeTensorFactory, tensor)(values, dtype=dtype)
    raise primitives.invalid()("PyTorch module does not expose as_tensor or tensor")


def prepare_torch_trainable_tensor(
    tensor: object,
    *,
    primitives: TensorConstructionPrimitives,
) -> object:
    """Detach, clone and enable native gradient recording when supported.

    Parameters
    ----------
    tensor
        Constructed native tensor. Detach and clone retain their SDK semantics.
    primitives
        Live capability and refusal operations from the owning entry point.

    Returns
    -------
    object
        Native tensor returned by ``requires_grad_(True)``.

    Raises
    ------
    RuntimeError
        The supplied tensor has no callable ``requires_grad_`` capability.

    """
    detach = primitives.attribute()(tensor, "detach", None)
    if primitives.predicate()(detach):
        tensor = cast(Callable[[], object], detach)()
    clone = primitives.attribute()(tensor, "clone", None)
    if primitives.predicate()(clone):
        tensor = cast(Callable[[], object], clone)()
    requires_grad = primitives.attribute()(tensor, "requires_grad_", None)
    if not primitives.predicate()(requires_grad):
        raise primitives.invalid()("PyTorch tensor does not expose requires_grad_")
    return cast(Callable[[bool], object], requires_grad)(True)


def construct_tensorflow_tensor(
    namespace: object,
    values: object,
    *,
    primitives: TensorConstructionPrimitives,
) -> object:
    """Convert values with TensorFlow's factory and available float64 dtype.

    Parameters
    ----------
    namespace
        Actual TensorFlow SDK namespace.
    values
        Values accepted by the native ``convert_to_tensor`` operation.
    primitives
        Live capability and refusal operations from the owning entry point.

    Returns
    -------
    object
        Native tensor returned by TensorFlow without host materialisation.

    Raises
    ------
    RuntimeError
        The namespace has no callable ``convert_to_tensor`` capability.

    """
    convert = primitives.attribute()(namespace, "convert_to_tensor", None)
    if not primitives.predicate()(convert):
        raise primitives.invalid()("TensorFlow module does not expose convert_to_tensor")
    dtype = primitives.attribute()(namespace, "float64", None)
    if dtype is None:
        return cast(NativeTensorFactory, convert)(values)
    return cast(NativeTensorFactory, convert)(values, dtype=dtype)


__all__ = [
    "NativeTensorFactory",
    "TensorAttributeLookup",
    "TensorConstructionPrimitives",
    "construct_tensorflow_tensor",
    "construct_torch_tensor",
    "prepare_torch_trainable_tensor",
]
