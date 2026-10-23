# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native tensor host materialization
"""Materialize native tensor values without selecting a model or host shape."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from .differentiable_tensor_construction import TensorAttributeLookup


@dataclass(frozen=True)
class TensorMaterializationPrimitives:
    """Read the caller's native capability operations lazily.

    Parameters
    ----------
    attribute
        Read the active attribute lookup before each capability lookup.
    predicate
        Read the active callable predicate before each capability test.

    """

    attribute: Callable[[], TensorAttributeLookup]
    predicate: Callable[[], Callable[[object], bool]]


def materialize_torch_host_values(
    values: object, *, primitives: TensorMaterializationPrimitives
) -> object:
    """Detach, transfer to CPU and expose host values when supported.

    Parameters
    ----------
    values
        Native tensor or host value accepted by the original entry point.
    primitives
        Live attribute and callable readers from the original owner.

    Returns
    -------
    object
        Last conversion result. Native storage sharing and exceptions persist.

    """
    candidate = values
    for name in ("detach", "cpu", "numpy"):
        method = primitives.attribute()(candidate, name, None)
        if primitives.predicate()(method):
            candidate = cast(Callable[[], object], method)()
    return candidate


def materialize_tensorflow_host_values(
    values: object, *, primitives: TensorMaterializationPrimitives
) -> object:
    """Expose eager TensorFlow host values when supported.

    Parameters
    ----------
    values
        Native tensor or host value accepted by the original entry point.
    primitives
        Live attribute and callable readers from the original owner.

    Returns
    -------
    object
        Native NumPy conversion result, or the unchanged input value.

    """
    method = primitives.attribute()(values, "numpy", None)
    if primitives.predicate()(method):
        return cast(Callable[[], object], method)()
    return values


__all__ = [
    "TensorMaterializationPrimitives",
    "materialize_tensorflow_host_values",
    "materialize_torch_host_values",
]
