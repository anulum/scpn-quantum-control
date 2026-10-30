# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch module and optimiser state operations
"""Describe and compare PyTorch state entries without loading them.

A candidate ``state_dict`` is compared with a module's current state by key,
shape and dtype. The comparison selects no model, builds no report and loads no
framework; the shape/dtype reader can be supplied as a resolver so a caller
keeps its own named operation live.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeAlias

import numpy as np

Resolver: TypeAlias = Callable[[], Callable[..., Any]]
"""Zero-argument callable returning the operation to call at that moment."""
StateMismatch: TypeAlias = tuple[str, tuple[int, ...], tuple[int, ...], str, str]
"""Key, expected shape, observed shape, expected dtype and observed dtype."""


def tensor_descriptor(
    value: object, *, numpy: Callable[[], Any] | None = None
) -> tuple[tuple[int, ...], str]:
    """Return shape and dtype metadata for a tensor-like state entry.

    Parameters
    ----------
    value
        Native tensor, or host array-like value without a ``shape``.
    numpy
        Resolver of the NumPy namespace used for host values; this module's
        NumPy by default. It is read only for a value without ``shape``.

    Returns
    -------
    tuple[tuple[int, ...], str]
        Integer shape and dtype name.

    """
    shape = getattr(value, "shape", None)
    if shape is None:
        array = (numpy or (lambda: np))().asarray(value)
        return tuple(int(axis) for axis in array.shape), str(array.dtype)
    dtype = getattr(value, "dtype", None)
    return tuple(int(axis) for axis in shape), str(dtype)


def state_key_partition(
    expected_state: Mapping[str, object], state_dict: Mapping[str, object]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return observed, expected, missing and unexpected state keys.

    Parameters
    ----------
    expected_state
        The module's current state entries.
    state_dict
        Candidate state mapping.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]
        Observed keys, expected keys, expected keys absent from the candidate,
        and candidate keys absent from the module.

    """
    observed_keys = tuple(str(key) for key in state_dict)
    expected_keys = tuple(expected_state)
    missing_keys = tuple(key for key in expected_keys if key not in state_dict)
    unexpected_keys = tuple(key for key in observed_keys if key not in expected_state)
    return observed_keys, expected_keys, missing_keys, unexpected_keys


def iter_state_mismatches(
    expected_state: Mapping[str, object],
    state_dict: Mapping[str, object],
    *,
    descriptor: Resolver | None = None,
) -> Iterator[StateMismatch]:
    """Yield shape or dtype mismatches of shared keys, one key at a time.

    Each mismatch is yielded right after that key's two descriptors are read,
    so a consumer can act on it before the next key is described.

    Parameters
    ----------
    expected_state
        The module's current state entries.
    state_dict
        Candidate state mapping.
    descriptor
        Resolver of the shape/dtype reader; :func:`tensor_descriptor` by default.

    Yields
    ------
    StateMismatch
        Key, expected and observed shapes, and expected and observed dtypes.

    """
    describe = descriptor or (lambda: tensor_descriptor)
    for key in tuple(expected_state):
        if key not in state_dict:
            continue
        expected_shape, expected_dtype = describe()(expected_state[key])
        observed_shape, observed_dtype = describe()(state_dict[key])
        if expected_shape != observed_shape or expected_dtype != observed_dtype:
            yield key, expected_shape, observed_shape, expected_dtype, observed_dtype


__all__ = ["StateMismatch", "iter_state_mismatches", "state_key_partition", "tensor_descriptor"]
