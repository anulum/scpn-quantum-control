# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared surrogate numeric identities
"""Bind surrogate training and validation data to one numeric custody rule."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_NUMERIC_CUSTODY_DECIMALS = 6


def _array_digest(values: FloatArray) -> str:
    """Return a shape-bound, cross-runtime SHA-256 identity for one float array."""
    rounded = np.round(np.asarray(values, dtype=np.float64), _NUMERIC_CUSTODY_DECIMALS)
    array = np.ascontiguousarray(np.where(rounded == 0.0, 0.0, rounded), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def input_row_digests(inputs: FloatArray) -> tuple[str, ...]:
    """Return one shape-bound SHA-256 identity per input row.

    Parameters
    ----------
    inputs
        Finite two-dimensional input matrix.

    Returns
    -------
    tuple[str, ...]
        Row identities in input order, with six-decimal numeric custody and
        signed-zero normalisation shared by fitting and validation.

    Raises
    ------
    ValueError
        If inputs are non-finite or are not a non-empty two-dimensional matrix.

    """
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("inputs must be a non-empty 2-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("inputs must contain only finite values.")
    return tuple(_array_digest(values[index]) for index in range(values.shape[0]))


# The public home of this function stays ``surrogates.train``, where it has
# always been importable: pickles, introspection and the recorded public export
# identity keep naming that module while fitting and validation share this body.
input_row_digests.__module__ = "scpn_quantum_control.surrogates.train"
