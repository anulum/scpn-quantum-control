# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace canonical encoding
"""Preserve the established workspace API through the shared contracts codec.

Canonical bytes, SHA-256 domains, resource limits and refusal semantics are
owned by :mod:`scpn_quantum_control.canonical_encoding`. The workspace functions
retain their defining names and pickle paths while delegating to that codec.
"""

from ..canonical_encoding import (
    MAX_DEPTH,
    MAX_INTEGER_DIGITS,
)
from ..canonical_encoding import (
    canonical_bytes as _canonical_bytes,
)
from ..canonical_encoding import (
    canonical_digest as _canonical_digest,
)


def canonical_bytes(schema: str, body: object) -> bytes:
    """Encode workspace values through the shared canonical codec.

    Parameters
    ----------
    schema
        Nonempty scalar string without CR or LF.
    body
        Finite JSON-shaped values; integers and binary64 floats stay distinct.

    Returns
    -------
    bytes
        Schema, LF and compact UTF-8 tagged JSON.

    Raises
    ------
    ValueError
        Prefix, scalar, key, recursion depth or ancestor cycle is unsupported.

    """
    return _canonical_bytes(schema, body)


def canonical_digest(schema: str, body: object) -> str:
    """Hash workspace values through the shared canonical codec.

    Parameters
    ----------
    schema
        Nonempty schema domain without CR or LF.
    body
        Values admitted by ``canonical_bytes``.

    Returns
    -------
    str
        Lowercase SHA-256 hex without an algorithm prefix.

    Raises
    ------
    ValueError
        Prefix, scalar, key, recursion depth or ancestor cycle is unsupported.

    """
    return _canonical_digest(schema, body)


__all__ = ["MAX_DEPTH", "MAX_INTEGER_DIGITS", "canonical_bytes", "canonical_digest"]
