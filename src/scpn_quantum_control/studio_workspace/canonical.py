# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace canonical encoding
"""Preserve the established workspace API through the shared contracts codec.

Canonical bytes, SHA-256 domains, resource limits and refusal semantics are
owned by :mod:`scpn_quantum_control.canonical_encoding`. These explicit aliases
retain the original workspace imports without duplicating the implementation.
"""

from ..canonical_encoding import (
    MAX_DEPTH,
    MAX_INTEGER_DIGITS,
    canonical_bytes,
    canonical_digest,
)

__all__ = ["MAX_DEPTH", "MAX_INTEGER_DIGITS", "canonical_bytes", "canonical_digest"]
