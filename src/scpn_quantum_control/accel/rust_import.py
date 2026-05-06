# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust Import Policy
"""Import policy for the optional Rust acceleration extension."""

from __future__ import annotations

from types import ModuleType


def optional_rust_engine() -> ModuleType | None:
    """Return the optional Rust extension or ``None`` when it is absent.

    Only true absence of ``scpn_quantum_engine`` is optional. Import failures
    raised while loading an installed extension are propagated so broken wheels,
    missing native libraries, and incompatible binary builds are visible.
    """
    try:
        import scpn_quantum_engine
    except ModuleNotFoundError as exc:
        if exc.name == "scpn_quantum_engine":
            return None
        raise
    return scpn_quantum_engine
