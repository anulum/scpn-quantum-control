# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the persistent-homology ripser fallback
"""Module-load fallback test for the optional ripser dependency."""

from __future__ import annotations

import importlib
import sys

from scpn_quantum_control.analysis import persistent_homology


def test_persistent_homology_disables_ripser_when_absent() -> None:
    """Reloading the module without ripser disables the availability flag.

    The module's top-level import guard sets ``_RIPSER_AVAILABLE`` to False when
    ripser cannot be imported; the module is restored afterwards.
    """
    original = sys.modules.get("ripser")
    sys.modules["ripser"] = None  # type: ignore[assignment]
    try:
        importlib.reload(persistent_homology)
        assert persistent_homology._RIPSER_AVAILABLE is False
    finally:
        if original is not None:
            sys.modules["ripser"] = original
        else:
            sys.modules.pop("ripser", None)
        importlib.reload(persistent_homology)
