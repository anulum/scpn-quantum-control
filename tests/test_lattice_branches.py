# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the psi-field lattice engine fallback
"""Module-load fallback test for the psi-field gauge-lattice native import."""

from __future__ import annotations

import importlib
import sys

from scpn_quantum_control.psi_field import lattice


def test_lattice_disables_rust_gauge_without_engine() -> None:
    """Reloading the module without the native engine disables the Rust gauge flag.

    The module's top-level import guard sets ``_HAS_RUST_GAUGE`` to False when
    ``scpn_quantum_engine`` is unavailable; the module is restored afterwards.
    """
    original = sys.modules.get("scpn_quantum_engine")
    sys.modules["scpn_quantum_engine"] = None  # type: ignore[assignment]
    try:
        importlib.reload(lattice)
        assert lattice._HAS_RUST_GAUGE is False
    finally:
        if original is not None:
            sys.modules["scpn_quantum_engine"] = original
        else:
            sys.modules.pop("scpn_quantum_engine", None)
        importlib.reload(lattice)
