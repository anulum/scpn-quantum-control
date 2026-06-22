# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the stable-core preflight helper
"""Branch tests for the stable-core preflight dependency probe."""

from __future__ import annotations

from scpn_quantum_control.stable_core_preflight import _dependency_available


def test_dependency_available_for_importable_module() -> None:
    """An importable module is reported as available."""
    assert _dependency_available("os") is True


def test_dependency_available_false_for_missing_module() -> None:
    """A missing module is reported as unavailable rather than raising."""
    assert _dependency_available("definitely_not_a_real_module_xyz") is False
