# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the backend dispatch availability probe
"""Availability-probe fallback test for the array-backend dispatcher."""

from __future__ import annotations

import sys

import pytest

from scpn_quantum_control.backend_dispatch import available_backends


def test_available_backends_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """When JAX cannot be imported it is omitted from the available backends."""
    monkeypatch.setitem(sys.modules, "jax", None)
    backends = available_backends()
    assert "numpy" in backends
    assert "jax" not in backends
