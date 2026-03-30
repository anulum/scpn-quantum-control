# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Cirq Adapter
"""Tests for Cirq backend adapter."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.cirq_adapter import is_cirq_available


class TestCirqAdapterImport:
    """Tests that work regardless of cirq availability."""

    def test_is_cirq_available_type(self):
        assert isinstance(is_cirq_available(), bool)

    def test_import_does_not_crash(self):
        from scpn_quantum_control.hardware import cirq_adapter  # noqa: F401

        assert True


@pytest.mark.skipif(not is_cirq_available(), reason="Cirq not installed")
class TestCirqRunner:
    def test_placeholder(self):
        assert is_cirq_available()
