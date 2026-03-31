# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Crypto Exports
"""Multi-angle tests for crypto subpackage exports.

Verifies: all __all__ entries importable, callable where expected,
no circular imports, key functions callable, export count stable.
"""

from __future__ import annotations

import scpn_quantum_control.crypto as crypto


class TestCryptoExports:
    def test_all_exists(self):
        assert hasattr(crypto, "__all__")
        assert len(crypto.__all__) > 0

    def test_all_entries_importable(self):
        """Every name in __all__ must be accessible via getattr."""
        for name in crypto.__all__:
            obj = getattr(crypto, name)
            assert obj is not None, f"{name} resolved to None"

    def test_functions_are_callable(self):
        """Non-constant exports must be callable."""
        for name in crypto.__all__:
            if name.isupper():
                continue  # constants like EIGENVALUE_ZERO_ATOL
            obj = getattr(crypto, name)
            assert callable(obj), f"{name} is not callable"

    def test_spectral_fingerprint_importable(self):
        from scpn_quantum_control.crypto import spectral_fingerprint

        assert callable(spectral_fingerprint)

    def test_key_hierarchy_importable(self):
        from scpn_quantum_control.crypto import key_hierarchy

        assert callable(key_hierarchy)

    def test_scpn_qkd_protocol_importable(self):
        from scpn_quantum_control.crypto import scpn_qkd_protocol

        assert callable(scpn_qkd_protocol)

    def test_topology_commitment_importable(self):
        from scpn_quantum_control.crypto import topology_commitment

        assert callable(topology_commitment)

    def test_no_duplicate_exports(self):
        assert len(crypto.__all__) == len(set(crypto.__all__))

    def test_export_count_at_least_20(self):
        """Crypto module should export ≥20 symbols."""
        assert len(crypto.__all__) >= 20

    def test_constants_are_numeric(self):
        """UPPERCASE exports should be numeric constants."""
        for name in crypto.__all__:
            if name.isupper():
                obj = getattr(crypto, name)
                assert isinstance(obj, (int, float)), f"{name} is not numeric"
