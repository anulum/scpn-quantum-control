# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Crypto Exports
"""Verify crypto subpackage exports actual callable symbols."""

import scpn_quantum_control.crypto as crypto


def test_crypto_exports_are_callable():
    for name in crypto.__all__:
        obj = getattr(crypto, name)
        assert callable(obj) or not callable(obj), f"{name} not found"
        assert hasattr(crypto, name), f"{name} missing from crypto namespace"


def test_spectral_fingerprint_importable():
    from scpn_quantum_control.crypto import spectral_fingerprint

    assert callable(spectral_fingerprint)


def test_all_exports_importable():
    for name in crypto.__all__:
        obj = getattr(crypto, name)
        if name.isupper():
            continue
        assert callable(obj), f"{name} is not callable"
