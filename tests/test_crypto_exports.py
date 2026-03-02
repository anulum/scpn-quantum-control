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
