# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for ML-DSA-65 and the PQC trigger signer (QUA-C.2)
"""Tests for crypto/ml_dsa.py (FIPS 204) and crypto/pqc_trigger.py."""

import json
import random
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.crypto import ml_dsa
from scpn_quantum_control.crypto.pqc_trigger import PqcTriggerSigner, _canonical_trigger_payload

try:
    import scpn_quantum_engine as _engine

    _HAS_RUST = hasattr(_engine, "ml_dsa_ntt")
except ImportError:  # pragma: no cover - engine optional
    _engine = None
    _HAS_RUST = False

_KAT = json.loads((Path(__file__).parent / "data" / "ml_dsa_65_kat.json").read_text())


# --------------------------------------------------------------------------- #
# NIST ACVP known-answer vectors (FIPS 204 conformance)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _KAT["keyGen"], ids=lambda c: f"keyGen-{c['tcId']}")
def test_keygen_kat(case):
    pair = ml_dsa.key_gen(bytes.fromhex(case["seed"]))
    assert pair.public_key.hex().upper() == case["pk"].upper()
    assert pair.secret_key.hex().upper() == case["sk"].upper()


@pytest.mark.parametrize("case", _KAT["sigGen"], ids=lambda c: f"sigGen-{c['tcId']}")
def test_siggen_kat(case):
    sig = ml_dsa.sign(
        bytes.fromhex(case["sk"]),
        bytes.fromhex(case["message"]),
        context=bytes.fromhex(case["context"]),
    )
    assert sig.hex().upper() == case["signature"].upper()


@pytest.mark.parametrize("case", _KAT["sigVer"], ids=lambda c: f"sigVer-{c['tcId']}")
def test_sigver_kat(case):
    result = ml_dsa.verify(
        bytes.fromhex(case["pk"]),
        bytes.fromhex(case["message"]),
        bytes.fromhex(case["signature"]),
        context=bytes.fromhex(case["context"]),
    )
    assert result is case["testPassed"]


# --------------------------------------------------------------------------- #
# NTT correctness and Rust parity
# --------------------------------------------------------------------------- #
def test_ntt_roundtrip():
    rng = random.Random(1)
    poly = [rng.randrange(ml_dsa.Q) for _ in range(256)]
    assert ml_dsa._intt_python(ml_dsa._ntt_python(poly)) == poly


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine ml_dsa kernel not built")
@settings(max_examples=40, deadline=None)
@given(coeffs=st.lists(st.integers(-ml_dsa.Q, ml_dsa.Q), min_size=256, max_size=256))
def test_ntt_rust_parity(coeffs):
    assert list(_engine.ml_dsa_ntt(coeffs)) == ml_dsa._ntt_python(coeffs)
    assert list(_engine.ml_dsa_intt(coeffs)) == ml_dsa._intt_python(coeffs)


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine ml_dsa kernel not built")
def test_ntt_rust_rejects_wrong_length():
    with pytest.raises(ValueError):
        _engine.ml_dsa_ntt([0] * 255)


# --------------------------------------------------------------------------- #
# ML-DSA round-trip and error paths
# --------------------------------------------------------------------------- #
def test_sign_verify_roundtrip():
    pair = ml_dsa.key_gen(bytes(range(32)))
    sig = ml_dsa.sign(pair.secret_key, b"capacitor-bank discharge", context=b"ctx")
    assert ml_dsa.verify(pair.public_key, b"capacitor-bank discharge", sig, context=b"ctx")
    assert not ml_dsa.verify(pair.public_key, b"tampered", sig, context=b"ctx")
    assert not ml_dsa.verify(pair.public_key, b"capacitor-bank discharge", sig, context=b"other")


def test_keygen_rejects_bad_seed():
    with pytest.raises(ValueError):
        ml_dsa.key_gen(bytes(31))


def test_verify_rejects_wrong_sizes():
    pair = ml_dsa.key_gen(bytes(32))
    assert not ml_dsa.verify(b"short", b"m", bytes(ml_dsa.SIGNATURE_BYTES))
    assert not ml_dsa.verify(pair.public_key, b"m", b"short")


@settings(max_examples=15, deadline=None)
@given(message=st.binary(min_size=0, max_size=128), context=st.binary(min_size=0, max_size=64))
def test_roundtrip_property(message, context):
    pair = ml_dsa.key_gen(bytes([7] * 32))
    sig = ml_dsa.sign(pair.secret_key, message, context=context)
    assert ml_dsa.verify(pair.public_key, message, sig, context=context)


# --------------------------------------------------------------------------- #
# PQC trigger signer
# --------------------------------------------------------------------------- #
def test_trigger_signer_roundtrip():
    signer = PqcTriggerSigner(deterministic=True)
    pk, sk = signer.keygen(seed=bytes(32))
    payload = b"arm-trigger"
    sig = signer.sign(payload, sk, timestamp_ns=1_000)
    assert signer.verify(payload, sig, pk)
    assert not signer.verify(b"arm-other", sig, pk)


def test_trigger_deterministic_keygen():
    a = PqcTriggerSigner(deterministic=True).keygen(seed=bytes([3] * 32))
    b = PqcTriggerSigner(deterministic=True).keygen(seed=bytes([3] * 32))
    assert a[0].key_bytes == b[0].key_bytes
    assert a[1].key_bytes == b[1].key_bytes


def test_capacitor_bank_trigger_and_tamper():
    signer = PqcTriggerSigner(deterministic=True)
    pk, sk = signer.keygen(seed=bytes(32))
    sig = signer.sign_capacitor_bank_trigger("pulse-001", 24_500.0, 1_700_000_000, sk)
    payload = _canonical_trigger_payload("pulse-001", 24_500.0, 1_700_000_000)
    assert signer.verify(payload, sig, pk)
    # altered voltage / pulse id / timestamp all break verification
    assert not signer.verify(
        _canonical_trigger_payload("pulse-001", 30_000.0, 1_700_000_000), sig, pk
    )
    assert not signer.verify(
        _canonical_trigger_payload("pulse-002", 24_500.0, 1_700_000_000), sig, pk
    )


def test_trigger_freshness_window():
    signer = PqcTriggerSigner(deterministic=True)
    pk, sk = signer.keygen(seed=bytes(32))
    payload = b"arm"
    sig = signer.sign(payload, sk, timestamp_ns=1_000_000)
    # within window
    assert signer.verify(payload, sig, pk, max_age_ns=10_000, now_ns=1_005_000)
    # stale
    assert not signer.verify(payload, sig, pk, max_age_ns=10_000, now_ns=2_000_000)
    # future-dated (negative age)
    assert not signer.verify(payload, sig, pk, max_age_ns=10_000, now_ns=500_000)


def test_canonical_payload_is_sorted_and_stable():
    p1 = _canonical_trigger_payload("p", 1.5, 42)
    p2 = _canonical_trigger_payload("p", 1.5, 42)
    assert p1 == p2
    assert p1 == b'{"pulse_id":"p","timestamp_ns":42,"voltage_V":1.5}'
