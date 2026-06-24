# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the ML-DSA-65 honesty-seal back-end (WS-1)
"""Tests for crypto/ml_dsa_seal.py — the post-quantum seal back-end.

Two layers: the back-end in isolation (protocol shape, key custody, signature
determinism, context binding) and its integration through the platform seal
(``seal`` → ``verify`` yielding the full VERIFIED / STRIPPED / FORGED / UNGRADED
lattice with an ML-DSA-65 key).
"""

from __future__ import annotations

import base64

import pytest

from scpn_quantum_control.crypto import ml_dsa
from scpn_quantum_control.crypto.ml_dsa_seal import (
    ALG,
    SEAL_CONTEXT,
    MLDSASigner,
    MLDSAVerifier,
)

_SEED_A = bytes(range(32))
_SEED_B = bytes(range(1, 33))
_KEY_ID = "scpn-quantum-control:2026-q2"


def _signer(seed: bytes = _SEED_A, key_id: str = _KEY_ID) -> MLDSASigner:
    return MLDSASigner.generate(key_id, seed=seed)


# ── back-end in isolation ──────────────────────────────────────────────


def test_parameter_set_is_ml_dsa_65() -> None:
    """The back-end is wired to the ML-DSA-65 parameter set, not another level."""
    assert ALG == "ML-DSA-65"
    assert MLDSASigner.alg == "ML-DSA-65"
    assert MLDSAVerifier.alg == "ML-DSA-65"
    assert ml_dsa.PUBLIC_KEY_BYTES == 1952
    assert ml_dsa.SIGNATURE_BYTES == 3309


def test_generate_is_deterministic_in_the_seed() -> None:
    """The same seed yields the same key; a different seed yields a different key."""
    a1 = _signer(_SEED_A).public_bytes()
    a2 = _signer(_SEED_A).public_bytes()
    b = _signer(_SEED_B).public_bytes()
    assert a1 == a2
    assert a1 != b
    assert len(a1) == ml_dsa.PUBLIC_KEY_BYTES


def test_sign_is_deterministic_and_round_trips() -> None:
    """A signed message verifies, and signing is byte-stable across runs."""
    signer = _signer()
    message = b"canonical-claim-unit-bytes"
    sig1 = signer.sign(message)
    sig2 = signer.sign(message)
    assert sig1 == sig2
    assert len(sig1) == ml_dsa.SIGNATURE_BYTES
    assert signer.verifier().verify(message, sig1) is True


def test_verify_rejects_a_tampered_message() -> None:
    """A signature is bound to its exact message."""
    signer = _signer()
    sig = signer.sign(b"message-one")
    assert signer.verifier().verify(b"message-two", sig) is False


def test_verify_rejects_a_wrong_length_signature() -> None:
    """A malformed (wrong-length) signature is rejected, never raised on."""
    verifier = _signer().verifier()
    assert verifier.verify(b"m", b"too-short") is False
    assert verifier.verify(b"m", bytes(ml_dsa.SIGNATURE_BYTES)) is False


def test_verify_rejects_a_signature_from_another_key() -> None:
    """A signature from one key does not verify under another's public key."""
    message = b"shared-message"
    sig = _signer(_SEED_A).sign(message)
    assert _signer(_SEED_B).verifier().verify(message, sig) is False


def test_seal_context_binds_the_signature_domain() -> None:
    """A signature made under a different context is invalid for the seal verifier.

    The signer mixes :data:`SEAL_CONTEXT` into every signature; a signature produced
    for another protocol (different context, same key) must not verify here.
    """
    signer = _signer()
    message = b"claim"
    # Sign the same message under a *different* context with the raw primitive and
    # the same key; it must not verify through the seal verifier.
    other_context_sig = ml_dsa.sign(
        ml_dsa.key_gen(_SEED_A).secret_key, message, context=b"other.protocol.v1"
    )
    assert signer.verifier().verify(message, other_context_sig) is False
    # A signature made under the seal context (what the signer emits) does verify.
    assert signer.verifier().verify(message, signer.sign(message)) is True
    # Sanity: the signer's own signature equals the raw primitive under SEAL_CONTEXT.
    raw_seal_sig = ml_dsa.sign(ml_dsa.key_gen(_SEED_A).secret_key, message, context=SEAL_CONTEXT)
    assert signer.sign(message) == raw_seal_sig


def test_verifier_rejects_wrong_length_public_key() -> None:
    """Constructing a verifier from a malformed public key fails closed."""
    with pytest.raises(ValueError, match="public_key must be"):
        MLDSAVerifier(b"\x00" * 10)


def test_public_bytes_round_trips_through_a_fresh_verifier() -> None:
    """A verifier rebuilt from published public bytes verifies the same signatures."""
    signer = _signer()
    message = b"published-key-path"
    sig = signer.sign(message)
    rebuilt = MLDSAVerifier(signer.public_bytes())
    assert rebuilt.verify(message, sig) is True
    # The verifier republishes the identical public key it was built from.
    assert rebuilt.public_bytes() == signer.public_bytes()


def test_signer_rejects_empty_key_id() -> None:
    """An empty or whitespace key_id is rejected — every envelope needs an identity."""
    keypair = ml_dsa.key_gen(_SEED_A)
    with pytest.raises(ValueError, match="key_id must be"):
        MLDSASigner("   ", keypair)
    with pytest.raises(ValueError, match="key_id must be"):
        MLDSASigner.generate("", seed=_SEED_A)


def test_generate_propagates_seed_length_error() -> None:
    """A wrong-length seed is rejected by the underlying key generation."""
    with pytest.raises(ValueError, match="seed must be"):
        MLDSASigner.generate(_KEY_ID, seed=b"short")


def test_key_id_is_recorded() -> None:
    """The signer exposes the stable identifier the envelope records."""
    assert _signer(key_id="studio:k1").key_id == "studio:k1"


# ── integration through the platform honesty seal ──────────────────────

_seal_mod = pytest.importorskip("scpn_studio_platform.seal", reason="studio extra not installed")


def _keyring(signer: MLDSASigner):
    from scpn_studio_platform.seal import Keyring

    ring = Keyring()
    ring.add(signer.key_id, signer.verifier())
    return ring


_UNIT = {
    "schema": "studio.hardware-result-pack.v1",
    "claim_status": "noise-limited",
    "evidence_kind": "measured",
}
_GRADER = {"name": "honesty-bridge", "version": "0.8.0"}
_ATTESTATION = {
    "provider": "ibm",
    "result_pack_digest": "sha256:" + "ab" * 32,
    "provider_sig": "ZGVhZGJlZWY=",
}


def _grade(_unit) -> str:
    return "noise-limited"


def test_backend_satisfies_platform_protocols() -> None:
    """``MLDSASigner`` / ``MLDSAVerifier`` structurally satisfy the seal protocols."""
    from scpn_studio_platform.seal import Signer, Verifier

    signer = _signer()
    assert isinstance(signer, Signer)
    assert isinstance(signer.verifier(), Verifier)


def test_sealed_unit_verifies_end_to_end() -> None:
    """A unit sealed with the ML-DSA back-end verifies through the platform verdict."""
    from scpn_studio_platform.seal import Verdict, seal, verify

    signer = _signer()
    envelope = seal(
        _UNIT,
        signer=signer,
        grader=_GRADER,
        verifiability_mode="attestation",
        exactness_class="bit-exact",
        attestation=_ATTESTATION,
    )
    assert envelope.signature["alg"] == "ML-DSA-65"
    verdict = verify(envelope.to_dict(), "noise-limited", keyring=_keyring(signer), regrade=_grade)
    assert verdict is Verdict.VERIFIED


def test_recompute_mode_seal_verifies() -> None:
    """The back-end also seals recompute-mode units (no attestation)."""
    from scpn_studio_platform.seal import Verdict, seal, verify

    signer = _signer()
    envelope = seal(
        _UNIT,
        signer=signer,
        grader=_GRADER,
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    )
    verdict = verify(envelope.to_dict(), "noise-limited", keyring=_keyring(signer), regrade=_grade)
    assert verdict is Verdict.VERIFIED


def test_tampered_unit_is_forged() -> None:
    """Rewriting the sealed unit after signing is caught as a forgery."""
    from scpn_studio_platform.seal import Verdict, seal, verify

    signer = _signer()
    envelope = seal(
        _UNIT,
        signer=signer,
        grader=_GRADER,
        verifiability_mode="attestation",
        exactness_class="bit-exact",
        attestation=_ATTESTATION,
    ).to_dict()
    envelope["unit"] = {**_UNIT, "claim_status": "reference-validated"}
    verdict = verify(
        envelope,
        "reference-validated",
        keyring=_keyring(signer),
        regrade=lambda _u: "reference-validated",
    )
    assert verdict is Verdict.FORGED


def test_corrupted_signature_is_forged() -> None:
    """Flipping a byte of the signature is caught."""
    from scpn_studio_platform.seal import Verdict, seal, verify

    signer = _signer()
    envelope = seal(
        _UNIT,
        signer=signer,
        grader=_GRADER,
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    ).to_dict()
    raw = bytearray(base64.b64decode(envelope["signature"]["sig"]))
    raw[0] ^= 0x01
    envelope["signature"]["sig"] = base64.b64encode(bytes(raw)).decode("ascii")
    verdict = verify(envelope, "noise-limited", keyring=_keyring(signer), regrade=_grade)
    assert verdict is Verdict.FORGED


def test_unknown_key_is_forged() -> None:
    """A signature whose key_id is not in the keyring is not admitted."""
    from scpn_studio_platform.seal import Keyring, Verdict, seal, verify

    signer = _signer()
    envelope = seal(
        _UNIT,
        signer=signer,
        grader=_GRADER,
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    ).to_dict()
    empty_ring = Keyring()
    verdict = verify(envelope, "noise-limited", keyring=empty_ring, regrade=_grade)
    assert verdict is Verdict.FORGED


def test_grade_disagreement_is_forged() -> None:
    """An authentic signature over axes that imply a *different* grade is forged."""
    from scpn_studio_platform.seal import Verdict, seal, verify

    signer = _signer()
    envelope = seal(
        _UNIT,
        signer=signer,
        grader=_GRADER,
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    ).to_dict()
    # Page renders 'reference-validated' but the signed unit grades to 'noise-limited'.
    verdict = verify(envelope, "reference-validated", keyring=_keyring(signer), regrade=_grade)
    assert verdict is Verdict.FORGED


def test_stripped_and_ungraded_states() -> None:
    """A rendered grade with no envelope is STRIPPED; a bare number is UNGRADED."""
    from scpn_studio_platform.seal import Verdict, verify

    ring = _keyring(_signer())
    assert verify(None, "validated", keyring=ring, regrade=_grade) is Verdict.STRIPPED
    assert verify(None, None, keyring=ring, regrade=_grade) is Verdict.UNGRADED
