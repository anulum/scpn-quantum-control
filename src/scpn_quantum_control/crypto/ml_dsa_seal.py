# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ML-DSA-65 back-end for the verifiable-honesty seal (WS-1)
"""Post-quantum signing back-end for the studio honesty seal (WS-1).

The SCPN-STUDIO honesty seal (``scpn_studio_platform.seal``) is algorithm-agnostic:
a producer signs the canonical bytes of a claim unit with any back-end that exposes
``sign(message) -> bytes`` plus ``key_id`` / ``alg``, and a verifier holding the
public key checks it. The platform ships **Ed25519** as the reference back-end and
names **ML-DSA-65** (FIPS 204, post-quantum) as the WS-1 target, to slot in behind
the same protocol *once a vetted library is selected*.

This module is that back-end. It wraps :mod:`scpn_quantum_control.crypto.ml_dsa`
— a from-specification ML-DSA-65 implementation reproducing the official NIST ACVP
known-answer vectors — as :class:`MLDSASigner` / :class:`MLDSAVerifier`, structurally
satisfying the platform's ``Signer`` / ``Verifier`` protocols. Every seal signature
is taken under a fixed domain-separation context (:data:`SEAL_CONTEXT`) so a
seal signature can never be replayed as a signature for another protocol that reuses
the same key, and vice versa.

Key custody is the caller's: :meth:`MLDSASigner.generate` takes an explicit 32-byte
seed (from a hardware RNG or a key-management system in production, a fixed vector in
tests) — the back-end never sources its own randomness, so a key is reproducible from
its seed and nothing is silently fabricated. The signing itself is ML-DSA's
deterministic (hedged-off) mode, so a unit seals to a byte-identical signature on
every run, which is what makes the seal independently recomputable.

This is *signing* assurance (the grade is the studio's own, unforged); it is not a
FIPS-140-validated module and carries no side-channel-resistance guarantee — the same
boundary :mod:`scpn_quantum_control.crypto.ml_dsa` states.
"""

from __future__ import annotations

from typing import Final

from .ml_dsa import (
    PUBLIC_KEY_BYTES,
    SIGNATURE_BYTES,
    MLDSAKeyPair,
    key_gen,
    sign,
    verify,
)

ALG: Final = "ML-DSA-65"
"""The signature-algorithm name recorded in every envelope this back-end seals."""

SEAL_CONTEXT: Final = b"scpn.studio.honesty-seal.v1"
"""Domain-separation context bound into every seal signature.

ML-DSA's context string is mixed into the signed message, so a signature produced
for the honesty seal is invalid under any other context and a signature produced for
another protocol is invalid here. Signer and verifier must use the identical
context; both sides of this module do, so the binding is invisible to the platform's
``Signer`` / ``Verifier`` protocol (which sees only ``message -> bytes`` and
``(message, signature) -> bool``).
"""


class MLDSAVerifier:
    """ML-DSA-65 public-key verifier satisfying the platform ``Verifier`` protocol.

    Parameters
    ----------
    public_key
        The 1952-byte ML-DSA-65 public key (FIPS 204 ``pk`` encoding).

    Raises
    ------
    ValueError
        If ``public_key`` is not exactly :data:`PUBLIC_KEY_BYTES` long.
    """

    alg: Final = ALG

    def __init__(self, public_key: bytes) -> None:
        if len(public_key) != PUBLIC_KEY_BYTES:
            raise ValueError(f"public_key must be {PUBLIC_KEY_BYTES} bytes, got {len(public_key)}")
        self._public_key = bytes(public_key)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Return ``True`` iff ``signature`` is a valid seal signature of ``message``.

        A malformed signature, a wrong-length signature, or a signature produced
        under a different context returns ``False`` — a verifier on an untrusted page
        reports a verdict, it never raises.

        Parameters
        ----------
        message
            The canonical bytes the signature is taken over.
        signature
            The detached ML-DSA-65 signature to check.
        """
        if len(signature) != SIGNATURE_BYTES:
            return False
        return verify(self._public_key, message, signature, context=SEAL_CONTEXT)

    def public_bytes(self) -> bytes:
        """Return the raw 1952-byte ML-DSA-65 public key (for keyring publication)."""
        return self._public_key


class MLDSASigner:
    """ML-DSA-65 private-key signer satisfying the platform ``Signer`` protocol.

    Construct via :meth:`generate` from a 32-byte seed rather than directly; the seed
    is the key's reproducible root and must come from the caller (a secure RNG or a
    key-management system), never from inside this module.

    Parameters
    ----------
    key_id
        The stable identifier recorded in every envelope this signer seals, by
        convention ``"<studio>:<keyid>"`` (e.g. ``"scpn-quantum-control:2026-q2"``).
    keypair
        The ML-DSA-65 key pair backing the signer.

    Raises
    ------
    ValueError
        If ``key_id`` is empty or whitespace.
    """

    alg: Final = ALG

    def __init__(self, key_id: str, keypair: MLDSAKeyPair) -> None:
        if not key_id.strip():
            raise ValueError("key_id must be a non-empty identifier")
        self._key_id = key_id
        self._keypair = keypair

    @property
    def key_id(self) -> str:
        """The stable identifier recorded in the envelope."""
        return self._key_id

    @classmethod
    def generate(cls, key_id: str, *, seed: bytes) -> MLDSASigner:
        """Create a signer whose key is deterministically derived from ``seed``.

        Parameters
        ----------
        key_id
            The stable identifier (``"<studio>:<keyid>"``) recorded in envelopes
            this signer seals.
        seed
            A 32-byte secret seed. The same seed always yields the same key, so the
            signer is reproducible; supply it from a secure RNG or a key-management
            system in production and a fixed vector in tests.

        Raises
        ------
        ValueError
            If ``seed`` is not exactly 32 bytes long (``key_gen`` enforces this) or
            ``key_id`` is empty.
        """
        return cls(key_id, key_gen(seed))

    def sign(self, message: bytes) -> bytes:
        """Return the detached ML-DSA-65 seal signature over ``message``.

        Deterministic: the same key and message always produce byte-identical output,
        which is what lets a third party recompute and compare the signature.

        Parameters
        ----------
        message
            The canonical bytes to sign.
        """
        return sign(self._keypair.secret_key, message, context=SEAL_CONTEXT)

    def verifier(self) -> MLDSAVerifier:
        """Return the public :class:`MLDSAVerifier` for this signer's key."""
        return MLDSAVerifier(self._keypair.public_key)

    def public_bytes(self) -> bytes:
        """Return the raw 1952-byte ML-DSA-65 public key (for keyring publication)."""
        return self._keypair.public_key


__all__ = [
    "ALG",
    "SEAL_CONTEXT",
    "MLDSASigner",
    "MLDSAVerifier",
]
