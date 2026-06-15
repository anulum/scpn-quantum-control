# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Post-quantum capacitor-bank trigger signer
"""Post-quantum (FIPS 204 ML-DSA-65) signer for high-voltage trigger commands.

Authorises capacitor-bank discharge commands ahead of arming the combinatorial
trigger. The signed message binds the payload to its timestamp, so neither can be
altered without invalidating the signature; verification optionally enforces a
freshness window. Built on the from-specification ML-DSA-65 implementation in
:mod:`~scpn_quantum_control.crypto.ml_dsa` (validated against the NIST ACVP
known-answer vectors).
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass

from . import ml_dsa

_ALGORITHM = "ML-DSA-65"
# FIPS 204 context string for domain separation of this signing surface.
_CONTEXT = b"scpn-quantum-control/pqc-trigger"


@dataclass(frozen=True)
class PublicKey:
    """An ML-DSA-65 public key."""

    algorithm: str
    key_bytes: bytes


@dataclass(frozen=True)
class PrivateKey:
    """An ML-DSA-65 private key."""

    algorithm: str
    key_bytes: bytes


@dataclass(frozen=True)
class Signature:
    """A timestamped ML-DSA-65 signature."""

    algorithm: str
    signature_bytes: bytes
    timestamp_ns: int


def _signed_message(payload: bytes, timestamp_ns: int) -> bytes:
    return payload + int(timestamp_ns).to_bytes(8, "big")


class PqcTriggerSigner:
    """ML-DSA-65 signer for capacitor-bank trigger authorisation."""

    def __init__(self, *, deterministic: bool = False) -> None:
        self._deterministic = bool(deterministic)
        self.algorithm = _ALGORITHM

    def keygen(self, *, seed: bytes | None = None) -> tuple[PublicKey, PrivateKey]:
        """Generate a key pair (deterministic when ``seed`` is supplied)."""
        material = seed if seed is not None else secrets.token_bytes(ml_dsa.SEED_BYTES)
        pair = ml_dsa.key_gen(material)
        return (
            PublicKey(algorithm=_ALGORITHM, key_bytes=pair.public_key),
            PrivateKey(algorithm=_ALGORITHM, key_bytes=pair.secret_key),
        )

    def sign(
        self, payload: bytes, private_key: PrivateKey, *, timestamp_ns: int | None = None
    ) -> Signature:
        """Sign ``payload`` bound to a timestamp."""
        if private_key.algorithm != _ALGORITHM:
            raise ValueError(f"private key algorithm must be {_ALGORITHM}")
        ts = int(timestamp_ns) if timestamp_ns is not None else time.time_ns()
        randomness = None if self._deterministic else secrets.token_bytes(32)
        signature = ml_dsa.sign(
            private_key.key_bytes,
            _signed_message(payload, ts),
            context=_CONTEXT,
            randomness=randomness,
        )
        return Signature(algorithm=_ALGORITHM, signature_bytes=signature, timestamp_ns=ts)

    def verify(
        self,
        payload: bytes,
        signature: Signature,
        public_key: PublicKey,
        *,
        max_age_ns: int | None = None,
        now_ns: int | None = None,
    ) -> bool:
        """Verify a signature and (optionally) enforce a freshness window."""
        if signature.algorithm != _ALGORITHM or public_key.algorithm != _ALGORITHM:
            return False
        if max_age_ns is not None:
            current = int(now_ns) if now_ns is not None else time.time_ns()
            age = current - signature.timestamp_ns
            if age < 0 or age > max_age_ns:
                return False
        return ml_dsa.verify(
            public_key.key_bytes,
            _signed_message(payload, signature.timestamp_ns),
            signature.signature_bytes,
            context=_CONTEXT,
        )

    def sign_capacitor_bank_trigger(
        self,
        pulse_id: str,
        voltage_v: float,
        timestamp_ns: int,
        private_key: PrivateKey,
    ) -> Signature:
        """Sign a capacitor-bank discharge command with a canonical payload."""
        payload = _canonical_trigger_payload(pulse_id, voltage_v, timestamp_ns)
        return self.sign(payload, private_key, timestamp_ns=timestamp_ns)


def _canonical_trigger_payload(pulse_id: str, voltage_v: float, timestamp_ns: int) -> bytes:
    """JSON-canonical (sorted-key, fixed-field) trigger payload."""
    if not isinstance(pulse_id, str):
        raise TypeError("pulse_id must be a string")
    record = {
        "pulse_id": pulse_id,
        "timestamp_ns": int(timestamp_ns),
        "voltage_V": float(voltage_v),
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "PqcTriggerSigner",
    "PrivateKey",
    "PublicKey",
    "Signature",
]
