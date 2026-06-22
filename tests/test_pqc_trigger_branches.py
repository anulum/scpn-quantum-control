# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the PQC trigger signer
"""Guard tests for the post-quantum trigger signer.

Covers the signing algorithm guard, the verification algorithm-mismatch
rejection and the canonical-payload pulse-id type guard.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from scpn_quantum_control.crypto.pqc_trigger import (
    PqcTriggerSigner,
    PrivateKey,
    PublicKey,
    Signature,
    _canonical_trigger_payload,
)


def test_sign_rejects_wrong_private_key_algorithm() -> None:
    """Signing with a non-ML-DSA private key is rejected."""
    signer = PqcTriggerSigner(deterministic=True)
    bad_key = PrivateKey(algorithm="bogus-alg", key_bytes=b"x")
    with pytest.raises(ValueError, match="private key algorithm must be"):
        signer.sign(b"payload", bad_key)


def test_verify_rejects_algorithm_mismatch() -> None:
    """Verifying a signature with a foreign algorithm returns False."""
    signer = PqcTriggerSigner(deterministic=True)
    signature = Signature(algorithm="bogus-alg", signature_bytes=b"s", timestamp_ns=0)
    public_key = PublicKey(algorithm="bogus-alg", key_bytes=b"p")
    assert signer.verify(b"payload", signature, public_key) is False


def test_canonical_payload_rejects_non_string_pulse_id() -> None:
    """A non-string pulse id is rejected by the canonical payload builder."""
    with pytest.raises(TypeError, match="pulse_id must be a string"):
        _canonical_trigger_payload(cast(Any, 123), 1.0, 1000)
