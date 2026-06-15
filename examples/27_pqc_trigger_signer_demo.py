# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ML-DSA-65 trigger signer demo
"""Sign and verify a capacitor-bank discharge command with ML-DSA-65 (FIPS 204).

Run with::

    python examples/27_pqc_trigger_signer_demo.py
"""

from __future__ import annotations

import time

from scpn_quantum_control.crypto.pqc_trigger import PqcTriggerSigner, _canonical_trigger_payload


def main() -> None:
    signer = PqcTriggerSigner()
    pk, sk = signer.keygen()
    print(f"public key  : {len(pk.key_bytes)} bytes ({pk.algorithm})")
    print(f"secret key  : {len(sk.key_bytes)} bytes")

    timestamp_ns = time.time_ns()
    sig = signer.sign_capacitor_bank_trigger("pulse-001", 24_500.0, timestamp_ns, sk)
    print(f"signature   : {len(sig.signature_bytes)} bytes")

    payload = _canonical_trigger_payload("pulse-001", 24_500.0, timestamp_ns)
    print(f"verify      : {signer.verify(payload, sig, pk)}")

    tampered = _canonical_trigger_payload("pulse-001", 30_000.0, timestamp_ns)
    print(f"tampered    : {signer.verify(tampered, sig, pk)}")

    fresh = signer.verify(payload, sig, pk, max_age_ns=10_000_000, now_ns=timestamp_ns + 5_000_000)
    stale = signer.verify(
        payload, sig, pk, max_age_ns=10_000_000, now_ns=timestamp_ns + 20_000_000
    )
    print(f"within 10 ms: {fresh}   stale: {stale}")


if __name__ == "__main__":
    main()
