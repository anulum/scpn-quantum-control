# Post-Quantum Trigger Signer (ML-DSA-65)

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.crypto.ml_dsa` is a from-specification implementation of
the **FIPS 204 ML-DSA-65** module-lattice digital signature scheme, and
`scpn_quantum_control.crypto.pqc_trigger` builds a capacitor-bank trigger
authorisation signer on top of it.

This is FIPS 204-**conformant** — it reproduces the official NIST ACVP
known-answer vectors (keyGen, deterministic sigGen, sigVer) bit for bit — but it
is **not** a FIPS-140-validated cryptographic module and offers no
side-channel-resistance guarantee. It authorises the discharge command that
*precedes* trigger arming; it does not sit on the sub-50 ns combinatorial path.

## ML-DSA-65

The polynomial ring is `R_q = Z_q[X]/(X^256 + 1)` with `q = 8380417`. Parameters
(FIPS 204 Table 1): `(k, l) = (6, 5)`, `eta = 4`, `gamma1 = 2^19`,
`gamma2 = (q-1)/32`, `tau = 49`, `omega = 55`, `d = 13`, `lambda = 192`. Public
key 1952 B, secret key 4032 B, signature 3309 B.

```python
from scpn_quantum_control.crypto import ml_dsa

pair = ml_dsa.key_gen(seed)                      # 32-byte seed -> key pair
sig = ml_dsa.sign(pair.secret_key, message, context=b"ctx")
assert ml_dsa.verify(pair.public_key, message, sig, context=b"ctx")
```

Conformance is asserted in `tests/test_ml_dsa_pqc.py` against the NIST ACVP
vectors in `tests/data/ml_dsa_65_kat.json`.

## Trigger signer

`PqcTriggerSigner` binds the payload to a timestamp inside the signed message,
so neither can be altered without invalidating the signature; verification
optionally enforces a freshness window.

```python
from scpn_quantum_control.crypto.pqc_trigger import PqcTriggerSigner

signer = PqcTriggerSigner()
pk, sk = signer.keygen()
sig = signer.sign_capacitor_bank_trigger("pulse-001", 24_500.0, timestamp_ns, sk)
# verify within a 10 ms freshness window:
ok = signer.verify(payload, sig, pk, max_age_ns=10_000_000)
```

## Acceleration

The negacyclic NTT (the dominant lattice operation) dispatches to a Rust kernel
that is **bit-true** (exact integer) with the Python reference; the rejection
loop, sampling, and encoding stay in Python.

Measured (release build, median of 21, `scripts/bench_ml_dsa.py`,
`results/ml_dsa_benchmark.json`, `functional_non_isolated`):

| operation | time |
|---|---|
| NTT (Python) | 209.7 µs |
| NTT (Rust) | 12.9 µs (16.2×) |
| key generation | 17.7 ms |
| signing | 27.5 ms |
| verification | 10.8 ms |

The millisecond-scale signing/verification latency is appropriate for the
authorisation gate, which precedes — and is not on — the fast trigger path.

## Consumers

SCPN-MIF-CORE imports `PqcTriggerSigner` to sign capacitor-bank discharge
commands prior to arming the combinatorial trigger.
