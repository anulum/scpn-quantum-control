# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — DLA-Protected Logical Synchronisation

# DLA-Protected Logical Synchronisation

The DLA-protected logical synchronisation module builds a finite
repetition-code memory sector inside the global parity decomposition of the
heterogeneous Kuramoto-XY Hamiltonian.

For `n_logical` logical oscillators and odd repetition distance `d`, each
logical oscillator occupies a contiguous block of `d` physical qubits:

```text
logical 0: q[0:d]
logical 1: q[d:2d]
...
```

A block is in the logical memory manifold when it is either all zero or all
one. Because `d` is odd, the physical block parity equals the logical bit.
Restricting the logical words to a fixed global parity therefore selects one
of the two DLA sectors from the parity theorem:

```text
DLA = su(2^(N-1)) ⊕ su(2^(N-1))
```

where `N = n_logical * d`.

## Certificate

```python
from scpn_quantum_control.qec import (
    DLAProtectedSubspaceSpec,
    certify_dla_protected_subspace,
)

spec = DLAProtectedSubspaceSpec(
    n_logical=3,
    code_distance=3,
    target_parity=1,
)

certificate = certify_dla_protected_subspace(spec)
print(certificate.is_provable)
print(certificate.physical_dla_dimension)
print(certificate.protected_logical_dim)
```

The certificate records:

- the physical DLA dimension `2^(2N-1) - 2`;
- the even and odd parity-sector Hilbert dimensions;
- the fixed-parity logical memory basis;
- the synchronised logical words inside that basis;
- proof obligations that must all hold for the certificate to pass.

## Memory Prototype

```python
from scpn_quantum_control.qec import build_dla_protected_memory_prototype

prototype = build_dla_protected_memory_prototype(
    spec,
    logical_word=(1, 1, 1),
)

qc = prototype.circuit
print(prototype.basis_index, qc.depth())
```

The prototype prepares one repetition-code logical word with `X` gates on
every physical qubit in logical blocks set to one. It rejects words whose
logical parity does not match the target DLA sector.

## Witness

```python
from scpn_quantum_control.qec import evaluate_dla_protected_memory

probabilities = ...
result = evaluate_dla_protected_memory(probabilities, spec=spec)

print(result.protected_weight)
print(result.sync_weight)
print(result.parity_leakage)
print(result.failure_reasons)
```

The witness accepts either a dense probability vector or measurement counts.
Count bitstrings use the usual Qiskit display order; the module reverses the
string before mapping into the contiguous block layout. Counts must be
non-negative integer shot counts; fractional or boolean values are rejected
before normalisation so the witness cannot silently rescale malformed hardware
or simulator payloads.

Failure criteria are configurable on the spec:

```python
spec = DLAProtectedSubspaceSpec(
    n_logical=4,
    code_distance=3,
    target_parity=0,
    min_protected_weight=0.9,
    min_sync_weight=0.75,
    max_parity_leakage=0.05,
    max_code_leakage=0.1,
)
```

The result fails when any of the following holds:

- protected fixed-parity memory weight is below `min_protected_weight`;
- synchronised memory weight is below `min_sync_weight`;
- opposite-parity leakage is above `max_parity_leakage`;
- repetition-code block leakage is above `max_code_leakage`.

## Rust Path

The Rust extension provides:

- `scpn_quantum_engine.dla_protected_memory_mask`;
- `scpn_quantum_engine.dla_protected_memory_metrics`.

The Python module uses those functions when available and falls back to NumPy
for source checkouts without the extension.
