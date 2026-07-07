# DLA-Protected Scar Memory

The scar-memory prototype uses the DLA parity theorem and the
fixed-parity repetition-code memory sector from
`qec.dla_protected_subspace`. It prepares a logical cat state across two
synchronised repetition-code words inside one DLA parity sector and evolves
it under a diagonal finite-dimensional Hamiltonian whose scar energies are
commensurate.

The result is a falsifiable memory primitive:

- the state leaves and returns to the initial scar packet over one revival
  period;
- the probability distribution stays inside the protected repetition-code
  sector at every sampled time;
- opposite-parity leakage remains directly measurable;
- Rust PyO3 trajectory metrics score protected, code, parity, and total
  weights over the full trajectory when `scpn_quantum_engine` is available.

## Public API

```python
from scpn_quantum_control.qec import (
    DLAProtectedScarSpec,
    build_dla_protected_scar_prototype,
    simulate_dla_protected_scar_memory,
)

spec = DLAProtectedScarSpec()
prototype = build_dla_protected_scar_prototype(spec)
result = simulate_dla_protected_scar_memory(prototype)

print(result.final_revival_fidelity)
print(result.min_protected_weight)
print(result.max_parity_leakage)
```

The default prototype uses four logical oscillators with distance-three
repetition blocks, giving twelve physical qubits. The default protected
scar words are the all-zero and all-one logical synchronisation memories,
which both live in the even DLA parity sector for an even number of logical
oscillators.

## Certificate

`build_dla_protected_scar_prototype()` carries the same analytic
certificate as `certify_dla_protected_subspace()`:

- odd repetition distance;
- fixed global DLA parity;
- protected logical dimension matching the target sector;
- synchronised scar words contained in the protected sector;
- heterogeneous XY DLA dimension
  $2^{2N-1} - 2 = \mathfrak{su}(2^{N-1}) \oplus
  \mathfrak{su}(2^{N-1})$.

## Revival Model

For scar basis states $|s_k\rangle$, the prototype prepares

$$
|\psi_0\rangle = \frac{1}{\sqrt m}\sum_{k=0}^{m-1}|s_k\rangle
$$

and assigns commensurate energies

$$
E_k = k\,\frac{2\pi}{T}.
$$

For the default two-state memory, the survival probability is

$$
|\langle\psi_0|\psi(t)\rangle|^2
= \cos^2\!\left(\frac{\pi t}{T}\right).
$$

The final sample at $t=T$ must revive to the configured fidelity
threshold, while the protected and parity weights are evaluated over the
whole trajectory.

## Count Snapshots

`evaluate_dla_protected_scar_counts()` accepts measured count dictionaries
at sampled times. Counts cannot recover the phase-sensitive survival
amplitude, so the count path treats scar support as the observable memory
survival proxy and still enforces protected-sector and parity-leakage
criteria.

## Failure Criteria

The typed result fails when any configured criterion is violated:

- `revival_fidelity_below_threshold`;
- `protected_weight_below_threshold`;
- `parity_leakage_above_threshold`;
- `scar_support_below_threshold`;
- `protection_certificate_failed`.

These criteria make the prototype usable as a pre-hardware witness: a
statevector simulation can validate phase revival, while measured counts can
validate protected memory support and DLA parity leakage.
