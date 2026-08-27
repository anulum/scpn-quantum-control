# Bounded entanglement-sync initial-state coherence evidence

- Schema: `entanglement_initial_state_evidence.v2`
- Functional passed: `true`
- Deterministic replay: `true`
- Content digest: `5aee67e29fe577332b38a36a229bd9a705d9aca49b80ac3e534467bb50665628`
- Execution: exact local statevector/density simulation; no provider, QPU, or hardware.

## Frozen state-family comparisons

| State | Initial linear entropy | Mean exchange coherence | Dephased control | Difference | Final difference |
|---|---:|---:|---:|---:|---:|
| product | 0 | 0.419481647 | 0.15443652 | 0.265045127 | 0.0934128992 |
| bell_pairs | 1 | 0.132115673 | 0.0910960776 | 0.0410195955 | 0.13222845 |
| ghz | 1 | 0 | 0 | 0 | 0 |
| w_state | 0.75 | 0.366405312 | 1.03712727e-16 | 0.366405312 | 0.244901947 |

## Interpretation

Bell-pair and W initial states differ from their population-matched dephased controls, while GHZ is the zero-difference negative control. The separable product state also differs from its dephased control. The measured effect is therefore an initial-coherence observation and is not attributable uniquely to entanglement.

The closed finite model has no drive, dissipation, limit cycle, or coupling scan. It cannot establish spontaneous synchronisation or a shifted critical coupling.

## Claim boundary

bounded deterministic closed-system statevector study for one frozen four-qubit Kuramoto-XY Hamiltonian and finite time grid; exchange coherence is a custom diagnostic, not a spontaneous-synchronisation certificate; dephased-control differences do not establish an entanglement-specific cause, lower critical coupling, universal enhancement, quantum advantage, hardware fidelity, or control authority
