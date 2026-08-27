# ENAQT bounded transport evidence

- Schema: `enaqt_transport_evidence.v1`
- Functional passed: `true`
- Bounded claim ready: `true`
- Intermediate cases: `1` of `3`
- Content digest: `98e29c17270963160af4b09fd969836e2ba82612224f7549cbe8a59f707a1969`
- Execution: deterministic local simulation; no provider, QPU, hardware, or setpoint action.

## Frozen scenarios

| Scenario | gamma* | Coherent efficiency | Optimal efficiency | High-noise efficiency | Enhancement | Interior optimum |
|---|---:|---:|---:|---:|---:|---|
| disordered_chain_intermediate | 3 | 0.0522738642 | 0.17656498 | 0.0114666137 | 3.37769137 | true |
| uniform_chain_coherent_control | 0 | 0.819542095 | 0.819542095 | 0.0714404983 | 1 | false |
| disconnected_target_control | 0 | 0 | 0 | 0 | 0 | false |

## Interpretation

The disordered chain exhibits a finite-grid intermediate optimum. The uniform chain is a coherent-endpoint negative control, and the disconnected target remains zero-transport. Therefore the evidence supports only a scenario-specific ENAQT result, not a universal optimum.

## Claim boundary

bounded deterministic single-excitation Lindblad transport evidence for the frozen finite networks and scan grids only; no universal optimum, biological tuning, Kuramoto synchronisation, BKT, consciousness, quantum advantage, hardware fidelity, or physical noise-setpoint claim
