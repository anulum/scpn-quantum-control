# DLA and Topology-Constrained Control Evidence

- Schema: `topology_control_evidence_v2`
- Generated: `2026-07-29`
- Content digest: `b97043be4bb4b5d3060d30c61f3afc25b290bd18520c60410e6eed4c887e4c5b`
- Claim boundary: finite synthetic parity-sector and fixed-active-set topology derivatives only; no full-DLA, controllability, persistent-homology derivative, hardware-protection, error-correction, provider, QPU, or deployment claim

## Synthetic parity-protected task

| Metric | Value |
|---|---:|
| Qubits | 4 |
| Sector | even |
| Initial objective | 5.06651816681 |
| Final objective | 8.65715781265e-25 |
| Initial leakage mass | 1.60797013733 |
| Final leakage mass | 0 |
| Accepted projected steps | 40 |

## Derivative checks

| Check | Maximum/absolute error |
|---|---:|
| Parity objective gradient | 3.59233087721e-10 |
| Parity projector JVP | 9.98454884353e-11 |
| Topology-ledger JVP | 3.93055310521e-11 |
| Topology JVP/VJP adjoint identity | 8.881784197e-16 |
| Existing projected optimiser final violation | 0 |

Unsupported topology derivative blockers: `sign_policy`, `uniform_bounds`, `total_weight`, `algebraic_connectivity_threshold`.

## Slice support

| Slice | Status | Derivative class | Evidence | Boundary |
|---|---|---|---|---|
| differentiability boundary | supported | affine | linear/affine branches are separated from non-smooth and discrete branches | classification is exact-local rather than a universal smoothness claim |
| existing contract inventory | supported | not_applicable | the facade composes DLA parity and topology_control owners | inventory is not new mathematical evidence |
| penalties and projections | supported | piecewise_smooth | parity JVP/VJP is exact and topology JVP/VJP is fixed-active-set only | PH, connectivity, kinks, and active budget rescaling fail closed |
| constrained optimiser | supported | affine | synthetic parity projection occurs inside every strict-decrease proposal | no physical Hamiltonian, controller, or hardware is actuated |
| deterministic evidence | supported | not_applicable | central differences, adjoint identity, custody digests, and blockers are frozen | one finite configuration is not generalisation evidence |
| optional QGNN wiring | descoped | not_applicable | no current QGNN consumer maps Hilbert-space parity onto graph-message topology | parity and graph topology are not conflated without a typed consumer |
| constraint versus witness documentation | supported | not_applicable | public docs distinguish projected constraints, diagnostics, and non-claims | documentation does not promote hardware DLA protection |

These rows are finite synthetic regression evidence. They are not a full-DLA,
controllability, differentiable-PH, hardware-protection, error-correction,
provider, QPU, advantage, or deployment result.
