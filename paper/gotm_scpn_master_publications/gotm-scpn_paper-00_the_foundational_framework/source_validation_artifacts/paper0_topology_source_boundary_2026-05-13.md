# Paper 0 Topology Source Boundary

- Schema key: `paper0.topology.source_boundary.v1`
- Layer count: `16`
- Meta-layer: `L16`
- Provider ready: `False`
- Hardware status: `source_boundary_only_no_provider_submission`
- Source equations: `11`
- Source ledger records: `31`

## Policy

No numeric coupling matrix is exported by this boundary. Paper 0 currently supports source-anchored layer, coupling-channel, field-port, and adaptive parameter provenance only. Synthetic chain, ring, and complete graphs are validation controls, not Paper 0 topology claims.

## Coupling Channels

- `downward_generation`: `14` directed edges
- `recursive_closure`: `2` directed edges
- `upward_inference`: `14` directed edges

## Synthetic Controls

- `control.chain`
- `control.ring`
- `control.complete`

## S19 Boundary

S19 scans may consume this file as a provenance boundary only. They must still supply an explicitly labelled experimental or synthetic numeric topology before any simulation, provider transpilation, or hardware run.
