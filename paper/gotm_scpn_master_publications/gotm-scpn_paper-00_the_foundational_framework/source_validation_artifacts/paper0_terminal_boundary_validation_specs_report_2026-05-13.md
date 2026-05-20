# Paper 0 Terminal Boundary Specs

- Source span: P0R07073 - P0R07080
- Source records consumed: 8
- Coverage match: True
- Terminal count: 7
- Table IDs: TBL020
- Hardware status: boundary_protocol_no_device_execution
- Claim boundary: source-bounded EBS terminal protocol; no unbound empirical claim

## Specs

### terminal_boundary.section_boundary
- Boundary: section
- Protocol: paper0.terminal_boundary.section_boundary
- Statement: The section defines Terminal Taxonomy and Enhanced Boundary Set integration.
- Null controls: 3

### terminal_boundary.terminal_taxonomy
- Boundary: T1-T7
- Protocol: paper0.terminal_boundary.terminal_taxonomy
- Statement: A finite T1-T7 terminal taxonomy is required for all world-facing exchanges.
- Null controls: 3

### terminal_boundary.ebs_binding
- Boundary: EBS
- Protocol: paper0.terminal_boundary.ebs_binding
- Statement: Each run binds local geometry, environmental fields, CGP, and operator state into a versioned EBS object.
- Null controls: 3

### terminal_boundary.claim_traceability
- Boundary: traceability
- Protocol: paper0.terminal_boundary.claim_traceability
- Statement: Claims about cosmic dependence, environmental sensitivity, or consciousness modulation must trace to EBS ID and hash.
- Null controls: 3
