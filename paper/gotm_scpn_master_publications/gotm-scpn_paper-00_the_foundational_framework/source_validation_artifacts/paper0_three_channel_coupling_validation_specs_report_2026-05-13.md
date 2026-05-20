# Paper 0 Three-Channel Coupling Specs

- Source span: P0R07081 - P0R07129
- Source records consumed: 49
- Coverage match: True
- Channel count: 3
- Sweet-spot window: [1e-06, 1e-05]
- Hardware status: parameter_scan_protocol_no_execution
- Claim boundary: source-bounded parameter scan; not empirical support

## Specs

### three_channel_coupling.section_boundary
- Scan: chapter25
- Protocol: paper0.three_channel_coupling.section_boundary
- Statement: Chapter 25 defines a unified coupling parameter scan across three constrained channels.
- Null controls: 3

### three_channel_coupling.geometry_factors
- Scan: geometry
- Protocol: paper0.three_channel_coupling.geometry_factors
- Statement: All Psi-sector couplings derive from one bare lambda0 through canonical warp geometry factors.
- Null controls: 3

### three_channel_coupling.fixed_ratios
- Scan: ratios
- Protocol: paper0.three_channel_coupling.fixed_ratios
- Statement: The source states fixed ratios between gravitational, electromagnetic, quantum, and scalar channels.
- Null controls: 3

### three_channel_coupling.experimental_constraints
- Scan: constraints
- Protocol: paper0.three_channel_coupling.experimental_constraints
- Statement: Gravitational, EM clock, and quantum coherence limits jointly constrain lambda0.
- Null controls: 3

### three_channel_coupling.cross_channel_propagation
- Scan: sweet_spot
- Protocol: paper0.three_channel_coupling.cross_channel_propagation
- Statement: The source defines a narrow lambda0 window and cross-channel propagation of bounds.
- Null controls: 3

### three_channel_coupling.falsification_fingerprint
- Scan: fingerprint
- Protocol: paper0.three_channel_coupling.falsification_fingerprint
- Statement: The three-channel pattern is falsified by isolated single-channel signals and supported only by predicted-ratio coincidence.
- Null controls: 3
