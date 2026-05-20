# Paper 0 UPDE Aggregate Validation Index

- Spec count: `5`
- Fixture result count: `5`
- Coverage status: `match`
- Hardware status: `simulator_only_no_provider_submission`
- Total local fixture runtime: `21.53322300000582` ms

## Fixtures

### upde.adaptive_coupling

- Protocol: `paper0.upde.adaptive_coupling.quasicritical_controller`
- Source equations: `EQ0045`
- Source ledgers: `P0R02910`
- Measured null controls: `4`
- Runtime: `1.2968219962203875` ms
- Hardware status: `simulator_only_no_provider_submission`

### upde.base_phase

- Protocol: `paper0.upde.base_phase.xy_gradient_and_locking`
- Source equations: `EQ0003, EQ0032, EQ0037, EQ0039, EQ0129`
- Source ledgers: `P0R00520, P0R02507, P0R02530, P0R02622, P0R06120`
- Measured null controls: `4`
- Runtime: `1.7314529977738857` ms
- Hardware status: `simulator_only_no_provider_submission`

### upde.field_coupling

- Protocol: `paper0.upde.field.global_phase_coupling`
- Source equations: `EQ0034, EQ0041, EQ0043`
- Source ledgers: `P0R02512, P0R02634, P0R02644`
- Measured null controls: `3`
- Runtime: `6.5162550017703325` ms
- Hardware status: `simulator_only_no_provider_submission`

### upde.interlayer_coupling

- Protocol: `paper0.upde.interlayer.directional_coupling`
- Source equations: `EQ0033, EQ0040`
- Source ledgers: `P0R02510, P0R02630`
- Measured null controls: `3`
- Runtime: `10.21192199550569` ms
- Hardware status: `simulator_only_no_provider_submission`

### upde.natural_gradient

- Protocol: `paper0.upde.natural_gradient.fim_free_energy`
- Source equations: `EQ0042`
- Source ledgers: `P0R02642`
- Measured null controls: `3`
- Runtime: `1.7767710087355226` ms
- Hardware status: `simulator_only_no_provider_submission`

## Policy

No provider submission is represented by this aggregate index. The records are simulator-only validation fixtures for source-anchored Paper 0 equations.
