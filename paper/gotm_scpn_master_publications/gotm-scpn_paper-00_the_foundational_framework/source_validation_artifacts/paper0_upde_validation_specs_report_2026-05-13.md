# Paper 0 UPDE Validation Specs

- Anchor records: `12`
- Consumed anchor records: `12`
- Coverage status: `match`
- Spec count: `5`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### upde.base_phase

- Protocol: `paper0.upde.base_phase.xy_gradient_and_locking`
- Source ledgers: `P0R00520, P0R02507, P0R02530, P0R02622, P0R06120`
- Source equations: `EQ0003, EQ0032, EQ0037, EQ0039, EQ0129`
- Implementation status: `partially_implemented_quantum_xy_mapping`
- Null controls: `4`
- Executable targets: `4`

### upde.interlayer_coupling

- Protocol: `paper0.upde.interlayer.directional_coupling`
- Source ledgers: `P0R02510, P0R02630`
- Source equations: `EQ0033, EQ0040`
- Implementation status: `validation_spec_pending_direct_implementation_audit`
- Null controls: `3`
- Executable targets: `3`

### upde.field_coupling

- Protocol: `paper0.upde.field.global_phase_coupling`
- Source ledgers: `P0R02512, P0R02634, P0R02644`
- Source equations: `EQ0034, EQ0041, EQ0043`
- Implementation status: `validation_spec_pending_direct_implementation_audit`
- Null controls: `3`
- Executable targets: `3`

### upde.natural_gradient

- Protocol: `paper0.upde.natural_gradient.fim_free_energy`
- Source ledgers: `P0R02642`
- Source equations: `EQ0042`
- Implementation status: `validation_spec_pending_direct_implementation_audit`
- Null controls: `3`
- Executable targets: `3`

### upde.adaptive_coupling

- Protocol: `paper0.upde.adaptive_coupling.quasicritical_controller`
- Source ledgers: `P0R02910`
- Source equations: `EQ0045`
- Implementation status: `validation_spec_pending_direct_implementation_audit`
- Null controls: `3`
- Executable targets: `3`

## Policy

UPDE anchors are promoted only into validation specifications. Provider submission remains out of scope until simulator fixtures, controls, and implementation audits pass.
