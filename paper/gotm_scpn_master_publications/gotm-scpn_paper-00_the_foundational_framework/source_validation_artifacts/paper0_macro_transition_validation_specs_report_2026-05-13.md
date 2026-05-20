# Paper 0 Macro-Transition Validation Specs

- Source records: `10`
- Consumed source records: `10`
- Coverage status: `match`
- Spec count: `2`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### nths.spin_glass_hamiltonian

- Protocol: `paper0.nths.spin_glass.phase_contrast`
- Source equations: `EQ0113`
- Source ledgers: `P0R00007, P0R00382, P0R05266, P0R05272, P0R05556, P0R05557, P0R05558`
- Null controls: `4`
- Executable targets: `4`
- Status: `validation_spec_pending_executable_fixture`

### macro_transition.effective_coupling_rg

- Protocol: `paper0.macro_transition.effective_coupling_rg_flow`
- Source equations: `EQ0114`
- Source ledgers: `P0R00538, P0R05636, P0R05639`
- Null controls: `4`
- Executable targets: `4`
- Status: `validation_spec_pending_executable_fixture`

## Policy

Provider submission remains out of scope. These records are source-anchored validation specifications only; executable simulator fixtures and null controls must pass before any paper claim or hardware plan is promoted.
