# Paper 0 HPC/UPDE Bridge Validation Specs

- Source records: `23`
- Consumed source records: `23`
- Coverage status: `match`
- Source span: `P0R06156, P0R06178`
- Spec count: `3`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### computational.hpc_bidirectional_flow

- Protocol: `paper0.computational.hpc.bidirectional_flow`
- Source equations: ``
- Source ledgers: `P0R06156, P0R06157, P0R06158`
- Null controls: `3`
- Executable targets: `3`
- Status: `validation_spec_pending_executable_fixture`

### computational.upde_phase_prediction_error

- Protocol: `paper0.computational.upde.phase_prediction_error`
- Source equations: ``
- Source ledgers: `P0R06159, P0R06160, P0R06161, P0R06162, P0R06163`
- Null controls: `3`
- Executable targets: `3`
- Status: `validation_spec_pending_executable_fixture`

### computational.upde_free_energy_gradient_bridge

- Protocol: `paper0.computational.upde.free_energy_gradient_bridge`
- Source equations: ``
- Source ledgers: `P0R06164, P0R06165, P0R06166, P0R06167, P0R06168, P0R06169, P0R06170, P0R06171, P0R06172, P0R06173, P0R06174, P0R06175, P0R06176, P0R06177, P0R06178`
- Null controls: `3`
- Executable targets: `3`
- Status: `validation_spec_pending_executable_fixture`

## Policy

No standalone equation IDs are invented for formula text that was not assigned canonical equation anchors during Paper 0 extraction. These records are source-anchored validation specifications only; the HPC, UPDE, and free-energy bridge interpretations require executable fixtures and falsification controls before claim promotion.
