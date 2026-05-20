# Paper 0 System-Robustness Validation Specs

- Source records: `3`
- Consumed source records: `3`
- Coverage status: `match`
- Source span: `P0R06215, P0R06217`
- Spec count: `3`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### applied.system_robustness.cascading_failure_percolation

- Protocol: `paper0.applied.system_robustness.cascading_failure_percolation`
- Source ledgers: `P0R06215, P0R06216, P0R06217`
- Null controls: `3`
- Executable targets: `3`
- Claim boundary: `simulator-only robustness boundary; not operational security evidence`

### applied.system_robustness.critical_slowing_recovery

- Protocol: `paper0.applied.system_robustness.critical_slowing_recovery`
- Source ledgers: `P0R06215, P0R06216, P0R06217`
- Null controls: `3`
- Executable targets: `3`
- Claim boundary: `simulator-only robustness boundary; not operational security evidence`

### applied.system_robustness.decoherence_attack_ms_qec_boundary

- Protocol: `paper0.applied.system_robustness.decoherence_attack_ms_qec_boundary`
- Source ledgers: `P0R06215, P0R06216, P0R06217`
- Null controls: `3`
- Executable targets: `3`
- Claim boundary: `simulator-only robustness boundary; not operational security evidence`

## Policy

These records are source-anchored simulator robustness specifications only. Passing any fixture is not operational security evidence and does not establish real-world attack resistance, safety, or resilience.
