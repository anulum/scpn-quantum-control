# Paper 0 System-Robustness Fixture

- Spec keys: `applied.system_robustness.cascading_failure_percolation, applied.system_robustness.critical_slowing_recovery, applied.system_robustness.decoherence_attack_ms_qec_boundary`
- Hardware status: `simulator_only_no_provider_submission`
- Largest component loss: `0.25`
- Recovery-time ratio: `9.99999999999999`
- Failure probability: `0.08056874390487534`
- Runtime: `2.623050007969141` ms

## Null Controls

- `asymmetric_coupling_rejection_label`: `1.0`
- `complete_graph_largest_component_fraction`: `1.0`
- `critical_point_rejection_label`: `1.0`
- `empty_graph_fragmentation_label`: `1.0`
- `far_from_transition_ratio`: `0.040000000000000036`
- `non_positive_sigma_rejection_label`: `1.0`
- `out_of_range_correction_rejection_label`: `1.0`
- `unit_interval_success_label`: `1.0`
- `zero_redundancy_rejection_label`: `1.0`

## Policy

No provider submission is represented. This is a simulator-only robustness fixture; passing it is not operational security evidence and does not establish real-world attack resistance, safety, or resilience.
