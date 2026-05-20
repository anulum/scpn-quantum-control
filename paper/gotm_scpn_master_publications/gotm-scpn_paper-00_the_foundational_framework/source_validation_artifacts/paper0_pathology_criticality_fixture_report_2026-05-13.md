# Paper 0 Pathology/Criticality Fixture

- Spec keys: `applied.pathology.coherence_breakdown_index, applied.pathology.criticality_deviation_classifier, applied.pathology.therapeutic_restoration_targets`
- Hardware status: `simulator_only_no_provider_submission`
- Pathology index: `2.4001045924503384`
- Baseline index: `0.35000000000000003`
- Index delta vs baseline: `2.0501045924503383`
- Sigma label: `supercritical`
- Restoration index delta: `-0.4977830356563837`
- Runtime: `3.9267970132641494` ms

## Null Controls

- `healthy_baseline_index`: `0.35000000000000003`
- `negative_probability_rejection_label`: `1.0`
- `negative_tolerance_rejection_label`: `1.0`
- `non_finite_observable_rejection_label`: `1.0`
- `non_positive_sigma_rejection_label`: `1.0`
- `out_of_range_order_parameter_rejection_label`: `1.0`
- `sigma_neutral_label_is_quasicritical`: `1.0`
- `wrong_direction_index_delta`: `0.4853605156578258`
- `zero_update_index_delta_abs`: `0.0`

## Policy

No clinical or provider submission is represented. This is a simulator-only finite systems-metric fixture and does not represent diagnosis, treatment guidance, medical advice, or empirical clinical evidence.
