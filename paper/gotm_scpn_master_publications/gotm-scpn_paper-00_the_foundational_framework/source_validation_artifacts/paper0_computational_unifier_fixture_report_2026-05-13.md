# Paper 0 Computational-Unifier Fixture

- Hardware status: `simulator_only_no_provider_submission`
- Runtime: `5.963422998320311` ms

## Cyclic Operator Boundary

- Spec key: `computational.cyclic_operator_boundary`
- Protocol: `paper0.computational.cyclic_operator.boundary_periodicity`
- Source equations: `EQ0115`
- Unitarity error: `1.1102230246251565e-16`
- Cycle-closure residual: `1.0778315928076987e-15`

## TSVF/ABL Boundary Probability

- Spec key: `computational.tsvf_abl_boundary`
- Protocol: `paper0.computational.tsvf_abl.boundary_probability`
- Source equations: `EQ0116`
- Probability normalisation error: `0.0`

## Information Thermodynamics

- Spec key: `computational.info_thermodynamics`
- Protocol: `paper0.computational.info_thermodynamics.entropy_budget`
- Source equations: `EQ0117, EQ0118`
- GSL margin: `0.08000000000000002`
- MI-negentropy error: `0.0`

## Null Controls

### Cyclic Operator Boundary

- `non_unitary_rejection_label`: `1.0`
- `wrong_period_residual`: `1.949855824363647`

### TSVF/ABL Boundary Probability

- `born_rule_reduction_l1`: `0.0`
- `projector_resolution_error`: `0.0`
- `zero_denominator_rejection_label`: `1.0`

### Information Thermodynamics

- `finite_gsl_margin_label`: `1.0`
- `independent_channel_negentropy_abs`: `0.0`
- `landauer_violation_label`: `1.0`

## Policy

No provider submission is represented. These are simulator-only boundary fixtures for source-anchored Paper 0 computational-unifier equations. They validate finite mathematical consistency checks, not empirical confirmation of the broader mechanism.
