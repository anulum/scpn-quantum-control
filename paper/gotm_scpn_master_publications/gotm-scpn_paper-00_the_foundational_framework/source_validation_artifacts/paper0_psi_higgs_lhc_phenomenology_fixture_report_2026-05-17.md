# Paper 0 Psi-Higgs LHC Phenomenology Fixture

- Source span: P0R01655 - P0R01668
- Hardware status: source_methodology_no_experiment
- Claim boundary: source-bounded Psi-Higgs LHC phenomenology bridge; not validation evidence
- Source records: 14
- Components: 3
- Next source boundary: P0R01669
- Protocol state: source_psi_higgs_lhc_phenomenology_only_no_experiment

## Components
- `phenomenology_bridge`: `psi_higgs_lhc_phenomenology_bridge_boundary`
- `scalar_mixing_mechanism`: `psi_higgs_sm_higgs_scalar_mixing_boundary`
- `scalar_potential_and_cross_term`: `higgs_portal_potential_cross_term_boundary`

## Null Controls
- `lhc_phenomenology_bridge_is_not_observed_lhc_signal`: 1.0
- `scalar_mixing_claim_is_not_measured_higgs_admixture`: 1.0
- `higgs_portal_potential_is_not_fitted_collider_model`: 1.0
