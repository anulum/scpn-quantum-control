# Paper 0 Mass Eigenstates Mixing Angle Specs

- Source span: P0R01669 - P0R01683
- Source records: 15
- Consumed source records: 15
- Coverage match: True
- Spec count: 3
- Claim boundary: source-bounded mass-eigenstates mixing-angle bridge; not validation evidence
- Hardware status: source_methodology_no_experiment
- Next source boundary: P0R01684

## Specs
### `mass_eigenstates_mixing_angle.mass_eigenstate_rotation`

The source defines physical h_SM and h_Psi mass eigenstates by rotating the bare scalar fields with a mixing angle theta.

- Context: `mass_eigenstate_rotation`
- Protocol: `paper0.mass_eigenstates_mixing_angle.mass_eigenstate_rotation`
- Source equations: P0R01669:mass_eigenstate_rotation, P0R01670:mass_eigenstate_rotation, P0R01671:mass_eigenstate_rotation, P0R01672:mass_eigenstate_rotation, P0R01673:mass_eigenstate_rotation, P0R01674:mass_eigenstate_rotation
- Null controls: rotation formalism must not be treated as measured Higgs mixing

### `mass_eigenstates_mixing_angle.lhc_invisible_decay_bound`

The source maps sin(theta) to Standard Model interaction suppression and cites an invisible-Higgs branching-ratio limit as a working bound.

- Context: `lhc_invisible_decay_bound`
- Protocol: `paper0.mass_eigenstates_mixing_angle.lhc_invisible_decay_bound`
- Source equations: P0R01675:lhc_invisible_decay_bound, P0R01676:lhc_invisible_decay_bound, P0R01677:lhc_invisible_decay_bound, P0R01678:lhc_invisible_decay_bound, P0R01679:lhc_invisible_decay_bound, P0R01680:lhc_invisible_decay_bound
- Null controls: empirical upper limit must not be reported as observed Psi-sector signal

### `mass_eigenstates_mixing_angle.perturbative_target_boundary`

The source frames perturbatively small lambda_mix and sin(theta) <= 0.31 as a working search target for falsification or discovery.

- Context: `perturbative_target_boundary`
- Protocol: `paper0.mass_eigenstates_mixing_angle.perturbative_target_boundary`
- Source equations: P0R01681:perturbative_target_boundary, P0R01682:perturbative_target_boundary, P0R01683:perturbative_target_boundary
- Null controls: working-bound language must not imply model confirmation
