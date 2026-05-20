# Paper 0 Cosmological EOS Specs

- Source span: P0R06916 - P0R06948
- Source records consumed: 33
- Coverage match: True
- Hardware status: cosmological_constraint_fixture_no_execution
- Claim boundary: source-bounded cosmological equation-of-state fixture; not empirical evidence

## Specs

### cosmological_eos.chapter_boundary
- Protocol: paper0.cosmological_eos.chapter_boundary
- Statement: Chapter 22 frames cosmological constraints through a Psi-field equation of state.
- Source equations: P0R06916:chapter_boundary
- Null controls: 3

### cosmological_eos.scalar_field_equations
- Protocol: paper0.cosmological_eos.scalar_field_equations
- Statement: Scalar-field density, pressure, and equation-of-state formulae are stated.
- Source equations: P0R06920:rho_psi, P0R06921:p_psi, P0R06923:w_psi
- Null controls: 3

### cosmological_eos.limiting_cases
- Protocol: paper0.cosmological_eos.limiting_cases
- Statement: Slow-roll, kinetic-dominated, and oscillatory limiting cases are distinguished.
- Source equations: P0R06925:slow_roll_limit, P0R06926:kinetic_limit, P0R06927:oscillatory_limit
- Null controls: 3

### cosmological_eos.observational_constraint
- Protocol: paper0.cosmological_eos.observational_constraint
- Statement: Planck 2018 plus supernova context constrains w0 near -1.
- Source equations: P0R06930:w0_planck_supernova
- Null controls: 3

### cosmological_eos.hybrid_split_and_homogeneity
- Protocol: paper0.cosmological_eos.hybrid_split
- Statement: The Psi field is split into homogeneous background and local perturbation terms.
- Source equations: P0R06934:psi_background_perturbation_split, P0R06938:stress_energy_split, P0R06942:smooth_dark_energy_constraint, P0R06943:gentle_variation_bound
- Null controls: 3

### cosmological_eos.quintessence_detection_target
- Protocol: paper0.cosmological_eos.quintessence_detection
- Statement: A mild w(z) deviation is recorded as a future-survey detection target.
- Source equations: P0R06946:low_redshift_quintessence_possibility, P0R06947:survey_detection_target
- Null controls: 3
