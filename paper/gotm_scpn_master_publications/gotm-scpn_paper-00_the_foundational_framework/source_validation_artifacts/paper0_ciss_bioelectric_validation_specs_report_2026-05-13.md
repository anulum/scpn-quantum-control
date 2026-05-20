# Paper 0 CISS-Bioelectric Feedback Specs

- Source span: P0R06560 - P0R06581
- Source records consumed: 22
- Coverage match: True
- Hardware status: simulator_only_no_provider_submission
- Claim boundary: source-bounded CISS-bioelectric feedback simulator contract; not empirical evidence

## Specs

### ciss_bioelectric.layer3_framing

Layer 3 is promoted as coupled CISS-bioelectric dynamics joining CISS spin filtering, radical-pair dynamics, a bioelectric cascade, and membrane-feedback coupling.

Formulae:

Mechanisms:
- Layer 3 is framed as a CISS-bioelectric feedback loop
- dual mechanism integration joins CISS spin filtering and radical-pair dynamics
- bioelectric cascade and coupled feedback are retained as separate mechanisms

Null controls:
- missing-CISS-channel control must be rejected
- missing-bioelectric-channel control must be rejected
- unbounded-empirical-claim control must be rejected

### ciss_bioelectric.ciss_spin_filter

The CISS spin-filter term is preserved as a source Hamiltonian with epsilon, splitting, spin-orbit, and spin-coupling contributions.

Formulae:
- H_total = epsilon_0 + (Delta/2) sigma_z + (lambda / L^2)(sigma dot L) + g S dot sigma
- lambda is spin-orbit coupling and generates effective B_eff in the 10-100 T source range

Mechanisms:
- lambda is retained as the spin-orbit coupling parameter
- effective B_eff is source-bounded to the 10-100 T range
- spin-filter Hamiltonian contributions remain additive

Null controls:
- non-positive-length control must be rejected
- non-finite-Hamiltonian-input control must be rejected
- out-of-range-effective-field control must be labelled

### ciss_bioelectric.radical_pair_modulation

Radical-pair dynamics are promoted with Zeeman, hyperfine, exchange, and CISS effective-field modulation of singlet/triplet ratio.

Formulae:
- H_RP = sum_i [omega_i S_iz + sum_k A_ik S_i dot I_k] + J(1/2 + 2 S_1 dot S_2)
- singlet/triplet ratio is modulated by B_eff from CISS

Mechanisms:
- Zeeman terms are retained in the radical-pair Hamiltonian
- hyperfine terms are retained in the radical-pair Hamiltonian
- exchange coupling contributes through J(1/2 + 2 S_1 dot S_2)
- CISS effective field modulates singlet/triplet ratio

Null controls:
- shape-mismatch control must be rejected
- non-finite-hyperfine control must be rejected
- zero-field-modulation control must be bounded

### ciss_bioelectric.bioelectric_cascade_feedback

Bioelectric target-gradient drive, calcium/CaMKII/chromatin cascade, membrane derivative, field-dependent CISS efficiency, and local-field radical-pair feedback are preserved.

Formulae:
- E = -grad V_target -> activates Ca_v channels -> intracellular Ca2+ spike
- dV_mem/dt = -I_ion(V_mem, B_eff(lambda(E))) + I_pump
- lambda(E) is a function of local electric field E
- H_RP = H_RP_base + B_local(V_mem) dot (g_1 S_1 + g_2 S_2)

Mechanisms:
- electric field is the negative target-potential gradient
- calcium, CaMKII, HDAC/HAT phosphorylation, and chromatin remodelling form the cascade
- membrane dynamics depend on ionic current under B_eff(lambda(E)) and pump current
- radical-pair Hamiltonian receives a local membrane-potential field coupling

Null controls:
- non-finite-gradient control must be rejected
- non-finite-current control must be rejected
- unsupported-morphogenesis-evidence control must be rejected

### ciss_bioelectric.observable_predictions

Observable predictions are retained as proposed validation targets: optogenetic bioelectric perturbation, chiral CISS blockade, and nonlinear radical-pair yield.

Formulae:

Mechanisms:
- bioelectric field perturbation by optogenetics predicts epigenetic changes
- CISS blockade by chiral molecular disruption predicts loss of field-guided morphogenesis
- radical-pair yield versus applied E-field is expected to show non-linear modulation

Null controls:
- missing-optogenetic-perturbation control must be rejected
- missing-CISS-blockade control must be rejected
- linear-only-yield control must be rejected
