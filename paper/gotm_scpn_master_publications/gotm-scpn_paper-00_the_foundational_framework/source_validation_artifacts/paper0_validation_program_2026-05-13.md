# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 validation programme

# Paper 0 Validation Programme

Date: 2026-05-13

This is the new source-first validation lane. It starts from the extracted
Paper 0 manuscript rather than from the current Paper 27 matrix helper.

## Scientific Rule

No mechanism is promoted to code, figures, hardware gates, or paper claims
unless it has:

- a Paper 0 source anchor;
- a canonical mathematical statement;
- units, domains, boundary conditions, and parameters;
- an executable simulator implementation;
- null controls and sensitivity tests;
- source-to-result provenance in generated artefacts.

## Exhaustive Capture Rule

The Paper 0 review surface is the exhaustive block register, not a cherry-picked
keyword list:

- `paper0_exhaustive_register_2026-05-13.jsonl` has one record per top-level
  Pandoc AST block;
- current coverage is 7,129 records for 7,129 top-level AST blocks;
- every record is `unreviewed` until canonical review promotes it to context,
  claim, mechanism, theorem, equation, validation target, or rejection;
- candidate labels are triage aids only and do not define the final claim set.

## Canonical Review Ledger

The current canonical-review ledger is
`paper0_canonical_review_ledger_2026-05-13.jsonl`.

Status:

- source records: 7,129;
- ledger records: 7,129;
- coverage status: match;
- context or structure accepted: 3,409;
- scientific domain review required: 3,720;
- UPDE source anchors linked to the Paper 0 equation register: 12.

Category queue:

- claim: 1,325;
- mechanism: 1,701;
- equation block: 123;
- validation target: 571;
- context: 2,813;
- figure: 163;
- table: 19;
- structural: 414.

Review queues:

- `paper0_domain_review_queue_2026-05-13.jsonl` contains the 3,720 records
  requiring scientific review.
- `paper0_upde_anchor_review_queue_2026-05-13.jsonl` contains the 12 records
  already linked to the current UPDE equation register.
- `paper0_upde_validation_specs_2026-05-13.json` promotes those 12 UPDE
  anchors into five source-covered validation specifications.

UPDE validation-spec status:

- source anchors: 12;
- consumed source anchors: 12;
- coverage status: match;
- promoted specs: `upde.base_phase`, `upde.interlayer_coupling`,
  `upde.field_coupling`, `upde.natural_gradient`,
  `upde.adaptive_coupling`;
- all promoted specs include variables, assumptions, null controls,
  executable validation targets, and implementation links;
- hardware status: simulator-only, no provider submission.

Base-phase executable fixture status:

- implementation: `src/scpn_quantum_control/paper0/upde_validation.py`;
- tests: `tests/test_paper0_upde_base_phase_validation.py`;
- result artefact:
  `paper0_upde_base_phase_fixture_result_2026-05-13.json`;
- report:
  `paper0_upde_base_phase_fixture_report_2026-05-13.md`;
- local 4-oscillator gradient error, L-infinity:
  `2.0708079695452852e-10`;
- local single-fixture runtime: `1.7314529977738857 ms`;
- hardware status: simulator-only, no provider submission.

Inter-layer executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/upde_interlayer_validation.py`;
- tests: `tests/test_paper0_upde_interlayer_validation.py`;
- result artefact:
  `paper0_upde_interlayer_fixture_result_2026-05-13.json`;
- report:
  `paper0_upde_interlayer_fixture_report_2026-05-13.md`;
- lower-to-downward response, L2: `0.10542163465303317`;
- upper-to-upward response, L2: `0.05679842904477164`;
- cross-channel leakage controls: `0.0`;
- disconnected-layer null, L-infinity: `0.0`;
- local single-fixture runtime: `10.21192199550569 ms`;
- hardware status: simulator-only, no provider submission.

Field-coupling executable fixture status:

- implementation: `src/scpn_quantum_control/paper0/upde_field_validation.py`;
- tests: `tests/test_paper0_upde_field_validation.py`;
- result artefact: `paper0_upde_field_fixture_result_2026-05-13.json`;
- report: `paper0_upde_field_fixture_report_2026-05-13.md`;
- coherent field-alignment projection: `0.48216188793076825`;
- zero-field baseline, L-infinity: `0.0`;
- randomised-phase projection absolute mean: `0.029577256961382775`;
- bounded amplitude `zeta_L * Psi_Global`: `0.714`;
- local single-fixture runtime: `6.5162550017703325 ms`;
- hardware status: simulator-only, no provider submission.

Natural-gradient executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/upde_natural_gradient_validation.py`;
- tests: `tests/test_paper0_upde_natural_gradient_validation.py`;
- result artefact:
  `paper0_upde_natural_gradient_fixture_result_2026-05-13.json`;
- report:
  `paper0_upde_natural_gradient_fixture_report_2026-05-13.md`;
- free energy: `0.175945`;
- finite-difference gradient error, L-infinity:
  `4.081540661005079e-12`;
- FIM condition number: `2.07465066550813`;
- Euclidean-versus-natural drift difference, L2:
  `0.10222084991284003`;
- identity-FIM versus Euclidean drift, L-infinity: `0.0`;
- regularised singular-metric response, L-infinity: `119880.0`;
- local single-fixture runtime: `1.7767710087355226 ms`;
- hardware status: simulator-only, no provider submission.

Adaptive-coupling executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/upde_adaptive_coupling_validation.py`;
- tests: `tests/test_paper0_upde_adaptive_coupling_validation.py`;
- result artefact:
  `paper0_upde_adaptive_coupling_fixture_result_2026-05-13.json`;
- report:
  `paper0_upde_adaptive_coupling_fixture_report_2026-05-13.md`;
- `K_dot` L-infinity: `0.1405`;
- `eta_dot`: `-0.08099999999999997`;
- bounded update L-infinity: `0.007025000000000001`;
- zero-gain nulls: `0.0`;
- wrong-sign `K_dot` response, L2: `0.509493866498901`;
- wrong-sign `eta_dot` response: `0.16199999999999995`;
- local single-fixture runtime: `1.2968219962203875 ms`;
- hardware status: simulator-only, no provider submission.

Aggregate UPDE validation index status:

- implementation: `scripts/build_paper0_upde_validation_index.py`;
- tests: `tests/test_build_paper0_upde_validation_index.py`;
- result artefact:
  `paper0_upde_aggregate_validation_index_2026-05-13.json`;
- report: `paper0_upde_aggregate_validation_index_2026-05-13.md`;
- promoted specs: 5;
- fixture results: 5;
- coverage status: match;
- total local fixture runtime: `21.53322300000582 ms`;
- all fixture results are simulator-only and no provider submission is
  represented.

Paper 0 topology source-boundary status:

- implementation: `src/scpn_quantum_control/paper0/topology_schema.py`;
- artefact builder: `scripts/build_paper0_topology_schema.py`;
- tests: `tests/test_paper0_topology_schema.py`,
  `tests/test_build_paper0_topology_schema.py`;
- result artefact: `paper0_topology_source_boundary_2026-05-13.json`;
- report: `paper0_topology_source_boundary_2026-05-13.md`;
- layers represented: 16, including Meta-Layer 16;
- coupling channels: downward generation, upward inference, recursive closure;
- field port: source-anchored global `C_Field` port;
- adaptive parameter set: source-anchored quasicritical controller;
- numeric coupling matrix: none exported;
- synthetic controls: chain, ring, and complete graph are labelled as controls,
  not Paper 0 topology;
- hardware status: source-boundary only, no provider submission.

S19 resource-signature bridge status:

- implementation: `scripts/generate_s19_resource_signature_scan.py`;
- tests: `tests/test_generate_s19_resource_signature_scan.py`;
- manifest: `data/s19_resource_signatures/s19_scan_manifest_2026-05-13.json`;
- rows: `data/s19_resource_signatures/s19_resource_rows_2026-05-13.csv`;
- claim boundary:
  `data/s19_resource_signatures/s19_claim_boundary_2026-05-13.md`;
- Paper 0 topology source boundary is attached as provenance only;
- each simulated row carries a separate numeric-topology label;
- current regenerated run uses `synthetic_control.ring`;
- `paper0_source_boundary_only` is rejected as non-numeric topology;
- legacy `paper27` is labelled `legacy.paper27_provisional_not_paper0` and
  remains outside Paper 0 topology claims;
- hardware status: simulator-only, no provider submission.

Macro-transition validation-spec status:

- implementation:
  `scripts/build_paper0_macro_transition_validation_specs.py`;
- tests: `tests/test_build_paper0_macro_transition_validation_specs.py`;
- result artefact:
  `paper0_macro_transition_validation_specs_2026-05-13.json`;
- report:
  `paper0_macro_transition_validation_specs_report_2026-05-13.md`;
- source records: 10;
- consumed source records: 10;
- coverage status: match;
- promoted specs: `nths.spin_glass_hamiltonian`,
  `macro_transition.effective_coupling_rg`;
- all promoted specs include source ledgers, canonical equations, variables,
  assumptions, null controls, executable validation targets, and
  implementation links;
- hardware status: simulator-only, no provider submission.

NTHS spin-glass executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/nths_spin_glass_validation.py`;
- runner: `scripts/run_paper0_nths_spin_glass_fixture.py`;
- tests: `tests/test_paper0_nths_spin_glass_validation.py`,
  `tests/test_run_paper0_nths_spin_glass_fixture.py`;
- result artefact:
  `paper0_nths_spin_glass_fixture_result_2026-05-13.json`;
- report:
  `paper0_nths_spin_glass_fixture_report_2026-05-13.md`;
- source equation consumed: `EQ0113`;
- exact finite-state count: 64;
- ground-state energy: `-7.14`;
- mean energy: `2.7755575615628914e-17`;
- ground-state magnetisation: `0.0`;
- Edwards-Anderson `q_EA`: `0.020833333333333332`;
- ultrametric violation: `0.33333333333333337`;
- shuffled-coupling energy delta: `8.42`;
- zero-field energy delta: `0.1200000000000001`;
- ferromagnetic-control aligned magnetisation absolute value: `1.0`;
- local single-fixture runtime: `3.70090598880779 ms`;
- hardware status: simulator-only, no provider submission.

Macro-transition RG executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/macro_transition_rg_validation.py`;
- runner: `scripts/run_paper0_macro_transition_rg_fixture.py`;
- tests: `tests/test_paper0_macro_transition_rg_validation.py`,
  `tests/test_run_paper0_macro_transition_rg_fixture.py`;
- result artefact:
  `paper0_macro_transition_rg_fixture_result_2026-05-13.json`;
- report:
  `paper0_macro_transition_rg_fixture_report_2026-05-13.md`;
- source equation consumed: `EQ0114`;
- finite positive scale grid count: 33;
- integration variable: `log(mu)`;
- initial `K_eff`: `0.22`;
- final `K_eff`: `1.1650566085435246`;
- fixed-point candidate: `1.25`;
- fixed-point stability: `stable`;
- zero-beta invariance L-infinity: `0.0`;
- constant-beta analytic error L-infinity: `2.220446049250313e-16`;
- reverse-beta final delta: `12.404552117294841`;
- local single-fixture runtime: `1.6594469925621524 ms`;
- hardware status: simulator-only, no provider submission.

Neurovascular validation-spec status:

- implementation:
  `scripts/build_paper0_neurovascular_validation_specs.py`;
- tests: `tests/test_build_paper0_neurovascular_validation_specs.py`;
- result artefact:
  `paper0_neurovascular_validation_specs_2026-05-13.json`;
- report:
  `paper0_neurovascular_validation_specs_report_2026-05-13.md`;
- source records: 15;
- consumed source records: 15;
- coverage status: match;
- promoted specs: `embodied.neurovascular_phase_coupling`;
- source equation consumed: `EQ0093`;
- all promoted specs include source ledgers, canonical equations, variables,
  assumptions, null controls, executable validation targets, biomedical
  boundary controls, and implementation links;
- hardware status: simulator-only, no provider submission.

Neurovascular executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/neurovascular_validation.py`;
- runner: `scripts/run_paper0_neurovascular_fixture.py`;
- tests: `tests/test_paper0_neurovascular_validation.py`,
  `tests/test_run_paper0_neurovascular_fixture.py`;
- result artefact:
  `paper0_neurovascular_fixture_result_2026-05-13.json`;
- report:
  `paper0_neurovascular_fixture_report_2026-05-13.md`;
- source equation consumed: `EQ0093`;
- two-oscillator sample count: 4,001;
- analysis start index: 2,000;
- phase-locking value: `1.0`;
- mean frequency slip: `1.904006730057972e-09`;
- final phase difference: `0.07148944988588823`;
- zero `K_NH` slip absolute value: `0.029999999999992432`;
- detuned phase-locking drop: `0.6233920792184904`;
- shuffled-drive phase-locking drop: `0.9870974275090869`;
- impaired-CBF boundary label: `1.0`;
- local single-fixture runtime: `23.570152989123017 ms`;
- hardware status: simulator-only, no provider submission.

## P0 Source Canonicalisation

1. **Equation canonicalisation register**
   - Input: `paper0_equations_2026-05-13.jsonl`.
   - Output: one canonical record per mathematical statement.
   - Required fields: equation id, manuscript section, corrected LaTeX,
     variables, dimensions, assumptions, valid regime, numerical parameters,
     and falsification target.

2. **UPDE family**
   - Canonicalise the base phase equation.
   - Canonicalise intra-layer coupling `K_ij^L`.
   - Canonicalise `C_InterLayer` and its top-down/bottom-up terms.
   - Canonicalise `C_Field`.
   - Canonicalise noise `eta_i^L(t)` and stochastic assumptions.
   - Canonicalise the information-geometric/natural-gradient lift.
   - Canonicalise adaptive coupling and criticality-control laws.

3. **Claim and mechanism canonicalisation register**
   - Input: `paper0_exhaustive_register_2026-05-13.jsonl`.
   - Required action: review every record exactly once.
   - Output classes: context, mathematical claim, physical mechanism,
     theorem/proof obligation, experiment proposal, parameter definition,
     topology/coupling definition, literature citation, figure/table evidence,
     duplicate, or rejected/non-operational text.
   - Required invariant: reviewed records plus explicitly deferred records
     equals 7,129.

4. **Paper 0 topology schema**
   - Define layers, oscillators, intra-layer edges, inter-layer edges,
     field-coupling ports, delays, and adaptive parameters.
   - Replace any unproven `paper27` helper usage with a provenance-labelled
     topology source.
   - Keep synthetic chain/ring/all-to-all controls explicitly labelled as
     controls, not Paper 0 topology.
   - Current status: source-boundary schema exists and deliberately exports no
     numeric `K_nm` matrix.
   - Current bridge status: S19 scans consume the boundary as provenance only
     and require explicit experimental or synthetic numeric-topology labels.

## P1 Executable Validation Experiments

1. **UPDE-to-XY gradient check**
   - Claim: Kuramoto/UPDE phase dynamics descend a negative-cosine potential
     under stated assumptions.
   - Test: finite-difference gradient check against analytic dynamics.
   - Null: random non-symmetric coupling, sign-flipped coupling, and shuffled
     phases must fail the same alignment.

2. **FIM natural-gradient check**
   - Claim: information-geometric lift behaves as natural gradient flow.
   - Test: compare Euclidean and FIM-preconditioned flows on known statistical
     manifolds.
   - Null: singular/ill-conditioned FIM must trigger regularisation or fail
     fast, not silently proceed.

3. **Inter-layer coupling decomposition**
   - Claim: `C_InterLayer` separates downward prediction and upward error
     aggregation.
   - Test: independently perturb lower and upper layer means and verify the
     correct term responds.
   - Null: disconnected inter-layer graph must report zero inter-layer
     transfer.

4. **Field-coupling control**
   - Claim: `C_Field` phase-locks local oscillators to global field phase under
     finite `zeta_L`.
   - Test: order-parameter and phase-error reduction versus zero-field control.
   - Null: randomised global phase must remove the effect.

5. **Adaptive coupling and quasicritical controller**
   - Claim: `dot K_ij^L` and `dot eta^L` stabilise the system near
     `sigma_L = 1`.
   - Test: convergence and overshoot under bounded perturbations.
   - Null: wrong-sign feedback must destabilise or miss the target.

6. **Boundary/topology dependence**
   - Claim: open-chain, periodic, complete, and heterogeneous topologies have
     distinct finite-size regimes.
   - Test: scan algebraic connectivity, graph spectra, onset estimates, and
     observable-family control status.
   - Null: topology labels must not be inferred without matrix evidence.

7. **Resource-signature bridge**
   - Claim: entanglement, magic, pairing, and Krylov features may align with
     Paper 0 synchronisation onsets in some regimes.
   - Test: reuse S19 diagnostics only after Paper 0 topology provenance is
     attached.
   - Null: off-onset controls remain mandatory.
   - Current status: provenance bridge complete for simulator-only S19 scans;
     larger ensembles still need explicit numeric-topology labels.

## P2 Broader Theorem Lanes

- MS-QEC hierarchy: map each error-correction claim to a substrate, noise
  model, code/stabiliser structure, and survival observable.
- Quasicriticality/SOC: test scale-free statistics, avalanche distributions,
  and controller stability under finite sampling.
- Noospheric spin-glass Hamiltonian: validate coupling/frustration semantics
  and metastable-minima detection.
- Tri-axial embodied UPDE: validate brain-heart-gut oscillator coupling using
  phase-locking, delay, and HRV-observable controls.
- Abiogenesis/cellular sigma dynamics: validate coupled `sigma`, calcium, and
  glial-control equations with parameter sweeps and null controls.

## Glial-Control Validation-Spec Status

- implementation:
  `scripts/build_paper0_glial_control_validation_specs.py`;
- tests: `tests/test_build_paper0_glial_control_validation_specs.py`,
  `tests/test_paper0_equation_register.py`;
- result artefact:
  `paper0_glial_control_validation_specs_2026-05-13.json`;
- report:
  `paper0_glial_control_validation_specs_report_2026-05-13.md`;
- source equations promoted: `EQ0105`, `EQ0106`, `EQ0107`, `EQ0108`,
  `EQ0109`, `EQ0110`, `EQ0111`, `EQ0112`;
- promoted specs: `embodied.quantum_immune_interface`,
  `embodied.glial_sigma_control`;
- source records: 26;
- consumed source records: 26;
- coverage status: match;
- null controls and executable validation targets: present for both specs;
- hardware status: simulator-only, no provider submission.

Glial-control executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/glial_control_validation.py`;
- runner: `scripts/run_paper0_glial_control_fixture.py`;
- tests: `tests/test_paper0_glial_control_validation.py`,
  `tests/test_run_paper0_glial_control_fixture.py`;
- result artefact:
  `paper0_glial_control_fixture_result_2026-05-13.json`;
- report:
  `paper0_glial_control_fixture_report_2026-05-13.md`;
- `EQ0105` immune-interface Hamiltonian: Hermiticity error `0.0`,
  cytokine spectral shift `0.11399999999999999`, zero-lambda operator norm
  `0.0`;
- `EQ0106`--`EQ0112` glial sigma-control ODE: final `sigma`
  `1.4412587576405256`, final `G` `1.2050071405534404`,
  integrated calcium drive `26.688677641175836`;
- blockade/null controls: `gamma_zero_blockade_attenuation`
  `0.4412587576405256`, zero-calcium final `G` `0.0`, baseline sigma
  relaxation error `1.4646062140855065e-12`;
- hardware status: simulator-only, no provider submission.

## Computational-Unifier Validation-Spec Status

- implementation:
  `scripts/build_paper0_information_thermodynamics_validation_specs.py`;
- tests: `tests/test_build_paper0_information_thermodynamics_validation_specs.py`,
  `tests/test_paper0_equation_register.py`;
- result artefact:
  `paper0_information_thermodynamics_validation_specs_2026-05-13.json`;
- report:
  `paper0_information_thermodynamics_validation_specs_report_2026-05-13.md`;
- source equations promoted: `EQ0115`, `EQ0116`, `EQ0117`, `EQ0118`;
- promoted specs: `computational.cyclic_operator_boundary`,
  `computational.tsvf_abl_boundary`,
  `computational.info_thermodynamics`;
- source records: 23;
- consumed source records: 23;
- coverage status: match;
- null controls and executable validation targets: present for all specs;
- hardware status: simulator-only, no provider submission.

Computational-unifier executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/computational_unifier_validation.py`;
- runner: `scripts/run_paper0_computational_unifier_fixture.py`;
- tests: `tests/test_paper0_computational_unifier_validation.py`,
  `tests/test_run_paper0_computational_unifier_fixture.py`;
- result artefact:
  `paper0_computational_unifier_fixture_result_2026-05-13.json`;
- report:
  `paper0_computational_unifier_fixture_report_2026-05-13.md`;
- `EQ0115` cyclic-operator boundary: unitarity error
  `1.1102230246251565e-16`, cycle-closure residual
  `1.0778315928076987e-15`, wrong-period residual
  `1.949855824363647`;
- `EQ0116` TSVF/ABL boundary probability: probabilities
  `[0.7777777777777779, 0.2222222222222222]`, probability-normalisation
  error `0.0`, zero-denominator rejection label `1.0`;
- `EQ0117`--`EQ0118` information-thermodynamics budget: negentropy rate
  `0.12`, information-entropy rate `0.2`, total entropy/GSL margin
  `0.08000000000000002`, MI-negentropy error `0.0`;
- hardware status: simulator-only, no provider submission.

## Computational-Threshold Validation-Spec Status

- implementation:
  `scripts/build_paper0_computational_threshold_validation_specs.py`;
- tests: `tests/test_build_paper0_computational_threshold_validation_specs.py`,
  `tests/test_paper0_equation_register.py`;
- result artefact:
  `paper0_computational_threshold_validation_specs_2026-05-13.json`;
- report:
  `paper0_computational_threshold_validation_specs_report_2026-05-13.md`;
- source equations promoted: `EQ0119`, `EQ0120`, `EQ0121`, `EQ0122`;
- promoted specs: `computational.iit_or_threshold`,
  `computational.coherence_noether_current`,
  `computational.information_energy_transduction`;
- source records: 16;
- consumed source records: 16;
- coverage status: match;
- null controls and executable validation targets: present for all specs;
- hardware status: simulator-only, no provider submission.

Computational-threshold executable fixture status:

- implementation:
  `src/scpn_quantum_control/paper0/computational_threshold_validation.py`;
- runner: `scripts/run_paper0_computational_threshold_fixture.py`;
- tests: `tests/test_paper0_computational_threshold_validation.py`,
  `tests/test_run_paper0_computational_threshold_fixture.py`;
- result artefact:
  `paper0_computational_threshold_fixture_result_2026-05-13.json`;
- report:
  `paper0_computational_threshold_fixture_report_2026-05-13.md`;
- `EQ0119` IIT-OR threshold boundary: proportionality residual `0.0`,
  threshold labels `[0, 0, 1, 1]`, alpha-zero energy max abs `0.0`;
- `EQ0120` Noether coherence current: global phase invariance error
  `3.552713678800501e-15`, divergence residual
  `2.9309593421095067e-13`, phase-broken divergence residual
  `4.257194491746628`;
- `EQ0121`--`EQ0122` IET quantum potential: constant-density max abs
  `0.0`, Gaussian residual RMS `4.1875808543721506e-05`,
  non-positive-rho rejection label `1.0`;
- hardware status: simulator-only, no provider submission.

## Immediate Next Slice

Promote the next Paper 0 source cluster after `EQ0122` into source-covered
validation specs. Keep the same gate: source records first, canonical equation
register, validation specs, executable simulator fixture, null controls,
provenance, and only then any paper claim or provider plan.
