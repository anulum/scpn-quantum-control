# Complete Python module and API catalog

This static catalog makes every ordinary Python module and every public
module-level class or function discoverable without importing optional
provider, accelerator, or scientific dependencies.

- **697 modules** across **39 package families**
- **3908 documented public module-level symbols**
- **841 root-package exports** governed by the stable API surface

The catalog is an inventory, not a stability or product claim. Start with
the [API selection guide](../api.md) and [stable facades](../stable_facades_api.md).
Advanced modules may require optional extras and may be research-only.

## How to use this catalog

Search for a module, class, function, or domain term. Each module gives its
source summary and exact source link. Symbol behavior, parameters, returns,
exceptions, and boundaries live in the source docstrings rendered by the
curated [advanced autodoc reference](../autodoc.md).

## `analog_mapping`

### `scpn_quantum_control.analog_mapping.calibrate`

Analytic design-unit coupling-scale objective and drift sensitivity.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analog_mapping/calibrate.py) · Public symbols: **4**

**Classes:** `CalibrationEvaluation`, `CalibrationSensitivity`

**Functions:** `coupling_scale_objective()`, `calibration_sensitivity()`

### `scpn_quantum_control.analog_mapping.compare`

Bounded mathematical-model comparison against a Lie–Trotter reference.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analog_mapping/compare.py) · Public symbols: **2**

**Classes:** `AnalogDigitalComparison`

**Functions:** `compare_analog_model_to_trotter()`

### `scpn_quantum_control.analog_mapping.contracts`

Typed contracts for bounded analog oscillator mapping feasibility.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analog_mapping/contracts.py) · Public symbols: **5**

**Classes:** `AnalogPlatformProfile`, `MappingRequest`, `FeasibilityDiagnostic`, `MappingResult`, `FeasibilityReport`

### `scpn_quantum_control.analog_mapping.evidence`

Deterministic analog-mapping evidence bundle builder, renderer, and writer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analog_mapping/evidence.py) · Public symbols: **4**

**Classes:** `AnalogMappingEvidenceBundle`

**Functions:** `build_analog_mapping_evidence()`, `write_analog_mapping_evidence()`, `analog_mapping_markdown()`

### `scpn_quantum_control.analog_mapping.feasibility`

Fail-closed analog topology, control, range, and measurement diagnostics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analog_mapping/feasibility.py) · Public symbols: **3**

**Functions:** `assess_mapping_feasibility()`, `classify_topology()`, `reconstruct_compiled_couplings()`

### `scpn_quantum_control.analog_mapping.platforms`

Load and validate the packaged static analog platform catalogue.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analog_mapping/platforms.py) · Public symbols: **2**

**Functions:** `load_platform_profiles()`, `platform_profile()`

## `analysis`

### `scpn_quantum_control.analysis.adaptive_fim_evidence`

Digest-bound calibration controls and offline FIM custody replay.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/adaptive_fim_evidence.py) · Public symbols: **8**

**Functions:** `canonical_adaptive_fim_json()`, `historical_replay_witnesses()`, `synthetic_calibration_witnesses()`, `adaptive_fim_evidence_payload()`, `validate_adaptive_fim_evidence()`, `render_adaptive_fim_evidence_markdown()`, `write_adaptive_fim_evidence()`, `main()`

### `scpn_quantum_control.analysis.adaptive_fim_feedback`

Uncertainty-aware, policy-bounded adaptive FIM batch proposals.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/adaptive_fim_feedback.py) · Public symbols: **13**

**Classes:** `AdaptiveFIMConfig`, `FIMWitness`, `BinomialInterval`, `AdaptiveFIMStep`, `AdaptiveFIMObserverRecord`, `AdaptiveFIMPlan`

**Functions:** `wilson_score_interval()`, `propose_next_lambda()`, `propose_count_aware_lambda()`, `adaptive_lambda_schedule()`, `adaptive_count_aware_schedule()`, `plan_adaptive_fim_schedule()`, `observer_record_from_step()`

### `scpn_quantum_control.analysis.berry_phase`

Finite-size Berry diagnostics for exact Kuramoto-XY ground-state scans.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/berry_phase.py) · Public symbols: **2**

**Classes:** `BerryPhaseResult`

**Functions:** `berry_phase_scan()`

### `scpn_quantum_control.analysis.bkt_analysis`

BKT phase transition analysis for Kuramoto-XY systems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/bkt_analysis.py) · Public symbols: **6**

**Classes:** `BKTResult`

**Functions:** `coupling_laplacian()`, `fiedler_eigenvalue()`, `estimate_t_bkt()`, `bkt_analysis()`, `scan_synchronization_transition()`

### `scpn_quantum_control.analysis.bkt_universals`

BKT universal amplitude-ratio audit: p_h1 = 0.72 remains an open question.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/bkt_universals.py) · Public symbols: **3**

**Classes:** `UniversalCheckResult`, `BKTUniversalsSummary`

**Functions:** `check_all_candidates()`

### `scpn_quantum_control.analysis.critical_concordance`

Finite-size critical-probe concordance for exact Kuramoto-XY scans.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/critical_concordance.py) · Public symbols: **2**

**Classes:** `ConcordanceResult`

**Functions:** `critical_concordance()`

### `scpn_quantum_control.analysis.dla_parity_exact_baseline`

Exact, noiseless statevector reference for the DLA-parity XY-Trotter circuits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/dla_parity_exact_baseline.py) · Public symbols: **6**

**Classes:** `ExactBaselineRow`

**Functions:** `coupling_matrix()`, `initial_parity()`, `build_statevector_circuit()`, `exact_parity_leakage()`, `exact_baseline_grid()`

### `scpn_quantum_control.analysis.dla_parity_theorem`

DLA parity theorem for the heterogeneous XY Hamiltonian.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/dla_parity_theorem.py) · Public symbols: **9**

**Classes:** `DLAParityTheoremResult`

**Functions:** `predicted_dla_dimension()`, `parity_sector_dimensions()`, `su_dimension()`, `verify_theorem()`, `verify_all_known()`, `parity_operator()`, `project_to_parity_sector()`, `decompose_state_by_parity()`

### `scpn_quantum_control.analysis.dla_parity_witness`

DLA parity witness observable for bitstring-count analyses.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/dla_parity_witness.py) · Public symbols: **1**

**Classes:** `DLAParityWitness`

### `scpn_quantum_control.analysis.dla_truncated_tn`

Fail-fast interface for future DLA-truncated tensor-network simulations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/dla_truncated_tn.py) · Public symbols: **1**

**Functions:** `dla_truncated_tn()`

### `scpn_quantum_control.analysis.dynamical_lie_algebra`

Dynamical Lie algebra (DLA) computation for Hamiltonian simulability analysis.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/dynamical_lie_algebra.py) · Public symbols: **8**

**Classes:** `DLAResult`

**Functions:** `compute_dla()`, `compute_dla_rust()`, `build_xy_generators()`, `build_ssgf_generators()`, `build_pgbo_generators()`, `build_tcbo_generators()`, `build_full_scpn_generators()`

### `scpn_quantum_control.analysis.enaqt`

Bounded environment-assisted quantum transport (ENAQT) simulation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/enaqt.py) · Public symbols: **2**

**Classes:** `ENAQTResult`

**Functions:** `enaqt_scan()`

### `scpn_quantum_control.analysis.enaqt_evidence`

Deterministic, digest-bound evidence for the bounded ENAQT ENAQT scan.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/enaqt_evidence.py) · Public symbols: **8**

**Classes:** `ENAQTScenario`

**Functions:** `frozen_enaqt_scenarios()`, `canonical_enaqt_json()`, `enaqt_evidence_payload()`, `validate_enaqt_evidence()`, `render_enaqt_evidence_markdown()`, `write_enaqt_evidence()`, `main()`

### `scpn_quantum_control.analysis.entanglement_enhanced_sync`

Compare initial-state coherence under bounded Kuramoto-XY evolution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/entanglement_enhanced_sync.py) · Public symbols: **11**

**Classes:** `InitialState`, `SyncTrajectory`, `InitialStateControlComparison`

**Functions:** `prepare_initial_state()`, `local_phase_observables()`, `transverse_exchange_coherence()`, `mean_single_qubit_linear_entropy()`, `simulate_sync_trajectory()`, `compare_all_initial_states()`, `compare_initial_states_with_dephased_controls()`, `entanglement_advantage()`

### `scpn_quantum_control.analysis.entanglement_entropy`

Entanglement entropy and Schmidt gap at the synchronization transition.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/entanglement_entropy.py) · Public symbols: **4**

**Classes:** `EntanglementResult`, `EntanglementScanResult`

**Functions:** `entanglement_at_coupling()`, `entanglement_vs_coupling()`

### `scpn_quantum_control.analysis.entanglement_percolation`

Finite-size entanglement percolation diagnostics vs synchronisation proxies.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/entanglement_percolation.py) · Public symbols: **4**

**Classes:** `PercolationScanResult`

**Functions:** `concurrence_map_exact()`, `fiedler_eigenvalue()`, `percolation_scan()`

### `scpn_quantum_control.analysis.entanglement_spectrum`

Entanglement spectrum analysis at the synchronization transition.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/entanglement_spectrum.py) · Public symbols: **7**

**Classes:** `EntanglementResult`

**Functions:** `entanglement_entropy_half_chain()`, `entanglement_spectrum_half_chain()`, `entropy_vs_subsystem_size()`, `fit_cft_central_charge()`, `entanglement_analysis()`, `entropy_vs_coupling_scan()`

### `scpn_quantum_control.analysis.entanglement_sync_evidence`

Generate deterministic, digest-bound entanglement-sync initial-state evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/entanglement_sync_evidence.py) · Public symbols: **7**

**Functions:** `frozen_entanglement_sync_scenario()`, `canonical_entanglement_sync_json()`, `entanglement_sync_evidence_payload()`, `validate_entanglement_sync_evidence()`, `render_entanglement_sync_evidence_markdown()`, `write_entanglement_sync_evidence()`, `main()`

### `scpn_quantum_control.analysis.fim_hamiltonian`

Offline diagnostics for the FIM-augmented Kuramoto-XY Hamiltonian.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/fim_hamiltonian.py) · Public symbols: **12**

**Classes:** `SpectrumSummary`

**Functions:** `computational_magnetisations()`, `fim_diagonal()`, `add_fim_feedback()`, `magnetisation_sector_indices()`, `summarise_spectrum()`, `sector_spectrum_rows()`, `adjacent_gap_ratio()`, `bipartite_entropy_from_statevector()`, `magnetisation_operator_diagonal()`, `commutator_frobenius_norm_with_diagonal()`, `sector_coupling_rows()`

### `scpn_quantum_control.analysis.finite_size_scaling`

Finite-size scaling for K_c extraction from small exact quantum systems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/finite_size_scaling.py) · Public symbols: **3**

**Classes:** `FSSFitDiagnostics`, `FSSResult`

**Functions:** `finite_size_scaling()`

### `scpn_quantum_control.analysis.graph_topology_scan`

Graph topology → p_h1 systematic scan.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/graph_topology_scan.py) · Public symbols: **2**

**Classes:** `GraphP_H1_Result`

**Functions:** `scan_graph_topologies()`

### `scpn_quantum_control.analysis.h1_persistence`

Evaluate H1 persistence: p_h1 = 0.72 remains an open question.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/h1_persistence.py) · Public symbols: **2**

**Classes:** `H1PersistenceResult`

**Functions:** `scan_h1_persistence()`

### `scpn_quantum_control.analysis.hamiltonian_learning`

Bounded inverse fitting of ``K_nm`` from exact ground-state correlators.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/hamiltonian_learning.py) · Public symbols: **3**

**Classes:** `HamiltonianLearningResult`

**Functions:** `measure_correlators()`, `learn_hamiltonian()`

### `scpn_quantum_control.analysis.hamiltonian_self_consistency`

Hamiltonian self-consistency loop: quantum measurement → K_nm recovery.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/hamiltonian_self_consistency.py) · Public symbols: **6**

**Classes:** `SelfConsistencyResult`

**Functions:** `correlators_from_counts()`, `correlator_shot_noise()`, `self_consistency_from_exact()`, `self_consistency_from_counts()`, `self_consistency_from_noisy_sim()`

### `scpn_quantum_control.analysis.integrated_information_phi`

Fail-closed interface for integrated-information requests.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/integrated_information_phi.py) · Public symbols: **1**

**Classes:** `IntegratedInformationPhi`

### `scpn_quantum_control.analysis.koopman`

Finite local Koopman-style closure for the Kuramoto model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/koopman.py) · Public symbols: **6**

**Classes:** `KoopmanResult`

**Functions:** `build_koopman_generator()`, `build_koopman_generator_rust()`, `koopman_analysis()`, `koopman_dimension()`, `koopman_to_hamiltonian()`

### `scpn_quantum_control.analysis.krylov_complexity`

Krylov complexity at the synchronization transition.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/krylov_complexity.py) · Public symbols: **4**

**Classes:** `KrylovResult`

**Functions:** `lanczos_coefficients()`, `krylov_complexity()`, `krylov_vs_coupling()`

### `scpn_quantum_control.analysis.lindblad_ness`

Non-equilibrium steady state (NESS) of driven-dissipative Kuramoto-XY.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/lindblad_ness.py) · Public symbols: **4**

**Classes:** `NESSResult`, `NESSScanResult`

**Functions:** `compute_ness()`, `ness_vs_coupling()`

### `scpn_quantum_control.analysis.logical_sync_witness`

Logical synchronisation witness backed by the DLA-protected QEC sector.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/logical_sync_witness.py) · Public symbols: **1**

**Classes:** `LogicalSyncWitness`

### `scpn_quantum_control.analysis.loschmidt_echo`

Loschmidt echo and dynamical quantum phase transitions.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/loschmidt_echo.py) · Public symbols: **3**

**Classes:** `LoschmidtResult`

**Functions:** `loschmidt_quench()`, `quench_scan()`

### `scpn_quantum_control.analysis.magic_nonstabilizerness`

Exact small-system stabilizer Rényi-2 diagnostics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/magic_nonstabilizerness.py) · Public symbols: **4**

**Classes:** `MagicResult`, `MagicScanResult`

**Functions:** `magic_at_coupling()`, `magic_vs_coupling()`

### `scpn_quantum_control.analysis.magnetisation_sectors`

U(1) symmetry exploitation for the XY Hamiltonian.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/magnetisation_sectors.py) · Public symbols: **8**

**Functions:** `basis_by_magnetisation()`, `sector_dimensions()`, `largest_sector_dim()`, `project_to_sector()`, `build_sector_hamiltonian()`, `eigh_by_magnetisation()`, `level_spacing_by_magnetisation()`, `memory_estimate()`

### `scpn_quantum_control.analysis.monte_carlo_xy`

Monte Carlo simulation of the XY model on the K_nm graph.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/monte_carlo_xy.py) · Public symbols: **6**

**Classes:** `MCResult`, `AHPResult`, `FiniteSizeResult`

**Functions:** `mc_simulate()`, `extract_a_hp()`, `finite_size_scaling()`

### `scpn_quantum_control.analysis.otoc`

Out-of-time-order correlator (OTOC) for quantum chaos detection.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/otoc.py) · Public symbols: **3**

**Classes:** `OTOC`, `OTOCResult`

**Functions:** `compute_otoc()`

### `scpn_quantum_control.analysis.otoc_sync_probe`

OTOC as a synchronization transition probe.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/otoc_sync_probe.py) · Public symbols: **3**

**Classes:** `OTOCSyncScanResult`

**Functions:** `otoc_sync_scan()`, `compare_otoc_vs_R()`

### `scpn_quantum_control.analysis.p_h1_derivation`

Negative-control audit: p_h1 = 0.72 remains an open question.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/p_h1_derivation.py) · Public symbols: **2**

**Classes:** `P_H1_Derivation`

**Functions:** `derive_p_h1()`

### `scpn_quantum_control.analysis.p_h1_open_guard`

Public-claim guard: ``p_h1 = 0.72`` remains an open question.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/p_h1_open_guard.py) · Public symbols: **5**

**Classes:** `P_H1OpenGuardViolation`, `P_H1OpenGuardReport`

**Functions:** `public_markdown_paths()`, `validate_p_h1_open_claim_text()`, `run_p_h1_open_guard()`

### `scpn_quantum_control.analysis.pairing_correlator`

Richardson-Gaudin pairing correlators in the synchronised state.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/pairing_correlator.py) · Public symbols: **3**

**Classes:** `PairingResult`

**Functions:** `pairing_map()`, `pairing_vs_anisotropy()`

### `scpn_quantum_control.analysis.persistent_homology`

Persistent homology of oscillator phase configurations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/persistent_homology.py) · Public symbols: **4**

**Classes:** `PersistenceResult`

**Functions:** `phase_distance_matrix()`, `compute_persistence()`, `p_h1_vs_temperature()`

### `scpn_quantum_control.analysis.phase_diagram`

Quantum Kuramoto phase diagram: K_c vs effective temperature.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/phase_diagram.py) · Public symbols: **8**

**Classes:** `PhaseBoundary`, `PhaseDiagramResult`

**Functions:** `critical_coupling_finite_graph()`, `critical_coupling_mean_field()`, `decoherence_temperature()`, `effective_temperature()`, `order_parameter_steady_state()`, `compute_phase_diagram()`

### `scpn_quantum_control.analysis.qfi`

Quantum Fisher Information for coupling parameter estimation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/qfi.py) · Public symbols: **3**

**Classes:** `QFIResult`

**Functions:** `compute_qfi()`, `qfi_gap_tradeoff()`

### `scpn_quantum_control.analysis.qfi_criticality`

QFI divergence at the synchronization critical point.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/qfi_criticality.py) · Public symbols: **3**

**Classes:** `QFICriticalityResult`

**Functions:** `qfi_single_coupling()`, `qfi_vs_coupling()`

### `scpn_quantum_control.analysis.qfi_geometric_crosscheck`

Cross-validate the spectral QFI against the quantum-geometric-tensor route.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/qfi_geometric_crosscheck.py) · Public symbols: **2**

**Classes:** `QFIGeometricCrosscheck`

**Functions:** `crosscheck_qfi_geometric()`

### `scpn_quantum_control.analysis.qrc_phase_detector`

Exact finite-size feature extraction for a QRC-style phase detector.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/qrc_phase_detector.py) · Public symbols: **5**

**Classes:** `QRCPhaseResult`

**Functions:** `generate_training_data()`, `train_linear_readout()`, `classify()`, `qrc_phase_detection()`

### `scpn_quantum_control.analysis.quantum_fisher_information`

Quantum Fisher Information observable wrappers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/quantum_fisher_information.py) · Public symbols: **1**

**Classes:** `QuantumFisherInformation`

### `scpn_quantum_control.analysis.quantum_mpemba`

Quantum Mpemba effect in synchronization dynamics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/quantum_mpemba.py) · Public symbols: **2**

**Classes:** `MpembaResult`

**Functions:** `mpemba_experiment()`

### `scpn_quantum_control.analysis.quantum_persistent_homology`

Persistent homology on quantum measurement data.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/quantum_persistent_homology.py) · Public symbols: **6**

**Classes:** `QuantumPHResult`

**Functions:** `correlation_matrix_from_counts()`, `correlation_to_distance()`, `quantum_persistent_homology()`, `compare_quantum_classical_ph()`, `ph_sync_scan()`

### `scpn_quantum_control.analysis.quantum_phi`

Minimum bipartite quantum mutual information from a density matrix.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/quantum_phi.py) · Public symbols: **6**

**Classes:** `PhiResult`

**Functions:** `von_neumann_entropy()`, `partial_trace()`, `mutual_information()`, `compute_quantum_phi()`, `phi_vs_coupling_scan()`

### `scpn_quantum_control.analysis.quantum_speed_limit`

Quantum speed limits for a bounded local-phase-order threshold.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/quantum_speed_limit.py) · Public symbols: **3**

**Classes:** `QSLResult`

**Functions:** `compute_qsl()`, `qsl_vs_coupling()`

### `scpn_quantum_control.analysis.research_lane_registry`

Governed catalogue of the package's analysis and gauge research lanes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/research_lane_registry.py) · Public symbols: **13**

**Classes:** `ResearchLaneMaturity`, `ResearchLaneDiffHook`, `ResearchLaneClaimStatus`, `ResearchLaneRecord`, `ResearchLaneInventoryReport`, `ResearchLaneRegistryReport`

**Functions:** `list_research_lanes()`, `get_research_lane()`, `discover_research_lane_modules()`, `validate_research_lane_inventory()`, `assert_research_lane_inventory()`, `build_research_lane_registry_report()`, `render_research_lane_registry_markdown()`

### `scpn_quantum_control.analysis.rl_discovery_agent`

Compatibility wrapper for governed Kuramoto witness discovery.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/rl_discovery_agent.py) · Public symbols: **1**

**Classes:** `RLDiscoveryAgent`

### `scpn_quantum_control.analysis.rl_pulse_optimizer`

Fail-fast, research-disabled interface for future RL pulse optimisation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/rl_pulse_optimizer.py) · Public symbols: **1**

**Classes:** `RLPulseOptimizer`

### `scpn_quantum_control.analysis.rl_research_governance`

Fail-closed governance for witness-search and pulse-optimisation research.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/rl_research_governance.py) · Public symbols: **13**

**Classes:** `RLResearchLane`, `RLResearchGovernanceError`, `RLResearchPolicy`, `RLResearchDecision`, `RLSeedEvaluation`, `RLSeedSuiteReport`

**Functions:** `estimate_witness_evaluation_budget()`, `assess_rl_research()`, `assert_rl_research_allowed()`, `build_witness_seed_suite()`, `run_governed_witness_seed_suite()`, `build_rl_research_evidence_report()`, `render_rl_research_evidence_markdown()`

### `scpn_quantum_control.analysis.sensing`

No-submit S11 sync-order quantum-sensing readiness model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/sensing.py) · Public symbols: **9**

**Classes:** `QuantumSensingReadinessConfig`, `SensingGainRow`, `SensingGainScan`, `CriticalitySensingTail`

**Functions:** `metrological_gain_vs_k()`, `optimal_sensing_k()`, `qfi_criticality_sensing_tail()`, `quantum_sensing_payload()`, `quantum_sensing_markdown()`

### `scpn_quantum_control.analysis.shadow_tomography`

Classical shadow tomography for efficient state characterisation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/shadow_tomography.py) · Public symbols: **3**

**Classes:** `ShadowResult`

**Functions:** `estimate_pauli_expectation()`, `classical_shadow_estimation()`

### `scpn_quantum_control.analysis.spectral_form_factor`

Finite-size spectral form-factor and adjacent-gap diagnostics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/spectral_form_factor.py) · Public symbols: **4**

**Classes:** `SFFResult`, `SFFScanResult`

**Functions:** `compute_sff()`, `sff_vs_coupling()`

### `scpn_quantum_control.analysis.symmetry_sectors`

Symmetry-aware exact diagonalisation for the XY Hamiltonian.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/symmetry_sectors.py) · Public symbols: **6**

**Functions:** `basis_indices_by_parity()`, `project_hamiltonian()`, `build_sector_hamiltonian()`, `eigh_by_sector()`, `level_spacing_by_sector()`, `memory_estimate_mb()`

### `scpn_quantum_control.analysis.sync_entanglement_witness`

The order parameter R as an entanglement witness.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/sync_entanglement_witness.py) · Public symbols: **6**

**Classes:** `EntanglementWitnessResult`

**Functions:** `R_separable_bound()`, `R_separable_bound_at_energy()`, `R_from_statevector()`, `detect_entanglement_from_R()`, `R_entanglement_scan()`

### `scpn_quantum_control.analysis.sync_order_parameter`

Z-basis synchronisation proxy observable for measurement counts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/sync_order_parameter.py) · Public symbols: **1**

**Classes:** `SyncOrderParameter`

### `scpn_quantum_control.analysis.sync_uncertainty`

Shot-noise uncertainty quantification for synchronisation metrics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/sync_uncertainty.py) · Public symbols: **5**

**Classes:** `UncertaintyInterval`

**Functions:** `order_parameter_estimate()`, `order_parameter_shot_noise()`, `order_parameter_bootstrap()`, `metric_bootstrap()`

### `scpn_quantum_control.analysis.sync_witness`

Quantum synchronization witness operators.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/sync_witness.py) · Public symbols: **8**

**Classes:** `WitnessResult`

**Functions:** `correlation_witness_from_counts()`, `fiedler_witness_from_correlator()`, `fiedler_witness_from_counts()`, `topological_witness_from_correlator()`, `evaluate_all_witnesses()`, `calibrate_thresholds()`, `build_correlation_witness_operator()`

### `scpn_quantum_control.analysis.tcbo_weighted_complex`

Coupling-weighted simplicial complex for TCBO p_h1 reconstruction.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/tcbo_weighted_complex.py) · Public symbols: **7**

**Classes:** `TCBOWeightedComplexResult`, `TCBOWeightedThresholdScan`, `TCBOWeightedReplayUncertainty`

**Functions:** `coupling_weighted_edge_matrix()`, `tcbo_weighted_complex()`, `tcbo_weighted_threshold_scan()`, `tcbo_weighted_uncertainty_replay()`

### `scpn_quantum_control.analysis.theory_hook_promotion`

Evidence-gated promotion records for experimental theory hooks.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/theory_hook_promotion.py) · Public symbols: **11**

**Classes:** `TheoryHookTier`, `TheoryHookRole`, `TheoryHookStatus`, `TheoryHookPromotionRecord`, `TheoryHookEvidenceRecord`, `TheoryHookPromotionReport`

**Functions:** `list_theory_hook_promotions()`, `get_theory_hook_promotion()`, `run_theory_hook_evidence()`, `build_theory_hook_promotion_report()`, `render_theory_hook_promotion_markdown()`

### `scpn_quantum_control.analysis.thermodynamic_witness`

Thermodynamic witness observable for calibrated work samples.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/thermodynamic_witness.py) · Public symbols: **1**

**Classes:** `ThermodynamicWitness`

### `scpn_quantum_control.analysis.translation_symmetry`

Translation symmetry exploitation for homogeneous Kuramoto-XY chains.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/translation_symmetry.py) · Public symbols: **4**

**Functions:** `is_translation_invariant()`, `momentum_sectors()`, `momentum_sector_dimensions()`, `eigh_with_translation()`

### `scpn_quantum_control.analysis.two_colour_schedule`

Genuine width-2 hand scheduling of the 1-D XY-Trotter chain (audit AUD-7).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/two_colour_schedule.py) · Public symbols: **6**

**Functions:** `two_colour_edges()`, `build_two_colour_circuit()`, `build_sequential_circuit()`, `two_colour_parity_leakage()`, `two_qubit_depth()`, `depth_comparison()`

### `scpn_quantum_control.analysis.vortex_binding`

Vortex binding energy and Kosterlitz renormalization group.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/vortex_binding.py) · Public symbols: **6**

**Classes:** `VortexBindingResult`

**Functions:** `vortex_pair_energy()`, `vortex_pair_entropy()`, `vortex_free_energy()`, `kosterlitz_rg_step()`, `compute_vortex_binding()`

### `scpn_quantum_control.analysis.witness_discovery`

Automated Kuramoto witness discovery with Bayesian and bandit search.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/witness_discovery.py) · Public symbols: **7**

**Classes:** `WitnessSearchMode`, `WitnessCandidate`, `WitnessDiscoverySpec`, `WitnessDiscoveryEvaluation`, `WitnessDiscoveryResult`

**Functions:** `discover_kuramoto_witnesses()`, `score_witness_candidates()`

### `scpn_quantum_control.analysis.xxz_phase_diagram`

Finite-size XXZ anisotropy diagnostics for the XY-to-Heisenberg crossover.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/xxz_phase_diagram.py) · Public symbols: **4**

**Classes:** `AnisotropyScanResult`, `PhaseDiagramResult`

**Functions:** `scan_coupling_at_delta()`, `anisotropy_phase_diagram()`

## `applications`

### `scpn_quantum_control.applications.app_plugins`

Application-specific plugin registry for benchmark datasets and workflows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/app_plugins.py) · Public symbols: **17**

**Classes:** `ApplicationPluginBenchmark`, `ApplicationPlugin`, `ApplicationPluginRegistry`, `EEGApplicationPlugin`, `PlasmaApplicationPlugin`, `PowerGridApplicationPlugin`, `FEPApplicationPlugin`

**Functions:** `eeg_application_plugin_factory()`, `plasma_application_plugin_factory()`, `power_grid_application_plugin_factory()`, `fep_application_plugin_factory()`, `get_application_plugin_registry()`, `discover_application_plugins()`, `get_application_plugin()`, `run_application_benchmark_suite()`, `load_application_dataset()`, `compile_application_problem()`

### `scpn_quantum_control.applications.cross_domain`

Cross-domain structural comparison for SCPN K_nm and domain matrices.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/cross_domain.py) · Public symbols: **2**

**Classes:** `CrossDomainResult`

**Functions:** `run_cross_domain_validation()`

### `scpn_quantum_control.applications.dataset_catalog`

Packaged application benchmark datasets exposed as QPU data artifacts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/dataset_catalog.py) · Public symbols: **6**

**Classes:** `ApplicationBenchmarkDescriptor`, `ApplicationBenchmarkPrivacyAudit`

**Functions:** `list_application_benchmark_descriptors()`, `get_application_benchmark_descriptor()`, `load_application_benchmark_artifact()`, `audit_application_benchmark_privacy()`

### `scpn_quantum_control.applications.disruption_classifier`

Quantum disruption classifier for tokamak plasma.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/disruption_classifier.py) · Public symbols: **5**

**Classes:** `DisruptionClassifierResult`

**Functions:** `generate_synthetic_disruption_data()`, `train_disruption_classifier()`, `predict_disruption()`, `run_disruption_benchmark()`

### `scpn_quantum_control.applications.eeg_benchmark`

EEG neural oscillator coupling benchmark.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/eeg_benchmark.py) · Public symbols: **3**

**Classes:** `EEGBenchmarkResult`

**Functions:** `eeg_coupling_matrix()`, `eeg_benchmark()`

### `scpn_quantum_control.applications.eeg_classification`

EEG state classification via structured VQE and quantum kernels.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/eeg_classification.py) · Public symbols: **3**

**Classes:** `EEGVQEResult`

**Functions:** `eeg_plv_to_vqe()`, `eeg_quantum_kernel()`

### `scpn_quantum_control.applications.fmo_benchmark`

FMO photosynthetic complex benchmark against SCPN coupling.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/fmo_benchmark.py) · Public symbols: **3**

**Classes:** `FMOBenchmarkResult`

**Functions:** `fmo_coupling_matrix()`, `fmo_benchmark()`

### `scpn_quantum_control.applications.honesty_kits`

Fail-closed claim and data boundaries for domain-facing applications.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/honesty_kits.py) · Public symbols: **9**

**Classes:** `ApplicationSupportStatus`, `ApplicationDataOrigin`, `DomainApplicationHonestyKit`, `ApplicationHonestyAuditReport`

**Functions:** `list_domain_application_honesty_kits()`, `get_domain_application_honesty_kit()`, `get_domain_application_honesty_kit_for_dataset()`, `build_application_honesty_audit_report()`, `render_application_honesty_audit_markdown()`

### `scpn_quantum_control.applications.iter_benchmark`

ITER synthetic data benchmark: tokamak MHD mode coupling.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/iter_benchmark.py) · Public symbols: **3**

**Classes:** `ITERBenchmarkResult`

**Functions:** `iter_coupling_matrix()`, `iter_benchmark()`

### `scpn_quantum_control.applications.josephson_array`

Josephson junction array mapping for the self-simulation narrative.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/josephson_array.py) · Public symbols: **4**

**Classes:** `JosephsonArrayParameters`, `JosephsonBenchmarkResult`

**Functions:** `jja_coupling_matrix()`, `josephson_benchmark()`

### `scpn_quantum_control.applications.josephson_magnitude_study`

Josephson-array K_nm magnitude-study preregistration.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/josephson_magnitude_study.py) · Public symbols: **5**

**Classes:** `JosephsonKnmCandidate`, `JosephsonMagnitudeGate`, `JosephsonMagnitudeStudyDesign`

**Functions:** `build_josephson_knm_magnitude_study_design()`, `render_josephson_knm_magnitude_study_markdown()`

### `scpn_quantum_control.applications.power_grid`

Power grid synchronisation benchmark using IEEE test cases.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/power_grid.py) · Public symbols: **5**

**Classes:** `PowerGridBenchmarkResult`

**Functions:** `ieee_14bus_susceptance_matrix()`, `ieee_14bus_admittance_coupling_matrix()`, `ieee_5bus_coupling_matrix()`, `power_grid_benchmark()`

### `scpn_quantum_control.applications.qrc_baseline`

QRC comparison helpers with a deterministic classical ESN baseline.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/qrc_baseline.py) · Public symbols: **7**

**Classes:** `ClassicalESNReadoutResult`, `QRCBaselineComparison`, `QRCHoldoutComparison`

**Functions:** `classical_esn_feature_matrix()`, `classical_esn_ridge_regression()`, `compare_quantum_reservoir_to_esn()`, `compare_quantum_reservoir_to_esn_holdout()`

### `scpn_quantum_control.applications.quantum_evs`

Quantum-enhanced EVS (Emergent Value Signature) for CCW.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/quantum_evs.py) · Public symbols: **2**

**Classes:** `QuantumEVSResult`

**Functions:** `quantum_evs_enhance()`

### `scpn_quantum_control.applications.quantum_kernel`

Finite simulator fidelity kernels informed by oscillator coupling.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/quantum_kernel.py) · Public symbols: **5**

**Classes:** `QuantumKernelResult`

**Functions:** `canonical_edge_pairs()`, `encode_topology_edge_features()`, `quantum_kernel_entry()`, `compute_kernel_matrix()`

### `scpn_quantum_control.applications.quantum_reservoir`

Quantum reservoir computing for exponential state space expansion.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/quantum_reservoir.py) · Public symbols: **4**

**Classes:** `ReservoirResult`

**Functions:** `reservoir_features()`, `reservoir_feature_matrix()`, `reservoir_ridge_regression()`

### `scpn_quantum_control.applications.quantum_reservoir_product`

Held-out synthetic QRC certificates and exact reservoir objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/applications/quantum_reservoir_product.py) · Public symbols: **6**

**Classes:** `ReservoirTaskKind`, `SyntheticReservoirDataset`, `ReservoirTrainingCertificate`, `ReservoirLinearObjective`

**Functions:** `generate_synthetic_reservoir_task()`, `certify_reservoir_training()`

## `benchmark_harness`

### `scpn_quantum_control.benchmark_harness.registry`

Registry of public benchmark-harness families and readiness status.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmark_harness/registry.py) · Public symbols: **3**

**Classes:** `BenchmarkFamily`

**Functions:** `list_benchmark_families()`, `benchmark_registry_payload()`

### `scpn_quantum_control.benchmark_harness.synchronisation`

Canonical synchronisation benchmark registry and schema.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmark_harness/synchronisation.py) · Public symbols: **3**

**Classes:** `SynchronisationBenchmarkInstance`

**Functions:** `list_synchronisation_benchmarks()`, `synchronisation_benchmark_registry_payload()`

### `scpn_quantum_control.benchmark_harness.synchronisation_compare`

Tolerance comparator for synchronisation benchmark result artefacts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmark_harness/synchronisation_compare.py) · Public symbols: **7**

**Classes:** `ObservableComparison`

**Functions:** `load_payload()`, `validate_payload_shape()`, `observable_index()`, `compare_payloads()`, `compare_files()`, `compare_default_artifacts()`

### `scpn_quantum_control.benchmark_harness.synchronisation_runner`

No-QPU runners for canonical synchronisation benchmark instances.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmark_harness/synchronisation_runner.py) · Public symbols: **12**

**Classes:** `ObservableRow`, `BenchmarkResultRow`

**Functions:** `ring_coupling_matrix()`, `decaying_chain_coupling_matrix()`, `natural_frequencies()`, `kuramoto_order_parameter()`, `run_classical_reference()`, `xy_hamiltonian()`, `run_exact_reference()`, `dependency_lock()`, `run_kuramoto_ring_n4_linear_omega()`, `run_kuramoto_chain_n8_decay_omega()`

## `benchmarks`

### `scpn_quantum_control.benchmarks.advantage_protocol`

Claim-bounded protocol for S2 scaling and advantage benchmarks.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/advantage_protocol.py) · Public symbols: **5**

**Classes:** `ScalingBaseline`, `ScalingProtocol`, `ScalingRowValidation`

**Functions:** `validate_scaling_rows()`, `default_s2_scaling_protocol()`

### `scpn_quantum_control.benchmarks.appqsim_protocol`

AppQSim benchmarking protocol: standardised metrics for publication.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/appqsim_protocol.py) · Public symbols: **2**

**Classes:** `AppQSimMetrics`

**Functions:** `appqsim_benchmark()`

### `scpn_quantum_control.benchmarks.classical_baselines`

Documented classical baselines for Kuramoto-XY workflows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/classical_baselines.py) · Public symbols: **6**

**Classes:** `ClassicalBaselineRun`

**Functions:** `available_baselines()`, `scipy_ode_baseline()`, `qutip_lindblad_baseline()`, `mps_tebd_baseline()`, `run_documented_classical_baselines()`

### `scpn_quantum_control.benchmarks.closed_loop_publication_run`

Reproducible software-in-the-loop closed-loop publication artifact.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/closed_loop_publication_run.py) · Public symbols: **7**

**Classes:** `FeedbackControllerLike`, `LatencyMeasurer`, `ControllerFactory`, `ClosedLoopRunConfig`, `ClosedLoopPublicationArtifact`

**Functions:** `dynamic_circuit_templates()`, `run_closed_loop_publication()`

### `scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence`

Attachment-grade isolated benchmark evidence for compiler promotion gates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/compiler_isolated_benchmark_evidence.py) · Public symbols: **5**

**Classes:** `CompilerIsolatedBenchmarkEvidenceFiles`, `CompilerIsolatedBenchmarkEvidence`

**Functions:** `build_compiler_isolated_benchmark_evidence()`, `render_compiler_isolated_benchmark_evidence_markdown()`, `write_compiler_isolated_benchmark_evidence()`

### `scpn_quantum_control.benchmarks.coupling_recovery_evidence`

Benchmark artefacts for bounded coupling time-series recovery.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/coupling_recovery_evidence.py) · Public symbols: **4**

**Classes:** `CouplingRecoveryEvidenceArtifact`

**Functions:** `coupling_recovery_evidence_payload()`, `render_coupling_recovery_evidence_markdown()`, `write_coupling_recovery_evidence_artifact()`

### `scpn_quantum_control.benchmarks.decisive_advantage_protocol`

Single-decision benchmark protocol for the Kuramoto-XY advantage question.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/decisive_advantage_protocol.py) · Public symbols: **6**

**Classes:** `DecisionCriterion`, `SubmissionGate`, `DecisionOutcome`, `DecisiveAdvantageProtocol`

**Functions:** `evaluate_decision()`, `default_decisive_advantage_protocol()`

### `scpn_quantum_control.benchmarks.decisive_run_harness`

Measured classical-baseline harness feeding the decisive-advantage gate.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/decisive_run_harness.py) · Public symbols: **9**

**Classes:** `DecisiveRunConfig`, `DecisiveRunArtifact`

**Functions:** `git_commit()`, `command_line()`, `dependency_versions()`, `dense_reference_row()`, `ode_row()`, `mps_row()`, `run_decisive_benchmark()`

### `scpn_quantum_control.benchmarks.differentiable_catalyst_comparison`

Catalyst compiler-workflow evidence boundaries for differentiable comparisons.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_catalyst_comparison.py) · Public symbols: **2**

**Classes:** `CatalystCompilerWorkflowComparison`

**Functions:** `catalyst_compiler_workflow_comparison()`

### `scpn_quantum_control.benchmarks.differentiable_evidence`

CI-only benchmark evidence metadata and artefact writers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_evidence.py) · Public symbols: **9**

**Classes:** `AcceleratorEvidenceMetadata`, `BenchmarkIsolationMetadata`, `DifferentiableBenchmarkEvidenceBundle`

**Functions:** `write_differentiable_benchmark_evidence_bundle()`, `capture_host_load()`, `read_cpu_governor()`, `read_cpu_frequency_mhz()`, `capture_accelerator_metadata()`, `infer_heavy_jobs_running()`

### `scpn_quantum_control.benchmarks.differentiable_external_comparison`

External framework comparison harness for bounded Phase-QNode claims.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py) · Public symbols: **5**

**Functions:** `run_differentiable_external_comparison_suite()`, `write_differentiable_external_comparison()`, `run_identical_circuit_gradient_comparison_suite()`, `write_identical_circuit_gradient_comparison()`, `external_comparison_failure_mode_rows()`

### `scpn_quantum_control.benchmarks.differentiable_external_contracts`

Immutable records and field vocabulary for external differentiable comparisons.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_external_contracts.py) · Public symbols: **4**

**Classes:** `ExternalComparisonRow`, `ExternalComparisonArtifact`, `IdenticalCircuitGradientComparisonRow`, `IdenticalCircuitGradientComparisonArtifact`

### `scpn_quantum_control.benchmarks.differentiable_hardening_gate`

Per-slice differentiable-programming hardening gate.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py) · Public symbols: **4**

**Classes:** `DifferentiableHardeningGateCheck`, `DifferentiableBenchmarkClassificationCase`, `DifferentiableHardeningSliceGateResult`

**Functions:** `run_differentiable_hardening_slice_gate()`

### `scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan`

Isolated benchmark batch plan for differentiable promotion evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py) · Public symbols: **6**

**Classes:** `DifferentiableIsolatedBenchmarkPlanRow`, `DifferentiableIsolatedBenchmarkPlan`, `DifferentiableIsolatedBenchmarkPlanValidation`

**Functions:** `run_differentiable_isolated_benchmark_plan()`, `validate_differentiable_isolated_benchmark_plan()`, `render_differentiable_isolated_benchmark_plan_markdown()`

### `scpn_quantum_control.benchmarks.differentiable_optimizer_convergence`

Benchmark artefacts for ground-state optimizer convergence rows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_optimizer_convergence.py) · Public symbols: **4**

**Classes:** `GroundStateOptimizerConvergenceArtifact`

**Functions:** `ground_state_optimizer_convergence_payload()`, `render_ground_state_optimizer_convergence_markdown()`, `write_ground_state_optimizer_convergence_artifact()`

### `scpn_quantum_control.benchmarks.differentiable_programming`

Deterministic differentiable-programming conformance benchmark cases.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_programming.py) · Public symbols: **3**

**Functions:** `run_differentiable_programming_benchmark_suite()`, `run_quantum_gradient_benchmark_suite()`, `run_differentiable_programming_external_reference_suite()`

### `scpn_quantum_control.benchmarks.differentiable_programming_contracts`

Dependency-light result contracts shared by differentiable benchmark suites.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_programming_contracts.py) · Public symbols: **3**

**Classes:** `DifferentiableProgrammingBenchmarkResult`, `DifferentiableProgrammingExternalReferenceResult`, `QuantumGradientBenchmarkResult`

### `scpn_quantum_control.benchmarks.differentiable_programming_quantum`

Quantum-gradient case builders for the differentiable conformance benchmark suite.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/differentiable_programming_quantum.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.benchmarks.gpu_baseline`

GPU baseline comparison for quantum simulation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/gpu_baseline.py) · Public symbols: **8**

**Classes:** `GPUBaselineResult`

**Functions:** `statevector_memory_gb()`, `statevector_flops()`, `estimate_gpu_time()`, `estimate_qpu_time()`, `gate_count_xy_trotter()`, `gpu_baseline_comparison()`, `scaling_comparison()`

### `scpn_quantum_control.benchmarks.hls_cosimulation_evidence`

Hash-bound software co-simulation evidence for the pulse→HLS lane.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/hls_cosimulation_evidence.py) · Public symbols: **7**

**Classes:** `CosimulationRunner`, `CosimulationEvidence`, `HLSCosimulationConfig`, `HLSCosimulationHandoff`

**Functions:** `host_compiler_identity()`, `run_hls_cosimulation()`, `run_hls_cosimulation_handoff()`

### `scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark`

Benchmark harness for the IQM Garnet layout-transfer preregistration.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/iqm_layout_transfer_benchmark.py) · Public symbols: **16**

**Classes:** `DepthParityResult`, `ArmPlan`, `SizeBlockPlan`, `LayoutTransferPlan`

**Functions:** `initial_bitstring()`, `exact_order_parameter()`, `coupling_map_from_calibration()`, `chain_swap_depth_provider()`, `naive_chain_layout()`, `optimised_initial_layout()`, `measured_physical_qubits()`, `per_qubit_one_probabilities()`, `per_qubit_readout_errors()`, `corrected_order_parameter()`, `depth_parity_gate()`, `build_layout_transfer_plan()`

### `scpn_quantum_control.benchmarks.iqm_layout_transfer_per_size`

Frozen IQM layout-transfer circuit matrix and decision-rule implementation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/iqm_layout_transfer_per_size.py) · Public symbols: **4**

**Classes:** `PerSizeLayoutTransferPlan`

**Functions:** `build_per_size_layout_transfer_plan()`, `holm_adjusted_p_values()`, `analyse_per_size_counts()`

### `scpn_quantum_control.benchmarks.isolated_host_readiness`

Decide whether a host can produce ``isolated_affinity`` benchmark evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/isolated_host_readiness.py) · Public symbols: **3**

**Classes:** `HostReadiness`

**Functions:** `assess_host_readiness()`, `capture_host_readiness()`

### `scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark`

Measured head-to-head comparison of our Kuramoto toolkit against external solvers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kuramoto_competitive_benchmark.py) · Public symbols: **2**

**Classes:** `KuramotoCompetitiveComparison`

**Functions:** `run_kuramoto_competitive_comparison()`

### `scpn_quantum_control.benchmarks.kuramoto_competitive_types`

Problem and row types shared by the competitive harness and its adapters.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kuramoto_competitive_types.py) · Public symbols: **3**

**Classes:** `KuramotoProblem`, `CompetitorRow`

**Functions:** `build_default_problem()`

### `scpn_quantum_control.benchmarks.kuramoto_external_competitors`

Real third-party solver adapters for the Kuramoto competitive comparison.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kuramoto_external_competitors.py) · Public symbols: **11**

**Functions:** `default_julia_runner()`, `default_dynamicalsystems_runner()`, `default_networkdynamics_runner()`, `default_scimlsensitivity_runner()`, `default_jitcdde_runner()`, `scipy_row()`, `julia_diffeq_row()`, `dynamicalsystems_row()`, `networkdynamics_row()`, `scimlsensitivity_row()`, `jitcdde_row()`

### `scpn_quantum_control.benchmarks.kyma.dynamics`

Differentiable Kuramoto RK4 integrator and order-parameter readout.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma/dynamics.py) · Public symbols: **3**

**Functions:** `kuramoto_rhs()`, `integrate_kuramoto()`, `cluster_order_parameter()`

### `scpn_quantum_control.benchmarks.kyma.models`

The motif substrate, the parameter-matched MLP baseline, and the chance floor.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma/models.py) · Public symbols: **10**

**Functions:** `substrate_param_count()`, `substrate_init()`, `substrate_readout()`, `train_substrate()`, `mlp_hidden_for_match()`, `mlp_param_count()`, `mlp_init()`, `mlp_forward()`, `train_mlp()`, `chance_floor_accuracy()`

### `scpn_quantum_control.benchmarks.kyma.probe`

Train, evaluate, and aggregate the KYMA composition probe over seeds.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma/probe.py) · Public symbols: **3**

**Classes:** `SeedResult`

**Functions:** `run_seed()`, `run_probe()`

### `scpn_quantum_control.benchmarks.kyma.task`

Cluster-pair relations, input encoding, and the compositional split.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma/task.py) · Public symbols: **7**

**Classes:** `ProbeConfig`, `TrialBatch`

**Functions:** `pair_members()`, `disjoint_conjunctions()`, `encode()`, `build_trials()`, `success_mask()`

### `scpn_quantum_control.benchmarks.kyma_v2.ablations`

Ablation models that each remove ONE of the two v2 fixes (KYMA v2.1 #1).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/ablations.py) · Public symbols: **5**

**Functions:** `shared_param_count()`, `shared_init()`, `shared_final_phases()`, `train_shared()`, `shared_predict()`

### `scpn_quantum_control.benchmarks.kyma_v2.baselines`

Stronger non-oscillator baselines for the v2 task (KYMA v2.1 #2).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/baselines.py) · Public symbols: **10**

**Functions:** `deep_mlp_param_count()`, `deep_mlp_init()`, `deep_mlp_logits()`, `train_deep_mlp()`, `deep_mlp_predict()`, `gnn_param_count()`, `gnn_init()`, `gnn_logits()`, `train_gnn()`, `gnn_predict()`

### `scpn_quantum_control.benchmarks.kyma_v2.coupling`

Assemble the per-trial gated coupling ``K_eff(code)`` (KYMA v2 fix 1).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/coupling.py) · Public symbols: **6**

**Functions:** `ambient_matrix()`, `partners_for()`, `readout_bridge_matrix()`, `base_coupling_matrix()`, `assemble_coupling()`, `symmetrise()`

### `scpn_quantum_control.benchmarks.kyma_v2.design`

Fix the v2 design constants from **teacher dynamics only** — never a model's held-out accuracy (pre-registration §5).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/design.py) · Public symbols: **4**

**Functions:** `single_relation_realisability()`, `non_separability_rate()`, `class_histogram()`, `select_config()`

### `scpn_quantum_control.benchmarks.kyma_v2.dynamics`

Differentiable Kuramoto RK4 integrator with a **per-trial** coupling matrix.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/dynamics.py) · Public symbols: **4**

**Functions:** `kuramoto_rhs_batched()`, `integrate_kuramoto_batched()`, `order_parameter()`, `phase_label()`

### `scpn_quantum_control.benchmarks.kyma_v2.models`

The trainable gated **student** substrate, a parameter-matched MLP baseline, and the chance floor.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/models.py) · Public symbols: **12**

**Functions:** `substrate_param_count()`, `student_init()`, `student_final_phases()`, `train_student()`, `student_predict()`, `mlp_hidden_for_match()`, `mlp_param_count()`, `mlp_init()`, `mlp_logits()`, `train_mlp()`, `mlp_predict()`, `chance_floor_accuracy()`

### `scpn_quantum_control.benchmarks.kyma_v2.probe`

Train, evaluate, and aggregate the KYMA v2 composition probe over seeds.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/probe.py) · Public symbols: **3**

**Classes:** `SeedResultV2`

**Functions:** `run_seed()`, `run_probe()`

### `scpn_quantum_control.benchmarks.kyma_v2.rigor`

Orchestrate the four v2.1 supplementary analyses (ablations, stronger baselines, MLP convergence, leave-one-out) against the frozen v2 task.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/rigor.py) · Public symbols: **4**

**Functions:** `run_ablations()`, `run_stronger_baselines()`, `mlp_convergence()`, `run_leave_one_out()`

### `scpn_quantum_control.benchmarks.kyma_v2.task`

Cluster-pair relations, gated-coupling masks, and the compositional split.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/task.py) · Public symbols: **8**

**Classes:** `ProbeConfigV2`, `TrialBatchV2`

**Functions:** `pair_members()`, `in_phase_mask()`, `anti_phase_masks()`, `disjoint_conjunctions()`, `encode()`, `build_trials()`

### `scpn_quantum_control.benchmarks.kyma_v2.teacher`

The fixed gated-oscillator **teacher** that defines every trial's label.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/kyma_v2/teacher.py) · Public symbols: **4**

**Functions:** `teacher_gates()`, `teacher_final_phases()`, `teacher_labels()`, `label_batch()`

### `scpn_quantum_control.benchmarks.layout_method_comparison`

Honest comparison of layout methods for the Kuramoto XY-Trotter circuit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/layout_method_comparison.py) · Public symbols: **10**

**Classes:** `RProvider`, `MetricsProvider`, `RoutedLayoutMetrics`, `MethodRow`, `LayoutComparisonConfig`, `LayoutComparisonArtifact`

**Functions:** `coupling_map_from_gate_errors()`, `ideal_xy_order_parameter()`, `routed_layout_metrics()`, `run_layout_method_comparison()`

### `scpn_quantum_control.benchmarks.layout_relaxation_experiment`

RESEARCH: the KT-4 preregistered seed-sweep experiment.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/layout_relaxation_experiment.py) · Public symbols: **5**

**Classes:** `RelaxationExperimentInstance`, `InstanceOutcome`, `RelaxationExperimentArtifact`

**Functions:** `preregistered_instances()`, `run_layout_relaxation_experiment()`

### `scpn_quantum_control.benchmarks.mps_baseline`

MPS tensor network baseline for quantum advantage comparison.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/mps_baseline.py) · Public symbols: **6**

**Classes:** `MPSBaselineResult`

**Functions:** `required_bond_dimension()`, `mps_memory()`, `exact_memory()`, `quantum_advantage_n()`, `mps_baseline_comparison()`

### `scpn_quantum_control.benchmarks.open_system_objective_evidence`

Benchmark artefacts for bounded Lindblad and MCWF objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/open_system_objective_evidence.py) · Public symbols: **4**

**Classes:** `OpenSystemObjectiveEvidenceArtifact`

**Functions:** `open_system_objective_evidence_payload()`, `render_open_system_objective_evidence_markdown()`, `write_open_system_objective_evidence_artifact()`

### `scpn_quantum_control.benchmarks.quantum_advantage`

Quantum vs classical scaling benchmark for Kuramoto Hamiltonian simulation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/quantum_advantage.py) · Public symbols: **5**

**Classes:** `AdvantageResult`

**Functions:** `classical_benchmark()`, `quantum_benchmark()`, `estimate_crossover()`, `run_scaling_benchmark()`

### `scpn_quantum_control.benchmarks.reproducible_comparison`

Deterministic head-to-head comparison of classical and quantum Kuramoto routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/reproducible_comparison.py) · Public symbols: **3**

**Classes:** `ComparisonMethodRow`, `ReproducibleKuramotoComparison`

**Functions:** `run_reproducible_kuramoto_comparison()`

### `scpn_quantum_control.benchmarks.s3_design_protocol`

Claim-bounded S3 pulse and ansatz design scoring protocol.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/s3_design_protocol.py) · Public symbols: **8**

**Classes:** `S3DesignCandidate`, `S3DesignRow`, `S3DesignProtocol`

**Functions:** `default_s3_design_protocol()`, `score_s3_candidates()`, `validate_s3_design_rows()`, `generate_s3_candidate_grid()`, `grid_s3_design_protocol()`

### `scpn_quantum_control.benchmarks.sync_witness_evidence`

Benchmark artefacts for bounded synchronisation-witness runs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/sync_witness_evidence.py) · Public symbols: **4**

**Classes:** `SyncWitnessEvidenceArtifact`

**Functions:** `sync_witness_evidence_payload()`, `render_sync_witness_evidence_markdown()`, `write_sync_witness_evidence_artifact()`

### `scpn_quantum_control.benchmarks.tn_mps_baseline_design`

Design a CPU-first tensor-network MPS scaling baseline.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/tn_mps_baseline_design.py) · Public symbols: **5**

**Classes:** `TNBaselineAdapter`, `TNBaselineSizePlan`, `TNBaselineDesign`

**Functions:** `build_tn_mps_baseline_design()`, `render_tn_mps_baseline_design_markdown()`

### `scpn_quantum_control.benchmarks.tn_mps_crossover_admission`

Admission contract for larger-than-16-node TN/MPS crossover rows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/benchmarks/tn_mps_crossover_admission.py) · Public symbols: **7**

**Classes:** `TNMPSCrossoverRowSchema`, `TNMPSCrossoverGate`, `TNMPSCrossoverAdmissionReport`, `TNMPSCrossoverRowValidation`

**Functions:** `build_tn_mps_crossover_admission()`, `validate_tn_mps_crossover_rows()`, `render_tn_mps_crossover_admission_markdown()`

## `bridge`

### `scpn_quantum_control.bridge.control_plasma_knm`

Compatibility bridge for plasma-native Knm builders from scpn-control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/control_plasma_knm.py) · Public symbols: **4**

**Functions:** `build_knm_plasma()`, `build_knm_plasma_spec()`, `build_knm_plasma_from_config()`, `plasma_omega()`

### `scpn_quantum_control.bridge.fusion_core_frc`

Calibrate the quantum FRC scheduler surrogate from a SCPN-FUSION-CORE run.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/fusion_core_frc.py) · Public symbols: **4**

**Classes:** `FRCEquilibriumLike`, `FusionCoreFRCCalibration`

**Functions:** `calibrate_frc_surrogate_from_equilibrium()`, `calibrate_frc_surrogate_from_inputs()`

### `scpn_quantum_control.bridge.knm_hamiltonian`

Knm coupling matrix -> Pauli Hamiltonian compiler.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/knm_hamiltonian.py) · Public symbols: **8**

**Functions:** `omega_for_oscillators()`, `build_knm_paper27()`, `build_kuramoto_ring()`, `knm_to_xxz_hamiltonian()`, `knm_to_hamiltonian()`, `knm_to_sparse_matrix()`, `knm_to_dense_matrix()`, `knm_to_ansatz()`

### `scpn_quantum_control.bridge.orchestrator_adapter`

Adapters between scpn-phase-orchestrator state and quantum bridge artifacts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/orchestrator_adapter.py) · Public symbols: **1**

**Classes:** `PhaseOrchestratorAdapter`

### `scpn_quantum_control.bridge.orchestrator_feedback`

Orchestrator bidirectional feedback loop.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/orchestrator_feedback.py) · Public symbols: **2**

**Classes:** `OrchestratorFeedback`

**Functions:** `compute_orchestrator_feedback()`

### `scpn_quantum_control.bridge.phase_artifact`

Shared phase-state artifact schema for SCPN classical/quantum interoperability.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/phase_artifact.py) · Public symbols: **3**

**Classes:** `LockSignatureArtifact`, `LayerStateArtifact`, `UPDEPhaseArtifact`

### `scpn_quantum_control.bridge.qpu_data_artifact`

QPU-ready oscillator artifact with provenance gates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/qpu_data_artifact.py) · Public symbols: **6**

**Classes:** `QPUDataArtifact`

**Functions:** `read_qpu_data_artifact()`, `write_qpu_data_artifact()`, `validate_qpu_data_artifact()`, `artifact_to_kuramoto_problem()`, `artifact_from_arrays()`

### `scpn_quantum_control.bridge.sc_to_quantum`

Bitstream probability <-> quantum rotation angle converters.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/sc_to_quantum.py) · Public symbols: **4**

**Functions:** `probability_to_angle()`, `angle_to_probability()`, `bitstream_to_statevector()`, `measurement_to_bitstream()`

### `scpn_quantum_control.bridge.scpn_upde_edge`

Bounded ``knm.scpn-upde`` edge payloads for SPO federation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/scpn_upde_edge.py) · Public symbols: **5**

**Classes:** `SCPNUPDEEdge`

**Functions:** `edge_content_digest()`, `build_scpn_upde_edge()`, `build_paper27_scpn_upde_edge()`, `validate_scpn_upde_edge_payload()`

### `scpn_quantum_control.bridge.snn_adapter`

SNN <> quantum bridge: spike trains to rotation angles, measurements to currents.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/snn_adapter.py) · Public symbols: **4**

**Classes:** `SNNQuantumBridge`, `ArcaneNeuronBridge`

**Functions:** `spike_train_to_rotations()`, `quantum_measurement_to_current()`

### `scpn_quantum_control.bridge.snn_backward`

SNN backward pass: Ry parameter-shift gradient through the quantum layer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/snn_backward.py) · Public symbols: **2**

**Classes:** `BackwardResult`

**Functions:** `parameter_shift_gradient()`

### `scpn_quantum_control.bridge.sparse_hamiltonian`

Sparse CSC/CSR Hamiltonian for large-N Kuramoto-XY systems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/sparse_hamiltonian.py) · Public symbols: **4**

**Functions:** `build_sparse_hamiltonian()`, `build_sparse_sector_hamiltonian()`, `sparse_eigsh()`, `sparsity_stats()`

### `scpn_quantum_control.bridge.spn_to_qcircuit`

SPN topology -> quantum circuit compiler.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/spn_to_qcircuit.py) · Public symbols: **2**

**Functions:** `spn_to_circuit()`, `inhibitor_anti_control()`

### `scpn_quantum_control.bridge.ssgf_adapter`

SSGF <> quantum bridge: geometry matrices to Hamiltonians, states to circuits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/ssgf_adapter.py) · Public symbols: **4**

**Classes:** `SSGFQuantumLoop`

**Functions:** `ssgf_w_to_hamiltonian()`, `ssgf_state_to_quantum()`, `quantum_to_ssgf_state()`

### `scpn_quantum_control.bridge.ssgf_w_adapter`

SSGF geometry W adaptation from quantum R_global.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bridge/ssgf_w_adapter.py) · Public symbols: **2**

**Classes:** `WAdaptResult`

**Functions:** `adapt_w_from_quantum()`

## `chimera_control`

### `scpn_quantum_control.chimera_control.evidence`

Deterministic chimera-control evidence construction, rendering, and byte custody.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/chimera_control/evidence.py) · Public symbols: **6**

**Classes:** `ChimeraSupportRow`, `SyntheticRegimeEvidence`, `ChimeraMultiscaleEvidence`

**Functions:** `build_chimera_multiscale_evidence()`, `render_chimera_multiscale_markdown()`, `write_chimera_multiscale_evidence()`

### `scpn_quantum_control.chimera_control.objectives`

Differentiable hierarchy targets composed from existing analytic terms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/chimera_control/objectives.py) · Public symbols: **3**

**Classes:** `PhaseControlProposal`

**Functions:** `build_chimera_control_objective()`, `propose_phase_control_step()`

### `scpn_quantum_control.chimera_control.observables`

Hierarchy-aware order parameters composed from oscillatools diagnostics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/chimera_control/observables.py) · Public symbols: **3**

**Classes:** `LevelOrderParameterSummary`, `MultiscaleOrderParameterReport`

**Functions:** `measure_multiscale_order_parameters()`

### `scpn_quantum_control.chimera_control.schema`

Immutable hierarchy and target contracts for synthetic chimera control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/chimera_control/schema.py) · Public symbols: **6**

**Classes:** `SyntheticRegime`, `HierarchyLevel`, `MultiscaleHierarchy`, `HierarchyTarget`, `ChimeraControlSpecification`

**Functions:** `two_population_hierarchy()`

### `scpn_quantum_control.chimera_control.synthetic`

Exact finite-N two-population Kuramoto-Sakaguchi trajectory generation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/chimera_control/synthetic.py) · Public symbols: **4**

**Classes:** `SyntheticChimeraConfig`, `SyntheticChimeraRun`

**Functions:** `build_two_population_coupling()`, `generate_two_population_chimera()`

### `scpn_quantum_control.chimera_control.topology`

Hierarchy summaries around the existing topology-constraint ledger.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/chimera_control/topology.py) · Public symbols: **3**

**Classes:** `HierarchyCouplingSummary`, `TopologyProjectionReport`

**Functions:** `project_chimera_coupling()`

## `codegen`

### `scpn_quantum_control.codegen.ultrascale_hls`

Convert a quantum control pulse waveform into a Vivado/Vitis HLS bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codegen/ultrascale_hls.py) · Public symbols: **9**

**Classes:** `HLSBundle`, `HLSArtifactFile`, `HLSArtifactManifest`, `HLSArtifactVerification`

**Functions:** `quantise_q_format()`, `pulse_to_vivado_hls()`, `emit_versioned_hls_artifact()`, `verify_hls_artifact_manifest()`, `write_bundle()`

## `codesign`

### `scpn_quantum_control.codesign.adapters`

co-design adapters over control-stack control ports and active-sensing/69/70 observers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/adapters.py) · Public symbols: **6**

**Classes:** `ControlAdapterEvidence`

**Functions:** `observer_inputs_from_products()`, `adaptive_fim_proposal_port()`, `consume_realtime_feedback_port()`, `consume_qaoa_mpc_port()`, `consume_cosimulation_port()`

### `scpn_quantum_control.codesign.components`

Deterministic estimator, evaluator, and controller components for co-design.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/components.py) · Public symbols: **5**

**Classes:** `ExponentialOrderEstimator`, `OpenSystemObjectiveConfig`, `PhaseObjectiveSimulator`, `GradientFeedbackController`

**Functions:** `component_claim_boundary()`

### `scpn_quantum_control.codesign.contracts`

Immutable contracts for the simulator-first co-design co-design loop.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/contracts.py) · Public symbols: **18**

**Classes:** `CoDesignMode`, `SafetyAction`, `StaleGradientAction`, `BackendCapabilities`, `LoopStepInput`, `StateEstimate`, `GradientPlanRecord`, `QuantumEvaluation`, `ControllerProposal`, `LatencyDecision`, `SafetyDecision`, `ObserverInputs`, `LoopStepOutput`, `PlasmaObjectiveTemplate`, `StateEstimatorPort`, `QuantumEvaluationPort`, `ControllerPort`

**Functions:** `plasma_objective_templates()`

### `scpn_quantum_control.codesign.evidence`

Functional non-isolated evidence writer for the bounded co-design workflow.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/evidence.py) · Public symbols: **7**

**Classes:** `FunctionalEvidence`

**Functions:** `build_demo_loop()`, `demo_inputs()`, `run_functional_evidence()`, `write_functional_evidence()`, `validate_functional_evidence()`, `main()`

### `scpn_quantum_control.codesign.loop`

Thin deterministic orchestration over co-design estimator and policy ports.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/loop.py) · Public symbols: **1**

**Classes:** `CoDesignLoop`

### `scpn_quantum_control.codesign.policies`

Fail-closed latency and controller-envelope policies for co-design.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/policies.py) · Public symbols: **2**

**Classes:** `LatencyPolicy`, `SafetyEnvelope`

### `scpn_quantum_control.codesign.replay`

Versioned trace recording and bit-stable replay verification for co-design.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/codesign/replay.py) · Public symbols: **3**

**Classes:** `ReplayTrace`

**Functions:** `record_replay_trace()`, `verify_replay_trace()`

## `compiler`

### `scpn_quantum_control.compiler.alias_activity_evidence`

Compiler alias-activity evidence assembled from Program AD lattice reports.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/alias_activity_evidence.py) · Public symbols: **4**

**Classes:** `CompilerAliasActivityCase`, `CompilerAliasActivityEvidence`

**Functions:** `build_compiler_alias_activity_evidence()`, `render_compiler_alias_activity_evidence_markdown()`

### `scpn_quantum_control.compiler.mlir`

Stable MLIR compiler facade over focused implementation leaves.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.compiler.mlir_enzyme_audit`

Enzyme/MLIR toolchain probing and bounded maturity aggregation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_enzyme_audit.py) · Public symbols: **1**

**Functions:** `run_enzyme_mlir_maturity_audit()`

### `scpn_quantum_control.compiler.mlir_enzyme_evidence`

Evidence records and builders for the Enzyme/MLIR compiler-AD maturity surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_enzyme_evidence.py) · Public symbols: **15**

**Classes:** `EnzymeMLIRCompilerADBreadthArtifactFiles`, `EnzymeMLIRToolchainStatus`, `EnzymeNativeExecutionEvidence`, `MLIRLLVMCorrectnessEvidence`, `EnzymeMLIRBenchmarkAttachment`, `EnzymeMLIRCompilerADBreadthCaseEvidence`, `EnzymeMLIRCompilerADBreadthArtifact`, `EnzymeMLIRCompilerADBreadthEvidence`, `EnzymeMLIRMaturityAuditResult`

**Functions:** `build_enzyme_mlir_benchmark_attachment()`, `build_enzyme_mlir_compiler_ad_breadth_artifact()`, `build_enzyme_mlir_compiler_ad_breadth_gap_artifact()`, `build_enzyme_mlir_compiler_ad_breadth_evidence()`, `render_enzyme_mlir_compiler_ad_breadth_artifact_markdown()`, `write_enzyme_mlir_compiler_ad_breadth_artifact()`

### `scpn_quantum_control.compiler.mlir_enzyme_execution_runner`

Capture real Enzyme/LLVM reverse-mode AD execution evidence beyond scalar replay.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_enzyme_execution_runner.py) · Public symbols: **4**

**Classes:** `EnzymeToolchainADCase`, `EnzymeToolchainADExecutionEvidence`

**Functions:** `resolve_enzyme_toolchain()`, `run_enzyme_toolchain_execution_evidence()`

### `scpn_quantum_control.compiler.mlir_executable_kernel`

Executable compiler-AD kernel: batching, verification and custom-derivative lowering.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_executable_kernel.py) · Public symbols: **3**

**Classes:** `ExecutableCompilerADKernel`

**Functions:** `make_executable_ad_kernel_batching_rule()`, `compile_custom_derivative_rule_to_mlir()`

### `scpn_quantum_control.compiler.mlir_llvm_jit_claim_gate`

Promotion gate for native LLVM/JIT differentiable-programming claims.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_llvm_jit_claim_gate.py) · Public symbols: **4**

**Classes:** `LLVMJITClaimGate`

**Functions:** `build_llvm_jit_claim_gate()`, `llvm_jit_claim_gate_from_dict()`, `render_llvm_jit_claim_gate_markdown()`

### `scpn_quantum_control.compiler.mlir_matrix_2x2_native_compilation`

Native LLVM/JIT autodiff compilation for closed-form 2x2 dense linear algebra.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_matrix_2x2_native_compilation.py) · Public symbols: **15**

**Functions:** `compile_matrix_2x2_determinant_ad_to_native_llvm_jit()`, `make_matrix_2x2_determinant_native_llvm_jit_lowering_rule()`, `make_matrix_2x2_determinant_native_llvm_jit_primitive_transform()`, `compile_matrix_2x2_inverse_ad_to_native_llvm_jit()`, `make_matrix_2x2_inverse_native_llvm_jit_lowering_rule()`, `make_matrix_2x2_inverse_native_llvm_jit_primitive_transform()`, `compile_matrix_2x2_solve_ad_to_native_llvm_jit()`, `make_matrix_2x2_solve_native_llvm_jit_lowering_rule()`, `make_matrix_2x2_solve_native_llvm_jit_primitive_transform()`, `compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit()`, `make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule()`, `make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform()`, `compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit()`, `make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule()`, `make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform()`

### `scpn_quantum_control.compiler.mlir_matrix_native_compilation`

Native LLVM/JIT autodiff compilation for matrix primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_matrix_native_compilation.py) · Public symbols: **15**

**Functions:** `compile_matrix_vector_product_ad_to_native_llvm_jit()`, `make_matrix_vector_product_native_llvm_jit_lowering_rule()`, `make_matrix_vector_product_native_llvm_jit_primitive_transform()`, `compile_matrix_matrix_product_ad_to_native_llvm_jit()`, `make_matrix_matrix_product_native_llvm_jit_lowering_rule()`, `make_matrix_matrix_product_native_llvm_jit_primitive_transform()`, `compile_matrix_trace_ad_to_native_llvm_jit()`, `make_matrix_trace_native_llvm_jit_lowering_rule()`, `make_matrix_trace_native_llvm_jit_primitive_transform()`, `compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit()`, `make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule()`, `make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform()`, `compile_matrix_quadratic_form_ad_to_native_llvm_jit()`, `make_matrix_quadratic_form_native_llvm_jit_lowering_rule()`, `make_matrix_quadratic_form_native_llvm_jit_primitive_transform()`

### `scpn_quantum_control.compiler.mlir_native_execution_evidence`

Evidence records for executed native LLVM/JIT whole-program autodiff.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_native_execution_evidence.py) · Public symbols: **4**

**Classes:** `NativeWholeProgramADExecutionCase`, `NativeWholeProgramADExecutionEvidence`

**Functions:** `build_native_whole_program_ad_execution_evidence()`, `run_native_whole_program_ad_execution_evidence()`

### `scpn_quantum_control.compiler.mlir_native_primitives`

Low-level primitives shared by the matrix-JIT and whole-program native lowering paths.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_native_primitives.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.compiler.mlir_phase_qnode_runtime`

Registered Phase-QNode MLIR lowering and verified runtime execution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_phase_qnode_runtime.py) · Public symbols: **3**

**Classes:** `PhaseQNodeMLIRRuntimeExecutable`

**Functions:** `lower_phase_qnode_circuit_to_mlir()`, `compile_phase_qnode_circuit_to_mlir_runtime()`

### `scpn_quantum_control.compiler.mlir_records`

Value records for the MLIR compilation and compiler-AD lowering surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_records.py) · Public symbols: **7**

**Classes:** `MLIRCompileConfig`, `MLIRModule`, `PrimitiveLoweringStatus`, `CompilerADTransformPlan`, `DifferentiableMLIRCompileConfig`, `CompilerADExecutableConfig`, `CompilerADKernelVerification`

### `scpn_quantum_control.compiler.mlir_scalar_native_compilation`

Native LLVM/JIT autodiff compilation for scalar primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_scalar_native_compilation.py) · Public symbols: **6**

**Functions:** `compile_scalar_quadratic_ad_to_native_llvm_jit()`, `make_scalar_quadratic_native_llvm_jit_lowering_rule()`, `compile_scalar_unary_elementwise_ad_to_native_llvm_jit()`, `make_scalar_unary_elementwise_native_llvm_jit_lowering_rule()`, `compile_scalar_binary_elementwise_ad_to_native_llvm_jit()`, `make_scalar_binary_elementwise_native_llvm_jit_lowering_rule()`

### `scpn_quantum_control.compiler.mlir_symmetric_native_compilation`

Native LLVM/JIT autodiff compilation for symmetric 2x2 primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_symmetric_native_compilation.py) · Public symbols: **6**

**Functions:** `compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit()`, `make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule()`, `make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform()`, `compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit()`, `make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule()`, `make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform()`

### `scpn_quantum_control.compiler.mlir_transform_plan_assembly`

Compiler-AD transform-plan assembly and deterministic MLIR interchange.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_transform_plan_assembly.py) · Public symbols: **2**

**Functions:** `build_compiler_ad_transform_plan()`, `compile_compiler_ad_transform_plan_to_mlir()`

### `scpn_quantum_control.compiler.mlir_vector_native_compilation`

Native LLVM/JIT autodiff compilation for vector primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_vector_native_compilation.py) · Public symbols: **6**

**Functions:** `compile_vector_dot_ad_to_native_llvm_jit()`, `make_vector_dot_native_llvm_jit_lowering_rule()`, `make_vector_dot_native_llvm_jit_primitive_transform()`, `compile_vector_squared_norm_ad_to_native_llvm_jit()`, `make_vector_squared_norm_native_llvm_jit_lowering_rule()`, `make_vector_squared_norm_native_llvm_jit_primitive_transform()`

### `scpn_quantum_control.compiler.mlir_whole_program_emitter`

LLVM IR emitter engine for whole-program autodiff native lowering.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_whole_program_emitter.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.compiler.mlir_whole_program_native`

Native and MLIR lowering of whole-program autodiff traces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_whole_program_native.py) · Public symbols: **11**

**Classes:** `ExecutableWholeProgramADBatchResult`, `WholeProgramADNativeLoweringReport`, `NativeWholeProgramADKernel`, `ExecutableWholeProgramADKernel`

**Functions:** `compile_whole_program_ad_trace_to_mlir()`, `compile_whole_program_ad_trace_to_executable()`, `compile_whole_program_ad_trace_to_native_llvm_jit()`, `native_whole_program_ad_linalg_support()`, `analyse_whole_program_ad_native_lowering()`, `native_whole_program_ad_compile_cache_stats()`, `clear_native_whole_program_ad_compile_cache()`

### `scpn_quantum_control.compiler.mlir_workload_compilation`

Kuramoto and custom-rule executable MLIR compilation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/mlir_workload_compilation.py) · Public symbols: **5**

**Functions:** `compile_kuramoto_to_mlir()`, `compile_custom_derivative_rule_to_executable()`, `compile_registered_primitive_to_executable()`, `make_program_ad_linalg_matrix_power_executable_lowering_rule()`, `make_program_ad_linalg_multi_dot_executable_lowering_rule()`

### `scpn_quantum_control.compiler.promotion_batch`

Non-promotional compiler evidence batch assembly.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler/promotion_batch.py) · Public symbols: **4**

**Classes:** `CompilerPromotionBatchEvidenceFile`, `CompilerPromotionBatch`

**Functions:** `build_compiler_promotion_batch()`, `render_compiler_promotion_batch_markdown()`

## `control`

### `scpn_quantum_control.control.adaptive_branching`

S8 mid-circuit adaptive branching readiness model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/adaptive_branching.py) · Public symbols: **9**

**Classes:** `AdaptiveBranchingConfig`, `AdaptiveBranchDecision`, `AdaptiveBranchingReadiness`

**Functions:** `classify_branch_state()`, `build_adaptive_branch_table()`, `required_s8_dynamic_features()`, `estimate_branching_readiness()`, `s8_adaptive_branching_payload()`, `s8_adaptive_branching_markdown()`

### `scpn_quantum_control.control.closed_loop_analysis`

Control-theoretic analysis of the measurement-feedback synchronisation loop.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/closed_loop_analysis.py) · Public symbols: **14**

**Classes:** `ResponseClass`, `ExecutionMode`, `ControlPerformance`, `ClosedLoopExecutionPolicy`, `ClosedLoopExecutionDecision`, `ClosedLoopControlEvidence`, `ClosedLoopLatencyBudget`, `ClosedLoopLatencyReport`, `ClosedLoopPublicationPackage`

**Functions:** `evaluate_closed_loop_policy()`, `analyse_closed_loop_response()`, `run_closed_loop_control()`, `measure_closed_loop_latency_budget()`, `build_closed_loop_publication_package()`

### `scpn_quantum_control.control.frc_pulsed_qaoa`

QAOA and classical optimisation of the FRC pulsed-shot scheduling cost.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/frc_pulsed_qaoa.py) · Public symbols: **5**

**Classes:** `FRCScheduleResult`

**Functions:** `enumerate_costs()`, `optimal_schedule()`, `classical_sqp_schedule()`, `solve_frc_pulsed_qaoa()`

### `scpn_quantum_control.control.hardware_topological_optimizer`

Hardware-in-the-Loop Topological Feedback.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/hardware_topological_optimizer.py) · Public symbols: **1**

**Classes:** `HardwareTopologicalOptimizer`

### `scpn_quantum_control.control.q_disruption`

Quantum disruption classifier.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/q_disruption.py) · Public symbols: **1**

**Classes:** `QuantumDisruptionClassifier`

### `scpn_quantum_control.control.q_disruption_iter`

ITER-specific disruption classifier with 11 physics-based features.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/q_disruption_iter.py) · Public symbols: **7**

**Classes:** `ITERFeatureSpec`, `DisruptionBenchmark`

**Functions:** `normalize_iter_features()`, `scpn_control_bridge_dependency_contract()`, `validate_scpn_control_bridge_dependency_contract()`, `generate_synthetic_iter_data()`, `from_fusion_core_shot()`

### `scpn_quantum_control.control.qaoa_mpc`

QAOA for MPC trajectory optimization.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/qaoa_mpc.py) · Public symbols: **1**

**Classes:** `QAOA_MPC`

### `scpn_quantum_control.control.qaoa_pulsed_cost`

Control-grade FRC pulsed-shot scheduling cost for QAOA-MPC.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/qaoa_pulsed_cost.py) · Public symbols: **4**

**Classes:** `FRCQAOAObjective`, `FRCPlasmaSurrogate`

**Functions:** `decode_schedule_to_field()`, `frc_pulsed_shot_cost()`

### `scpn_quantum_control.control.qpetri`

Quantum Petri nets with superposition-token dynamics for control systems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/qpetri.py) · Public symbols: **3**

**Classes:** `QuantumPetriStepReport`, `QuantumPetriCampaignReport`, `QuantumPetriNet`

### `scpn_quantum_control.control.realtime_feedback`

Closed-loop Kuramoto-XY synchronisation feedback.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/realtime_feedback.py) · Public symbols: **6**

**Classes:** `RealtimeFeedbackConfig`, `FeedbackStep`, `RealtimeSyncFeedbackController`

**Functions:** `build_monitored_feedback_circuit()`, `build_open_loop_feedback_control_circuit()`, `feedback_policy_numpy()`

### `scpn_quantum_control.control.realtime_runtime`

Deadline-aware realtime control runtime.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/realtime_runtime.py) · Public symbols: **15**

**Classes:** `RealtimeClock`, `MonotonicRealtimeClock`, `VirtualRealtimeClock`, `RealtimeRuntimeConfig`, `RealtimeTickRecord`, `RealtimeRunResult`, `RealtimeSLAConfig`, `RealtimeSLAReport`, `CycleSample`, `SubMicrosecondReport`, `SubMicrosecondTracker`

**Functions:** `run_realtime_control_loop()`, `evaluate_realtime_sla()`, `enforce_realtime_sla()`, `summarise_cycle_samples()`

### `scpn_quantum_control.control.structured_ansatz`

Structured Kuramoto-XY ansatz construction helpers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/structured_ansatz.py) · Public symbols: **1**

**Classes:** `StructuredAnsatz`

### `scpn_quantum_control.control.topological_optimizer`

Topological Quantum Reinforcement Learning / Optimizer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/topological_optimizer.py) · Public symbols: **1**

**Classes:** `TopologicalCouplingOptimizer`

### `scpn_quantum_control.control.vqls_gs`

Residual-certified VQLS surface for a bounded Grad-Shafranov proxy.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control/vqls_gs.py) · Public symbols: **2**

**Classes:** `VQLSGradShafranovResult`, `VQLS_GradShafranov`

## `cosimulation`

### `scpn_quantum_control.cosimulation.knm_partition`

Split a K_nm coupling network into a quantum-strong core and classical bath.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/cosimulation/knm_partition.py) · Public symbols: **3**

**Classes:** `ConservationReport`, `KnmPartition`

**Functions:** `partition_knm()`

### `scpn_quantum_control.cosimulation.quantum_classical`

Mean-field co-simulation of a quantum-strong core inside a classical bath.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/cosimulation/quantum_classical.py) · Public symbols: **2**

**Classes:** `CoSimulationResult`

**Functions:** `cosimulate()`

## `crypto`

### `scpn_quantum_control.crypto.entanglement_qkd`

SCPN-QKD: Topology-authenticated quantum key distribution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/entanglement_qkd.py) · Public symbols: **3**

**Functions:** `scpn_qkd_protocol()`, `correlator_matrix()`, `bell_inequality_test()`

### `scpn_quantum_control.crypto.hierarchical_keys`

SCPN layer hierarchy to key derivation tree.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/hierarchical_keys.py) · Public symbols: **9**

**Functions:** `derive_master_key()`, `derive_layer_key()`, `key_hierarchy()`, `verify_key_chain()`, `evolve_key_phases()`, `rotating_key_schedule()`, `group_key()`, `hmac_verify_key()`, `hmac_sign()`

### `scpn_quantum_control.crypto.knm_key`

K_nm coupling matrix to key material pipeline.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/knm_key.py) · Public symbols: **4**

**Functions:** `prepare_key_state()`, `extract_raw_key()`, `estimate_qber()`, `privacy_amplification()`

### `scpn_quantum_control.crypto.ml_dsa`

ML-DSA-65 module-lattice digital signatures, implemented from FIPS 204.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/ml_dsa.py) · Public symbols: **6**

**Classes:** `MLDSAKeyPair`

**Functions:** `ntt()`, `intt()`, `key_gen()`, `sign()`, `verify()`

### `scpn_quantum_control.crypto.ml_dsa_seal`

Post-quantum signing back-end for the studio honesty seal (WS-1).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/ml_dsa_seal.py) · Public symbols: **2**

**Classes:** `MLDSAVerifier`, `MLDSASigner`

### `scpn_quantum_control.crypto.noise_analysis`

Security analysis under noise and eavesdropping.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/noise_analysis.py) · Public symbols: **6**

**Functions:** `depolarizing_channel()`, `amplitude_damping_single()`, `noisy_concurrence()`, `intercept_resend_qber()`, `devetak_winter_rate()`, `security_analysis()`

### `scpn_quantum_control.crypto.percolation`

Entanglement percolation analysis on K_nm coupling graph.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/percolation.py) · Public symbols: **7**

**Functions:** `concurrence_map()`, `percolation_threshold()`, `active_channel_graph()`, `key_rate_per_channel()`, `robustness_random_removal()`, `robustness_targeted_removal()`, `best_entanglement_path()`

### `scpn_quantum_control.crypto.pqc_trigger`

Post-quantum (FIPS 204 ML-DSA-65) signer for high-voltage trigger commands.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/pqc_trigger.py) · Public symbols: **4**

**Classes:** `PublicKey`, `PrivateKey`, `Signature`, `PqcTriggerSigner`

### `scpn_quantum_control.crypto.topology_auth`

Spectral fingerprint authentication for K_nm topology.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/crypto/topology_auth.py) · Public symbols: **11**

**Functions:** `spectral_fingerprint()`, `normalized_laplacian_fingerprint()`, `verify_fingerprint()`, `topology_distance()`, `topology_commitment()`, `verify_commitment()`, `challenge_response_prove()`, `challenge_response_verify()`, `fingerprint_noise_tolerance()`, `row_hash_fingerprint()`, `verify_row_hash()`

## `deployment`

### `scpn_quantum_control.deployment.cloud_native`

Deterministic cloud-native manifest generation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/deployment/cloud_native.py) · Public symbols: **4**

**Classes:** `ContainerResources`, `CloudDeploymentSpec`, `CloudManifestBundle`

**Functions:** `generate_cloud_manifests()`

## `dla_parity`

### `scpn_quantum_control.dla_parity.baselines`

Classical (noiseless) leakage reference for the DLA-parity protocol.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_parity/baselines.py) · Public symbols: **4**

**Classes:** `ClassicalLeakagePoint`, `ClassicalLeakageReference`

**Functions:** `available_baselines()`, `compute_classical_leakage_reference()`

### `scpn_quantum_control.dla_parity.dataset`

DLA-parity dataset loader.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_parity/dataset.py) · Public symbols: **2**

**Classes:** `DatasetIntegrityError`

**Functions:** `load_dla_parity_dataset()`

### `scpn_quantum_control.dla_parity.reproduce`

Statistical re-computation of the DLA-parity published numbers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_parity/reproduce.py) · Public symbols: **6**

**Classes:** `ReproductionTolerance`, `FisherResult`, `ReproductionResult`

**Functions:** `recompute_parity_leakage()`, `compute_depth_summaries()`, `reproduce_statistics()`

### `scpn_quantum_control.dla_parity.schema`

Typed dataclasses for the DLA-parity dataset.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_parity/schema.py) · Public symbols: **5**

**Classes:** `DlaParityCircuitMeta`, `DlaParityCircuit`, `DlaParityRun`, `DlaParityDataset`, `StatisticalSummary`

## `dla_topology_control`

### `scpn_quantum_control.dla_topology_control.evidence`

Deterministic evidence custody for DLA/topology constrained control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_topology_control/evidence.py) · Public symbols: **4**

**Classes:** `DlaTopologyControlEvidence`

**Functions:** `build_dla_topology_control_evidence()`, `render_dla_topology_control_markdown()`, `write_dla_topology_control_evidence()`

### `scpn_quantum_control.dla_topology_control.objectives`

Analytic synthetic objective inside a fixed DLA-parity sector.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_topology_control/objectives.py) · Public symbols: **2**

**Classes:** `ParityProtectedObjectiveEvaluation`, `ParityProtectedQuadraticObjective`

### `scpn_quantum_control.dla_topology_control.optimizer`

Deterministic projected-gradient loop for a synthetic parity-sector task.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_topology_control/optimizer.py) · Public symbols: **4**

**Classes:** `ProjectedGradientConfig`, `ProjectedGradientStep`, `ParityProjectedOptimisationTrace`

**Functions:** `optimise_parity_protected_state()`

### `scpn_quantum_control.dla_topology_control.parity`

Linear parity-sector projection with exact JVP, VJP, and leakage gradient.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_topology_control/parity.py) · Public symbols: **2**

**Classes:** `ParityLeakageEvaluation`, `ParitySectorProjector`

### `scpn_quantum_control.dla_topology_control.projection`

Fail-closed JVP/VJP contracts around the existing topology ledger.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_topology_control/projection.py) · Public symbols: **4**

**Classes:** `TopologyProjectionDifferential`

**Functions:** `topology_projection_support()`, `topology_projection_jvp()`, `topology_projection_vjp()`

### `scpn_quantum_control.dla_topology_control.schema`

Immutable support contracts for DLA/topology-constrained control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_topology_control/schema.py) · Public symbols: **5**

**Classes:** `DifferentiabilityKind`, `ParitySector`, `UnsupportedDifferentiableConstraintError`, `ConstraintSupportRow`, `DifferentiabilityReport`

## `entropy`

### `scpn_quantum_control.entropy.fips_140_2`

FIPS 140-2 Annex C power-up randomness tests on a 20 000-bit sample.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/entropy/fips_140_2.py) · Public symbols: **3**

**Classes:** `FipsHealthReport`

**Functions:** `fips_140_2_tests()`, `enforce_fips_140_2()`

### `scpn_quantum_control.entropy.nist_sp800_22`

NIST SP 800-22 Revision 1a statistical tests for (pseudo)random bit streams.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/entropy/nist_sp800_22.py) · Public symbols: **18**

**Classes:** `NistTestResult`

**Functions:** `as_bits()`, `frequency_test()`, `block_frequency_test()`, `runs_test()`, `longest_run_of_ones_test()`, `dft_spectral_test()`, `serial_test()`, `approximate_entropy_test()`, `cumulative_sums_test()`, `binary_matrix_rank_test()`, `non_overlapping_template_test()`, `overlapping_template_test()`, `maurers_universal_test()`, `berlekamp_massey()`, `linear_complexity_test()`, `random_excursions_test()`, `random_excursions_variant_test()`

### `scpn_quantum_control.entropy.qrng_stream`

Production streaming quantum random-number generator.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/entropy/qrng_stream.py) · Public symbols: **2**

**Classes:** `EntropyHealthReport`, `QRNGStream`

### `scpn_quantum_control.entropy.quantum_source`

Quantum measurement entropy sources for the QRNG streaming harness.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/entropy/quantum_source.py) · Public symbols: **3**

**Classes:** `EntropyBackend`, `AerQuantumEntropySource`

**Functions:** `von_neumann_debias()`

## `fep`

### `scpn_quantum_control.fep.predictive_coding`

Predictive coding across SCPN layers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/fep/predictive_coding.py) · Public symbols: **3**

**Classes:** `PredictiveCodingResult`

**Functions:** `hierarchical_prediction_error()`, `predictive_coding_step()`

### `scpn_quantum_control.fep.variational_free_energy`

Variational free energy computation for the SCPN.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/fep/variational_free_energy.py) · Public symbols: **5**

**Classes:** `FreeEnergyResult`

**Functions:** `kl_divergence_gaussian()`, `variational_free_energy()`, `evidence_lower_bound()`, `free_energy_gradient()`

## `forecasting`

### `scpn_quantum_control.forecasting.kuramoto_neural_operator`

Backward-compatible re-export shim for the relocated Kuramoto neural operator.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/kuramoto_neural_operator.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.forecasting.multimodal_bridge`

Bounded multimodal-forecasting composition into active-sensing sensing and co-design proposals.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/multimodal_bridge.py) · Public symbols: **4**

**Classes:** `ForecastActiveSensingBridge`, `ForecastControllerInitialisation`

**Functions:** `plan_forecast_active_sensing()`, `forecast_to_controller_initialisation()`

### `scpn_quantum_control.forecasting.multimodal_forecaster`

Classical reference forecasting over immutable multimodal-forecasting multimodal batches.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/multimodal_forecaster.py) · Public symbols: **6**

**Classes:** `MultimodalPointForecast`, `DomainForecastAccuracy`, `ForecastAccuracyCertificate`, `MultimodalRidgeForecaster`

**Functions:** `fit_multimodal_ridge_forecaster()`, `evaluate_point_forecast()`

### `scpn_quantum_control.forecasting.multimodal_report`

Deterministic JSON and Markdown evidence for the bounded multimodal-forecasting product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/multimodal_report.py) · Public symbols: **4**

**Classes:** `MultimodalSupportRow`, `MultimodalForecastingEvidence`

**Functions:** `render_multimodal_forecasting_markdown()`, `write_multimodal_forecasting_evidence()`

### `scpn_quantum_control.forecasting.multimodal_schema`

Immutable multimodal observation custody for bounded synthetic forecasts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/multimodal_schema.py) · Public symbols: **3**

**Classes:** `SyntheticDomainTag`, `MultimodalObservationBatch`

**Functions:** `assert_disjoint_batches()`

### `scpn_quantum_control.forecasting.neural_operator_advantage`

Evaluate whether the Kuramoto neural-operator surrogate beats direct simulation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/neural_operator_advantage.py) · Public symbols: **3**

**Classes:** `HeldOutFidelity`, `NeuralOperatorAdvantage`

**Functions:** `evaluate_neural_operator_advantage()`

### `scpn_quantum_control.forecasting.neural_operator_cost_model`

Host-independent operation-count model for the Kuramoto neural-operator surrogate.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/neural_operator_cost_model.py) · Public symbols: **9**

**Classes:** `SurrogateCostModel`

**Functions:** `rk4_right_hand_side_evaluations()`, `networked_force_flops()`, `rk4_step_flops()`, `direct_simulation_flops()`, `deeponet_forward_flops()`, `training_flops()`, `amortised_break_even_queries()`, `build_cost_model()`

### `scpn_quantum_control.forecasting.partial_observation`

Observed-phase and exact Kuramoto-residual scoring for multimodal-forecasting forecasts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/partial_observation.py) · Public symbols: **5**

**Classes:** `PartialObservationWeights`, `PartialObservationScore`, `PartialObservationBatchCertificate`

**Functions:** `evaluate_partial_observation_objective()`, `evaluate_partial_observation_batch()`

### `scpn_quantum_control.forecasting.real_data_sync`

Held-out synchronisation forecasting on observed or source-backed traces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/real_data_sync.py) · Public symbols: **7**

**Classes:** `SynchronisationForecastDataset`, `ForecastModelRun`, `SynchronisationForecastBenchmarkResult`

**Functions:** `load_hardware_kuramoto_4osc_trace()`, `load_ieee5bus_sync_forecast_case()`, `run_real_data_sync_forecast_benchmark()`, `run_real_data_sync_forecast_suite()`

### `scpn_quantum_control.forecasting.synthetic_multimodal`

Deterministic simulation-only datasets for multimodal-forecasting forecast certificates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/synthetic_multimodal.py) · Public symbols: **3**

**Classes:** `SyntheticMultimodalConfig`, `SyntheticMultimodalDataset`

**Functions:** `generate_synthetic_multimodal_dataset()`

### `scpn_quantum_control.forecasting.uncertainty`

Split sample-level residual intervals for independent multimodal-forecasting trajectories.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/forecasting/uncertainty.py) · Public symbols: **7**

**Classes:** `ResidualIntervalCalibrator`, `MultimodalIntervalForecast`, `DomainIntervalCoverage`, `IntervalCoverageCertificate`

**Functions:** `fit_residual_interval_calibrator()`, `apply_residual_interval()`, `certify_interval_coverage()`

## `gauge`

### `scpn_quantum_control.gauge.cft_analysis`

CFT central charge extraction at the XY critical point.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gauge/cft_analysis.py) · Public symbols: **4**

**Classes:** `CFTResult`

**Functions:** `find_critical_coupling()`, `extract_central_charge()`, `cft_analysis()`

### `scpn_quantum_control.gauge.confinement`

Confinement-deconfinement transition in the U(1) Kuramoto gauge theory.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gauge/confinement.py) · Public symbols: **4**

**Classes:** `ConfinementResult`

**Functions:** `extract_string_tension()`, `confinement_analysis()`, `confinement_vs_coupling()`

### `scpn_quantum_control.gauge.lattice_crosscheck`

Joint confinement report: quantum Wilson loops vs classical U(1) lattice MC.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gauge/lattice_crosscheck.py) · Public symbols: **2**

**Classes:** `GaugeLatticeCrosscheck`

**Functions:** `crosscheck_confinement_on_lattice()`

### `scpn_quantum_control.gauge.universality`

BKT/noisy-Kuramoto universality class analysis.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gauge/universality.py) · Public symbols: **5**

**Classes:** `UniversalityResult`

**Functions:** `correlation_vs_distance()`, `fit_correlation_exponent()`, `check_nelson_kosterlitz()`, `universality_analysis()`

### `scpn_quantum_control.gauge.vortex_detector`

Vortex density measurement for the Kuramoto-XY quantum model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gauge/vortex_detector.py) · Public symbols: **4**

**Classes:** `VortexResult`

**Functions:** `plaquette_vorticity()`, `measure_vortex_density()`, `vortex_density_vs_coupling()`

### `scpn_quantum_control.gauge.wilson_loop`

U(1) Wilson loop measurement on the Kuramoto-XY coupling graph.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gauge/wilson_loop.py) · Public symbols: **3**

**Classes:** `WilsonLoopResult`

**Functions:** `wilson_loop_expectation()`, `compute_wilson_loops()`

## `hardware`

### `scpn_quantum_control.hardware._count_integrity`

Shared strict count coercion utilities for hardware adapters.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/_count_integrity.py) · Public symbols: **6**

**Functions:** `strict_non_negative_count()`, `strict_integer_value()`, `strict_binary_bitstring_key()`, `strict_fixed_width_bitstring_key()`, `strict_provider_job_id()`, `strict_shot_conservation()`

### `scpn_quantum_control.hardware._experiment_helpers`

Shared helper functions used by experiment sub-modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/_experiment_helpers.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.hardware.aggregators`

First-class aggregator/provider route matrix for the hardware HAL.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/aggregators.py) · Public symbols: **5**

**Classes:** `AggregatorProviderRoute`, `ResolvedAggregatorProviderRoute`

**Functions:** `built_in_aggregator_provider_routes()`, `aggregator_provider_routes_for()`, `resolve_aggregator_provider_route()`

### `scpn_quantum_control.hardware.analog_kuramoto`

Native analog Kuramoto backend interface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/analog_kuramoto.py) · Public symbols: **15**

**Classes:** `AnalogKuramotoPlatform`, `AnalogProviderTarget`, `AnalogCouplingTerm`, `AnalogDriveTerm`, `AnalogFeedbackTerm`, `AnalogKuramotoProgram`, `ProviderAnalogPayload`, `ProviderAnalogExecutionPlan`, `AnalogBackendCapabilities`, `AnalogKuramotoBackendProtocol`, `AnalogKuramotoBackend`

**Functions:** `compile_analog_kuramoto()`, `analog_kuramoto_factory()`, `export_provider_payload()`, `prepare_provider_execution_plan()`

### `scpn_quantum_control.hardware.analog_native_readiness`

No-submit S10 analog-native Kuramoto readiness model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/analog_native_readiness.py) · Public symbols: **7**

**Classes:** `AnalogNativeReadinessConfig`, `AnalogNativePrimitiveComparison`, `AnalogProviderReadinessRow`

**Functions:** `compare_native_to_digital_primitives()`, `provider_readiness_rows()`, `analog_native_payload()`, `analog_native_markdown()`

### `scpn_quantum_control.hardware.async_runner`

Concurrent IBM job submission via asyncio.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/async_runner.py) · Public symbols: **2**

**Classes:** `AsyncJobHandle`, `AsyncHardwareRunner`

### `scpn_quantum_control.hardware.backends`

Plugin / backend extension API.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/backends.py) · Public symbols: **22**

**Classes:** `BackendProtocol`, `QuantumBackendDescriptor`, `BackendRegistrationError`, `BackendRegistry`

**Functions:** `get_registry()`, `register_backend()`, `unregister_backend()`, `get_backend()`, `describe_backend()`, `discover_backends()`, `list_backends()`, `list_quantum_backends()`, `describe_hal_backend_profile()`, `list_hal_backend_descriptors()`, `qiskit_ibm_factory()`, `qiskit_aer_factory()`, `cirq_factory()`, `braket_factory()`, `pennylane_factory()`, `iqm_factory()`, `analog_kuramoto_factory()`, `hybrid_digital_analog_factory()`

### `scpn_quantum_control.hardware.circuit_cutting`

Circuit cutting for scaling to 32-64 oscillators.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/circuit_cutting.py) · Public symbols: **5**

**Classes:** `CircuitCuttingPlan`

**Functions:** `count_inter_partition_couplings()`, `optimal_partition()`, `circuit_cutting_plan()`, `scaling_analysis()`

### `scpn_quantum_control.hardware.circuit_export`

Export Kuramoto-XY Trotter circuits to multiple quantum platforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/circuit_export.py) · Public symbols: **5**

**Functions:** `build_trotter_circuit()`, `to_qasm3()`, `to_cirq()`, `to_quil()`, `export_all()`

### `scpn_quantum_control.hardware.cirq_adapter`

Google Cirq backend adapter.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/cirq_adapter.py) · Public symbols: **3**

**Classes:** `CirqResult`, `CirqRunner`

**Functions:** `is_cirq_available()`

### `scpn_quantum_control.hardware.classical`

Classical reference computations for hardware experiment comparison.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/classical.py) · Public symbols: **5**

**Functions:** `classical_kuramoto_reference()`, `classical_exact_diag()`, `classical_exact_evolution()`, `bloch_vectors_from_json()`, `classical_brute_mpc()`

### `scpn_quantum_control.hardware.cutting_runner`

Circuit cutting runner: execute partitioned simulations for N > 16.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/cutting_runner.py) · Public symbols: **2**

**Classes:** `CuttingRunResult`

**Functions:** `run_cutting_simulation()`

### `scpn_quantum_control.hardware.dynq_layout_pass`

Qiskit ``AnalysisPass`` adapter for the DynQ qubit mapper.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/dynq_layout_pass.py) · Public symbols: **2**

**Classes:** `DynQLayoutPass`

**Functions:** `calibration_from_target()`

### `scpn_quantum_control.hardware.error_aware_chain`

Error-aware selection of 1-D nearest-neighbour qubit chains.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/error_aware_chain.py) · Public symbols: **3**

**Classes:** `ChainSelection`

**Functions:** `select_error_aware_chain()`, `longest_error_aware_chain()`

### `scpn_quantum_control.hardware.experiment_control`

QAOA-MPC, UPDE snapshot, Bell test, correlator, and QKD experiments.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/experiment_control.py) · Public symbols: **5**

**Functions:** `qaoa_mpc_4_experiment()`, `upde_16_snapshot_experiment()`, `bell_test_4q_experiment()`, `correlator_4q_experiment()`, `qkd_qber_4q_experiment()`

### `scpn_quantum_control.hardware.experiment_dynamics`

Kuramoto evolution experiments on quantum hardware.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/experiment_dynamics.py) · Public symbols: **4**

**Functions:** `kuramoto_4osc_experiment()`, `kuramoto_8osc_experiment()`, `kuramoto_4osc_trotter2_experiment()`, `sync_threshold_experiment()`

### `scpn_quantum_control.hardware.experiment_mitigation`

ZNE, dynamical decoupling, and noise characterisation experiments.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/experiment_mitigation.py) · Public symbols: **6**

**Functions:** `kuramoto_4osc_zne_experiment()`, `noise_baseline_experiment()`, `kuramoto_8osc_zne_experiment()`, `upde_16_dd_experiment()`, `zne_higher_order_experiment()`, `decoherence_scaling_experiment()`

### `scpn_quantum_control.hardware.experiment_vqe`

VQE and ansatz experiments on quantum hardware.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/experiment_vqe.py) · Public symbols: **5**

**Functions:** `vqe_4q_experiment()`, `vqe_8q_experiment()`, `vqe_8q_hardware_experiment()`, `ansatz_comparison_hw_experiment()`, `vqe_landscape_experiment()`

### `scpn_quantum_control.hardware.experiments`

Concrete hardware experiments for IBM Quantum.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/experiments.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.hardware.fast_classical`

High-performance sparse statevector engine.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/fast_classical.py) · Public symbols: **1**

**Functions:** `fast_sparse_evolution()`

### `scpn_quantum_control.hardware.feedback_capability_probe`

No-submit capability probes for S1 feedback target selection.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/feedback_capability_probe.py) · Public symbols: **5**

**Classes:** `BackendCapabilitySnapshot`, `FeedbackCapabilityDecision`

**Functions:** `required_s1_dynamic_features()`, `assess_feedback_backend_capability()`, `assess_feedback_backend_fleet()`

### `scpn_quantum_control.hardware.feedback_dryrun`

No-submit provider dry-run payloads for S1 feedback jobs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/feedback_dryrun.py) · Public symbols: **5**

**Classes:** `FeedbackDryRunPayload`

**Functions:** `build_ibm_runtime_dry_run()`, `build_openqasm3_gate_dry_run()`, `build_analog_native_review_payload()`, `build_s1_feedback_dry_run_bundle()`

### `scpn_quantum_control.hardware.feedback_hardware_scheduler`

Approval-gated hardware scheduler boundary for S1 feedback jobs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/feedback_hardware_scheduler.py) · Public symbols: **4**

**Classes:** `HardwareApprovalRecord`, `HardwareSubmissionRecord`, `ApprovalGatedFeedbackHardwareScheduler`

**Functions:** `hash_package_manifest()`

### `scpn_quantum_control.hardware.feedback_loop`

Cross-shot hybrid classical-quantum feedback orchestration.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/feedback_loop.py) · Public symbols: **10**

**Classes:** `FeedbackLoopConfig`, `FeedbackLoopLatencySLA`, `FeedbackCommand`, `FeedbackResult`, `FeedbackStepRecord`, `FeedbackScheduler`, `FeedbackObserver`, `FeedbackRunner`, `RealtimeControllerScheduler`, `ProportionalMetricObserver`

### `scpn_quantum_control.hardware.feedback_provider_metadata`

Provider metadata adapters for S1 no-submit capability probes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/feedback_provider_metadata.py) · Public symbols: **2**

**Functions:** `snapshot_from_generic_metadata()`, `snapshot_from_qiskit_backend()`

### `scpn_quantum_control.hardware.feedback_submission`

Provider-neutral S1 feedback submission-readiness packaging.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/feedback_submission.py) · Public symbols: **9**

**Classes:** `FeedbackPlatformCapability`, `FeedbackBudgetEstimate`, `FeedbackCircuitSummary`, `PlatformReadiness`, `FeedbackSubmissionPackage`

**Functions:** `default_s1_platforms()`, `build_s1_feedback_submission_package()`, `summarise_feedback_circuit()`, `assess_platform_readiness()`

### `scpn_quantum_control.hardware.gpu_accel`

GPU acceleration via cupy for matrix-heavy quantum operations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/gpu_accel.py) · Public symbols: **7**

**Functions:** `is_gpu_available()`, `gpu_device_name()`, `eigvalsh()`, `eigh()`, `expm()`, `matmul()`, `gpu_memory_free_mb()`

### `scpn_quantum_control.hardware.hal`

Provider-neutral hardware abstraction layer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal.py) · Public symbols: **9**

**Classes:** `BackendCapabilities`, `BackendProfile`, `QuantumWorkload`, `QuantumJobRef`, `QuantumJobResult`, `QuantumBackend`, `LocalDeterministicSimulator`, `HardwareAbstractionLayer`

**Functions:** `built_in_backend_profiles()`

### `scpn_quantum_control.hardware.hal_azure`

Azure Quantum adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_azure.py) · Public symbols: **2**

**Classes:** `AzureQuantumHALAdapter`

**Functions:** `azure_openqasm3_to_workload()`

### `scpn_quantum_control.hardware.hal_braket`

Amazon Braket adapters for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_braket.py) · Public symbols: **3**

**Classes:** `BraketLocalHALAdapter`, `BraketAwsHALAdapter`

**Functions:** `braket_circuit_to_workload()`

### `scpn_quantum_control.hardware.hal_cirq`

Local Cirq simulator adapter for the provider-neutral HAL.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_cirq.py) · Public symbols: **2**

**Classes:** `CirqLocalHALAdapter`

**Functions:** `cirq_circuit_workload()`

### `scpn_quantum_control.hardware.hal_dwave`

Direct D-Wave Leap BQM adapter for the provider-neutral HAL.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_dwave.py) · Public symbols: **2**

**Classes:** `DWaveLeapHALAdapter`

**Functions:** `dwave_bqm_workload()`

### `scpn_quantum_control.hardware.hal_ionq`

Direct IonQ Cloud adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_ionq.py) · Public symbols: **2**

**Classes:** `IonQCloudHALAdapter`

**Functions:** `ionq_qis_workload()`

### `scpn_quantum_control.hardware.hal_iqm`

IQM Qiskit adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_iqm.py) · Public symbols: **2**

**Classes:** `IQMHALAdapter`

**Functions:** `iqm_qiskit_workload()`

### `scpn_quantum_control.hardware.hal_oqc`

Direct OQC QCAAS adapter for the provider-neutral HAL.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_oqc.py) · Public symbols: **2**

**Classes:** `OQCHALAdapter`

**Functions:** `oqc_openqasm3_workload()`

### `scpn_quantum_control.hardware.hal_pasqal`

Pasqal/Pulser adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_pasqal.py) · Public symbols: **2**

**Classes:** `PasqalPulserHALAdapter`

**Functions:** `pulser_sequence_workload()`

### `scpn_quantum_control.hardware.hal_pennylane`

PennyLane-backed adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_pennylane.py) · Public symbols: **2**

**Classes:** `PennyLaneDeviceHALAdapter`

**Functions:** `pennylane_gate_workload()`

### `scpn_quantum_control.hardware.hal_qbraid`

qBraid runtime adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_qbraid.py) · Public symbols: **2**

**Classes:** `QbraidRuntimeHALAdapter`

**Functions:** `qbraid_program_to_workload()`

### `scpn_quantum_control.hardware.hal_qiskit`

Qiskit-backed adapters for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_qiskit.py) · Public symbols: **4**

**Classes:** `QiskitAerHALAdapter`, `QiskitRuntimeHALAdapter`

**Functions:** `qiskit_circuit_to_workload()`, `qiskit_circuit_to_qasm3_workload()`

### `scpn_quantum_control.hardware.hal_quandela`

Direct Quandela/Perceval adapter for the provider-neutral HAL.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_quandela.py) · Public symbols: **2**

**Classes:** `QuandelaPercevalHALAdapter`

**Functions:** `quandela_perceval_workload()`

### `scpn_quantum_control.hardware.hal_quantinuum`

Quantinuum pytket adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_quantinuum.py) · Public symbols: **2**

**Classes:** `QuantinuumCloudHALAdapter`

**Functions:** `quantinuum_tket_workload()`

### `scpn_quantum_control.hardware.hal_quera_bloqade`

QuEra Bloqade adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_quera_bloqade.py) · Public symbols: **2**

**Classes:** `QuEraBloqadeHALAdapter`

**Functions:** `bloqade_ahs_workload()`

### `scpn_quantum_control.hardware.hal_rigetti`

Rigetti pyQuil adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_rigetti.py) · Public symbols: **2**

**Classes:** `RigettiQCSHALAdapter`

**Functions:** `rigetti_quil_workload()`

### `scpn_quantum_control.hardware.hal_strangeworks`

Strangeworks Compute adapter for :mod:`scpn_quantum_control.hardware.hal`.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hal_strangeworks.py) · Public symbols: **2**

**Classes:** `StrangeworksComputeHALAdapter`

**Functions:** `strangeworks_program_to_workload()`

### `scpn_quantum_control.hardware.hybrid_digital_analog`

Hybrid digital-analog execution plans for Kuramoto-XY workloads.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/hybrid_digital_analog.py) · Public symbols: **9**

**Classes:** `HybridRoute`, `HybridCouplingAssignment`, `HybridCouplingPartition`, `HybridDigitalAnalogProgram`, `HybridDigitalAnalogBackendProtocol`, `HybridDigitalAnalogBackend`

**Functions:** `compile_hybrid_digital_analog()`, `hybrid_digital_analog_factory()`, `partition_kuramoto_couplings()`

### `scpn_quantum_control.hardware.ibm_latency_probe`

Helpers for IBM Runtime latency telemetry extraction and normalisation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/ibm_latency_probe.py) · Public symbols: **4**

**Functions:** `iso_utc_now()`, `parse_timestamp()`, `extract_job_telemetry()`, `derive_timing_windows()`

### `scpn_quantum_control.hardware.iqm_backend`

IQM backend adapter for Qiskit-compatible circuit execution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/iqm_backend.py) · Public symbols: **5**

**Classes:** `IQMBackendConfig`, `IQMRunResult`, `IQMQuantumBackend`

**Functions:** `is_iqm_available()`, `iqm_factory()`

### `scpn_quantum_control.hardware.iqm_lattice_calibration`

IQM square-lattice calibration → Kuramoto layout-cost inputs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/iqm_lattice_calibration.py) · Public symbols: **5**

**Classes:** `LatticeCalibration`, `ChainRegion`

**Functions:** `lattice_calibration_from_backend()`, `enumerate_chain_regions()`, `best_chain_region()`

### `scpn_quantum_control.hardware.jax_accel`

JAX-accelerated exact dense quantum analysis.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/jax_accel.py) · Public symbols: **5**

**Functions:** `is_jax_available()`, `is_jax_gpu_available()`, `jax_device_name()`, `eigensolve_batch_jax()`, `entanglement_scan_jax()`

### `scpn_quantum_control.hardware.job_dossier`

Standard documentation schema for submission-ready hardware jobs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/job_dossier.py) · Public symbols: **2**

**Classes:** `HardwareJobDossier`

**Functions:** `build_s1_feedback_job_dossier()`

### `scpn_quantum_control.hardware.kuramoto_layout_cost`

Kuramoto-XY-aware discrete cost model for qubit-layout selection.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/kuramoto_layout_cost.py) · Public symbols: **6**

**Classes:** `DepthProvider`, `CostWeights`, `LayoutCost`

**Functions:** `dynq_mean_gate_fidelity()`, `routed_layout_depth()`, `kuramoto_layout_cost()`

### `scpn_quantum_control.hardware.kuramoto_layout_optimiser`

Discrete layout optimiser over the Kuramoto-XY-aware cost model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/kuramoto_layout_optimiser.py) · Public symbols: **3**

**Classes:** `LayoutSearchConfig`, `LayoutSearchResult`

**Functions:** `optimise_kuramoto_layout()`

### `scpn_quantum_control.hardware.kuramoto_layout_relaxation`

RESEARCH: Sinkhorn continuous relaxation of the Kuramoto layout search.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/kuramoto_layout_relaxation.py) · Public symbols: **6**

**Classes:** `SinkhornRelaxationConfig`, `RelaxationSearchResult`

**Functions:** `sinkhorn_normalise()`, `coupling_graph_distances()`, `swap_distance_surrogate()`, `relax_kuramoto_layout()`

### `scpn_quantum_control.hardware.noise_model`

Heron r2 noise model for realistic local simulation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/noise_model.py) · Public symbols: **1**

**Functions:** `heron_r2_noise_model()`

### `scpn_quantum_control.hardware.openpulse_control`

OpenPulse schedule construction and calibration workflow primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/openpulse_control.py) · Public symbols: **10**

**Classes:** `OpenPulseWaveform`, `OpenPulseInstruction`, `OpenPulseSchedule`, `RabiCalibrationPoint`, `OpenPulseCalibrationWorkflow`, `RabiPiCalibrationEstimate`

**Functions:** `compile_hypergeometric_openpulse_schedule()`, `build_rabi_amplitude_calibration_workflow()`, `estimate_rabi_pi_amplitude()`, `schedule_to_qiskit_pulse()`

### `scpn_quantum_control.hardware.pennylane_adapter`

PennyLane backend adapter for cross-platform quantum execution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/pennylane_adapter.py) · Public symbols: **3**

**Classes:** `PennyLaneResult`, `PennyLaneRunner`

**Functions:** `is_pennylane_available()`

### `scpn_quantum_control.hardware.plugin_registry`

Extensible plugin architecture for quantum backends.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/plugin_registry.py) · Public symbols: **1**

**Classes:** `PluginRegistry`

### `scpn_quantum_control.hardware.provenance`

Run-time provenance capture for hardware and simulator results.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provenance.py) · Public symbols: **1**

**Functions:** `capture_provenance()`

### `scpn_quantum_control.hardware.provider_capability_cloud_adapters`

No-submit metadata adapters for cloud and broker quantum provider routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_capability_cloud_adapters.py) · Public symbols: **6**

**Functions:** `snapshot_from_azure_target()`, `snapshot_from_braket_device()`, `snapshot_from_qiskit_runtime_backend()`, `snapshot_from_qbraid_device()`, `snapshot_from_strangeworks_backend()`, `normalize_calibration_timestamp()`

### `scpn_quantum_control.hardware.provider_capability_core`

Provider-neutral no-submit capability contracts and readiness decisions.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_capability_core.py) · Public symbols: **6**

**Classes:** `ProviderCapabilitySnapshot`, `ProviderCapabilityDecision`, `OpenPulseControlReadiness`

**Functions:** `build_openpulse_control_readiness()`, `probe_aggregator_provider_capability()`, `assess_provider_capability_snapshot()`

### `scpn_quantum_control.hardware.provider_capability_discovery`

No-submit provider metadata adapters and compatibility facade.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_capability_discovery.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.hardware.provider_capability_gate_adapters`

No-submit metadata adapters for direct gate-model provider routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_capability_gate_adapters.py) · Public symbols: **5**

**Functions:** `snapshot_from_ionq_backend()`, `snapshot_from_iqm_backend()`, `snapshot_from_oqc_target()`, `snapshot_from_quantinuum_backend()`, `snapshot_from_rigetti_qcs()`

### `scpn_quantum_control.hardware.provider_capability_normalization`

Provider-independent metadata access and normalization primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_capability_normalization.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.hardware.provider_capability_specialized_adapters`

No-submit metadata adapters for specialized quantum provider routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_capability_specialized_adapters.py) · Public symbols: **4**

**Functions:** `snapshot_from_dwave_solver()`, `snapshot_from_quera_bloqade()`, `snapshot_from_pasqal_target()`, `snapshot_from_quandela_processor()`

### `scpn_quantum_control.hardware.provider_smoke`

Metadata-only optional dependency smoke checks for HAL provider routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/provider_smoke.py) · Public symbols: **7**

**Classes:** `ProviderOptionalDependencyRow`, `AggregatorProviderOptionalDependencyRow`, `IsolatedProviderSmokeLane`

**Functions:** `provider_optional_dependency_matrix()`, `aggregator_provider_optional_dependency_matrix()`, `isolated_provider_smoke_lanes()`, `main()`

### `scpn_quantum_control.hardware.pulse_feasibility`

No-submit provider feasibility probes for S3 pulse schedules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/pulse_feasibility.py) · Public symbols: **7**

**Classes:** `PulseProviderSnapshot`, `PulseScheduleSummary`, `PulseFeasibilityDecision`

**Functions:** `summarise_pulse_schedule()`, `assess_pulse_provider_feasibility()`, `assess_pulse_provider_fleet()`, `pulse_snapshot_from_metadata()`

### `scpn_quantum_control.hardware.qasm_export`

OpenQASM 3 circuit export for cross-platform portability.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/qasm_export.py) · Public symbols: **4**

**Classes:** `QASMExportResult`

**Functions:** `export_trotter_qasm()`, `export_ansatz_qasm()`, `export_measurement_qasm()`

### `scpn_quantum_control.hardware.qcvv`

QCVV: Quantum Characterisation, Verification, and Validation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/qcvv.py) · Public symbols: **6**

**Classes:** `QCVVResult`

**Functions:** `state_fidelity()`, `mirror_circuit_fidelity()`, `cross_entropy_score()`, `simulate_xeb()`, `qcvv_certify()`

### `scpn_quantum_control.hardware.qiskit_compat`

Qiskit version compatibility layer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/qiskit_compat.py) · Public symbols: **7**

**Functions:** `qiskit_version()`, `qiskit_major()`, `get_pauli_evolution_gate()`, `get_lie_trotter()`, `get_statevector()`, `get_sparse_pauli_op()`, `check_qiskit_compatibility()`

### `scpn_quantum_control.hardware.qpu_result_pack_bridge`

Emit a ``studio.qpu-result-pack.v1`` unit from a live provider-neutral result.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/qpu_result_pack_bridge.py) · Public symbols: **3**

**Functions:** `raw_results_digest()`, `job_result_provenance()`, `qpu_result_pack_from_job()`

### `scpn_quantum_control.hardware.qubit_mapper`

DynQ-inspired qubit placement via quality-weighted community detection.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/qubit_mapper.py) · Public symbols: **6**

**Classes:** `ExecutionRegion`, `QubitMappingResult`

**Functions:** `build_calibration_graph()`, `detect_execution_regions()`, `select_best_region()`, `dynq_initial_layout()`

### `scpn_quantum_control.hardware.realtime_latency_scenarios`

Dedicated realtime-control latency scenarios independent from S1 scientific batches.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/realtime_latency_scenarios.py) · Public symbols: **4**

**Classes:** `RealtimeLatencyScenario`

**Functions:** `build_dynamic_feedback_circuit()`, `build_open_loop_reference_circuit()`, `default_realtime_latency_scenarios()`

### `scpn_quantum_control.hardware.runner`

IBM Quantum hardware runner.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/runner.py) · Public symbols: **2**

**Classes:** `JobResult`, `HardwareRunner`

### `scpn_quantum_control.hardware.s1_feedback_ibm`

S1 IBM paired-arm submission and raw-count conversion contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/s1_feedback_ibm.py) · Public symbols: **9**

**Classes:** `S1FeedbackArmCircuit`

**Functions:** `build_s1_feedback_arm_circuits()`, `build_s1_xy_observable_arm_circuits()`, `binary_phase_synchrony_from_counts()`, `pauli_expectation_from_counts()`, `raw_count_package_from_feedback_results()`, `raw_count_package_from_xy_observable_results()`, `build_s1_arm_command()`, `run_ibm_sampler_arm()`

### `scpn_quantum_control.hardware.trapped_ion`

Representative trapped-ion noise model for cross-platform benchmarking.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware/trapped_ion.py) · Public symbols: **2**

**Functions:** `trapped_ion_noise_model()`, `transpile_for_trapped_ion()`

## `identity`

### `scpn_quantum_control.identity.binding_spec`

Arcane Sapience identity binding spec: 6-layer, 18-oscillator Kuramoto topology.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity/binding_spec.py) · Public symbols: **4**

**Functions:** `build_identity_attractor()`, `solve_identity()`, `quantum_to_orchestrator_phases()`, `orchestrator_to_quantum_phases()`

### `scpn_quantum_control.identity.coherence_budget`

Coherence budget calculator for identity quantum circuits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity/coherence_budget.py) · Public symbols: **2**

**Functions:** `fidelity_at_depth()`, `coherence_budget()`

### `scpn_quantum_control.identity.entanglement_witness`

Entanglement witness for disposition pairs via CHSH inequality.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity/entanglement_witness.py) · Public symbols: **2**

**Functions:** `chsh_from_statevector()`, `disposition_entanglement_map()`

### `scpn_quantum_control.identity.ground_state`

Identity attractor basin via VQE ground state analysis.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity/ground_state.py) · Public symbols: **1**

**Classes:** `IdentityAttractor`

### `scpn_quantum_control.identity.identity_key`

Quantum identity fingerprint from coupling topology.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity/identity_key.py) · Public symbols: **4**

**Functions:** `identity_fingerprint()`, `identity_fingerprint_from_binding_spec()`, `verify_identity()`, `prove_identity()`

### `scpn_quantum_control.identity.robustness`

Adiabatic robustness certificate for identity binding.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity/robustness.py) · Public symbols: **4**

**Classes:** `RobustnessCertificate`

**Functions:** `compute_robustness_certificate()`, `perturbation_fidelity()`, `gap_vs_perturbation_scan()`

## `l16`

### `scpn_quantum_control.l16.director_contracts`

Immutable contracts for bounded L16 indicator evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/l16/director_contracts.py) · Public symbols: **4**

**Classes:** `L16ScenarioSpec`, `L16IndicatorCertificate`, `L16RouteEvidence`, `L16DirectorEvidence`

### `scpn_quantum_control.l16.director_evidence`

Digest-bound JSON and Markdown evidence for the bounded L16 director.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/l16/director_evidence.py) · Public symbols: **6**

**Functions:** `canonical_l16_json()`, `l16_evidence_payload()`, `render_l16_evidence_markdown()`, `validate_l16_evidence()`, `write_l16_evidence()`, `main()`

### `scpn_quantum_control.l16.director_product`

Policy-gated L16 indicators and conservative co-design safety routing.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/l16/director_product.py) · Public symbols: **7**

**Classes:** `L16DirectorPolicyError`

**Functions:** `frozen_l16_scenarios()`, `observer_inputs_from_l16()`, `run_l16_indicator_scenario()`, `run_l16_director_suite()`, `informative_l16_indicators()`, `l16_promotion_blockers()`

### `scpn_quantum_control.l16.quantum_director`

Quantum L16 indicator bundle for heuristic cybernetic routing.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/l16/quantum_director.py) · Public symbols: **5**

**Classes:** `L16Result`

**Functions:** `loschmidt_echo()`, `energy_variance()`, `fidelity_susceptibility()`, `compute_l16_lyapunov()`

## `mitigation`

### `scpn_quantum_control.mitigation.compound_mitigation`

Compound Error Mitigation: CPDR + Z2 Symmetry Verification.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/compound_mitigation.py) · Public symbols: **2**

**Classes:** `CompoundMitigationResult`

**Functions:** `compound_mitigate_pipeline()`

### `scpn_quantum_control.mitigation.cpdr`

Clifford Perturbation Data Regression (CPDR) error mitigation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/cpdr.py) · Public symbols: **7**

**Classes:** `CPDRResult`

**Functions:** `generate_training_circuits()`, `compute_ideal_values()`, `compute_noisy_values_from_counts()`, `fit_regression()`, `cpdr_mitigate()`, `cpdr_full_pipeline()`

### `scpn_quantum_control.mitigation.dd`

Dynamical decoupling sequences for idle-qubit decoherence suppression.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/dd.py) · Public symbols: **2**

**Classes:** `DDSequence`

**Functions:** `insert_dd_sequence()`

### `scpn_quantum_control.mitigation.mitiq_integration`

Mitiq integration for production-quality error mitigation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/mitiq_integration.py) · Public symbols: **3**

**Functions:** `is_mitiq_available()`, `zne_mitigated_expectation()`, `ddd_mitigated_expectation()`

### `scpn_quantum_control.mitigation.pec`

Probabilistic Error Cancellation for local depolarizing channels.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/pec.py) · Public symbols: **3**

**Classes:** `PECResult`

**Functions:** `pauli_twirl_decompose()`, `pec_sample()`

### `scpn_quantum_control.mitigation.readout_matrix`

Full-basis readout confusion-matrix mitigation utilities.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/readout_matrix.py) · Public symbols: **11**

**Classes:** `ReadoutConfusionMatrix`

**Functions:** `computational_basis_labels()`, `bitstring_index()`, `counts_to_probabilities()`, `build_readout_confusion_matrix()`, `mitigate_probabilities()`, `mitigate_counts()`, `probability_state_retention()`, `probability_parity_leakage()`, `probability_magnetisation_leakage()`, `probability_mean_magnetisation()`

### `scpn_quantum_control.mitigation.symmetry_decay`

GUESS: Guiding Extrapolations from Symmetry Decays.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/symmetry_decay.py) · Public symbols: **5**

**Classes:** `SymmetryDecayModel`, `GUESSResult`

**Functions:** `learn_symmetry_decay()`, `guess_extrapolate()`, `xy_magnetisation_ideal()`

### `scpn_quantum_control.mitigation.symmetry_sector_compiler`

Planning contract for symmetry- and sector-aware error mitigation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/symmetry_sector_compiler.py) · Public symbols: **3**

**Classes:** `SymmetrySectorProblem`, `SymmetrySectorPlan`

**Functions:** `plan_symmetry_sector_mitigation()`

### `scpn_quantum_control.mitigation.symmetry_sector_fixtures`

Deterministic fixtures for the symmetry-sector mitigation planner.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/symmetry_sector_fixtures.py) · Public symbols: **7**

**Functions:** `fixture_problems()`, `fixture_payload()`, `replay_fixture_rows()`, `normalised_json()`, `write_json()`, `fixture_markdown()`, `write_text()`

### `scpn_quantum_control.mitigation.symmetry_sector_replay`

Offline raw-count replay for symmetry-sector mitigation plans.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/symmetry_sector_replay.py) · Public symbols: **2**

**Classes:** `SymmetrySectorReplayResult`

**Functions:** `replay_symmetry_sector_counts()`

### `scpn_quantum_control.mitigation.symmetry_verification`

Z₂ parity symmetry verification for XY Hamiltonian error mitigation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/symmetry_verification.py) · Public symbols: **7**

**Classes:** `SymmetryVerificationResult`

**Functions:** `bitstring_parity()`, `initial_state_parity()`, `parity_postselect()`, `symmetry_expand()`, `parity_verified_expectation()`, `parity_verified_R()`

### `scpn_quantum_control.mitigation.zne`

Zero-Noise Extrapolation via global unitary folding.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/zne.py) · Public symbols: **3**

**Classes:** `ZNEResult`

**Functions:** `gate_fold_circuit()`, `zne_extrapolate()`

### `scpn_quantum_control.mitigation.zne_uncertainty`

Uncertainty propagation for zero-noise extrapolation (ZNE).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/mitigation/zne_uncertainty.py) · Public symbols: **2**

**Classes:** `ZNEUncertaintyResult`

**Functions:** `zne_extrapolate_with_uncertainty()`

## `ml_examples`

### `scpn_quantum_control.ml_examples.contracts`

Immutable contracts for bounded QNN/QGNN/QSNN convergence examples.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ml_examples/contracts.py) · Public symbols: **6**

**Classes:** `ModelFamily`, `FrameworkStatus`, `ConvergenceExampleSpec`, `ConvergenceCertificate`, `FrameworkEvidenceRow`, `ConvergenceSuiteEvidence`

### `scpn_quantum_control.ml_examples.evidence`

Digest-bound JSON and Markdown evidence for the convergence-example convergence suite.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ml_examples/evidence.py) · Public symbols: **6**

**Functions:** `canonical_json()`, `evidence_payload()`, `render_evidence_markdown()`, `write_ml_convergence_evidence()`, `validate_ml_convergence_evidence()`, `main()`

### `scpn_quantum_control.ml_examples.qgnn_convergence`

Frozen graph-regression convergence task over the existing bounded QGNN.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ml_examples/qgnn_convergence.py) · Public symbols: **3**

**Functions:** `qgnn_example_spec()`, `run_qgnn_convergence_example()`, `qgnn_framework_rows()`

### `scpn_quantum_control.ml_examples.qnn_convergence`

Frozen phase-QNN convergence task and real framework agreement rows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ml_examples/qnn_convergence.py) · Public symbols: **3**

**Functions:** `qnn_example_spec()`, `run_qnn_convergence_example()`, `run_qnn_framework_rows()`

### `scpn_quantum_control.ml_examples.qsnn_convergence`

Frozen synapse-angle task over the existing QSNN parameter-shift trainer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ml_examples/qsnn_convergence.py) · Public symbols: **3**

**Functions:** `qsnn_example_spec()`, `run_qsnn_convergence_example()`, `qsnn_framework_rows()`

### `scpn_quantum_control.ml_examples.suite`

Compose the three existing ML training surfaces into one evidence suite.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ml_examples/suite.py) · Public symbols: **1**

**Functions:** `run_ml_convergence_suite()`

## `pgbo`

### `scpn_quantum_control.pgbo.quantum_bridge`

Quantum PGBO: phase-geometry bridge operator.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/pgbo/quantum_bridge.py) · Public symbols: **2**

**Classes:** `PGBOResult`

**Functions:** `compute_pgbo_tensor()`

## `phase`

### `scpn_quantum_control.phase.adapt_vqe`

Adaptive layered VQE for the Kuramoto-XY ground state.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/adapt_vqe.py) · Public symbols: **2**

**Classes:** `ADAPTResult`

**Functions:** `adapt_vqe()`

### `scpn_quantum_control.phase.adiabatic_preparation`

Adiabatic state preparation for the synchronization ground state.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/adiabatic_preparation.py) · Public symbols: **3**

**Classes:** `AdiabaticResult`

**Functions:** `adiabatic_ramp()`, `adiabatic_time_scaling()`

### `scpn_quantum_control.phase.ancilla_lindblad`

Simulate open-system Kuramoto-XY dynamics with one ancilla qubit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/ancilla_lindblad.py) · Public symbols: **3**

**Classes:** `AncillaCircuitStats`

**Functions:** `build_ancilla_lindblad_circuit()`, `ancilla_circuit_stats()`

### `scpn_quantum_control.phase.ansatz_bench`

Compare K_nm-informed and generic variational ansatz families.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/ansatz_bench.py) · Public symbols: **3**

**Classes:** `AnsatzBenchmarkRow`

**Functions:** `benchmark_ansatz()`, `run_ansatz_benchmark()`

### `scpn_quantum_control.phase.ansatz_methodology`

Coupling-topology-informed ansatz: formal benchmark methodology.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/ansatz_methodology.py) · Public symbols: **4**

**Classes:** `AnsatzBenchmarkResult`

**Functions:** `benchmark_single_ansatz()`, `run_full_benchmark()`, `summarize_benchmark()`

### `scpn_quantum_control.phase.avqds`

McLachlan variational quantum real-time dynamics with a fixed ansatz.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/avqds.py) · Public symbols: **2**

**Classes:** `AVQDSResult`

**Functions:** `avqds_simulate()`

### `scpn_quantum_control.phase.backend_selector`

Auto-select simulation backend based on system size and available resources.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/backend_selector.py) · Public symbols: **4**

**Classes:** `BackendRecommendation`, `AutoSolveResult`

**Functions:** `recommend_backend()`, `auto_solve()`

### `scpn_quantum_control.phase.contraction_optimiser`

Optimal tensor contraction paths for MPS operations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/contraction_optimiser.py) · Public symbols: **6**

**Classes:** `ContractionPathInfo`, `ContractionBenchmarkResult`

**Functions:** `is_cotengra_available()`, `optimal_contraction_path()`, `contract()`, `benchmark_contraction()`

### `scpn_quantum_control.phase.coupling_learning`

Parameter-shift coupling learning for oscillator observation models.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/coupling_learning.py) · Public symbols: **5**

**Classes:** `CouplingLearningResult`, `CouplingGradientVerificationResult`

**Functions:** `coupling_matrix_from_edge_vector()`, `learn_couplings_from_observations()`, `verify_coupling_parameter_shift_gradient()`

### `scpn_quantum_control.phase.coupling_time_series_recovery`

Recover bounded Kuramoto/XY coupling matrices from synthetic time series.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/coupling_time_series_recovery.py) · Public symbols: **12**

**Classes:** `CouplingRecoveryCase`, `CouplingRecoveryRecord`, `CouplingRecoveryBoundaryRow`, `CouplingRecoverySuiteResult`

**Functions:** `simulate_kuramoto_phase_time_series()`, `simulate_xy_pair_energy_time_series()`, `inject_time_series_noise_and_missing()`, `recover_kuramoto_couplings_from_time_series()`, `recover_xy_couplings_from_pair_energy_series()`, `coupling_recovery_boundary_rows()`, `default_coupling_recovery_cases()`, `run_coupling_recovery_suite()`

### `scpn_quantum_control.phase.cross_domain_transfer`

Cross-domain VQE transfer learning.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/cross_domain_transfer.py) · Public symbols: **7**

**Classes:** `TransferSummary`, `TransferResult`, `PhysicalSystem`

**Functions:** `build_systems()`, `transfer_experiment()`, `run_transfer_matrix()`, `summarize_transfer()`

### `scpn_quantum_control.phase.differentiable_audit`

Reviewer-facing differentiable quantum gradient audit reports.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/differentiable_audit.py) · Public symbols: **7**

**Functions:** `verify_parameter_shift_analytic_gradient()`, `run_ml_framework_gradient_audit()`, `run_parameter_shift_audit_suite()`, `run_known_phase_gradient_audit()`, `run_finite_shot_gradient_uncertainty_audit()`, `run_differentiable_workflow_audit_suite()`, `run_phase_gradient_benchmark_suite()`

### `scpn_quantum_control.phase.differentiable_audit_contracts`

Immutable reports and serializers for differentiable gradient audits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/differentiable_audit_contracts.py) · Public symbols: **7**

**Classes:** `ParameterShiftAnalyticAgreement`, `DifferentiableQuantumAuditReport`, `PhaseGradientBenchmarkSuiteResult`, `DifferentiableWorkflowAuditSuiteResult`, `FiniteShotGradientAuditResult`, `MLFrameworkGradientAuditRecord`, `MLFrameworkGradientAuditSuiteResult`

### `scpn_quantum_control.phase.differentiable_readiness`

Unified readiness ledger for differentiable-programming evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/differentiable_readiness.py) · Public symbols: **5**

**Classes:** `DifferentiableReadinessSurface`, `DifferentiableReadinessAuditRecord`, `DifferentiableReadinessAuditResult`

**Functions:** `default_differentiable_readiness_surfaces()`, `run_differentiable_readiness_audit()`

### `scpn_quantum_control.phase.domain_benchmark_datasets`

Exact-answer differentiable benchmark datasets for bounded phase models.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/domain_benchmark_datasets.py) · Public symbols: **13**

**Classes:** `DifferentiableQNNExactAnswerCase`, `DifferentiableKuramotoExactAnswerCase`, `DifferentiableDomainBenchmarkDatasetSuite`, `DifferentiableDomainBenchmarkValidationResult`, `DifferentiableDomainBenchmarkValidationSuite`, `DifferentiablePublishedDomainBenchmarkCase`, `DifferentiablePublishedDomainBenchmarkSuite`, `DifferentiablePublishedDomainBenchmarkValidationResult`, `DifferentiablePublishedDomainBenchmarkValidationSuite`

**Functions:** `load_differentiable_domain_benchmark_datasets()`, `run_differentiable_domain_benchmark_dataset_validation()`, `load_differentiable_published_domain_benchmark_cases()`, `run_differentiable_published_domain_benchmark_validation()`

### `scpn_quantum_control.phase.floquet_kuramoto`

Floquet-Kuramoto: periodically driven XY synchronization.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/floquet_kuramoto.py) · Public symbols: **3**

**Classes:** `FloquetResult`

**Functions:** `floquet_evolve()`, `scan_drive_amplitude()`

### `scpn_quantum_control.phase.general_unitary`

U3 and arbitrary single-qubit unitary support via a registered ZYZ decomposition.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/general_unitary.py) · Public symbols: **2**

**Functions:** `su2_zyz_angles()`, `build_u3_operations()`

### `scpn_quantum_control.phase.generalised_parameter_shift`

Generalised parameter-shift plans for finite generator spectra.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/generalised_parameter_shift.py) · Public symbols: **8**

**Classes:** `GeneralisedParameterShiftTerm`, `GeneralisedParameterShiftPlan`, `GeneralisedParameterShiftResult`, `GeneralisedStochasticParameterShiftResult`

**Functions:** `plan_generalised_parameter_shift()`, `value_and_generalised_parameter_shift_grad()`, `generalised_parameter_shift_gradient()`, `estimate_generalised_parameter_shift_shot_noise()`

### `scpn_quantum_control.phase.gpu_batch_vqe`

Parallel VQE evaluation on GPU using PyTorch or JAX.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/gpu_batch_vqe.py) · Public symbols: **3**

**Functions:** `batch_energy_numpy()`, `batch_energy_torch()`, `batch_vqe_scan()`

### `scpn_quantum_control.phase.gradient_backend`

Backend-aware quantum-gradient planning for phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/gradient_backend.py) · Public symbols: **8**

**Classes:** `QuantumGradientBackendCapability`, `QuantumGradientPlan`, `QuantumGradientRejectedMethod`, `QuantumGradientShotPolicy`, `QuantumGradientMethodExplanation`

**Functions:** `quantum_gradient_backend_capability()`, `plan_quantum_gradient_backend()`, `explain_quantum_gradient_method()`

### `scpn_quantum_control.phase.gradient_descent`

Auditable parameter-shift gradient-descent training for phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/gradient_descent.py) · Public symbols: **5**

**Classes:** `ParameterShiftTrainingStep`, `ParameterShiftTrainingResult`, `ParameterShiftTrainingCertificate`

**Functions:** `parameter_shift_gradient_descent()`, `validate_parameter_shift_training()`

### `scpn_quantum_control.phase.gradient_support_matrix`

Executable support matrix for quantum-gradient combinations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/gradient_support_matrix.py) · Public symbols: **8**

**Classes:** `GradientSupportCapability`, `GradientSupportPlan`, `GradientSupportMatrixAuditResult`

**Functions:** `list_gradient_support_capabilities()`, `gradient_support_capability()`, `plan_gradient_support()`, `assert_gradient_support()`, `run_gradient_support_matrix_audit()`

### `scpn_quantum_control.phase.gradient_tape`

Context-managed quantum-gradient tape for phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/gradient_tape.py) · Public symbols: **6**

**Classes:** `TapeGradientRecord`, `GradientTapeContractCheck`, `GradientTapeContractAuditResult`, `QuantumGradientTape`

**Functions:** `run_gradient_tape_contract_audit()`, `gradient_tape()`

### `scpn_quantum_control.phase.hardware_gradient_campaign`

No-submit hardware-gradient campaign specifications for XY Hamiltonians.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/hardware_gradient_campaign.py) · Public symbols: **7**

**Classes:** `HardwareGradientReplaySchema`, `HardwareGradientCampaignSpec`, `HardwareGradientCampaignPlan`, `HardwareGradientCampaignSuite`

**Functions:** `default_hardware_gradient_campaign_specs()`, `plan_hardware_gradient_campaign()`, `run_hardware_gradient_campaign_readiness_suite()`

### `scpn_quantum_control.phase.hardware_gradient_policy`

Fail-closed policy checks for hardware quantum-gradient preparation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/hardware_gradient_policy.py) · Public symbols: **7**

**Classes:** `HardwareGradientRequest`, `HardwareGradientPolicy`, `HardwareGradientPolicyDecision`, `HardwareGradientReadinessSuiteResult`

**Functions:** `evaluate_hardware_gradient_policy()`, `assert_hardware_gradient_policy_approved()`, `run_hardware_gradient_policy_readiness_suite()`

### `scpn_quantum_control.phase.hardware_gradient_publication`

Publication package scaffold for no-submit XY hardware-gradient campaigns.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/hardware_gradient_publication.py) · Public symbols: **7**

**Classes:** `HardwareGradientPreregistration`, `HardwareGradientMethodSection`, `HardwareGradientArtifactMapEntry`, `HardwareGradientClaimLedgerRow`, `HardwareGradientBenchmarkPlaceholder`, `HardwareGradientPublicationPackage`

**Functions:** `build_hardware_gradient_publication_package()`

### `scpn_quantum_control.phase.jax_bridge`

Optional JAX execution and compatibility facade for phase gradients.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_bridge.py) · Public symbols: **18**

**Functions:** `is_phase_jax_available()`, `jax_parameter_shift_value_and_grad()`, `check_jax_parameter_shift_agreement()`, `jax_native_qnn_value_and_grad()`, `jax_custom_vjp_qnn_value_and_grad()`, `jax_phase_qnode_value_and_grad()`, `jax_phase_qnode_native_transform_audit()`, `jax_phase_qnode_pytree_transform_audit()`, `jax_phase_qnode_sharding_transform_audit()`, `jax_phase_qnode_aot_export_audit()`, `run_jax_jit_compatibility_audit()`, `run_jax_vmap_compatibility_audit()`, `run_jax_sharding_compatibility_audit()`, `run_jax_pytree_compatibility_audit()`, `run_jax_nested_transform_algebra_audit()`, `run_jax_phase_qnode_lowering_matrix()`, `plan_jax_cloud_validation_batch()`, `run_jax_maturity_audit()`

### `scpn_quantum_control.phase.jax_bridge_contracts`

Immutable result contracts for the optional phase JAX bridge.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_bridge_contracts.py) · Public symbols: **19**

**Classes:** `PhaseJAXParameterShiftResult`, `PhaseJAXGradientAgreementResult`, `PhaseJAXNativeQNNGradientResult`, `PhaseJAXCustomVJPQNNGradientResult`, `PhaseJAXPhaseQNodeStatevectorResult`, `PhaseJAXPhaseQNodeNativeTransformResult`, `PhaseJAXPhaseQNodePyTreeTransformResult`, `PhaseJAXPhaseQNodeShardingTransformResult`, `PhaseJAXPhaseQNodeAOTExportResult`, `PhaseJAXJITCompatibilityResult`, `PhaseJAXVMAPCompatibilityResult`, `PhaseJAXShardingCompatibilityResult`, `PhaseJAXPyTreeCompatibilityResult`, `PhaseJAXMaturityAuditResult`, `PhaseJAXNestedTransformRoute`, `PhaseJAXNestedTransformAlgebraResult`, `PhaseJAXCloudValidationRunSpec`, `PhaseJAXPhaseQNodeLoweringRoute`, `PhaseJAXPhaseQNodeLoweringMatrixResult`

### `scpn_quantum_control.phase.jax_compatibility`

Bounded phase-QNN JAX compatibility and nested-transform audits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_compatibility.py) · Public symbols: **5**

**Functions:** `run_jax_jit_compatibility_audit()`, `run_jax_vmap_compatibility_audit()`, `run_jax_sharding_compatibility_audit()`, `run_jax_pytree_compatibility_audit()`, `run_jax_nested_transform_algebra_audit()`

### `scpn_quantum_control.phase.jax_gradients`

Bounded parameter-shift and QNN gradient execution for the JAX bridge.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_gradients.py) · Public symbols: **4**

**Functions:** `jax_parameter_shift_value_and_grad()`, `check_jax_parameter_shift_agreement()`, `jax_native_qnn_value_and_grad()`, `jax_custom_vjp_qnn_value_and_grad()`

### `scpn_quantum_control.phase.jax_maturity`

JAX lowering declarations, cloud planning, and maturity aggregation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_maturity.py) · Public symbols: **3**

**Functions:** `run_jax_phase_qnode_lowering_matrix()`, `plan_jax_cloud_validation_batch()`, `run_jax_maturity_audit()`

### `scpn_quantum_control.phase.jax_nqs`

JAX-based exact-enumeration RBM wavefunction with automatic differentiation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_nqs.py) · Public symbols: **3**

**Functions:** `is_jax_available()`, `jax_rbm_energy()`, `jax_vmc_ground_state()`

### `scpn_quantum_control.phase.jax_qnode_transforms`

Native JAX execution and transforms for registered local Phase-QNodes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/jax_qnode_transforms.py) · Public symbols: **5**

**Functions:** `jax_phase_qnode_value_and_grad()`, `jax_phase_qnode_native_transform_audit()`, `jax_phase_qnode_pytree_transform_audit()`, `jax_phase_qnode_sharding_transform_audit()`, `jax_phase_qnode_aot_export_audit()`

### `scpn_quantum_control.phase.kuramoto_variants`

Higher-order, monitored, and PT-symmetric Kuramoto trajectories.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/kuramoto_variants.py) · Public symbols: **10**

**Classes:** `KuramotoVariant`, `KuramotoVariantResult`, `HigherOrderKuramotoSpec`, `MonitoredKuramotoSpec`, `PTSymmetricKuramotoSpec`

**Functions:** `build_triadic_ring_terms()`, `simulate_higher_order_kuramoto()`, `simulate_monitored_kuramoto()`, `simulate_pt_symmetric_kuramoto()`, `validate_variant_kuramoto_inputs()`

### `scpn_quantum_control.phase.lindblad`

Lindblad master equation solver for open Kuramoto-XY systems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/lindblad.py) · Public symbols: **1**

**Classes:** `LindbladKuramotoSolver`

### `scpn_quantum_control.phase.lindblad_engine`

Lindblad Master Equation and Quantum Trajectory Solver.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/lindblad_engine.py) · Public symbols: **1**

**Classes:** `LindbladSyncEngine`

### `scpn_quantum_control.phase.model_training_evidence`

Registered medium-scale local differentiable-model training evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/model_training_evidence.py) · Public symbols: **6**

**Classes:** `DifferentiableModelTrainingRecord`, `DifferentiableModelTrainingEvidenceSuite`, `RegisteredDifferentiableTrainingSuiteRecord`, `RegisteredDifferentiableTrainingSuiteAuditResult`

**Functions:** `run_differentiable_model_training_evidence_suite()`, `run_registered_differentiable_training_suite_audit()`

### `scpn_quantum_control.phase.mps_evolution`

Matrix Product State backend for large-N Kuramoto-XY.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/mps_evolution.py) · Public symbols: **3**

**Functions:** `is_quimb_available()`, `dmrg_ground_state()`, `tebd_evolution()`

### `scpn_quantum_control.phase.natural_gradient`

Metric-aware parameter-shift optimisation for supported phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/natural_gradient.py) · Public symbols: **8**

**Classes:** `NaturalGradientRegularizationPolicy`, `NaturalGradientDirection`, `ParameterShiftNaturalGradientStep`, `ParameterShiftNaturalGradientResult`, `ParameterShiftNaturalGradientCertificate`

**Functions:** `solve_natural_gradient_direction()`, `parameter_shift_natural_gradient_descent()`, `validate_natural_gradient_training()`

### `scpn_quantum_control.phase.nqs_ansatz`

Restricted Boltzmann Machine (RBM) ansatz for Kuramoto-XY ground state.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/nqs_ansatz.py) · Public symbols: **2**

**Classes:** `RBMWavefunction`

**Functions:** `vmc_ground_state()`

### `scpn_quantum_control.phase.objective_audit`

Reviewer-facing correctness evidence for composed phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/objective_audit.py) · Public symbols: **4**

**Classes:** `ComposedObjectiveGradientAgreement`, `ComposedObjectiveAuditSuiteResult`

**Functions:** `verify_composed_objective_gradient()`, `run_composed_objective_audit_suite()`

### `scpn_quantum_control.phase.objective_planner`

Fail-closed execution planning for composed phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/objective_planner.py) · Public symbols: **5**

**Classes:** `ComposedObjectiveExecutionPlan`, `ComposedObjectivePlannerAuditResult`

**Functions:** `plan_composed_objective_execution()`, `assert_composed_objective_execution_supported()`, `run_composed_objective_planner_audit()`

### `scpn_quantum_control.phase.objectives`

Composable differentiable objectives for phase-control training.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/objectives.py) · Public symbols: **15**

**Classes:** `ObjectiveTermValue`, `ObjectiveGradientEvaluation`, `ObjectiveTerm`, `ComposedPhaseObjective`, `ComposedObjectiveTrainingStep`, `ComposedObjectiveTrainingResult`, `ComposedObjectiveTrainingCertificate`

**Functions:** `phase_energy_term()`, `phase_fidelity_target_term()`, `periodic_regularization_term()`, `phase_symmetry_penalty_term()`, `smooth_box_safety_penalty_term()`, `build_phase_control_objective()`, `train_composed_phase_objective()`, `validate_composed_objective_training()`

### `scpn_quantum_control.phase.open_system_objectives`

Bounded Lindblad and MCWF objective certificates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/open_system_objectives.py) · Public symbols: **13**

**Classes:** `BoundedOpenSystemObjectiveCase`, `DensityMatrixInvariantCertificate`, `MCWFReproducibilityCertificate`, `OpenSystemObjectiveRecord`, `OpenSystemObjectiveBoundaryRow`, `OpenSystemObjectiveSuiteResult`

**Functions:** `default_open_system_objective_cases()`, `evaluate_lindblad_objective()`, `evaluate_mcwf_objective()`, `certify_density_matrix_invariants()`, `certify_mcwf_reproducibility()`, `open_system_objective_boundary_rows()`, `run_open_system_objective_suite()`

### `scpn_quantum_control.phase.optimizer_audit`

Multi-start convergence evidence for parameter-shift phase optimizers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/optimizer_audit.py) · Public symbols: **3**

**Classes:** `OptimizerConvergenceRecord`, `OptimizerComparisonSuiteResult`

**Functions:** `run_parameter_shift_optimizer_comparison()`

### `scpn_quantum_control.phase.optimizer_convergence_suite`

Convergence certificates for small phase ground-state objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/optimizer_convergence_suite.py) · Public symbols: **7**

**Classes:** `KnownGroundStateObjective`, `GroundStateConvergenceCertificate`, `GroundStateOptimizerRunRecord`, `GroundStateOptimizerBoundaryRow`, `GroundStateOptimizerConvergenceSuiteResult`

**Functions:** `default_ground_state_optimizer_objectives()`, `run_ground_state_optimizer_convergence_suite()`

### `scpn_quantum_control.phase.param_shift`

Parameter-shift gradients for phase and Kuramoto-XY VQE objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/param_shift.py) · Public symbols: **17**

**Classes:** `ParamShiftConvergenceDiagnostics`, `ParamShiftVQEResult`, `GenericParameterShiftEvaluationPlan`, `GradientVerificationResult`, `HessianVerificationResult`

**Functions:** `parameter_shift_gradient()`, `plan_generic_parameter_shift_evaluations()`, `parameter_shift_hessian()`, `verify_parameter_shift_gradient()`, `verify_parameter_shift_hessian()`, `verify_vqe_parameter_shift_gradient()`, `verify_vqe_parameter_shift_hessian()`, `parameter_shift_gradient_with_uncertainty()`, `plan_parameter_shift_shots()`, `validate_param_shift_convergence()`, `value_and_vqe_grad()`, `vqe_with_param_shift()`

### `scpn_quantum_control.phase.pennylane_bridge`

Optional PennyLane agreement checks for phase parameter-shift gradients.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/pennylane_bridge.py) · Public symbols: **10**

**Classes:** `PennyLaneGradientAgreementResult`, `PennyLaneRoundTripResult`, `PennyLaneQNodeConversionResult`, `PennyLaneMaturityAuditResult`

**Functions:** `is_phase_pennylane_available()`, `check_pennylane_parameter_shift_agreement()`, `check_pennylane_qnode_round_trip()`, `build_pennylane_qnode_from_phase_qnode()`, `check_pennylane_phase_qnode_round_trip()`, `run_pennylane_maturity_audit()`

### `scpn_quantum_control.phase.pennylane_import`

Convert a PennyLane quantum tape into a registered Phase-QNode circuit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/pennylane_import.py) · Public symbols: **5**

**Classes:** `PennyLaneImportResult`, `PennyLaneImportRoundTripResult`

**Functions:** `is_pennylane_import_available()`, `import_phase_qnode_from_pennylane()`, `check_pennylane_phase_qnode_import_round_trip()`

### `scpn_quantum_control.phase.pennylane_provider_plugin`

PennyLane provider-plugin gradient artefacts and fail-closed route matrix.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/pennylane_provider_plugin.py) · Public symbols: **7**

**Classes:** `PennyLanePluginMatrixRoute`, `PennyLaneProviderPluginExecutionArtifact`, `PennyLaneProviderGradientParityArtifact`, `PennyLaneHardwarePluginExecutionArtifact`, `PennyLaneProviderEvidenceBundle`, `PennyLanePluginMatrixResult`

**Functions:** `run_pennylane_plugin_matrix()`

### `scpn_quantum_control.phase.phase_vqe`

VQE for Kuramoto/XY Hamiltonian ground state.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/phase_vqe.py) · Public symbols: **1**

**Classes:** `PhaseVQE`

### `scpn_quantum_control.phase.provider_gradient`

Provider-safe parameter-shift gradient execution contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/provider_gradient.py) · Public symbols: **6**

**Classes:** `ProviderExpectationSample`, `ProviderParameterShiftRecord`, `ProviderGradientExecutionResult`, `ProviderHardwareGradientPreparationResult`

**Functions:** `prepare_provider_hardware_parameter_shift_gradient()`, `execute_provider_parameter_shift_gradient()`

### `scpn_quantum_control.phase.provider_gradient_audit`

Executable readiness audit for provider-safe quantum gradients.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/provider_gradient_audit.py) · Public symbols: **5**

**Classes:** `ProviderGradientReadinessScenario`, `ProviderGradientReadinessRecord`, `ProviderGradientReadinessAuditResult`

**Functions:** `default_provider_gradient_readiness_scenarios()`, `run_provider_gradient_readiness_audit()`

### `scpn_quantum_control.phase.provider_hardware_gradient_audit`

Executable audit for provider hardware-gradient preparation readiness.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py) · Public symbols: **5**

**Classes:** `ProviderHardwareGradientPreparationScenario`, `ProviderHardwareGradientPreparationRecord`, `ProviderHardwareGradientPreparationAuditResult`

**Functions:** `default_provider_hardware_gradient_preparation_scenarios()`, `run_provider_hardware_gradient_preparation_audit()`

### `scpn_quantum_control.phase.provider_hardware_safety_audit`

Aggregate safety gate for differentiable provider and hardware-gradient paths.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/provider_hardware_safety_audit.py) · Public symbols: **3**

**Classes:** `DifferentiableProviderHardwareSafetySurface`, `DifferentiableProviderHardwareSafetyAuditResult`

**Functions:** `run_differentiable_provider_hardware_safety_audit()`

### `scpn_quantum_control.phase.pulse_shaping`

PMP-optimal ICI pulse sequences and (α,β)-hypergeometric pulse shaping.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/pulse_shaping.py) · Public symbols: **10**

**Classes:** `ICIPulse`, `HypergeometricPulse`, `PulseSchedule`

**Functions:** `ici_mixing_angle()`, `build_ici_pulse()`, `ici_three_level_evolution()`, `hypergeometric_envelope()`, `build_hypergeometric_pulse()`, `infidelity_bound()`, `build_trotter_pulse_schedule()`

### `scpn_quantum_control.phase.qgnn`

A local quantum graph neural network that maps K_nm graphs to circuit outputs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qgnn.py) · Public symbols: **10**

**Classes:** `KnmGraph`, `QGNNConfig`, `QGNNTrainingResult`

**Functions:** `validate_graph()`, `parameter_count()`, `initialise_parameters()`, `predict()`, `predict_and_gradient()`, `synthetic_kuramoto_target()`, `train()`

### `scpn_quantum_control.phase.qiskit_bridge`

Compatibility facade for Qiskit gradient and Runtime evidence routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qiskit_bridge.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.phase.qiskit_bridge_contracts`

Qiskit gradient, Runtime, and provider-evidence record contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qiskit_bridge_contracts.py) · Public symbols: **9**

**Classes:** `QiskitParameterShiftRecord`, `QiskitParameterShiftGradientResult`, `QiskitRuntimePrimitiveExecutionArtifact`, `QiskitRuntimeQPUExecutionArtifact`, `QiskitRawCountReplayArtifact`, `QiskitCalibrationStatevectorComparisonArtifact`, `QiskitProviderGradientWorkflowArtifact`, `QiskitRuntimeQPUProviderEvidenceBundle`, `QiskitMaturityAuditResult`

### `scpn_quantum_control.phase.qiskit_gradients`

Qiskit shifted-circuit generation and local gradient execution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qiskit_gradients.py) · Public symbols: **3**

**Functions:** `generate_qiskit_parameter_shift_circuits()`, `execute_qiskit_statevector_parameter_shift()`, `execute_qiskit_finite_shot_parameter_shift()`

### `scpn_quantum_control.phase.qiskit_runtime`

Qiskit Runtime evidence capture and maturity orchestration.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qiskit_runtime.py) · Public symbols: **4**

**Functions:** `build_qiskit_runtime_qpu_execution_artifact()`, `build_qiskit_provider_gradient_workflow_artifact()`, `build_qiskit_runtime_qpu_provider_evidence_bundle()`, `run_qiskit_maturity_audit()`

### `scpn_quantum_control.phase.qnn_conformance`

Conformance evidence for bounded phase-QNN differentiable workflows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_conformance.py) · Public symbols: **5**

**Classes:** `ParameterShiftQNNUnsupportedScenario`, `ParameterShiftQNNConformanceCaseResult`, `ParameterShiftQNNConformanceSuiteResult`

**Functions:** `summarize_parameter_shift_qnn_unsuitable_scenarios()`, `run_parameter_shift_qnn_conformance_suite()`

### `scpn_quantum_control.phase.qnn_convergence`

Deterministic convergence evidence for bounded phase-QNN training.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_convergence.py) · Public symbols: **9**

**Classes:** `ParameterShiftQNNConvergenceUnsuitableScenario`, `ParameterShiftQNNConvergenceCaseResult`, `ParameterShiftQNNConvergenceSuiteResult`, `ParameterShiftQNNMultiSeedConvergenceRunResult`, `ParameterShiftQNNMultiSeedConvergenceCaseResult`, `ParameterShiftQNNMultiSeedConvergenceSuiteResult`

**Functions:** `summarize_parameter_shift_qnn_convergence_unsuitable_scenarios()`, `run_parameter_shift_qnn_convergence_suite()`, `run_parameter_shift_qnn_multi_seed_convergence_suite()`

### `scpn_quantum_control.phase.qnn_finite_shot`

Seeded finite-shot evidence for bounded phase-QNN gradients and training.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_finite_shot.py) · Public symbols: **8**

**Classes:** `ParameterShiftQNNFiniteShotProbeRecord`, `ParameterShiftQNNFiniteShotGradientResult`, `ParameterShiftQNNFiniteShotConvergenceCaseResult`, `ParameterShiftQNNFiniteShotUnsupportedScenario`, `ParameterShiftQNNFiniteShotConvergenceSuiteResult`

**Functions:** `estimate_parameter_shift_qnn_finite_shot_gradient()`, `summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios()`, `run_parameter_shift_qnn_finite_shot_convergence_suite()`

### `scpn_quantum_control.phase.qnn_framework_agreement`

Caller-supplied framework-gradient agreement for bounded phase QNNs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_framework_agreement.py) · Public symbols: **5**

**Classes:** `ParameterShiftQNNFrameworkGradientAgreement`, `ParameterShiftQNNFrameworkAgreementResult`, `ParameterShiftQNNFrameworkAgreementSuiteResult`

**Functions:** `verify_parameter_shift_qnn_framework_agreement()`, `run_parameter_shift_qnn_framework_agreement_suite()`

### `scpn_quantum_control.phase.qnn_framework_bridge_matrix`

Fail-closed support matrix for bounded phase-QNN framework bridges.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py) · Public symbols: **4**

**Classes:** `BoundedQNNFrameworkBridgeCapability`, `BoundedQNNFrameworkBridgeMatrixResult`

**Functions:** `run_bounded_qnn_framework_bridge_matrix()`, `assert_bounded_qnn_framework_bridge_supported()`

### `scpn_quantum_control.phase.qnn_loss_landscape`

Deterministic loss-landscape evidence for bounded phase-QNN training.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_loss_landscape.py) · Public symbols: **4**

**Classes:** `ParameterShiftQNNLossLandscapePoint`, `ParameterShiftQNNLossLandscapeCaseResult`, `ParameterShiftQNNLossLandscapeSuiteResult`

**Functions:** `run_parameter_shift_qnn_loss_landscape_suite()`

### `scpn_quantum_control.phase.qnn_optimizer_benchmark`

Functional optimizer benchmarks for bounded phase-QNN training.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py) · Public symbols: **4**

**Classes:** `QNNOptimizerBaselineResult`, `ParameterShiftQNNOptimizerBenchmarkCaseResult`, `ParameterShiftQNNOptimizerBenchmarkSuiteResult`

**Functions:** `run_parameter_shift_qnn_optimizer_benchmark_suite()`

### `scpn_quantum_control.phase.qnn_training`

Deterministic parameter-shift training for bounded phase QNN classifiers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnn_training.py) · Public symbols: **9**

**Classes:** `ParameterShiftQNNPredictionResult`, `ParameterShiftQNNTrainingResult`, `ParameterShiftQNNExternalGradientAgreement`, `ParameterShiftQNNGradientVerificationResult`

**Functions:** `predict_parameter_shift_qnn_classifier()`, `parameter_shift_qnn_classifier_loss()`, `parameter_shift_qnn_classifier_gradient()`, `verify_parameter_shift_qnn_classifier_gradient()`, `train_parameter_shift_qnn_classifier()`

### `scpn_quantum_control.phase.qnode_affinity_benchmark`

Small Phase-QNode benchmark harness with isolation metadata.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_affinity_benchmark.py) · Public symbols: **6**

**Classes:** `PhaseQNodeAffinityBenchmarkMetadata`, `PhaseQNodeAffinityBenchmarkResult`, `PhaseQNodeAffinityArtifactValidation`

**Functions:** `classify_affinity_evidence()`, `run_phase_qnode_affinity_benchmark()`, `validate_phase_qnode_affinity_artifact()`

### `scpn_quantum_control.phase.qnode_circuit`

Executable facade for registered Phase-QNode circuit routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_circuit.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.phase.qnode_circuit_builders`

Registered Phase-QNode vocabulary, observables, decompositions, and templates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_circuit_builders.py) · Public symbols: **8**

**Functions:** `registered_phase_qnode_gates()`, `registered_phase_qnode_observables()`, `registered_phase_qnode_templates()`, `registered_phase_qnode_decompositions()`, `registered_phase_qnode_noise_channels()`, `build_sparse_ising_chain_hamiltonian()`, `decompose_phase_qnode_controlled_gate()`, `build_phase_qnode_template()`

### `scpn_quantum_control.phase.qnode_circuit_contracts`

Phase-QNode declarations, result records, registries, and validation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_circuit_contracts.py) · Public symbols: **20**

**Classes:** `PhaseQNodeOperation`, `PhaseQNodeNoiseChannel`, `PauliTerm`, `SparsePauliHamiltonian`, `DenseHermitianObservable`, `PauliCovarianceObservable`, `PhaseQNodeSupportReport`, `PhaseQNodeSupportError`, `PhaseQNodeCircuit`, `PhaseQNodeDensityCircuit`, `PhaseQNodeTemplateSpec`, `PhaseQNodeDepthProfile`, `PhaseQNodeRegisteredCircuitSpec`, `PhaseQNodeExecutionResult`, `PhaseQNodeDensityExecutionResult`, `PhaseQNodeGradientResult`, `PhaseQNodeGradientEvaluationGroup`, `PhaseQNodeGradientEvaluationPlan`, `PhaseQNodeMetricTensorResult`, `PhaseQNodeClassicalFisherResult`

### `scpn_quantum_control.phase.qnode_circuit_differentiation`

Gradient, Fisher, and metric orchestration for Phase-QNode circuits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_circuit_differentiation.py) · Public symbols: **5**

**Functions:** `phase_qnode_computational_basis_fisher_support_report()`, `parameter_shift_phase_qnode_gradient()`, `phase_qnode_quantum_fisher_information()`, `phase_qnode_computational_basis_fisher_information()`, `phase_qnode_natural_gradient_metric()`

### `scpn_quantum_control.phase.qnode_circuit_execution`

Numerical statevector and density execution for Phase-QNode circuits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_circuit_execution.py) · Public symbols: **2**

**Functions:** `execute_phase_qnode_circuit()`, `execute_phase_qnode_density_matrix()`

### `scpn_quantum_control.phase.qnode_circuit_support`

Declarative support analysis and shift planning for Phase-QNode circuits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_circuit_support.py) · Public symbols: **7**

**Functions:** `build_registered_phase_qnode_circuit()`, `phase_qnode_depth_profile()`, `phase_qnode_support_report()`, `phase_qnode_density_support_report()`, `phase_qnode_gradient_support_report()`, `plan_phase_qnode_parameter_shift_evaluations()`, `phase_qnode_metric_support_report()`

### `scpn_quantum_control.phase.qnode_framework_parity`

Real optional-framework parity checks for a bounded Phase-QNode circuit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_framework_parity.py) · Public symbols: **3**

**Classes:** `PhaseQNodeFrameworkParityRecord`, `PhaseQNodeFrameworkParitySuiteResult`

**Functions:** `run_phase_qnode_framework_parity_suite()`

### `scpn_quantum_control.phase.qnode_provider_transforms`

Provider-callback QNode transform execution evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_provider_transforms.py) · Public symbols: **5**

**Classes:** `ProviderQNodeTransformResult`, `ProviderQNodeTransformReadinessSuiteResult`

**Functions:** `execute_provider_qnode_transform()`, `execute_provider_qnode_vmap_grad()`, `run_provider_qnode_transform_readiness_suite()`

### `scpn_quantum_control.phase.qnode_tape`

QNode-style differentiable tape records for supported phase objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_tape.py) · Public symbols: **5**

**Classes:** `PhaseQNodeTapeRecord`, `PhaseQNodeTapeReadinessSuiteResult`, `PhaseQNodeTape`

**Functions:** `phase_qnode_tape()`, `run_phase_qnode_tape_readiness_suite()`

### `scpn_quantum_control.phase.qnode_transforms`

Executable scalar phase-QNode transform evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_transforms.py) · Public symbols: **7**

**Classes:** `PhaseQNodeComplexDerivativeContract`, `PhaseQNodeTransformResult`, `PhaseQNodeTransformReadinessSuiteResult`

**Functions:** `execute_phase_qnode_transform()`, `phase_qnode_complex_derivative_contract()`, `execute_phase_qnode_hessian_vector_product()`, `run_phase_qnode_transform_readiness_suite()`

### `scpn_quantum_control.phase.qnode_vector_transforms`

Executable vector-output phase-QNode Jacobian and native vmap evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qnode_vector_transforms.py) · Public symbols: **8**

**Classes:** `PhaseQNodeVectorTransformResult`, `PhaseQNodeVectorTransformReadinessSuiteResult`

**Functions:** `execute_phase_qnode_vector_jacobian()`, `execute_phase_qnode_vector_hessian()`, `execute_phase_qnode_vector_jvp()`, `execute_phase_qnode_vector_vjp()`, `execute_phase_qnode_vmap_grad()`, `run_phase_qnode_vector_transform_readiness_suite()`

### `scpn_quantum_control.phase.qsvt_evolution`

QSVT-based Hamiltonian simulation for the Kuramoto-XY model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/qsvt_evolution.py) · Public symbols: **8**

**Classes:** `QSVTResourceEstimate`

**Functions:** `hamiltonian_1norm()`, `hamiltonian_spectral_norm()`, `qsvt_query_count()`, `trotter1_step_count()`, `trotter2_step_count()`, `qsvt_resource_estimate()`, `qsp_phase_angles()`

### `scpn_quantum_control.phase.results`

Typed result objects for phase-dynamics APIs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/results.py) · Public symbols: **1**

**Classes:** `TrajectoryResult`

### `scpn_quantum_control.phase.structured_ansatz`

General-purpose structured VQE ansatz based on physical coupling matrices.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/structured_ansatz.py) · Public symbols: **1**

**Functions:** `build_structured_ansatz()`

### `scpn_quantum_control.phase.synchronisation_objectives`

Differentiable synchronisation losses for phase-control objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/synchronisation_objectives.py) · Public symbols: **6**

**Functions:** `kuramoto_order_parameter()`, `kuramoto_order_parameter_gradient()`, `kuramoto_order_parameter_target_term()`, `phase_locking_target_term()`, `cluster_synchronisation_target_term()`, `build_synchronisation_objective()`

### `scpn_quantum_control.phase.synchronisation_witness`

Order-parameter and persistent-homology synchronisation witnesses.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/synchronisation_witness.py) · Public symbols: **12**

**Classes:** `SyncWitnessCase`, `SyncWitnessRecord`, `SyncWitnessBoundaryRow`, `SyncWitnessSuiteResult`

**Functions:** `harmonic_order_parameter()`, `geodesic_phase_distance_matrix()`, `vietoris_rips_persistence()`, `betti_curve()`, `phase_cloud_synchronisation_witness()`, `default_sync_witness_cases()`, `sync_witness_boundary_rows()`, `run_sync_witness_suite()`

### `scpn_quantum_control.phase.tensor_jump`

Monte Carlo Wave Function (MCWF) method for open Kuramoto-XY systems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/tensor_jump.py) · Public symbols: **2**

**Functions:** `mcwf_trajectory()`, `mcwf_ensemble()`

### `scpn_quantum_control.phase.tensorflow_bridge`

Signature-stable facade for optional TensorFlow phase-gradient routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/tensorflow_bridge.py) · Public symbols: **10**

**Functions:** `is_phase_tensorflow_available()`, `run_tensorflow_phase_qnode_lowering_matrix()`, `tensorflow_parameter_shift_value_and_grad()`, `tensorflow_bounded_qnn_value_and_grad()`, `run_tensorflow_gradient_tape_compatibility_audit()`, `run_tensorflow_function_compatibility_audit()`, `run_tensorflow_xla_compatibility_audit()`, `tensorflow_bounded_qnn_keras_layer()`, `run_tensorflow_keras_layer_wrapper_audit()`, `run_tensorflow_maturity_audit()`

### `scpn_quantum_control.phase.tensorflow_bridge_contracts`

TensorFlow bridge result, compatibility, lowering, and maturity records.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/tensorflow_bridge_contracts.py) · Public symbols: **9**

**Classes:** `PhaseTensorFlowParameterShiftResult`, `PhaseTensorFlowQNNGradientResult`, `PhaseTensorFlowGradientTapeCompatibilityResult`, `PhaseTensorFlowFunctionCompatibilityResult`, `PhaseTensorFlowXLACompatibilityResult`, `PhaseTensorFlowKerasLayerWrapperAuditResult`, `PhaseTensorFlowMaturityAuditResult`, `PhaseTensorFlowPhaseQNodeLoweringRoute`, `PhaseTensorFlowPhaseQNodeLoweringMatrixResult`

### `scpn_quantum_control.phase.tensorflow_compatibility`

Bounded TensorFlow compatibility, lowering, and maturity execution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/tensorflow_compatibility.py) · Public symbols: **7**

**Functions:** `run_tensorflow_phase_qnode_lowering_matrix()`, `run_tensorflow_gradient_tape_compatibility_audit()`, `run_tensorflow_function_compatibility_audit()`, `run_tensorflow_xla_compatibility_audit()`, `tensorflow_bounded_qnn_keras_layer()`, `run_tensorflow_keras_layer_wrapper_audit()`, `run_tensorflow_maturity_audit()`

### `scpn_quantum_control.phase.tensorflow_gradients`

Bounded TensorFlow gradient execution and direct validation primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/tensorflow_gradients.py) · Public symbols: **2**

**Functions:** `tensorflow_parameter_shift_value_and_grad()`, `tensorflow_bounded_qnn_value_and_grad()`

### `scpn_quantum_control.phase.tensorflow_maintenance`

TensorFlow maintenance decision for differentiable framework parity.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/tensorflow_maintenance.py) · Public symbols: **3**

**Classes:** `PhaseTensorFlowMaintenanceRoute`, `PhaseTensorFlowMaintenanceReport`

**Functions:** `run_tensorflow_maintenance_decision()`

### `scpn_quantum_control.phase.torch_aot_autograd_export`

AOTAutograd FX graph persistence for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_aot_autograd_export.py) · Public symbols: **4**

**Classes:** `PhaseTorchAOTAutogradGraphRecord`, `PhaseTorchAOTAutogradExportRoute`, `PhaseTorchAOTAutogradExportResult`

**Functions:** `run_torch_aot_autograd_export_audit()`

### `scpn_quantum_control.phase.torch_autograd_function`

Bounded PyTorch ``autograd.Function`` utilities for phase-QNN gradients.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_autograd_function.py) · Public symbols: **4**

**Classes:** `PhaseTorchAutogradFunctionRoute`, `PhaseTorchAutogradFunctionResult`

**Functions:** `torch_autograd_function_qnn_loss()`, `run_torch_autograd_function_audit()`

### `scpn_quantum_control.phase.torch_bridge`

Optional PyTorch execution facade for phase gradients.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_bridge.py) · Public symbols: **18**

**Functions:** `is_phase_torch_available()`, `run_torch_phase_qnode_lowering_matrix()`, `torch_parameter_shift_value_and_grad()`, `torch_phase_qnode_value_and_grad()`, `torch_phase_qnode_transform_audit()`, `torch_phase_qnode_compile_audit()`, `torch_phase_qnode_compile_boundary_audit()`, `torch_bounded_qnn_value_and_grad()`, `torch_autograd_qnn_value_and_grad()`, `run_torch_func_compatibility_audit()`, `run_torch_compile_compatibility_audit()`, `torch_bounded_qnn_module()`, `torch_bounded_qnn_layer()`, `run_torch_module_wrapper_audit()`, `run_torch_training_loop_audit()`, `run_torch_ecosystem_maturity_audit()`, `plan_torch_cloud_validation_batch()`, `run_torch_maturity_audit()`

### `scpn_quantum_control.phase.torch_bridge_contracts`

Immutable NumPy/stdlib result contracts for the optional Torch bridge.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_bridge_contracts.py) · Public symbols: **19**

**Classes:** `PhaseTorchParameterShiftResult`, `PhaseTorchQNNGradientResult`, `PhaseTorchAutogradQNNGradientResult`, `PhaseTorchFuncCompatibilityResult`, `PhaseTorchCompileCompatibilityResult`, `PhaseTorchModuleWrapperAuditResult`, `PhaseTorchTrainingLoopAuditResult`, `PhaseTorchPhaseQNodeStatevectorResult`, `PhaseTorchPhaseQNodeTransformResult`, `PhaseTorchPhaseQNodeCompileResult`, `PhaseTorchCompileBoundaryRoute`, `PhaseTorchCompileBoundaryAuditResult`, `PhaseTorchLiveOverlayEvidence`, `PhaseTorchEcosystemMaturityRoute`, `PhaseTorchEcosystemMaturityAuditResult`, `PhaseTorchCloudValidationRunSpec`, `PhaseTorchMaturityAuditResult`, `PhaseTorchPhaseQNodeLoweringRoute`, `PhaseTorchPhaseQNodeLoweringMatrixResult`

### `scpn_quantum_control.phase.torch_checkpoint`

Checkpoint replay utilities for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_checkpoint.py) · Public symbols: **3**

**Classes:** `PhaseTorchCheckpointRoute`, `PhaseTorchCheckpointAuditResult`

**Functions:** `run_torch_module_checkpoint_audit()`

### `scpn_quantum_control.phase.torch_checkpoint_matrix`

Long-lived checkpoint matrix utilities for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_checkpoint_matrix.py) · Public symbols: **5**

**Classes:** `PhaseTorchCheckpointMatrixRoute`, `PhaseTorchCheckpointMatrixTensorMetadata`, `PhaseTorchCheckpointRuntimeFingerprint`, `PhaseTorchCheckpointMatrixResult`

**Functions:** `run_torch_long_lived_checkpoint_matrix()`

### `scpn_quantum_control.phase.torch_compatibility`

Bounded Torch transforms, modules, and deterministic training evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_compatibility.py) · Public symbols: **6**

**Functions:** `run_torch_func_compatibility_audit()`, `run_torch_compile_compatibility_audit()`, `torch_bounded_qnn_module()`, `torch_bounded_qnn_layer()`, `run_torch_module_wrapper_audit()`, `run_torch_training_loop_audit()`

### `scpn_quantum_control.phase.torch_device_state`

Device-state replay utilities for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_device_state.py) · Public symbols: **3**

**Classes:** `PhaseTorchDeviceStateRoute`, `PhaseTorchDeviceStateAuditResult`

**Functions:** `run_torch_module_device_state_audit()`

### `scpn_quantum_control.phase.torch_dynamic_shape_export`

Dynamic-batch ``torch.export`` replay for bounded phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_dynamic_shape_export.py) · Public symbols: **6**

**Classes:** `PhaseTorchDynamicShapeExportReplayCase`, `PhaseTorchDynamicShapeExportRecord`, `PhaseTorchDynamicShapeExportRoute`, `PhaseTorchDynamicShapeExportResult`

**Functions:** `default_torch_dynamic_shape_export_replay_cases()`, `run_torch_dynamic_shape_export_audit()`

### `scpn_quantum_control.phase.torch_export`

Export-persistence utilities for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_export.py) · Public symbols: **3**

**Classes:** `PhaseTorchExportRoute`, `PhaseTorchExportAuditResult`

**Functions:** `run_torch_module_export_audit()`

### `scpn_quantum_control.phase.torch_export_shape_matrix`

Static-shape export matrix for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_export_shape_matrix.py) · Public symbols: **6**

**Classes:** `PhaseTorchExportShapeScenario`, `PhaseTorchExportShapeMatrixRoute`, `PhaseTorchExportShapeMatrixRecord`, `PhaseTorchExportShapeMatrixResult`

**Functions:** `default_torch_export_shape_scenarios()`, `run_torch_export_shape_matrix()`

### `scpn_quantum_control.phase.torch_gradients`

Bounded Torch gradient execution and its direct runtime primitives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_gradients.py) · Public symbols: **3**

**Functions:** `torch_parameter_shift_value_and_grad()`, `torch_bounded_qnn_value_and_grad()`, `torch_autograd_qnn_value_and_grad()`

### `scpn_quantum_control.phase.torch_maturity`

Torch lowering declarations, cloud planning, and maturity aggregation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_maturity.py) · Public symbols: **4**

**Functions:** `run_torch_phase_qnode_lowering_matrix()`, `run_torch_ecosystem_maturity_audit()`, `plan_torch_cloud_validation_batch()`, `run_torch_maturity_audit()`

### `scpn_quantum_control.phase.torch_module_state`

State-dictionary utilities for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_module_state.py) · Public symbols: **6**

**Classes:** `PhaseTorchModuleStateTensorMismatch`, `PhaseTorchModuleStateValidationResult`, `PhaseTorchModuleStateRoute`, `PhaseTorchModuleStateAuditResult`

**Functions:** `validate_torch_bounded_qnn_state_dict()`, `run_torch_module_state_audit()`

### `scpn_quantum_control.phase.torch_qnode_transforms`

Native Torch execution and compiler diagnostics for registered Phase-QNodes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_qnode_transforms.py) · Public symbols: **4**

**Functions:** `torch_phase_qnode_value_and_grad()`, `torch_phase_qnode_transform_audit()`, `torch_phase_qnode_compile_audit()`, `torch_phase_qnode_compile_boundary_audit()`

### `scpn_quantum_control.phase.torch_training_loop_matrix`

Training-loop matrix utilities for bounded PyTorch phase-QNN modules.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/torch_training_loop_matrix.py) · Public symbols: **6**

**Classes:** `PhaseTorchTrainingLoopScenario`, `PhaseTorchTrainingLoopMatrixRoute`, `PhaseTorchTrainingLoopMatrixRecord`, `PhaseTorchTrainingLoopMatrixResult`

**Functions:** `default_torch_training_loop_scenarios()`, `run_torch_training_loop_matrix()`

### `scpn_quantum_control.phase.trainability`

Barren-plateau diagnostics and finite-shot dry-run planning.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/trainability.py) · Public symbols: **4**

**Classes:** `TrainabilityGradientSample`, `AdaptiveShotAllocationDryRun`, `BarrenPlateauTrainabilityReport`

**Functions:** `run_barren_plateau_trainability_report()`

### `scpn_quantum_control.phase.transform_nesting`

Fail-closed transform-nesting planner for quantum gradients.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/transform_nesting.py) · Public symbols: **5**

**Classes:** `GradientTransformNestingPlan`, `GradientTransformNestingAuditResult`

**Functions:** `plan_gradient_transform_nesting()`, `assert_gradient_transform_nesting_supported()`, `run_gradient_transform_nesting_audit()`

### `scpn_quantum_control.phase.trotter_error`

Trotter error analysis for the XY Kuramoto Hamiltonian.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/trotter_error.py) · Public symbols: **7**

**Functions:** `trotter_error_norm()`, `trotter_error_sweep()`, `commutator_norm_bound()`, `nested_commutator_norm_bound()`, `trotter_error_bound()`, `optimal_dt()`, `frequency_heterogeneity()`

### `scpn_quantum_control.phase.trotter_upde`

Quantum 16-layer UPDE solver: multi-site spin chain.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/trotter_upde.py) · Public symbols: **3**

**Classes:** `UPDEStepResult`, `UPDETrajectoryResult`, `QuantumUPDESolver`

### `scpn_quantum_control.phase.variational_metric`

Exact variational-dynamics linear system for a fixed parametrised ansatz.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/variational_metric.py) · Public symbols: **5**

**Functions:** `assert_single_parameter_rotations()`, `analytic_state_derivatives()`, `mclachlan_metric()`, `real_time_force()`, `imaginary_time_force()`

### `scpn_quantum_control.phase.varqite`

Variational Quantum Imaginary Time Evolution (VarQITE).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/varqite.py) · Public symbols: **2**

**Classes:** `VarQITEResult`

**Functions:** `varqite_ground_state()`

### `scpn_quantum_control.phase.xy_compiler`

Domain-specific circuit compiler for XY Hamiltonian evolution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/xy_compiler.py) · Public symbols: **3**

**Functions:** `xy_gate()`, `compile_xy_trotter()`, `depth_comparison()`

### `scpn_quantum_control.phase.xy_kuramoto`

Quantum Kuramoto solver via XY spin Hamiltonian + Trotter evolution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase/xy_kuramoto.py) · Public symbols: **2**

**Classes:** `TrotterEvolutionConfig`, `QuantumKuramotoSolver`

## `psi_field`

### `scpn_quantum_control.psi_field.infoton`

Infoton: scalar field coupled to the U(1) gauge via covariant derivative.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/psi_field/infoton.py) · Public symbols: **4**

**Classes:** `InfitonField`

**Functions:** `gauge_covariant_kinetic()`, `matter_action()`, `create_infoton()`

### `scpn_quantum_control.psi_field.lattice`

Compact U(1) lattice gauge theory on arbitrary graph topologies.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/psi_field/lattice.py) · Public symbols: **3**

**Classes:** `PlaquetteResult`, `U1LatticGauge`

**Functions:** `hmc_update()`

### `scpn_quantum_control.psi_field.observables`

Gauge-invariant observables for the U(1) Ψ-field lattice.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/psi_field/observables.py) · Public symbols: **4**

**Functions:** `polyakov_loop()`, `topological_charge()`, `string_tension_from_wilson()`, `average_link()`

### `scpn_quantum_control.psi_field.scpn_mapping`

Map the 15+1 SCPN layer hierarchy onto a lattice gauge topology.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/psi_field/scpn_mapping.py) · Public symbols: **2**

**Classes:** `SCPNLattice`

**Functions:** `scpn_to_lattice()`

## `qec`

### `scpn_quantum_control.qec.biological_cli`

CLI utilities for biological surface-code execution artefacts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/biological_cli.py) · Public symbols: **2**

**Functions:** `build_parser()`, `main()`

### `scpn_quantum_control.qec.biological_diagnostics`

Biology-oriented topology diagnostics for Biological Surface Code graphs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/biological_diagnostics.py) · Public symbols: **2**

**Classes:** `BiologicalSurfaceDiagnostics`

**Functions:** `analyse_biological_surface_code()`

### `scpn_quantum_control.qec.biological_pipeline`

End-to-end biological surface-code pipeline helpers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/biological_pipeline.py) · Public symbols: **4**

**Classes:** `BiologicalQecExecution`, `BiologicalQecBatchExecution`

**Functions:** `run_biological_qec_execution()`, `run_biological_qec_batch_execution()`

### `scpn_quantum_control.qec.biological_surface_code`

Topological Quantum Error Correction: Biological Surface Code.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/biological_surface_code.py) · Public symbols: **2**

**Classes:** `BiologicalSurfaceCode`, `BiologicalMWPMDecoder`

### `scpn_quantum_control.qec.control_qec`

QEC for quantum control signals using a toric surface code and MWPM decoder.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/control_qec.py) · Public symbols: **3**

**Classes:** `SurfaceCode`, `MWPMDecoder`, `ControlQEC`

### `scpn_quantum_control.qec.dla_protected_scar`

DLA-protected scar-memory prototypes with finite-time revivals.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/dla_protected_scar.py) · Public symbols: **6**

**Classes:** `DLAProtectedScarSpec`, `DLAProtectedScarPrototype`, `DLAProtectedScarSimulationResult`

**Functions:** `build_dla_protected_scar_prototype()`, `simulate_dla_protected_scar_memory()`, `evaluate_dla_protected_scar_counts()`

### `scpn_quantum_control.qec.dla_protected_subspace`

DLA-protected logical synchronisation subspaces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/dla_protected_subspace.py) · Public symbols: **11**

**Classes:** `DLAProtectedSubspaceSpec`, `DLAProtectionCertificate`, `DLAProtectedMemoryPrototype`, `DLAProtectedWitnessResult`, `DLAProtectedLogicalSyncWitness`

**Functions:** `certify_dla_protected_subspace()`, `build_dla_protected_memory_prototype()`, `protected_memory_mask()`, `sync_memory_mask()`, `protected_logical_words()`, `evaluate_dla_protected_memory()`

### `scpn_quantum_control.qec.error_budget`

Surface code error budget for Kuramoto-XY simulation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/error_budget.py) · Public symbols: **5**

**Classes:** `ErrorBudget`

**Functions:** `logical_error_rate()`, `minimum_code_distance()`, `compute_error_budget()`, `compare_error_budgets()`

### `scpn_quantum_control.qec.fault_tolerant`

Repetition-code UPDE simulation with bit-flip protected logical qubits.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/fault_tolerant.py) · Public symbols: **2**

**Classes:** `LogicalQubit`, `RepetitionCodeUPDE`

### `scpn_quantum_control.qec.logical_dla_parity`

Logical-level DLA parity resource roadmap.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/logical_dla_parity.py) · Public symbols: **9**

**Classes:** `LogicalDLAParityRow`, `MultiscaleComparison`

**Functions:** `surface_code_physical_qubits()`, `repetition_scaffold_physical_qubits()`, `estimate_logical_dla_parity_row()`, `estimate_s7_resource_table()`, `compare_flat_surface_code_to_multiscale()`, `logical_dla_parity_payload()`, `logical_dla_parity_markdown()`

### `scpn_quantum_control.qec.multiscale_qec`

Hierarchical QEC across SCPN layers via concatenated surface codes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/multiscale_qec.py) · Public symbols: **5**

**Classes:** `QECLevel`, `MultiscaleQECResult`

**Functions:** `knm_between_domains()`, `concatenated_logical_rate()`, `build_multiscale_qec()`

### `scpn_quantum_control.qec.surface_code_upde`

Build a structural surface-code UPDE circuit and resource scaffold.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/surface_code_upde.py) · Public symbols: **2**

**Classes:** `SurfaceCodeSpec`, `SurfaceCodeUPDE`

### `scpn_quantum_control.qec.syndrome_flow`

Syndrome information flow between MS-QEC levels.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qec/syndrome_flow.py) · Public symbols: **3**

**Classes:** `SyndromeFlow`

**Functions:** `syndrome_flow_between_levels()`, `syndrome_flow_analysis()`

## `qsnn`

### `scpn_quantum_control.qsnn.dynamic_coupling`

Dynamic Quantum-Classical Co-Evolution (Quantum Hebbian Learning).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/dynamic_coupling.py) · Public symbols: **1**

**Classes:** `DynamicCouplingEngine`

### `scpn_quantum_control.qsnn.qlayer`

Quantum dense layer: multi-qubit entangled spiking network.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/qlayer.py) · Public symbols: **1**

**Classes:** `QuantumDenseLayer`

### `scpn_quantum_control.qsnn.qlif`

Quantum LIF neuron: Ry rotation + Z-basis measurement.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/qlif.py) · Public symbols: **1**

**Classes:** `QuantumLIFNeuron`

### `scpn_quantum_control.qsnn.qstdp`

Quantum STDP learning via the parameter-shift rule.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/qstdp.py) · Public symbols: **1**

**Classes:** `QuantumSTDP`

### `scpn_quantum_control.qsnn.qsynapse`

Quantum synapse: controlled-Ry gate.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/qsynapse.py) · Public symbols: **1**

**Classes:** `QuantumSynapse`

### `scpn_quantum_control.qsnn.quantum_neuromorphic_bridge`

Quantum neuromorphic bridge for QSNN experiments.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/quantum_neuromorphic_bridge.py) · Public symbols: **7**

**Classes:** `RecurrentCouplingPolicy`, `QuantumLIFConfig`, `TraceSTDPConfig`, `DynamicCouplingConfig`, `TraceSTDPState`, `NeuromorphicStepResult`, `QuantumNeuromorphicBridge`

### `scpn_quantum_control.qsnn.training`

Parameter-shift gradient training for QuantumDenseLayer.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qsnn/training.py) · Public symbols: **4**

**Classes:** `QSNNTrainingDiagnostics`, `QSNNTrainingRun`, `QSNNParameterShiftDescentRun`, `QSNNTrainer`

## `sensing`

### `scpn_quantum_control.sensing.nv_magnetometry_20T`

Nitrogen-vacancy (NV) centre magnetometry response model into the 20 T regime.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/sensing/nv_magnetometry_20T.py) · Public symbols: **9**

**Classes:** `NVCenter`, `NVFieldCalibration`

**Functions:** `nv_ground_state_hamiltonian()`, `nv_energy_levels_hz()`, `odmr_resonances_hz()`, `cw_odmr_dc_sensitivity_t_per_sqrt_hz()`, `odmr_spectrum()`, `simulate_odmr_measurement()`, `calibrate_field_from_odmr()`

## `ssgf`

### `scpn_quantum_control.ssgf.quantum_costs`

Quantum cost terms for SSGF integration.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ssgf/quantum_costs.py) · Public symbols: **5**

**Classes:** `QuantumCosts`

**Functions:** `compute_c_micro()`, `compute_c4_tcbo()`, `compute_c_pgbo()`, `compute_quantum_costs()`

### `scpn_quantum_control.ssgf.quantum_gradient`

SSGF quantum gradient: dC_quantum/dz via parameter-shift rule.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ssgf/quantum_gradient.py) · Public symbols: **3**

**Classes:** `QuantumGradientResult`

**Functions:** `quantum_cost()`, `compute_quantum_gradient()`

### `scpn_quantum_control.ssgf.quantum_outer_cycle`

Quantum SSGF outer cycle: variational geometry descent via quantum cost.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ssgf/quantum_outer_cycle.py) · Public symbols: **3**

**Classes:** `OuterCycleResult`

**Functions:** `classical_cost()`, `quantum_outer_cycle()`

### `scpn_quantum_control.ssgf.quantum_spectral`

Spectral bridge: Fiedler value from quantum phase estimation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ssgf/quantum_spectral.py) · Public symbols: **6**

**Classes:** `SpectralBridgeResult`

**Functions:** `laplacian_spectrum()`, `entrainment_criterion()`, `qpe_resource_estimate()`, `spectral_bridge_analysis()`, `spectral_bridge_vs_coupling()`

## `studio`

### `scpn_quantum_control.studio.benchmark_databank_bundle`

Federate the committed native-speedup benchmark rows as a schema-B bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/benchmark_databank_bundle.py) · Public symbols: **2**

**Functions:** `build_benchmark_databank_bundle()`, `main()`

### `scpn_quantum_control.studio.coupling_invariant`

Emit the effective-coupling invariant as a schema-B evidence bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/coupling_invariant.py) · Public symbols: **6**

**Classes:** `CouplingInvariantSource`, `CouplingInvariantPayload`

**Functions:** `build_coupling_invariant_payload()`, `validate_coupling_invariant_payload()`, `build_coupling_invariant_bundle()`, `main()`

### `scpn_quantum_control.studio.coverage_frontier`

Coverage-frontier report for QUANTUM's differentiable claim ledger.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/coverage_frontier.py) · Public symbols: **5**

**Classes:** `CoverageFrontierReport`

**Functions:** `map_claim_status()`, `measure_coverage_frontier()`, `measure_coverage_frontier_from_certifications()`, `render_coverage_frontier_markdown()`

### `scpn_quantum_control.studio.evidence_bundle`

Emit schema-B STUDIO evidence bundles from QUANTUM ledgers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/evidence_bundle.py) · Public symbols: **8**

**Classes:** `StudioBundleValidation`

**Functions:** `evidence_axes()`, `build_claim_ledger_bundle()`, `build_claim_ledger_bundles()`, `build_hardware_result_pack_bundle()`, `build_hardware_result_pack_bundles()`, `validate_bundle()`, `validate_bundles()`

### `scpn_quantum_control.studio.executive`

Executive action spine for the standalone SCPN-QUANTUM-CONTROL studio.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive.py) · Public symbols: **12**

**Classes:** `VerbContract`, `ExecutiveRequest`, `ExecutionPlan`, `ExecutionResult`, `GeneratedScript`, `ExecutiveRecord`, `ActionHandler`, `ActionRegistry`

**Functions:** `resolve_verb_contract()`, `build_generated_script()`, `preview_action()`, `run_action()`

### `scpn_quantum_control.studio.executive_analyse`

The ``analyse`` executive action handler — synchronisation witness of a phase cloud.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_analyse.py) · Public symbols: **1**

**Classes:** `AnalyseActionHandler`

### `scpn_quantum_control.studio.executive_benchmark`

The ``benchmark`` executive action handler — native construction speedup.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_benchmark.py) · Public symbols: **4**

**Classes:** `BenchmarkActionHandler`

**Functions:** `reference_dense_xy_hamiltonian()`, `native_dense_xy_hamiltonian()`, `measure_p50_us()`

### `scpn_quantum_control.studio.executive_cli`

The ``scpn-studio-run`` executive dispatch command line interface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_cli.py) · Public symbols: **3**

**Functions:** `build_default_registry()`, `run()`, `main()`

### `scpn_quantum_control.studio.executive_compile`

The ``compile`` executive action handler — bounded XY compile of a network.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_compile.py) · Public symbols: **1**

**Classes:** `CompileActionHandler`

### `scpn_quantum_control.studio.executive_differentiate`

The ``differentiate`` executive action handler — the first spine plugin.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_differentiate.py) · Public symbols: **3**

**Classes:** `DifferentiateActionHandler`

**Functions:** `build_effect_ir()`, `default_registry()`

### `scpn_quantum_control.studio.executive_execute`

The ``execute`` executive action handler — approval-gated QPU deployment.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_execute.py) · Public symbols: **1**

**Classes:** `ExecuteActionHandler`

### `scpn_quantum_control.studio.executive_mitigate`

The ``mitigate`` executive action handler — zero-noise extrapolation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_mitigate.py) · Public symbols: **1**

**Classes:** `MitigateActionHandler`

### `scpn_quantum_control.studio.executive_replay`

The ``replay`` executive action handler — hardware-result-pack re-verification.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_replay.py) · Public symbols: **1**

**Classes:** `ReplayActionHandler`

### `scpn_quantum_control.studio.executive_simulate`

The ``simulate`` executive action handler — bounded XY-Kuramoto evolution.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_simulate.py) · Public symbols: **1**

**Classes:** `SimulateActionHandler`

### `scpn_quantum_control.studio.executive_validate`

The ``validate`` executive action handler — claim-ledger reference validation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/executive_validate.py) · Public symbols: **1**

**Classes:** `ValidateActionHandler`

### `scpn_quantum_control.studio.federation`

The QUANTUM studio's federation document for STUDIO/Hub ingestion.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/federation.py) · Public symbols: **5**

**Functions:** `build_architecture_map_extension()`, `build_federation_document()`, `write_federation_document()`, `studio_manifest_drift()`, `main()`

### `scpn_quantum_control.studio.kuramoto_reference`

Python reference for the WASM Kuramoto live simulator.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/kuramoto_reference.py) · Public symbols: **5**

**Classes:** `KuramotoRun`

**Functions:** `encode_kuramoto_input()`, `order_parameter()`, `simulate()`, `decode_output()`

### `scpn_quantum_control.studio.kuramoto_scenario_artifact`

Committed-artefact emission for the WASM Kuramoto Play panel's ground truth.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/kuramoto_scenario_artifact.py) · Public symbols: **3**

**Functions:** `build_kuramoto_scenario_artifact()`, `validate_kuramoto_scenario_artifact()`, `main()`

### `scpn_quantum_control.studio.manifest`

The QUANTUM studio's capability manifest (schema A) on the platform contract.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/manifest.py) · Public symbols: **2**

**Functions:** `declared_surface()`, `build_manifest()`

### `scpn_quantum_control.studio.program_ad_replay_artifact`

Committed-artefact emission for the browser-verifiable program-AD gradient replay.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/program_ad_replay_artifact.py) · Public symbols: **6**

**Classes:** `ProgramADReplayArtifactValidation`

**Functions:** `encode_replay_input()`, `build_program_ad_replay_artifact()`, `inspect_program_ad_replay_artifact()`, `validate_program_ad_replay_artifact()`, `main()`

### `scpn_quantum_control.studio.qec_readiness_bundle`

Federate the committed offline QEC-readiness decision as a schema-B bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/qec_readiness_bundle.py) · Public symbols: **2**

**Functions:** `build_qec_readiness_bundle()`, `main()`

### `scpn_quantum_control.studio.qpu_result_pack`

Emit and present the attestation-verifiable ``studio.qpu-result-pack.v1`` unit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/qpu_result_pack.py) · Public symbols: **4**

**Classes:** `QpuResultPackPresentation`

**Functions:** `build_qpu_result_pack_unit()`, `present_qpu_result_pack()`, `seal_qpu_result_pack()`

### `scpn_quantum_control.studio.readout_mitigation_bundle`

Federate the committed readout-mitigation evidence as a schema-B bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/readout_mitigation_bundle.py) · Public symbols: **2**

**Functions:** `build_readout_mitigation_bundle()`, `main()`

### `scpn_quantum_control.studio.recompute_kernel`

Recompute-verifiable Studio units for deterministic compile claims.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/recompute_kernel.py) · Public symbols: **7**

**Classes:** `DecodedXYCompileInput`, `XYCompileRecomputeUnit`

**Functions:** `canonical_xy_compile_input_bytes()`, `decode_xy_compile_input_bytes()`, `xy_compile_digest_python()`, `build_xy_compile_recompute_unit()`, `verify_xy_compile_recompute_unit()`

### `scpn_quantum_control.studio.reference_validation`

Per-claim reference-validation certifications for the Studio frontier.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/reference_validation.py) · Public symbols: **4**

**Classes:** `ReferenceValidationCertification`, `ReferenceValidationRegistryValidation`, `ReferenceValidationRegistry`

**Functions:** `load_reference_validation_registry()`

### `scpn_quantum_control.studio.result_pack_seal`

Seal a QUANTUM hardware result pack into a verifiable honesty envelope (WS-1).

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/result_pack_seal.py) · Public symbols: **3**

**Functions:** `build_result_pack_unit()`, `build_provider_attestation()`, `seal_result_pack()`

### `scpn_quantum_control.studio.scorecard_bundle`

Emit the differentiable baseline scorecard as a schema-B evidence bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/scorecard_bundle.py) · Public symbols: **2**

**Functions:** `build_scorecard_bundle()`, `main()`

### `scpn_quantum_control.studio.support_matrix_bundle`

Emit the transform-algebra support matrix as a schema-B evidence bundle.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/support_matrix_bundle.py) · Public symbols: **2**

**Functions:** `build_support_matrix_bundle()`, `main()`

### `scpn_quantum_control.studio.verbs`

The SCPN-QUANTUM-CONTROL studio's verbs, on the locked platform contract.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/verbs.py) · Public symbols: **2**

**Functions:** `evidence_schemas()`, `verb_substrates()`

### `scpn_quantum_control.studio.xy_compile_recompute_artifact`

Committed-artefact emission for a browser-verifiable XY-compile recompute unit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio/xy_compile_recompute_artifact.py) · Public symbols: **3**

**Functions:** `build_xy_compile_recompute_artifact()`, `validate_xy_compile_recompute_artifact()`, `main()`

## `surrogates`

### `scpn_quantum_control.surrogates.fidelity`

Held-out value and gradient fidelity gates for classical surrogates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/surrogates/fidelity.py) · Public symbols: **5**

**Classes:** `SurrogateFidelityThresholds`, `SurrogateFidelityCertificate`, `SurrogateGradientCertificate`

**Functions:** `certify_surrogate_fidelity()`, `certify_surrogate_gradient()`

### `scpn_quantum_control.surrogates.hybrid`

co-design proposal composition with mandatory exact local validation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/surrogates/hybrid.py) · Public symbols: **2**

**Classes:** `ExactValidatedSurrogateProposal`

**Functions:** `propose_and_validate_surrogate_step()`

### `scpn_quantum_control.surrogates.models`

Smooth classical surrogates for bounded quantum-objective studies.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/surrogates/models.py) · Public symbols: **1**

**Classes:** `GaussianRBFSurrogate`

### `scpn_quantum_control.surrogates.report`

Deterministic JSON and Markdown evidence for the quantum-reservoir product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/surrogates/report.py) · Public symbols: **4**

**Classes:** `SurrogateSupportRow`, `QuantumReservoirSurrogateEvidence`

**Functions:** `render_quantum_reservoir_surrogate_markdown()`, `write_quantum_reservoir_surrogate_evidence()`

### `scpn_quantum_control.surrogates.train`

Deterministic fitting for differentiable Gaussian-RBF surrogates.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/surrogates/train.py) · Public symbols: **3**

**Classes:** `SurrogateFitConfig`

**Functions:** `input_row_digests()`, `fit_gaussian_rbf_surrogate()`

## `tcbo`

### `scpn_quantum_control.tcbo.quantum_observer`

Compute small-system TCBO proxy diagnostics from exact ground states.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/tcbo/quantum_observer.py) · Public symbols: **2**

**Classes:** `TCBOResult`

**Functions:** `compute_tcbo_observables()`

## `thermodynamics`

### `scpn_quantum_control.thermodynamics.readiness`

No-submit S9 quantum-thermodynamics readiness model.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/thermodynamics/readiness.py) · Public symbols: **14**

**Classes:** `EntropyProductionRate`, `CalibratedWorkIdentity`, `IrreversibilityResidual`, `HeatDissipationRate`, `ThermodynamicSweepConfig`, `ThermodynamicSweepRow`, `ThermodynamicSweepResult`

**Functions:** `entropy_production_rate()`, `calibrated_work_identity()`, `irreversibility_residual()`, `heat_dissipation_rate()`, `run_k_sweep_protocol()`, `quantum_thermo_payload()`, `quantum_thermo_markdown()`

## `top_level`

### `scpn_quantum_control._constants`

Shared numerical constants for scpn-quantum-control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/_constants.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control._paths`

Locate repository-scoped data files from source and installed layouts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/_paths.py) · Public symbols: **2**

**Functions:** `project_data_root()`, `project_data_path()`

### `scpn_quantum_control._rust_accel`

Single resilient entry point to the optional :mod:`oscillatools` Rust engine.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/_rust_accel.py) · Public symbols: **1**

**Functions:** `optional_rust_engine()`

### `scpn_quantum_control.active_sensing_product`

Policy-bounded active sensing over existing sensing and S3 surfaces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/active_sensing_product.py) · Public symbols: **9**

**Classes:** `SensingInventoryRow`, `InformationGainCandidate`, `InformationGainScore`, `ActiveSensingObserverRecord`, `ActiveSensingPlan`

**Functions:** `sensing_surface_inventory()`, `score_expected_information_gain()`, `plan_active_sensing()`, `demo_information_gain_candidates()`

### `scpn_quantum_control.adjoint_replay_product`

Fail-closed **adjoint differentiation via reversible replay** product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/adjoint_replay_product.py) · Public symbols: **15**

**Classes:** `AdjointReplaySurfaceRow`, `CheckpointPolicy`, `ReversibilityReport`, `PathEligibilityDecision`, `MaterialisedAdjointReplayProbe`

**Functions:** `list_adjoint_replay_surface_ids()`, `get_adjoint_replay_surface()`, `iter_adjoint_replay_surfaces()`, `build_checkpoint_policy()`, `assess_reversibility()`, `decide_adjoint_replay_path()`, `materialise_demo_adjoint_replay_probe()`, `map_adjoint_replay_public_surfaces()`, `build_adjoint_replay_product_registry()`, `assert_adjoint_replay_product_integrity()`

### `scpn_quantum_control.advanced_witnesses_product`

Fail-closed **advanced witnesses** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/advanced_witnesses_product.py) · Public symbols: **27**

**Classes:** `WitnessCapabilityRow`, `WitnessBoundaryRow`, `WitnessEstimate`, `PathEligibilityDecision`, `MaterialisedKrylovProbe`, `MaterialisedOtocProbe`, `MaterialisedShadowProbe`

**Functions:** `iter_witness_capabilities()`, `list_witness_capability_ids()`, `get_witness_capability()`, `iter_witness_boundaries()`, `list_witness_boundary_ids()`, `get_witness_boundary()`, `list_witness_glossary_keys()`, `get_witness_glossary_entry()`, `list_witness_ambient_inventory()`, `map_advanced_witnesses_public_surfaces()`, `decide_witness_path()`, `materialise_krylov_probe()`, `materialise_demo_krylov_probe()`, `materialise_otoc_probe()`, `materialise_demo_otoc_probe()`, `materialise_shadow_probe()`, `materialise_demo_shadow_probe()`, `materialise_harmonic_order_parameter_compose()`, `build_advanced_witnesses_product_registry()`, `assert_advanced_witnesses_product_integrity()`

### `scpn_quantum_control.advantage_language_protocol`

Fail-closed advantage-language governance and protocol catalogue.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/advantage_language_protocol.py) · Public symbols: **11**

**Classes:** `AdvantageProtocolRecord`, `NoAdvantageCertificate`, `AdvantageLanguageProbeResult`

**Functions:** `list_advantage_protocol_ids()`, `get_advantage_protocol()`, `iter_advantage_protocols()`, `build_advantage_language_registry()`, `issue_no_advantage_certificate()`, `find_advantage_language_triggers()`, `probe_advantage_language()`, `assert_advantage_language_registry_integrity()`

### `scpn_quantum_control.attested_result_pack`

Strip-resistant attested result-pack digests and local envelopes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/attested_result_pack.py) · Public symbols: **9**

**Classes:** `AttestedEnvelope`, `AttestationVerdict`

**Functions:** `canonical_content_digest()`, `build_unsigned_envelope()`, `verify_attested_envelope()`, `refuse_invent_green_hardware_attestation()`, `build_attestation_report()`, `default_claim_axes()`, `envelope_from_mapping()`

### `scpn_quantum_control.backend_dispatch`

Runtime backend selection for array operations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/backend_dispatch.py) · Public symbols: **6**

**Functions:** `set_backend()`, `get_backend()`, `get_array_module()`, `to_numpy()`, `from_numpy()`, `available_backends()`

### `scpn_quantum_control.bench_cli`

One-command benchmark artefact regeneration for the methods papers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/bench_cli.py) · Public symbols: **4**

**Classes:** `ExecutionSurfacePolicy`, `Harness`

**Functions:** `run()`, `main()`

### `scpn_quantum_control.campaign_harness_product`

Fail-closed **campaign harness productisation** surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/campaign_harness_product.py) · Public symbols: **15**

**Classes:** `CampaignHarnessRow`, `PathEligibilityDecision`, `MaterialisedCampaignProbe`

**Functions:** `list_campaign_harness_ids()`, `get_campaign_harness()`, `iter_campaign_harnesses()`, `list_ambient_benchmark_family_ids()`, `decide_campaign_path()`, `materialise_appqsim_probe()`, `materialise_iqm_layout_probe()`, `materialise_closed_loop_probe()`, `materialise_demo_campaign_probe()`, `map_campaign_harness_public_surfaces()`, `build_campaign_harness_product_registry()`, `assert_campaign_harness_product_integrity()`

### `scpn_quantum_control.circuit_cutting_product`

Fail-closed circuit-cutting product for large synchronisation workloads.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/circuit_cutting_product.py) · Public symbols: **12**

**Classes:** `CuttingSurfaceRow`, `CuttingResourceCertificate`, `SyntheticReconstructionCertificate`, `CuttingPathDecision`

**Functions:** `list_cutting_surface_ids()`, `get_cutting_surface()`, `iter_cutting_surfaces()`, `build_cutting_resource_certificate()`, `certify_synthetic_reconstruction()`, `decide_cutting_path()`, `build_circuit_cutting_product_registry()`, `assert_circuit_cutting_product_integrity()`

### `scpn_quantum_control.cloud_native_deployment_product`

Fail-closed **cloud-native deployment boundary** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/cloud_native_deployment_product.py) · Public symbols: **15**

**Classes:** `DeploymentPatternRow`, `ThreatModelRow`, `PathEligibilityDecision`, `MaterialisedDeployDryRunProbe`

**Functions:** `list_deployment_pattern_ids()`, `list_threat_ids()`, `get_deployment_pattern()`, `iter_deployment_patterns()`, `decide_deploy_path()`, `materialise_deploy_dry_run_probe()`, `materialise_demo_deploy_dry_run_probe()`, `map_cloud_native_deployment_public_surfaces()`, `build_cloud_native_deployment_product_registry()`, `assert_cloud_native_deployment_product_integrity()`, `compute_spec_digest()`

### `scpn_quantum_control.competitive_baseline_watch`

Fail-closed continuous competitive-baseline watch surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/competitive_baseline_watch.py) · Public symbols: **10**

**Classes:** `CompetitiveWatchRecord`, `RefreshProbeResult`, `FeedProbeResult`

**Functions:** `list_competitor_ids()`, `get_competitive_watch()`, `iter_competitive_watch()`, `probe_refresh()`, `probe_feed()`, `build_competitive_baseline_watch_registry()`, `assert_competitive_baseline_watch_integrity()`

### `scpn_quantum_control.compile_budget`

Resource guards for sparse Pauli-operator construction in the Kuramoto compiler.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compile_budget.py) · Public symbols: **6**

**Classes:** `PauliOperatorBudgetError`, `PauliOperatorEstimate`

**Functions:** `pauli_term_upper_bound()`, `pauli_budget_bytes()`, `estimate_pauli_operator()`, `require_pauli_operator_budget()`

### `scpn_quantum_control.compiler_boundary_product`

Fail-closed **external compiler boundary register** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/compiler_boundary_product.py) · Public symbols: **12**

**Classes:** `CompilerBoundaryRow`, `PathEligibilityDecision`, `MaterialisedCompilerBoundaryProbe`

**Functions:** `list_compiler_ids()`, `get_compiler_boundary()`, `iter_compiler_boundaries()`, `decide_compiler_path()`, `materialise_compiler_boundary_probe()`, `materialise_demo_compiler_boundary_probe()`, `map_compiler_boundary_public_surfaces()`, `build_compiler_boundary_product_registry()`, `assert_compiler_boundary_product_integrity()`

### `scpn_quantum_control.config`

Single source of truth for runtime configuration.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/config.py) · Public symbols: **3**

**Classes:** `SCPNConfig`

**Functions:** `get_config()`, `reload_config()`

### `scpn_quantum_control.control_stack_compose_product`

Fail-closed **compose existing control/*** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control_stack_compose_product.py) · Public symbols: **15**

**Classes:** `OwnershipRow`, `AdapterPortRow`, `PathEligibilityDecision`, `MaterialisedClosedLoopTelemetryProbe`

**Functions:** `list_ownership_module_ids()`, `list_adapter_port_ids()`, `get_ownership_row()`, `get_adapter_port()`, `iter_ownership_rows()`, `decide_control_compose_path()`, `materialise_closed_loop_telemetry_probe()`, `materialise_demo_closed_loop_telemetry_probe()`, `map_control_stack_compose_public_surfaces()`, `build_control_stack_compose_product_registry()`, `assert_control_stack_compose_product_integrity()`

### `scpn_quantum_control.control_stack_runtime_adapters`

Executable control-stack adapters over existing control and co-simulation surfaces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/control_stack_runtime_adapters.py) · Public symbols: **12**

**Classes:** `PolicyGatedAdapterError`, `RealtimeFeedbackPort`, `QaoaMpcPort`, `RealtimeFeedbackAdapterResult`, `QaoaMpcAdapterResult`, `CosimulationPartitionTelemetry`, `CosimulationPartitionAdapterResult`, `PulseComposeBoundaryDecision`

**Functions:** `run_realtime_feedback_adapter()`, `run_qaoa_mpc_adapter()`, `run_cosimulation_partition_adapter()`, `decide_pulse_compose_boundary()`

### `scpn_quantum_control.custom_derivatives_product`

Fail-closed **custom / registered derivatives** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/custom_derivatives_product.py) · Public symbols: **16**

**Classes:** `CustomDerivativeContractRow`, `RegistrationResult`

**Functions:** `list_custom_derivative_contract_ids()`, `get_custom_derivative_contract()`, `iter_custom_derivative_contracts()`, `registration_contract_policy()`, `parse_product_identity()`, `build_example_scaled_linear_rule()`, `new_product_registry()`, `register_product_custom_rule()`, `require_product_custom_rule()`, `list_product_registered_identities()`, `probe_example_rule_round_trip()`, `map_custom_derivatives_public_surfaces()`, `build_custom_derivatives_product_registry()`, `assert_custom_derivatives_product_integrity()`

### `scpn_quantum_control.dense_budget`

Memory guards for dense Hilbert-space allocations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dense_budget.py) · Public symbols: **9**

**Classes:** `DenseAllocationError`, `DenseAllocationEstimate`

**Functions:** `hilbert_dimension()`, `dense_object_bytes()`, `available_memory_bytes()`, `dense_budget_bytes()`, `estimate_dense_allocation()`, `require_dense_allocation()`, `require_dense_eigensolver_workspace()`

### `scpn_quantum_control.diff`

Canonical first-path namespace for differentiable quantum-control workflows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/diff.py) · Public symbols: **10**

**Classes:** `ShotPolicy`, `EstimatorProvenance`, `BackendCapabilityMetadata`, `DifferentiableCircuitDiagnostics`, `JITExplanation`, `DifferentiableCircuit`

**Functions:** `differentiable_circuit()`, `jit_or_explain()`, `supported_transforms()`, `namespace_metadata()`

### `scpn_quantum_control.diff_contract_audit`

Executable differentiable-contract contract audit for differentiable circuit facades.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/diff_contract_audit.py) · Public symbols: **3**

**Classes:** `DifferentiableCircuitContractCheck`, `DifferentiableCircuitContractAuditResult`

**Functions:** `run_differentiable_circuit_contract_audit()`

### `scpn_quantum_control.differentiable`

Native differentiable-programming primitives for SCPN quantum objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.differentiable_api`

Unified differentiable-programming facade over supported local routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_api.py) · Public symbols: **18**

**Functions:** `differentiable_value()`, `differentiable_gradient()`, `differentiable_jacobian()`, `differentiable_hessian()`, `differentiable_support_report()`, `explain_differentiability()`, `differentiable_compile_report()`, `differentiable_benchmark_report()`, `differentiable_baseline_scorecard_report()`, `differentiable_competitive_baseline_refresh_report()`, `differentiable_rust_python_inventory_report()`, `differentiable_architecture_map_report()`, `differentiable_dependency_environment_map_report()`, `differentiable_isolated_benchmark_plan_report()`, `differentiable_transform_algebra_report()`, `differentiable_qfi_fss_report()`, `differentiable_frontend_report()`, `differentiable_api()`

### `scpn_quantum_control.differentiable_api_contracts`

Immutable envelopes and type contracts for the unified differentiable API.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_api_contracts.py) · Public symbols: **4**

**Classes:** `UnifiedDifferentiableAPIResult`, `DifferentiabilityDiagnosticReport`, `DifferentiableDashboardCapabilityRow`, `DifferentiableDashboardStatus`

### `scpn_quantum_control.differentiable_architecture_map`

Architecture and Rustification map for differentiable-programming governance.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_architecture_map.py) · Public symbols: **6**

**Classes:** `DifferentiableArchitectureMapLayer`, `DifferentiableArchitectureMap`, `DifferentiableArchitectureMapValidation`

**Functions:** `run_differentiable_architecture_map()`, `validate_differentiable_architecture_map()`, `render_differentiable_architecture_map_markdown()`

### `scpn_quantum_control.differentiable_baseline_scorecard`

Baseline scorecard governance for differentiable computing claims.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_baseline_scorecard.py) · Public symbols: **8**

**Classes:** `DifferentiableBaselineScorecardRow`, `DifferentiableBaselineScorecard`, `DifferentiableBaselineScorecardValidation`, `DifferentiablePromotionLanguageAudit`

**Functions:** `run_differentiable_baseline_scorecard()`, `validate_differentiable_baseline_scorecard()`, `audit_differentiable_promotion_language()`, `render_differentiable_baseline_scorecard_markdown()`

### `scpn_quantum_control.differentiable_batch_helpers`

Batch and sample tensor validation helpers for native transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_batch_helpers.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.differentiable_benchmark_report`

Claim-bounded local differentiable benchmark report builders.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_benchmark_report.py) · Public symbols: **2**

**Classes:** `DifferentiableBenchmarkReport`

**Functions:** `build_differentiable_benchmark_report()`

### `scpn_quantum_control.differentiable_canonical_api`

Canonical differentiable transform dispatchers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_canonical_api.py) · Public symbols: **2**

**Functions:** `value_and_grad()`, `grad()`

### `scpn_quantum_control.differentiable_claim_ledger`

Claim ledger for bounded differentiable Phase-QNode evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_claim_ledger.py) · Public symbols: **10**

**Classes:** `ClaimLedgerRow`, `ClaimLedger`, `ClaimLedgerValidation`, `DifferentiableSupportSurfaceAlignment`

**Functions:** `load_differentiable_claim_ledger()`, `load_differentiable_support_surface_alignment()`, `validate_claim_ledger()`, `validate_public_language_against_ledger()`, `validate_differentiable_support_surface_alignment()`, `validate_public_claim_table()`

### `scpn_quantum_control.differentiable_claim_rendering`

Render bounded differentiable claim evidence as deterministic Markdown.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_claim_rendering.py) · Public symbols: **3**

**Functions:** `render_claim_ledger_markdown()`, `render_differentiable_support_surface_alignment_markdown()`, `render_public_claim_table()`

### `scpn_quantum_control.differentiable_competitive_baselines`

Freshness gate for differentiable-computing competitive baselines.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_competitive_baselines.py) · Public symbols: **9**

**Classes:** `CompetitiveBaselineRow`, `CompetitiveBaselineRefresh`, `CompetitiveBaselineValidation`, `CompetitiveBaselinePromotionGate`

**Functions:** `run_competitive_baseline_refresh()`, `load_competitive_baseline_refresh()`, `validate_competitive_baseline_refresh()`, `audit_competitive_baseline_promotion_gate()`, `render_competitive_baseline_refresh_markdown()`

### `scpn_quantum_control.differentiable_consistency`

Consistency diagnostics for native differentiable transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_consistency.py) · Public symbols: **2**

**Functions:** `check_parameter_shift_consistency()`, `check_custom_derivative_consistency()`

### `scpn_quantum_control.differentiable_custom_derivatives`

Exact custom JVP, VJP, and Jacobian transform wrappers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_custom_derivatives.py) · Public symbols: **12**

**Functions:** `custom_jvp()`, `batch_custom_jvp()`, `batch_value_and_custom_jvp()`, `value_and_custom_jvp()`, `custom_vjp()`, `batch_custom_vjp()`, `batch_value_and_custom_vjp()`, `value_and_custom_vjp()`, `custom_jacobian()`, `batch_custom_jacobian()`, `batch_value_and_custom_jacobian()`, `value_and_custom_jacobian()`

### `scpn_quantum_control.differentiable_dashboard`

Claim-bounded capability catalog for differentiable dashboard consumers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_dashboard.py) · Public symbols: **1**

**Functions:** `differentiable_dashboard_status()`

### `scpn_quantum_control.differentiable_dependency_environment_evidence`

Version-pin and execution-route evidence for differentiable environments.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_dependency_environment_evidence.py) · Public symbols: **2**

**Classes:** `DifferentiableDependencyEnvironmentEvidence`

**Functions:** `build_differentiable_dependency_environment_evidence()`

### `scpn_quantum_control.differentiable_dependency_environment_map`

Dependency and environment evidence map for differentiable-programming governance.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_dependency_environment_map.py) · Public symbols: **6**

**Classes:** `DifferentiableDependencyEnvironmentProfile`, `DifferentiableDependencyEnvironmentMap`, `DifferentiableDependencyEnvironmentMapValidation`

**Functions:** `run_differentiable_dependency_environment_map()`, `validate_differentiable_dependency_environment_map()`, `render_differentiable_dependency_environment_map_markdown()`

### `scpn_quantum_control.differentiable_exact_modes`

Exact scalar forward- and reverse-mode gradient wrappers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_exact_modes.py) · Public symbols: **4**

**Functions:** `value_and_forward_mode_grad()`, `forward_mode_gradient()`, `value_and_reverse_mode_grad()`, `reverse_mode_gradient()`

### `scpn_quantum_control.differentiable_external_validation`

External-validation package manifests for differentiable evidence.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_external_validation.py) · Public symbols: **15**

**Classes:** `EnvironmentLockfileSummary`, `ExternalValidationEnvironmentLock`, `ExternalValidationEnvironmentLockValidation`, `ExternalValidationArtifactEntry`, `ExternalValidationArtifactBundle`

**Functions:** `summarize_environment_lockfile()`, `summarize_artifact_entry()`, `build_external_validation_environment_lock()`, `build_external_validation_artifact_bundle()`, `load_external_validation_environment_lock()`, `load_external_validation_artifact_bundle()`, `validate_external_validation_environment_lock()`, `validate_external_validation_artifact_bundle()`, `render_external_validation_environment_lock_markdown()`, `render_external_validation_artifact_bundle_markdown()`

### `scpn_quantum_control.differentiable_finite_difference`

Finite-difference and complex-step diagnostic differentiable transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_finite_difference.py) · Public symbols: **37**

**Functions:** `finite_difference_gradient()`, `complex_step_gradient()`, `batch_complex_step_gradient()`, `batch_value_and_complex_step_grad()`, `value_and_jacobian()`, `jacobian()`, `value_and_jacfwd()`, `jacfwd()`, `value_and_jacrev()`, `jacrev()`, `value_and_hessian()`, `hessian()`, `batch_value_and_finite_difference_grad()`, `value_and_complex_step_grad()`, `value_and_finite_difference_grad()`, `finite_difference_jacobian()`, `value_and_finite_difference_jacobian()`, `finite_difference_jvp()`, `value_and_jvp()`, `jvp()`, `value_and_finite_difference_jvp()`, `batch_finite_difference_jvp()`, `batch_value_and_finite_difference_jvp()`, `vector_jacobian_product()`, `finite_difference_vjp()`, `value_and_finite_difference_vjp()`, `value_and_vjp()`, `vjp()`, `batch_vector_jacobian_product()`, `batch_finite_difference_vjp()`, `batch_value_and_finite_difference_vjp()`, `finite_difference_hessian()`, `value_and_finite_difference_hessian()`, `finite_difference_hvp()`, `value_and_finite_difference_hvp()`, `batch_finite_difference_hvp()`, `batch_value_and_finite_difference_hvp()`

### `scpn_quantum_control.differentiable_fisher`

Empirical-Fisher matrix-free solves for residual-map derivatives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_fisher.py) · Public symbols: **3**

**Functions:** `empirical_fisher_vector_product()`, `empirical_fisher_conjugate_gradient()`, `least_squares_covariance()`

### `scpn_quantum_control.differentiable_framework_overlay`

Reproducible CPU-only overlay profile for optional AD frameworks.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_framework_overlay.py) · Public symbols: **9**

**Classes:** `FrameworkOverlayManifest`, `FrameworkOverlayVerification`

**Functions:** `default_framework_overlay_path()`, `build_framework_overlay_manifest()`, `install_framework_overlay()`, `verify_framework_overlay_manifest()`, `verify_framework_overlay_path()`, `framework_overlay_pythonpath()`, `main()`

### `scpn_quantum_control.differentiable_gradient_descent`

Native gradient-descent optimizer for differentiable scalar objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_gradient_descent.py) · Public symbols: **1**

**Classes:** `DifferentiableOptimizer`

### `scpn_quantum_control.differentiable_implicit_sensitivity`

Implicit stationary and fixed-point sensitivity solvers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_implicit_sensitivity.py) · Public symbols: **2**

**Functions:** `implicit_stationary_sensitivity()`, `implicit_fixed_point_sensitivity()`

### `scpn_quantum_control.differentiable_jax_adapter`

Optional JAX autodiff adapter for native differentiable objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_jax_adapter.py) · Public symbols: **2**

**Functions:** `is_jax_autodiff_available()`, `jax_value_and_grad()`

### `scpn_quantum_control.differentiable_levenberg_marquardt`

Levenberg-Marquardt and Gauss-Newton residual optimization helpers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_levenberg_marquardt.py) · Public symbols: **7**

**Classes:** `LevenbergMarquardtOptimizer`

**Functions:** `gauss_newton_gradient()`, `custom_gauss_newton_gradient()`, `levenberg_marquardt_step()`, `custom_levenberg_marquardt_step()`, `evaluate_levenberg_marquardt_step()`, `update_levenberg_marquardt_damping()`

### `scpn_quantum_control.differentiable_module_hardening_audit`

Module coverage and diagnostic audit for differentiable-programming surfaces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_module_hardening_audit.py) · Public symbols: **4**

**Classes:** `DifferentiableModuleHardeningRecord`, `DifferentiableModuleHardeningAuditResult`

**Functions:** `differentiable_module_hardening_registry()`, `run_differentiable_module_hardening_audit()`

### `scpn_quantum_control.differentiable_natural_gradient`

Natural-gradient preconditioning and scalar line-search helpers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_natural_gradient.py) · Public symbols: **4**

**Classes:** `NaturalGradientOptimizer`

**Functions:** `armijo_backtracking_line_search()`, `weighted_gradient_sum()`, `natural_gradient()`

### `scpn_quantum_control.differentiable_parameter_contracts`

Parameter metadata and validation contracts for differentiable objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_parameter_contracts.py) · Public symbols: **4**

**Classes:** `Parameter`, `ParameterBounds`, `ParameterShiftRule`

**Functions:** `multi_frequency_parameter_shift_rule()`

### `scpn_quantum_control.differentiable_parameter_shift`

Parameter-shift transforms for scalar differentiable objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_parameter_shift.py) · Public symbols: **5**

**Functions:** `parameter_shift_gradient()`, `batch_parameter_shift_gradient()`, `batch_value_and_parameter_shift_grad()`, `value_and_parameter_shift_grad()`, `parameter_shift_gradient_with_uncertainty()`

### `scpn_quantum_control.differentiable_registered_custom`

Registry-backed custom derivative wrappers for native transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_registered_custom.py) · Public symbols: **3**

**Functions:** `registered_custom_jvp()`, `registered_custom_vjp()`, `registered_custom_jacobian()`

### `scpn_quantum_control.differentiable_residual_weights`

IRLS residual weighting helpers for differentiable least-squares paths.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_residual_weights.py) · Public symbols: **2**

**Functions:** `huber_residual_weights()`, `soft_l1_residual_weights()`

### `scpn_quantum_control.differentiable_result_contracts`

Validated result records for native differentiable-programming transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_result_contracts.py) · Public symbols: **32**

**Classes:** `GradientResult`, `FiniteShotSampleProvenance`, `ParameterShiftSampleRecord`, `StochasticGradientResult`, `SPSAObjectiveSample`, `SPSAProbeRecord`, `SPSAGradientResult`, `ScoreFunctionSampleRecord`, `ScoreFunctionGradientResult`, `ShotAllocationResult`, `OptimizationResult`, `ArmijoLineSearchResult`, `GradientCheckResult`, `CustomDerivativeCheckResult`, `JacobianResult`, `JVPResult`, `VJPResult`, `HessianResult`, `SparseMatrixResult`, `HVPResult`, `NaturalGradientResult`, `NaturalGradientOptimizationResult`, `LevenbergMarquardtStep`, `LevenbergMarquardtTrial`, `LevenbergMarquardtDampingUpdate`, `LevenbergMarquardtResult`, `LeastSquaresCovarianceResult`, `FisherVectorProductResult`, `FisherConjugateGradientResult`, `WeightedGradientResult`, `ImplicitSensitivityResult`, `FixedPointSensitivityResult`

### `scpn_quantum_control.differentiable_rust_python_inventory`

Rust/Python inventory governance for differentiable rustification planning.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_rust_python_inventory.py) · Public symbols: **6**

**Classes:** `DifferentiableRustPythonInventoryRow`, `DifferentiableRustPythonInventory`, `DifferentiableRustPythonInventoryValidation`

**Functions:** `run_differentiable_rust_python_inventory()`, `validate_differentiable_rust_python_inventory()`, `render_differentiable_rust_python_inventory_markdown()`

### `scpn_quantum_control.differentiable_scalar_kernels`

Scalar forward- and reverse-mode automatic differentiation kernels.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_scalar_kernels.py) · Public symbols: **10**

**Classes:** `DualNumber`, `ReverseNode`

**Functions:** `dual_sin()`, `dual_cos()`, `dual_exp()`, `dual_log()`, `reverse_sin()`, `reverse_cos()`, `reverse_exp()`, `reverse_log()`

### `scpn_quantum_control.differentiable_sparse_derivatives`

Sparse derivative conversion helpers for native differentiable transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_sparse_derivatives.py) · Public symbols: **5**

**Functions:** `dense_to_sparse_matrix()`, `sparse_jacobian()`, `sparse_hessian()`, `empirical_fisher_metric()`, `sparse_empirical_fisher_metric()`

### `scpn_quantum_control.differentiable_stochastic_estimators`

SPSA, score-function, and shot-budget stochastic gradient helpers.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_stochastic_estimators.py) · Public symbols: **3**

**Functions:** `spsa_gradient_estimate()`, `score_function_gradient_estimate()`, `allocate_parameter_shift_shots()`

### `scpn_quantum_control.differentiable_stochastic_policy`

Fail-closed stochastic-gradient uncertainty policy contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_stochastic_policy.py) · Public symbols: **3**

**Classes:** `GradientFailurePolicy`, `StochasticGradientConfidenceInterval`

**Functions:** `gradient_confidence_interval()`

### `scpn_quantum_control.differentiable_transform_algebra`

Metamorphic transform-algebra gate for differentiable local routes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_transform_algebra.py) · Public symbols: **4**

**Classes:** `TransformAlgebraCase`, `TransformAlgebraAudit`

**Functions:** `run_transform_algebra_audit()`, `assert_transform_algebra_audit_passes()`

### `scpn_quantum_control.differentiable_transform_helpers`

Shared scalar, parameter, bounds, and tape helpers for native transforms.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_transform_helpers.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.differentiable_transform_support_matrix`

Generated support-matrix rows for the transform-algebra audit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_transform_support_matrix.py) · Public symbols: **3**

**Classes:** `TransformAlgebraCaseLike`, `TransformAlgebraSupportMatrixRow`

**Functions:** `build_transform_algebra_support_matrix()`

### `scpn_quantum_control.differentiable_transform_support_matrix_artifact`

Committed-artefact emission for the transform-algebra support matrix.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_transform_support_matrix_artifact.py) · Public symbols: **5**

**Classes:** `TransformSupportMatrixArtifactValidation`

**Functions:** `build_transform_support_matrix_artifact()`, `validate_transform_support_matrix_artifact()`, `render_transform_support_matrix_markdown()`, `main()`

### `scpn_quantum_control.differentiable_vmap`

Eager vectorization transform for differentiable-programming objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/differentiable_vmap.py) · Public symbols: **1**

**Functions:** `vmap()`

### `scpn_quantum_control.error_mitigation_product`

Fail-closed **differentiable error-mitigation taxonomy** product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/error_mitigation_product.py) · Public symbols: **19**

**Classes:** `MitigatorTaxonomyRow`, `MitigationBoundaryRow`, `PathEligibilityDecision`, `MaterialisedZneProbe`, `MaterialisedReadoutProbe`

**Functions:** `list_mitigator_ids()`, `list_mitigation_boundary_ids()`, `get_mitigator()`, `get_mitigation_boundary()`, `iter_mitigators()`, `iter_mitigation_boundaries()`, `decide_mitigation_path()`, `materialise_zne_probe()`, `materialise_demo_zne_probe()`, `materialise_readout_probe()`, `studio_mitigate_claim_boundary()`, `map_error_mitigation_public_surfaces()`, `build_error_mitigation_product_registry()`, `assert_error_mitigation_product_integrity()`

### `scpn_quantum_control.execution_surface`

Static execution-surface scanner for notebooks and publication scripts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/execution_surface.py) · Public symbols: **8**

**Classes:** `ExecutionSurfaceFinding`, `ExecutionSurfaceManifestEntry`, `ExecutionSurfaceViolation`

**Functions:** `load_execution_surface_manifest()`, `evaluate_execution_surface_manifest()`, `iter_execution_surface_paths()`, `find_unmanifested_high_risk_surfaces()`, `scan_execution_surface_path()`

### `scpn_quantum_control.fault_tolerant_resource_product`

Compose existing QEC resource primitives into a conservative fault-tolerant resource report.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/fault_tolerant_resource_product.py) · Public symbols: **12**

**Classes:** `FaultTolerantResourceBoundaryError`, `FormulaReference`, `SyncProblemResourceRequest`, `ResourceEstimate`, `SensitivityPoint`, `RegimeComparisonRow`, `FaultTolerantResourceProduct`

**Functions:** `estimate_ft_resources()`, `build_ft_sensitivity()`, `build_regime_comparison()`, `build_fault_tolerant_resource_product()`, `render_ft_resource_markdown()`

### `scpn_quantum_control.geometric_control_product`

Fail-closed **geometric quantum control** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/geometric_control_product.py) · Public symbols: **21**

**Classes:** `GeometryCapabilityRow`, `GeometryBoundaryRow`, `PathEligibilityDecision`, `MaterialisedMetricDiagnosticsProbe`, `MaterialisedQngDirectionProbe`

**Functions:** `list_geometry_capability_ids()`, `list_geometry_boundary_ids()`, `list_geometry_glossary_keys()`, `get_geometry_glossary_entry()`, `get_geometry_capability()`, `get_geometry_boundary()`, `iter_geometry_capabilities()`, `iter_geometry_boundaries()`, `list_geometry_ambient_inventory()`, `decide_geometry_path()`, `materialise_metric_diagnostics_probe()`, `materialise_demo_metric_diagnostics_probe()`, `materialise_qng_direction_probe()`, `map_geometric_control_public_surfaces()`, `build_geometric_control_product_registry()`, `assert_geometric_control_product_integrity()`

### `scpn_quantum_control.governed_route_matrix`

Fail-closed multi-ecosystem differentiable route matrix and explain API.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/governed_route_matrix.py) · Public symbols: **9**

**Classes:** `RouteCapability`, `GovernedRouteRecord`, `RouteExplanation`

**Functions:** `list_governed_route_ids()`, `get_governed_route()`, `iter_governed_routes()`, `build_governed_route_matrix()`, `explain_route()`, `assert_no_blank_matrix_cells()`

### `scpn_quantum_control.gradient_plan_explanation_artifact`

Committed gradient-plan explanations for the Studio cockpit.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/gradient_plan_explanation_artifact.py) · Public symbols: **5**

**Classes:** `GradientPlanExplanationArtifactValidation`

**Functions:** `build_gradient_plan_explanation_artifact()`, `validate_gradient_plan_explanation_artifact()`, `render_gradient_plan_explanation_markdown()`, `main()`

### `scpn_quantum_control.hardware_result_pack_evidence`

Generate a release-audit evidence packet for hardware result packs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware_result_pack_evidence.py) · Public symbols: **9**

**Functions:** `sha256()`, `utc_stamp()`, `run_json_command()`, `run_log_command()`, `load_manifest()`, `select_packs()`, `rel()`, `parse_pack_ids()`, `main()`

### `scpn_quantum_control.hardware_result_packs`

Offline verification and deterministic export for hardware result packs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware_result_packs.py) · Public symbols: **14**

**Functions:** `default_repo_root()`, `sha256()`, `digest_bytes()`, `walk_values()`, `contains_text()`, `load_manifest()`, `select_packs()`, `artifact_path()`, `verify_manifest()`, `tarinfo_for_bytes()`, `write_deterministic_tar_gz()`, `export_result_packs()`, `parse_pack_ids()`, `main()`

### `scpn_quantum_control.hardware_safe_execution`

Fail-closed hardware-safe gradient execution policy product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hardware_safe_execution.py) · Public symbols: **13**

**Classes:** `ExecutionPolicy`, `DryRunPlan`, `EnforceDecision`, `AuditRecord`

**Functions:** `list_execution_policy_ids()`, `get_execution_policy()`, `iter_execution_policies()`, `default_execution_policy()`, `dry_run_execution_plan()`, `enforce_execution_request()`, `build_audit_record()`, `build_hardware_safe_execution_registry()`, `assert_hardware_safe_execution_integrity()`

### `scpn_quantum_control.hermetic_reproduction_kit`

Hermetic external reproduction kit contract.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/hermetic_reproduction_kit.py) · Public symbols: **13**

**Classes:** `KitDigestSpec`, `HermeticKitEntry`, `DigestCheckResult`

**Functions:** `list_hermetic_kit_entry_ids()`, `get_hermetic_kit_entry()`, `iter_hermetic_kit_entries()`, `build_hermetic_reproduction_kit()`, `sha256_hex_of()`, `verify_digest()`, `verify_kit_entry_digests()`, `probe_hermetic_kit_entry()`, `assert_hermetic_kit_integrity()`, `fixture_payload()`

### `scpn_quantum_control.identity_observer_product`

Fail-closed control observers over the existing identity metrics.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/identity_observer_product.py) · Public symbols: **7**

**Classes:** `IdentityMetricInventoryRow`, `IdentityObserverThresholds`, `IdentityObserverRecord`, `IdentitySafetyDecision`

**Functions:** `identity_metric_inventory()`, `identity_observer_unsuitable_scenarios()`, `evaluate_identity_safety()`

### `scpn_quantum_control.jax_nqs_baseline_product`

Build claim-bounded JAX NQS evidence against exact diagonalisation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/jax_nqs_baseline_product.py) · Public symbols: **7**

**Classes:** `JAXNQSBaselineSpec`, `JAXNQSEnvironment`, `JAXNQSComparison`, `JAXNQSBaselineProduct`

**Functions:** `run_jax_nqs_baseline()`, `render_jax_nqs_baseline_markdown()`, `write_jax_nqs_baseline_evidence()`

### `scpn_quantum_control.kuramoto`

Backward-compatible re-export shim for the relocated Kuramoto toolkit facade.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/kuramoto.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.kuramoto_core`

Small public facade for Kuramoto-XY problems.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/kuramoto_core.py) · Public symbols: **10**

**Classes:** `KuramotoProblem`

**Functions:** `validate_kuramoto_inputs()`, `build_kuramoto_problem()`, `compile_hamiltonian()`, `compile_dense_hamiltonian()`, `compile_trotter_circuit()`, `compile_analog_program()`, `compile_hybrid_program()`, `measure_order_parameter()`, `simulate_variant_trajectory()`

### `scpn_quantum_control.kyma_mechanism_benchmark_product`

Fail-closed **KYMA / KYMA v2 public mechanism-only benchmark** product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/kyma_mechanism_benchmark_product.py) · Public symbols: **15**

**Classes:** `KymaSuiteRow`, `FrozenDesignConstants`, `PathEligibilityDecision`, `MaterialisedMechanismCertificateProbe`

**Functions:** `load_frozen_design_constants()`, `list_kyma_suite_ids()`, `get_kyma_suite()`, `iter_kyma_suites()`, `get_frozen_design_constants()`, `decide_kyma_path()`, `materialise_mechanism_certificate_probe()`, `materialise_demo_mechanism_certificate_probe()`, `map_kyma_mechanism_benchmark_public_surfaces()`, `build_kyma_mechanism_benchmark_product_registry()`, `assert_kyma_mechanism_benchmark_product_integrity()`

### `scpn_quantum_control.logging_setup`

Structlog-backed logging bootstrap.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/logging_setup.py) · Public symbols: **3**

**Functions:** `configure_logging()`, `get_logger()`, `reset_for_testing()`

### `scpn_quantum_control.metamorphic_ad_verification`

Versioned metamorphic AD verification catalogue and pure residual checks.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/metamorphic_ad_verification.py) · Public symbols: **10**

**Classes:** `MetamorphicLawRecord`, `MetamorphicCheckResult`

**Functions:** `list_metamorphic_law_ids()`, `get_metamorphic_law()`, `iter_metamorphic_laws()`, `build_metamorphic_ad_registry()`, `probe_metamorphic_law()`, `evaluate_linearity_residual()`, `evaluate_chain_rule_residual()`, `assert_metamorphic_registry_integrity()`

### `scpn_quantum_control.migration_guides_product`

Fail-closed **PennyLane + Qiskit migration guides** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/migration_guides_product.py) · Public symbols: **13**

**Classes:** `MigrationConceptRow`, `PathEligibilityDecision`, `MaterialisedPennyLaneRoundTrip`, `MaterialisedQiskitLocalGradient`

**Functions:** `list_migration_concept_ids()`, `get_migration_concept()`, `iter_migration_concepts()`, `decide_migration_path()`, `materialise_demo_pennylane_round_trip()`, `materialise_demo_qiskit_local_gradient()`, `map_migration_guides_public_surfaces()`, `build_migration_guides_product_registry()`, `assert_migration_guides_product_integrity()`

### `scpn_quantum_control.multi_hal_federation_product`

Fail-closed **Multi-HAL provider federation** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/multi_hal_federation_product.py) · Public symbols: **14**

**Classes:** `HalCapabilityRecord`, `PathEligibilityDecision`, `MaterialisedFederationDryRunProbe`

**Functions:** `list_hal_backend_ids()`, `list_hal_providers()`, `get_hal_capability()`, `iter_hal_capabilities()`, `build_federation_matrix()`, `decide_federation_route()`, `materialise_federation_dry_run_probe()`, `materialise_demo_federation_dry_run_probe()`, `map_multi_hal_federation_public_surfaces()`, `build_multi_hal_federation_product_registry()`, `assert_multi_hal_federation_product_integrity()`

### `scpn_quantum_control.neural_operator_baseline_product`

Compose existing neural-operator baselines under fail-closed governance.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/neural_operator_baseline_product.py) · Public symbols: **8**

**Classes:** `BaselineSurfaceRow`, `ArtifactVerification`, `DatasetAdmission`, `IntegrationDisposition`, `NeuralOperatorBaselineProduct`

**Functions:** `verify_neural_operator_artifact()`, `assess_forecast_dataset()`, `build_neural_operator_baseline_product()`

### `scpn_quantum_control.notebook_programme_product`

Fail-closed **differentiable notebook programme** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/notebook_programme_product.py) · Public symbols: **12**

**Classes:** `CurriculumNotebookRow`, `PathEligibilityDecision`, `MaterialisedCurriculumProbe`

**Functions:** `list_curriculum_notebook_ids()`, `get_curriculum_notebook()`, `iter_curriculum_notebooks()`, `decide_notebook_programme_path()`, `resolve_curriculum_directory()`, `materialise_curriculum_probe()`, `map_notebook_programme_public_surfaces()`, `build_notebook_programme_registry()`, `assert_notebook_programme_product_integrity()`

### `scpn_quantum_control.open_system_mcwf_product`

Fail-closed **open-system MCWF completeness** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/open_system_mcwf_product.py) · Public symbols: **22**

**Classes:** `OpenSystemSurfaceRow`, `OpenSystemBoundaryRow`, `PathEligibilityDecision`, `MaterialisedMcwfEnsembleProbe`, `MaterialisedReproducibilityProbe`

**Functions:** `list_open_system_surface_ids()`, `list_open_system_boundary_ids()`, `get_open_system_surface()`, `get_open_system_boundary()`, `iter_open_system_surfaces()`, `iter_open_system_boundaries()`, `decide_open_system_path()`, `materialise_mcwf_ensemble_probe()`, `materialise_demo_mcwf_ensemble_probe()`, `materialise_reproducibility_probe()`, `export_sim_noise_model()`, `import_sim_noise_model()`, `map_open_system_mcwf_public_surfaces()`, `list_ambient_objective_boundary_ids()`, `list_default_objective_case_ids()`, `build_open_system_mcwf_product_registry()`, `assert_open_system_mcwf_product_integrity()`

### `scpn_quantum_control.pgbo_qgt_product`

Fail-closed **PGBO quantum geometric tensor** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/pgbo_qgt_product.py) · Public symbols: **16**

**Classes:** `QgtCapabilityRow`, `QgtBoundaryRow`, `PathEligibilityDecision`, `MaterialisedPgboTensorProbe`

**Functions:** `list_qgt_capability_ids()`, `list_qgt_boundary_ids()`, `get_qgt_capability()`, `get_qgt_boundary()`, `iter_qgt_capabilities()`, `iter_qgt_boundaries()`, `decide_qgt_path()`, `materialise_pgbo_tensor_probe()`, `materialise_demo_pgbo_tensor_probe()`, `map_pgbo_qgt_public_surfaces()`, `build_pgbo_qgt_product_registry()`, `assert_pgbo_qgt_product_integrity()`

### `scpn_quantum_control.phase_qnode_product`

Fail-closed Phase-QNode **product** catalogue and journey map.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/phase_qnode_product.py) · Public symbols: **9**

**Classes:** `PhaseQNodeJourney`, `PhaseQNodeJourneyDecision`

**Functions:** `list_phase_qnode_journey_ids()`, `get_phase_qnode_journey()`, `iter_phase_qnode_journeys()`, `dry_run_phase_qnode_journey()`, `map_phase_qnode_public_surfaces()`, `build_phase_qnode_product_registry()`, `assert_phase_qnode_product_integrity()`

### `scpn_quantum_control.polyglot_edge_ad_product`

Fail-closed polyglot edge Program-AD product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/polyglot_edge_ad_product.py) · Public symbols: **12**

**Classes:** `EdgeADRuntimeRow`, `EdgeADPathDecision`, `CommittedWasmReplayCertificate`

**Functions:** `list_edge_ad_runtime_ids()`, `get_edge_ad_runtime()`, `iter_edge_ad_runtimes()`, `decide_edge_ad_path()`, `load_committed_wasm_replay_payload()`, `materialise_wasm_replay_certificate()`, `map_polyglot_edge_ad_public_surfaces()`, `build_polyglot_edge_ad_product_registry()`, `assert_polyglot_edge_ad_product_integrity()`

### `scpn_quantum_control.polyglot_parity_certificate`

Fail-closed **bit-exact polyglot parity certificate** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/polyglot_parity_certificate.py) · Public symbols: **14**

**Classes:** `ParityFamily`, `PolyglotParityCertificate`, `CertificateVerifyDecision`

**Functions:** `list_parity_family_ids()`, `get_parity_family()`, `iter_parity_families()`, `canonical_json_bytes()`, `digest_payload()`, `build_sample_certificate()`, `certificate_from_dict()`, `verify_certificate()`, `map_parity_public_surfaces()`, `build_polyglot_parity_product_registry()`, `assert_polyglot_parity_product_integrity()`

### `scpn_quantum_control.program_ad_adjoint`

Program AD reverse-adjoint generation result records and accessors.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_adjoint.py) · Public symbols: **7**

**Classes:** `ProgramADAdjointStep`, `ProgramADAdjointResult`

**Functions:** `program_adjoint_result()`, `program_adjoint_gradient()`, `program_adjoint_replay_gradient()`, `program_adjoint_grad()`, `program_adjoint_value_and_grad()`

### `scpn_quantum_control.program_ad_adjoint_generation`

Reverse-mode adjoint generation over the stabilised whole-program AD IR.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_adjoint_generation.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.program_ad_alias_analysis`

Program AD alias/effect analysis drivers and static-lattice provenance parsing.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_alias_analysis.py) · Public symbols: **2**

**Functions:** `analyze_program_ad_alias_effects()`, `program_ad_static_alias_lattice_report()`

### `scpn_quantum_control.program_ad_alias_contracts`

Immutable Program AD alias, lattice, and provenance result contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_alias_contracts.py) · Public symbols: **10**

**Classes:** `ProgramADAliasSet`, `ProgramADAliasEffectAnalysis`, `ProgramADStaticAliasLatticeComponent`, `ProgramADUnknownAliasEdge`, `ProgramADViewAliasProvenance`, `ProgramADListAliasProvenance`, `ProgramADLoopCarriedStateProvenance`, `ProgramADControlPathAliasProvenance`, `ProgramADRebindingAliasProvenance`, `ProgramADStaticAliasLatticeReport`

### `scpn_quantum_control.program_ad_array_indexing`

Static array-indexing derivative rules for Program AD registry dispatch.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_array_indexing.py) · Public symbols: **6**

**Functions:** `program_ad_array_getitem_derivative_rule()`, `program_ad_array_take_derivative_rule()`, `program_ad_array_take_along_axis_derivative_rule()`, `program_ad_array_delete_derivative_rule()`, `program_ad_array_pad_derivative_rule()`, `program_ad_array_insert_derivative_rule()`

### `scpn_quantum_control.program_ad_assembly_primitives`

Program AD assembly derivative factories and registry contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_assembly_primitives.py) · Public symbols: **4**

**Functions:** `program_ad_assembly_split_derivative_rule()`, `program_ad_assembly_tril_derivative_rule()`, `program_ad_assembly_triu_derivative_rule()`, `program_ad_assembly_diagonal_derivative_rule()`

### `scpn_quantum_control.program_ad_broadcast_assembly`

Static broadcast assembly derivative rules for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_broadcast_assembly.py) · Public symbols: **2**

**Functions:** `program_ad_assembly_broadcast_to_derivative_rule()`, `program_ad_assembly_broadcast_arrays_derivative_rule()`

### `scpn_quantum_control.program_ad_cumulative_primitives`

Static cumulative derivative rules for Program AD registry dispatch.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_cumulative_primitives.py) · Public symbols: **3**

**Functions:** `program_ad_cumulative_cumsum_derivative_rule()`, `program_ad_cumulative_cumprod_derivative_rule()`, `program_ad_cumulative_diff_derivative_rule()`

### `scpn_quantum_control.program_ad_effect_ir`

Validated Program AD effect-IR records and metadata parser.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_effect_ir.py) · Public symbols: **7**

**Classes:** `ProgramADSSAValue`, `ProgramADEffect`, `ProgramADAliasEdge`, `ProgramADPhiNode`, `ProgramADControlRegion`, `ProgramADEffectIR`

**Functions:** `parse_program_ad_effect_ir()`

### `scpn_quantum_control.program_ad_elementwise_primitives`

Program AD elementwise primitive contracts and direct derivative factories.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_elementwise_primitives.py) · Public symbols: **1**

**Functions:** `program_ad_elementwise_binary_derivative_rule()`

### `scpn_quantum_control.program_ad_fuzz_assurance`

Fail-closed **Rust Program AD fuzz assurance** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_fuzz_assurance.py) · Public symbols: **14**

**Classes:** `FuzzTarget`, `FuzzPolicy`, `FuzzProbeDecision`

**Functions:** `list_fuzz_target_ids()`, `get_fuzz_target()`, `iter_fuzz_targets()`, `fuzz_assurance_policy()`, `validate_time_box_seconds()`, `dry_run_fuzz_target()`, `corpus_governance_policy()`, `crash_pipeline_policy()`, `map_fuzz_public_surfaces()`, `build_fuzz_assurance_registry()`, `assert_fuzz_assurance_integrity()`

### `scpn_quantum_control.program_ad_interpolation_primitives`

Static interpolation derivative rules for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_interpolation_primitives.py) · Public symbols: **1**

**Functions:** `program_ad_interpolation_interp_derivative_rule()`

### `scpn_quantum_control.program_ad_linalg_primitives`

Static linear-algebra derivative rules and registry contracts for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_linalg_primitives.py) · Public symbols: **14**

**Classes:** `ProgramADLinalgConditioningDiagnostic`

**Functions:** `diagnose_program_ad_linalg_conditioning()`, `program_ad_linalg_solve_derivative_rule()`, `program_ad_linalg_matrix_power_derivative_rule()`, `program_ad_linalg_multi_dot_derivative_rule()`, `program_ad_linalg_trace_derivative_rule()`, `program_ad_linalg_diag_derivative_rule()`, `program_ad_linalg_diagflat_derivative_rule()`, `program_ad_linalg_eig_derivative_rule()`, `program_ad_linalg_eigvals_derivative_rule()`, `program_ad_linalg_eigh_derivative_rule()`, `program_ad_linalg_eigvalsh_derivative_rule()`, `program_ad_linalg_svdvals_derivative_rule()`, `program_ad_linalg_pinv_derivative_rule()`

### `scpn_quantum_control.program_ad_product_primitives`

Program AD product/contraction derivative factories and registry contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_product_primitives.py) · Public symbols: **5**

**Functions:** `program_ad_product_matmul_derivative_rule()`, `program_ad_product_tensordot_derivative_rule()`, `program_ad_product_inner_derivative_rule()`, `program_ad_product_outer_derivative_rule()`, `program_ad_product_einsum_derivative_rule()`

### `scpn_quantum_control.program_ad_reduction_primitives`

Program AD reduction primitive contracts and direct derivative factories.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_reduction_primitives.py) · Public symbols: **10**

**Functions:** `program_ad_reduction_sum_derivative_rule()`, `program_ad_reduction_mean_derivative_rule()`, `program_ad_reduction_var_derivative_rule()`, `program_ad_reduction_std_derivative_rule()`, `program_ad_reduction_max_derivative_rule()`, `program_ad_reduction_min_derivative_rule()`, `program_ad_reduction_median_derivative_rule()`, `program_ad_reduction_quantile_derivative_rule()`, `program_ad_reduction_percentile_derivative_rule()`, `program_ad_reduction_prod_derivative_rule()`

### `scpn_quantum_control.program_ad_registry`

Primitive registry contracts and Program AD registry-dispatch coverage.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_registry.py) · Public symbols: **20**

**Classes:** `CustomDerivativeRule`, `PrimitiveIdentity`, `PrimitiveTransformRule`, `PrimitiveContract`, `ProgramADRegistryDispatchCoverageRow`, `ProgramADRegistryDispatchCoverageReport`, `CustomDerivativeRegistry`

**Functions:** `register_custom_derivative_rule()`, `register_primitive_transform_rule()`, `register_primitive_batching_rule()`, `register_primitive_lowering_rule()`, `primitive_shape_rule_for()`, `primitive_dtype_rule_for()`, `primitive_static_argument_rule_for()`, `primitive_nondifferentiable_policy_for()`, `primitive_effect_for()`, `primitive_contract_for()`, `primitive_complete_contract_for()`, `program_ad_registry_dispatch_coverage_report()`, `custom_derivative_rule_for()`

### `scpn_quantum_control.program_ad_rust_bridge`

Rust bridge wrappers for bounded Program AD effect IR replay.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_rust_bridge.py) · Public symbols: **7**

**Classes:** `ProgramADEffectIRLike`, `RustProgramADInterpreterResult`, `RustProgramADValueAndGradientResult`, `RustProgramADRegistryMetadataMirrorResult`

**Functions:** `interpret_program_ad_effect_ir_with_rust()`, `value_and_grad_program_ad_effect_ir_with_rust()`, `mirror_program_ad_registry_metadata_with_rust()`

### `scpn_quantum_control.program_ad_selection_primitives`

Program AD selection derivative factories and registry contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_selection_primitives.py) · Public symbols: **2**

**Functions:** `program_ad_selection_where_derivative_rule()`, `program_ad_selection_clip_derivative_rule()`

### `scpn_quantum_control.program_ad_shape_transforms`

Static shape-transform derivative rules for Program AD registry dispatch.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_shape_transforms.py) · Public symbols: **17**

**Functions:** `program_ad_shape_reshape_derivative_rule()`, `program_ad_shape_ravel_derivative_rule()`, `program_ad_shape_transpose_derivative_rule()`, `program_ad_shape_expand_dims_derivative_rule()`, `program_ad_shape_squeeze_derivative_rule()`, `program_ad_shape_swapaxes_derivative_rule()`, `program_ad_shape_moveaxis_derivative_rule()`, `program_ad_shape_roll_derivative_rule()`, `program_ad_shape_flip_derivative_rule()`, `program_ad_shape_flipud_derivative_rule()`, `program_ad_shape_fliplr_derivative_rule()`, `program_ad_shape_rot90_derivative_rule()`, `program_ad_shape_repeat_derivative_rule()`, `program_ad_shape_tile_derivative_rule()`, `program_ad_shape_atleast_1d_derivative_rule()`, `program_ad_shape_atleast_2d_derivative_rule()`, `program_ad_shape_atleast_3d_derivative_rule()`

### `scpn_quantum_control.program_ad_signal_primitives`

Static convolution and correlation derivative rules for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_signal_primitives.py) · Public symbols: **2**

**Functions:** `program_ad_signal_convolve_derivative_rule()`, `program_ad_signal_correlate_derivative_rule()`

### `scpn_quantum_control.program_ad_stack_block_assembly`

Static stack, append, concatenate, and block assembly rules for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_stack_block_assembly.py) · Public symbols: **8**

**Functions:** `program_ad_assembly_concatenate_derivative_rule()`, `program_ad_assembly_stack_derivative_rule()`, `program_ad_assembly_hstack_derivative_rule()`, `program_ad_assembly_vstack_derivative_rule()`, `program_ad_assembly_column_stack_derivative_rule()`, `program_ad_assembly_dstack_derivative_rule()`, `program_ad_assembly_append_derivative_rule()`, `program_ad_assembly_block_derivative_rule()`

### `scpn_quantum_control.program_ad_stencil_primitives`

Static finite-difference stencil derivative rules for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_stencil_primitives.py) · Public symbols: **1**

**Functions:** `program_ad_stencil_gradient_derivative_rule()`

### `scpn_quantum_control.program_ad_trapezoid_primitives`

Static trapezoidal-integration derivative rules for Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/program_ad_trapezoid_primitives.py) · Public symbols: **1**

**Functions:** `program_ad_reduction_trapezoid_derivative_rule()`

### `scpn_quantum_control.public_api_stability`

Fail-closed public-vs-internal API stability catalogue.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/public_api_stability.py) · Public symbols: **14**

**Classes:** `PublicApiSymbolRecord`, `PathClassification`, `DeprecationProbe`, `BreakingChangeDecision`

**Functions:** `list_public_api_symbol_ids()`, `get_public_api_symbol()`, `iter_public_api_symbols()`, `classify_api_path()`, `probe_deprecation()`, `validate_breaking_change()`, `deprecated_public()`, `build_public_api_stability_registry()`, `assert_public_api_stability_integrity()`, `version_compatibility_note()`

### `scpn_quantum_control.qpu_compute`

Public QPU compute façade and command-line entry point.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qpu_compute.py) · Public symbols: **1**

**Functions:** `main()`

### `scpn_quantum_control.qpu_compute_product`

Fail-closed qpu_compute plan/runtime product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qpu_compute_product.py) · Public symbols: **13**

**Classes:** `ComputePlanKind`, `ComputePlanRecord`, `ComputePlanDecision`

**Functions:** `list_plan_kind_ids()`, `get_plan_kind()`, `iter_plan_kinds()`, `list_supported_kernels()`, `list_supported_backend_policies()`, `construct_compute_plan()`, `dry_run_compute_plan()`, `audit_compute_plan_decision()`, `build_qpu_compute_product_registry()`, `assert_qpu_compute_product_integrity()`

### `scpn_quantum_control.qpu_compute_runtime`

Simulator runtime and JSON I/O for QPU compute contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qpu_compute_runtime.py) · Public symbols: **14**

**Functions:** `deterministic_counts()`, `make_compute_request()`, `execute_simulator_request()`, `write_compute_request()`, `read_compute_request()`, `write_compute_result()`, `read_compute_result()`, `write_node_descriptor()`, `read_node_descriptor()`, `write_stream_delta()`, `read_stream_delta()`, `write_fusion_result()`, `read_fusion_result()`, `run_simulator_from_artifact()`

### `scpn_quantum_control.qpu_compute_types`

Provider-neutral QPU compute request/result contracts.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/qpu_compute_types.py) · Public symbols: **9**

**Classes:** `QPUComputeRequest`, `QPUComputeResult`, `QPUNodeDescriptor`, `QPUStreamDelta`, `QPUFusionResult`

**Functions:** `require_non_empty()`, `json_sha256()`, `counts_sha256()`, `fuse_compute_results()`

### `scpn_quantum_control.quantum_sync_challenge_oracle_product`

Fail-closed **Quantum Sync Challenge oracle** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/quantum_sync_challenge_oracle_product.py) · Public symbols: **17**

**Classes:** `ProblemFamilyRow`, `MetricCatalogueRow`, `BaselineCatalogueRow`, `PathEligibilityDecision`, `MaterialisedOracleProbe`

**Functions:** `list_problem_family_ids()`, `list_metric_ids()`, `list_baseline_ids()`, `get_problem_family()`, `iter_problem_families()`, `compute_instance_digest()`, `decide_challenge_path()`, `materialise_oracle_probe()`, `materialise_demo_oracle_probe()`, `map_quantum_sync_challenge_oracle_public_surfaces()`, `build_quantum_sync_challenge_oracle_product_registry()`, `assert_quantum_sync_challenge_oracle_product_integrity()`

### `scpn_quantum_control.resource_budget_gate`

Fail-closed compile & dense resource budget product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/resource_budget_gate.py) · Public symbols: **12**

**Classes:** `BudgetDimension`, `ResourceBudgetEstimate`, `ResourceBudgetDecision`, `ResourceBudgetExceededError`

**Functions:** `list_budget_dimension_ids()`, `get_budget_dimension()`, `iter_budget_dimensions()`, `estimate_resource_budget()`, `check_resource_budget()`, `enforce_resource_budget()`, `build_resource_budget_registry()`, `assert_resource_budget_integrity()`

### `scpn_quantum_control.scorecard_acceptance_engine`

Fail-closed baseline-scorecard acceptance / promotion engine.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/scorecard_acceptance_engine.py) · Public symbols: **8**

**Classes:** `ScorecardCategoryRecord`, `PromoteDecision`

**Functions:** `list_scorecard_category_ids()`, `get_scorecard_category()`, `iter_scorecard_categories()`, `build_scorecard_acceptance_registry()`, `promote_scorecard_category()`, `assert_scorecard_acceptance_integrity()`

### `scpn_quantum_control.ssgf_geometry_gradient_product`

Governed SSGF quantum-in-the-loop geometry-gradient product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/ssgf_geometry_gradient_product.py) · Public symbols: **15**

**Classes:** `SsgfPublicSurfaceRow`, `GradientRouteDecision`, `QuantumCostCertificate`, `GeometryGradientCertificate`, `SsgfGeometryObserverRecord`, `OuterCycleEvidence`

**Functions:** `list_ssgf_public_surfaces()`, `ssgf_gradient_unsuitable_scenarios()`, `decide_ssgf_gradient_route()`, `certify_quantum_cost()`, `certify_geometry_gradient()`, `geometry_observer_from_certificate()`, `materialise_outer_cycle_evidence()`, `build_ssgf_geometry_gradient_registry()`, `assert_ssgf_geometry_gradient_integrity()`

### `scpn_quantum_control.stable_core`

Stable first-path contracts for SCPN quantum-control workflows.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/stable_core.py) · Public symbols: **21**

**Classes:** `Problem`, `Backend`, `Experiment`, `Result`

**Functions:** `build_problem()`, `build_backend()`, `classical_reference_backend()`, `hardware_replay_backend()`, `qiskit_backend()`, `qutip_backend()`, `pennylane_backend()`, `pulser_surrogate_backend()`, `backend_capability_matrix()`, `stable_core_capability_payload()`, `normalised_stable_core_json()`, `stable_core_capability_markdown()`, `write_stable_core_capability_artifacts()`, `build_experiment()`, `build_result()`, `problem_from_kuramoto()`, `problem_to_kuramoto()`

### `scpn_quantum_control.stable_core_preflight`

Backend preflight checks for stable core experiments.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/stable_core_preflight.py) · Public symbols: **6**

**Classes:** `StableCorePreflightResult`

**Functions:** `run_stable_core_preflight()`, `stable_core_backend_dependencies()`, `stable_core_preflight_fixtures_payload()`, `stable_core_preflight_fixtures_json()`, `stable_core_preflight_fixtures_markdown()`

### `scpn_quantum_control.stable_core_product`

Fail-closed **stable_core experiment model** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/stable_core_product.py) · Public symbols: **29**

**Classes:** `StableCoreContractRow`, `StableCoreRoundTripResult`

**Functions:** `list_stable_core_contract_ids()`, `get_stable_core_contract()`, `iter_stable_core_contracts()`, `schema_version_policy()`, `validate_model_schema_version()`, `problem_from_dict()`, `backend_from_dict()`, `experiment_from_dict()`, `result_from_dict()`, `wrap_model_envelope()`, `unwrap_model_envelope()`, `canonical_json_bytes()`, `digest_stable_core_payload()`, `serialise_problem()`, `serialise_backend()`, `serialise_experiment()`, `serialise_result()`, `deserialise_problem()`, `deserialise_backend()`, `deserialise_experiment()`, `deserialise_result()`, `round_trip_problem()`, `round_trip_experiment()`, `build_demo_experiment()`, `map_stable_core_public_surfaces()`, `build_stable_core_product_registry()`, `assert_stable_core_product_integrity()`

### `scpn_quantum_control.stochastic_estimators_product`

Fail-closed **stochastic estimators & policies** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/stochastic_estimators_product.py) · Public symbols: **12**

**Classes:** `StochasticEstimatorRow`, `EstimatorDryRunDecision`, `MaterialisedSPSAProbe`

**Functions:** `list_stochastic_estimator_ids()`, `get_stochastic_estimator()`, `iter_stochastic_estimators()`, `build_product_failure_policy()`, `dry_run_stochastic_estimator()`, `materialise_demo_spsa_probe()`, `map_stochastic_estimators_public_surfaces()`, `build_stochastic_estimators_product_registry()`, `assert_stochastic_estimators_product_integrity()`

### `scpn_quantum_control.structured_log_fallback`

Structured logging with a stdlib fallback that tolerates event kwargs.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/structured_log_fallback.py) · Public symbols: **2**

**Classes:** `KwargTolerantLoggerAdapter`

**Functions:** `get_structured_logger()`

### `scpn_quantum_control.studio_executive_product`

Fail-closed **Studio executive + coverage frontier** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/studio_executive_product.py) · Public symbols: **12**

**Classes:** `ExecutiveVerbRow`, `PathEligibilityDecision`, `MaterialisedCoverageFrontierProbe`

**Functions:** `list_executive_verb_ids()`, `get_executive_verb()`, `iter_executive_verbs()`, `decide_executive_path()`, `compute_coverage_frontier_score()`, `materialise_demo_coverage_frontier_probe()`, `map_studio_executive_public_surfaces()`, `build_studio_executive_product_registry()`, `assert_studio_executive_product_integrity()`

### `scpn_quantum_control.thermo_readiness_product`

Fail-closed **thermodynamics readiness** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/thermo_readiness_product.py) · Public symbols: **19**

**Classes:** `ReadinessCapabilityRow`, `FepInventoryRow`, `PathEligibilityDecision`, `MaterialisedKSweepProbe`

**Functions:** `verify_ambient_claim_boundary()`, `list_readiness_capability_ids()`, `list_fep_module_ids()`, `get_readiness_capability()`, `get_fep_inventory_row()`, `iter_readiness_capabilities()`, `iter_fep_inventory()`, `decide_readiness_path()`, `materialise_k_sweep_probe()`, `materialise_demo_k_sweep_probe()`, `materialise_quantum_thermo_payload_probe()`, `map_thermo_readiness_public_surfaces()`, `build_thermo_readiness_product_registry()`, `assert_thermo_readiness_product_integrity()`, `compute_k_sweep_request_digest()`

### `scpn_quantum_control.unsuitable_scenario_registry`

Versioned unsuitable-scenario and anti-silent-wrong-gradient registry.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/unsuitable_scenario_registry.py) · Public symbols: **8**

**Classes:** `UnsuitableScenarioRecord`, `ScenarioProbeResult`

**Functions:** `list_unsuitable_scenario_ids()`, `get_unsuitable_scenario()`, `iter_unsuitable_scenarios()`, `build_unsuitable_scenario_registry()`, `probe_unsuitable_scenario()`, `assert_unsuitable_registry_integrity()`

### `scpn_quantum_control.visualisation_dashboard_product`

Fail-closed **fixture-driven visualisation dashboard** product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/visualisation_dashboard_product.py) · Public symbols: **13**

**Classes:** `VisualisationPanelRow`, `PathEligibilityDecision`, `SecretsScanResult`, `MaterialisedStaticReportProbe`

**Functions:** `list_visualisation_panel_ids()`, `get_visualisation_panel()`, `iter_visualisation_panels()`, `scan_export_for_secrets()`, `decide_visualisation_path()`, `materialise_demo_static_report_probe()`, `map_visualisation_dashboard_public_surfaces()`, `build_visualisation_dashboard_product_registry()`, `assert_visualisation_dashboard_product_integrity()`

### `scpn_quantum_control.whole_program_ad_api`

Public whole-program automatic differentiation entry points.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_ad_api.py) · Public symbols: **2**

**Functions:** `whole_program_value_and_grad()`, `whole_program_grad()`

### `scpn_quantum_control.whole_program_ad_product`

Fail-closed whole-program AD **product** catalogue and journey map.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_ad_product.py) · Public symbols: **10**

**Classes:** `WholeProgramADJourney`, `WholeProgramADJourneyDecision`

**Functions:** `list_whole_program_ad_journey_ids()`, `get_whole_program_ad_journey()`, `iter_whole_program_ad_journeys()`, `dry_run_whole_program_ad_journey()`, `map_whole_program_ad_public_surfaces()`, `map_whole_program_ad_architecture_layers()`, `build_whole_program_ad_product_registry()`, `assert_whole_program_ad_product_integrity()`

### `scpn_quantum_control.whole_program_ad_result`

Whole-program automatic-differentiation result records.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_ad_result.py) · Public symbols: **3**

**Classes:** `WholeProgramTraceEvent`, `WholeProgramIRNode`, `WholeProgramADResult`

### `scpn_quantum_control.whole_program_frontend`

Static bytecode/source inspection pipeline for whole-program Program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_frontend.py) · Public symbols: **1**

**Functions:** `compile_whole_program_frontend()`

### `scpn_quantum_control.whole_program_frontend_contracts`

Immutable IR and report contracts for static whole-program frontend inspection.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_frontend_contracts.py) · Public symbols: **9**

**Classes:** `WholeProgramBytecodeInstruction`, `WholeProgramSourceIRFeature`, `WholeProgramBytecodeBasicBlock`, `WholeProgramSourceRegion`, `WholeProgramSourceBytecodeLineMap`, `WholeProgramSymbolScopeEntry`, `WholeProgramUnsupportedSemanticDiagnostic`, `WholeProgramSemanticsReport`, `WholeProgramCompilerFrontendReport`

### `scpn_quantum_control.whole_program_trace_metadata`

Static normalisation of whole-program AD trace operation shapes and axes.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_trace_metadata.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.whole_program_trace_predicates`

Derivative-safe primal control-flow predicates for whole-program AD.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_trace_predicates.py) · Public symbols: **1**

**Classes:** `TraceADPredicateArray`

### `scpn_quantum_control.whole_program_trace_runtime`

Runtime trace-context builders for whole-program automatic differentiation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_trace_runtime.py) · Public symbols: **0**

No public module-level class or function is declared.

### `scpn_quantum_control.whole_program_trace_values`

Operator-intercepted forward-AD trace value classes and their operations.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/whole_program_trace_values.py) · Public symbols: **2**

**Classes:** `TraceADScalar`, `TraceADArray`

### `scpn_quantum_control.wirtinger_calculus`

Wirtinger calculus: holomorphic and non-holomorphic complex derivatives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/wirtinger_calculus.py) · Public symbols: **7**

**Classes:** `WirtingerDerivative`, `WirtingerOptimisationResult`

**Functions:** `wirtinger_partials()`, `is_holomorphic()`, `holomorphic_gradient()`, `real_objective_gradient()`, `minimise_real_objective()`

### `scpn_quantum_control.wirtinger_implicit_product`

Fail-closed **Wirtinger + implicit differentiation** product surface.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/wirtinger_implicit_product.py) · Public symbols: **13**

**Classes:** `WirtingerImplicitSurfaceRow`, `MaterialisedWirtingerProbe`, `MaterialisedImplicitProbe`, `ComplexContractDecision`

**Functions:** `list_wirtinger_implicit_surface_ids()`, `get_wirtinger_implicit_surface()`, `iter_wirtinger_implicit_surfaces()`, `decide_complex_objective_contract()`, `materialise_demo_wirtinger_probe()`, `materialise_demo_implicit_stationary_probe()`, `map_wirtinger_implicit_public_surfaces()`, `build_wirtinger_implicit_product_registry()`, `assert_wirtinger_implicit_product_integrity()`

## `topology_control`

### `scpn_quantum_control.topology_control.artefacts`

Serialisation and digest artefacts for topological optimisation traces.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/artefacts.py) · Public symbols: **2**

**Classes:** `TopologyOptimisationArtifact`

**Functions:** `export_topology_optimisation_artifact()`

### `scpn_quantum_control.topology_control.complexes`

Persistent-H1 complex builders and backend abstractions for topology control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/complexes.py) · Public symbols: **9**

**Classes:** `PersistenceDiagram`, `H1Summary`, `PersistentHomologyBackend`, `NetworkCycleBackend`, `RipserPHBackend`

**Functions:** `max_h1_for_vertices()`, `build_coupling_distance_matrix()`, `build_correlation_distance_matrix()`, `spike_trace_correlation_distance()`

### `scpn_quantum_control.topology_control.constraints`

Constraint ledgers and projection routines for coupling graph control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/constraints.py) · Public symbols: **6**

**Classes:** `CouplingGraphBounds`, `HardwareEmbeddingConstraint`, `ConstraintViolation`, `TopologyConstraintLedger`

**Functions:** `canonical_edge()`, `algebraic_connectivity()`

### `scpn_quantum_control.topology_control.hardware_integration`

No-QPU hardware manifest gate for topology-control experiments.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/hardware_integration.py) · Public symbols: **2**

**Classes:** `TopologyHardwareManifest`

**Functions:** `validate_topology_hardware_manifest()`

### `scpn_quantum_control.topology_control.objectives`

Degeneracy-safe persistent-H1 objectives for coupling graph control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/objectives.py) · Public symbols: **5**

**Classes:** `DegeneracyMode`, `ObjectiveBreakdown`, `CouplingTopologyObjective`

**Functions:** `classify_degeneracy()`, `objective_sha256_payload()`

### `scpn_quantum_control.topology_control.optimizers`

Projected optimisers for non-smooth persistent-H1 objectives.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/optimizers.py) · Public symbols: **4**

**Classes:** `TopologyOptimisationStep`, `TopologyOptimisationTrace`, `ProjectedSPSAOptimizer`, `ProjectedScipyOptimizer`

### `scpn_quantum_control.topology_control.qsnn_integration`

QSNN dynamic-coupling policy hooks for persistent-H1 control.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_control/qsnn_integration.py) · Public symbols: **1**

**Classes:** `TopologicalDynamicCouplingPolicy`

## `topology_kernel_product`

### `scpn_quantum_control.topology_kernel_product.classifier`

Custody-checked binary kernel ridge fitting and evaluation.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_kernel_product/classifier.py) · Public symbols: **4**

**Classes:** `KernelRidgeClassifier`

**Functions:** `fit_kernel_ridge()`, `predict_kernel_ridge()`, `evaluate_kernel_ridge()`

### `scpn_quantum_control.topology_kernel_product.evidence`

Deterministic evidence and claim-boundary rendering for topology-kernel.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_kernel_product/evidence.py) · Public symbols: **5**

**Classes:** `KernelSupportRow`, `TopologyKernelEvidence`

**Functions:** `build_topology_kernel_evidence()`, `render_topology_kernel_markdown()`, `write_topology_kernel_evidence()`

### `scpn_quantum_control.topology_kernel_product.kernels`

Validated exact-statevector and classical-control kernel construction.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_kernel_product/kernels.py) · Public symbols: **7**

**Functions:** `validate_topology()`, `topology_digest()`, `validate_feature_matrix()`, `fidelity_kernel_matrix()`, `rbf_kernel_matrix()`, `permute_topology()`, `permute_edge_features()`

### `scpn_quantum_control.topology_kernel_product.schema`

Immutable contracts for the bounded topology-kernel quantum-kernel product.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_kernel_product/schema.py) · Public symbols: **4**

**Classes:** `TopologyKernelConfig`, `TopologyKernelMatrix`, `TopologyKernelDataset`, `KernelEvaluation`

### `scpn_quantum_control.topology_kernel_product.synthetic`

Deterministic graph controls and teacher-aligned synthetic data.

[Source](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/topology_kernel_product/synthetic.py) · Public symbols: **5**

**Functions:** `ring_topology()`, `path_topology()`, `complete_topology()`, `zero_topology()`, `build_teacher_aligned_dataset()`
