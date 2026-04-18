# Roadmap

## Completed

### v0.1.0–v0.8.0 (February 2026)

- Core modules: qsnn, phase, control, bridge, qec, hardware, crypto, mitigation
- 20 hardware experiments, ZNE + DD error mitigation, Heron r2 noise model
- 5 Jupyter notebooks, 10 examples, GitHub Pages docs
- Property-based tests (hypothesis), integration + regression suites
- Identity subpackage: VQE attractor, coherence budget, entanglement witness, fingerprint
- 483 tests, 99%+ coverage

### v0.9.0 (March 2026)

- 100% line coverage, security scanning (bandit + pip-audit)
- CODEOWNERS, SPDX headers on 130 files, AGPL-3.0 dual-license
- Enterprise hardening: 7 CI workflows, Dockerfile, Makefile, GOVERNANCE, SUPPORT
- Identity subpackage: 4 modules, 43 tests
- 553 tests

### v0.9.0+9modules (March 2026)

9 v1.0 modules implemented:

| Module | Description |
|--------|-------------|
| `mitigation/pec.py` | Probabilistic Error Cancellation (Temme et al. PRL 119 180509) |
| `hardware/trapped_ion.py` | Trapped-ion noise model (MS gate, T1/T2) + transpilation |
| `control/q_disruption_iter.py` | ITER 11-feature disruption classifier + synthetic data |
| `benchmarks/quantum_advantage.py` | Classical vs quantum scaling + crossover extrapolation |
| `bridge/snn_adapter.py` | SNN ↔ quantum bridge + ArcaneNeuronBridge (sc-neurocore) |
| `bridge/ssgf_adapter.py` | SSGF ↔ quantum bridge + SSGFQuantumLoop |
| `identity/binding_spec.py` | 6-layer 18-oscillator identity topology + orchestrator mapping |
| `qsnn/training.py` | Parameter-shift gradient training for QuantumDenseLayer |
| `qec/fault_tolerant.py` | Repetition-code logical qubits + transversal RZZ |

Cross-repo integrations wired:

- **sc-neurocore**: ArcaneNeuron spike collection → quantum forward → current feedback
- **SSGF engine**: W/theta read → Trotter evolve → theta writeback (quantum-in-the-loop)
- **scpn-phase-orchestrator**: 18↔35 oscillator phase mapping (identity_coherence domainpack)
- **scpn-fusion-core**: NPZ archive shot data → ITER 11-feature vector

679 tests, 100% coverage, all 6 preflight gates passing.

### v0.9.1 (March 2026)

- 15-dimension codebase audit: 5 critical, 12 high, 6 medium findings fixed (39 files)
- Removed hardcoded IBM CRN, fabricated CVE, broken Dockerfile
- SPDX headers on all files, line-6 descriptors on all `__init__.py`
- CI tool pins: ruff 0.15.6, mypy 1.19.1, bandit 1.9.4
- `knm_to_hamiltonian` dedup, Makefile/pre-commit preflight fix
- 1789 tests, 100% coverage

### v0.9.2 (March 2026)

- Coverage expanded to include `runner.py` and `experiments.py` (previously omitted)
- 38 new runner tests covering all simulator-path methods
- 22 new experiment tests covering all 20 experiment functions
- Rust engine (`scpn_quantum_engine`) rebuilt, parity tests green
- Stale README cross-refs updated (phase-orchestrator v0.5.0, test count, hardware status)
- Failed/cancelled CI runs cleaned
- 1932+ tests

### v0.9.3 (March 2026)

- Rust engine expanded 11→15 functions: `lanczos_b_coefficients`, `otoc_from_eigendecomp`,
  `build_xy_hamiltonian_dense`, `all_xy_expectations`
- Measured benchmarks: 5401× Hamiltonian (n=4), 264× OTOC (n=4), 27× Lanczos (n=3) vs Python
- 8 modules migrated to Rust Hamiltonian path (`knm_to_dense_matrix`), zero `.to_matrix()` callers
- **IBM hardware campaign complete: 20/20 experiments on ibm_fez (Heron r2)**
  - CHSH S=2.165 (Bell inequality violated, >8σ)
  - QKD QBER 5.5% (below BB84 threshold)
  - 16-qubit Kuramoto with dynamical decoupling
  - ZNE stable across fold levels 1-9
  - Knm ansatz outperforms TwoLocal by 32% entropy
- 14 publication figures (simulation + hardware)
- JAX GPU backend (`jax_accel.py`) — vectorised coupling scans
- PyPI Rust wheel CI for 5 platforms (`rust-wheels.yml`)
- Kaggle registered, notebook pushed, ORCID profile filled
- New docs: `rust_engine.md` with benchmark tables, API updates across 4 doc pages

### v0.9.4 (March 2026)

- 81 new tests (PennyLane mock, JAX mock, ripser mock, hardware runner mock, fallbacks)
- OpenSSF Best Practices badge (100% passing)
- 3 benchmark API docs: gpu_baseline, mps_baseline, appqsim_protocol
- Coverage 95%→98%, 2715 tests
- Experiment roadmap + crypto branch updated for completed March hardware campaign

### v0.9.5 (March–April 2026)

- **10X Architecture:** Strange Loop co-evolution engine (DynamicCouplingEngine, TopologicalCouplingOptimizer)
- **BiologicalSurfaceCode:** native topological QEC on SCPN 16-layer graph
- **LindbladSyncEngine:** MCWF trajectory path for large-N open systems
- **StructuredAnsatz:** topology-informed variational circuits for arbitrary coupling graphs
- **EEG Classification:** PLV-to-quantum pipeline for brain state analysis
- **27 FIM notebooks** (NB14–47): 19 discoveries, 6 honest negative results
- **IBM hardware v2:** 9 equal-depth fair experiments on ibm_fez confirming dual protection
- **Rust engine expanded:** 15→22 functions (correlation_matrix_xy, lindblad_jump_ops_coo, lindblad_anti_hermitian_diag, parity_filter_mask)
- **Documentation audit:** 21 discrepancies fixed, 26 analysis + 10 phase + 3 bridge exports added
- Fixed backend_dispatch jax.numpy AttributeError
- 165 Python modules, 22 Rust functions, 47 notebooks, 21 examples
- 2813+ tests, 95% coverage

## v1.0.0 (Target: Q3 2026)

Remaining items:

- [x] IBM Heron r2 hardware campaign (20/20 experiments complete)
- [ ] Coverage push to 100% (currently 95%, 572 uncovered lines)
- [ ] arXiv preprint: "Quantum simulation of coupled-oscillator synchronization on a 156-qubit superconducting processor"
- [ ] Quantum advantage figure: exponential crossover curve (hardware data)
- [ ] IBM Quantum Credits campaign (applied 2026-03-29, pending review)
- [ ] Version bump to 1.0.0

## Future

- Fault-tolerant UPDE on surface code logical qubits (post-2030, hardware-dependent)
- QSNN training loop on real hardware (parameter-shift STDP)
- Quantum disruption classifier on ITER disruption database
- Trapped-ion hardware runs (IonQ Aria / Quantinuum H2)
- SSGF quantum-in-the-loop with live SSGFEngine on GPU

## Strategic Differentiation (post-v1.0)

Thirty-five post-v1.0 differentiation tracks are scoped in
[`docs/strategic_roadmap.md`](docs/strategic_roadmap.md). All are
**DEFERRED / CEO-gated**; no execution until individually activated.
Quarterly review cadence.

Infrastructure + ML (S1–S7):

- **S1** Hybrid classical–quantum feedback loop
- **S2** Quantum advantage benchmarks at scale (N = 4 → 20+)
- **S3** ML-augmented pulse / ansatz design
- **S4** Multi-hardware backend + pulse-level control
- **S5** Open-data + classical validation harness
- **S6** Decoupled `quantum-kuramoto` subpackage
- **S7** Fault-tolerant / logical-level extension roadmap

Scientific directions — batch 1 (S8–S14):

- **S8** Mid-circuit adaptive branching (Dynamic Circuits follow-up to S1)
- **S9** Quantum thermodynamics of synchronisation transitions
- **S10** Analog-native Kuramoto backends (Rydberg / neutral-atom / CV photonic)
- **S11** DLA-driven quantum sensing via sync order parameter
- **S12** Automated phase-diagram exploration via Bayesian optimisation
- **S13** Bosonic / continuous-variable quantum Kuramoto
- **S14** Hybrid quantum-classical forecasting engine

Scientific directions — batch 2 (S15–S21):

- **S15** DLA-protected many-body scars for long-lived synchronisation
- **S16** Quantum network tomography (reconstruct K_nm from observables)
- **S17** Higher-order (simplicial / hypergraph) quantum Kuramoto
- **S18** Synchronisation-protected quantum memories and repeaters
- **S19** Entanglement phase diagram + magic + Krylov complexity
- **S20** Quantum Kuramoto universal control-benchmark suite
- **S21** Multi-scale quantum → classical bridging layer

Scientific directions — batch 3 (S22–S28):

- **S22** Non-Hermitian / PT-symmetric Kuramoto with exceptional points
- **S23** Quantum reservoir computing on Kuramoto transients
- **S24** Quantum speed limits for collective synchronisation
- **S25** Topological defects + vortex dynamics on 2D quantum lattices
- **S26** Entanglement-mediated long-range synchronisation
- **S27** Hardware-in-the-loop inverse design of oscillator networks
- **S28** Synchronisation-enhanced distributed quantum metrology

Scientific directions — batch 4 (S29–S35):

- **S29** Floquet Kuramoto time crystals + subharmonic sync
- **S30** Quantum Kuramoto for community detection and modularity
- **S31** DLA-protected many-body localisation / delocalisation
- **S32** Monitored quantum Kuramoto (measurement-induced transitions)
- **S33** Quantum-enhanced Lyapunov spectra for chaotic Kuramoto
- **S34** Self-organising Kuramoto (autonomous drive engineering)
- **S35** Quantum Kuramoto native simulator for active matter

Foundational tracks (S36–S53) — geometric, categorical, field-theoretic, foundations:

- **S36** Information geometry on quantum sync manifolds
- **S37** Categorical / compositional quantum Kuramoto
- **S38** Quantum Kuramoto field theory continuum limit + RG flows
- **S39** Autopoietic / self-referential networks
- **S40** Holographic duals via quantum synchronisation
- **S41** Quantum causal discovery with intervention
- **S42** Symplectic structure-preserving Trotterisation
- **S43** Full resource theory of quantum synchronisation
- **S44** Objective collapse / macroscopic foundations testbed
- **S45** Biologically faithful Kuramoto simulator + IIT
- **S46** Phase-transition / attractor-landscape quantum programming
- **S47** Analogue gravity (relativistic, cosmological, baryogenesis, emergent spacetime)
- **S48** Self-healing qubit fabrics + continuous QEC via sync
- **S49** Quantum fluctuation theorems across sync transitions
- **S50** Quantum kernels from sync manifolds (ML)
- **S51** Hayden–Preskill / black-hole information dynamics simulator
- **S52** Distributed quantum consensus via global sync (quantum internet)
- **S53** Engineered self-organised criticality

Applied verticals (cross-cutting over S1–S53, no separate physics
tracks): fusion plasma stabilisation; tipping-point early-warning;
IIT consciousness testbed; quantum biology engineering; quantum
internet infrastructure; autonomous AI physicist (discovery
engine). Each applied vertical is an activation target for one or
more physics tracks listed above.
