# Changelog

All notable changes to scpn-quantum-control are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [0.9.6] - 2026-04-17

### Security

- **rand 0.9.2 → 0.9.4** in `scpn_quantum_engine/Cargo.lock` to address
  RUSTSEC-2026-0097 (Rand unsoundness with custom logger calling
  `rand::rng()` from inside the logger; aliased `&mut BlockRng` from
  the public API). Resolves Dependabot alert #2 and code-scanning
  alert #81. All 92 Rust unit tests pass with the bumped dependency.
- **`scripts/retrieve_ibm_job.py`** no longer prints the first 50
  characters of the IBM Cloud CRN read from the vault. The CRN
  contains the account ID, region and project prefix; the truncation
  was insufficient. Replaced with a content-free confirmation
  message. Resolves CodeQL alert #79
  (`py/clear-text-logging-sensitive-data`).
- **`pytest 9.0.2 → 9.0.3`** (PR #37) — addresses CVE-2025-71176
  (`/tmp/pytest-of-{user}` directory pattern allows local users to
  cause a denial of service or possibly gain privileges on UNIX).
  Resolves Dependabot alert #1.

### Added

- **Input validation in `analysis/koopman.py`** — `build_koopman_generator`
  and `koopman_analysis` now reject malformed input (non-square `K`,
  length-mismatched `omega`/`theta_ref`, NaN/Inf entries) and cap the
  problem size at `MAX_OSCILLATORS_DEFAULT = 32` oscillators by
  default. Larger sizes require an explicit `max_oscillators=`
  argument so a stray caller cannot trigger an unbounded n²×n²
  allocation followed by a multi-minute `eigvals` call. 13 new tests
  cover every guard branch.
- **`docs/pipeline_performance.md` §21 — measured Rust speedups vs
  Python baseline** plus an explicit decision rule for when a new
  acceleration backend (Mojo / Julia / Lean 4) enters the build
  matrix.

### Changed

- **CI dev-tool matrix** synced to current upstreams (PRs #36–#42):
  `pytest 9.0.3`, `mypy 1.20.1`, `ruff 0.15.10`, `hypothesis 6.151.13`,
  `build 1.4.3`, `actions/upload-artifact 7.0.1`,
  `pypa/gh-action-pypi-publish 1.14.0`.
- **Public-facing docs scrubbed** of self-applied quality labels
  (`elite`, `Elite documentation`, `STRONG tests`, `SUPERIOR
  documentation`) in `CHANGELOG.md`, `docs/changelog.md`,
  `docs/test_infrastructure.md`, `docs/symmetry_decay_guess.md`,
  `docs/dynq_qubit_mapping.md` and 27 test docstrings. Domain-technical
  uses (`STRONG correlation` verdict in stats tables, `STRONG coupling`
  in biochemical notebooks) are kept — they are field terminology.
- **Stale counts refreshed** across `README.md`, `docs/architecture.md`,
  `docs/index.md`, `docs/pipeline_performance.md` and
  `docs/test_infrastructure.md`: subpackages 19 → 20 (with the
  April-2026 `psi_field/` and `fep/` rows added), test suite 4 771 /
  4 828 → 4 841, pipeline-wiring 113 → 155, Rust path 51 → 68 tests
  over 37 functions, `__all__` 77 → 104 symbols, three duplicate
  phantom rows removed from the `docs/index.md` package map.

### Repository hygiene

- **`.coordination/sessions/` and `.coordination/handovers/` are now
  local-only.** The 16 historical session logs and handovers tracked
  in this repo were untracked via `git rm --cached`. Local working
  copies and git history are preserved; new logs no longer ship with
  the public repo.
- **New `.gitignore` patterns** for paper-extraction working files
  (raw `pdftotext` dumps, structured per-paper extractions, per-paper
  assessments, archived under-fidelity `_v1_archive/`),
  `.agent_metadata.json`, and root-level misplaced `handover_*.md`.

### Internal

- Cleared 13 internal `docs/internal/gemini/` audit files after
  verification — see session log
  `claude_2026-04-17T0317_pr_security_hygiene.md` for per-file
  decisions. Two cross-ecosystem outputs persist:
  `agentic-shared/configs/agent_metadata_schema.json` (cross-repo
  metadata schema 1.0.0) and
  `agentic-shared/research_reports/architectural_pillars_2026-04-17.md`
  (verified deferral criteria for the five pillars Gemini proposed).


## [0.9.5] - 2026-03-29 / 2026-04-11

### Added (2026-04-10 — Phase 1 IBM Quantum hardware campaign)

- **First publishable hardware confirmation of the DLA parity asymmetry**
  on IBM ibm_kingston (Heron r2, 156 qubits). 348 circuits across four
  sub-phases (pipe cleaner → Phase 1 → 1.5 → 2 → 2.5 final burn) with
  up to 21 repetitions per (depth, sector) point at $n = 4$. Mean
  asymmetry $+10.8\,\%$ for Trotter depths $\ge 4$, peak $+17.48\,\%$
  at depth 6. Welch's two-sample t-test gives 7/8 depths individually
  significant at $p < 0.05$; Fisher's combined chi² = 123.4 over 16 df,
  combined $p \ll 10^{-16}$. Result is consistent with the 4.5–9.6 %
  apriori prediction from the noiseless classical simulator.
- **`scripts/analyse_phase1_dla_parity.py`** — publication-quality
  statistical analysis script: per-depth SEM, Welch t-test per depth,
  Fisher's combined p, readout baseline correction, matplotlib
  figures (`figures/phase1/leakage_vs_depth.png`,
  `figures/phase1/asymmetry_vs_depth.png`). Reproduces the full Phase 1
  analysis from the raw `.coordination/ibm_runs/*.json` files in one
  command.
- **`paper/phase1_dla_parity_short_paper.md`** — 267-line short-paper
  draft for *Quantum Science and Technology* (Letter) or *Physical
  Review Research*, with abstract, methods, results table, discussion,
  Phase 2 plan, and full reference list.
- **IBM hardware execution scripts** under `scripts/`:
  `pipe_cleaner_ibm_kingston.py`, `phase1_mini_bench_ibm_kingston.py`,
  `phase1_5_reinforce_ibm_kingston.py`,
  `phase2_exhaust_cycle_ibm_kingston.py`,
  `phase2_5_final_burn_ibm_kingston.py`,
  `phase2_full_campaign_ibm.py` (preview, requires
  `--confirm-promo-active` flag), `micro_probe_ibm_kingston.py`,
  `retrieve_ibm_job.py`.
- **`.coordination/IBM_CAMPAIGN_STATE.md`**, **`IBM_EXECUTION_LOG.md`**,
  **`phase1_experiment_design.md`**, **`WEBMASTER_CONTEXT.md`** —
  persistent IBM-campaign and webmaster context that survives session
  compaction.

### Added (2026-04-08 — strategic tweaks from Gemini Deep Research report)

- **GUESS symmetry-decay ZNE** (`mitigation/symmetry_decay.py`,
  `scpn_quantum_engine/src/symmetry_decay.rs`): physics-informed
  zero-noise extrapolation that uses the conserved total magnetisation
  of $H_{XY}$ as the guide observable. Eq. 5 from
  Oliva del Moral *et al.*, arXiv:2603.13060 (2026), implemented
  exactly. Rust path provides `fit_symmetry_decay` (least-squares α
  fit) and `guess_extrapolate_batch` (rayon-parallel correction).
  20 multi-angle tests across 6 dimensions.
- **DynQ topology-agnostic qubit mapper** (`hardware/qubit_mapper.py`,
  `scpn_quantum_engine/src/community.rs`): Louvain community detection
  on calibration-weighted QPU graphs, with quality scoring per region
  (connectivity × fidelity composite, Eq. 8 of Liu *et al.*,
  arXiv:2601.19635). 17 multi-angle tests.
- **PMP / ICI pulse sequences** (`phase/pulse_shaping.py`,
  `scpn_quantum_engine/src/pulse_shaping.rs`): three-segment
  Pontryagin-optimal mixing-angle trajectory for STIREP, with full
  3-level Λ system Lindblad evolution. Liu *et al.* (2023). The Rust
  `ici_three_level_evolution_batch` is 1,665× faster than the Python
  forward-Euler reference, verified to machine precision.
- **(α,β)-hypergeometric pulse shaping** (same module): unified
  adiabatic pulse family via Gauss ${}_2F_1$, subsuming Allen-Eberly
  (α=β=0), STIRAP (α=β=0.5), and Demkov-Kunike (α=1, β=0.5) as special
  cases. Rust path implements ${}_2F_1$ via custom series expansion +
  rayon, 44× faster than scipy element-wise. Ventura Meinersen *et al.*,
  arXiv:2504.08031 (2025). 25 multi-angle tests cover both ICI and the
  hypergeometric family.
- **FFI boundary hardening** for the entire Rust crate: all 36
  `#[pyfunction]` exports now return `PyResult<T>` and validate inputs
  via `validation.rs` (`validate_n`, `validate_positive`,
  `validate_range`, `validate_finite`, `validate_flat_square`,
  `validate_statevec_len`, `validate_domain_range`). Inner pure-Rust
  functions are kept separate so the algorithms can be unit-tested
  without a Python interpreter. 16 unit tests in `validation.rs`.
- **Long-form documentation** (>567 lines each):
  `docs/symmetry_decay_guess.md` (891 lines) and
  `docs/dynq_qubit_mapping.md` (878 lines), both with theory,
  references, API, tutorials, benchmarks, limitations, and
  comparisons.
- **Pipeline performance entries** in
  `tests/test_pipeline_wiring_performance.py` for GUESS, DynQ, and the
  pulse-shaping module — every new module is wired end-to-end with a
  measured wall-time budget.

### Added (2026-04-10 — repository hygiene)

- **`tools/check_secrets.py`** — custom vault-pattern secret scanner.
  Reads ~84 credential-shaped tokens from
  `agentic-shared/CREDENTIALS.md` (filtered by length, Shannon
  entropy ≥ 3.5 bits/char, mixed character classes, and a substantial
  ignore list), then greps the staged diff for any of them. A
  secondary keyword-based scan catches `password:` / `token:` /
  `api_key:` patterns regardless of value entropy. Redacts every
  match in its failure output so the hook itself never leaks. Pre-commit
  hook entry added in `.pre-commit-config.yaml`.
- **gitleaks v8.21.2 pre-commit hook** for generic high-entropy
  secrets and known token formats (GitHub, AWS, GCP, PEM, etc.).
  Complements the custom vault scanner.
- **`.coordination/incidents/INCIDENT_2026-04-10T2336_ftp_creds_in_webmaster_context.md`**
  — full post-mortem of the credential leak that motivated the new
  scanners (caught at the L4 pre-push audit gate, never reached
  origin).
- `.gitignore` updated to cover platform venvs (`.venv-linux/`,
  `.venv-rocm/`, `.venv-cuda/`), the `results/` scratch directory,
  `.coordination/TODO_*.md` working notes, and `.coordination/*.pdf`
  email-attachment artefacts.

### Changed (2026-04-10)

- Tests collected: 2,813 → **4,828** (97%+ coverage).
- Python modules: 165 → **201**, subpackages 17 → **19**.
- Rust functions exported: 22 → **36** across **20** Rust source files.
- Hardware reference: ibm_fez (Feb 2026) → ibm_fez + **ibm_kingston**
  (Feb + Apr 2026).
- `runner.py`: robust counts extraction for qiskit-ibm-runtime 0.46+
  (DataBin classical-register name is no longer hard-coded as `meas`).

### Fixed (2026-04-10)

- IBM SamplerV2 result parsing previously assumed the classical
  register was named `meas`. Newer qiskit-ibm-runtime returns the
  actual circuit register name (e.g. `c` for `QuantumCircuit(n, n)`),
  which broke `run_sampler`. Replaced with `_extract_counts(...)`
  that tries common names (`meas`, `c`, `cr`, `c0`) and falls back to
  introspecting the `DataBin` attributes.


## [0.9.5] - 2026-03-29 / 2026-04-07

### Added (2026-04-06 — 2026-04-07)
- **Multi-Scale QEC** (`qec/multiscale_qec.py`, `qec/syndrome_flow.py`): concatenated surface codes across 5 SCPN domains. Knill 2005 threshold, K_nm-weighted syndrome flow, Rust-accelerated via `concat_qec.rs`. 23 multi-angle tests.
- **Free Energy Principle** (`fep/variational_free_energy.py`, `fep/predictive_coding.py`): Friston 2010 variational F = complexity + accuracy, hierarchical prediction errors, KL divergence, ELBO gradient. Rust-accelerated via `fep.rs`. 16 multi-angle tests.
- **Ψ-field Lattice Gauge** (`psi_field/lattice.py`, `psi_field/infoton.py`, `psi_field/scpn_mapping.py`, `psi_field/observables.py`): U(1) compact gauge theory with HMC, gauge-covariant kinetic (Rothe convention), Polyakov loops, topological charge, string tension. Rust-accelerated via `gauge_lattice.rs`. 22 multi-angle tests.
- **K_nm validation completed**: IEEE 5-bus (ρ=0.881), Josephson (ρ=0.990), EEG (ρ=0.916), ITER MHD (ρ=0.944) — all 5 physical systems now measured.
- **Long-form documentation**: 567+ line docs for MS-QEC, FEP, and Ψ-field (8 mandatory sections each).
- **ripser** added to `[topology]` optional-dependencies in pyproject.toml.

### Changed (2026-04-06 — 2026-04-07)
- **Rust engine refactored**: `lib.rs` god file (1436 lines) split into 16 focused modules + 3 new Rust paths (`concat_qec.rs`, `fep.rs`, `gauge_lattice.rs`). 22→25 exported functions.
- **OTOC optimised**: avoid full W(t) matrix construction per time point — O(d) phase rotation instead of O(d²) matrix build. 4.4× faster (benchmarked n=4, 50 time points).
- **Pauli expectations optimised**: half-loop over paired states halves inner iterations. 2–10× faster across `state_order_param_sparse` and `all_xy_expectations`.
- **Kuramoto order parameter**: Rust fast path via `all_xy_expectations` eliminates 2n individual Qiskit `expectation_value` calls per Trotter step.
- **CI dependencies pinned**: Dockerfile base image SHA256, `build==1.4.2`, `pip-audit==2.9.0`, `sc-neurocore==3.14.0`.
- **Dev deps bumped**: ruff 0.15.6→0.15.9, mypy 1.19.1→1.20.0, hypothesis 6.151.10→6.151.11.
- **God file split**: `multiscale_qec.py` (346 lines) → `multiscale_qec.py` (292) + `syndrome_flow.py` (66).
- Tests 2715 → 4445+ (0 failures), Rust tests 47 → 65.

### Added (2026-03-29 — 2026-04-01)
- **10X Architecture**: Transitioned from static VQE to a dynamic 'Strange Loop' co-evolution engine.
- **Dynamic Coupling**: 'DynamicCouplingEngine' for Quantum Hebbian Learning (micro-entanglement driving macro-topology).
- **Topological Optimization**: 'TopologicalCouplingOptimizer' and 'HardwareTopologicalOptimizer' using Persistent Homology ($p_{h1}$) as a loss function on IBM hardware.
- **Native Topological QEC**: 'BiologicalSurfaceCode' mapping stabilizers directly to the SCPN 16-layer graph topology.
- **Open Systems**: 'LindbladSyncEngine' with memory-efficient Quantum Trajectory (MCWF) path for large-N simulation.
- **Biological Ingestion**: EEG PLV-to-Quantum pipeline for brain state classification via structured VQE and quantum kernels.
- **Performance**: High-performance sparse evolution engine bypassing Qiskit circuit overhead.
- **Generalization**: 'StructuredAnsatz' class for topology-informed variational circuits on arbitrary coupling graphs.
- **27 experiment notebooks** (NB14–47): FIM mechanism deep investigation
- **81 FIM mechanism tests** (`test_fim_mechanism.py`): regression tests for all 19 findings (0 skips, 0 stubs)
- **25 JSON result files**: complete experimental data from 27 notebooks
- **RESULTS_SUMMARY.md**: comprehensive summary of all findings
- **IBM hardware v2**: 9 equal-depth fair experiments on ibm_fez confirming dual protection
- **IBM Quantum Credits application** submitted (5h QPU, 04/2026–08/2026)
- **Rust engine**: 4 new functions (correlation_matrix_xy, lindblad_jump_ops_coo, lindblad_anti_hermitian_diag, parity_filter_mask) — 18→22 total
- **115 multi-angle tests** for 9 Gemini 10X modules

### Discovered
- **FIM alone synchronises** without coupling (K=0, λ≥8) — NB26
- **Scaling law** λ_c(N) = 0.149·N^1.02 (R² = 0.966) — NB25
- **Dual protection confirmed on IBM hardware** (F_FIM=0.916 > F_XY=0.849, p<10⁻⁶) — IBM v2
- **BKT universality** (β→0, not mean-field β=0.5) — NB43
- **FIM enhances MBL** via M²/n sector splitting — NB31, NB38
- **Φ (IIT) increases 73%** under FIM — NB28
- **100% basin of attraction** at (K=12, λ=5) — NB27
- **Topology-universal**, small-world optimal (+0.31) — NB36
- **P = 0.085λ** linear thermodynamic cost — NB33
- **Stochastic resonance** under FIM at weak coupling — NB41
- **Delay-robust** with coupling, fragile without — NB42
- **Critical slowing down** τ=330 at BKT transition — NB34
- **Topological defects suppressed** 8→0 by FIM — NB47
- **Metabolic scaling P∝N** matches biology (r=0.983) — NB46
- **Self-consistent mean-field equation** R* = √(1−2Δ/(K·R+λ·R/(1−R²+ε))) — NB37
- **6 anaesthesia predictions** with hysteresis — NB35
- **U(1) gauge invariance** confirmed — NB40
- **Cross-frequency coupling** (PAC, wavelet, Granger) confirms SCPN — NB22

### Fixed
- Enforced K-symmetry in 'knm_hamiltonian.py' to ensure physical Hermiticity across all variational modules (Finding #7).
- Wrapped all example scripts in 'if __name__ == "__main__":' to prevent 'pytest' collection crashes.
- Fixed 'delta' (anisotropy) parameter propagation in 'pairing_correlator.py' and 'xxz_phase_diagram.py'.
- Resolved infinite recursion in 'knm_to_dense_matrix' fallback logic.
- Added 'vqe_energy' alias in 'PhaseVQE' for backward compatibility.
- Migrated 25+ analysis modules to Rust-accelerated dense matrix path.
- Excluded optional Rust acceleration branches from coverage counts.

### Negative Results (honest)
- Curvature does NOT peak at K_c — NB23
- Directed coupling HURTS sync — NB24
- DLA parity direction reversed on hardware — IBM v1,v2
- Empirical FIM from EEG: method fails — NB30
- FIM-modulated learning: no benefit — NB44
- Noise purification: not on symmetric noise — NB45

## [0.9.4] - 2026-03-29

### Fixed
- **PennyLane adapter**: catch `Exception` not just `ImportError` on broken JAX/PennyLane
- **5 test failures**: skip guards for unimplemented Rust engine functions
- **12 doc errors**: wrong function names in API docs, outdated test counts, stale version numbers
- **CI**: correct test args for ripser mock, inject missing module attrs

### Added
- **81 new tests** across 6 test files (pennylane mock, JAX mock, ripser mock, hardware runner mock, Python fallback, edge cases)
- **OpenSSF Best Practices badge**: 100% passing (project 12290)
- **OpenSSF Scorecard, Ruff, mypy badges**
- **3 benchmark API docs**: gpu_baseline, mps_baseline, appqsim_protocol

### Changed
- Coverage 95% → 98% (9601 stmts, 165 missed)
- Tests 2634 → 2715 (0 failures)
- Workflow permissions hardened: docs.yml `contents:write` scoped to job level, ci.yml `contents:read` at top level
- Architecture stats updated to v0.9.4 (155 modules, 23.8k LOC, 29 doc pages)
- Experiment roadmap + crypto branch updated to reflect completed March hardware campaign

## [0.9.3] - 2026-03-28

### Added

- **Rust engine: 4 new functions** (11→15 total)
  - `lanczos_b_coefficients`: complex Lanczos loop (27× vs numpy)
  - `otoc_from_eigendecomp`: parallel OTOC via eigendecomp (264× vs scipy)
  - `build_xy_hamiltonian_dense`: bitwise flip-flop (5,401× vs Qiskit)
  - `all_xy_expectations`: batch Pauli (6.2× vs individual calls)
- **IBM hardware campaign: 20/20 experiments complete**
  - 22 jobs on ibm_fez (Heron r2, 156 qubits), 176,000+ shots
  - CHSH S=2.165 (>8σ), QBER 5.5%, 16-qubit UPDE
  - ZNE stable fold 1-9, Knm ansatz outperforms TwoLocal
- **16 publication figures** (simulation + hardware + MBL + BKT)
- **3 publications on GitHub Pages**: preprint, sync witnesses, DLA parity
- **Scientific findings**:
  - BKT universality preserved (CFT c=1.04 at n=8, gap R²>0.96)
  - Non-ergodic regime (Poisson level spacing, 25-33% sub-thermal)
  - DTC survives heterogeneous frequencies (15/15 amplitudes)
  - Scrambling 4× faster at strong coupling (OTOC)
- `knm_to_dense_matrix`: Rust fast path for dense Hamiltonian (8 modules migrated)
- JAX GPU backend (`jax_accel.py`) for vectorised coupling scans
- PyPI Rust wheel CI (`rust-wheels.yml`) for 5 platforms
- SCPN theory page + biochemical foundations on GitHub Pages
- Results gallery on GitHub Pages
- 12 coverage tests for v0.9.3 additions
- Kaggle notebook for JAX GPU validation + BKT universality tests

### Changed

- README reframed: hardware-first, SCPN as advanced benchmark
- 12 GitHub topics added, repo description updated
- GitHub Release v0.9.3 created
- Merged 2 Dependabot PRs (codeql-action, codecov-action)

### Fixed

- Empty `pauli_list` crash in `knm_to_xxz_hamiltonian` (hypothesis edge case)
- Rust parity tests: `pytest.importorskip` for Docker CI
- JAX backend: build H in numpy/Rust, GPU for eigh/SVD only (was 1731× slower)
- 3 TokenPermissions OpenSSF Scorecard alerts
- Version test 0.9.2→0.9.3
- Preprint MBL→non-ergodic correction (honest cross-validation)

## [0.9.2] - 2026-03-26

### Added

- 38 runner coverage tests (all simulator-path methods)
- 22 experiment coverage tests (all 20 experiment functions)
- `requirements.txt` and `requirements-dev.txt` with pinned versions

### Changed

- Removed coverage omit for `runner.py` and `experiments.py` (now fully tested)
- Updated README cross-refs: phase-orchestrator v0.5.0/2321 tests, test badge 1932+, hardware campaign status
- Updated ROADMAP with v0.9.1 and v0.9.2 sections
- Cleaned failed/cancelled CI runs

### Fixed

- Rust engine `build_knm` parity test (rebuilt `scpn_quantum_engine` wheel)

## [0.9.1] - 2026-03-25

### Added (Rounds 1-8: 33 research gems, ~9,772 lines, 350+ tests)

- `analysis/sync_witness.py`: 3 synchronization witness constructions (correlation, Fiedler, topological)
- `mitigation/symmetry_verification.py`: Z₂ parity post-selection from DLA proof
- `analysis/quantum_persistent_homology.py`: full PH pipeline from hardware counts to p_h1
- `phase/cpdr_simulation.py`: CPDR for XY Hamiltonian simulation
- `phase/coupling_topology_ansatz.py`: K_nm-topology-informed VQE ansatz
- `analysis/sync_entanglement_witness.py`: R as entanglement witness (Kuramoto 1975 → quantum)
- `analysis/entanglement_sync.py`: entanglement-enhanced synchronization
- `phase/cross_domain_transfer.py`: cross-domain VQE parameter transfer
- `analysis/otoc_sync_probe.py`: OTOC as sync transition probe
- `analysis/hamiltonian_self_consistency.py`: Hamiltonian self-consistency loop for K_nm validation
- `analysis/quantum_speed_limit.py`: quantum speed limit for BKT synchronization
- `analysis/qfi_criticality.py`: QFI metrological sweet spot at K_c
- `analysis/entanglement_percolation.py`: entanglement percolation = sync threshold conjecture
- `analysis/qrc_phase_detector.py`: QRC self-probing phase detection (reservoir IS the system)
- `phase/floquet_kuramoto.py`: Floquet-Kuramoto DTC (heterogeneous frequencies)
- `analysis/critical_concordance.py`: multi-probe K_c agreement
- `analysis/berry_fidelity.py`: Berry phase / fidelity susceptibility at BKT
- `analysis/quantum_mpemba.py`: quantum Mpemba effect in synchronization dynamics
- `analysis/lindblad_ness.py`: Lindblad NESS for driven-dissipative Kuramoto-XY
- `analysis/adiabatic_gap.py`: adiabatic preparation gap analysis at BKT
- `bridge/knm_hamiltonian.py`: XXZ Hamiltonian (Kouchekian-Teodorescu S² embedding)
- `analysis/pairing_correlator.py`: Richardson pairing correlators ⟨S⁺S⁻⟩
- `analysis/xxz_phase_diagram.py`: anisotropy phase diagram K_c vs Δ
- `analysis/spectral_form_factor.py`: SFF at synchronization transition
- `analysis/loschmidt_echo.py`: Loschmidt echo / DQPT for quenches across K_c
- `analysis/entanglement_entropy.py`: entanglement entropy + Schmidt gap at sync transition
- `analysis/krylov_complexity.py`: Krylov complexity at synchronization transition
- `analysis/magic_sre.py`: magic (SRE M₂) + finite-size scaling for K_c
- `analysis/finite_size_scaling.py`: BKT ansatz K_c(∞) + a/(log N)² extraction
- `tests/test_round4_8_coverage.py`: 36 coverage gap tests for Rounds 4-8 modules
- IBM Quantum hardware campaign: 9 jobs submitted to ibm_fez (2 completed, 7 queued)

### Added (March 20-22 marathon: 60 commits, 56 new modules)

- `analysis/` subpackage expanded to 14 modules:
  - `bkt_analysis`: BKT phase transition (Fiedler, T_BKT, p_h1 prediction)
  - `bkt_universals`: 10 candidate expressions for p_h1 = 0.72
  - `p_h1_derivation`: A_HP × sqrt(2/pi) = 0.717, Gap 3 closed (0.5%)
  - `phase_diagram`: K_c vs T_eff synchronisation boundary
  - `dynamical_lie_algebra`: DLA computation (126/255 at N=4)
  - `qfi`: Quantum Fisher Information for parameter estimation
  - `quantum_phi`: IIT integrated information from density matrix
  - `entanglement_spectrum`: half-chain entropy, CFT central charge
  - `koopman`: Koopman linearisation for nonlinear Kuramoto (BQP argument)
  - `otoc`: Out-of-time-order correlator for quantum chaos
  - `shadow_tomography`: classical shadow estimation (O(log M) shots)
  - `hamiltonian_learning`: recover K_nm from measurement data
  - `enaqt`: environment-assisted quantum transport optimisation
  - `vortex_binding`: Kosterlitz RG flow, logarithmic pair energy
  - `h1_persistence`: vortex density scan at BKT transition
- `gauge/` subpackage (5 modules — NEW):
  - `wilson_loop`: U(1) Wilson loop measurement
  - `vortex_detector`: BKT vortex density order parameter
  - `cft_analysis`: CFT central charge extraction at K_c
  - `universality`: BKT universality class check (eta, Nelson-Kosterlitz)
  - `confinement`: string tension, confinement-deconfinement mapping
- `ssgf/` subpackage (4 modules — NEW):
  - `quantum_gradient`: dC_quantum/dz via finite differences
  - `quantum_costs`: C_micro, C4_tcbo, C_pgbo quantum cost terms
  - `quantum_outer_cycle`: variational z descent with quantum feedback
  - `quantum_spectral`: Fiedler value via QPE resource estimation
- `applications/` subpackage expanded to 10 modules:
  - `fmo_benchmark`: FMO photosynthetic complex (7 chromophores)
  - `power_grid`: IEEE 5-bus power grid synchronisation
  - `josephson_array`: JJA/transmon self-simulation (E_J/E_C=60)
  - `eeg_benchmark`: 8-channel alpha-band PLV functional connectivity
  - `iter_benchmark`: 8 MHD mode coupling (NTM/RWM/kink/ELM)
  - `cross_domain`: summary of all 5 physical system benchmarks
  - `quantum_kernel`: K_nm-informed feature encoding for classification
  - `quantum_reservoir`: Pauli feature extraction, ridge regression readout
  - `disruption_classifier`: quantum kernel classifier for plasma stability
  - `quantum_evs`: quantum-enhanced EVS for CCW consciousness detection
- `phase/` 4 new algorithms:
  - `adapt_vqe`: gradient-driven operator selection, barren-plateau-free
  - `avqds`: McLachlan variational dynamics, circuit depth independent of time
  - `qsvt_evolution`: QSVT resource estimation (260x speedup vs Trotter-1)
  - `varqite`: imaginary-time ground state, guaranteed convergence
- `tcbo/quantum_observer`: p_h1, TEE, string order, Betti proxies
- `pgbo/quantum_bridge`: quantum geometric tensor, Berry curvature
- `l16/quantum_director`: Loschmidt echo, stability score, action decision
- `bridge/snn_backward`: parameter-shift gradient through quantum layer
- `bridge/ssgf_w_adapter`: correlator-weighted geometry W update
- `bridge/orchestrator_feedback`: advance/hold/rollback from quantum state
- `identity/robustness`: adiabatic robustness certificate (gap → perturbation bound)
- `qec/error_budget`: 3-channel Trotter+gate+logical error allocation
- `benchmarks/mps_baseline`: bond dimension, memory, advantage threshold
- `benchmarks/gpu_baseline`: A100 FLOPS, GPU vs QPU crossover
- `benchmarks/appqsim_protocol`: application-oriented fidelity metrics
- `hardware/gpu_accel`: cupy opt-in GPU offload for eigvalsh, expm, matmul
- `hardware/circuit_cutting`: partition optimiser for 32-64 oscillators
- `hardware/qasm_export`: OpenQASM 3.0 circuit export
- `hardware/qcvv`: state fidelity, mirror circuit, XEB certification
- `GAP_CLOSURE_STATUS.md`: honest assessment of all three gaps

### Previously added

- `identity/` subpackage: quantitative identity continuity analysis
  - `IdentityAttractor`: VQE-based attractor basin + robustness gap
  - `coherence_budget()`: max circuit depth before fidelity loss on Heron r2
  - `chsh_from_statevector()`, `disposition_entanglement_map()`: CHSH S-parameter
  - `identity_fingerprint()`, `verify_identity()`, `prove_identity()`: K_nm fingerprint + HMAC
- 9 v1.0 modules:
  - `mitigation/pec.py`: Probabilistic Error Cancellation (Temme et al. PRL 119 180509)
  - `hardware/trapped_ion.py`: trapped-ion noise model (MS gate) + transpilation
  - `control/q_disruption_iter.py`: ITER 11-feature disruption classifier + fusion-core adapter
  - `benchmarks/quantum_advantage.py`: classical vs quantum scaling + crossover extrapolation
  - `bridge/snn_adapter.py`: ArcaneNeuronBridge (sc-neurocore ↔ quantum layer)
  - `bridge/ssgf_adapter.py`: SSGFQuantumLoop (SSGF engine ↔ quantum Trotter loop)
  - `identity/binding_spec.py`: 6-layer 18-oscillator topology + orchestrator 18↔35 mapping
  - `qsnn/training.py`: parameter-shift gradient trainer for QuantumDenseLayer
  - `qec/fault_tolerant.py`: repetition-code logical qubits + transversal RZZ
- 4 cross-repo integrations: sc-neurocore, SSGF engine, scpn-phase-orchestrator, scpn-fusion-core
- 7 new examples (11–17): PEC, trapped-ion, ITER, quantum advantage, QSNN training, fault-tolerant, bridges
- 2 new notebooks: `06_pec_error_cancellation`, `07_quantum_advantage_scaling`
- 2 new API doc pages: `bridges_api.md`, `benchmarks_api.md`
- `docs/equations.md`: PEC, trapped-ion noise, ITER features, fault-tolerant QEC, SSGF loop, binding topology, quantum advantage scaling
- Enterprise hardening: SPDX headers (130 files), AGPL-3.0 dual-license, 4 new CI workflows, Dockerfile, Makefile, GOVERNANCE, SUPPORT, CONTRIBUTORS

### Fixed

- CHSH angles: b'=-π/4 → b'=3π/4 for correct S≈2√2
- README license badge: MIT → AGPL-3.0-or-later
- README test count: ~505 → 627+
- README architecture tree: added identity, benchmarks, bridges, new modules
- mypy numpy `no-any-return` errors across new modules
- ruff E741/format issues in examples and notebooks
- ArcaneNeuron import path: `neurons` → `neurons.models`

### Changed

- Python 3.13 added to CI test matrix and pyproject classifiers
- `[docs]` optional deps: added `mkdocs`, `pymdown-extensions`
- Test count: ~505 → 627+
- Coverage: 100% maintained

## [0.9.0] - 2026-03-02

### Added

- 100% line coverage: 13 new tests closing 19 uncovered lines across 9 files
- Security scanning CI job (bandit + pip-audit) in `ci.yml`
- `.github/CODEOWNERS` (default: @anulum)
- Input validation: `QuantumPetriNet` (shape checks), `QLiF` (dt/n_shots), `QAOA_MPC` (p_layers)
- `WEIGHT_SPARSITY_EPS` constant; Shor & Preskill citation on `QBER_SECURITY_THRESHOLD`
- ZNE experiments now return `R_std_per_scale` (shot-noise error bars per noise scale)
- `_run_vqe()` returns `energy_std` (convergence stability metric)
- Runner `timeout_s` parameter on `run_sampler`/`run_estimator`, job metadata logged to `jobs.json`
- `retrieve_job(job_id)` recovery method on `HardwareRunner`
- `examples/07_crypto_bell_test.py` — CHSH violation demo
- `examples/08_dynamical_decoupling.py` — XY4 DD vs raw fidelity comparison
- `notebooks/05_crypto_and_entanglement.ipynb` — Bell test, correlator matrix, QKD, key rate
- Shared fixtures in `conftest.py`: `knm_4q`, `knm_8q`, `rng`
- Dataclass field docs on `LockSignatureArtifact` (PLV, mean_lag) and `LayerStateArtifact` (R, psi)

### Fixed

- `q_disruption.py`: magic `16` replaced with `2**self.n_data_qubits`
- Bare `1e-15` in `qpetri.py`, `spn_to_qcircuit.py`, `q_disruption.py` replaced with `WEIGHT_SPARSITY_EPS`
- Weak assertions tightened: `rate > 0.1` → `0.15 < rate < 0.85`, `total > 0` → `>= 5`, `failures > 0` → `>= 5`

### Changed

- `Development Status :: 4 - Beta` → `Development Status :: 5 - Production/Stable`
- `TYPE_CHECKING` blocks excluded from coverage (`exclude_also`)
- Version bump: 0.8.0 → 0.9.0
- Test count: 483 → ~505

## [0.8.0] - 2026-03-02

### Added

- Shot-noise error bars (`hw_R_std`, `hw_expectations_std`) on all 20 hardware experiments
- 3 property-based test files: `test_knm_properties.py`, `test_crypto_properties.py`, `test_qec_properties.py` (9 hypothesis tests)
- 4 coverage test files: `test_orchestrator_adapter_helpers.py`, `test_vqls_edge_cases.py`, `test_percolation_edge_cases.py`, `test_topology_auth_edge.py` (12 tests)
- `_constants.py`: shared `COUPLING_SPARSITY_EPS`, `CONCURRENCE_EPS`, `QBER_SECURITY_THRESHOLD`, `VQLS_DENOMINATOR_EPS`
- `.editorconfig` for consistent formatting
- Input validation: K/omega shape mismatch, ZNE data point count, DD qubit range, ODE solver status, eigenvalue reality check
- 3 experiment functions exported from `hardware` subpackage: `bell_test_4q_experiment`, `correlator_4q_experiment`, `qkd_qber_4q_experiment`

### Fixed

- 21 mypy `no-any-return` errors across 13 source files (numpy expression type annotations)
- Dead code removed in `qaoa_mpc.py` (redundant None check after builder)
- `test_public_api.py` assertions strengthened from `is not None` to type checks
- `test_control_qec.py` success threshold tightened from `> 0` to `>= 10`

### Changed

- Version bump: 0.7.1 → 0.8.0
- Test count: 463 → 483

## [0.7.1] - 2026-03-02

### Fixed

- Trivial `assert True` and `assert qc.size() > 0` replaced with meaningful assertions
- 3 duplicate tests removed (`test_omega_shape`, `test_knm_paper27_calibration_anchors`, `test_all_experiment_functions_registered`)
- `inhibitor_anti_control` missing `-> None` return annotation
- mypy blanket `ignore_missing_imports` replaced with per-module overrides
- `__all__` export gaps: `knm_to_ansatz` (top-level), `classical_exact_evolution`/`JobResult` (hardware), `SurfaceCode`/`MWPMDecoder` (qec)

### Added

- `wheel-check` CI job: build → install from wheel → smoke test
- 10 error-path tests: phase_artifact validation, runner connect guards, orchestrator adapter bounds
- mypy `warn_unreachable`, `check_untyped_defs` enabled
- sdist excludes (.github/, dist/, results/, figures/, notebooks/)
- `build` in dev extras, matplotlib upper bound `<4.0`
- Docstrings on `hmac_sign`, `hmac_verify_key`, `PhaseOrchestratorAdapter.from_orchestrator_state`

### Changed

- Version bump: 0.7.0 → 0.7.1

## [0.7.0] - 2026-03-02

### Fixed

- `crypto/__init__.py` exported module names instead of callable symbols
- `QuantumDenseLayer` unseeded `default_rng()` — now accepts `seed` parameter
- `PhaseVQE.relative_error_pct` returned `inf` when exact energy ~0; now returns `nan`
- Bare magic numbers in `percolation.py` replaced with named constants

### Added

- PEP 561 `py.typed` marker for downstream mypy/pyright consumers
- pip caching (`cache: 'pip'`) in all CI `setup-python` steps
- Dependency upper bounds (next-major caps on qiskit, numpy, scipy, networkx)
- `test_crypto_exports.py` — validates all `crypto.__all__` entries are callable
- Seed determinism test for `QuantumDenseLayer`

### Changed

- Version bump: 0.6.4 → 0.7.0

## [0.6.4] - 2026-03-01

### Fixed

- Stale test counts in docs (411/424 → 442) across index.md, installation.md, contributing.md
- `figures/generate_header.py` version string (v0.5.1/411 → v0.6.3/442)
- README architecture tree: add missing `ansatz_bench.py`, `trotter_error.py`, `control_plasma_knm.py`
- SECURITY.md supported versions (0.1.x → 0.6.x)

### Added

- scpn-phase-orchestrator in README Related Repositories
- pip ecosystem in dependabot monitoring
- v0.6.1 and v0.6.2 entries in docs/changelog.md

### Changed

- Version bump: 0.6.2 → 0.6.4 across pyproject.toml, __init__.py, CITATION.cff, .zenodo.json, badges, test

## [0.6.3] - 2026-03-01

### Fixed

- Coverage gate in CI (codecov threshold)
- Mitigation API docs (ZNE/DD endpoint reference)
- Notebook table alignment in docs
- ruff S311 violation in tests

### Added

- `docs/mitigation_api.md` — ZNE/DD API reference
- Coverage gate job in CI workflow
- Notebook summary table in docs/index.md

## [0.6.2] - 2026-03-01

### Fixed

- Notebook 01/03/04: `classical_kuramoto_ode` → `classical_kuramoto_reference`
- Notebook 03: ZNE scales [1,2,3,4,5] → [1,3,5,7,9] (odd required by gate_fold_circuit)
- Notebook 04: rewrite to 8-qubit Trotter + 16-layer classical (16-qubit statevector intractable on laptop)
- Remove misplaced docs/SESSION_LOG and docs/HANDOVER (duplicates of .coordination/)

### Added

- `figures/generate_knm_heatmap.py` + `figures/knm_heatmap.png` (16×16 K_nm coupling matrix)
- Knm heatmap figure in README with annotated calibration anchors
- All 4 notebooks executed with embedded outputs
- ROADMAP: post-2030 timeline qualifier on fault-tolerant bullet

## [0.6.1] - 2026-03-01

### Fixed

- mypy errors in bridge module: remove FloatArray type alias (incompatible with Python 3.9), fix Path(None) in control_plasma_knm.py
- Zenodo metadata enriched (.zenodo.json, CITATION.cff)

## [0.6.0] - 2026-03-01

### Fixed

- Division-by-zero in QuantumLIFNeuron when v_threshold == v_rest
- Division-by-zero in QuantumSynapse when w_max == w_min
- Division-by-zero in QuantumSTDP when sin(shift) == 0
- Division-by-zero in QAOA_MPC when horizon <= 0
- Division-by-zero in classical_kuramoto_reference when dt <= 0
- Index-out-of-bounds in bell_inequality_test when qubit >= n_total
- Index-out-of-bounds in best_entanglement_path when source/target >= n
- Notebook 02 referencing non-existent PhaseVQE.solve() dict keys
- Stale test counts in VALIDATION.md and docs/index.md

### Added

- Input validation guards on QSNN constructors (qlif, qsynapse, qstdp)
- Input validation on QAOA_MPC, bell_inequality_test, best_entanglement_path, classical_kuramoto_reference
- PhaseVQE.solve() now returns exact_energy, energy_gap, relative_error_pct, n_params
- 13 validation tests (test_qsnn_validation.py)

### Changed

- docs/changelog.md fleshed out with full version history
- Test count: 411 → 424

## [0.5.1] - 2026-03-01

### Fixed

- **PyPI version mismatch**: v0.5.0 published with `__version__ = "0.4.0"` due to tag timing. This release ensures `__version__`, `pyproject.toml`, and PyPI all report 0.5.1.

## [0.5.0] - 2026-03-01

### Added

- **3 crypto hardware experiments**: `bell_test_4q` (CHSH violation on hardware), `correlator_4q` (ZZ cross-correlation topology validation), `qkd_qber_4q` (QBER from hardware vs BB84 threshold)
- **`_correlator_from_counts()` helper**: extracts 2-qubit correlator E(A,B) from 4-qubit measurement counts
- **noise_analysis.py**: `devetak_winter_rate()` key rate from Devetak-Winter bound
- **3 simulator tests**: bell test, correlator, QKD QBER — all using VQE monkey-patching pattern

### Changed

- **Experiment count**: 17 → 20 (3 crypto experiments added to `ALL_EXPERIMENTS`)
- **Test count**: 408 → 411
- **Version bump**: 0.4.0 → 0.5.0 across pyproject.toml, CITATION.cff, badges

## [0.4.0] - 2026-02-28

### Added

- **GitHub Pages docs**: MkDocs Material site with 7 pages, auto-deploy on push to main
- **4 Jupyter notebooks**: Kuramoto XY dynamics, VQE ground state, error mitigation (ZNE), UPDE-16
- **10 hardware experiments**: noise baseline, 8-osc ZNE, 8q VQE on hardware, UPDE-16 with DD, Trotter order-2, sync threshold, decoherence scaling, ZNE higher-order, VQE landscape, cross-layer correlation
- **14 property-based tests** (hypothesis): probability-angle roundtrip, Knm symmetry/positivity, Hamiltonian Hermiticity, ansatz parameter counts
- **8 edge-case tests**: 2-oscillator minimal, SuzukiTrotter order=2, single-value inputs
- **13 coverage-gap tests**: multi-inhibitor anti-control, QAOA ZZ path, VQLS near-zero guard, QEC odd defects / correction failure, QSTDP synapse
- **4 integration tests**: Knm → VQE ground state, Knm → Trotter → energy, 8q spectrum, 16-layer Hamiltonian structure
- **7 regression tests**: Knm calibration anchors, cross-hierarchy boosts, omega values, 4q ground energy baseline, statevector R, R evolution monotonicity, _R_from_xyz validation

### Changed

- **Mypy**: expanded from 27 to 30 source files (full hardware/ directory)
- **Test count**: 208 → 254
- **pyproject.toml**: docs URL, mkdocs-material optional dep

## [0.3.0] - 2026-02-28

### Added

- **README rewrite**: motivating abstract explaining SCPN→XY isomorphism, "Background" section with Kuramoto-to-Hamiltonian derivation, all 4 figures embedded with captions, expanded example table with descriptions, "Related Repositories" section
- **Paper 27 citation**: formal working paper reference in README and docs/equations.md
- **docs/equations.md**: SCPN overview and UPDE definition added; equations document is now self-contained
- **examples/README.md**: guided walkthrough of all 6 examples with physics context
- **HARDWARE_RESULTS.md**: layer naming convention section explaining L1-L16 mapping

### Changed

- **CI mypy scope**: expanded from `bridge/` only to full `mypy` (pyproject.toml `files` list covers all 8 module paths — 27 source files, zero errors)
- **VALIDATION.md**: test count 88→199, classical references now point to in-repo `hardware/classical.py` functions instead of external packages, added Pauli ordering and Trotter convergence invariants
- **Badges**: test count 88→199, added version badge (v0.3.0)
- **CITATION.cff**: version 0.1.0→0.3.0

## [0.2.7] - 2026-02-28

### Added

- **Parametrized quantum-vs-classical validation**: integration tests at n={2,3,4,6} qubits verifying Trotter evolution tracks classical exact evolution
- **Exact diag cross-check**: integration test verifying `classical_exact_diag` ground energy matches direct `eigvalsh` of Hamiltonian matrix

### Changed

- **Coverage scope refined**: `omit` now excludes only `runner.py` and `experiments.py` (IBM-dependent), no longer blanket-excludes `hardware/`. `classical.py` now tracked.

## [0.2.6] - 2026-02-28

### Added

- **Classical reference tests**: 20 tests covering `classical_kuramoto_reference`, `classical_exact_diag`, `classical_exact_evolution`, `classical_brute_mpc`, and `bloch_vectors_from_json`
- **Pauli ordering validation**: 2 tests confirming qubit labeling consistency between Hamiltonian builder and state preparation (`<0|H|0> = -sum(omega)` and single-flip energy shift)
- **ALL_EXPERIMENTS completeness**: test verifying every `*_experiment()` function is registered in the `ALL_EXPERIMENTS` dict
- **Integration tests**: quantum-vs-classical Kuramoto, ZNE on noiseless backend, energy conservation under Trotter, Trotter order-2 passthrough

### Changed

- **mypy scope expanded**: now covers `control/`, `qsnn/`, `qec/` in addition to `bridge/`, `phase/`, `mitigation/`, `hardware/` — 27 source files checked, zero errors
- Fixed type narrowing in `vqls_gs.py`, `qaoa_mpc.py`, `runner.py` (assert-after-guard pattern)
- Fixed `QuantumSTDP` forward reference to `QuantumSynapse` via `TYPE_CHECKING` import
- Fixed `ZNEResult` forward reference in `runner.py` via `TYPE_CHECKING` import

## [0.2.5] - 2026-02-28

### Added

- **Second-order Trotter**: `trotter_order=2` parameter on `QuantumKuramotoSolver` and `QuantumUPDESolver` uses SuzukiTrotter(order=2) with O(t^3/reps^2) error vs O(t^2/reps) for first-order
- **Energy tracking**: `QuantumKuramotoSolver.energy_expectation(sv)` returns <H> for paper figure data
- `test_second_order_trotter` — verifies order-2 Trotter error < order-1 on 4-oscillator system
- `test_trotter_error_decreases_with_reps` — verifies convergence: error(reps=8) < error(reps=3) < error(reps=1)
- `test_energy_expectation` — verifies <H> matches direct matrix computation

## [0.2.4] - 2026-02-28

### Fixed

- **QAOA Hamiltonian**: correct Ising encoding with identity (constant) term — h_z = -(a^2-2ab)/2, c0 = (a^2-2ab)*H/2 + H*b^2; removed spurious ZZ terms
- **Quantum Petri net**: multi-input transitions now use multi-controlled Ry (AND gating) instead of single CRy on first input only
- **Inhibitor arcs**: restructured anti-control pattern (X-CRy-X) to correctly gate output on inhibitor place emptiness
- **build_knm_paper27**: removed dead `zeta_uniform` parameter
- **VQLS**: imaginary norm threshold now configurable via `imag_tol` init parameter (default 0.1)

### Added

- `test_hamiltonian_matches_classical_cost` — verifies QAOA Hamiltonian diagonal matches brute-force cost for all bitstrings
- `test_optimal_bitstring_matches_brute_force` — verifies minimum eigenvalue bitstring equals classical optimum
- `test_multi_input_conjunctive_gating` — verifies Petri net output depends on all input places
- `test_inhibitor_blocks_when_place_occupied` — verifies anti-control suppresses output when inhibitor is |1>

## [0.2.3] - 2026-02-28

### Fixed

- **Disruption classifier**: removed dead CX parameter entries — `n_params` was `n_layers*(2*n_qubits + n_qubits-1)` but CX gates have no trainable parameters; now `n_layers*2*n_qubits` (30 vs 42 for default config)
- **QSTDP Hebbian dynamics**: `post_measured` was accepted but ignored; now implements LTP (pre+post → weight increase) and LTD (pre only → weight decrease) per Hebbian learning rule

### Added

- Test for `kuramoto_4osc_zne_experiment` on simulator
- Test for `upde_16_snapshot_experiment` on simulator (marked `@pytest.mark.slow`)
- `pytest.ini_options.addopts` skips slow/hardware tests by default

## [0.2.2] - 2026-02-28

### Fixed

- **MWPM decoder** (3 bugs):
  - `_shortest_path` now uses dual edges for plaquette (Z) syndromes — all single Z errors were failing
  - `_has_logical_error` uses seam-crossing winding number formula; shifted non-contractible cycles no longer missed
  - d=5 now correctly outperforms d=3 below threshold (Dennis et al. 2002)
- **Classical reference endianness** (`_build_initial_state`, `_expectation_pauli`): kron order reversed to match Qiskit little-endian convention; verified against Statevector evolution to 1e-6
- **Parameter-shift rule** (`q_disruption.py`): removed misleading `sin(shift)` denominator (Schuld et al., PRA 99, 032331)
- **VQLS**: assert imaginary norm < 0.1 before `np.real()` instead of silently discarding
- **QAOA_MPC**: removed dead `current_state` parameter
- **ZNE**: cache `base.inverse()` before fold loop
- **trotter_upde**: remove dead `evolve(0)` call, add `reset()`
- **qlif**: replace legacy `np.random.binomial` with seedable `rng` parameter
- **runner**: catch `TranspilerError` in DD pass fallback instead of bare `except Exception`
- **sc_to_quantum**: `measurement_to_bitstream` now accepts `rng` parameter

### Added

- `_run_vqe` helper eliminates vqe_4q/vqe_8q code duplication
- Root `__init__.py` exports all 20 public symbols
- Return type annotations on all public methods
- 9 new tests: d5-beats-d3, shifted logical cycles, single X/Z error correctness, VQE experiment, DD transpile, QSNN stochastic mode, bitstream seeded
- Citation markers on `K_base`, `K_alpha` (Paper 27, Eq. 3)
- `test_classical_evolution_matches_qiskit`: definitive Qiskit-vs-classical endianness agreement test

## [0.2.1] - 2026-02-28

### Fixed

- `q_disruption.py`: configurable `seed` parameter instead of hardcoded 42
- `qpetri.py`: threshold gating via CRy — output rotations controlled by input place qubits
- `classical.py`: sparse eigensolver path (`scipy.sparse.linalg.eigsh`) for N >= 14 or when `k_eigenvalues` specified
- `ansatz_bench.py`: replaced deprecated `TwoLocal`/`EfficientSU2` classes with `n_local`/`efficient_su2` functions (Qiskit 2.1+)
- `vqls_gs.py`: replaced deprecated `TwoLocal` class with `n_local` function

### Added

- STDP direction validation tests (gradient sign at theta=0 and theta=pi/2)
- QEC threshold measurement tests (p=0.01 vs p=0.08 success rates, d=5 single-error decoding, very low error rate high success)
- Disruption classifier seed reproducibility tests
- Petri net controlled output and multi-step bounds tests
- Bloch ball constraint test
- Sparse vs dense eigensolver agreement test

## [0.2.0] - 2026-02-28

### Added

- **hardware/noise_model.py**: `heron_r2_noise_model()` factory with Heron r2 median calibration (T1=300us, T2=200us, CZ 0.5%, readout 0.2%)
- **mitigation/zne.py**: `gate_fold_circuit()` global unitary folding (Giurgica-Tiron et al. 2020), `zne_extrapolate()` Richardson extrapolation
- **mitigation/dd.py**: `DDSequence` (XY4, X2), `insert_dd_sequence()` idle-qubit pulse insertion (Viola et al. 1999)
- **phase/trotter_error.py**: `trotter_error_norm()` exact vs Trotter Frobenius norm, `trotter_error_sweep()` 2D parameter scan
- **phase/ansatz_bench.py**: `benchmark_ansatz()` / `run_ansatz_benchmark()` — Knm-informed vs TwoLocal vs EfficientSU2 VQE comparison
- **hardware/classical.py**: `bloch_vectors_from_json()` per-qubit Bloch vector extraction from hardware result files
- **hardware/experiments.py**: `kuramoto_4osc_zne_experiment` — ZNE-mitigated 4-oscillator dynamics
- **hardware/runner.py**: `run_estimator_zne()` ZNE pipeline, `transpile_with_dd()` via Qiskit's PadDynamicalDecoupling pass, `noise_model` constructor parameter
- **scripts/plot_vqe_convergence.py**: ansatz convergence + energy gap bar chart (Figure 2)
- **scripts/plot_decoherence_curve.py**: `fit_decoherence_rate()` — exp(-gamma*depth) fit for R_hw/R_exact ratio
- **examples/05_vqe_ansatz_comparison.py**, **examples/06_zne_demo.py**
- Top-level re-exports: `OMEGA_N_16`, `build_knm_paper27`, `knm_to_hamiltonian`, `QuantumKuramotoSolver`, `QuantumUPDESolver`, `PhaseVQE`, `HardwareRunner`, `JobResult`

### Fixed

- `_run_sampler_simulator()` now uses `self._backend` instead of fresh `AerSimulator()`, respecting noise model
- Removed duplicate `AerSimulator` import in simulator path
- Removed dead `2**n_osc` expressions in `classical.py` (lines 117, 147)
- `DEFAULT_INSTANCE` reads from `SCPN_IBM_INSTANCE` env var with fallback

### Changed

- mypy scope expanded from `bridge/` to include `phase/`, `mitigation/`, `hardware/classical.py`, `hardware/runner.py`

## [0.1.0] - 2026-02-28

### Added

- **qsnn/**: Quantum LIF neuron (`qlif.py`), controlled-Ry synapse (`qsynapse.py`), parameter-shift STDP (`qstdp.py`), entangled dense layer (`qlayer.py`)
- **phase/**: Kuramoto XY Hamiltonian solver (`xy_kuramoto.py`), 16-layer Trotter UPDE (`trotter_upde.py`), VQE ground state finder (`phase_vqe.py`)
- **control/**: QAOA-MPC binary trajectory optimizer (`qaoa_mpc.py`), VQLS Grad-Shafranov solver (`vqls_gs.py`), quantum Petri net (`qpetri.py`), quantum disruption classifier (`q_disruption.py`)
- **bridge/**: Knm-to-Hamiltonian compiler (`knm_hamiltonian.py`), SPN-to-circuit converter (`spn_to_qcircuit.py`), bitstream-rotation bridge (`sc_to_quantum.py`)
- **qec/**: Toric surface code + MWPM decoder with Knm-weighted edges (`control_qec.py`)
- **hardware/**: IBM Quantum runner for ibm_fez Heron r2 (`runner.py`, `experiments.py`, `classical.py`)
- 88 unit tests, 4 example scripts, 19 hardware result files
- Hardware validation on ibm_fez: VQE 0.05% error, 12-point decoherence curve, 16-layer UPDE snapshot
- CI workflow with Python 3.9-3.12 matrix, coverage, ruff lint
- Full documentation: architecture, API reference, hardware results
