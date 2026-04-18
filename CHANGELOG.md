# Changelog

Dated list of changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- 2026-04-18 — `qiskit` pin bumped `>=1.0,<2.0` → `>=2.2,<3.0`; `qiskit-aer` pin bumped `>=0.14,<1.0` → `>=0.15,<1.0`; `qiskit-ibm-runtime` pin bumped `>=0.20,<1.0` → `>=0.40,<1.0`. Misleading `PauliEvolutionGate bug (GH #15476)` comment removed — upstream closed that report as not-a-bug (the 2.2 change correctly treats `PauliEvolutionGate` as an abstract exact-evolution object; pre-2.2 was silently injecting Trotter error). Full pytest suite: 5045 passed / 0 failed on qiskit 2.4.0 without code changes.

## [0.9.6] - 2026-04-17

### Security
- Bumped `rand` 0.9.2 → 0.9.4 (RUSTSEC-2026-0097).
- Removed IBM Cloud CRN prefix log in `scripts/retrieve_ibm_job.py`.
- Bumped `pytest` 9.0.2 → 9.0.3 (CVE-2025-71176).

### Added
- Input-validation guards + `MAX_OSCILLATORS_DEFAULT = 32` cap in `analysis/koopman.py`.
- `docs/pipeline_performance.md` §21 (measured Rust speedups + new-backend decision rule).
- `docs/language_policy.md` (multi-language accel chain: Rust + Julia + Go + Mojo + Python floor).
- `docs/triage.md` (issue triage policy and label taxonomy).
- `docs/falsification.md`, `docs/preregistration.md`, `docs/THREAT_MODEL.md`, `docs/EXPORT_CONTROL.md`, `docs/adopters.md`, `docs/datasets.md`, `docs/mutation_testing.md`, `DEPRECATIONS.md`.
- `.well-known/security.txt` (RFC 9116).
- `src/scpn_quantum_control/config.py` — `SCPNConfig` via `pydantic-settings`.
- `src/scpn_quantum_control/logging_setup.py` — `structlog` bootstrap.
- `src/scpn_quantum_control/hardware/backends.py` — entry-points backend registry.
- `src/scpn_quantum_control/hardware/async_runner.py` — `AsyncHardwareRunner` over `asyncio`.
- `src/scpn_quantum_control/accel/` — multi-language dispatcher + Julia tier (`order_parameter`).
- `src/scpn_quantum_control/hardware/provenance.py` — `capture_provenance()`.
- `tests/test_phase1_dla_parity_reproduces.py` (8 tests, Phase 1 reproducer).
- `tests/test_cross_validation_qutip_dynamiqs.py` (43 tests, XY Hamiltonian vs QuTiP / Dynamiqs).
- `tests/test_pennylane_vendor_backends.py` (67 tests, mocked IBM / IonQ / Rigetti / Quantinuum / Braket / Cirq).
- `tests/test_backend_registry.py` (26 tests).
- `tests/test_config.py` (20 tests).
- `tests/test_logging_setup.py` (19 tests).
- `tests/test_async_runner.py` (17 tests, mocked `qiskit_ibm_runtime`).
- `tests/test_accel_dispatch.py` (27 tests, Rust → Julia → Python floor).
- `tests/test_phase_artifact_fuzz.py` (18 Hypothesis tests).
- `tests/test_hardware_classical_fuzz.py` (13 Hypothesis tests).
- `tests/test_qec_validators_fuzz.py` (15 Hypothesis tests).
- `tests/test_perf_regression.py` (Rust ≥ 2× Python floor).
- `tests/test_provenance.py` (11 tests).
- `scripts/bench_order_parameter_tiers.py` + `docs/benchmarks/order_parameter_tiers.json`.
- `scpn_quantum_engine/benches/hot_paths.rs` (Criterion harness) + `.github/workflows/rust-benches.yml`.
- `.github/workflows/notebooks.yml`, `link-check.yml`, `docs-strict.yml`, `sbom.yml`, `mutation-testing.yml`, `commit-trailers.yml`.
- `tools/check_commit_trailers.py`, `tools/mutmut_runner.sh`.
- `pyproject.toml` extras: `xvalidate`, `config`, `logging`, `julia`; entry-point group `scpn_quantum_control.backends`.
- `.coordination/launch_copy/` — 9 visibility drafts (HN / r/QuantumComputing / r/qiskit / Slack / Discord / LinkedIn / X / arXiv / README).

### Changed
- CI dev-tool matrix (`pytest 9.0.3`, `mypy 1.20.1`, `ruff 0.15.10`, `hypothesis 6.151.13`, `build 1.4.3`, `actions/upload-artifact 7.0.1`, `pypa/gh-action-pypi-publish 1.14.0`).
- `_order_param` in `hardware/classical.py` now dispatches through the accel chain.
- Five remaining `print()` calls in `hardware/runner.py` → `structlog` events.
- `HardwareRunner.DEFAULT_INSTANCE` → `HardwareRunner._default_instance()` (reads SCPNConfig).
- Stale counts refreshed in `README.md`, `docs/architecture.md`, `docs/index.md`, `docs/pipeline_performance.md`, `docs/test_infrastructure.md`.
- Self-applied quality labels scrubbed across `CHANGELOG.md`, `docs/changelog.md`, `docs/test_infrastructure.md`, `docs/symmetry_decay_guess.md`, `docs/dynq_qubit_mapping.md`, and 27 test docstrings.

### Repository hygiene
- `.coordination/sessions/` and `.coordination/handovers/` untracked; local-only going forward.
- `.gitignore` patterns for paper-extraction working files, `.agent_metadata.json`, root-level `handover_*.md`, `.coordination/refactor_backups/`, `.coordination/contributing.md`.
- Agent-name mentions stripped from public-facing tracked files (`CHANGELOG.md`, `docs/triage.md`, `docs/PAPER_CLAIMS.md`, `docs/changelog.md`, `figures/generate_ansatz_comparison.py`, `tests/test_koopman.py`, and nine files referencing the internal audit filename).

## [0.9.5] - 2026-03-29 / 2026-04-11

### Added
- 2026-04-10: Phase 1 IBM Quantum hardware campaign on `ibm_kingston` (Heron r2, 156 qubits). 348 circuits, up to 21 reps per (depth, sector) at n = 4. Mean DLA-parity asymmetry +10.8 % for depths ≥ 4, peak +17.48 % at depth 6. Fisher combined χ²(16) = 123.4, p ≪ 10⁻¹⁶. Apriori simulator band was 4.5–9.6 %.
- `scripts/analyse_phase1_dla_parity.py`, `paper/phase1_dla_parity_short_paper.md`.
- IBM execution scripts: `pipe_cleaner_ibm_kingston.py`, `phase1_mini_bench_ibm_kingston.py`, `phase1_5_reinforce_ibm_kingston.py`, `phase2_exhaust_cycle_ibm_kingston.py`, `phase2_5_final_burn_ibm_kingston.py`, `phase2_full_campaign_ibm.py`, `micro_probe_ibm_kingston.py`, `retrieve_ibm_job.py`.
- `.coordination/IBM_CAMPAIGN_STATE.md`, `IBM_EXECUTION_LOG.md`, `phase1_experiment_design.md`, `WEBMASTER_CONTEXT.md`.
- 2026-04-08: `mitigation/symmetry_decay.py` + `scpn_quantum_engine/src/symmetry_decay.rs` — GUESS symmetry-decay ZNE (Oliva del Moral et al., arXiv:2603.13060, 2026). 20 tests.
- `hardware/qubit_mapper.py` + `scpn_quantum_engine/src/community.rs` — DynQ topology-agnostic mapper (Liu et al., arXiv:2601.19635). 17 tests.
- `phase/pulse_shaping.py` + `scpn_quantum_engine/src/pulse_shaping.rs` — PMP / ICI pulse sequences (Liu et al. 2023). Rust `ici_three_level_evolution_batch` 1 665× vs Python.
- `(α,β)`-hypergeometric pulse family via Gauss ${}_2F_1$ (Ventura Meinersen et al., arXiv:2504.08031, 2025). Rust 44× vs scipy. 25 tests for ICI + hypergeometric.
- FFI boundary hardening across all 36 `#[pyfunction]` exports (`PyResult<T>`, `validate_n`, `validate_positive`, `validate_range`, `validate_finite`, `validate_flat_square`, `validate_statevec_len`, `validate_domain_range`). 16 `validation.rs` tests.
- `docs/symmetry_decay_guess.md` (891 lines), `docs/dynq_qubit_mapping.md` (878 lines).
- 2026-04-10: `tools/check_secrets.py` (vault-pattern scanner) + gitleaks v8.21.2 pre-commit hook.
- `.coordination/incidents/INCIDENT_2026-04-10T2336_ftp_creds_in_webmaster_context.md`.
- `.gitignore` patterns for `.venv-linux/`, `.venv-rocm/`, `.venv-cuda/`, `results/`, `.coordination/TODO_*.md`, `.coordination/*.pdf`.

### Changed
- Tests collected: 2 813 → 4 828 (97 %+ coverage).
- Python modules: 165 → 201; subpackages 17 → 19.
- Rust functions exported: 22 → 36 across 20 source files.
- Hardware reference: `ibm_fez` → `ibm_fez + ibm_kingston` (Feb + Apr 2026).
- `runner.py`: counts extraction now tries `meas`, `c`, `cr`, `c0` then introspects `DataBin`.

### Fixed
- 2026-04-10: SamplerV2 result parsing no longer hard-codes the classical-register name as `meas`.

## [0.9.5] - 2026-03-29 / 2026-04-07

### Added
- 2026-04-06 / -07: `qec/multiscale_qec.py`, `qec/syndrome_flow.py` — concatenated surface codes across 5 SCPN domains. 23 tests.
- `fep/variational_free_energy.py`, `fep/predictive_coding.py` — Friston 2010 variational F. 16 tests.
- `psi_field/lattice.py`, `psi_field/infoton.py`, `psi_field/scpn_mapping.py`, `psi_field/observables.py` — U(1) compact gauge theory with HMC. 22 tests.
- K_nm validation on IEEE 5-bus (ρ = 0.881), Josephson (ρ = 0.990), EEG (ρ = 0.916), ITER MHD (ρ = 0.944).
- `ripser` under `[topology]`.
- 2026-03-29 / 2026-04-01: dynamic-coupling engine (Quantum Hebbian learning).
- Topological coupling optimiser + `HardwareTopologicalOptimizer` (persistent homology as loss).
- `BiologicalSurfaceCode` — stabilisers on the 16-layer SCPN graph.
- `LindbladSyncEngine` (MCWF path for large-N).
- EEG PLV → quantum pipeline.
- Sparse evolution engine (bypasses Qiskit overhead).
- `StructuredAnsatz` — topology-informed variational circuits on arbitrary K.
- 27 experiment notebooks (NB14–47, FIM mechanism).
- `test_fim_mechanism.py` — 81 regression tests.
- 25 JSON result files from 27 notebooks.
- IBM hardware v2 — 9 fair-depth experiments on `ibm_fez` (F_FIM = 0.916 vs F_XY = 0.849, p < 10⁻⁶).
- Rust: `correlation_matrix_xy`, `lindblad_jump_ops_coo`, `lindblad_anti_hermitian_diag`, `parity_filter_mask` (18 → 22 total).

### Changed
- `lib.rs` god file (1 436 lines) split into 16 focused modules + 3 new Rust paths (`concat_qec.rs`, `fep.rs`, `gauge_lattice.rs`). 22 → 25 exported functions.
- OTOC: 4.4× faster (n = 4, 50 time points) via O(d) phase rotation.
- Pauli expectations: 2–10× faster via half-loop over paired states.
- Kuramoto order parameter: Rust fast path via `all_xy_expectations`.
- Dockerfile base image pinned by SHA256; `build==1.4.2`, `pip-audit==2.9.0`, `sc-neurocore==3.14.0`.
- Dev: `ruff 0.15.6 → 0.15.9`, `mypy 1.19.1 → 1.20.0`, `hypothesis 6.151.10 → 6.151.11`.
- `multiscale_qec.py` (346 lines) → `multiscale_qec.py` (292) + `syndrome_flow.py` (66).
- Tests: 2 715 → 4 445+; Rust tests: 47 → 65.

### Fixed
- `knm_hamiltonian.py` K-symmetry enforced for Hermiticity.
- All example scripts wrapped in `if __name__ == "__main__":`.
- `delta` parameter propagation in `pairing_correlator.py`, `xxz_phase_diagram.py`.
- Infinite recursion in `knm_to_dense_matrix` fallback.
- `vqe_energy` alias on `PhaseVQE` for backward compatibility.
- 25+ analysis modules migrated to Rust-accelerated dense-matrix path.

## [0.9.4] - 2026-03-29

### Added
- 81 tests across 6 new files (pennylane mock, JAX mock, ripser mock, hardware runner mock, Python fallback, edge cases).
- OpenSSF Best Practices badge (project 12290).
- OpenSSF Scorecard, Ruff, mypy badges.
- 3 benchmark API docs: `gpu_baseline`, `mps_baseline`, `appqsim_protocol`.

### Changed
- Coverage 95 % → 98 % (9 601 stmts, 165 missed).
- Tests: 2 634 → 2 715.
- `docs.yml` `contents:write` scoped to job; `ci.yml` `contents:read` at top level.
- Architecture stats refreshed (155 modules, 23.8k LOC, 29 doc pages).

### Fixed
- PennyLane adapter: catch `Exception` not only `ImportError`.
- 5 test failures: skip guards for unimplemented Rust functions.
- 12 doc errors: function names, test counts, version numbers.
- CI: ripser mock test args; injected missing module attrs.

## [0.9.3] - 2026-03-28

### Added
- Rust: `lanczos_b_coefficients` (27× vs numpy), `otoc_from_eigendecomp` (264× vs scipy), `build_xy_hamiltonian_dense` (5 401× vs Qiskit), `all_xy_expectations` (6.2× vs individual calls). 11 → 15 total.
- IBM hardware campaign 20/20 experiments complete (22 jobs on `ibm_fez`, 176 000+ shots). CHSH S = 2.165 (>8σ), QBER 5.5 %, 16-qubit UPDE. ZNE stable fold 1–9.
- 16 publication figures (simulation + hardware + MBL + BKT).
- 3 publications on GitHub Pages (preprint, sync witnesses, DLA parity).
- `knm_to_dense_matrix` Rust fast path (8 modules migrated).
- JAX GPU backend (`jax_accel.py`) for vectorised coupling scans.
- PyPI Rust wheel CI (`rust-wheels.yml`) for 5 platforms.
- SCPN theory + biochemical foundations pages on GitHub Pages.
- Results gallery on GitHub Pages.
- 12 coverage tests for v0.9.3 additions.
- Kaggle notebook for JAX GPU validation + BKT universality.

### Changed
- README reframed: hardware-first.
- 12 GitHub topics added.
- GitHub Release v0.9.3 created.
- 2 Dependabot PRs merged (codeql-action, codecov-action).

### Fixed
- Empty `pauli_list` crash in `knm_to_xxz_hamiltonian` (Hypothesis edge case).
- Rust parity tests: `pytest.importorskip` for Docker CI.
- JAX backend: build H in numpy/Rust, GPU only for `eigh` / `svd` (was 1 731× slower).
- 3 TokenPermissions OpenSSF Scorecard alerts.
- Version test 0.9.2 → 0.9.3.
- Preprint MBL → non-ergodic correction.

## [0.9.2] - 2026-03-26

### Added
- 38 runner coverage tests (simulator-path methods).
- 22 experiment coverage tests (all 20 experiment functions).
- `requirements.txt` and `requirements-dev.txt` (pinned).

### Changed
- Removed coverage omit for `runner.py` and `experiments.py`.
- README cross-refs updated; hardware campaign status noted.
- ROADMAP: v0.9.1 + v0.9.2 sections.

### Fixed
- Rust engine `build_knm` parity test (wheel rebuilt).

## [0.9.1] - 2026-03-25

### Added
- `analysis/sync_witness.py` — correlation, Fiedler, topological witnesses.
- `mitigation/symmetry_verification.py` — Z₂ parity post-selection.
- `analysis/quantum_persistent_homology.py` — hardware counts → $p_{h1}$.
- `phase/cpdr_simulation.py` — CPDR for XY simulation.
- `phase/coupling_topology_ansatz.py` — K_nm-informed VQE ansatz.
- `analysis/sync_entanglement_witness.py` — R as entanglement witness.
- `analysis/entanglement_sync.py`, `phase/cross_domain_transfer.py`, `analysis/otoc_sync_probe.py`, `analysis/hamiltonian_self_consistency.py`, `analysis/quantum_speed_limit.py`, `analysis/qfi_criticality.py`, `analysis/entanglement_percolation.py`, `analysis/qrc_phase_detector.py`, `phase/floquet_kuramoto.py`, `analysis/critical_concordance.py`, `analysis/berry_fidelity.py`, `analysis/quantum_mpemba.py`, `analysis/lindblad_ness.py`, `analysis/adiabatic_gap.py`, `analysis/pairing_correlator.py`, `analysis/xxz_phase_diagram.py`, `analysis/spectral_form_factor.py`, `analysis/loschmidt_echo.py`, `analysis/entanglement_entropy.py`, `analysis/krylov_complexity.py`, `analysis/magic_sre.py`, `analysis/finite_size_scaling.py`.
- `bridge/knm_hamiltonian.py` XXZ Hamiltonian (S² embedding).
- `tests/test_round4_8_coverage.py` — 36 coverage-gap tests.
- IBM Quantum campaign: 9 jobs submitted to `ibm_fez` (2 completed, 7 queued).
- March 20–22 marathon: 60 commits, 56 new modules.
  - `analysis/` expanded to 14 modules: `bkt_analysis`, `bkt_universals`, `p_h1_derivation`, `phase_diagram`, `dynamical_lie_algebra`, `qfi`, `quantum_phi`, `entanglement_spectrum`, `koopman`, `otoc`, `shadow_tomography`, `hamiltonian_learning`, `enaqt`, `vortex_binding`, `h1_persistence`.
  - `gauge/` (new): `wilson_loop`, `vortex_detector`, `cft_analysis`, `universality`, `confinement`.
  - `ssgf/` (new): `quantum_gradient`, `quantum_costs`, `quantum_outer_cycle`, `quantum_spectral`.
  - `applications/` expanded to 10 modules: `fmo_benchmark`, `power_grid`, `josephson_array`, `eeg_benchmark`, `iter_benchmark`, `cross_domain`, `quantum_kernel`, `quantum_reservoir`, `disruption_classifier`, `quantum_evs`.
  - `phase/` new algorithms: `adapt_vqe`, `avqds`, `qsvt_evolution`, `varqite`.
  - `tcbo/quantum_observer`, `pgbo/quantum_bridge`, `l16/quantum_director`, `bridge/snn_backward`, `bridge/ssgf_w_adapter`, `bridge/orchestrator_feedback`, `identity/robustness`, `qec/error_budget`, `benchmarks/mps_baseline`, `benchmarks/gpu_baseline`, `benchmarks/appqsim_protocol`, `hardware/gpu_accel`, `hardware/circuit_cutting`, `hardware/qasm_export`, `hardware/qcvv`.
- `GAP_CLOSURE_STATUS.md`.
- `identity/` subpackage: `IdentityAttractor`, `coherence_budget`, `chsh_from_statevector`, `disposition_entanglement_map`, `identity_fingerprint`, `verify_identity`, `prove_identity`.
- v1.0 modules: `mitigation/pec.py`, `hardware/trapped_ion.py`, `control/q_disruption_iter.py`, `benchmarks/quantum_advantage.py`, `bridge/snn_adapter.py`, `bridge/ssgf_adapter.py`, `identity/binding_spec.py`, `qsnn/training.py`, `qec/fault_tolerant.py`.
- 7 new examples (11–17); 2 notebooks (`06_pec_error_cancellation`, `07_quantum_advantage_scaling`).
- 2 API docs (`bridges_api.md`, `benchmarks_api.md`); `docs/equations.md`.
- Enterprise hardening: SPDX headers on 130 files, AGPL-3.0 dual-license, 4 new CI workflows, Dockerfile, Makefile, GOVERNANCE, SUPPORT, CONTRIBUTORS.

### Fixed
- CHSH angles: `b' = -π/4` → `b' = 3π/4` (for S ≈ 2√2).
- README license badge: MIT → AGPL-3.0-or-later.
- README test count: ~505 → 627+.
- README architecture tree updated.
- mypy numpy `no-any-return` errors across new modules.
- Ruff E741 / format in examples and notebooks.
- ArcaneNeuron import path: `neurons` → `neurons.models`.

### Changed
- Python 3.13 added to CI matrix and classifiers.
- `[docs]` extras: added `mkdocs`, `pymdown-extensions`.
- Tests: ~505 → 627+; coverage 100 %.

## [0.9.0] - 2026-03-02

### Added
- 100 % line coverage: 13 new tests closing 19 uncovered lines across 9 files.
- Security CI job (bandit + pip-audit).
- `.github/CODEOWNERS` (default: @anulum).
- Input validation: `QuantumPetriNet`, `QLiF`, `QAOA_MPC`.
- `WEIGHT_SPARSITY_EPS`; Shor & Preskill citation on `QBER_SECURITY_THRESHOLD`.
- ZNE: `R_std_per_scale` per noise scale.
- `_run_vqe()` returns `energy_std`.
- Runner `timeout_s` on `run_sampler` / `run_estimator`; metadata logged to `jobs.json`.
- `retrieve_job(job_id)` on `HardwareRunner`.
- `examples/07_crypto_bell_test.py`, `examples/08_dynamical_decoupling.py`.
- `notebooks/05_crypto_and_entanglement.ipynb`.
- Shared fixtures in `conftest.py`: `knm_4q`, `knm_8q`, `rng`.
- Dataclass field docs on `LockSignatureArtifact`, `LayerStateArtifact`.

### Fixed
- `q_disruption.py` magic `16` → `2**self.n_data_qubits`.
- Bare `1e-15` in `qpetri.py`, `spn_to_qcircuit.py`, `q_disruption.py` → `WEIGHT_SPARSITY_EPS`.
- Weak assertions tightened.

### Changed
- Classifier: Beta → Production/Stable.
- Coverage `exclude_also`: `TYPE_CHECKING` blocks.
- Tests: 483 → ~505.

## [0.8.0] - 2026-03-02

### Added
- Shot-noise error bars (`hw_R_std`, `hw_expectations_std`) on all 20 hardware experiments.
- `test_knm_properties.py`, `test_crypto_properties.py`, `test_qec_properties.py` (9 Hypothesis tests).
- `test_orchestrator_adapter_helpers.py`, `test_vqls_edge_cases.py`, `test_percolation_edge_cases.py`, `test_topology_auth_edge.py` (12 tests).
- `_constants.py`: `COUPLING_SPARSITY_EPS`, `CONCURRENCE_EPS`, `QBER_SECURITY_THRESHOLD`, `VQLS_DENOMINATOR_EPS`.
- `.editorconfig`.
- Input validation: K/omega shapes, ZNE data points, DD qubit range, ODE solver status, eigenvalue reality.
- Hardware exports: `bell_test_4q_experiment`, `correlator_4q_experiment`, `qkd_qber_4q_experiment`.

### Fixed
- 21 mypy `no-any-return` errors across 13 files.
- Dead code in `qaoa_mpc.py`.
- `test_public_api.py`: type-check assertions.
- `test_control_qec.py`: threshold `> 0` → `>= 10`.

### Changed
- Tests: 463 → 483.

## [0.7.1] - 2026-03-02

### Fixed
- Trivial `assert True` replaced with meaningful assertions.
- 3 duplicate tests removed.
- `inhibitor_anti_control` return annotation `-> None`.
- mypy: per-module `ignore_missing_imports`.
- `__all__` gaps: `knm_to_ansatz`, `classical_exact_evolution`, `JobResult`, `SurfaceCode`, `MWPMDecoder`.

### Added
- `wheel-check` CI job.
- 10 error-path tests.
- mypy `warn_unreachable`, `check_untyped_defs`.
- sdist excludes (`.github/`, `dist/`, `results/`, `figures/`, `notebooks/`).
- `build` in dev extras; matplotlib capped `<4.0`.
- Docstrings on `hmac_sign`, `hmac_verify_key`, `PhaseOrchestratorAdapter.from_orchestrator_state`.

## [0.7.0] - 2026-03-02

### Fixed
- `crypto/__init__.py` export fix (modules → callables).
- `QuantumDenseLayer` accepts `seed`.
- `PhaseVQE.relative_error_pct` returns `nan` for exact ≈ 0 instead of `inf`.
- Bare magic numbers in `percolation.py` → named constants.

### Added
- PEP 561 `py.typed`.
- pip `cache: 'pip'` in CI `setup-python`.
- Dependency upper bounds (next-major caps).
- `test_crypto_exports.py`.
- Seed determinism test for `QuantumDenseLayer`.

## [0.6.4] - 2026-03-01

### Fixed
- Stale test counts in docs (411 / 424 → 442).
- `figures/generate_header.py` version string.
- README architecture tree (`ansatz_bench.py`, `trotter_error.py`, `control_plasma_knm.py`).
- `SECURITY.md` supported versions (0.1.x → 0.6.x).

### Added
- `scpn-phase-orchestrator` in README Related Repositories.
- pip in Dependabot.
- v0.6.1 + v0.6.2 entries in `docs/changelog.md`.

### Changed
- Version 0.6.2 → 0.6.4 across `pyproject.toml`, `__init__.py`, `CITATION.cff`, `.zenodo.json`, badges, test.

## [0.6.3] - 2026-03-01

### Fixed
- Coverage gate in CI (codecov threshold).
- Mitigation API docs (ZNE / DD reference).
- Notebook table alignment.
- Ruff S311 in tests.

### Added
- `docs/mitigation_api.md`.
- Coverage gate job.
- Notebook summary table in `docs/index.md`.

## [0.6.2] - 2026-03-01

### Fixed
- Notebooks 01 / 03 / 04: `classical_kuramoto_ode` → `classical_kuramoto_reference`.
- Notebook 03: ZNE scales [1,2,3,4,5] → [1,3,5,7,9] (odd required).
- Notebook 04: 16-qubit → 8-qubit Trotter + 16-layer classical.
- Duplicate `docs/SESSION_LOG`, `docs/HANDOVER` removed.

### Added
- `figures/generate_knm_heatmap.py` + `figures/knm_heatmap.png` (16×16 K_nm).
- Knm heatmap in README.
- All 4 notebooks executed with embedded outputs.
- ROADMAP: post-2030 qualifier on fault-tolerant bullet.

## [0.6.1] - 2026-03-01

### Fixed
- mypy: `FloatArray` removed (incompatible with Python 3.9); `Path(None)` in `control_plasma_knm.py`.
- Zenodo metadata enriched (`.zenodo.json`, `CITATION.cff`).

## [0.6.0] - 2026-03-01

### Fixed
- Division-by-zero in `QuantumLIFNeuron`, `QuantumSynapse`, `QuantumSTDP`, `QAOA_MPC`, `classical_kuramoto_reference`.
- Index-out-of-bounds in `bell_inequality_test`, `best_entanglement_path`.
- Notebook 02 `PhaseVQE.solve()` dict keys.
- Stale test counts in `VALIDATION.md`, `docs/index.md`.

### Added
- Input-validation guards on QSNN constructors (`qlif`, `qsynapse`, `qstdp`).
- Input validation on `QAOA_MPC`, `bell_inequality_test`, `best_entanglement_path`, `classical_kuramoto_reference`.
- `PhaseVQE.solve()` returns `exact_energy`, `energy_gap`, `relative_error_pct`, `n_params`.
- 13 validation tests (`test_qsnn_validation.py`).

### Changed
- `docs/changelog.md` expanded with full version history.
- Tests: 411 → 424.

## [0.5.1] - 2026-03-01

### Fixed
- v0.5.0 published with `__version__ = "0.4.0"` (tag-timing bug). This release realigns `__version__`, `pyproject.toml`, and PyPI to 0.5.1.

## [0.5.0] - 2026-03-01

### Added
- 3 crypto hardware experiments: `bell_test_4q` (CHSH), `correlator_4q` (ZZ topology), `qkd_qber_4q` (QBER vs BB84 threshold).
- `_correlator_from_counts()` (2-qubit E(A,B)).
- `noise_analysis.py` `devetak_winter_rate()` (Devetak-Winter bound).
- 3 simulator tests (bell, correlator, QKD).

### Changed
- Experiments: 17 → 20 (3 crypto added to `ALL_EXPERIMENTS`).
- Tests: 408 → 411.
- Version 0.4.0 → 0.5.0.

## [0.4.0] - 2026-02-28

### Added
- GitHub Pages (MkDocs Material, 7 pages, auto-deploy).
- 4 Jupyter notebooks: Kuramoto XY, VQE ground state, ZNE, UPDE-16.
- 10 hardware experiments: noise baseline, 8-osc ZNE, 8q VQE hardware, UPDE-16 with DD, Trotter order-2, sync threshold, decoherence scaling, ZNE higher-order, VQE landscape, cross-layer correlation.
- 14 Hypothesis tests: probability-angle roundtrip, Knm symmetry/positivity, Hamiltonian Hermiticity, ansatz parameter counts.
- 8 edge-case tests: 2-oscillator minimal, `SuzukiTrotter(order=2)`, single-value inputs.
- 13 coverage-gap tests: multi-inhibitor anti-control, QAOA ZZ, VQLS near-zero guard, QEC odd defects / correction failure, QSTDP synapse.
- 4 integration tests: Knm → VQE ground state, Knm → Trotter → energy, 8q spectrum, 16-layer Hamiltonian structure.
- 7 regression tests: Knm calibration anchors, cross-hierarchy boosts, omega values, 4q ground energy baseline, statevector R, R evolution monotonicity, `_R_from_xyz`.

### Changed
- mypy: 27 → 30 source files (full `hardware/`).
- Tests: 208 → 254.
- `pyproject.toml`: docs URL; `mkdocs-material` in extras.

## [0.3.0] - 2026-02-28

### Added
- README rewrite (SCPN → XY isomorphism motivation, Kuramoto-to-Hamiltonian derivation, 4 figures, expanded example table, Related Repositories).
- Paper 27 citation in README and `docs/equations.md`.
- `docs/equations.md`: SCPN overview and UPDE definition.
- `examples/README.md` walkthrough.
- `HARDWARE_RESULTS.md`: L1–L16 naming section.

### Changed
- CI mypy scope: `bridge/` → all 8 module paths (27 files, zero errors).
- `VALIDATION.md`: tests 88 → 199; classical references now in-repo.
- Badges: tests 88 → 199; version 0.3.0 added.
- `CITATION.cff`: 0.1.0 → 0.3.0.

## [0.2.7] - 2026-02-28

### Added
- Parametrised quantum-vs-classical validation at n = {2,3,4,6}.
- Exact-diag cross-check against `eigvalsh` of Hamiltonian matrix.

### Changed
- Coverage omit narrowed: only `runner.py` and `experiments.py` excluded; `classical.py` now tracked.

## [0.2.6] - 2026-02-28

### Added
- Classical reference tests (20 across `classical_kuramoto_reference`, `classical_exact_diag`, `classical_exact_evolution`, `classical_brute_mpc`, `bloch_vectors_from_json`).
- Pauli ordering validation tests (2).
- `ALL_EXPERIMENTS` completeness test.
- Integration tests: quantum-vs-classical Kuramoto, ZNE on noiseless backend, energy conservation under Trotter, Trotter order-2 passthrough.

### Changed
- mypy scope expanded to `control/`, `qsnn/`, `qec/` (27 files, zero errors).
- Type narrowing in `vqls_gs.py`, `qaoa_mpc.py`, `runner.py` (assert-after-guard).
- `QuantumSTDP` forward ref to `QuantumSynapse` via `TYPE_CHECKING`.
- `ZNEResult` forward ref in `runner.py` via `TYPE_CHECKING`.

## [0.2.5] - 2026-02-28

### Added
- `trotter_order=2` on `QuantumKuramotoSolver` and `QuantumUPDESolver` (SuzukiTrotter order 2, O(t³/reps²) vs O(t²/reps)).
- `QuantumKuramotoSolver.energy_expectation(sv)`.
- `test_second_order_trotter`, `test_trotter_error_decreases_with_reps`, `test_energy_expectation`.

## [0.2.4] - 2026-02-28

### Fixed
- QAOA Hamiltonian: correct Ising encoding with identity (constant) term (`h_z`, `c0`); removed spurious ZZ terms.
- Quantum Petri net: multi-input transitions use multi-controlled Ry (AND gating).
- Inhibitor arcs: X-CRy-X anti-control pattern correctly gates on inhibitor emptiness.
- `build_knm_paper27`: dead `zeta_uniform` removed.
- VQLS: `imag_tol` init parameter (default 0.1).

### Added
- `test_hamiltonian_matches_classical_cost`.
- `test_optimal_bitstring_matches_brute_force`.
- `test_multi_input_conjunctive_gating`.
- `test_inhibitor_blocks_when_place_occupied`.

## [0.2.3] - 2026-02-28

### Fixed
- Disruption classifier `n_params`: CX gates have no trainable parameters; corrected to `n_layers*2*n_qubits` (30 vs 42 for default).
- QSTDP Hebbian: `post_measured` now implements LTP/LTD per Hebbian rule.

### Added
- Test for `kuramoto_4osc_zne_experiment`.
- Test for `upde_16_snapshot_experiment` (`@pytest.mark.slow`).
- `pytest.ini_options.addopts` skips slow/hardware by default.

## [0.2.2] - 2026-02-28

### Fixed
- MWPM decoder (3 bugs): dual edges for plaquette (Z) syndromes; seam-crossing winding-number logical error; d = 5 outperforms d = 3 below threshold (Dennis et al. 2002).
- Classical reference endianness (`_build_initial_state`, `_expectation_pauli`): `kron` order reversed to Qiskit little-endian; verified vs `Statevector` evolution to 1e-6.
- Parameter-shift rule (`q_disruption.py`): misleading `sin(shift)` denominator removed (Schuld et al., PRA 99, 032331).
- VQLS: assert imaginary norm < 0.1 before `np.real()`.
- `QAOA_MPC`: dead `current_state` parameter removed.
- ZNE: `base.inverse()` cached before fold loop.
- `trotter_upde`: dead `evolve(0)` removed; `reset()` added.
- `qlif`: `np.random.binomial` → seedable `rng`.
- `runner`: `TranspilerError` caught in DD pass fallback instead of bare `except`.
- `sc_to_quantum.measurement_to_bitstream` accepts `rng`.

### Added
- `_run_vqe` helper (eliminates vqe_4q / vqe_8q duplication).
- Root `__init__.py` exports 20 public symbols.
- Return-type annotations on public methods.
- 9 new tests: d5 > d3, shifted logical cycles, single X/Z correctness, VQE experiment, DD transpile, QSNN stochastic, bitstream seeded.
- Citation markers on `K_base`, `K_alpha` (Paper 27 Eq. 3).
- `test_classical_evolution_matches_qiskit` (Qiskit-vs-classical endianness).

## [0.2.1] - 2026-02-28

### Fixed
- `q_disruption.py`: configurable `seed`.
- `qpetri.py`: threshold gating via CRy.
- `classical.py`: sparse eigensolver (`eigsh`) for N ≥ 14 or when `k_eigenvalues` specified.
- `ansatz_bench.py`: deprecated `TwoLocal` / `EfficientSU2` → `n_local` / `efficient_su2` (Qiskit 2.1+).
- `vqls_gs.py`: `TwoLocal` → `n_local`.

### Added
- STDP direction validation (gradient sign at θ = 0, π/2).
- QEC threshold measurement tests (p = 0.01 vs p = 0.08; d = 5 single-error decoding).
- Disruption classifier seed reproducibility.
- Petri net controlled output + multi-step bounds.
- Bloch-ball constraint test.
- Sparse vs dense eigensolver agreement.

## [0.2.0] - 2026-02-28

### Added
- `hardware/noise_model.py`: `heron_r2_noise_model()` (T1 = 300 µs, T2 = 200 µs, CZ 0.5 %, readout 0.2 %).
- `mitigation/zne.py`: `gate_fold_circuit()` (Giurgica-Tiron et al. 2020), `zne_extrapolate()` (Richardson).
- `mitigation/dd.py`: `DDSequence` (XY4, X2), `insert_dd_sequence()` (Viola et al. 1999).
- `phase/trotter_error.py`: `trotter_error_norm()`, `trotter_error_sweep()`.
- `phase/ansatz_bench.py`: `benchmark_ansatz()`, `run_ansatz_benchmark()` (Knm-informed vs TwoLocal vs EfficientSU2).
- `hardware/classical.py`: `bloch_vectors_from_json()`.
- `hardware/experiments.py`: `kuramoto_4osc_zne_experiment`.
- `hardware/runner.py`: `run_estimator_zne()`, `transpile_with_dd()`, `noise_model` constructor parameter.
- `scripts/plot_vqe_convergence.py`, `scripts/plot_decoherence_curve.py`.
- `examples/05_vqe_ansatz_comparison.py`, `examples/06_zne_demo.py`.
- Top-level re-exports: `OMEGA_N_16`, `build_knm_paper27`, `knm_to_hamiltonian`, `QuantumKuramotoSolver`, `QuantumUPDESolver`, `PhaseVQE`, `HardwareRunner`, `JobResult`.

### Fixed
- `_run_sampler_simulator` uses `self._backend` (respects noise model).
- Duplicate `AerSimulator` import removed.
- Dead `2**n_osc` in `classical.py` (lines 117, 147).
- `DEFAULT_INSTANCE` reads `SCPN_IBM_INSTANCE` with fallback.

### Changed
- mypy scope: `bridge/` → `bridge/`, `phase/`, `mitigation/`, `hardware/classical.py`, `hardware/runner.py`.

## [0.1.0] - 2026-02-28

### Added
- `qsnn/`: `qlif.py`, `qsynapse.py`, `qstdp.py`, `qlayer.py`.
- `phase/`: `xy_kuramoto.py`, `trotter_upde.py`, `phase_vqe.py`.
- `control/`: `qaoa_mpc.py`, `vqls_gs.py`, `qpetri.py`, `q_disruption.py`.
- `bridge/`: `knm_hamiltonian.py`, `spn_to_qcircuit.py`, `sc_to_quantum.py`.
- `qec/`: `control_qec.py` (toric surface code + MWPM with Knm-weighted edges).
- `hardware/`: `runner.py`, `experiments.py`, `classical.py` (`ibm_fez`).
- 88 unit tests, 4 example scripts, 19 hardware result files.
- Hardware validation on `ibm_fez`: VQE 0.05 % error, 12-point decoherence curve, 16-layer UPDE snapshot.
- CI workflow with Python 3.9–3.12 matrix, coverage, ruff lint.
- Documentation: architecture, API reference, hardware results.
