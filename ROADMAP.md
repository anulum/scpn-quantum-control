# Roadmap

## Canonical work queue

This file is the single source of truth for active roadmap and TODO
selection. Older planning files under `.coordination/` and
`docs/internal/` are retained as historical context only unless an item
is copied here.

### Current top priority: repository hygiene and release safety

- [x] **GitHub Actions history audit.** Completed 2026-05-06:
  current failed and cancelled workflow-run queries are empty after
  resolved superseded runs were classified and removed.
- [x] **Latest CI/link-check failures.** Completed 2026-05-06:
  latest `main` CI run `25412632454` passed lint, tests, security,
  Rust audit, optional integration, hardware smoke, and CI gate.
- [x] **Security alert audit.** Completed 2026-05-06: open CodeQL,
  Dependabot security, and secret-scanning alert queries returned empty
  lists.
- [x] **PR and branch hygiene.** Completed 2026-05-06 for active
  repository state: no open pull requests were present during the
  hygiene pass.
- [x] **Safe workflow-run cleanup.** Completed 2026-05-06: deleted only
  resolved or superseded failed/cancelled workflow runs after their
  causes were fixed by later successful commits.
- [x] **Session log, handover, and Arcane notification.** Completed
  2026-05-06 with append-only coordination records and a new SNN
  stimulus.

### Next repository hygiene follow-up

- [x] **Full historical Actions audit automation.** Implemented
  2026-05-06: `tools/audit_github_actions_history.py` classifies
  `gh run list --json ...` history into clean successes, in-progress
  runs, unresolved failures, resolved failures, unresolved cancellations,
  superseded cancellations, and safe delete candidates without deleting
  any run.
- [x] **Actions audit GitHub workflow integration.** Implemented
  2026-05-06: `.github/workflows/actions-history-audit.yml` runs the
  classifier on a schedule or manually, uploads read-only audit
  artefacts, and performs no deletions.
- [x] **Link-check history automation.** Implemented 2026-05-06:
  `tools/audit_link_check_history.py` records live link-check failures,
  resolved historical failures, accepted external-transient failures,
  and safe delete candidates from workflow history.
- [x] **Actions-history dashboard documentation.** Implemented
  2026-05-06: `docs/actions_history_dashboard.md` documents audit
  artefacts, bucket meanings, accepted external-transient Link Check
  failures, and the manual safe-delete rule.
- [x] **Post-push Actions history workflow observation.** Completed
  2026-05-06: manually triggered `Actions History Audit` run
  `25436780329`; the `classify-history` job passed and uploaded the
  `actions-history-audit` artefact.

### Archived repository hygiene checklist

- [x] **GitHub Actions history audit.** Classify the full workflow-run
  history into resolved failures, unresolved failures, and cancelled
  superseded runs. Do not delete any run until the corresponding failure
  is demonstrably resolved by a later successful run or superseded by a
  closed branch/PR.
- [x] **Latest CI/link-check failures.** Inspect the latest failing
  `Link Check` runs on `main`, fix any live issue, and record whether
  older link failures are resolved by the fix.
- [x] **Security alert audit.** Check open CodeQL, Dependabot security,
  and secret-scanning alerts. Fix true positives; document API-permission
  limits or accepted false positives internally.
- [x] **PR and branch hygiene.** Review open Dependabot PRs and stale
  branches, decide whether to merge, rebase, close, or leave blocked,
  then document the decision.
- [x] **Safe workflow-run cleanup.** After classification, delete only
  cancelled or fully resolved failed runs where deletion does not remove
  the only evidence for an unresolved defect.
- [x] **Session log, handover, and Arcane notification.** Keep a new
  timestamped log and handover for the repository-hygiene audit, and
  emit a factual Arcane stimulus.

### Active release tasks

- [ ] **Coverage and test-quality closure.** Push the release baseline
  from the documented `~97.6 %` coverage state toward 100 %, then audit
  behavioural value rather than relying on line coverage alone.
- [x] **Behavioural-test audit automation.** Implemented 2026-05-06:
  `tools/audit_test_behaviour.py` inventories test modules for
  assertion-bearing tests, exception contracts, parametrisation, and
  smoke-only tests so the manual behavioural audit can be prioritised.
- [x] **Behavioural assertion-helper recognition.** Implemented
  2026-05-06: the behavioural audit now counts assertion helper calls
  such as `assert_*` functions instead of misclassifying them as smoke
  tests.
- [x] **Class-based test audit coverage.** Implemented 2026-05-06:
  `tools/audit_test_behaviour.py` now counts `test_` methods inside
  `Test*` classes, so class-based pytest modules are no longer
  misreported as empty.
- [ ] **Behavioural-test audit.** Review every test module for
  assertions that constrain numerical invariants, state transitions,
  exception contracts, provenance, or integration behaviour.
- [x] **Behavioural-test audit: topological coupling guard.**
  Implemented 2026-05-06: `tests/test_topological_coupling_guard.py`
  now asserts coupling symmetrisation, diagonal clearing, omega/config
  retention, and no state mutation on the missing optional dependency
  path.
- [x] **Behavioural-test audit: readout matrix guards.** Implemented
  2026-05-06: `tests/test_readout_matrix.py` now asserts confusion
  matrix column stochasticity, condition number and shot accounting,
  bitstring parsing, invalid-count rejection, mitigation normalisation,
  and mean-magnetisation observables.
- [x] **Public API smoke-only audit closure.** Implemented 2026-05-06:
  `tests/test_public_api.py` now asserts explicit export-count
  contracts in addition to delegated import/type checks, closing the
  remaining smoke-only module found by `tools/audit_test_behaviour.py`.
- [x] **Backend registry no-op contract.** Implemented 2026-05-06:
  `test_unregister_missing_is_silent` now asserts registry state is
  unchanged when unregistering a missing backend.
- [x] **Coverage 0.95 import contracts.** Implemented 2026-05-06:
  `tests/test_coverage_095_push.py` import checks now assert module
  identity and public surface instead of only importing modules.
- [x] **Phase-artifact fuzz boundary contracts.** Implemented
  2026-05-06: inclusive PLV and `R` boundary tests now assert retained
  endpoint values instead of only constructing artefacts.
- [x] **VQLS denominator guard contract.** Implemented 2026-05-06:
  `test_denominator_guard_path` now asserts the patched statevector
  path is exercised while forcing the near-zero denominator branch.
- [x] **STDP pipeline update contract.** Implemented 2026-05-06:
  pipeline wiring now asserts Hebbian pre/post firing increases the
  synaptic weight while preserving configured clamp bounds.
- [x] **Rust benchmark timing contracts.** Implemented 2026-05-06:
  pure performance rows in `tests/test_rust_path_benchmarks.py` now
  assert finite non-negative timing samples instead of only printing
  benchmark lines.
- [x] **E2E boundary audit automation.** Implemented 2026-05-06:
  `tools/audit_e2e_contract_boundaries.py` inventories hardware/QPU,
  bridge, SC-NeuroCore, Phase Orchestrator, notebook, and example
  workflow test boundaries without fabricating coverage for missing
  categories.
- [x] **Example workflow contract.** Implemented 2026-05-06:
  `tests/test_example_workflows.py` statically verifies every example
  script parses, exposes `main()`, has an execution guard, and is listed
  in `examples/README.md`; the README now includes examples 19--21.
- [x] **Notebook workflow contract.** Implemented 2026-05-06:
  `tests/test_notebook_workflows.py` statically verifies committed
  notebooks are valid nbformat-4 JSON artefacts with cells, metadata,
  recognised cell types, and notebook-compatible source fields.
- [x] **E2E and contract audit.** Completed 2026-05-06:
  `tools/audit_e2e_contract_boundaries.py --fail-on-missing` now
  reports all six tracked boundaries covered; static notebook/example
  contracts are documented in `docs/e2e_contract_boundaries.md` without
  claiming executed scientific validation.
- [x] **Mutation-test expansion: XY Kuramoto invariants.** Implemented
  2026-05-06: `tests/test_xy_kuramoto.py` now includes mutation guards
  for zero-time identity evolution, internal Pauli qubit ordering, and
  trajectory time-grid endpoints.
- [x] **Mutation-test expansion: async runner state contracts.**
  Implemented 2026-05-06: `tests/test_async_runner.py` now guards
  round-robin index progression, submitted timestamp capture, underlying
  job-handle preservation, and timeout propagation into result polling.
- [x] **Mutation-test expansion: backend registry state contracts.**
  Implemented 2026-05-06: `tests/test_backend_registry.py` now guards
  discovery-state reset on `clear()`, sorted known-backend diagnostics
  for missing names, and one-shot discovery semantics after broken
  plugin load attempts.
- [x] **Mutation-test expansion: Knm Hamiltonian bridge invariants.**
  Implemented 2026-05-06: `tests/test_knm_hamiltonian.py` now guards
  Kuramoto ring self-coupling/edge count, inclusive ansatz thresholding,
  and delta-zero equivalence across XY, XXZ, sparse, and dense
  Hamiltonian paths.
- [x] **Mutation-test expansion: DLA parity invariants.** Implemented
  2026-05-06: `tests/test_dla_parity_theorem.py` now guards parity
  operator popcount ordering, even/odd projector orthogonality and
  reconstruction, and unnormalised weight accounting.
- [x] **Mutation-test expansion: ZNE mitigation invariants.**
  Implemented 2026-05-06: `tests/test_zne.py` now guards terminal
  measurement stripping/re-append behaviour, exact odd-scale fold gate
  counts, and immutable result copies for extrapolation inputs.
- [x] **Mutation-test expansion: PEC mitigation invariants.**
  Implemented 2026-05-06: `tests/test_pec.py` now guards exact
  inverse-channel coefficient formulae, sign-distribution support, and
  circuit-size exponentiation of PEC overhead.
- [x] **Mutation-test expansion: readout-matrix mitigation invariants.**
  Implemented 2026-05-06: `tests/test_readout_matrix.py` now guards
  full-basis matrix stochasticity, prepared-state shot accounting,
  invalid observed labels, and probability-observable contracts.
- [x] **Mutation-test expansion: symmetry-decay mitigation invariants.**
  Implemented 2026-05-06: `tests/test_symmetry_decay.py` now guards
  copied decay-model sequences and real positive GUESS correction
  factors under sign-flipped symmetry readings.
- [x] **Mutation-test expansion: compound-mitigation pipeline guards.**
  Implemented 2026-05-06: `tests/test_compound_mitigation.py` now
  asserts the CPDR plus symmetry pipeline calls the backend exactly once
  with the generated training circuits and preserves requested
  training-count boundaries.
- [x] **Mutation-test expansion: CPDR mitigation guards.**
  Implemented 2026-05-06: `tests/test_cpdr.py` now guards Clifford
  snapping without target mutation, little-endian observable-qubit
  extraction, zero-slope fallback, and backend training-count
  boundaries.
- [x] **Mutation-test expansion: DD mitigation guards.**
  Implemented 2026-05-06: `tests/test_dd.py` now guards negative idle
  qubit rejection, exact DD sequence ordering, idle-qubit placement, and
  source-circuit immutability.
- [x] **Mutation-test expansion: symmetry-verification guards.**
  Implemented 2026-05-06: `tests/test_symmetry_verification.py` now
  guards invalid Z2 parity labels and invalid computational-basis count
  labels before post-selection or symmetry expansion.
- [x] **Mutation-test expansion: Mitiq integration guards.**
  Implemented 2026-05-06: `tests/test_mitiq_integration.py` now guards
  that default ZNE and DDD executor paths forward the requested shot
  count into the Qiskit executor instead of silently using its default.
- [x] **Mutation-test expansion.** Completed 2026-05-06 across the
  listed initial targets: `phase/xy_kuramoto.py`, hardware runner and
  backend registry state contracts, Hamiltonian bridge invariants,
  DLA-parity analysis contracts, and mitigation modules.
- [x] **Mock/stub audit automation.** Implemented 2026-05-06:
  `tools/audit_mock_stub_usage.py` inventories `Mock`, `MagicMock`,
  `patch`, `monkeypatch`, and fake/stub helper usage, flags apparent
  third-party boundaries, and highlights result-term contexts for manual
  review without deleting or rewriting tests.
- [x] **Mock/stub audit.** Completed 2026-05-06:
  an internal local audit record captures the
  `tools/audit_mock_stub_usage.py` inventory, manual classification of
  the 22 result-term contexts, and the decision that current mocks model
  branch control, optional dependencies, configuration, or executor
  boundaries rather than fabricated publication data or successful
  hardware execution. Ensure future mocks only model third-party
  boundaries and never fabricate scientific results, provenance,
  datasets, or successful hardware execution.
  Initial target list reviewed: `phase/xy_kuramoto.py`,
  `hardware/async_runner.py`, `hardware/backends.py`, `bridge/*`,
  `analysis/dla*`, and `mitigation/*`.
- [x] **Full-suite ordering audit automation.** Implemented
  2026-05-06: `tools/audit_test_ordering_state.py` inventories module
  reloads, monkeypatch state changes, environment mutation, module
  injection, random seed mutation, and backend/cache/registry global
  mutations so ordering risks can be reviewed without relying on ad-hoc
  greps.
- [x] **Full-suite ordering audit.** Completed 2026-05-06: a local
  internal audit record documents the
  `tools/audit_test_ordering_state.py` result (`420` findings), accepted
  isolation patterns for monkeypatch-managed state, optional dependency
  module injections, and paired reload tests, plus the high-density files
  to prioritise if ordering randomisation later exposes flakes.
- [x] **Static-analysis pass.** Completed 2026-05-06: Semgrep was added
  as a development dependency and run with explicit Python/security
  rules (`0` findings); Bandit was run over `src`, `scripts`, and
  `tools`, easy true positives were fixed, and the remaining `57` low
  findings were documented locally as accepted subprocess/provenance
  tooling or deterministic non-cryptographic shuffling.
- [x] **`QuantumKuramotoSolver` validation.** Completed 2026-05-06:
  validates oscillator count, coupling-matrix shape/squareness,
  symmetric XY semantics, finite coupling and `omega` values, matching
  finite `omega` length, and canonicalises the solver-owned coupling
  diagonal to zero without mutating caller input.
- [x] **Trotter/config public surface.** Completed 2026-05-06:
  `TrotterEvolutionConfig` now exposes typed defaults for evolution
  order, `evolve()` Trotter steps, and `run()` per-step Trotterisation;
  legacy `trotter_order`, `trotter_steps`, and `trotter_per_step`
  arguments remain supported and take explicit precedence.
- [x] **Expectation hot path.** Implemented 2026-05-06:
  statevector X/Y/Z expectation fallbacks now use bitwise-vectorised
  NumPy instead of per-qubit dense Kronecker or Qiskit Pauli object
  construction; Rust remains the preferred accelerated path when the
  extension is available.
- [x] **Rust import failure policy.** Implemented 2026-05-06:
  optional Rust acceleration imports now share
  `optional_rust_engine()`, treating only true
  `scpn_quantum_engine` absence as optional while propagating broken
  installed-extension import failures.
- [x] **README and public framing sync.** Completed 2026-05-06:
  README, publication framing guide, and hardware status ledger now
  reflect promoted Phase 2 DLA artefacts, SCPN/FIM negative hardware
  results, benchmark dashboard/CLI availability, and conservative
  no-advantage/no-DLA-only/no-FIM-protection boundaries.
- [x] **Architecture data-flow diagram.** Completed 2026-05-06:
  `docs/architecture.md` now documents the stable artefact-first
  pipeline from `K_nm/omega` through Hamiltonian compilation,
  circuit/simulator kernels, CPU/GPU/Rust/QPU execution, raw counts,
  mitigation, DLA/sync/FIM/VQE observables, and reports/ledgers.
- [x] **Curated researcher examples.** Completed 2026-05-06:
  `docs/application_benchmarks.md`, `examples/README.md`, and
  `data/public_application_benchmarks/README.md` now promote selected
  GraphML/CSV, EEG, power-grid, plasma/tokamak, and notebook workflows
  with provenance boundaries and deterministic no-QPU smoke commands.

### Active paper and submission tasks

- [x] **DLA parity preprint submission package.** Completed 2026-05-06:
  `docs/dla_parity_submission_checklist_2026-05-06.md` freezes the
  conservative parity-sector/excitation-number framing, required job IDs,
  committed artefact index, unsupported claims, and final no-QPU
  pre-upload gate.
- [x] **Rust/VQE methods paper package.** Completed 2026-05-06:
  `docs/rust_vqe_methods_submission_checklist_2026-05-06.md` freezes
  the artefact-first table boundary, supported/unsupported claims,
  generator-script index, `scpn-bench reproduce-methods` gate, and
  no-QPU final pre-upload checklist.
- [x] **JOSS-style software paper package.** Completed 2026-05-06:
  `docs/joss_software_submission_checklist_2026-05-06.md` freezes the
  software-paper claim boundary, JOSS/pyOpenSci framing, metadata gate,
  reproducibility gate, and no-QPU final pre-upload checklist.
- [x] **SCPN/FIM Hamiltonian paper package.** Completed 2026-05-06:
  `docs/scpn_fim_submission_checklist_2026-05-06.md` freezes the
  negative hardware falsification boundary, committed artefact index,
  IBM job IDs, `scpn-bench fim-all` gate, full-basis readout-mitigation
  scope, and blocked hardware-protection claims.
- [x] **Combined submission checklist.** Completed 2026-05-06:
  `docs/combined_submission_checklist_2026-05-06.md` ties the four
  paper packages together with final PDF build commands, no-QPU
  reproduction gates, arXiv metadata draft, URL/identifier checks, and a
  minimal venue-conditional AI disclosure policy.
- [x] **IBM Quantum Credits follow-up.** Completed 2026-05-06:
  `docs/ibm_quantum_credits_followup_2026-05-06.md` records the
  5--10 hour QPU allocation boundary, current evidence package,
  affiliation wording, draft locations, and per-run spend gates. The
  older allocation draft now matches the 5--10 hour scope.

### Active Phase 4 follow-up tasks

- [x] **One-command reproducibility CLI.** Implemented 2026-05-06:
  `scpn-bench reproduce-methods`, `scpn-bench fim-all`, and
  `scpn-bench all` regenerate committed benchmark artefacts and report
  drift without submitting IBM jobs.
- [x] **Public benchmark dashboard.** Implemented 2026-05-06:
  `docs/methods_benchmark_dashboard.md` is wired into MkDocs and links
  artefacts, generator scripts, provenance, reproducibility commands,
  optional GPU/scaling/readout harnesses, and no-QPU-spend boundaries.
- [x] **Ansatz scaling plus tensor-network baseline: initial harness.**
  Implemented 2026-05-06 with n=4--12 ansatz scaling rows and
  tensor-network truncation diagnostics.
- [x] **Ansatz scaling sparse strengthening.** Implemented 2026-05-06
  using sparse eigensolver references where feasible for larger-n rows.
- [x] **Richer ansatz/TN reference comparisons.** Implemented
  2026-05-06: `benchmark_ansatz_scaling_tn.py` now emits generated
  per-`n` reference-comparison rows pairing MPS truncation diagnostics
  with committed VQE aggregate references and marks missing larger-n VQE
  rows as skipped rather than extrapolated.
- [x] **Native or analogue FIM compiler path.** Initial
  `lambda_fim` compiler payload implemented 2026-05-06 by decomposing
  `-lambda M^2/n` into all-to-all `Z_i Z_j` terms for backend design
  studies.
- [x] **Provider-specific analogue backend export layer.** Implemented
  2026-05-06: generic analogue programmes can now be exported into
  Pulser, Bloqade, and IBM pulse-level design payloads with SDK
  availability metadata, platform-compatibility checks, and explicit
  `can_submit=False` boundaries.
- [x] **Executable provider analogue backend: approval-gated plan.**
  Implemented 2026-05-06: provider exports can now be wrapped in
  `ProviderAnalogExecutionPlan` via `prepare_provider_execution_plan`,
  requiring calibration metadata and explicit approval before SDK-object
  construction or emulator execution is marked possible. Cloud submission
  remains blocked until a separately approved provider runner exists.
- [x] **Adaptive lambda feedback scaffold.** Implemented 2026-05-06:
  `AdaptiveFIMConfig`, `FIMWitness`, `propose_next_lambda`, and
  `adaptive_lambda_schedule`.
- [x] **Adaptive-QPU protocol.** Implemented 2026-05-06:
  `docs/adaptive_fim_qpu_protocol_2026-05-06.md` defines the
  non-submitting adaptive `lambda_fim` hardware boundary, QPU budget
  gate, live transpilation gate, falsification rules, artefact names,
  and blocked claims before any IBM submission.
- [x] **FIM repeated full-basis readout mitigation.** Implemented
  2026-05-06 for the repeated dataset where the required 16-state
  calibration basis exists.
- [x] **Readout-mitigation eligibility markers.** Implemented
  2026-05-06: `audit_readout_mitigation_eligibility.py` generates
  `data/readout_mitigation_eligibility/readout_mitigation_eligibility_2026-05-06.json`
  with dataset-level and per-`n` markers for full-basis, partial
  exact-state, and missing-calibration readout-mitigation eligibility
  before any new QPU calibration spend.

### Hardware experiment candidates

These are candidates, not authorisations to spend QPU time. Each needs
offline artefacts, a preregistered manifest, depth/shot gates, and an
explicit QPU-time estimate before submission.

- [x] **Multi-device DLA replication preregistration.** Completed
  2026-05-06: `docs/dla_multidevice_replication_prereg_2026-05-06.md`
  defines the second-Heron backend rule, reduced `n=4` circuit matrix,
  148-circuit scope, 3--6 minute estimate, 10-minute ceiling,
  live-depth gates, analysis plan, falsification rules, and output
  artefact paths. QPU execution remains separate and approval-gated.
- [ ] **Multi-device DLA replication execution.** Execute the
  preregistered second-Heron run only after live backend selection,
  transpilation/depth gates, QPU budget confirmation, and explicit
  approval.
- [ ] **Systematic state/layout randomisation.** Separate symmetry
  sector, excitation count, physical layout, readout, and topology
  contributions.
- [ ] **Full readout-mitigation calibration where missing.** Only run
  complete basis calibration when the scientific value outweighs QPU
  cost.
- [ ] **GUESS / symmetry-decay calibration.** Use parity leakage as a
  real hardware witness for symmetry-guided extrapolation if stability
  is sufficient.
- [ ] **Layer-selective qubit assignment.** Assign strongly coupled
  layers to low-error physical qubits and compare with default layout.
- [ ] **Entanglement entropy or tomography check.** Use shadow
  tomography or small-n tomography only after cost and claim boundaries
  are fixed.
- [ ] **Depth-optimal native decomposition.** Reduce Kuramoto evolution
  depth by targeting Heron native gates directly.
- [ ] **Variational quantum simulation alternative.** Compare VQS
  against Trotter where it can reduce compiled depth.
- [ ] **Multi-circuit QEC demonstration.** Evaluate physics-aware
  decoding only with clear logical-error metrics.

### Visibility and registration tasks

- [x] **GitHub topics.** Completed 2026-04-17.
- [x] **QOSF awesome-quantum-software.** Merged 2026-03-30.
- [x] **CiteAs verification.** Verified 2026-04-17.
- [x] **PyPI publication.** Completed for the current package line.
- [x] **OpenSSF Best Practices.** Passing badge is present.
- [x] **Zenodo DOI.** Existing DOI is live.
- [x] **GitHub Pages docs.** Published.
- [ ] **Software Heritage SWHID follow-up.** Check the submitted save
  request and publish the SWHID once the crawl completes.
- [ ] **Zenodo communities and metadata refresh.** Add the DOI to
  quantum/physics communities and refresh stale version metadata during
  a manual Zenodo login session.
- [ ] **Qiskit Ecosystem Catalog.** Submit ecosystem metadata before
  attempting `awesome-qiskit`.
- [ ] **awesome-qiskit.** Blocked until Qiskit Ecosystem membership is
  accepted.
- [ ] **Conda-forge recipe.** Prepare and submit staged-recipes package.
- [ ] **Metriq submission.** Submit only validated, bounded benchmark
  results.
- [ ] **pyOpenSci review.** Submit the software package for review and
  possible JOSS fast-track.
- [ ] **JOSS submission.** Submit after paper package and metadata are
  aligned.
- [ ] **arXiv submission.** Submit the paper set once final PDFs,
  references, source tarballs, and artefact links are aligned.
- [ ] **SciPy 2026 CFP.** Prepare talk/poster proposal if strategically
  useful.
- [ ] **Community announcements.** Prepare Reddit, Qiskit Slack,
  Unitary Discord, Hacker News, LinkedIn, and X posts only after the
  public preprints are live.
- [ ] **Registry listings.** Quantiki, QOSF project list, best-of-python,
  Papers With Code, SciCrunch RRID, Open Hub, and Research Software
  Directory.

### Deferred / CEO-gated strategic tracks

These tracks remain scoped in `docs/strategic_roadmap.md` and must not
be executed until individually activated.

- [ ] **S1** Hybrid classical--quantum feedback loop.
- [ ] **S2** Quantum advantage benchmarks at scale.
- [ ] **S3** ML-augmented pulse / ansatz design.
- [ ] **S4** Multi-hardware backend + pulse-level control.
- [ ] **S5** Open-data + classical validation harness.
- [ ] **S6** Decoupled `quantum-kuramoto` subpackage.
- [ ] **S7** Fault-tolerant / logical-level extension roadmap.
- [ ] **S8--S53** Scientific, foundational, and applied post-v1.0
  differentiation tracks listed in the strategic roadmap.

## Recently closed

- **S5** Open-data + classical validation for the DLA-parity dataset
  — **closed 2026-04-18**. New subpackage `scpn_quantum_control.dla_parity`:
  schema, JSON loader + SHA-256 integrity check, statistical reproducer
  against the published summary, classical (numpy + qutip) leakage
  reference, one-call `run_full_harness`, CLI at
  `scripts/run_dla_parity_suite.py`, docs page, `[dla-parity]` extra,
  CI smoke step. 100 % line coverage across all modules; 128 / 128
  tests green.
- **B1** arXiv LaTeX preprint — **closed 2026-04-18** (commit
  `457b734`): `paper/phase1_dla_parity.tex` compiles to a 4-page
  PDF; `paper/README.md` documents the submission packaging.
- **Qiskit 1.x → 2.x migration** — **closed 2026-04-18** (commit
  `b786fc2`): pin bump to `qiskit>=2.2,<3.0`, closed the Dependabot
  PR #44 technical debt, misleading PauliEvolutionGate comment
  corrected. No source changes needed (our call sites were already
  2.x-compatible). Full test suite 5045 / 0 / 95.

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
- **IBM hardware evidence ledgered for ibm_fez (Heron r2)**
  - legacy artifact-backed Bell, QKD, VQE, ZNE, Trotter, and UPDE rows retained
  - no broad-advantage or frontier claim promoted from the baseline campaign
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
- **IBM hardware v2:** aggregate-only ibm_fez artifacts retained but unpromoted
  until raw counts, retrieval manifest, and reproduction analysis are reviewed
- **Rust engine expanded:** 15→22 functions (correlation_matrix_xy, lindblad_jump_ops_coo, lindblad_anti_hermitian_diag, parity_filter_mask)
- **Documentation audit:** 21 discrepancies fixed, 26 analysis + 10 phase + 3 bridge exports added
- Fixed backend_dispatch jax.numpy AttributeError
- 165 Python modules, 22 Rust functions, 47 notebooks, 21 examples
- 2813+ tests, 95% coverage

## v1.0.0 (Target: Q3 2026)

Remaining items:

- [x] IBM Heron r2 hardware evidence ledgered; promoted claims narrowed to
  raw-count-backed or artifact-named rows
- [ ] Coverage push to 100% (tracked in the internal coverage queue; latest documented baseline is ~97.6%, 315 uncovered lines)
- [ ] arXiv preprint: "Quantum simulation of coupled-oscillator synchronization on a 156-qubit superconducting processor"
- [ ] Quantum advantage figure: exact-simulation crossover and hardware-budget boundary; no broad hardware-advantage claim until a preregistered raw-count campaign passes the validation gate
- [ ] IBM Quantum Credits campaign (applied 2026-03-29, pending review)
- [ ] Version bump to 1.0.0

## Future

- **Phase 4 / follow-up validation and reproducibility**
  - **One-command reproducibility CLI — implemented 2026-05-06.**
    `scpn-bench reproduce-methods`, `scpn-bench fim-all`, and
    `scpn-bench all` now run committed benchmark harness groups, regenerate
    JSON/CSV artefacts, and report drift against committed files without
    submitting IBM jobs.
  - **Public benchmark dashboard — implemented 2026-05-06.**
    `docs/methods_benchmark_dashboard.md` is wired into MkDocs and links
    benchmark artefacts, generator scripts, machine provenance,
    reproducibility commands, optional GPU/scaling/readout harnesses, and
    no-QPU-spend boundaries.
  - **Ansatz scaling plus tensor-network baseline — initial harness
    implemented 2026-05-06; next strengthening task.**
    `scripts/benchmark_ansatz_scaling_tn.py` generates n=4--12 ansatz
    scaling rows and tensor-network truncation diagnostics. Sparse eigensolver
    references now strengthen larger-n rows where feasible. Remaining work:
    extend beyond ground-state truncation diagnostics into richer MPS/VQE
    reference comparisons, then update methods-paper claims only from
    regenerated artefacts.
  - **Native or analogue FIM implementation — initial compiler path
    implemented 2026-05-06.**
    The analogue Kuramoto compiler accepts `lambda_fim` and decomposes
    `-lambda M^2/n` into the documented all-to-all `Z_i Z_j` payload for
    backend design studies. Remaining work: wire a real provider-specific
    Pulser/Bloqade/pulse-level backend before making any execution claim.
  - **Adaptive lambda feedback loop — scaffold implemented 2026-05-06.**
    `AdaptiveFIMConfig`, `FIMWitness`, `propose_next_lambda`, and
    `adaptive_lambda_schedule` provide a deterministic controller over
    measured or simulated witnesses. Remaining work: design a separately
    approved hardware protocol before any adaptive-QPU claim.
  - **Scalable readout-mitigation cross-check — FIM repeated dataset
    implemented 2026-05-06.**
    Full 16-state readout-matrix inversion is implemented for the repeated
    SCPN/FIM follow-up, where the required calibration basis exists. Remaining
    work: add dataset-level eligibility markers for any n<=8 campaign that
    lacks complete basis calibration, before spending QPU time on new
    calibration circuits.
- Fault-tolerant UPDE on surface code logical qubits (post-2030, hardware-dependent)
- QSNN training loop on real hardware (parameter-shift STDP)
- Quantum disruption classifier on ITER disruption database
- Trapped-ion hardware runs (IonQ Aria / Quantinuum H2)
- SSGF quantum-in-the-loop with live SSGFEngine on GPU

## Strategic Differentiation (post-v1.0)

Fifty-three post-v1.0 differentiation tracks are scoped in
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
