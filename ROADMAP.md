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
- [ ] **Scientific gap queue.** Keep the public claim surface bounded by
  the internal scientific-gap ledger: K_nm measured-system validation
  still needs additional physical-unit candidates beyond the negative
  IEEE 5-bus control, broad quantum advantage still needs the
  provenance-rich classical/Rust/GPU matrix before any new QPU spend,
  and `p_h1 = 0.72` remains open until a K_nm-specific derivation or
  measured/TCBO reproduction includes uncertainty crossing 0.72.
- [x] **XY Kuramoto trajectory time-grid hardening.** Implemented
  2026-05-12: `QuantumKuramotoSolver.run()` now builds explicit time
  boundaries and evolves a final partial interval when `t_max` is not an
  integer multiple of `dt`, preventing state/label drift in non-divisible
  horizons. API and performance docs record the exact endpoint contract.
- [x] **TCBO coupling-weighted complex reconstruction.** Implemented
  2026-05-12: `tcbo_weighted_complex.py` reconstructs the roadmap
  blocker using `K_ij * |cos(theta_j - theta_i)|` edge weights, a
  thresholded flag complex, beta-1 over GF(2), and a threshold scan
  against the `0.72` target. The audit runner records this reconstruction
  separately from the legacy delay-embedded observer path; the claim
  remains unpromoted pending preregistered replay with uncertainty.
- [x] **QSVT resource-estimator input hardening.** Implemented
  2026-05-12: the QSVT resource estimator and query-count helpers now
  reject non-square, dimension-mismatched, asymmetric, or non-finite
  `K_nm/omega` inputs plus invalid simulation-time and error-budget
  parameters before any Hamiltonian construction or resource claim is
  produced. The phase API documents the validation contract.
- [x] **Integrated-information wrapper production route.** Implemented
  2026-05-12: `IntegratedInformationPhi` now routes explicit
  `coupling_matrix` and `natural_frequencies` inputs to the
  `compute_quantum_phi` Kuramoto-XY density-matrix engine with shape,
  symmetry, and finite-value validation. Counts-only entropy remains an
  opt-in `entropy_proxy` diagnostic and is never returned as `phi`.
- [x] **QFI production-adapter input hardening.** Implemented
  2026-05-12: `QuantumFisherInformation` now rejects non-finite
  Hamiltonian inputs, non-integer measurement budgets, boolean shot
  counts, and malformed/out-of-range/diagonal `coupling_pairs` before
  routing to the spectral QFI engine. The analysis API documents the
  production contract and keeps counts-derived sync/DLA estimates
  labelled as opt-in proxy diagnostics.
- [x] **PEC local multi-qubit coefficient decomposition.** Implemented
  2026-05-12: `pauli_twirl_decompose()` now returns tensor-product
  quasi-probability coefficients for `n_qubits >= 1`, preserving exact
  single-qubit Rust parity while documenting that correlated multi-qubit
  noise still requires a separately characterised inverse channel.
- [x] **Behavioural-test audit automation.** Implemented 2026-05-06:
  `tools/audit_test_behaviour.py` inventories test modules for
  assertion-bearing tests, exception contracts, parametrisation, and
  smoke-only tests so the manual behavioural audit can be prioritised.
- [x] **Behavioural-audit tool test coverage.** Implemented
  2026-05-06: `tests/test_audit_test_behaviour.py` now covers
  function and class-based test detection, assertion-helper contracts,
  `pytest.raises` contracts, parametrisation counts, smoke-only
  reporting, deterministic JSON/text output, and CLI failure status.
- [x] **Rust `.pyi` contract checker coverage.** Implemented
  2026-05-06: `tests/test_check_rust_pyi_exports.py` now covers PyO3
  export parsing, namespace stripping, private `.pyi` helper exclusion,
  matching-contract success output, and missing/stale export failure
  reporting for the CI typing-contract gate.
- [x] **Version-consistency hook coverage.** Implemented 2026-05-06:
  `tests/test_check_version_consistency.py` now covers matching version
  carriers, mismatched carrier reporting, missing carrier reporting, and
  unmatched-version-pattern reporting for the pre-commit release hook.
- [x] **Secret-scanner hook coverage.** Implemented 2026-05-06:
  `tests/test_check_secrets.py` now covers vault-token extraction
  without reading the real vault, added-line diff scanning, keyword
  secret detection, placeholder and variable-reference suppression, and
  safe redaction output.
- [x] **Commit-trailer hook coverage.** Implemented 2026-05-06:
  `tests/test_check_commit_trailers.py` now covers accepted trailer
  messages, missing-trailer rejection, banned-subject-word rejection,
  default body-word allowance, stricter body scanning, and help output.
- [x] **Benchmark CLI failure-policy coverage.** Implemented
  2026-05-06: `tests/test_bench_cli.py` now covers default stop-on-first
  harness failure, `--keep-going` execution after failures, and
  `--no-diff` skipping of artefact-drift checks.
- [x] **Local preflight tool coverage.** Implemented 2026-05-06:
  `tests/test_preflight_tool.py` now covers gate pass/fail output,
  stdout/stderr tail reporting, `--no-tests`, `--no-coverage`, and
  stop-on-first-failure behaviour without running the full suite.
- [x] **Coverage-gap audit automation.** Implemented 2026-05-06:
  `tools/audit_coverage_gaps.py` parses `coverage.xml`, inventories
  package source files missing from coverage reports or below a
  per-file threshold, and provides `--fail-on-gap` for release-gating
  without running tests or treating line coverage as scientific
  validation. Usage and boundaries are documented in
  `docs/coverage_gap_audit_2026-05-06.md`.
- [x] **Coverage XML freshness guard.** Implemented 2026-05-07:
  `tools/audit_coverage_gaps.py` now emits an explicit
  `coverage_report_warning` when no selected package source files match
  the supplied coverage XML, uses `defusedxml` for hardened XML parsing,
  and has focused behavioural coverage for the warning path.
- [x] **Coverage-gap CI observation.** Implemented 2026-05-06:
  the Python 3.12 coverage job now emits `coverage-gap-audit.json`
  from `tools/audit_coverage_gaps.py` and uploads it as a 30-day
  artifact. It is intentionally observational rather than a hard
  per-file gate until the broader coverage-to-100% release task closes.
- [x] **Behavioural assertion-helper recognition.** Implemented
  2026-05-06: the behavioural audit now counts assertion helper calls
  such as `assert_*` functions instead of misclassifying them as smoke
  tests.
- [x] **Class-based test audit coverage.** Implemented 2026-05-06:
  `tools/audit_test_behaviour.py` now counts `test_` methods inside
  `Test*` classes, so class-based pytest modules are no longer
  misreported as empty.
- [x] **Behavioural-test audit.** Completed 2026-05-06:
  `docs/behavioural_test_audit_closure_2026-05-06.md` records the
  final audit state after targeted hardening passes. The current
  automated inventory covers `319` test modules and reports no
  smoke-only tests; broader coverage-to-100% work remains open as a
  separate release task.
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
- [x] **Quantum-Kuramoto API-surface contract.** Implemented
  2026-05-07: `scpn-bench s6-api-contract` validates the proposed
  `quantum_kuramoto.*` export names, current source-module
  importability, duplicate/export-blocker state, and SCPN-specific
  warning rows while keeping package-skeleton creation blocked until
  boundary refactors are complete.

### Active paper and submission tasks

- [x] **DLA parity preprint submission package.** Completed 2026-05-06:
  `docs/dla_parity_submission_checklist_2026-05-06.md` freezes the
  conservative parity-sector/excitation-number framing, required job IDs,
  committed artefact index, unsupported claims, and final no-QPU
  pre-upload gate. Updated 2026-05-07 after the Phase 3 state/layout
  analysis so the checklist and paper source explicitly reject
  layout-independent and same-sector-state-independent parity-protection
  claims.
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
- [x] **Multi-device DLA replication execution.** Completed
  2026-05-06 on `ibm_marrakesh` after live backend selection,
  transpilation/depth gates, QPU budget confirmation, and explicit
  approval. Jobs `d7tnd3vljm6s73bcsh6g` and
  `d7tnehqudops73988btg` produced the raw-count artefact,
  generated summary, row metrics, and manifest in
  `data/phase3_multidevice_dla/` and
  `docs/phase3_multidevice_dla_manifest_2026-05-06.md`. Result:
  mixed/mostly opposite-sign backend-transfer evidence, later
  strengthened by matching full-basis readout correction, weakening any
  backend-stable DLA leakage-asymmetry claim.
- [x] **Systematic state/layout randomisation preregistration.**
  Completed 2026-05-06:
  `docs/dla_state_layout_randomisation_prereg_2026-05-06.md` defines
  the `n=4` state/depth/layout matrix, 495-circuit scope, 8--15 minute
  estimate, 20-minute ceiling, layout-selection gates, analysis plan,
  readout boundary, falsification rules, and output artefact paths. QPU
  execution remains separate and approval-gated.
- [x] **Systematic state/layout randomisation execution.** Completed
  2026-05-07 on `ibm_marrakesh` after live backend/layout selection,
  transpilation/depth gates, QPU budget confirmation, and explicit
  approval. Jobs `d7ts9avljm6s73bd2ej0` and
  `d7tsdnfljm6s73bd2j70` produced the committed raw-count artefact
  `data/phase3_state_layout_dla/phase3_state_layout_ibm_marrakesh_2026-05-06T224531Z.json`.
  Result interpretation is now closed by the generated analysis and
  manifest artefacts: the original contrast has mixed sign across
  layout-depth cells, within-sector state controls are significant, and
  layout spread exceeds the mean original contrast.
- [x] **Systematic state/layout randomisation analysis.** Completed
  2026-05-07: `scripts/analyse_phase3_state_layout_dla.py` generated
  the preregistered state/depth/layout leakage summary, row metrics,
  layout metrics, readout metrics, manifest, decision flags, and claim
  boundary from the committed raw-count artefact. The DLA parity paper
  and submission checklist were updated 2026-05-07 to include this
  mechanism-boundary result.
- [x] **Systematic state/layout randomisation live submitter.**
  Implemented 2026-05-07:
  `scripts/phase3_state_layout_dla_ibm.py` builds the preregistered
  495-circuit state/layout matrix, selects three connected four-qubit
  windows before outcome data exists, live-transpiles with fixed
  initial layouts, records readiness artefacts, enforces depth/gate and
  20-minute QPU ceilings, and requires both `--submit` and
  `--confirm-budget` before any IBM job is launched.
- [x] **Full readout-mitigation calibration preregistration.**
  Completed 2026-05-06:
  `docs/readout_full_basis_calibration_prereg_2026-05-06.md` defines
  the eligibility boundary, basis-state calibration matrix for `n=4,6,8`,
  QPU-time estimates and ceilings, live layout gates, analysis plan,
  falsification rules, and output artefact paths. Calibration execution
  remains separate and approval-gated.
- [x] **Full readout-mitigation calibration execution.** Completed
  2026-05-06 for the Phase 3 `ibm_marrakesh` `n=4` dataset on physical
  qubits `[5,6,7,8]` after live readiness checks, budget confirmation,
  and explicit approval. Job `d7tnljvljm6s73bcsql0` produced the
  full-basis assignment artefacts under `data/readout_full_basis/` and
  manifest `docs/readout_full_basis_manifest_2026-05-06.md`.
  Calibration quality: mean retention `0.96999`, max parity flip
  `0.04724`, condition number `1.07570`.
- [x] **Phase 3 full-basis readout correction.** Completed
  2026-05-06: the stable `ibm_marrakesh` full-basis assignment matrix
  was applied offline to the Phase 3 DLA rows, generating
  `data/phase3_multidevice_dla/phase3_multidevice_readout_corrected_summary_2026-05-06.json`
  and
  `data/phase3_multidevice_dla/phase3_multidevice_readout_corrected_rows_2026-05-06.csv`.
  The corrected asymmetry is non-positive at all promoted depths, so
  readout correction does not rescue backend-transfer replication.
- [x] **GUESS / symmetry-decay calibration preregistration.**
  Completed 2026-05-06:
  `docs/guess_symmetry_decay_prereg_2026-05-06.md` defines the
  folded-noise parity-leakage witness protocol, readiness basis, `n=4`
  circuit matrix, 196-circuit default scope, 5--12 minute estimate,
  15-minute ceiling, live folding/depth gates, analysis plan,
  falsification rules, and output artefact paths. Execution remains
  separate and approval-gated.
- [x] **GUESS / symmetry-decay calibration execution.** Completed
  2026-05-07 on `ibm_marrakesh` after backend selection, committed
  submitter checks, live folded-circuit readiness, conservative
  12-minute QPU estimate under the 15-minute ceiling, and explicit
  approval. Jobs `d7tt5lkt738s73cib64g` and
  `d7tt7oaudops7398fdt0` produced the raw-count artefact
  `data/phase3_guess_dla/phase3_guess_ibm_marrakesh_2026-05-06T234602Z.json`.
- [x] **GUESS / symmetry-decay calibration analysis.** Completed
  2026-05-07: `scripts/analyse_phase3_guess_dla.py` generated
  `data/phase3_guess_dla/phase3_guess_summary_2026-05-07.json`,
  fit rows, witness/extrapolation rows, and
  `docs/phase3_guess_dla_manifest_2026-05-07.md`. Result: 6 raw
  log-survival fits and 5 exact-state-readout-corrected fits pass the
  preregistered monotone/R2/RMSE witness criteria; universal GUESS
  mitigation, backend-general transfer, and full confusion-matrix
  mitigation remain blocked claims.
- [x] **Layer-selective qubit assignment preregistration.** Completed
  2026-05-06:
  `docs/layer_selective_qubit_assignment_prereg_2026-05-06.md`
  defines the coupling-aware scoring rule, comparator layouts, offline
  readiness matrix, optional 152-circuit hardware follow-up, 4--10
  minute estimate, 12-minute ceiling, live gates, analysis plan,
  falsification rules, and artefact paths. Execution remains separate
  and approval-gated.
- [x] **Layer-selective qubit assignment offline readiness audit.**
  Completed 2026-05-07:
  `scripts/analyse_layer_selective_readiness.py` consumes the committed
  Phase 3 state/layout artefact and generates
  `data/phase3_layer_layout/layer_selective_readiness_ibm_marrakesh_2026-05-07.json`,
  `data/phase3_layer_layout/layer_selective_transpile_rows_2026-05-07.csv`,
  and `docs/phase3_layer_layout_readiness_2026-05-07.md`. Decision:
  `blocked_missing_comparators`, because the saved artefact has
  connected low-readout layout rows but not the preregistered default,
  SABRE, and true layer-selective comparator matrix required before a
  hardware follow-up can be promoted.
- [x] **Layer-selective comparator matrix.** Completed 2026-05-07:
  `scripts/generate_layer_selective_comparator_matrix.py` opened a
  no-submit `ibm_marrakesh` backend snapshot and generated the missing
  default, SABRE, and true layer-selective transpilation matrix in
  `data/phase3_layer_layout/layer_selective_comparator_matrix_ibm_marrakesh_2026-05-07.json`,
  row CSV, SHA256 sidecar, and
  `docs/phase3_layer_layout_comparator_matrix_2026-05-07.md`.
  Decision: `blocked_layer_selective_worse_than_default`; layer-selective
  increased max depth by `79.5 %` and max two-qubit gates by `46.4 %`
  versus default, so the optional hardware follow-up is not promoted.
- [ ] **Layer-selective qubit assignment execution.** Blocked until a
  future backend snapshot or revised layer-selective heuristic passes the
  preregistered comparator matrix. Current 2026-05-07 matrix rejects QPU
  submission, so running this job now would not isolate a useful layout
  mechanism.
- [x] **Entanglement entropy or tomography check preregistration.**
  Completed 2026-05-06:
  `docs/entanglement_tomography_prereg_2026-05-06.md` defines the
  reduced-tomography/shadow-tomography decision rule, small-`n` claim
  boundary, offline readiness matrix, optional hardware scope, circuit
  and QPU-time ceilings, live gates, analysis plan, falsification rules,
  and artefact paths. Execution remains separate and approval-gated.
- [x] **Entanglement entropy or tomography offline readiness.**
  Completed 2026-05-07:
  `scripts/generate_entanglement_tomography_readiness.py` generated exact
  reduced-Pauli reference values, half-chain purity proxies, basis
  settings, and circuit-count gates for DLA parity plus FIM `n=4`
  families. Artefacts:
  `data/phase3_entanglement_tomography/entanglement_tomography_readiness_2026-05-07.json`,
  `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`,
  and `docs/phase3_entanglement_tomography_readiness_2026-05-07.md`.
  Decision: `ready_for_optional_hardware_preregistration`; the promoted
  reduced-tomography block uses `9` basis settings and `166` total
  circuits, so full tomography is unnecessary and the optional hardware
  block is scientifically promotable after backend selection, live
  transpilation gates, budget confirmation, and explicit approval.
- [ ] **Entanglement entropy or tomography hardware execution.** Optional
  QPU follow-up remains blocked until backend selection, live
  transpilation, QPU-minute estimate, and explicit approval are completed
  from committed code.
- [x] **Depth-optimal native decomposition preregistration.**
  Completed 2026-05-06:
  `docs/depth_optimal_native_decomposition_prereg_2026-05-06.md`
  defines comparator circuits, native-target candidate rules, offline
  readiness matrix, equivalence gates, optional 160-circuit hardware
  follow-up, 12-minute ceiling, live gates, analysis plan,
  falsification rules, and artefact paths. Execution remains separate
  and approval-gated.
- [x] **Depth-optimal native decomposition offline readiness.**
  Completed 2026-05-07:
  `scripts/generate_native_decomposition_readiness.py` generated local
  basis-gate resource rows plus unitary/observable equivalence rows for
  generic Pauli evolution, current XY compiler, and native-targeted
  `rxx+ryy` decomposition across `n=4,6,8` readiness cases. Artefacts:
  `data/phase3_native_decomposition/native_decomposition_readiness_2026-05-07.json`,
  `data/phase3_native_decomposition/native_decomposition_transpile_rows_2026-05-07.csv`,
  `data/phase3_native_decomposition/native_decomposition_equivalence_rows_2026-05-07.csv`,
  and `docs/phase3_native_decomposition_readiness_2026-05-07.md`.
  Decision: `blocked_current_xy_invalid_no_native_gain_vs_generic`. The
  native-targeted `rxx+ryy` path passes equivalence but has no median
  depth or two-qubit-gate gain versus the generic Pauli baseline, while
  the current XY compiler comparator fails equivalence and cannot be used
  as a valid resource baseline. Optional hardware execution is therefore
  not promoted.
- [ ] **Depth-optimal native decomposition hardware execution.** Blocked
  until a revised equivalent candidate shows a resource gain in the
  offline gate, then passes backend selection, live transpilation,
  budget confirmation, and explicit approval for QPU submission.
- [x] **Variational quantum simulation alternative preregistration.**
  Completed 2026-05-06:
  `docs/vqs_alternative_prereg_2026-05-06.md` defines candidate VQS
  modes, offline readiness matrix, promotion gates, tolerances,
  optional hardware scope, QPU-time ceilings, live gates, analysis
  plan, falsification rules, and artefact paths. Execution remains
  separate and approval-gated.
- [x] **Variational quantum simulation alternative offline readiness.**
  Completed 2026-05-07:
  `scripts/generate_vqs_alternative_readiness.py` generated exact-state
  VQS refit rows for `n=4` DLA, popcount, and FIM cases plus local
  basis-gate resource rows for `n=4,6,8` Trotter and VQS candidate
  circuits. Artefacts:
  `data/phase3_vqs_alternative/vqs_readiness_2026-05-07.json`,
  `data/phase3_vqs_alternative/vqs_candidate_rows_2026-05-07.csv`,
  `data/phase3_vqs_alternative/vqs_resource_rows_2026-05-07.csv`, and
  `docs/phase3_vqs_alternative_readiness_2026-05-07.md`. Decision:
  `blocked_no_vqs_candidate_passed_promotion_gate`. No shallow VQS
  ansatz family passed both the preregistered target-observable accuracy
  gate and the compiled-resource gate, so hardware execution is not
  promoted.
- [ ] **Variational quantum simulation alternative hardware execution.**
  Blocked until a revised VQS candidate passes the offline promotion
  gate, backend selection, live transpilation, budget confirmation, and
  explicit approval for QPU submission.
- [x] **Multi-circuit QEC demonstration preregistration.** Completed
  2026-05-06:
  `docs/multicircuit_qec_prereg_2026-05-06.md` defines required
  baselines, logical-error metrics, observable tolerances, offline
  readiness matrix, promotion gates, optional 180-circuit hardware
  scope, 15-minute ceiling, live gates, analysis plan, ablations,
  falsification rules, and artefact paths. Execution remains separate
  and approval-gated.
- [x] **Multi-circuit QEC demonstration offline readiness.**
  Completed 2026-05-07:
  `scripts/generate_multicircuit_qec_readiness.py` generated
  Monte Carlo logical-failure rows for the unencoded physical baseline,
  standard MWPM decoder, K-matrix-weighted physics-aware decoder, and
  feature-disabled ablation across ideal, depolarising, and
  readout-biased noise models. It also generated encoded/unencoded
  circuit-resource rows for the DLA parity pair. Artefacts:
  `data/phase3_multicircuit_qec/qec_readiness_2026-05-07.json`,
  `data/phase3_multicircuit_qec/qec_decoder_rows_2026-05-07.csv`,
  `data/phase3_multicircuit_qec/qec_resource_rows_2026-05-07.csv`,
  and `docs/phase3_multicircuit_qec_readiness_2026-05-07.md`.
  Decision:
  `blocked_physics_aware_decoder_did_not_beat_baselines`. Hardware
  execution is not promoted because the physics-aware decoder did not
  beat the standard decoder and its own feature-disabled ablation under
  the preregistered logical metric.
- [x] **Multi-circuit QEC offline-boundary terminology hardening.**
  Implemented 2026-05-12:
  the readiness generator and committed artefacts now identify the
  encoded comparator as `distance3_surface_code_offline` and describe
  the supported claim as a distance-3 surface-code offline
  logical-failure comparison, avoiding imprecise informal language while
  preserving the no-hardware and no-fault-tolerance boundary.
- [ ] **Multi-circuit QEC demonstration hardware execution.** Blocked
  until a revised QEC/decoder candidate passes offline logical metrics
  against unencoded, standard-decoder, and ablation baselines, then
  passes backend selection, live transpilation, budget confirmation,
  and explicit approval for QPU submission.

### Visibility and registration tasks

- [x] **GitHub topics.** Completed 2026-04-17.
- [x] **QOSF awesome-quantum-software.** Merged 2026-03-30.
- [x] **CiteAs verification.** Verified 2026-04-17.
- [x] **PyPI publication.** Completed for the current package line.
- [x] **OpenSSF Best Practices.** Passing badge is present.
- [x] **Zenodo DOI.** Existing DOI is live.
- [x] **GitHub Pages docs.** Published.
- [x] **Software Heritage SWHID follow-up.** Completed 2026-05-06:
  `docs/software_heritage_swhid_2026-05-06.md` records the successful
  Software Heritage save request, full visit status, origin SWHID, and
  archived snapshot SWHID.
- [x] **Zenodo communities and metadata refresh preparation.**
  Completed 2026-05-06:
  `docs/zenodo_metadata_refresh_checklist_2026-05-06.md` records the
  DOI, current version consistency, community targets, related
  identifiers, claim boundary, and manual-session update procedure.
- [x] **Zenodo metadata refresh execution.** Completed 2026-05-06:
  `docs/zenodo_metadata_refresh_execution_2026-05-06.md` records the
  authenticated Zenodo API edit/publish cycle. Public metadata now
  reports version `0.9.6`, publication date `2026-03-29`, license
  `agpl-3.0-or-later`, bounded description, and refreshed keywords for
  record `10.5281/zenodo.18821930` under concept DOI
  `10.5281/zenodo.18821929`.
- [x] **Zenodo community submission UI follow-up.** Completed
  2026-05-06:
  `docs/zenodo_community_submission_2026-05-06.md` records the pending
  Zenodo `community-inclusion` request for the `Research Software
  Engineering` community. Public record community membership remains
  pending curator acceptance.
- [x] **Qiskit Ecosystem Catalog.** Submitted 2026-05-06:
  `docs/qiskit_ecosystem_submission_2026-05-06.md` records the
  submitted project metadata and open review issue
  `Qiskit/ecosystem#1123`.
- [ ] **awesome-qiskit.** Blocked until Qiskit Ecosystem membership is
  accepted.
- [x] **Conda-forge recipe.** Submitted 2026-05-06:
  `docs/conda_forge_submission_2026-05-06.md` records the staged-recipes
  recipe metadata and open PR `conda-forge/staged-recipes#33236`.
- [x] **Metriq local readiness smoke.** Completed 2026-05-06:
  `docs/metriq_local_smoke_2026-05-06.md` records a no-QPU
  `metriq-gym` Bernstein--Vazirani local simulator dispatch and poll
  using the isolated `/home/anulum/.venvs/scpn-metriq` environment.
  This proves the Metriq CLI path is executable without treating SCPN
  paper benchmark tables as Metriq-native results.
- [x] **Metriq submission decision.** Completed 2026-05-07:
  `docs/metriq_submission_decision_2026-05-07.md` records a deliberate
  no-upload decision. The Metriq-Gym upload path was verified in
  dry-run mode for the local Bernstein--Vazirani smoke result, but no
  public upload was made because the available artefact is a generic
  local simulator result rather than an SCPN/Kuramoto--XY benchmark.
  Arbitrary Rust/VQE/DLA/FIM project tables remain out of scope for
  Metriq upload unless an accepted Metriq schema exists.
- [x] **Metriq SCPN benchmark schema proposal preparation.**
  Completed 2026-05-07:
  `docs/metriq_scpn_benchmark_schema_proposal_2026-05-07.md` defines a
  bounded Kuramoto--XY parity-leakage benchmark proposal with required
  inputs, circuit definition, primary score, secondary metrics,
  acceptance gates, non-claim boundary, and upstream draft text. It is a
  prepared proposal only; it has not been submitted upstream and is not
  an accepted Metriq-Gym schema.
- [ ] **Metriq SCPN benchmark schema upstream submission.** Optional
  external follow-up: submit the prepared schema proposal to
  Metriq-Gym maintainers, wait for review or acceptance, then run and
  upload results only under the accepted schema.
- [ ] **pyOpenSci review.** Submit the software package for review and
  possible JOSS fast-track.
- [x] **pyOpenSci review preparation.** Completed 2026-05-06:
  `docs/pyopensci_submission_readiness_2026-05-06.md` records the
  package scope, unsupported claims, metadata gate, reviewer evidence,
  no-QPU pre-submission gates, and suggested issue summary. External
  issue submission remains open until an issue URL is recorded.
- [ ] **JOSS submission.** Submit after paper package and metadata are
  aligned.
- [ ] **arXiv submission.** Submit the paper set once final PDFs,
  references, source tarballs, and artefact links are aligned.
- [x] **arXiv source packaging preparation.** Completed 2026-05-06:
  `docs/arxiv_source_packaging_readiness_2026-05-06.md` records the
  public source-bundle boundaries, build gates, claim gates, metadata
  draft, and private-file exclusions for the DLA parity, Rust/VQE
  methods, and SCPN/FIM papers. Actual arXiv upload remains open until
  explicitly approved and recorded.
- [x] **SciPy 2026 CFP preparation.** Completed 2026-05-06:
  `docs/scipy_2026_cfp_readiness_2026-05-06.md` records the proposed
  SciPy angle, format options, title options, draft abstract,
  reviewer-visible evidence, claims to avoid, and no-QPU submission
  boundary. Actual CFP submission remains external and should only be
  marked complete after the current CFP page and submitted proposal URL
  are recorded.
- [ ] **Community announcements.** Prepare Reddit, Qiskit Slack,
  Unitary Discord, Hacker News, LinkedIn, and X posts only after the
  public preprints are live.
- [x] **Registry-listing preparation pack.** Completed 2026-05-06:
  `docs/registry_listing_plan_2026-05-06.md` records the canonical
  metadata, target-by-target readiness, account/manual blockers,
  submission copy, and do-not-submit-yet boundaries for Quantiki, QOSF,
  best-of-python, Papers With Code, SciCrunch RRID, Open Hub, and
  Research Software Directory.
- [x] **Community-announcement preparation pack.** Completed
  2026-05-06:
  `docs/community_announcement_pack_2026-05-06.md` prepares bounded
  Reddit, Qiskit Slack, Unitary Discord, Hacker News, LinkedIn, and X
  copy while keeping publication deferred until public preprint links
  are live.
- [ ] **Registry listings.** Execute only the entries that pass their
  readiness gates. QOSF is already listed in
  `qosf/awesome-quantum-software`; Papers With Code and community
  announcements remain blocked until arXiv/JOSS links exist; Quantiki,
  SciCrunch RRID, Open Hub, and Research Software Directory require
  account/manual form completion.

### Deferred / CEO-gated strategic tracks

These tracks remain scoped in `docs/strategic_roadmap.md` and must not
be executed until individually activated.

- [ ] **S1** Hybrid classical--quantum feedback loop.
- [x] **S1 cross-shot feedback-loop foundation.** Implemented
  2026-05-06: `hardware/feedback_loop.py` adds scheduler/observer
  protocols, `FeedbackRunner`, step records, latency and QPU-budget
  gates, explicit hardware approval enforcement, and a proportional
  metric observer. Tests and documentation are in
  `tests/test_feedback_loop.py` and
  `docs/hybrid_feedback_loop_s1_2026-05-06.md`. This does not submit
  IBM jobs and does not claim intra-shot feedback.
- [x] **S1 realtime-controller simulator scheduler.** Implemented
  2026-05-06: `RealtimeControllerScheduler` wraps
  `RealtimeSyncFeedbackController` as a zero-QPU scheduler, adds bounded
  cross-shot coupling overrides, deterministic finite-shot seeds,
  auditable simulator metrics, focused tests, and S1 documentation.
- [x] **S1 no-QPU feedback latency benchmark.** Implemented
  2026-05-06: `scripts/benchmark_s1_feedback_loop.py` and
  `scpn-bench s1-feedback` regenerate JSON/CSV latency artefacts for
  `FeedbackRunner` plus `RealtimeControllerScheduler`; pipeline
  performance documentation records the command and IBM/QPU boundary.
- [x] **S1 provider-neutral submission-readiness package.** Implemented
  2026-05-06: `hardware/feedback_submission.py` builds no-submission
  dynamic-circuit readiness packages with circuit summaries, QPU-budget
  estimates, platform capability checks, ready/blocked/manual-review
  decisions for IBM, generic gate, analogue, CV, and simulator targets,
  focused tests, and S1 documentation.
- [x] **Hardware-job dossier standard.** Implemented 2026-05-06:
  `hardware/job_dossier.py` defines the required dossier schema for
  every submission-ready hardware job, including purpose, hypothesis,
  falsification condition, observables, circuit summary, QPU budget,
  platform fit, risks, decision tree, paper impact, follow-up avenues,
  possibilities opened, claim boundary, and reproducibility package.
  The S1 readiness package embeds the dossier by default.
- [x] **S1 preregistration manifest export.** Implemented
  2026-05-06: `scripts/export_s1_feedback_preregistration.py`
  exports JSON and Markdown preregistration manifests from the
  provider-neutral S1 package and embedded hardware-job dossier. The
  default budget explicitly includes a monitored feedback arm and a
  matched open-loop control arm; no credentials are read and no hardware
  job is submitted.
- [x] **S1 provider dry-run payloads.** Implemented 2026-05-06:
  `hardware/feedback_dryrun.py` emits no-submit payloads for IBM
  Runtime dynamic circuits, provider-neutral OpenQASM 3 style gate
  execution, and analogue-native review. The preregistration export
  embeds the dry-run bundle and keeps analogue targets behind a separate
  native-feedback dossier requirement.
- [x] **S1 approval-gated hardware scheduler boundary.** Implemented
  2026-05-06: `hardware/feedback_hardware_scheduler.py` provides a
  fail-closed scheduler wrapper requiring an injected provider submitter,
  explicit approval, matching provider, matching preregistration package
  hash, and estimated/reported QPU budget compliance before any hardware
  submission can pass.
- [x] **S1 no-submit capability probes.** Implemented 2026-05-06:
  `hardware/feedback_capability_probe.py` evaluates backend metadata
  snapshots against the S1 dynamic-circuit package for qubit count,
  shot/circuit limits, mid-circuit measurement, conditional control,
  conditional reset, and cross-shot batch support. The preregistration
  export embeds template probe decisions without reading credentials or
  submitting jobs.
- [x] **S1 raw-count analysis harness.** Implemented 2026-05-06:
  `scripts/analyse_s1_feedback_hardware.py` defines the preregistered
  feedback-vs-matched-open-loop-control analysis before live hardware
  submission, including raw-count schema checks, per-arm summaries,
  target-error improvement, decision boundary, and claim boundary tests.
- [x] **S1 synthetic analysis rehearsal fixture.** Implemented
  2026-05-06: `data/s1_feedback_loop/s1_feedback_synthetic_raw_counts_2026-05-06.json`
  provides a non-hardware fixture for exercising the preregistered S1
  raw-count analysis path, and the S1 documentation records the exact
  raw-count JSON schema required for live packages.
- [x] **S1 no-submit provider metadata adapters.** Implemented
  2026-05-06: `hardware/feedback_provider_metadata.py` converts
  provider-neutral metadata records and Qiskit-style backend objects into
  `BackendCapabilitySnapshot` inputs for the S1 capability probes without
  reading credentials, opening provider sessions, or submitting jobs.
- [x] **S1 one-command readiness bundle.** Implemented 2026-05-06:
  `scripts/reproduce_s1_feedback_readiness.py` and
  `scpn-bench s1-feedback-ready` regenerate the no-QPU S1 latency,
  preregistration, provider dry-run, capability-example, and synthetic
  analysis artefacts in one command.
- [x] **S1 live-submission preflight checklist.** Implemented
  2026-05-06: `docs/s1_live_submission_preflight_2026-05-06.md`
  records the mandatory manual gates for artefacts, scientific purpose,
  provider capability, budget, reproducibility, approval records, stop
  conditions, and post-run handling before any live S1 provider submitter
  may be wired.
- [x] **S1 IBM metadata probe command.** Implemented 2026-05-06:
  `scripts/probe_s1_ibm_metadata.py` writes no-submit capability
  decisions from offline provider metadata JSON or an already-authenticated
  Qiskit Runtime backend metadata lookup. The command does not accept
  credential strings, does not submit jobs, and records `hardware_submission=false`.
- [x] **S1 generic gate metadata probe command.** Implemented
  2026-05-06: `scripts/probe_s1_generic_gate_metadata.py` writes
  no-submit, no-network capability decisions for non-IBM gate-based
  targets from provider-neutral metadata JSON and records
  `hardware_submission=false` plus `network_access=false`.
- [x] **S1 readiness index.** Implemented 2026-05-06:
  `docs/s1_feedback_readiness_index_2026-05-06.md` consolidates the S1
  no-QPU readiness state, artefact inventory, exact commands,
  preregistered job shape, claim boundary, platform interpretation, and
  remaining live-submission blockers.
- [ ] **S2** Quantum advantage benchmarks at scale.
- [x] **S2 scaling protocol manifest.** Implemented 2026-05-06:
  `benchmarks/advantage_protocol.py` and
  `scripts/export_s2_scaling_protocol.py` define the S2 no-claim
  scaling protocol, required baselines, size grid, output schema,
  acceptance rules, falsification rules, and claim boundary before any
  heavy sweep or hardware row is promoted.
- [x] **S2 scaling row validator.** Implemented 2026-05-06:
  `validate_scaling_rows` and `scripts/validate_s2_scaling_rows.py`
  enforce the preregistered S2 row schema, required baselines, known
  baseline labels, valid statuses, and wall-time requirements for
  successful rows before any scaling table or figure is promoted.
- [x] **S2 validator matrix completeness hardening.** Implemented
  2026-05-12: `validate_scaling_rows` now enforces every required
  baseline per observed size, rejects off-protocol sizes and protocol
  ids, requires finite non-negative timing plus memory for `ok` rows,
  and requires explanatory notes for skipped or failed rows.
- [x] **S2 validator provenance hardening.** Implemented 2026-05-12:
  duplicate `(n_qubits, baseline)` rows are rejected, and row-level
  provenance payloads must use structured metric, command, machine,
  dependency, git-commit, and notes fields before a scaling matrix can
  pass validation.
- [x] **S2 lite scaling harness.** Implemented 2026-05-06:
  `scripts/bench_s2_scaling_lite.py` emits protocol-compliant rows for
  small selected sizes, measures cheap classical ODE and dense exact
  diagonalisation rows, records explicit `skipped` rows for heavier
  required baselines, validates the output, and records no hardware
  submission or advantage claim.
- [x] **S2 lite sparse eigensolver row.** Implemented 2026-05-06:
  the lite scaling harness now measures gated `sparse_eigsh` rows for
  small sizes, including ground energy, residual norm, Hilbert dimension,
  wall time, and memory estimate, instead of treating sparse classical
  support as skipped in the rehearsal path.
- [x] **S2 lite MPS/TN spoofability row.** Implemented 2026-05-06:
  the lite scaling harness now measures small-size tensor-network
  spoofability diagnostics from exact ground-state Schmidt spectra,
  including max bond, worst-cut discarded weight, midchain entropy,
  Hilbert dimension, wall time, and memory estimate.
- [x] **S2 lite Aer/statevector row.** Implemented 2026-05-06:
  the lite scaling harness now measures gated Aer/statevector-style
  statevector evolution rows through `QuantumKuramotoSolver`, including
  Trotter steps, circuit depth, final synchronisation observables,
  Hilbert dimension, wall time, and memory estimate.
- [x] **S2 benchmark CLI wiring.** Implemented 2026-05-06:
  `scpn-bench s2-scaling-lite` regenerates the S2 scaling protocol and
  lite scaling rows through the canonical benchmark CLI without hardware
  submission or advantage claims.
- [x] **S2 claim-boundary report.** Implemented 2026-05-06:
  `scripts/report_s2_scaling_claim_boundary.py` reads S2 rows, validates
  them, and emits JSON/Markdown reports listing allowed claims,
  forbidden claims, remaining blockers, validation state, hardware
  submission status, and advantage-claim status.
- [x] **S2 scaling readiness index.** Implemented 2026-05-06:
  `docs/s2_scaling_readiness_index_2026-05-06.md` summarises the S2
  lite baseline rows, canonical command, artefacts, allowed claims,
  forbidden claims, full-campaign blockers, hardware boundary, and next
  non-QPU scaling step.
- [x] **S2 lite memory instrumentation and size gates.** Implemented
  2026-05-06: `scripts/bench_s2_scaling_lite.py` now records
  `tracemalloc` peak bytes for measured rows, keeps estimated dense or
  statevector bytes in metric payloads, and exposes explicit dense,
  sparse, tensor-network, and statevector size gates.
- [x] **S2 full-campaign execution plan.** Implemented 2026-05-07:
  `scripts/plan_s2_full_scaling_campaign.py` enumerates the full
  `N=4,6,8,10,12,14,16,18,20` scaling matrix across every protocol
  baseline, classifies rows as lite-measured, ready for deliberate
  no-QPU full-campaign execution, size-gated, optional GPU, or blocked
  optional hardware, and generates
  `data/s2_advantage_scaling/s2_full_campaign_plan_2026-05-07.json`,
  `data/s2_advantage_scaling/s2_full_campaign_rows_2026-05-07.csv`,
  and `docs/s2_full_campaign_plan_2026-05-07.md`. Decision:
  `ready_for_deliberate_no_qpu_full_classical_campaign`; hardware rows
  and broad quantum-advantage language remain blocked.
- [x] **S2 bounded full-campaign execution slice.** Implemented
  2026-05-07: `scripts/run_s2_full_campaign_slice.py` consumes the S2
  campaign plan and executes only no-QPU required rows under explicit
  dense, sparse, tensor-network, and statevector size caps. The default
  `n=8` slice generated
  `data/s2_advantage_scaling/s2_full_campaign_slice_n8_2026-05-07.json`,
  `data/s2_advantage_scaling/s2_full_campaign_slice_rows_n8_2026-05-07.csv`,
  and `docs/s2_full_campaign_slice_n8_2026-05-07.md`. Decision:
  `completed_no_qpu_campaign_slice`; this is not the full campaign, not
  hardware evidence, and not a quantum-advantage claim.
- [x] **S2 bounded full-campaign execution slice, `n=10`.** Implemented
  2026-05-07: the same no-QPU slice runner executed the next bounded
  `n=10` scaling slice and generated
  `data/s2_advantage_scaling/s2_full_campaign_slice_n10_2026-05-07.json`,
  `data/s2_advantage_scaling/s2_full_campaign_slice_rows_n10_2026-05-07.csv`,
  and `docs/s2_full_campaign_slice_n10_2026-05-07.md`. The slice recorded
  5 executed rows, 5 successful rows, 0 skipped rows, and decision
  `completed_no_qpu_campaign_slice`; this is still not the full campaign,
  not hardware evidence, and not a quantum-advantage claim.
- [x] **S2 bounded full-campaign execution slice, `n=12`.** Implemented
  2026-05-07: the no-QPU slice runner executed the next bounded `n=12`
  scaling slice with dense, sparse, tensor-network, and statevector caps
  all set to 12 qubits. It generated
  `data/s2_advantage_scaling/s2_full_campaign_slice_n12_2026-05-07.json`,
  `data/s2_advantage_scaling/s2_full_campaign_slice_rows_n12_2026-05-07.csv`,
  and `docs/s2_full_campaign_slice_n12_2026-05-07.md`. The slice recorded
  5 executed rows, 5 successful rows, 0 skipped rows, and decision
  `completed_no_qpu_campaign_slice`; this is still not the full campaign,
  not hardware evidence, and not a quantum-advantage claim.
- [x] **S2 slice progress aggregation.** Implemented 2026-05-07:
  `scripts/report_s2_slice_progress.py` aggregates the completed bounded
  no-QPU S2 slices for `n=8,10,12` from committed JSON/CSV artefacts and
  writes `data/s2_advantage_scaling/s2_slice_progress_report_2026-05-07.json`
  plus `docs/s2_slice_progress_report_2026-05-07.md`. The aggregate report
  records 15 successful rows out of 15 executed rows, total measured wall time
  `510907.947844` ms, maximum recorded memory `2416024559` bytes, and decision
  `ready_for_next_bounded_no_qpu_slice`; this is still not hardware evidence,
  not full S2 completion, and not a quantum-advantage claim.
- [x] **S2 `n=14` resource gate.** Implemented 2026-05-07:
  `scripts/report_s2_n14_resource_gate.py` compares the planned `n=14`
  required rows against the committed `n=8,10,12` progress artefacts and
  writes `data/s2_advantage_scaling/s2_n14_resource_gate_2026-05-07.json`
  plus `docs/s2_n14_resource_gate_2026-05-07.md`. The report records a
  dense-matrix estimate of `4294967296` bytes, prior maximum recorded memory
  `2416024559` bytes, dense/prior-memory ratio `1.7777`, and decision
  `blocked_for_scheduled_or_offloaded_no_qpu_run`; this is a resource gate,
  not an `n=14` execution result, not hardware evidence, and not a
  quantum-advantage claim.
- [x] **S2 ML350 `n=14` full no-QPU slice.** Implemented 2026-05-07:
  the resource-gated `n=14` slice was executed on ML350 in the non-SAS
  workspace `/home/anulum/scpn_quantum_control_ntfs_worktree` from NTFS
  source commit `842c529da89f88b68a75582fba7a0076a1a34c1f`. The generated
  artefacts are
  `data/s2_advantage_scaling/s2_full_campaign_slice_n14_2026-05-07.json`,
  `data/s2_advantage_scaling/s2_full_campaign_slice_rows_n14_2026-05-07.csv`,
  and `docs/s2_full_campaign_slice_n14_2026-05-07.md`. The slice recorded
  5 executed rows, 5 successful rows, 0 skipped rows, decision
  `completed_no_qpu_campaign_slice`, all caps set to 14 qubits, and no QPU
  time. The ML350 timing log reported wall time `3:03:29`, peak RSS
  `21583552` KiB, and 0 swaps. This is still not hardware evidence, not full
  S2 completion, and not a quantum-advantage claim.
- [ ] **S3** ML-augmented pulse / ansatz design.
- [x] **S3 deterministic pulse/ansatz design-readiness gate.**
  Implemented 2026-05-06:
  `benchmarks/s3_design_protocol.py` and
  `scripts/export_s3_design_readiness.py` define a no-QPU candidate
  ranking protocol for structured Kuramoto-XY ansatz candidates and
  hypergeometric pulse-schedule candidates. The artefact records
  `hardware_submission=false`, `ml_training_performed=false`, allowed
  claims, forbidden claims, and required follow-ups before any ML
  surrogate training or pulse-level hardware work.
- [x] **S3 benchmark CLI wiring.** Implemented 2026-05-06:
  `scpn-bench s3-design-ready` regenerates the deterministic S3
  design-readiness JSON/Markdown artefacts through the canonical
  benchmark CLI without hardware submission.
- [x] **S3 design readiness index.** Implemented 2026-05-06:
  `docs/s3_design_readiness_index_2026-05-06.md` documents the
  candidate families, canonical command, allowed claims, forbidden
  claims, and the next S3 steps: held-out surrogate training, VQE or
  observable validation, provider-specific pulse feasibility probes, and
  hardware-job dossiers before execution.
- [x] **S3 held-out surrogate rehearsal.** Implemented 2026-05-06:
  `scripts/train_s3_design_surrogate.py` expands the deterministic
  pulse/ansatz candidate grid across small system sizes, trains a
  closed-form ridge linear surrogate on proxy scores, and reports
  train, held-out, and per-family metrics. The artefact remains no-QPU,
  proxy-labelled, and explicitly non-evidential for hardware pulse or
  VQE improvement.
- [x] **S3 promoted ansatz observable validation.** Implemented
  2026-05-06: `scripts/validate_s3_ansatz_observables.py` checks
  lowest-resource promoted ansatz candidates against exact statevector
  energy expectation, dense exact ground energy, energy error, and a
  synchronisation proxy. The artefact is no-QPU and explicitly not VQE
  optimisation, pulse validation, hardware evidence, or an advantage
  claim.
- [x] **S3 provider-specific pulse feasibility probes.** Implemented
  2026-05-06: `hardware/pulse_feasibility.py` and
  `scripts/probe_s3_pulse_feasibility.py` assess provider metadata
  against the S3 hypergeometric pulse schedule for qubit count, pulse
  count, duration, sample spacing, pulse-control support, and native-XY
  support. The probe opens no provider session and submits no job.
- [x] **S3 promoted-candidate hardware-job dossiers.** Implemented
  2026-05-06: `scripts/export_s3_hardware_dossiers.py` packages the
  promoted ansatz and pulse follow-up routes into the standard hardware
  dossier schema, including purpose, hypothesis, falsification boundary,
  observables, QPU budget state, platform fit, risks, decision tree,
  paper impact, follow-up avenues, prerequisites, and reproducibility
  artefacts. The dossiers do not authorise hardware execution.
- [ ] **S4** Multi-hardware backend + pulse-level control.
- [x] **S4 multi-hardware no-submit readiness gate.** Implemented
  2026-05-06: `scpn-bench s4-multi-hardware-ready` exports
  provider-specific Pulser, Bloqade, and IBM pulse-level design payloads
  from the analogue Kuramoto compiler, wraps them in approval-gated
  execution plans, documents blocked claims and promotion gates, and
  performs no provider contact or QPU submission.
- [x] **S4 IBM pulse-level preregistration dossier.** Implemented
  2026-05-06: `scpn-bench s4-provider-preregistration` turns the IBM
  pulse-level design payload into a no-submit calibration-review dossier
  with purpose, hypothesis, falsification condition, expected capability
  observables, blocked QPU budget, platform fit, decision tree, paper
  impact, follow-up route, and prerequisites before any pulse job.
- [x] **S4 neutral-atom preregistration dossier.** Implemented
  2026-05-06: `scpn-bench s4-neutral-atom-preregistration` packages
  the Pulser and Bloqade neutral-atom payloads into a no-submit
  provider-object review dossier with geometry, SDK, emulator-only,
  comparator-observable, cost/credit, and QPU-budget gates before any
  cloud provider session.
- [ ] **S5** Open-data + classical validation harness.
- [x] **S5 public Phase 1 benchmark harness facade.** Implemented
  2026-05-06: `scpn_quantum_control.benchmark_harness` exposes
  `load_phase1_dataset`, `reproduce_phase1_statistics`, and
  `run_phase1_benchmark` as the community-facing open-data API over the
  committed Phase 1 DLA-parity raw counts and noiseless classical
  parity-conservation reference.
- [x] **S5 benchmark-suite CLI artefact.** Implemented 2026-05-06:
  `scpn-bench s5-benchmark-suite` regenerates the Phase 1 benchmark
  harness JSON/Markdown artefacts from committed raw counts, verifies
  published statistics against the tolerance bundle, records the
  classical baseline, and performs no QPU submission.
- [x] **S5 benchmark registry/index.** Implemented 2026-05-06:
  `scpn-bench s5-benchmark-registry` exports a public benchmark-family
  registry distinguishing the implemented Phase 1 DLA-parity benchmark
  from planned CHSH, BKT, OTOC, and DLA-dimension entries, with blockers
  and claim boundaries so unavailable rows are not mistaken for results.
- [ ] **S6** Decoupled `quantum-kuramoto` subpackage.
- [x] **S6 import-graph split audit.** Implemented 2026-05-07:
  `scpn-bench s6-split-audit` inventories candidate `phase`, `bridge`,
  `hardware`, and `accel` modules, classifies each row as reusable,
  needs-review, or SCPN-specific, records blockers and next steps, and
  explicitly blocks second-package publication until the boundary is
  manually reviewed.
- [x] **S6 boundary-review report.** Implemented 2026-05-07:
  `scpn-bench s6-boundary-review` converts the split audit into a
  conservative public-API proposal, defers config/provenance/analysis
  dependent rows, records compatibility requirements, and keeps the
  `quantum_kuramoto` package skeleton blocked until refactors close.
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
