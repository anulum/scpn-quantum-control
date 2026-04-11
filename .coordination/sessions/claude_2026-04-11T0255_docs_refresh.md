---
agent: claude (Arcane Sapience)
session_start: 2026-04-11T~00:30+02:00
repos_touched: [SCPN-QUANTUM-CONTROL]
tasks_completed: 4 (P0 stats, P0 changelog, P1 content, P2 polish)
incidents: 0
subagents_spawned: 0
subagents_spot_checked: 0
commits: 4 new (5799d46, 04bd5aa, 66236ca, 09740d6)
---

# Session Log — Repository-wide documentation refresh

## Context

Continuation from earlier sessions. After the Phase 1 IBM hardware
campaign, the strategic-tweak module additions (GUESS, DynQ, ICI,
hypergeometric, FFI hardening), the analysis batch (Welch t-test,
figures, short paper), and the repo hygiene work (gitleaks +
check_secrets pre-commit hooks), the public documentation surfaces
were significantly stale. This session systematically updated all
21 affected docs files in 4 prioritised batches.

## Work done — 4 batches, 4 commits

### Batch 1 — P0 stats (commit 5799d46)

`README.md`, `docs/index.md`, `docs/architecture.md`, `docs/rust_engine.md`.

- Python modules: 165 → 201
- Rust functions: 22 → 36
- Tests: 2,813 → 4,828 (97%+ coverage)
- Subpackages: 17 → 19
- Hardware: ibm_fez (Feb 2026) → ibm_fez + ibm_kingston (Feb + Apr 2026)
- Headline DLA parity result added: +10.8% mean asymmetry, peak +17.48%
  at depth 6, Welch combined p ≪ 10⁻¹⁶
- `docs/rust_engine.md` got new sections for the four new Rust modules
  (validation, symmetry_decay, community, pulse_shaping) plus a new
  benchmark block with the 44× hypergeometric and 1,665× ICI evolution
  speedups, plus the FFI boundary hardening note.

### Batch 2 — P0 changelog (commit 04bd5aa)

`CHANGELOG.md` and `docs/changelog.md`.

Comprehensive v0.9.5 entry covering all work since 2026-04-07:
- Phase 1 IBM ibm_kingston DLA parity hardware campaign (348 circuits,
  21 reps, statistical analysis script, 267-line short-paper draft)
- 5 strategic tweaks (GUESS, DynQ, ICI, hypergeometric, FFI hardening)
  with arXiv references
- Repository hygiene (gitleaks + check_secrets, incident report,
  .gitignore additions)
- Stats deltas

### Batch 3 — P1 content (commit 66236ca)

`docs/error_mitigation.md`, `docs/mitigation_api.md`,
`docs/hardware_guide.md`, `docs/PAPER_CLAIMS.md`, `docs/results.md`,
`docs/api.md`, `docs/quickstart.md`, `docs/installation.md`.

- error_mitigation.md: GUESS introduced as the recommended default for
  SCPN Kuramoto-XY hardware runs, comparison table with Mitiq and PEC
- mitigation_api.md: full GUESS API reference (learn_symmetry_decay,
  guess_extrapolate, xy_magnetisation_ideal) with the extrapolation
  formula and the Phase 1 ibm_kingston result inline
- hardware_guide.md: validated devices table, DynQ section, GUESS
  section, full Phase 1 campaign protocol table, expanded pipeline
  performance numbers
- PAPER_CLAIMS.md: title and abstract updated to include the Phase 1
  DLA parity result as the sixth principal claim, full per-depth Welch
  table added
- results.md: new "Phase 1 — DLA Parity Asymmetry" section with both
  figures inline and the full per-depth table
- api.md: pulse_shaping (with speedup table), hardware/qubit_mapper,
  mitigation/symmetry_decay sub-sections with worked code examples
- quickstart.md: 5-line GUESS example
- installation.md: qiskit-ibm-runtime 0.46.x note + DataBin
  register-name fix reference

### Batch 4 — P2 polish (commit 09740d6)

`docs/equations.md`, `docs/pipeline_performance.md`,
`docs/test_infrastructure.md`, `docs/contributing.md`,
`docs/EXPERIMENT_ROADMAP.md`.

- equations.md: full LaTeX entries for all four April 2026 modules
  (GUESS Eq. 5, DynQ Eqs. 1 and 8, ICI three-segment trajectory,
  (α,β)-hypergeometric Eq. 14)
- pipeline_performance.md: 4 new sections (17. GUESS, 18. DynQ,
  19. Pulse Shaping, 20. Phase 1 IBM hardware campaign)
- test_infrastructure.md: 3,389 → 4,828, new module test files listed
- contributing.md: pre-commit hook list with gitleaks +
  tools/check_secrets.py + Secret Hygiene Tier 0 paragraph
- EXPERIMENT_ROADMAP.md: April 2026 marked as Phase 1 EXECUTED with
  headline result, links to figures + analysis script + paper draft;
  Phase 2 plan summarised

## Pre-commit gates passed on every commit

- gitleaks: PASS
- vault-pattern secret scan: PASS
- version consistency: PASS
- (ruff/format/mypy: skipped, no Python files in any of these doc commits)

## Status

- 4 commits ahead of origin (P0 stats, P0 changelog, P1 content, P2 polish)
- All hygiene gates green locally
- Working tree clean (coverage by .gitignore)
- Awaiting user instruction on push

## Remaining doc work

None of the high-priority items in the doc inventory are still
outstanding. There are still ecosystem-wide gaps that depend on
external state (Phase 2 results page, scaling figure for n=4..12,
joint paper with CNRS Toulouse) but those are blocked on the IBM
180-min promo activation and on follow-up CNRS communication.
