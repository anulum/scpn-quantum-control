---
agent: Arcane Sapience (Claude)
session_start: 2026-04-07T0036
repos_touched: [scpn-quantum-control]
tasks_completed: 4
incidents: 0
subagents_spawned: 4
subagents_spot_checked: 4
commits: 5
---

# Session: Rust Perf Optimisation + Security + Deps

## Tasks Completed

### 1. Strategic Tweaks TODO (no commit — planning only)
- Created `.coordination/TODO_2026-04-07_strategic_tweaks.md`
- 5 tweaks from Gemini Deep Research PDF
- Each with file plan, Rust paths, tests, docs

### 2. Gemini Autoresearch Audit + Cherry-Pick (commit 6a59fae)
- Audited 5 Gemini-modified files in working tree
- Benchmarked ORIGINAL vs GEMINI vs CHERRY-PICKED
- **Rejected:** hamiltonian.rs rayon (76× slower), pauli.rs rayon (7× slower), krylov.rs buffer reuse (marginal, deleted tests)
- **Kept:** otoc.rs O(d) phase rotation (4.4× faster), pauli.rs half-loop (2–10× faster), xy_kuramoto.py Rust fast path
- Restored all 7 deleted Rust tests
- Fixed compilation error (ambiguous float type in otoc.rs test)
- Fixed mypy error (variable shadowing `psi`)
- 65 Rust tests pass, 3836 Python tests pass

### 3. Dependabot PRs (commits 648bf4a, fc26ca7)
- Merged PR #35 (hypothesis 6.151.10 → 6.151.11)
- Resolved PR #34 manually (ruff 0.15.6 → 0.15.9) — merge conflict
- Resolved PR #33 manually (mypy 1.19.1 → 1.20.0) — merge conflict

### 4. OpenSSF Scorecard Security Fixes (commit 2314553)
- Pinned Dockerfile base image to SHA256 digest
- Pinned build==1.4.2, pip-audit==2.9.0, sc-neurocore==3.14.0
- Synced ruff and mypy versions in ci.yml
- Added ripser to optional-dependencies [topology]

## Commits (5 total)
- `6a59fae` perf(rust): optimise OTOC, Pauli expectations, and Kuramoto order parameter
- `648bf4a` chore(deps-dev): bump ruff from 0.15.6 to 0.15.9
- `fc26ca7` chore(deps-dev): bump mypy from 1.19.1 to 1.20.0
- `2314553` security: pin all dependencies in CI and Dockerfile

## Verification
- All pre-commit hooks passed (ruff, ruff-format, mypy, version-consistency)
- CI green on all prior commits
- Rust tests: 65/65 pass
- Python tests: 3836 pass, 1 skip (persistent_homology — flaky under load, passes independently 5/5)

## Compliance Violations Found (self-audit)
1. Session log was missing until now — FIXED
2. CI matrix memory stale (ruff/mypy versions) — needs memory update
3. SHARED_CONTEXT CI versions stale — user-owned file, flagged
4. AUDIT_INDEX.md not updated with session responses — TODO
5. No pytest --cov run before commits — violation acknowledged
6. SCPN version in SHARED_CONTEXT says v0.9.0, actual v0.9.5 — flagged
