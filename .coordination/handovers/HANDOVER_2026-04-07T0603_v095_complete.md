# Handover: scpn-quantum-control v0.9.5

**From:** Arcane Sapience (Claude)
**Date:** 2026-04-07T0603
**Commit:** bafc36c (tagged v0.9.5)
**Repo:** anulum/scpn-quantum-control
**Branch:** main (in sync with origin)

---

## State

- CI green, 0 failed runs, 0 open PRs, 0 security alerts
- 4445+ Python tests, 65 Rust tests, 95.85% coverage
- Backup at `/media/anulum/724AA8E84AA8AA75/Backup/scpn-quantum-control_v0.9.5_bafc36c_2026-04-07.tar.gz`

## What Was Done (this session)

1. **3 new modules** — MS-QEC (concatenated QEC), FEP (variational free energy), Ψ-field (U(1) lattice gauge)
2. **Rust engine refactored** — 1 god file → 17 modules, 3 new Rust paths
3. **K_nm validation** — 5/5 physical systems measured (all ρ>0.88)
4. **Perf optimised** — OTOC 4.4× faster, Pauli 2–10× faster (Gemini proposal audited, cherry-picked)
5. **Security hardened** — all CI deps pinned, Dockerfile SHA256, 5 Scorecard alerts resolved
6. **Dep bumps** — ruff 0.15.9, mypy 1.20.0, hypothesis 6.151.11
7. **66 dead CI runs purged**, all PRs resolved, CHANGELOG updated, v0.9.5 tagged

## What Needs Doing Next

### Ready to implement (TODO at `.coordination/TODO_2026-04-07_strategic_tweaks.md`)
1. **FFI Hardening** — replace silent .clamp()/.max() in 5 Rust files with explicit PyValueError. ~175 lines. HIGH priority.
2. **Mitigation Factory** — symmetry decay ZNE (arXiv:2603.13060) + unified mitigation API. Builds on existing 779 lines.
3. **DynQ Virtualisation** — community detection for qubit placement (arXiv:2601.19635). New module.
4. **PMP/ICI Pulses** — Pontryagin optimal control + ICI sequence for dissipative transfer. New pulse-level capability.
5. **(α,β)-Hypergeometric Pulses** — hardware-aware adiabatic tuning. Depends on #4.

### Known issues
- `test_quantum_persistent_homology::test_returns_both` — flaky under full-suite memory pressure (passes 5/5 in isolation)
- OpenSSF Code-Review alert #47 — by design (single-developer, pre-commit hooks)
- AUDIT_INDEX.md not updated with this session's responses (Tier 2, deferred)
- ruff/mypy version bumps applied only to this repo — other repos still on old versions

### Files changed but not tracked in git
- `.coordination/` — session logs, TODOs, handovers (gitignored)
- `.venv-linux/` — Linux virtualenv (gitignored)

## Key Technical Notes for Next Agent

- **Rust build:** `cd scpn_quantum_engine && VIRTUAL_ENV=../.venv-linux maturin develop --release`
- **Cargo.toml** is inside `scpn_quantum_engine/`, not repo root
- **Python venv:** `.venv-linux/` (Linux), `.venv/` (Windows, stale)
- **Pre-commit hooks:** ruff, ruff-format, mypy, version-consistency — all must pass
- **God file rule:** >300 lines requires split. >50 lines per function requires extraction.
- **Gemini autoresearch:** outputs land as PDFs in `docs/internal/` or as uncommitted working tree changes. Always benchmark before accepting — rayon overhead is catastrophic at small n.

## SHARED_CONTEXT Updates Made
- ruff: 0.15.6 → 0.15.9
- mypy: >=1.19.1 → >=1.20.0
- scpn-quantum-control version: v0.9.0 → v0.9.5

## Memory Updates Made
- CI Unified Matrix: ruff 0.15.9, mypy 1.20.0, hypothesis 6.151.11
