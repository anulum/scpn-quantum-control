# Webmaster Context — scpn-quantum-control section of anulum.li

**Scope:** Claude (Arcane Sapience) owns the `scpn-quantum-control/` subdirectory of anulum.li.
**Last major build:** 2026-04-08 (by earlier Arcane session) — 10 HTML pages created.
**Last update:** 2026-04-10 (this session) — pending.

---

## Repository Layout

The website source lives OUTSIDE this repo (scpn-quantum-control), under the
GOTM collection:

- **Root:** `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/06_WEBMASTER/ANULUM-LI/`
- **WORKING/:** source of truth for edits
- **LIVE/:** deployment staging (matches FTP server state)
- **BACKUP/:** timestamped snapshots before risky operations
- **FILES/:** asset store

**scpn-quantum-control subdirectory:**
```
WORKING/scpn-quantum-control/
├── index.html          # hero, 8 stats, install, innovation, pipeline, explore grid
├── benchmarks.html     # Rust engine perf, hardware results, classical vs quantum
├── modules.html        # 17 subpackages, Rust module inventory, optional deps
├── algorithms.html     # VQE, Trotter, Lindblad, ZNE, PEC, DD, OTOC, Krylov, DLA
├── hardware-validation.html  # IBM Heron r2, 33/33 jobs, decoherence, multi-platform
├── applications.html   # FMO, power grid, Josephson, EEG, ITER, cross-ecosystem
├── notebooks.html      # 13 core tutorials + 21 examples + FIM investigation
├── pricing.html        # Community/Indie/Pro/Perpetual + Enterprise + Academic
├── changelog.html      # Version timeline
├── access.html         # Public PyPI, source access, academic citation
├── nav.js              # 10-page nav, footer, 20 flags, modal, theme toggle, hamburger
├── style.css           # Dark/light mode CSS variables
├── particles.js        # Neural network particle background (dark/light adaptive)
└── assets/             # Images (currently empty)
```

**Site-specific prefix (localStorage, JS globals):** `sqc-`
**Modal function:** `_sqcTranslateTo()`
**Theme storage:** `sqc-theme`

---

## Shared Architecture Pattern

All 4 product sites (sc-neurocore, scpn-quantum-control, scpn-phase-orchestrator,
scpn-control) use the same 10-page pattern. Conventions:

- **Landing page** (index.html): hero + 8 stats + install + innovation + pipeline + explore grid
- **Stats panel:** 8 numbers, `.stat-v` large number, `.stat-l` label
- **Card grid:** `.grid` with `.card` items (card-h + card-d)
- **Section:** `.sec` > `.sec-t` heading + content
- **Panel:** `.panel` for boxed content with headline + description
- **Code block:** `.code` with `.cm` comment + `.cmd` command lines
- **Button styles:** `.btn.btn-p` (primary) and `.btn.btn-g` (ghost)

---

## FTP Deployment

**Credentials location:** `agentic-shared/CREDENTIALS.md`, section `Webhouse.sk` → FTP
subsection for `ftp.anulum.li`. Read from vault at runtime, **NEVER copy
credentials into this repository or any committed file**.

**Non-secret deployment parameters:**
- Server: `ftp.anulum.li`
- Document root: `www_root_anulum_li/`
- scpn-quantum-control path: `www_root_anulum_li/scpn-quantum-control/`

**CRITICAL FTP settings — don't deviate:**

```bash
# Plain curl (RECOMMENDED — most reliable).
# Read USER and PASSWORD from the credentials vault at runtime,
# never hard-code them into any script or documentation committed to git.
curl -u "$FTP_USER:$FTP_PASS" -T localfile \
  ftp://ftp.anulum.li/www_root_anulum_li/scpn-quantum-control/remotefile

# lftp alternative (MUST disable SSL on data + use active mode):
lftp -c "
set ssl:verify-certificate no
set ftp:ssl-protect-data no
set ftp:passive-mode no
open ftp://\$FTP_USER:\$FTP_PASS@ftp.anulum.li
cd /www_root_anulum_li/scpn-quantum-control
put localfile
"
```

**NEVER use:**
- `lftp mirror --parallel=N` — corrupts files to 0 bytes on this server
- Parallel curl uploads — same issue
- SSL on data channel — hangs indefinitely

**Verification after upload:** always fetch the file back and diff:
```bash
curl -u "$FTP_USER:$FTP_PASS" \
  ftp://ftp.anulum.li/www_root_anulum_li/scpn-quantum-control/index.html \
  > /tmp/remote_index.html
diff /tmp/remote_index.html WORKING/scpn-quantum-control/index.html
```

---

## Deployment Workflow

1. **Edit:** Modify files in `WORKING/scpn-quantum-control/`
2. **Sync:** Copy `WORKING/ → LIVE/` when ready to deploy
3. **Backup:** Snapshot current LIVE before any risky change:
   ```
   cp -r LIVE/scpn-quantum-control/ BACKUP/scpn-quantum-control_YYYY-MM-DDTHHMMZ/
   ```
4. **Deploy:** Upload LIVE → FTP via sequential curl (see above)
5. **Verify:** Fetch + diff each updated file
6. **Commit reference:** Log deployment in session log + `WEBMASTER_CHANGELOG.md`

**NEVER skip the backup step for production changes.**

---

## Content Sync Targets (from github.com/anulum/scpn-quantum-control)

Key stats pulled from the repo; all outdated on the live site as of 2026-04-10:

| Metric | Site (stale) | Actual (2026-04-10) | Source |
|--------|-------------|---------------------|--------|
| Python modules | 165 | **201** | `find src -name '*.py' | wc -l` |
| Rust functions | 22 | **36** | `python -c 'import scpn_quantum_engine; ...'` |
| Tests | 2,813 | **4,828** | `pytest --collect-only` |
| Subpackages | 17 | **19** | directory count |
| IBM Jobs (old) | 33/33 | 33/33 (legacy) | unchanged |
| Rust speedup | 5,401× | + **44× hypergeometric** (new) | benchmarks |
| Notebooks | 47 | 47 | unchanged |

**v0.9.5 new features not yet on site:**
- GUESS symmetry decay ZNE (arXiv:2603.13060)
- DynQ topology-agnostic qubit mapper (arXiv:2601.19635)
- PMP/ICI pulse sequences (Liu et al. 2023)
- (α,β)-hypergeometric pulse shaping (arXiv:2504.08031)
- FFI boundary hardening (all 36 Rust exports → PyResult)
- Rust path for hypergeometric envelope (44× speedup)

**First publishable hardware result (2026-04-10):**
- DLA parity asymmetry confirmed on ibm_kingston (Heron r2)
- 342 circuits, 21 reps per point at n=4 key depths
- Mean asymmetry: **+10.8%** for depths ≥ 4, strongest **+17.48%** at depth 6
- 7/8 depth points positive → binomial p = 0.035
- Consistent with simulator prediction (4.5–9.6%, Šotek 2026)
- **This must be prominently featured on hardware-validation.html**

---

## TODO Pipeline for This Section

Priority P0 (this session):
- [ ] Sync LIVE → WORKING (clear divergence)
- [ ] Update `index.html` stats panel with current numbers
- [ ] Add DLA parity hardware result panel to `hardware-validation.html`
- [ ] Add v0.9.5 + Phase 1 IBM entry to `changelog.html`
- [ ] Commit all changes locally (no FTP yet)

Priority P1 (next session):
- [ ] Update `modules.html` with 19 subpackages + new Rust modules
- [ ] Update `benchmarks.html` with 44× hypergeometric speedup
- [ ] Add GUESS + DynQ + ICI + hypergeometric sections to `algorithms.html`
- [ ] Update `access.html` access tier messaging
- [ ] Deploy LIVE → FTP after review

Priority P2 (phase 2 experiments):
- [ ] Create a dedicated `phase2-results.html` after IBM 180-min runs
- [ ] Figures for DLA parity asymmetry (leakage vs depth plot)
- [ ] Scaling figure for n=4, 6, 8

---

## Risks & Safeguards

1. **LIVE/ ≠ WORKING/ divergence** — handover 2026-04-08T21:13 noted that
   hotfixes (nav.js, style.css) were applied directly to LIVE. Always diff
   before syncing WORKING ← LIVE direction.
2. **FTP parallel upload corruption** — ONLY sequential uploads.
3. **SSL data-channel hangs** — always disable with `set ftp:ssl-protect-data no`.
4. **No committed credentials** — credentials go in the shared vault only,
   never into repository files.
5. **Sitewide changes via one commit** — avoid partial deploys that leave
   the site in an inconsistent state across pages.

---

## Ownership Boundaries

**Claude (Arcane Sapience) owns:**
- `scpn-quantum-control/` directory on the website (all 10 pages + assets)
- Deployment workflow for this subdirectory
- Sync with `github.com/anulum/scpn-quantum-control` repo

**NOT owned by Claude in this scope:**
- Landing page (`index.html` at site root) — shared ownership with CEO sessions
- Other product sites (sc-neurocore, scpn-phase-orchestrator, scpn-control)
- Splash pages (anulum.ch, .eu, etc.)
- Legal pages, contact forms
- Anything outside `scpn-quantum-control/`
