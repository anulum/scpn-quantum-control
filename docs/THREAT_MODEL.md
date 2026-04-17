# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Threat Model

# Threat Model

This document names the things we are trying to protect
(`assets`), the kinds of actors we plan against (`adversaries`),
where they can try to interact (`attack surface`), and the
mitigations currently in the repository. It is a companion to
`SECURITY.md` — that file tells researchers how to report a
vulnerability; this file explains what we consider a
vulnerability.

Version 1 — 2026-04-17. Review cycle: quarterly, or sooner if any
of the three crypto / QKD / hardware surfaces change in a
non-trivial way.

## Assets

1. **Scientific integrity of published numbers** — the hardware
   result JSON files in `data/`, the statistics quoted in
   `CHANGELOG.md`, `docs/results.md`, and `docs/preprint.md`. An
   attacker who can flip a depth-point or a sign silently
   invalidates a publication.
2. **Research credentials** — the IBM Quantum token, Zenodo token,
   GoatCounter key, PyPI Trusted Publisher identity. Exfiltration
   would enable forged submissions under the ANULUM identity.
3. **End-user cryptographic output** — the output of every function
   in `crypto/`. Includes BB84 session keys, Bell-test verification
   bits, QKD-derived secrets. Downstream code that trusts these
   bytes as secret material must hold against a passive network
   adversary.
4. **Host resources of anyone who imports the library** — CPU,
   memory, disk, QPU quota. A pathological input to
   `analysis/koopman.py` or `bridge/knm_hamiltonian.py` should not
   exhaust the caller.
5. **Downstream consumer trust** — the chain that takes a git
   commit, through a wheel, through PyPI, to a user's import. A
   substituted wheel that passes `pip install` invalidates every
   user we have.

## Adversaries we plan against

| # | Adversary | Capability | Incentive |
|---|-----------|-----------|-----------|
| A1 | Passive public-internet observer | Reads traffic between a user and PyPI / IBM Quantum / GitHub Pages / Zenodo | Eavesdrop on QKD sessions, infer research directions from download patterns |
| A2 | Malicious PyPI mirror / cache | Serves an altered sdist or wheel under our name | Supply-chain code execution on researcher workstations |
| A3 | Compromised Dependabot / GitHub Actions third-party action | Runs our CI with attacker-controlled code | Exfiltrate `PYPI_API_TOKEN`, `QISKIT_IBM_TOKEN`, `ZENODO_TOKEN`, or alter the release artefact |
| A4 | Forged hardware-result author | Submits fabricated `data/<experiment>/*.json` via a PR | Inject false scientific claims we cite as evidence |
| A5 | Hostile user of the library | Calls public APIs with crafted input | Denial of service on the caller; code execution via deserialisation if we added any |
| A6 | Insider with commit access | Pushes a commit that silently tightens `feedback_no_simplifications` or removes a statistical caveat | Long-range scientific fraud |

We explicitly do **not** plan against:

- Nation-state adversaries with implants on the maintainer's
  laptop. Mitigation there is out-of-scope for this repo.
- Adversaries with full write access to PyPI itself. Sigstore
  Trusted Publishing (`.github/workflows/publish.yml`) raises the
  bar but does not defeat PyPI compromise.
- Adversaries with physical access to IBM Quantum data centres.
  Any mitigation belongs on IBM's side.

## Attack surface

### S1 — Crypto subpackage (`crypto/`)

- `bb84.py`, `entanglement_qkd.py`, `bell_test.py`, `qkd_parameter.py`,
  `topology_qkd.py`, `key_hierarchy.py`.
- Output is treated as secret. Every function must be deterministic
  in its randomness (i.e. accept a `numpy.random.Generator` rather
  than using `numpy.random.seed`).
- No function here imports from `os.urandom`, `secrets`, or reads
  `/dev/random` today. Downstream callers who need CSPRNG-grade
  bits beyond toy demos must pair the QKD output with their own
  CSPRNG stretch.

### S2 — Hardware runner (`hardware/runner.py`)

- Reads `QISKIT_IBM_TOKEN` / `~/.qiskit/qiskit-ibm.json` — never
  logged, never written to a result JSON. Audit: grep
  `QISKIT_IBM_TOKEN` / `api_key` / `token` in `src/` returns only
  read paths.
- Writes result JSONs with `save_result`; includes a
  `provenance` block (see `hardware/provenance.py`) so an outsider
  can detect silent rewrites. CI gate: `tests/test_phase1_dla_parity_reproduces.py`
  fails if any published statistic drifts.
- Circuit-depth bound (`max_depth`), shot-count bound (100 000),
  RNG isolation per module. Documented in `SECURITY.md`.

### S3 — `analysis/koopman.py` and friends

- Past gap: unbounded `eigvals` on a caller-supplied n²×n² matrix.
  Fixed in commit `c7d4ccd` by `MAX_OSCILLATORS_DEFAULT = 32` and
  an explicit opt-in argument. 13 tests in `tests/test_koopman.py`
  exercise every guard branch.
- Other modules with ingesting-user-input surface should receive
  similar validators. Open audit item B8.

### S4 — Supply chain

- GitHub Actions SHAs are pinned, not tag-pinned, in every
  workflow. Dependabot monitors actions weekly, pip weekly, cargo
  weekly.
- Pre-commit hooks include `gitleaks` and a custom
  `tools/check_secrets.py` vault-pattern scanner.
  `tools/check_commit_trailers.py` enforces `Co-Authored-By` and
  an anti-slop word-list on commit subjects.
- Releases carry CycloneDX SBOMs (`sbom.yml`) and Sigstore
  signatures (`publish.yml`). Downstream verifies without a PyPI
  round-trip.

### S5 — Dataset / result integrity

- `data/<experiment>/` files are tracked in git; any change
  shows up in `git diff`. The provenance block added in commit
  `819aded` embeds the producing commit SHA in every new result
  file, so a silent overwrite becomes detectable as a `git`
  hash that does not match the claimed campaign.

### S6 — `docs/internal/` and `.coordination/`

- Gitignored. Contain drafts, session logs, agent-generated
  audits. No secret should live there, but private plans do.
- Repo gitleaks hook scans both anyway in `--all` mode.

## Mitigations per adversary

| Adversary | Primary mitigation | Residual risk |
|-----------|-------------------|---------------|
| A1 Passive observer | TLS everywhere (PyPI, IBM, GitHub, Zenodo all HTTPS-only). QKD output is intended to be the shared secret for a higher-level protocol, not the full protocol. | None beyond the TLS CA trust assumption. |
| A2 Malicious PyPI mirror | Sigstore-signed wheels + Trusted Publisher attestations. Reader verifies with `sigstore verify identity`. | Attacker who compromises the `anulum` GitHub org OIDC identity can mint a valid signature; not defeated. |
| A3 Compromised GH Action | Every third-party action pinned by full commit SHA, not by tag. OpenSSF Scorecard `Pinned-Dependencies` gate. Dependabot flags new SHAs weekly. | A compromise of the upstream repo that the SHA was taken from, before we updated, is still exploitable until detected. |
| A4 Forged hardware result | PRs with changes under `data/` require maintainer review. Reproducer test (`tests/test_phase1_dla_parity_reproduces.py`) detects any drift in published numbers. | A forger who edits both the JSON and the claimed statistics consistently would pass the reproducer. Human review is the last line. |
| A5 Hostile user | Input validation (koopman, planned for the rest). Depth + shot caps on hardware runner. No pickle deserialisation (per `SECURITY.md`). | Novel crafted inputs may still find unchecked paths; audit item B8 tracks this. |
| A6 Insider with commit access | `feedback_no_simplifications` is a hard rule for agents. `DEPRECATIONS.md` forbids silent API removals. `docs/falsification.md` pins each scientific claim to an observable refutation criterion. CI runs `mypy` + `ruff` + tests, but cannot detect a coherent lie. | Real threat. Maintainer is currently a single person; a second reviewer for `CHANGELOG.md` + `docs/preprint.md` edits is future governance work. |

## Known gaps (tracked in the audit)

- **B6** — Criterion-level Rust benchmarks with their own
  regression gate. Today we have the Python-side regression gate
  in `tests/test_perf_regression.py`; Rust side is only timed
  indirectly.
- **B7** — Mutation testing baseline. `mutmut` has never been run;
  we do not know the test-quality score.
- **B8** — Fuzz tests on the non-koopman input boundaries.
- **C4** — Formal export-control (ECCN 5D002) assessment for the
  `crypto/` subpackage.
- **C7** — Cross-validation against Dynamiqs / QuTiP / PennyLane
  on the same problem to detect silent numerical divergences.

Each of these degrades the confidence of one or more mitigations
above and is logged as an open item in
`docs/internal/audit_2026-04-17T0800_claude_gap_audit.md`.
