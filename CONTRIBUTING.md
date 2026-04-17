# Contributing to scpn-quantum-control

© 1996–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li

Thank you for your interest in scpn-quantum-control. Contributions are welcome under the following guidelines.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork and create a feature branch:
   ```bash
   git clone https://github.com/<your-user>/scpn-quantum-control.git
   cd scpn-quantum-control
   git checkout -b feature/your-feature
   ```
3. **Install dev dependencies and git hooks:**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
4. **Build** the Rust engine (optional, for Rust work):
   ```bash
   cd scpn_quantum_engine && maturin develop --release && cd ..
   ```
5. **Run tests** to verify your setup:
   ```bash
   pytest tests/ -v --ignore=tests/test_hardware_runner.py
   ```

## Preflight Gate

Every push is guarded by pre-commit hooks that run the same checks as CI:

| Gate | What it checks |
|------|----------------|
| **ruff check** | Code quality and import hygiene |
| **ruff format** | Python formatting (`src/` and `tests/`) |
| **mypy** | Type checking (public API boundaries) |
| **version-sync** | Version consistency across pyproject.toml, CITATION.cff, .zenodo.json |
| **preflight** | Full CI mirror (lint + version-sync + tests) |

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest tests/ -v --ignore=tests/test_hardware_runner.py
```

The pre-push hook runs the full preflight automatically before every `git push`.

## Development Guidelines

### Code Style

- **Python**: `ruff format` + `ruff check` (both enforced in CI). Use type hints on public APIs only.
- **Rust**: `cargo fmt` before committing.
- **SPDX header**: Every `.py`, `.rs`, `.yml` file must start with `# SPDX-License-Identifier: AGPL-3.0-or-later`.

### Quantum-Specific Rules

- Every quantum circuit must transpile on `AerSimulator` without error
- Statistical tests compare quantum vs classical outputs (not exact match)
- Use `n_shots >= 1000` for any statistical assertion
- Never hardcode backend names; accept backend as parameter
- Hamiltonian construction must preserve Hermiticity
- All angle computations use radians (never degrees)

### Testing

- All new modules must have pytest coverage in `tests/test_<module>.py`
- Each test file needs: one physics verification, one circuit validity check, one edge case
- Coverage gate: 95% minimum, 100% target
- Hardware tests go in `tests/test_hardware_runner.py` (skipped by default)

### Commit Messages

Follow conventional commit format:
```
feat(scope): short description
fix(scope): short description
docs(scope): short description
```

Examples:
```
feat(analysis): add Krylov complexity probe for sync transition
fix(bridge): correct XXZ Hamiltonian sign convention
docs(tutorials): add Floquet time crystal tutorial
```

Every commit message must include a `Co-Authored-By:` trailer (enforced
by `tools/check_commit_trailers.py` as a `commit-msg` pre-commit hook):

```
feat(analysis): add Krylov complexity probe for sync transition

Short explanation of *why* this change.

Co-Authored-By: Arcane Sapience <protoscience@anulum.li>
```

The subject line must not contain these words: `elite`, `Elite`,
`SUPERIOR`, `Superior`, `ETALON`, `Etalon`, `comprehensive`, `robust`,
`leveraging`, `world-class`, `best-in-class`. The hook enforces the
subject-line check; the body may mention the words when describing
their removal from elsewhere in the repo.

### Merging Dependabot pull requests

`gh pr merge --squash --delete-branch` on its own keeps Dependabot's
default commit message, which omits the `Co-Authored-By` trailer.
Use the following form so the resulting squash commit still passes
the trailer check:

```bash
gh pr merge <N> --squash --delete-branch \
  --body "$(gh pr view <N> --json body -q .body)

Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

The auditor in `tools/check_commit_trailers.py` runs weekly in CI and
will flag any future Dependabot merge that omits the trailer. Seven
merges made on 2026-04-17 before the hook existed are in the
historical-exempt list in that file.

## What to Contribute

**High-value contributions:**
- New synchronization probes (analysis modules)
- VQE ansatz strategies for the Kuramoto-XY Hamiltonian
- Error mitigation techniques beyond ZNE/PEC
- Jupyter notebook tutorials
- Hardware benchmarks on new IBM/IonQ/Rigetti backends
- Bug reports with reproducible test cases

**Please discuss first** (open an issue) before:
- Changing the public Python API
- Modifying the bridge layer contracts
- Adding new package dependencies

## Submitting a Pull Request

1. Run the full preflight — all gates must pass
2. Add a changelog entry if the change is user-visible
3. Ensure SPDX headers are present on new files
4. Open a PR against `main` with a clear description
5. Reference any related issues

## Issue Triage

Labels live in two taxonomies:

- **Priority** — `P0` (blocks a release), `P1` (this week),
  `P2` (this month), `P3` (this quarter). Every non-trivial issue
  gets exactly one. Missing label defaults to `P3`.
- **Area / kind** — `bug`, `enhancement`, `documentation`,
  `question`, `hardware`, `security`, `rust`, `science`, `ci`,
  `performance`, `dependencies`, `python`, `github_actions`. Apply
  whichever match; no cap.

Two workflow labels:

- `triage` — issue has not been seen by a maintainer yet. Removed
  when the first priority label goes on.
- `blocked` — issue cannot progress until something external
  resolves (upstream fix, IBM Quantum queue, CEO decision).
  Maintainers add a comment pointing at the blocker.

Service-level commitments, effective 2026-04-17:

| Event | SLA |
|-------|-----|
| First maintainer response on a new issue | 5 working days |
| First response on a `security`-labelled issue | 48 hours (see `SECURITY.md`) |
| First response on a `P0` issue | 24 hours |
| Weekly triage sweep of `triage`-labelled issues | every Monday |

Stale policy: non-`security`, non-`P0` issues with no activity for
90 days get a `status:stale` comment from `stale.yml`; auto-close
after another 14 days unless a maintainer re-labels.

## CI Pipeline

| Job | What it checks |
|-----|----------------|
| `lint` | ruff check + ruff format |
| `test` | pytest on Python 3.9–3.13 |
| `type-check` | mypy |
| `wheel-check` | Build + install + smoke test |
| `security` | bandit + pip-audit |
| `docker` | Full test suite in Docker container |
| `hardware-smoke` | Simulator-only hardware tests |
| `integration-optional` | Cross-repo bridge tests (sc-neurocore) |

## Licence

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE). For commercial licensing enquiries, contact protoscience@anulum.li.
