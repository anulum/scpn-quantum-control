# Contributing

Copyright 1996-2026 Miroslav Sotek. All rights reserved.
Contact: protoscience@anulum.li

This repository accepts focused changes with tests, clear claim boundaries, and
no live hardware side effects in automated checks.

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

For Rust engine work:

```bash
cd scpn_quantum_engine
maturin develop --release
cd ..
```

## Before Opening A PR

Run the relevant focused tests, then the local preflight when the change is not
trivial:

```bash
python -m pytest tests/<focused_test_file>.py -q
python tools/preflight.py --no-coverage
```

For full local verification:

```bash
python tools/preflight.py
```

## Code Rules

- Format Python with Ruff and type public APIs.
- Format Rust with `cargo fmt`.
- Keep new dependencies justified and optional unless they are required by the
  core package.
- Preserve scientific claim boundaries. Simulator output, generated fixtures,
  and planning metadata are not hardware evidence.
- Do not contact live quantum providers from tests or CI unless a maintainer has
  explicitly approved the run.
- Keep secrets, raw credentials, local logs, and private planning artefacts out
  of tracked files.

## Tests

- Add tests with the behaviour change.
- Prefer module-specific tests over broad bucket tests.
- Cover the happy path, at least one edge case, and the relevant failure path.
- For numerical code, assert invariants such as Hermiticity, finite values,
  shape contracts, probability normalisation, or documented error bounds.
- For hardware-facing code, use simulator or mocked provider boundaries by
  default.

## Commit Messages

Use conventional subjects:

```text
feat(scope): short description
fix(scope): short description
docs(scope): short description
```

Every commit must include the repository authorship line enforced by
`tools/check_commit_trailers.py`:

```text
Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
```

## Pull Requests

- Keep the PR scoped to one logical change.
- State what changed, how it was tested, and any remaining limitations.
- Add or update docs when behaviour, public APIs, workflows, or claim boundaries
  change.
- Do not include generated build output unless the repository already tracks
  that exact artefact class.

## Security

Report vulnerabilities through the process in `SECURITY.md`. Do not open a
public issue for secrets, credentials, or exploitable security defects.

## Licence

Contributions are licensed under the GNU Affero General Public License v3.0 or
later. Commercial licensing is available via protoscience@anulum.li.
