# Contributing

Copyright 1996-2026 Miroslav Sotek. All rights reserved.
Contact: protoscience@anulum.li

This repository accepts focused changes with tests, clear claim boundaries, and
no live hardware side effects in automated checks.

## Setup

Use Python 3.11 or newer.

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

## Keeping The Local Environment Aligned

A local run only means something if it ran against the environment CI runs
against, and the pins move: locks get regenerated, action versions get bumped,
a runner image ships a newer default. Check before trusting a local result, and
again whenever a lock or a workflow pin changes:

```bash
python tools/audit_local_environment_parity.py
```

It reads the repository as the source of truth — the workflows for the
interpreter and tool pins, `requirements-ci-py312-linux.txt` for distributions,
`rust-toolchain.toml` for the toolchain — and names every axis where this
machine says something different. It is a report, not a gate: a workstation
legitimately differs from a runner on the optional tiers.

Three pieces are not in the base lock and have to be provisioned:

```bash
# the native engine, with the arguments CI builds it with
maturin build --release --features extension-module --out dist \
  -m scpn_quantum_engine/Cargo.toml
python -m pip install --force-reinstall --no-deps dist/scpn_quantum_engine-*.whl

# the pnpm the Studio jobs pin, project-scoped rather than global
corepack prepare pnpm@11.9.0

# the Julia tier, in its own environment as CI keeps it
python -m venv .venv-julia
.venv-julia/bin/python -m pip install --require-hashes -r requirements-ci-py312-linux.txt
.venv-julia/bin/python -m pip install --no-deps --require-hashes -r requirements-ci-julia-tier.txt
```

Two axes cannot be closed on a workstation and the report says so rather than
passing them silently: `actions/setup-python` ships a different build of the
same CPython version, and `juliapkg` reuses a system Julia when one is on PATH
while a clean runner downloads its own.

The other direction — whether a CI job installs what it runs — is a gate rather
than a report, and runs in static analysis and in the local preflight:

```bash
python tools/audit_workflow_environment_contracts.py
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
- Name modules, symbols, APIs, serialized fields, evidence artefacts, workflow
  jobs, and public documentation by their domain purpose. Internal work-item or
  roadmap identifiers belong only in private traceability records, not product
  names. Run `python tools/audit_descriptive_production_naming.py` to verify the
  boundary.
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

The whole test tree has measured legacy typing debt. CI and local preflight
therefore enforce an additive strict-mypy cohort instead of pretending all test
files are already strict:

```bash
python tools/audit_test_typing_policy.py
```

The ordered cohort schedule and exact enforced paths live in
`tools/test_typing_policy.json`. Add a test file only in a source-owned slice
that also passes focused pytest, Ruff check/format, and strict mypy. Keep
intentional invalid-input calls and use a narrow error-code suppression only
where the type system cannot express the negative case.

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

That line is the only attribution a commit carries. A commit message is a
public surface, so the same checker rejects any additional trailer that
attributes the work to a tool vendor or model identity — for example a
`Co-Authored-By:` line naming an assistant, or a `<Vendor>-Session:` link.
Several assistant tools append such trailers automatically; strip them before
committing. The check runs at the `commit-msg` stage, so it refuses the message
rather than leaving the commit to be corrected afterwards.

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
