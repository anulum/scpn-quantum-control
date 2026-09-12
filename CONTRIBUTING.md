# Contributing

Copyright 1996-2026 Miroslav Sotek. All rights reserved.
Contact: protoscience@anulum.li

This repository accepts focused changes with tests, clear claim boundaries, and
no live hardware side effects in automated checks.

## Local reference maintenance

The main-only reference hook permits `git pack-refs` to remove a loose copy
only when an identical packed ref remains and no `packed-refs.lock` exists.
This is Git files-backend maintenance, not permission to delete `main`.
Missing, ambiguous or locked storage fails closed; pre-push deletion rules
are unchanged. Do not bypass the hook to resolve a maintenance failure.

The dedicated `tests/test_enforce_main_branch_policy.py` suite exercises real
packing and refused main deletion in loose, packed, mixed and stale-packed
states, both with and without an expected old OID. Recheck these behaviours
when changing Git versions or ref backends. Storage inference does not defend
against an operator directly editing Git's own files.

The project hook installer never replaces an existing hook or symlink.
Reinstalling its identical executable is a no-op; an older, modified or
nonexecutable hook requires explicit review of the existing chain. This also
protects shared GOTM wrappers: do not remove them to make installation pass.
New hooks are published as complete executables through an atomic, exclusive
hard link; unsupported storage or competing installation fails without an
overwrite fallback. Git's configured hook directory is respected.
If neither the current nor primary checkout contains the policy script, the
installed shim refuses the transaction. Restore the actual policy owner rather
than disabling the hook. Installer regressions live in
`tests/test_main_branch_hook_installation.py`.

## Container memory admission

The Docker workflow builds the repository's test image, then runs
`tools/check_container_memory_budget.py --limit-bytes <bytes>` inside separate
1 GiB and 2 GiB memory-limited containers before the image's full test command.
The combined memory/swap limit equals the memory limit, allowing no extra swap;
each check has one CPU and a 60-second deadline. The checker imports the real
package, requires positive finite cgroup
headroom, bounds the default dense budget, admits a two-byte vector and refuses
an above-limit estimate. It does not allocate either estimated buffer or test
the kernel OOM killer. JSON output records byte counts in the job log.

An environment budget override or unavailable cgroup allowance fails the check;
it never treats an unrestricted host as container evidence. Cgroup readings are
snapshots, not reservations or a guarantee against concurrent memory pressure.
`tests/test_check_container_memory_budget.py` exercises injected controller
files and workflow wiring locally. Only an actual successful Docker job at the
published commit proves the hosted container behaviour.

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

This is a static check of recognised literal commands, not proof that jobs ran.
It rejects missing or empty workflow inventories, malformed job/step shapes and
unreadable YAML, and scans both `.yml` and `.yaml`. Requirements and project
installs must precede their consumers across steps and literal newline, `&&`
or semicolon boundaries. Reusable workflow references are admitted without
executing actions or expanding remote workflows. Shell branches, dynamic
expressions, step conditions and actual package imports still need runner
validation. Exit status 1 reports findings; 0 means only that these static
checks found none.

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

## Documentation Scope Gate

The closed Python documentation scopes are checked with
`python tools/audit_documentation_scopes.py`. Every configured scope must exist
as a directory, and the combined inventory must contain Python source. Each
scope's Python file count is reported, including zero for non-Python areas.
A missing scope is an input error, not evidence of zero documentation debt.
Exit status 0 means all listed
scopes were inspected without unexempted findings, 1 reports documentation debt,
and 2 reports an incomplete scan. This gate does not certify test documentation,
other languages or the semantic accuracy of docstrings.

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

New commits permit exactly one authorship line and no `Co-Authored-By:` trailer,
including the former project coauthor trailer. CI checks introduced revisions
with `python tools/check_commit_trailers.py --strict --range BASE..HEAD` using
the same rules as the hook. Strict mode requires a nonempty explicit range,
does not accept historical exemptions and cannot be bypassed by backdating a
commit. Scheduled checks retain the historical audit and additionally apply
strict policy after the fixed published review boundary
`a1760207032178e3b926c2dd25a5d83367daf76b`. This does not certify or waive older
attribution debt, and no published history needs rewriting.

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
