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

Five pieces are not in the base lock and have to be provisioned:

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

# the shell linter, in the project environment as the CI job installs it
python -m pip install --no-deps --require-hashes -r requirements-ci-shell-lint.txt

# the licence-information linter; its shared dependencies follow the base lock
python -m pip install --require-hashes -r requirements-ci-licence-lint.txt
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

A third axis is the repository against itself. A quality tool that runs both as
a pre-commit hook and from the hash-locked CI requirements has two declarations
of its version, and neither of the checks above compares them: the parity report
measures this machine, and the contract gate measures a job's own install. When
the two declarations drift, a locally clean commit can fail CI and the agreement
between them is luck. That comparison is a gate, and runs in the hook set, the
local preflight and static analysis:

```bash
python tools/check_toolchain_pin_alignment.py
```

It collects every version this repository states for a tool — pre-commit hook
revisions, `requirements*.txt` pins, `pyproject.toml` dependency ranges,
workflow action inputs, pinned `cargo install` commands and the setup commands
in this file — and refuses any two that disagree. It also refuses a pinned
version the project range forbids, a mapped hook with no requirements pin to
compare, and a remote hook repository classified neither as a mirrored Python
distribution nor as a reasoned non-Python hook. Adding a hook therefore means
classifying it. Exit status 1 reports findings, 2 means the declarations could
not be read, and an unreadable configuration is an error rather than a pass.

`--list` prints the evidence behind the verdict, which is the fastest way to
see every place a tool is declared before changing one of them:

```bash
python tools/check_toolchain_pin_alignment.py --list
```

Declarations without a version are deliberately out of scope. A `language:
system` hook runs whatever the environment installed, so it cannot split from a
pin. Whether the agreed version is also the newest compatible release is a
separate, network-bound question for the periodic dependency census, not for a
commit hook.

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
  boundary. Standard URI authorities and expanded XML namespace names are
  technical references; their paths and local names still obey the naming rule.
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

## Test Documentation Ceiling

Test files are measured separately with
`python tools/audit_test_documentation_ceiling.py`, using the same isolated
NumPy documentation profile over `tests` and `oscillatools/tests`. The recorded
per-file counts in `tools/test_documentation_ceiling.json` are a ceiling that
only falls:

- a test file that is not in the ceiling must have no documentation finding;
- a recorded file may not gain findings;
- when a file's findings fall, or a file becomes clean or is removed, lower the
  ceiling in the same change with `--lower` (it refuses while any file grew);
- with `--changed-against <revision>` every test file that differs from that
  revision must have no finding at all. CI compares with the previous head and
  the pre-push check compares with `origin/main`, so a test file that is
  touched is documented completely in the same change.

The counts depend on the Ruff release. The ceiling records the release it was
measured with, and the gate refuses to compare across releases; after a Ruff
upgrade record the new measurement with `--rebaseline`, which keeps the previous
totals in the file's history. The gate checks that documentation is present and
well formed; it does not judge whether a description is accurate.

## Web Source Ceiling

The TypeScript and stylesheet sources of `studio-web` are linted and
format-checked with Biome, pinned as an exact development dependency in
`studio-web/package.json` and configured in `studio-web/biome.jsonc`. The
sources were never formatted by a tool, so the gate does not demand one mass
rewrite: `python tools/audit_web_source_ceiling.py` compares every tracked
`.ts`, `.tsx` and `.css` file with `tools/web_source_ceiling.json`, which
records the lint finding count per file and the files that are not formatted.
The ceiling only falls:

- a source that is not in the ceiling must have no lint finding and be
  formatted;
- a recorded file may not gain findings, and a formatted file may not lose its
  layout;
- when a file's findings fall, or a file becomes clean, formatted or is
  removed, lower the ceiling in the same change with `--lower` (it refuses
  while any debt grew);
- with `--changed-against <revision>` every web source that differs from that
  revision must have no finding and be formatted. CI compares with the
  previous head and the pre-push check compares with `origin/main`, so a web
  source that is touched is cleaned and formatted completely in the same
  change: `pnpm exec biome format --write <file>` and
  `pnpm exec biome lint <file>` in `studio-web`.

The gate uses the Biome executable that `pnpm install --frozen-lockfile` puts
into the workspace and refuses any release other than the pinned one. After a
Biome upgrade record the new measurement with `--rebaseline`, which keeps the
previous totals in the file's history. A parse error is a failure of the gate,
not a counted finding. The gate reports what Biome reports under the recorded
configuration; it does not judge accessibility or behaviour beyond those rules.

## Git Location Isolation In Tests

Git exports `GIT_DIR`, `GIT_INDEX_FILE` and related variables to its hooks, and
a caller may export them to address a repository from outside its work tree. A
test that builds a temporary repository would then run `git init` and
`git commit` against the inherited repository instead of its own. The shared
configuration in `tests/conftest.py` therefore drops every variable that
`git rev-parse --local-env-vars` lists, once, before any test module is
collected, and the session header names what was dropped.

- A test that needs such a variable sets it itself, for its own child process.
- A test that runs Git belongs under `tests/`, where this configuration
  applies; a run with `--noconftest` has no isolation.
- Do not export these variables around a test run or any Git command that
  writes. `python -m pytest tests/test_git_location_isolation.py` proves the
  isolation with real Git and a nested session.

## Source Surface Inventory

Lint, type and documentation gates name the paths they read, so a file in a new
language or a new top-level directory is read by none of them.
`python tools/audit_source_surface_inventory.py` classifies every tracked file
by kind (its suffix, or its name when it has none) and top-level directory and
compares the result with `tools/source_surface_policy.json`:

- a file of an unknown kind, or a known source kind in a directory without a
  row, fails the gate. Add the row together with the gate that reads it;
- a *gated* row names a workflow, a job and a command of that job. The gate
  checks that the command is still written in one of the job's `run` steps;
- an *open* row records what is not enforced yet. With
  `--changed-against <revision>` no row may be open that was not open at that
  revision, so existing debt stays listed and new debt is refused. CI compares
  with the previous head and the pre-push check with `origin/main`;
- an *evidence* row is allowed under `data` only, for sources recorded with
  the results they produced;
- kinds that carry no source (documents, data, configuration) are listed once
  under `outside`. A row or an outside kind that no tracked file uses must be
  removed.

The gate proves that a recorded command exists. It does not prove that the job
passes or that the command reads every file of the row.

## Shell Script Lint And Format

`python tools/audit_shell_scripts.py` runs ShellCheck over every tracked shell
script and fails on any finding, down to style level, and fails on any script
that the `shfmt` formatter would change. A script is a tracked `.sh` or `.bash`
file, or a tracked file without a suffix whose first line names `sh`, `bash`,
`dash` or `ksh` as the interpreter, such as `.githooks/pre-push`.

Both tools are pinned in `requirements-ci-shell-lint.txt`, and the gate refuses
to run with other versions. Install them into the project environment with
`python -m pip install --no-deps --require-hashes -r requirements-ci-shell-lint.txt`;
the gate uses the binaries beside the interpreter that runs it (for ShellCheck
it falls back to the one on `PATH`). To change a pin, edit
`requirements-ci-shell-lint.in` and regenerate the lock with the command
recorded at the top of the `.txt` file; keep the pins at the newest releases,
like every other pin.

Format a script with `shfmt -w <script>`. The formatter reads `.editorconfig`,
so the layout is the repository's: four spaces. It changes layout only; when a
script that cannot be executed here is reformatted, compare `shfmt -mn` of the
old and the new text to show that nothing else changed.

Silence a ShellCheck finding only where it is wrong, with a `# shellcheck`
directive on the line above and a comment that says why. The gate runs in CI
and in the pre-push check, with the same pinned tools.

## Licence Information

`python -m reuse lint` requires copyright and licence information for every
file, tracked or not yet tracked, and fails on a licence expression it cannot
parse. The linter is pinned in `requirements-ci-licence-lint.txt`. The gate runs
in CI and in the pre-push check.

- A source file carries the header described above.
- A file that cannot carry a comment (JSON, binary, generated data) gets a
  `<name>.license` sidecar, or its directory gets an annotation in `REUSE.toml`.
- A file that *contains* a licence tag as data, such as a generator that emits
  a header, wraps those lines in `REUSE-IgnoreStart` / `REUSE-IgnoreEnd`
  comments and says why.
- Tool output that is not source belongs in `.gitignore`.

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
