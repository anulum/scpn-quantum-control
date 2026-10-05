# Repository boundary

**Status:** Active. Recorded 2026-10-05.

This record states what the `scpn-quantum-control` repository contains, what
it does not contain, how the boundary is checked, and what the record does not
claim. The reasoning behind the module structure inside the package is in
[Architecture Decisions](../ARCHITECTURE_DECISIONS.md).

## Context

The repository grew from one Python package into a workbench: the package,
two distributions with their own package metadata, a web portal, the tools and
tests that gate all of them, and the committed evidence that the
documentation cites. A plan exists to split the package into focused
repositories. Until a split is reviewed and released, contributors and readers
need one answer to "does this belong here, and who owns it?".

## Decision

Everything stays in this one repository until a split is released. Ownership is
recorded as data, not as convention:

- `data/split_preparation/split_domain_map.json` assigns every unit of the
  package to a domain and every domain to one planned destination. A unit
  whose files belong to different destinations lists each file. Each
  classification that needed a decision carries the reason and the decision.
- Every tracked path outside the package has an owner by path rule in the same
  file. Tools, workflows, data, documentation, notebooks and examples belong to
  the workbench itself.
- The dependency direction between destinations is a declared graph. Imports
  that cross it against the direction are listed, one row per exception, in
  `data/split_preparation/boundary_baseline.json`. The list can shrink; a new
  or widened exception fails the check.

## What the repository contains

| Path | Content | Owner in the map |
|---|---|---|
| `src/scpn_quantum_control/` | The Python package | By unit, to the planned package destinations |
| `src/scpn/` | Compatibility import path for the differentiation interface | Workbench |
| `oscillatools/` | Coupled-phase-oscillator toolkit with its own package metadata | Sibling distribution |
| `scpn_quantum_engine/`, `src/scpn_quantum_engine.pyi` | Rust engine and its type stubs | Sibling distribution |
| `studio-web/` | Web portal | Studio destination |
| `tools/`, `tests/`, `.github/` | Gates, their tests and the hosted workflows | Workbench; tests follow what they import |
| `data/`, `results/`, `figures/` | Committed evidence and its digests | Workbench |
| `docs/`, `paper/`, `notebooks/`, `examples/`, `benchmarks/` | Documentation and worked material | Workbench |
| `experimental_workers/` | Isolated experimental worker | Workbench |

## What the repository does not contain

- The classical SCPN engines `sc-neurocore` and `scpn-fusion-core`. The package
  talks to them through adapters; their code lives in their own repositories.
- The Studio platform package `scpn-studio-platform`. The Studio lane installs
  it from a hash-locked requirements file.
- Provider credentials, account identifiers and queue tokens. Hardware access
  is configured outside the repository.
- A permissively licensed core. All code here is `AGPL-3.0-or-later` with a
  commercial licence route; see [Core Package Boundary](../core_package_boundary.md).
- Any of the planned split repositories. None exists yet.

## Risks accepted

- **One repository, one long gate.** A full hosted run takes more than an hour
  (measured 2026-10-04). A change in one domain waits for the gates of all.
- **Existing exceptions are debt.** On 2026-10-05 the guard reported 174
  exception rows covering 292 import occurrences; its output gives the current
  figures. They are frozen, not solved.
- **Sibling distributions share the history.** `oscillatools` and the Rust
  engine carry their own versions but are built and published by this
  repository's workflows, from this repository's commits.
- **Evidence is bound to sources by digest.** Several committed evidence files
  record the digests of source and documentation files; changing such a file
  requires refreshing the evidence in the same change.

## Commands

```bash
# Who owns every tracked path, and how the units depend on each other
python tools/audit_split_ownership.py --out /tmp/split-ownership

# No new, widened or stale exception to the declared dependency graph
python tools/split_boundary_guard.py

# Every tracked source kind is under a named gate
python tools/audit_source_surface_inventory.py
```

The hosted step "Validate split dependency contracts" runs the guard and the
tests of the ownership tools on every push.

## Non-claims

- No split has been carried out. The destination names in the map are planned
  targets; they are not existing repositories, packages or teams.
- The map records architectural responsibility. It is not a list of
  maintainers and grants no review rights.
- No file is relicensed or dual-licensed by this record.
- The exception baseline does not show that the remaining cross-destination
  imports are acceptable, only that their number cannot grow unnoticed.
