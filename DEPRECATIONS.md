# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Deprecation & SemVer Policy

# Deprecation Policy and SemVer Contract

This document records what `scpn-quantum-control` considers a public
API surface, what counts as a breaking change, and how deprecations
are staged. The rules here are binding from **v1.0.0** onwards.
Until v1.0.0 the project follows the pre-1.0 clause of SemVer 2.0.0
(anything may change in a 0.y.z release).

## Public API surface

The following is stable and covered by this contract:

- Every name exported from `scpn_quantum_control.__all__`
  (currently 104 symbols).
- Every name exported from a subpackage `__init__.__all__`.
- The CLI entry points declared in `pyproject.toml` (`project.scripts`).
- The command-line arguments of scripts in `scripts/` that are
  referenced from `README.md` or `CHANGELOG.md`.
- The JSON schema of hardware-result files under `data/` (keys
  documented in that directory's `README.md`).
- The `scpn_quantum_engine` Rust crate's `#[pyfunction]` exports
  and their type signatures.

Everything else — private modules (`_foo.py`, `foo._bar`), internal
Rust modules without `#[pyfunction]`, undocumented helper
constants, fixture files under `tests/`, and agent-facing docs under
`docs/internal/` — is explicitly **not** part of the public contract
and may change without notice between releases.

## What counts as a breaking change

Any of the following in the public API surface above:

1. Removing a symbol, method, attribute, or CLI flag.
2. Renaming a symbol (treated as removal + addition).
3. Changing a function signature in a way that rejects input
   previously accepted — e.g. adding a required parameter,
   tightening a type annotation from `float | None` to `float`,
   or reducing the accepted domain of a numeric argument.
4. Changing the return type or return shape — e.g. changing a
   `numpy.ndarray` return to a `qiskit.SparsePauliOp`.
5. Changing observable numerical behaviour in a way that breaks a
   test exercising a documented invariant (e.g. the `k_nm` anchor
   values in `bridge/knm_hamiltonian.py` or the DLA dimension
   formula in `analysis/dla_parity_theorem.py`).
6. Changing the JSON schema of a result file in a way older readers
   cannot decode.
7. Tightening Python or Rust version requirements inside a minor
   release (e.g. bumping `requires-python` from `>=3.10` to
   `>=3.11`).
8. Changing the default value of a keyword argument where the old
   default produced a different published number.

Adding new public APIs, adding optional keyword arguments with safe
defaults, adding new fields to result JSON (non-removing), and
loosening input validation are **non-breaking** — they go in a
minor release.

## Deprecation staging

When a public API needs to change, the following sequence is
mandatory before the removal lands:

1. **Release N (minor)** — the new behaviour lands alongside the old
   with a `DeprecationWarning` raised by the old call path. The
   deprecation is recorded in this file (`DEPRECATIONS.md`) with the
   release number and the removal target.
2. **At least two more minor releases** must pass with the
   deprecation warning in place, giving downstream callers a minimum
   of one full quarter to migrate.
3. **Next major release** (Nth → (N+1)th.0) — the old call path is
   removed. `CHANGELOG.md` flags the removal as breaking.

Example:

- `v1.2.0` — deprecate `foo(x, y)` in favour of `foo2(x, y, z=None)`;
  `DeprecationWarning` emitted.
- `v1.3.0`, `v1.4.0` — warning remains; both APIs work.
- `v2.0.0` — `foo()` removed; `CHANGELOG.md` lists it under
  **Removed**.

Deprecations cannot be skipped ("silent removal") even when the
author is confident no-one depends on the symbol. If the API was
public, someone downstream is depending on it.

## Out-of-cycle exceptions

Two narrow exceptions that bypass the staging sequence:

1. **Security fixes** that require a breaking change may ship in a
   patch release (`v1.2.1` instead of `v2.0.0`). The fix is
   highlighted in the release's `### Security` section and a
   pointer is added to the next
   `DEPRECATIONS.md` revision.
2. **Upstream requirement flag days** — when a required dependency
   (e.g. Qiskit, PyO3) ends support for a Python or Rust version
   that we promised to support, we follow the upstream timeline
   even if it falls inside a minor release. This is logged in
   `CHANGELOG.md` under `### Changed` with the upstream advisory
   linked.

## Current deprecations

None as of v0.9.6.

Once v1.0.0 ships this section becomes the authoritative ledger of
outstanding deprecations, their announcing release, their removal
target, and their migration guidance. Agents that propose a
breaking change must append an entry here before the change is
merged.
