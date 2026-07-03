# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto standalone package decision

# Kuramoto Standalone Package Decision

Decision status: APPROVED 2026-07-04 (CEO/IP). This record supersedes the prior deferral
(2026-06-26) and satisfies the Promotion Gate by recording all eight required decisions below.
Implementation proceeds in phases; the standalone package is created only after the pending
in-repository work lands so it captures the current surface, and the outward-facing publish steps
remain separately owner-gated (see *Implementation phasing*).

## Decision

The Kuramoto toolkit — the `scpn_quantum_control.accel` package (the accelerated primitives), the
`scpn_quantum_control.kuramoto` facade, and the `forecasting` neural-operator surface — is approved for
extraction into a **standalone, light, pip-installable distribution** named **`oscillatools`**
(PyPI name verified available 2026-07-04). The distribution depends only on NumPy and SciPy; every
other tier (Rust engine, Julia, JAX, PyTorch, matplotlib) is an optional extra. `scpn-quantum-control`
becomes a consumer of `oscillatools` and retains backward-compatible re-export access to
`scpn_quantum_control.kuramoto` / `scpn_quantum_control.accel` under a staged deprecation.

The prior decision (keep Kuramoto in-repository, expose it only through
`scpn_quantum_control.kuramoto`, `scpn_quantum_control.kuramoto_core`, `scpn_quantum_control.accel`,
and `scpn_quantum_engine`) is retained as the *current* surface until the migration lands, and remains
valid for `kuramoto_core` (the quantum-control compiler contracts, which are NOT part of this split).

## Promotion Gate — the eight recorded decisions

1. **Package name and import namespace.** `oscillatools` (distribution and import namespace).
   Verified available on PyPI 2026-07-04. The names `kuramoto`, `quantum-kuramoto`, `scpn-kuramoto`,
   and `scpn-quantum-kuramoto` remain unused.
2. **License and commercial-use policy.** AGPL-3.0-or-later with a commercial license available —
   identical to `scpn-quantum-control`. The package carries `LICENSE`, `LICENSES/`, `NOTICE.md`,
   `REUSE.toml`, and the SPDX header convention.
3. **Dependency budget and optional-extra policy.** Hard dependencies: `numpy` and `scipy` only.
   Optional extras: `[rust]` (`scpn-quantum-engine`), `[julia]` (`juliacall`), `[jax]`, `[torch]`,
   `[viz]` (`matplotlib`). No Qiskit, no provider SDKs.
4. **Ownership of the Rust engine dependency and wheel strategy.** The package **depends on**
   `scpn-quantum-engine` as the `[rust]` extra and does **not** vendor or duplicate the engine
   bindings. The default wheel is a pure-Python `py3-none-any` build (hatchling), matching the
   optional-engine, NumPy-floor design already used by `accel`.
5. **Migration policy for `scpn_quantum_control.kuramoto` users.** `scpn-quantum-control` depends on
   `oscillatools`; `scpn_quantum_control.kuramoto` and `scpn_quantum_control.accel` become re-export
   shims that emit a `DeprecationWarning` naming the new import path, staged per `DEPRECATIONS.md` (new
   path lands beside the old with a warning at the next minor, the warning is kept for at least two
   further minor releases, and the old path is removed no earlier than the next major).
6. **Documentation and tutorial ownership.** `oscillatools` carries its own documentation site
   (handbook, example gallery, tier and competitive benchmark pages); `scpn-quantum-control` links to
   it and records that the surface has moved.
7. **External benchmark harness scope and claim boundary.** The competitive benchmark harness moves
   with the package and keeps its fail-closed, boundary-guarded claim discipline; performance ratios
   remain boundary-guarded (host-dependent) rather than headline claims.
8. **Release, signing, SBOM, security, and support process.** A dual-tag `oscillatools-v*`
   trusted-publishing workflow (PyPI OIDC + Sigstore attestations) mirroring the existing
   `publish.yml`, CycloneDX SBOM emission, a `SECURITY.md`, and Dependabot coverage.

## Implementation phasing

- **F1 (this record).** Records the CEO/IP approval and the eight decisions; updates the enforcing
  test. No package directory or namespace is created yet.
- **F2.** Physically lift `accel/` + the `kuramoto` facade (+ the forecasting neural-operator module)
  into `oscillatools/`. This runs **after the pending in-repository combined push lands**, so the
  extracted surface is complete and current (it must include the delay-sensitivity, saddle-node,
  neural-operator, JAX-backend, and visualisation work still in flight).
- **F3.** Re-export shims + `DEPRECATIONS.md` entry in `scpn-quantum-control`.
- **F4.** Documentation site, publish/SBOM/signing workflow, and the enforcing-test update to the
  post-extraction state.
- **F5 (owner-gated, NOT autonomous).** The PyPI publish (`oscillatools-v0.1.0` tag), the Zenodo DOI
  mint, and the JOSS paper's name/install-command update happen only on an explicit, separate owner
  instruction.

## Operational Rules (superseding the prior deferral)

- The approved distribution name is `oscillatools`; do not use `kuramoto`, `quantum-kuramoto`,
  `scpn-kuramoto`, or `scpn-quantum-kuramoto`.
- Do not rename the existing `scpn-quantum-control` package.
- Do not duplicate the Rust engine bindings; depend on `scpn-quantum-engine` as an optional extra.
- Do not publish to PyPI, mint a Zenodo DOI, or advertise install commands until the owner-gated F5
  instruction is given.
- Keep `scpn_quantum_control.kuramoto` working via the re-export shim throughout the deprecation window.
