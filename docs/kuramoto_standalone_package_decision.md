# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto standalone package decision

# Kuramoto Standalone Package Decision

Decision status: deferred pending CEO/IP approval.

As of 2026-06-26, the Kuramoto toolkit remains part of the
`scpn-quantum-control` distribution. Phase 5 establishes that the toolkit has a
coherent public facade, handbook, worked workflow, notebook, and benchmark
evidence, but it does not create a standalone package, PyPI project, namespace,
wheel, source distribution, release workflow, or commercial distribution
boundary.

## Decision

The current decision is to keep Kuramoto in-repository and expose it through:

- `scpn_quantum_control.kuramoto` for the 200-symbol classical Kuramoto facade.
- `scpn_quantum_control.kuramoto_core` for stable quantum-control problem,
  Hamiltonian, circuit, and measurement contracts.
- `scpn_quantum_control.accel` for the lower-level accelerated primitives.
- `scpn_quantum_engine` for the existing Rust PyO3 engine bindings.

No package named `kuramoto`, `quantum-kuramoto`, `scpn-kuramoto`, or
`scpn-quantum-kuramoto` is approved by this decision record. No code should
import a standalone namespace, duplicate the engine, or publish a split
distribution until the CEO/IP decision is recorded in a later decision record.

## Rationale

The split is attractive because the Kuramoto surface is now coherent enough to
stand on its own: the facade is grouped, the handbook is generated from live
capabilities, the worked example is executable, and the tier benchmark has
CI/local evidence with explicit claim boundaries.

The split is not approved yet because it changes ownership, licensing,
support, release, and compatibility obligations. A separate package would need
its own name clearance, package metadata, dependency budget, release cadence,
security policy, documentation site, support boundary, and commercial licensing
route. Those are CEO/IP decisions, not implementation defaults.

## Readiness Snapshot

| Surface | Current state | Split implication |
|---|---|---|
| Public API | `scpn_quantum_control.kuramoto` groups the toolkit into documented capabilities. | Candidate import surface exists, but the namespace is not approved. |
| Engine | Local engine remains the 171-function Rust PyO3 build. | A split package must decide whether it depends on `scpn_quantum_engine` or vendors no Rust code. |
| Documentation | Handbook, examples, notebooks, and benchmark pages are wired into this repository. | A split package needs its own docs navigation and release docs. |
| Benchmarks | CI/local tier benchmark evidence is committed and fail-closed for external claims. | External package comparisons still require a separate harness. |
| Licensing | Repository remains AGPL with commercial license available. | A split requires explicit licensing and commercial-use terms. |
| Distribution | `scpn-quantum-control` remains the only approved PyPI distribution for this surface. | A split requires a new package name, trusted publishing, signing, and release governance. |

## Promotion Gate

A future standalone package decision must record all of the following before
implementation:

1. Approved package name and import namespace.
2. License and commercial-use policy.
3. Dependency budget and optional-extra policy.
4. Ownership of the Rust engine dependency and wheel strategy.
5. Migration policy for `scpn_quantum_control.kuramoto` users.
6. Documentation and tutorial ownership.
7. External benchmark harness scope and claim boundary.
8. Release, signing, SBOM, security, and support process.

Until that record exists, Phase 5.6 is closed as a deferred decision, not as a
package split.

## Operational Rules

- Do not add a standalone package directory or import namespace.
- Do not add a new PyPI publish workflow for Kuramoto.
- Do not rename the existing `scpn-quantum-control` package.
- Do not duplicate the Rust engine bindings.
- Do not advertise standalone install commands.
- Continue documenting the current facade as an in-repository surface.
