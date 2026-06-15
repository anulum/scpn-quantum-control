# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Licensing FAQ

# Licensing FAQ

This page states the current licensing boundary for `scpn-quantum-control`.
It is documentation for adopters and release reviewers; it does not change any
licence grant.

## Current Licence

The repository is published under `AGPL-3.0-or-later`, with a commercial
licence route available for proprietary use. The package metadata, SPDX
headers, README licence section, and `LICENSE` file must continue to agree
before a release is tagged.

## Common Routes

| Use case | Route |
|---|---|
| Academic research, teaching, and AGPL-compatible open work | Use the `AGPL-3.0-or-later` terms in `LICENSE`. |
| Closed-source products, internal proprietary tools, SaaS, consulting deliverables, or embedded deployments | Obtain a commercial licence grant before distribution or network-service use. |
| Future lightweight Kuramoto-XY core package | Not available as a permissive package today. |

## Core-Split Boundary

The repository documents a possible future lightweight core boundary in
[`core_package_boundary.md`](core_package_boundary.md). That document is a
planning boundary only. It does not relicense any file, package, symbol,
backend, benchmark, or generated artefact.

A future permissive split requires a reviewed release decision that names the
exact files, symbols, dependencies, SPDX headers, package metadata, and target
licence. Until that happens, all in-repository code remains under the
AGPL/commercial terms.

## Release Gate

Run the license-readiness gate before a tag or licence-affecting change:

```bash
python tools/audit_license_readiness.py --root .
```

The gate checks:

- `pyproject.toml` project licence and classifiers;
- `LICENSE`, README, core-boundary, and this FAQ for consistent public wording;
- SPDX and commercial-licence headers on Python source, tool, and script files.

The release-readiness audit also includes the same gate:

```bash
python tools/audit_release_readiness.py --project-root . --fail-on-blocker
```

If the gate fails, do not tag or publish until the blocker is fixed or an
approved licence-split release changes all affected surfaces in one reviewed
commit.
