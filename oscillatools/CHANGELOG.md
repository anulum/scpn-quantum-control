<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Changelog

All notable changes to `oscillatools` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial extraction of the coupled-phase-oscillator (Kuramoto) toolkit from
  `scpn-quantum-control` into a standalone, NumPy+SciPy-floor distribution: the model
  family, integrators, exact mean-field reductions, stability/continuation analysis,
  reverse-mode and forward differentiation, and control primitives.
- Optional acceleration and interoperability extras: `[rust]`, `[julia]`, `[jax]`,
  `[torch]`, `[sklearn]`, `[viz]`.
- A documentation site (mkdocs-material): a generated handbook, a capability
  snapshot, tier and competitive benchmark pages, and an example gallery, with the
  renderers under `tools/` and a deterministic worked-workflow example under
  `examples/`. The `[docs]` extra and a `mkdocs build --strict` CI job build it.
