<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Release Readiness Gate -->

# Release Readiness Gate

Date added: 2026-05-18

This page records the release-blocker closure path for the next package tag.
The release decision is no longer based on scattered manual judgement across
coverage, behavioural-test quality, source ingestion, and scientific claim
boundaries. The deterministic gate is:

```bash
./.venv-linux/bin/python tools/audit_release_readiness.py --fail-on-blocker
```

The audit composes four release checks:

| Check | Release meaning |
|---|---|
| Version consistency | `pyproject.toml`, package `__version__`, `CITATION.cff`, and `.zenodo.json` carry the same version. |
| Required release artefacts | Paper 0, coverage, behavioural-test, K_nm, and S2 blocker artefacts are present. |
| Coverage gap gate | A fresh `coverage.xml` exists, aggregate package coverage meets the release threshold, and unjustified missing files are blocked. Per-file gaps remain reported; `--fail-on-file-gap` can promote them to hard blockers. |
| Behavioural quality gate | Tests satisfy the smoke-only, assertion-density, and exception-contract-density thresholds. |

## Coverage and test-quality closure boundary

The release gate keeps coverage and behavioural value enforceable without
pretending that total line coverage is already 100 percent. A fresh coverage
XML report is required for tag readiness. The default tag gate blocks on
aggregate coverage below 95 percent and unjustified missing files. Per-file gaps
remain a release follow-up queue unless explicitly promoted to hard blockers
with `--fail-on-file-gap`. The 100 percent coverage target remains a future
improvement, not a blocker for tagging a bounded release.

Reviewed release exclusions live in:

```bash
docs/coverage_justified_exclusions_2026-05-18.json
```

The current exclusions are limited to optional Julia-runtime wrappers and the
generated Paper 0 source-accounting package. Hand-maintained source files remain
under the normal coverage and behavioural-quality gates.

## Scientific gap closure boundary

Open scientific questions do not block a software release when their claim
boundaries are enforced by hard gates. The current release boundary is:

| Gap | Release gate |
|---|---|
| K_nm measured-system validation | Physical-validation promotion remains blocked unless units, uncertainty, full pairwise coverage, tolerance, response, and null-model requirements pass. |
| TCBO `p_H1` reproduction | Promotion remains blocked without a named preregistered dataset and uncertainty crossing the threshold gate. |
| S2/S5 broad advantage | IBM advantage readiness remains blocked until the full benchmark matrix, hardware rows, and claim-boundary requirements pass. |
| Paper 0 downstream programme | Paper 0 is processed as source-bounded ingestion; downstream experiments require lane registry, methodology outline, and preregistered measured-system design before stronger claims. |

This means the release can be tagged when software gates pass. It does not mean
that source ingestion, simulator output, or partial benchmark rows have become
external scientific validation.

## Tagging procedure

Before tagging:

1. Generate a fresh coverage XML report with the release test command.
2. Run `tools/audit_release_readiness.py --fail-on-blocker`.
3. Run the scoped docs build and version-consistency checks.
4. Commit with the required trailer after staged-diff audit.
5. Push the commit and wait for CI before creating a release tag.
