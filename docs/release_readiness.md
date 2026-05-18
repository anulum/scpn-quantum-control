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

The audit composes five release checks:

| Check | Release meaning |
|---|---|
| Version consistency | `pyproject.toml`, package `__version__`, `CITATION.cff`, and `.zenodo.json` carry the same version. |
| Required release artefacts | Paper 0, coverage, behavioural-test, K_nm, stable core contracts, stable core contract fixtures, backend capability artefacts, and S2 blocker artefacts are present. |
| Coverage gap gate | A fresh `coverage.xml` exists, aggregate package coverage meets the release threshold, and unjustified missing files are blocked. Per-file gaps remain reported; `--fail-on-file-gap` can promote them to hard blockers. |
| Behavioural quality gate | Tests satisfy the smoke-only, assertion-density, and exception-contract-density thresholds. |
| Core-artifact determinism | Stable core contracts and backend capability artefacts are reproducible from committed metadata, and exported digests are checked before tagging. |

For any package tag or public release note touching stable-core contracts or backend
capabilities, the preferred gate is:

```bash
scpn-bench stable-core-release-gate
```

The bundle is a no-QPU reproducibility command that composes:

- `scpn-bench stable-core-capability-gate`
- `scpn-bench stable-core-contract-gate`

Use component gates only for targeted component-only verification.

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

The same closed-set discipline applies to stable core contracts and backend
capability artefacts. These artefacts are not only generated for runtime use;
they are treated as release-blocking evidence surfaces alongside version,
coverage, and behavioural gates.

For package tags or public release notes that touch stable-core contract
shape, adaptor boundaries, or contract-facing text, add:

```bash
scpn-bench stable-core-contract-gate
```

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


## Synchronisation benchmark release gate

Before any tag that touches benchmark code, benchmark registry rows, generated
benchmark artefacts, or paper-facing benchmark claims, run:

```bash
scpn-bench sync-benchmark-gate
```

The gate regenerates the standardised synchronisation benchmark registry,
regenerates all committed no-QPU reference rows, and then runs the
multi-instance comparator. It is a no-QPU release gate: it must not submit IBM
or other hardware jobs, and it must not promote quantum-advantage or hardware
performance claims.

The release audit now treats the synchronisation benchmark registry, n=4/n=8
reference rows, and public benchmark pages as required release artefacts.

## Symmetry-sector mitigation release gate

Before any tag that touches symmetry-sector mitigation planning, planner
fixtures, or mitigation-eligibility claims, run:

```bash
scpn-bench symmetry-sector-mitigation-gate
```

The gate regenerates the committed planner-fixture JSON and public fixture
summary, then compares the regenerated planner output against the committed
artefact. The gate is offline, does not submit hardware jobs, and only locks the
planner eligibility/blocker contract.

## Stable-core release/repro gate

When release notes, API docs, or tags touch stable-core claims, run:

```bash
scpn-bench stable-core-release-gate
```

This preferred gate is no-QPU and reproducible. It is composed of:

- `scpn-bench stable-core-capability-gate`
- `scpn-bench stable-core-contract-gate`

Use these component gates for isolated changes to only one stable-core surface.

## Hardware result-pack release gate

Any tag, paper-facing update, website update, or release note that cites
promoted IBM hardware evidence must attach a hardware result-pack evidence
packet. The packet records three facts that must stay together:

1. verifier output from `scpn-verify-hardware-packs --json`;
2. deterministic export digests from `scpn-verify-hardware-packs --export-dir ... --json`;
3. reproduction logs for every cited pack's count-to-statistic command.

The packet schema and checklist are defined in
[`hardware_result_pack_release_checklist.md`](hardware_result_pack_release_checklist.md).
The release audit accepts the packet through:

```bash
./.venv-linux/bin/python tools/audit_release_readiness.py \
  --fail-on-blocker \
  --hardware-result-pack-evidence docs/internal/releases/<packet>.json
```

If a release does not cite promoted hardware evidence, record that decision in
the release notes. Do not cite raw-count hardware claims from README, website,
papers, or release notes without the evidence packet.

## Tagging procedure

Before tagging:

1. Generate a fresh coverage XML report with the release test command.
2. Run `scpn-bench sync-benchmark-gate` if the release touches benchmark code,
   benchmark registry rows, generated benchmark artefacts, or benchmark claims.
3. Run `scpn-bench stable-core-release-gate` if the release touches stable-core
   contracts, contract fixtures, capability claims, or stable-core API text.
5. Run `scpn-bench symmetry-sector-mitigation-gate` if the release touches
   symmetry-sector mitigation planning, planner fixtures, or mitigation claims.
6. If the release cites promoted hardware evidence, generate the hardware
   result-pack evidence packet and pass it to the release audit.
7. Run `tools/audit_release_readiness.py --fail-on-blocker` with the hardware
   result-pack evidence argument when applicable.
8. Run the scoped docs build and version-consistency checks.
9. Commit with the required trailer after staged-diff audit.
10. Push the commit and wait for CI before creating a release tag.
