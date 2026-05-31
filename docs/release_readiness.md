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
- `scpn-bench stable-core-preflight-gate`

Use component gates only for targeted component-only verification.

## Paper 0 lane registry gate

For release notes, API docs, or pathway text that touch Paper 0 downstream
lanes, run:

```bash
scpn-bench paper0-lane-registry-gate
```

This is a no-QPU reproducibility gate. It confirms that the public Paper 0 lane
registry and its JSON companion are regenerated from repository artefacts. It
does not establish external validation, measured-system evidence, or hardware
readiness for any Paper 0 lane.

## Coverage and test-quality closure boundary

The release gate keeps coverage and behavioural value enforceable without
pretending that total line coverage is already 100 percent. A fresh coverage
XML report is required for tag readiness. The default tag gate blocks on
aggregate coverage below 95 percent and unjustified missing files. Per-file gaps
remain a release follow-up queue unless explicitly promoted to hard blockers
with `--fail-on-file-gap`. The 100 percent coverage target remains a future
improvement, not a blocker for tagging a bounded release.

Reviewed release-audit notes and local coverage exclusions are internal
operational artefacts, not part of the public documentation site. Hand-maintained
source files remain under the normal coverage and behavioural-quality gates.

The same closed-set discipline applies to stable core contracts and backend
capability artefacts. These artefacts are not only generated for runtime use;
they are treated as release-blocking evidence surfaces alongside version,
coverage, and behavioural gates.

For package tags or public release notes that touch stable-core contract
shape, adaptor boundaries, or contract-facing text, add:

```bash
scpn-bench stable-core-contract-gate
```

For tags that change stable-core preflight fixture expectations, add:

```bash
scpn-bench stable-core-preflight-gate
```

## Scientific gap closure boundary

Open scientific questions do not block a software release when their claim
boundaries are enforced by hard gates. The current release boundary is:

| Gap | Release gate |
|---|---|
| K_nm measured-system validation | Required release artefacts include the EEG PLV, IEEE 5-bus, and IEEE 14-bus comparison payloads plus the measured-coupling checklist. Physical-validation promotion remains blocked unless units, uncertainty, full pairwise coverage, tolerance, response, and null-model requirements pass. |
| TCBO `p_H1` reproduction | Promotion remains blocked without a named preregistered dataset and uncertainty crossing the threshold gate. |
| S2/S5 broad advantage | IBM advantage readiness remains blocked until the full benchmark matrix, hardware rows, and claim-boundary requirements pass. |
| Paper 0 downstream programme | Paper 0 is processed as source-bounded ingestion; downstream experiments require lane registry, methodology outline, and preregistered measured-system design before stronger claims. |
| S7 logical-level DLA parity | Required release artefacts include the logical-DLA roadmap JSON and Markdown note. DLA parity survival under logical encoding remains blocked until the theory, logical-observable, and simulation prerequisites pass. |
| S8 adaptive branching | Required release artefacts include the adaptive-branching readiness JSON and Markdown note. Adaptive advantage remains blocked until backend dynamic-circuit support, preregistration, and equal-depth open-loop falsification pass. |
| S9 quantum thermodynamics | Required release artefacts include the quantum-thermodynamics readiness JSON and Markdown note. Entropy-production peak claims remain blocked until theory review, classical reference, raw-count execution, and falsification controls pass. |
| S10 analog-native Kuramoto | Required release artefacts include the analog-native readiness JSON and Markdown note. Analog advantage and provider-execution claims remain blocked until calibrated provider SDK construction, matched-tolerance digital baselines, and raw execution records pass. |
| S11 sync-order quantum sensing | Required release artefacts include the quantum-sensing readiness JSON and Markdown note. Sensing-advantage claims remain blocked until a preregistered perturbation benchmark, classical Fisher baseline, hardware shot budget, and raw-count uncertainty archive pass. |

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

## S7 logical-DLA roadmap gate

Before any tag that touches logical-level DLA parity, fault-tolerant outlook,
or post-NISQ QEC claims, run:

```bash
scpn-bench s7-logical-dla-roadmap
```

The gate regenerates the S7 logical-DLA parity resource table and roadmap note.
It is offline, performs no hardware submission, and keeps DLA parity survival
claims blocked until the representation-theory and logical-observable work is
complete.

## S8 adaptive-branching readiness gate

Before any tag that touches mid-circuit adaptive branching, dynamic-circuit
feedback, or adaptive-advantage framing, run:

```bash
scpn-bench s8-adaptive-branching-readiness
```

The gate regenerates the S8 branch-policy table and readiness note. It is
offline, performs no hardware submission, and keeps adaptive-advantage claims
blocked until backend support and the preregistered equal-depth comparison pass.

## S9 quantum-thermodynamics readiness gate

Before any tag that touches entropy-production, irreversibility, heat-current,
or quantum-thermodynamics synchronisation-transition claims, run:

```bash
scpn-bench s9-quantum-thermo-readiness
```

The gate regenerates the S9 calibrated-observable K-sweep readiness table and
public note. It is offline, performs no hardware submission, and keeps
thermodynamic peak claims blocked until the formalism, classical reference,
and preregistered raw-count comparison pass.

## S10 analog-native readiness gate

Before any tag that touches analog-native Kuramoto backends, provider exports,
or analog-advantage framing, run:

```bash
scpn-bench s10-analog-native-readiness
```

The gate regenerates the S10 primitive-accounting table and provider-readiness
note. It is offline, performs no hardware submission, and keeps analog
advantage claims blocked until provider SDK construction, calibrated unit
constraints, matched-tolerance digital baselines, and raw execution records
pass.

## S11 quantum-sensing readiness gate

Before any tag that touches QFI-based sensing, synchronisation-order sensing,
or sensing-advantage framing, run:

```bash
scpn-bench s11-quantum-sensing-readiness
```

The gate regenerates the S11 QFI/classical-Fisher proxy table and public note.
It is offline, performs no hardware submission, and keeps sensing-advantage
claims blocked until the preregistered perturbation benchmark, classical Fisher
baseline, shot budget, and raw-count uncertainty archive pass.

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
- `scpn-bench stable-core-preflight-gate`

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
4. Run `scpn-bench stable-core-preflight-gate` if the release changes stable-core
   preflight fixtures or preflight-facing API/docs text.
5. Run `scpn-bench symmetry-sector-mitigation-gate` if the release touches
   symmetry-sector mitigation planning, planner fixtures, or mitigation claims.
6. Run `scpn-bench knm-measured-candidate-gate` if the release touches Paper 0
   K_nm measured-system audit artefacts, unit-class promotion logic, or
   measured-candidate claim text.
7. If the release cites promoted hardware evidence, generate the hardware
   result-pack evidence packet and pass it to the release audit.
8. Run `tools/audit_release_readiness.py --fail-on-blocker` with the hardware
   result-pack evidence argument when applicable.
9. Run the scoped docs build and version-consistency checks.
10. Commit with the required authorship line after staged-diff audit.
11. Push the commit and wait for CI before creating a release tag.
