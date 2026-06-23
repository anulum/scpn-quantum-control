<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Release Readiness Gate -->

# Release Readiness Gate

Date added: 2026-05-18

## Purpose

This is the release control page for software changes that touch stable contracts,
core evidence, or claim-facing documentation. It gives teams one deterministic
route to prove what is and is not safe to ship.

Use this gate before tagging:

- release managers preparing a package tag;
- reviewers closing a milestone;
- any PR that changes release-blocking artefacts, contracts, or claim tables.

## What it governs

The page is a blocker policy for artefact-backed releases. It governs:

- version consistency across public package metadata,
- licence and commercial-route consistency across metadata, docs, and headers,
- stable-contract fixture and capability generation reproducibility,
- coverage and behavioural gating thresholds,
- open-surface scientific gaps that remain bounded until explicit gates pass.

This page records the release-blocker closure path for the next package tag.
The release decision is no longer based on scattered manual judgement across
coverage, behavioural-test quality, source ingestion, and scientific claim
boundaries. The deterministic gate is:

```bash
./.venv-linux/bin/python tools/audit_release_readiness.py --fail-on-blocker
```

The audit composes six release checks:

## For teams shipping with this repository

These checks exist to separate “works on my machine” from releasable outcomes.
They were designed so that:

- claims are tied to committed artefacts, not prose-only;
- changes to core interfaces are checked against contracts;
- source-accounting and performance/validation gates are explicit before a new tag.

If you are preparing a pilot, demo, or publication-facing release, this is the
single place to prove that the same route is repeatable across machines.

| Check | Release meaning |
|---|---|
| Version consistency | `pyproject.toml`, package `__version__`, `CITATION.cff`, and `.zenodo.json` carry the same version. |
| Required release artefacts | Coverage, behavioural-test, K_nm, stable core contracts, stable core contract fixtures, backend capability artefacts, release coverage exclusions, and S2 blocker artefacts are present. |
| Coverage gap gate | A fresh `coverage.xml` exists, aggregate package coverage meets the release threshold, and unjustified missing files are blocked. Intentional CPU-only omissions must be listed in [`release_coverage_exclusions.json`](release_coverage_exclusions.json). Per-file gaps remain reported; `--fail-on-file-gap` can promote them to hard blockers. |
| Behavioural quality gate | Tests satisfy the smoke-only, assertion-density, and exception-contract-density thresholds. |
| Licence readiness gate | `pyproject.toml`, `LICENSE`, README, [`core_package_boundary.md`](core_package_boundary.md), [`licensing_faq.md`](licensing_faq.md), and source/tool SPDX headers agree that this repository is `AGPL-3.0-or-later` with a commercial route until an approved split changes all surfaces. |
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


## Coverage and test-quality closure boundary

The release gate keeps coverage and behavioural value enforceable without
pretending that total line coverage is already 100 percent. A fresh coverage
XML report is required for tag readiness. The default CI gate blocks on
aggregate coverage below 70 percent and unjustified missing files. Per-file gaps
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
  --hardware-result-pack-evidence private internal records/releases/<packet>.json
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
7. If the release cites promoted hardware evidence, generate the hardware
   result-pack evidence packet and pass it to the release audit.
8. Confirm intentional CPU-only coverage omissions are justified in
   [`release_coverage_exclusions.json`](release_coverage_exclusions.json).
9. Run `tools/audit_release_readiness.py --fail-on-blocker` with the hardware
   result-pack evidence argument when applicable.
10. Run the scoped docs build and version-consistency checks.
11. Commit with the required authorship line after staged-diff audit.
12. Push the commit and wait for CI before creating a release tag.

## 0.9.12 release, documentation, and repository-hygiene scope

The `0.9.12` source release packages the June 2026 differentiable-programming
hardening queue and refreshes the public documentation surface so first-time
users can understand the software, its applications, its commercial route, and
its claim boundaries before reading subsystem internals. It does not promote
broad quantum advantage, clinical validation, arbitrary simulator autodiff, or
unbounded hardware-gradient execution.

The release scope is:

- version-consistent `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, README,
  capability manifest, documentation site pages, and changelog entries;
- README, site home, onboarding, tutorials, notebook guide, API overview, and
  release-readiness pages that route users by job, evidence class, and adoption
  objective;
- differentiable-programming additions from the current queue, including
  bounded QNN/QNode gradient, convergence, framework, finite-shot, stochastic,
  Rust/PyO3 parity, and compiler/adapter evidence surfaces already committed in
  the release branch;
- explicit benchmark classification boundaries: non-isolated rows remain local
  regression or functional evidence, while production performance wording still
  requires isolated-affinity artefacts;
- GitHub push, tag, CI validation, security-alert inspection, open-PR review,
  and safe Actions/deployment cleanup before public release promotion.


## 0.9.11 documentation, native AD, and release-polish scope

The `0.9.11` source release packages the latest documentation and
differentiable-programming hardening round. It does not promote arbitrary
program AD, hardware gradients, full ML-framework-native autodiff, or broad
quantum advantage. The release scope is:

- public docs now route first-time users through onboarding, applications,
  tutorials, notebooks, differentiable programming, and claim boundaries before
  low-level APIs;
- version-consistent `pyproject.toml`, `CITATION.cff`, `.zenodo.json`,
  capability manifest, README snapshot, reproducibility metadata, and release
  notes;
- native compiler-backed whole-program AD determinant support widened from
  helper-backed `6x6`-`16x16` to verified `6x6`-`19x19`;
- static dense `20x20+` determinant traces are explicitly documented as
  fail-closed after strict native verification rejected the current helper
  formulation;
- repository hygiene remains bound by module-specific tests, release-readiness
  gates, security scans, and GitHub CI before public promotion.

## 0.9.10 documentation and differentiable-programming release scope

The `0.9.10` source release aligns the public documentation with the current
differentiable-programming implementation state. It does not promote arbitrary
program AD, hardware gradients, full ML-framework-native autodiff, or broad
quantum advantage. The release scope is:

- clearer README, onboarding, tutorial, notebook, API, and gradient-route entry
  points;
- version-consistent `pyproject.toml`, `CITATION.cff`, `.zenodo.json`,
  capability manifest, and reproducibility metadata;
- explicit support boundaries for parameter-shift, composed objectives,
  provider-gradient readiness, transform nesting, compiler/program AD
  primitives, and unsupported routes;
- continued requirement that hardware and scientific claims cite committed
  evidence artefacts before promotion.

## 0.9.9 release-documentation scope

The `0.9.9` source release documentation surface adds a public
differentiable-programming route. The release docs must explain parameter-shift
gradient support, compiler/program AD boundaries, public API entry points,
planned tape/framework-adapter work, notebooks, benchmarks, and fail-closed
unsupported scenarios without promoting roadmap items as completed features.

## 0.9.8 release-documentation scope

The `0.9.8` source release documentation surface is required to explain the
project before users reach low-level APIs. At minimum, the public site must
include:

- onboarding overview with purpose, user routes, application lanes, commercial
  route, and claim boundaries;
- quickstart route that runs without IBM credentials;
- tutorials and notebook maps that separate exploration from release evidence;
- API overview that points users to stable facades first;
- hardware and release-readiness pages that keep simulator, hardware, and open
  scientific claims separated.

This documentation scope is a release artefact because misunderstanding the
claim boundary is a product and scientific safety risk.
