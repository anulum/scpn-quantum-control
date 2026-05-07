<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Metriq submission decision -->

# Metriq Submission Decision

Date: 2026-05-07

## Decision

No public Metriq upload was performed.

This closes the current Metriq roadmap item as a deliberate
no-upload decision rather than an omitted task. The reason is scientific
and procedural: the only available Metriq-native artefact is the local
Bernstein--Vazirani simulator smoke result from
`docs/metriq_local_smoke_2026-05-06.md`. Uploading that result would test
Metriq plumbing, but it would not represent `scpn-quantum-control`, the
Kuramoto--XY workflow, the DLA/FIM papers, or the Rust/VQE benchmark
claims.

## Dry-Run Evidence

The upload path was checked with Metriq-Gym dry-run mode:

```bash
/home/anulum/.venvs/scpn-metriq/bin/mgym job upload \
  b96914ac-2e2f-461e-84a8-b61d81300fb2 \
  --dry-run \
  --repo unitaryfoundation/metriq-data
```

Dry-run result:

```text
Preparing job upload...
[Cached result data]
DRY-RUN: wrote mock file at /tmp/mgym-dryrun-9oje8wcq/metriq-data/metriq-gym/v0.7/local/aer_simulator/2026-05-06_18-47-23_bernstein-vazirani_7d11b013.json; would create branch 'mgym/upload-b96914ac-2e2f-461e-84a8-b61d81300fb2' and open PR to unitaryfoundation/metriq-data (base: main) with title: mgym upload: Bernstein-Vazirani on local/aer_simulator
```

## Valid Future Paths

Future Metriq publication should use one of these routes:

1. Run a Metriq-native benchmark that is scientifically relevant to the
   package and upload that benchmark result with explicit approval.
2. Propose an SCPN/Kuramoto--XY benchmark schema upstream to Metriq-Gym,
   wait for acceptance, then run and upload results under the accepted
   schema.

## Non-Submission Boundary

The following were not submitted:

- project-specific Rust/VQE benchmark tables;
- DLA parity raw-count tables;
- SCPN/FIM hardware artefacts;
- GPU or cross-machine timing artefacts;
- arbitrary JSON/CSV outputs outside the Metriq-Gym schema.

This preserves the repository's artefact-first discipline and avoids
misrepresenting SCPN-specific paper data as a Metriq-native benchmark.
