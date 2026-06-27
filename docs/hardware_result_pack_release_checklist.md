<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- hardware result-pack release checklist -->

# Hardware Result-Pack Release Checklist

This checklist is mandatory for any tag, paper-facing update, website update,
or release note that cites promoted IBM hardware evidence. It is intentionally
offline: the release process must preserve committed artefacts and replay
count-to-statistic paths, not submit new QPU jobs.

## Required commands

Preferred generator from the release candidate commit:

```bash
scpn-generate-hardware-pack-evidence \
  --output-dir private internal records/releases \
  --export-dir dist/hardware-result-packs
```

For a release that cites only selected packs, add one or more `--pack-id`
arguments. For a release that cites no promoted hardware evidence, write a
non-citing packet instead:

```bash
scpn-generate-hardware-pack-evidence \
  --non-citing \
  --reason "Release notes do not cite promoted IBM hardware evidence."
```

The generator runs the verifier, writes deterministic exports, runs every
selected pack's reproduction command, stores logs under `<private-internal-record>`,
computes log SHA-256 digests, and writes the evidence packet. Manual verifier
and export commands remain acceptable only when the packet matches the schema
below.

The generator admits each verifier, export, and reproduction command only after
resolving the executable token to an absolute executable file. Missing or
non-executable commands fail closed and write the stderr/log evidence beside
the release packet outputs.

## Evidence packet schema

Pass the packet to the release audit with
`--hardware-result-pack-evidence <path>`. The packet must be JSON with this
shape:

```json
{
  "schema_version": 1,
  "hardware_evidence_cited": true,
  "verifier_summary_path": "private internal records/releases/hardware_result_packs_verify_YYYY-MM-DD.json",
  "export_summary_path": "private internal records/releases/hardware_result_packs_export_YYYY-MM-DD.json",
  "reproduction_logs": [
    {
      "pack_id": "phase2_dla_parity_ag_ibm_kingston_2026_05_05",
      "command": "python scripts/analyse_phase2_dla_parity.py --verify-integrity",
      "log_path": "private internal records/releases/phase2_dla_parity_ag_reproduction_YYYY-MM-DD.log",
      "sha256": "<sha256 of log file>"
    }
  ]
}
```

If no promoted hardware evidence is cited, use:

```json
{
  "schema_version": 1,
  "hardware_evidence_cited": false,
  "reason": "Release notes do not cite promoted IBM hardware evidence."
}
```

## Acceptance gates

- `schema_version` is `1`.
- `hardware_evidence_cited=true` packets include verifier and export summaries.
- Verifier summary reports at least one pack and no blockers.
- Export summary includes one export digest per cited pack.
- Every cited pack has a reproduction log entry.
- Every reproduction log exists and matches its declared SHA-256 digest.
- No packet may promote claims outside `docs/hardware_status_ledger.md`.

## Claim boundary

The evidence packet proves integrity and replayability of committed artefacts.
It does not prove new hardware performance, broad quantum advantage, clinical
validity, or unregistered scaling claims.
