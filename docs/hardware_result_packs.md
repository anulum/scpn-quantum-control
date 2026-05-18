<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- hardware result packs -->

# Reproducible Hardware Result Packs

Hardware result packs are the repository's offline trust surface for promoted
IBM hardware evidence. A pack binds raw counts, reproduced summaries, IBM job
identifiers, byte sizes, SHA-256 digests, reproduction commands, and explicit
claim boundaries into one committed manifest.

The pack layer does not submit QPU jobs and does not widen scientific claims.
It preserves what is already promoted in the hardware status ledger and makes
that evidence installable, checkable, and release-auditable.

## Current packs

| Pack | Evidence class | Claim boundary |
|---|---|---|
| `phase1_dla_parity_ibm_kingston_2026_04` | Promoted raw-count evidence | Phase 1 `n=4` DLA parity asymmetry observation only. |
| `phase2_dla_parity_ag_ibm_kingston_2026_05_05` | Promoted raw-count evidence | Reduced Phase 2 `n=4` A+G replication and readout-control claim only. |
| `phase2_scaling_bc_ibm_kingston_2026_05_05` | Promoted mixed raw-count evidence | Mixed B-C scaling evidence; falsifies a simple monotone scaling story. |
| `phase2_popcount_control_ibm_kingston_2026_05_05` | Promoted control raw-count evidence | Excitation-count confound control only. |
| `scpn_fim_negative_ibm_kingston_2026_05_05` | Promoted negative result | Simple digital `lambda=4` FIM hardware protection fails on the tested circuit family. |

## Manifest

The canonical manifest is:

```text
data/hardware_result_packs/manifest.json
```

Each pack records:

- stable pack identifier;
- backend and hardware family;
- execution date;
- required IBM job identifiers;
- reproduction command;
- promoted claim scope;
- non-claims that must not be inferred;
- artefact paths with byte size and SHA-256 digest.

## Offline verification

Run from a source checkout:

```bash
python scripts/verify_hardware_result_packs.py
```

Installed entry point:

```bash
scpn-verify-hardware-packs --repo-root /path/to/scpn-quantum-control
```

For archived or relocated manifests, pass both roots explicitly:

```bash
scpn-verify-hardware-packs \
  --repo-root /path/to/scpn-quantum-control \
  --manifest /path/to/scpn-quantum-control/data/hardware_result_packs/manifest.json
```

The verifier checks:

- manifest schema version;
- safe repository-relative artefact paths;
- artefact existence;
- exact byte size;
- exact SHA-256 digest;
- JSON parseability for JSON artefacts;
- presence of every declared IBM job identifier inside the pack artefacts.

Machine-readable output:

```bash
python scripts/verify_hardware_result_packs.py --json
```

Verify one pack only:

```bash
scpn-verify-hardware-packs \
  --repo-root /path/to/scpn-quantum-control \
  --pack-id phase2_dla_parity_ag_ibm_kingston_2026_05_05
```

## Deterministic release exports

After verification, the same CLI can emit deterministic per-pack archives:

```bash
scpn-verify-hardware-packs \
  --repo-root /path/to/scpn-quantum-control \
  --export-dir dist/hardware-result-packs
```

Each archive is named `<pack-id>.tar.gz` and contains:

- `PACK_MANIFEST.json` with the single-pack manifest and source-manifest path;
- every artefact listed for that pack under its repository-relative path.

Archive metadata is normalised with fixed timestamps, owner IDs, owner names,
and sorted entries, so repeated exports from the same artefacts have stable
SHA-256 digests. The CLI reports archive byte sizes and SHA-256 digests in
plain text or under the `exports` key in `--json` output.

## Release gate usage

Before publishing a release or paper-facing update that cites promoted hardware
results, run the verifier and the relevant pack reproduction commands. The
verifier proves artefact integrity; the reproduction command proves the
count-to-statistic path.

The result-pack verifier is intentionally narrower than full CI. It is a
hardware-evidence integrity gate, not a replacement for tests, docs builds, or
claim-boundary review.

## Next planned extensions

1. Add a release checklist that requires verifier output, export digests, and
   reproduction logs.
2. Add figure rebuild scripts to every pack where the current pack only records
   reproduced JSON summaries.
3. Add DOI/archive metadata once an external archive is minted.
