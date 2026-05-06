<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Zenodo Metadata Refresh Checklist -->

# Zenodo Metadata Refresh Checklist

Date prepared: 2026-05-06

This checklist prepares the manual Zenodo metadata refresh for the public
software DOI. It does not log in to Zenodo, edit the Zenodo record, publish a
new version, or change repository release metadata.

## Record to Refresh

Software DOI:

```text
10.5281/zenodo.18821929
```

Repository:

```text
https://github.com/anulum/scpn-quantum-control
```

Current in-repo metadata is internally consistent:

| Field | Value |
|-------|-------|
| `pyproject.toml` version | `0.9.6` |
| `src/scpn_quantum_control/__init__.py` version | `0.9.6` |
| `CITATION.cff` version | `0.9.6` |
| `.zenodo.json` version | `0.9.6` |
| `CITATION.cff` DOI | `10.5281/zenodo.18821929` |
| README citation DOI | `10.5281/zenodo.18821929` |

## Manual Zenodo Session Tasks

During an authenticated Zenodo session:

1. Open the record for DOI `10.5281/zenodo.18821929`.
2. Confirm the record title matches:

   ```text
   scpn-quantum-control: Quantum-Native SCPN Phase Dynamics and Control
   ```

3. Confirm creator metadata:

   ```text
   Sotek, Miroslav
   ORCID: 0009-0009-3560-0851
   Affiliation: ANULUM CH & LI
   ```

4. Confirm license:

   ```text
   AGPL-3.0-or-later
   ```

5. Add or request inclusion in appropriate Zenodo communities only where the
   community moderators and scope match the record:

   - quantum computing;
   - physics;
   - research software;
   - open science / reproducibility.

6. Confirm related identifiers include:

   - GitHub repository: `https://github.com/anulum/scpn-quantum-control`;
   - PyPI package: `https://pypi.org/project/scpn-quantum-control/`;
   - documentation: `https://anulum.github.io/scpn-quantum-control`;
   - Software Heritage origin or snapshot SWHID where Zenodo accepts it.

7. Confirm keywords remain bounded to the software/package scope:

   - quantum computing;
   - Kuramoto model;
   - XY Hamiltonian;
   - Qiskit;
   - NISQ;
   - IBM Heron r2;
   - error mitigation;
   - quantum error correction;
   - phase dynamics;
   - reproducible research software.

8. Do not add unsupported quantum-advantage, medical, biological, or
   consciousness claims to the public Zenodo record.

9. If Zenodo requires publishing a new version for metadata changes, record the
   new version DOI and update:

   - `CITATION.cff`;
   - `.zenodo.json`;
   - README citation block;
   - `docs/EXPORT_CONTROL.md`;
   - any active paper source that cites the software DOI.

10. After the manual session, create a dated follow-up note with:

    - communities requested or added;
    - record URL;
    - DOI or version DOI;
    - moderation status if any community requires approval;
    - metadata fields changed;
    - fields intentionally left unchanged.

## Claim Boundary

The Zenodo record is a software and artefact citation surface. It must not be
used to promote claims that are not already supported by committed papers,
datasets, manifests, and analysis scripts.

## Current Status

Preparation complete. Authenticated Zenodo execution remains open because it
requires a manual login session and potentially community-moderator approval.
