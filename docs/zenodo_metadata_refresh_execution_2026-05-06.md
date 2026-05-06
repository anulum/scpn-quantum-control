<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Zenodo Metadata Refresh Execution -->

# Zenodo Metadata Refresh Execution

Date executed: 2026-05-06

This note records the authenticated Zenodo metadata refresh for the public
software record. No access token is stored in this document.

## Record

Concept DOI:

```text
10.5281/zenodo.18821929
```

Version DOI:

```text
10.5281/zenodo.18821930
```

Record URL:

```text
https://zenodo.org/records/18821930
```

## Authenticated API Result

The `codex` Zenodo token was added to the shared credential vault before this
execution. The token has `deposit:actions`, `deposit:write`, and `user:email`
scopes.

Actions completed:

1. Opened published record `18821930` for edit via the Zenodo deposition API.
2. Updated draft metadata.
3. Published the metadata edit.
4. Verified the public record after publication.

Publish response status:

```text
202 Accepted
```

## Public Metadata After Refresh

Verified public fields:

| Field | Value after refresh |
|-------|---------------------|
| Version | `0.9.6` |
| Publication date | `2026-03-29` |
| License | `agpl-3.0-or-later` |
| Language | `eng` |
| Concept DOI | `10.5281/zenodo.18821929` |
| Version DOI | `10.5281/zenodo.18821930` |

Keywords now include:

- quantum computing;
- Kuramoto model;
- XY Hamiltonian;
- Trotter decomposition;
- VQE;
- QAOA;
- SCPN;
- qiskit;
- NISQ;
- IBM Heron r2;
- error mitigation;
- quantum error correction;
- phase dynamics;
- reproducible research software.

Related identifiers verified publicly after publish:

- `https://github.com/anulum/scpn-quantum-control`
- `https://anulum.github.io/scpn-quantum-control`

The PyPI alternate identifier was included in the accepted draft metadata but
did not appear in the public record after publication. This appears to be a
Zenodo metadata-normalisation behaviour for the published record.

## Community Submission Status

Attempted community metadata:

```text
rse
```

The draft metadata update accepted the `rse` community identifier, but the
published public record still reports:

```text
communities: null
```

Zenodo's current public help describes submission of published records to
communities as a separate review workflow from the record page's communities
menu. Zenodo's public developer documentation lists deposit, records, and files
as stable REST APIs and describes communities APIs as still in testing. No
stable community-submission REST endpoint was found during this execution.

Community submission therefore remains a browser/UI follow-up, not a completed
API action.

Relevant public documentation checked:

- `https://help.zenodo.org/docs/share/submit-to-community/`
- `https://developers.zenodo.org/`

## Claim Boundary

This execution refreshed the public Zenodo software metadata. It did not:

- publish a new repository archive version;
- change files attached to the Zenodo record;
- complete community inclusion;
- assert any claim beyond the software and artefact citation surface.

The community-submission follow-up should be closed only after Zenodo shows the
record in the target community or a pending community-review request is
documented.
