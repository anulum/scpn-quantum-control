<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Zenodo Community Submission -->

# Zenodo Community Submission

Date checked: 2026-05-06

This note records the Zenodo community-submission state for the public
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

## Target Community

Community:

```text
Research Software Engineering
```

Community slug:

```text
rse
```

Community URL:

```text
https://zenodo.org/communities/rse/
```

Community identifier reported by the Zenodo API:

```text
5ca0f144-78b9-4807-8043-0a78215456f6
```

## Submission State

Zenodo reports one open request for the record:

| Field | Value |
|-------|-------|
| Request type | `community-inclusion` |
| Request status | `submitted` |
| Request number | `2cstt` |
| Request identifier | `8813fc98-3dd9-4443-99ab-fa63c85565e1` |
| Topic record | `18821930` |
| Receiver community | `5ca0f144-78b9-4807-8043-0a78215456f6` |

The request page is:

```text
https://zenodo.org/me/requests/8813fc98-3dd9-4443-99ab-fa63c85565e1
```

The request is pending curator action. The public Zenodo record should
therefore not be described as already accepted into the community until the
record's public community list changes or the request status changes to an
accepted/closed state.

## API and UI Boundary

Zenodo documents two distinct community workflows:

- `Submit for review` is for unpublished drafts.
- `Submit to community` is for already published records.

The stable Zenodo developer documentation lists deposit, records, and files as
the stable REST API surfaces, while community APIs are still described as being
in testing. During this check, the draft-review API rejected the published
record path with:

```text
You cannot create a review for an already published record.
```

The record-level request list then confirmed that the published-record
community-inclusion request exists and is submitted.

## Draft State

An authenticated draft view for record `18821930` was present after the
metadata/community workflow. A comparison between the public record core fields
and the draft core fields found no metadata, access, PID, or file-content
changes beyond the expected API file-content URL changing from the published
record path to the draft path.

The draft was not deleted or published during this check, because Zenodo's
published-record community-inclusion workflow grants curators edit access while
the request is pending. The safe operational rule is to leave the Zenodo draft
state untouched until the community request is accepted, declined, or manually
cancelled from the Zenodo request page.

## Claim Boundary

This check confirms:

- the software record metadata refresh is already public;
- a community-inclusion request for the `Research Software Engineering`
  community is submitted;
- community acceptance is still pending.

This check does not claim:

- accepted community membership;
- a new Zenodo archive version;
- any change to attached Zenodo files.
