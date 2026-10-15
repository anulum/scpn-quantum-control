# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — read-only provider observation export
"""Project original durable provider evidence for Studio without a provider call."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import cast

from ..hardware.provider_job_journal import ProviderJobJournal
from ..studio_workspace.canonical import canonical_digest
from ..studio_workspace.json_transport import write_json

PROVIDER_OBSERVATION_SCHEMA = "studio.provider-observation.v1"
"""Schema domain for a detached observation, not an execution request."""
MAX_OBSERVATION_BYTES = 4 * 1024 * 1024
"""Maximum UTF-8 export size; larger original evidence stays in its journal."""


def export_provider_observation(journal: ProviderJobJournal, attempt_id: str) -> str:
    """Export exact request identity and ordered observations without submission.

    Parameters
    ----------
    journal
        Existing original SQLite journal. Its payload integrity is checked by
        the journal owner; export does not modify the request or event history.
    attempt_id
        Existing attempt UUID. An unknown submission stays unknown and requires
        explicit reconciliation through the original runner API.

    Returns
    -------
    str
        Lossless JSON with the original QPY bytes encoded as base64, original
        event values and a domain-separated canonical digest. Provider state is
        the latest explicit native status, or null when none was retained.
        Local lifecycle state and unknown billing remain distinct. The digest
        proves content identity, not provider authenticity or live execution.

    Raises
    ------
    KeyError
        If the attempt does not exist.
    ValueError
        If stored integrity, a status value, canonical encoding or the four
        mebibyte export budget fails. Original journal evidence remains intact.

    """
    snapshot = journal.snapshot(attempt_id)
    events = cast(list[dict[str, object]], snapshot["events"])
    provider_state: str | None = None
    for event in events:
        observation = cast(Mapping[str, object], event["observation"])
        if "provider_status" in observation:
            status = observation["provider_status"]
            if not isinstance(status, str) or not 0 < len(status) <= 1024:
                raise ValueError("retained provider status must be nonempty bounded text")
            provider_state = status
    body = {
        "journal_schema": snapshot["schema"],
        "attempt_id": snapshot["attempt_id"],
        "target": snapshot["target"],
        "payload_sha256": snapshot["payload_sha256"],
        "payload_base64": base64.b64encode(cast(bytes, snapshot["payload"])).decode("ascii"),
        "shots": snapshot["shots"],
        "circuits": snapshot["circuits"],
        "experiment": snapshot["experiment"],
        "local_state": snapshot["state"],
        "provider_job_id": snapshot["provider_job_id"],
        "provider_state": provider_state,
        "billing": snapshot["billing"],
        "events": events,
        "no_submit": True,
    }
    unit: dict[str, object] = {
        "schema": PROVIDER_OBSERVATION_SCHEMA,
        "body": body,
        "extensions": {},
    }
    unit["sha256"] = canonical_digest(PROVIDER_OBSERVATION_SCHEMA, unit)
    text = write_json(unit)
    if len(text.encode("utf-8")) > MAX_OBSERVATION_BYTES:
        raise ValueError("provider observation exceeds the four mebibyte export budget")
    return text
