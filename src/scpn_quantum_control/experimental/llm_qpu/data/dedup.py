# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — source-family leakage guard
"""Check source and prompt identity before any group-safe split is admitted."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence

from .tasks import MemoryRecord


def normalized_source_digest(text: str) -> str:
    """Hash casefolded, whitespace-normalized source text for duplicate checks."""
    if type(text) is not str or not text:
        raise ValueError("source text must be nonempty")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_source_families(records: Sequence[MemoryRecord]) -> None:
    """Refuse duplicate samples or source identities assigned to different groups."""
    if not records or len(records) > 8192:
        raise ValueError("memory record inventory out of bounds")
    samples: set[str] = set()
    source_groups: dict[str, str] = {}
    prompt_groups: dict[str, str] = {}
    for record in records:
        if type(record) is not MemoryRecord:
            raise ValueError("memory inventory contains an unvalidated record")
        if record.sample_id in samples:
            raise ValueError("duplicate memory sample ID")
        samples.add(record.sample_id)
        previous_group = source_groups.setdefault(record.source_id, record.group_id)
        if previous_group != record.group_id:
            raise ValueError("one source appears in multiple groups")
        digest = normalized_source_digest(record.prefix_text)
        previous_prompt_group = prompt_groups.setdefault(digest, record.group_id)
        if previous_prompt_group != record.group_id:
            raise ValueError("normalized prompt duplicated across source groups")
