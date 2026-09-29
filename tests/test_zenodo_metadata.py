# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Zenodo archive metadata
"""Zenodo archive metadata must use values the Zenodo vocabularies accept.

The GitHub release integration rejects a ``.zenodo.json`` whose related
identifiers name a relation outside the Zenodo relation-type vocabulary, and
no archive version is created for that release. ``isRelatedTo`` and
``isAlternateIdentifier`` are DataCite relations that Zenodo does not accept.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_FILES = (REPO_ROOT / ".zenodo.json", REPO_ROOT / "oscillatools" / ".zenodo.json")

# Zenodo relation-type vocabulary (``/api/vocabularies/relationtypes``), 2026-09-29.
ZENODO_RELATION_TYPES = frozenset(
    {
        "cites",
        "compiles",
        "continues",
        "describes",
        "documents",
        "hasmetadata",
        "haspart",
        "hasversion",
        "iscitedby",
        "iscompiledby",
        "iscontinuedby",
        "isderivedfrom",
        "isdescribedby",
        "isdocumentedby",
        "isidenticalto",
        "ismetadatafor",
        "isnewversionof",
        "isobsoletedby",
        "isoriginalformof",
        "ispartof",
        "ispreviousversionof",
        "ispublishedin",
        "isreferencedby",
        "isrequiredby",
        "isreviewedby",
        "issourceof",
        "issupplementedby",
        "issupplementto",
        "isvariantformof",
        "isversionof",
        "obsoletes",
        "references",
        "requires",
        "reviews",
    }
)


@pytest.mark.parametrize("path", ZENODO_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_zenodo_related_identifier_relations_are_in_zenodo_vocabulary(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    relations = [item["relation"] for item in metadata["related_identifiers"]]
    assert relations
    rejected = [
        relation for relation in relations if relation.lower() not in ZENODO_RELATION_TYPES
    ]
    assert rejected == []
