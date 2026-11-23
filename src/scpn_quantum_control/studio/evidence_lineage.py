# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — evidence lineage from claims to their artefacts, packs and replications
"""Join claims, hardware result packs and replication analyses into one evidence lineage.

The lineage keeps every source-owned status beside the others and never merges
or relabels them:

- a claim carries its ledger ``promotion_status``, the schema-B bundle that
  :mod:`scpn_quantum_control.studio.evidence_bundle` emits for it, and its
  reference-validation certification, or the reason it has none. The registry
  is validated against the ledger first, and an invalid registry refuses the
  build;
- each evidence identifier a claim declares is resolved by ``artifact_id``
  against the committed data. It is *resolved* to exactly one file,
  *ambiguous* with every candidate listed, or *missing*; one candidate is never
  picked;
- a hardware result pack carries its manifest ``status`` verbatim next to its
  bundle, and each of its artefacts carries the recorded digest and the digest
  of the committed file;
- a declared replication analysis carries its decision field verbatim, for
  example ``negative_sign_replicates: false``.

Every record carries ``record_digest``, the shared canonical digest of the
record without that field, so a consumer can check that the record it shows is
the one this module built.

Bundles keep their verdict axes. The fields that describe the emitting host,
namely the activity host, the agent's Studio version and the recompute
toolchain's Python version, are left out, so the lineage is the same on every
host.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..differentiable_claim_ledger import REPO_ROOT, ClaimLedger, load_differentiable_claim_ledger
from ..hardware_result_packs import MANIFEST_RELATIVE_PATH, load_manifest
from ..studio_workspace.canonical import canonical_digest
from .evidence_bundle import build_claim_ledger_bundle, build_hardware_result_pack_bundle
from .reference_validation import (
    DEFAULT_REFERENCE_VALIDATION_PATH,
    ReferenceValidationRegistry,
    load_reference_validation_registry,
)

EVIDENCE_LINEAGE_SCHEMA: Final[str] = "scpn_qc_studio_evidence_lineage_v1"
"""Wire schema of the committed evidence lineage."""

LINEAGE_RECORD_SCHEMA: Final[str] = "scpn_qc_studio_evidence_lineage_record_v1"
"""Canonical domain of each record's ``record_digest``."""

ABSENT_CERTIFICATION_REASON: Final[str] = (
    "no reference-validation certification is committed for this claim"
)
"""Reason recorded for a claim without a certification."""


@dataclass(frozen=True, slots=True)
class ReplicationSource:
    """One declared replication analysis and where its decision is written.

    Parameters
    ----------
    path
        Repository-relative path of the committed analysis.
    section
        Top-level key of the replication record inside the analysis.
    decision
        Key of the boolean decision inside that record.

    """

    path: str
    section: str
    decision: str


REPLICATION_SOURCES: Final[tuple[ReplicationSource, ...]] = (
    ReplicationSource(
        "data/iqm_paper_replication/iqm_dla_d10_retest_primary_analysis_2026-07-22.json",
        "primary_sign_replication",
        "negative_sign_replicates",
    ),
)
"""Replication analyses whose decision the lineage reports."""

_HOST_FIELDS: Final[tuple[tuple[str, ...], ...]] = (
    ("prov", "activity", "host"),
    ("prov", "agent", "studio_version"),
    ("recompute_environment", "toolchain", "python"),
)


def sealed_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return a record with its ``record_digest`` added.

    Parameters
    ----------
    record
        A lineage record without ``record_digest``.

    Returns
    -------
    dict[str, object]
        The record and the canonical digest of it.

    """
    return {**record, "record_digest": canonical_digest(LINEAGE_RECORD_SCHEMA, dict(record))}


def _sha256(data: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    """Read one committed JSON object.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    dict[str, object]
        The parsed object.

    Raises
    ------
    ValueError
        The file does not hold a JSON object.

    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return document


def portable_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    """Return a bundle without the fields that describe the emitting host.

    Parameters
    ----------
    bundle
        A schema-B bundle as ``EvidenceBundle.to_dict`` returns it.

    Returns
    -------
    dict[str, object]
        A deep copy without the activity host, the agent's Studio version and the
        recompute toolchain's Python version. Every verdict axis is kept.

    """
    copy = json.loads(json.dumps(bundle))
    for *parents, leaf in _HOST_FIELDS:
        holder = copy
        for key in parents:
            holder = holder.get(key) if isinstance(holder, dict) else None
        if isinstance(holder, dict):
            holder.pop(leaf, None)
    return dict(copy)


def artifact_index(root: Path) -> dict[str, list[dict[str, str]]]:
    """Map every committed ``artifact_id`` under ``data`` to the files that declare it.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        For each identifier, its files in path order, each with its digest.

    """
    index: dict[str, list[dict[str, str]]] = {}
    for path in sorted((root / "data").rglob("*.json")):
        data = path.read_bytes()
        try:
            document = json.loads(data)
        except ValueError:
            continue
        identifier = document.get("artifact_id") if isinstance(document, dict) else None
        if isinstance(identifier, str) and identifier:
            index.setdefault(identifier, []).append(
                {"path": path.relative_to(root).as_posix(), "sha256": _sha256(data)}
            )
    return index


def _resolution(
    identifier: str, role: str, index: Mapping[str, list[dict[str, str]]]
) -> dict[str, object]:
    """Resolve one declared evidence identifier without choosing among candidates.

    Parameters
    ----------
    identifier
        The declared ``artifact_id``.
    role
        ``"evidence"`` or ``"benchmark"``, the ledger field that declared it.
    index
        The committed artefact index.

    Returns
    -------
    dict[str, object]
        The identifier, its role, its state and every candidate file.

    """
    candidates = index.get(identifier, [])
    state = "missing" if not candidates else "resolved" if len(candidates) == 1 else "ambiguous"
    return {"artifact_id": identifier, "role": role, "state": state, "candidates": candidates}


def _claims(
    ledger: ClaimLedger,
    registry: ReferenceValidationRegistry,
    index: Mapping[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Return one lineage record per ledger claim, in ledger order.

    Parameters
    ----------
    ledger
        The committed claim ledger.
    registry
        The reference-validation registry.
    index
        The committed artefact index.

    Returns
    -------
    list[dict[str, object]]
        Claim records with their bundle, certification and resolved evidence.

    Raises
    ------
    ValueError
        The registry does not validate against the ledger.

    """
    certified = set(registry.reference_validated_claim_ids(ledger))
    certificates = {item.claim_id: item.to_dict() for item in registry.certifications}
    records: list[dict[str, object]] = []
    for row in ledger.rows:
        bundle = build_claim_ledger_bundle(row, reference_validated=row.claim_id in certified)
        certification: dict[str, object] = (
            {"state": "certified", "record": certificates[row.claim_id]}
            if row.claim_id in certified
            else {"state": "absent", "reason": ABSENT_CERTIFICATION_REASON}
        )
        evidence = [_resolution(item, "evidence", index) for item in row.evidence_artifact_ids]
        evidence += [_resolution(item, "benchmark", index) for item in row.benchmark_artifact_ids]
        records.append(
            sealed_record(
                {
                    "claim_id": row.claim_id,
                    "claim_text": row.claim_text,
                    "claim_boundary": row.claim_boundary,
                    "promotion_status": row.promotion_status,
                    "bundle": portable_bundle(bundle.to_dict()),
                    "certification": certification,
                    "evidence": evidence,
                }
            )
        )
    return records


def _custody(root: Path, artefact: Mapping[str, object]) -> str:
    """Compare a pack artefact's recorded digest with the committed file.

    Parameters
    ----------
    root
        Repository root.
    artefact
        One artefact record of a pack.

    Returns
    -------
    str
        ``"digest-matches"``, ``"digest-differs"`` or ``"missing"``.

    """
    path = root / str(artefact.get("path", ""))
    if not path.is_file():
        return "missing"
    return (
        "digest-matches"
        if _sha256(path.read_bytes()) == artefact.get("sha256")
        else "digest-differs"
    )


def _packs(root: Path) -> list[dict[str, object]]:
    """Return one lineage record per committed hardware result pack.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    list[dict[str, object]]
        Pack records with the manifest status, the bundle and artefact custody.

    """
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    records: list[dict[str, object]] = []
    for pack in manifest["packs"]:
        artefacts = [
            {
                "path": item["path"],
                "role": item["role"],
                "recorded_sha256": item["sha256"],
                "bytes": item["bytes"],
                "custody": _custody(root, item),
            }
            for item in pack["artifacts"]
        ]
        records.append(
            sealed_record(
                {
                    "id": pack["id"],
                    "title": pack["title"],
                    "manifest_status": pack["status"],
                    "claim_scope": pack["claim_scope"],
                    "non_claims": list(pack["non_claims"]),
                    "job_ids": list(pack["required_job_ids"]),
                    "bundle": portable_bundle(build_hardware_result_pack_bundle(pack).to_dict()),
                    "artefacts": artefacts,
                }
            )
        )
    return records


def _replications(root: Path, sources: Sequence[ReplicationSource]) -> list[dict[str, object]]:
    """Return one record per declared replication analysis, its decision verbatim.

    Parameters
    ----------
    root
        Repository root.
    sources
        The declared replication analyses.

    Returns
    -------
    list[dict[str, object]]
        Replication records with their decision, test level and job identifiers.

    Raises
    ------
    ValueError
        A declared analysis lacks its replication record or its boolean decision.

    """
    records: list[dict[str, object]] = []
    for source in sources:
        path = root / source.path
        document = _json_object(path)
        section = document.get(source.section)
        value = section.get(source.decision) if isinstance(section, dict) else None
        if not isinstance(section, dict) or not isinstance(value, bool):
            raise ValueError(
                f"{source.path}: {source.section}.{source.decision} must be a boolean"
            )
        records.append(
            sealed_record(
                {
                    "path": source.path,
                    "sha256": _sha256(path.read_bytes()),
                    "campaign": document.get("campaign"),
                    "kind": document.get("kind"),
                    "section": source.section,
                    "decision": {"field": source.decision, "value": value},
                    "alpha": section.get("alpha"),
                    "job_ids": document.get("job_ids"),
                    "interpretation_boundary": document.get("interpretation_boundary"),
                }
            )
        )
    return records


def build_evidence_lineage(
    root: Path = REPO_ROOT,
    *,
    replication_sources: Sequence[ReplicationSource] = REPLICATION_SOURCES,
) -> dict[str, object]:
    """Build the evidence lineage of the committed claims, packs and replications.

    Parameters
    ----------
    root
        Repository root to read the committed sources from.
    replication_sources
        Replication analyses to report.

    Returns
    -------
    dict[str, object]
        ``{"schema", "claims", "packs", "replications"}``, ready for JSON.

    Raises
    ------
    ValueError
        The certification registry is invalid, or a declared replication analysis
        lacks its decision.

    """
    ledger_path = root / "data/differentiable_phase_qnode/claim_ledger.json"
    registry_path = root / DEFAULT_REFERENCE_VALIDATION_PATH.relative_to(REPO_ROOT)
    index = artifact_index(root)
    return {
        "schema": EVIDENCE_LINEAGE_SCHEMA,
        "claims": _claims(
            load_differentiable_claim_ledger(ledger_path),
            load_reference_validation_registry(registry_path),
            index,
        ),
        "packs": _packs(root),
        "replications": _replications(root, replication_sources),
    }


__all__ = [
    "ABSENT_CERTIFICATION_REASON",
    "EVIDENCE_LINEAGE_SCHEMA",
    "LINEAGE_RECORD_SCHEMA",
    "REPLICATION_SOURCES",
    "ReplicationSource",
    "artifact_index",
    "build_evidence_lineage",
    "portable_bundle",
    "sealed_record",
]
