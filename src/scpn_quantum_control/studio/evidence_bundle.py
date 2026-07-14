# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B EvidenceBundle emission
"""Emit schema-B STUDIO evidence bundles from QUANTUM ledgers.

The schema-A manifest advertises what the QUANTUM studio can produce. This module
builds the load-bearing schema-B objects: concrete
``scpn_studio_platform.evidence.EvidenceBundle`` records for the committed
differentiable claim ledger and hardware result-pack manifest.

The mapping is intentionally conservative. Differentiable ledger rows become
curated, bounded model bundles unless a separate reference-validation certificate
is supplied. Hardware result packs become measured hardware-unmitigated bundles;
they do not become hardware-validated or reference-validated solely because raw
counts exist. Falsified rows must use the platform's ``falsified``/``refuted``
pair, preserving the LOCK-4 rule that negative findings are explicit and never
laundered into validation.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    BlockedOn,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    DerivedEdge,
    DerivedKind,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    FederationVerdict,
    Freshness,
    NumericProvenance,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    RecomputeEnvironment,
    Substrate,
    ValidityDomain,
    validate_studio_bundle,
)

from ..differentiable_claim_ledger import (
    ClaimLedger,
    ClaimLedgerRow,
    PromotionStatus,
    load_differentiable_claim_ledger,
)
from ..hardware_result_packs import MANIFEST_RELATIVE_PATH, load_manifest
from .coverage_frontier import map_claim_status
from .manifest import STUDIO_VERSION
from .verbs import HARDWARE_RESULT_PACK_SCHEMA, STUDIO_ID, VALIDATE

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
EvidenceSource = Literal[
    "theory",
    "simulator",
    "hardware-unmitigated",
    "hardware-mitigated",
    "falsification",
    "noise-floor",
]

DEFAULT_ACTIVITY_TIMESTAMP = "2026-06-26T00:00:00Z"
"""Stable timestamp for deterministic bundles built from committed artefacts."""


@dataclass(frozen=True, slots=True)
class StudioBundleValidation:
    """Validation result for one emitted schema-B bundle.

    Parameters
    ----------
    bundle
        The in-memory platform bundle.
    verdict
        The platform federation verdict for ``bundle.to_dict()``.
    """

    bundle: EvidenceBundle
    verdict: FederationVerdict

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready validation summary."""
        return {
            "schema": self.bundle.schema,
            "entity_id": self.bundle.entity.entity_id,
            "admitted": self.verdict.admitted,
            "mode": self.verdict.mode,
            "rejections": list(self.verdict.rejections),
            "unresolved_upstreams": list(self.verdict.unresolved_upstreams),
        }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic JSON bytes for digesting bundle inputs."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _digest_payload(payload: Mapping[str, Any]) -> str:
    """Return a ``sha256:<hex>`` digest for a JSON-like mapping."""
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _studio_activity(*, verb: str, command: str, timestamp: str) -> ProvActivity:
    """Build the deterministic PROV activity common to emitted bundles."""
    return ProvActivity(
        verb=verb,
        studio=STUDIO_ID,
        started=timestamp,
        ended=timestamp,
        regenerated_by=command,
        host=platform.platform(),
    )


def _studio_agent(*, operator: str = "scpn-quantum-control/studio-emitter") -> ProvAgent:
    """Return the PROV agent for local source-tree bundle generation."""
    return ProvAgent(studio_version=STUDIO_VERSION, operator=operator)


def _committed_artifact_edge(artifact_path: Path, *, label: str) -> DerivedEdge:
    """Return the content-addressed derivation edge for a committed artefact.

    Parameters
    ----------
    artifact_path
        Path of the committed artefact file to content-address.
    label
        Human-readable artefact label used in the fail-closed error message.

    Returns
    -------
    DerivedEdge
        Evidence edge carrying the SHA-256 of the artefact bytes.

    Raises
    ------
    ValueError
        If the artefact path does not exist — a requested derivation edge that
        cannot be content-addressed fails closed instead of being dropped.
    """
    if not artifact_path.exists():
        raise ValueError(f"{label} does not exist: {artifact_path.as_posix()}")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return DerivedEdge(
        kind=DerivedKind.EVIDENCE,
        studio=STUDIO_ID,
        entity_digest=f"sha256:{digest}",
    )


def evidence_axes(source: EvidenceSource) -> tuple[EvidenceKind, Substrate]:
    """Map QUANTUM evidence-source classes onto STUDIO kind and substrate axes.

    Parameters
    ----------
    source
        The QUANTUM source class being emitted.

    Returns
    -------
    tuple[EvidenceKind, Substrate]
        The platform evidence kind and execution substrate.

    Raises
    ------
    ValueError
        If ``source`` is not part of the declared QUANTUM mapping.
    """
    if source == "theory":
        return EvidenceKind.CURATED, Substrate.CLASSICAL_REFERENCE
    if source == "simulator":
        return EvidenceKind.MEASURED, Substrate.SIMULATOR
    if source == "hardware-unmitigated":
        return EvidenceKind.MEASURED, Substrate.HARDWARE_UNMITIGATED
    if source == "hardware-mitigated":
        return EvidenceKind.HARDWARE_VALIDATED, Substrate.HARDWARE_MITIGATED
    if source == "falsification":
        return EvidenceKind.FALSIFIED, Substrate.NUMERICAL_MODEL
    if source == "noise-floor":
        return EvidenceKind.NOISE_LIMITED, Substrate.HARDWARE_UNMITIGATED
    raise ValueError(f"unknown evidence source: {source!r}")


def _promotion_boundary(
    status: PromotionStatus,
    *,
    claim_boundary: str,
    reference_validated: bool,
    blocked_on: Sequence[str],
) -> ClaimBoundary:
    """Build the platform claim boundary for a differentiable ledger row."""
    claim_status = map_claim_status(status, reference_validated=reference_validated)
    if claim_status is ClaimStatus.EXTERNAL_DEPENDENCY_BLOCKED:
        blockers = tuple(
            BlockedOn(dependency=blocker, kind="evidence") for blocker in blocked_on
        ) or (BlockedOn(dependency="unresolved external evidence", kind="evidence"),)
        return ClaimBoundary(
            status=claim_status,
            admission=AdmissionDecision.REJECTED,
            blocked_on=blockers,
        )
    admission = (
        AdmissionDecision.ADMITTED
        if claim_status in {ClaimStatus.BOUNDED_MODEL, ClaimStatus.REFERENCE_VALIDATED}
        else AdmissionDecision.REJECTED
    )
    domain = (
        ValidityDomain(note=claim_boundary)
        if claim_status in {ClaimStatus.BOUNDED_MODEL, ClaimStatus.REFERENCE_VALIDATED}
        else None
    )
    return ClaimBoundary(status=claim_status, admission=admission, validity_domain=domain)


def build_claim_ledger_bundle(
    row: ClaimLedgerRow,
    *,
    reference_validated: bool = False,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build a schema-B bundle for one differentiable claim-ledger row.

    Parameters
    ----------
    row
        A committed differentiable claim-ledger row.
    reference_validated
        Whether an external WS-3 process has attached reference-validation
        evidence to this promoted row.
    activity_timestamp
        Deterministic PROV timestamp for the source-tree emission.

    Returns
    -------
    EvidenceBundle
        The validated platform dataclass. Call :func:`validate_bundle` for the
        federation gate verdict.
    """
    row_payload = row.to_dict()
    digest = _digest_payload(row_payload)
    status = row.promotion_status
    evidence_kind = EvidenceKind.CURATED
    substrate = Substrate.NUMERICAL_MODEL
    return EvidenceBundle(
        schema="studio.evidence-replay.v1",
        entity=ProvEntity(entity_id=f"{STUDIO_ID}:claim-ledger:{row.claim_id}", digest=digest),
        activity=_studio_activity(
            verb=VALIDATE.name,
            command="python -m scpn_quantum_control.studio.evidence_bundle --claim-ledger",
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.TAXONOMY,
        evidence_kind=evidence_kind,
        claim_boundary=_promotion_boundary(
            status,
            claim_boundary=row.claim_boundary,
            reference_validated=reference_validated,
            blocked_on=row.known_gaps,
        ),
        substrate=substrate,
        freshness=(
            Freshness.VERIFIED_AT_SOURCE if reference_validated else Freshness.TRACEABLE_UNCHECKED
        ),
        numeric_provenance=NumericProvenance(
            active_backend="differentiable-claim-ledger",
            reference_backend="committed-ledger-json",
        ),
        cases=tuple(
            CaseResult(
                operation_family="claim-ledger-surface",
                dimension=index,
                status="present",
            )
            for index, _ in enumerate(
                (
                    *row.implementation_surface,
                    *row.test_surface,
                    *row.docs_surface,
                ),
                start=1,
            )
        ),
    )


def build_claim_ledger_bundles(
    rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow] | None = None,
    *,
    reference_validated_claim_ids: Iterable[str] = (),
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> tuple[EvidenceBundle, ...]:
    """Build schema-B bundles for the differentiable claim ledger.

    Parameters
    ----------
    rows_or_ledger
        A loaded ledger, explicit rows, or ``None`` to load the committed ledger.
    reference_validated_claim_ids
        Claim IDs certified by the external reference-validation process.
    activity_timestamp
        Deterministic PROV timestamp for the source-tree emission.

    Returns
    -------
    tuple[EvidenceBundle, ...]
        One bundle per claim row.
    """
    ledger_rows = (
        load_differentiable_claim_ledger().rows
        if rows_or_ledger is None
        else rows_or_ledger.rows
        if isinstance(rows_or_ledger, ClaimLedger)
        else tuple(rows_or_ledger)
    )
    certified = frozenset(reference_validated_claim_ids)
    return tuple(
        build_claim_ledger_bundle(
            row,
            reference_validated=row.claim_id in certified,
            activity_timestamp=activity_timestamp,
        )
        for row in ledger_rows
    )


def _pack_artifact_digest_edges(pack: Mapping[str, Any]) -> tuple[DerivedEdge, ...]:
    """Return content-addressed derivation edges for pack artifacts."""
    artifacts = pack.get("artifacts", ())
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, str | bytes):
        return ()
    edges: list[DerivedEdge] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        digest = artifact.get("sha256")
        if isinstance(digest, str) and digest:
            edges.append(
                DerivedEdge(
                    kind=DerivedKind.EVIDENCE,
                    studio=STUDIO_ID,
                    entity_digest=f"sha256:{digest}",
                )
            )
    return tuple(edges)


def _pack_cases(pack: Mapping[str, Any]) -> tuple[CaseResult, ...]:
    """Return per-artifact case rows for a hardware result-pack bundle."""
    artifacts = pack.get("artifacts", ())
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, str | bytes):
        return ()
    cases: list[CaseResult] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        role = str(artifact.get("role", "artifact"))
        size = artifact.get("bytes", 0)
        dimension = size if isinstance(size, int) and not isinstance(size, bool) else 0
        cases.append(
            CaseResult(
                operation_family=f"hardware-result-pack:{role}",
                dimension=dimension,
                status=ClaimStatus.BOUNDED_SUPPORT.value,
            )
        )
    return tuple(cases)


def build_hardware_result_pack_bundle(
    pack: Mapping[str, Any],
    *,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build a schema-B bundle for one committed hardware result pack.

    Parameters
    ----------
    pack
        One pack record from ``data/hardware_result_packs/manifest.json``.
    activity_timestamp
        Deterministic PROV timestamp for the source-tree emission.

    Returns
    -------
    EvidenceBundle
        The platform schema-B bundle for the pack.

    Raises
    ------
    ValueError
        If the pack lacks an ``id``.
    """
    pack_id = str(pack.get("id", "")).strip()
    if not pack_id:
        raise ValueError("hardware result pack must carry a non-empty id")
    evidence_kind, substrate = evidence_axes("hardware-unmitigated")
    command = str(pack.get("reproduce_command", "")).strip()
    return EvidenceBundle(
        schema=HARDWARE_RESULT_PACK_SCHEMA,
        entity=ProvEntity(
            entity_id=f"{STUDIO_ID}:hardware-result-pack:{pack_id}", digest=_digest_payload(pack)
        ),
        activity=_studio_activity(
            verb="execute",
            command=command or "python scripts/verify_hardware_result_packs.py --json",
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=evidence_kind,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_SUPPORT,
            admission=AdmissionDecision.ADMITTED,
        ),
        substrate=substrate,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend=str(pack.get("backend", "hardware-result-pack")),
            reference_backend="committed-raw-counts",
        ),
        cases=_pack_cases(pack),
        recompute_environment=RecomputeEnvironment(
            toolchain={"python": platform.python_version(), "command": command or "verify"},
            available=True,
        ),
        derived_from=_pack_artifact_digest_edges(pack),
    )


def build_hardware_result_pack_bundles(
    *,
    manifest_path: Path | None = None,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> tuple[EvidenceBundle, ...]:
    """Build schema-B bundles for all committed hardware result packs.

    Parameters
    ----------
    manifest_path
        Optional manifest path. Defaults to the repository result-pack manifest.
    activity_timestamp
        Deterministic PROV timestamp for the source-tree emission.

    Returns
    -------
    tuple[EvidenceBundle, ...]
        One bundle per manifest pack.
    """
    path = manifest_path or Path.cwd() / MANIFEST_RELATIVE_PATH
    manifest = load_manifest(path)
    packs = cast(list[object], manifest["packs"])
    return tuple(
        build_hardware_result_pack_bundle(pack, activity_timestamp=activity_timestamp)
        for pack in packs
        if isinstance(pack, Mapping)
    )


def validate_bundle(bundle: EvidenceBundle) -> StudioBundleValidation:
    """Validate a bundle through the platform federation gate.

    Parameters
    ----------
    bundle
        The platform dataclass to validate in wire form.

    Returns
    -------
    StudioBundleValidation
        The bundle plus its federation verdict.
    """
    return StudioBundleValidation(
        bundle=bundle,
        verdict=validate_studio_bundle(bundle.to_dict()),
    )


def validate_bundles(bundles: Iterable[EvidenceBundle]) -> tuple[StudioBundleValidation, ...]:
    """Validate several schema-B bundles through the platform gate.

    Parameters
    ----------
    bundles
        Bundles to validate.

    Returns
    -------
    tuple[StudioBundleValidation, ...]
        Validation summaries in input order.
    """
    return tuple(validate_bundle(bundle) for bundle in bundles)


__all__ = [
    "DEFAULT_ACTIVITY_TIMESTAMP",
    "EvidenceSource",
    "StudioBundleValidation",
    "build_claim_ledger_bundle",
    "build_claim_ledger_bundles",
    "build_hardware_result_pack_bundle",
    "build_hardware_result_pack_bundles",
    "evidence_axes",
    "validate_bundle",
    "validate_bundles",
]
