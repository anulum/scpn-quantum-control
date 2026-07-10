# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B QEC-readiness bundle emission
"""Federate the committed offline QEC-readiness decision as a schema-B bundle.

The Phase 3 multicircuit-QEC readiness artefact
(``data/phase3_multicircuit_qec/qec_readiness_2026-05-07.json``) records the
offline distance-3 surface-code decoder comparison: logical-failure aggregates
per decoder, family, label, and noise model, plus the promotion-or-reject
decision taken **before** any live backend submission. This module federates
that artefact as one ``studio.qec-readiness.v1`` bundle — an additive family
produced by the ``validate`` verb.

The mapping is deliberately honest. Every decoder aggregate rides in
``cases[]`` verbatim with its logical-failure rate as the case error and an
explicit ``simulated`` status (the aggregates are offline Monte-Carlo decoder
runs under modelled noise, never hardware results), and the bundle claim
boundary carries the artefact's own supported/blocked lists verbatim — fault
tolerance, scalable QEC, and hardware logical-error reduction stay blocked.
An artefact missing its aggregates, its distance, or its blocked list is
refused outright.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    NumericProvenance,
    ProvEntity,
    Substrate,
    ValidityDomain,
)

from ..differentiable_claim_ledger import REPO_ROOT
from .evidence_bundle import (
    DEFAULT_ACTIVITY_TIMESTAMP,
    _committed_artifact_edge,
    _digest_payload,
    _studio_activity,
    _studio_agent,
    validate_bundle,
)
from .verbs import QEC_READINESS_SCHEMA, STUDIO_ID, VALIDATE

DEFAULT_QEC_READINESS_ARTIFACT_PATH = Path(
    "data/phase3_multicircuit_qec/qec_readiness_2026-05-07.json"
)
"""Repository-relative path of the committed QEC readiness artefact."""

QEC_READINESS_ARTIFACT_ID = "phase3-multicircuit-qec-readiness-2026-05-07"

_REGENERATED_BY = "python -m scpn_quantum_control.studio.qec_readiness_bundle"


def _load_readiness(artifact_path: Path) -> dict[str, Any]:
    """Load and shape-check the committed QEC readiness artefact, failing closed.

    Parameters
    ----------
    artifact_path
        Committed artefact path (relative paths resolve against the repo root).

    Returns
    -------
    dict[str, Any]
        The parsed readiness payload.

    Raises
    ------
    ValueError
        If the artefact is missing its decoder aggregates, its code distance,
        its readiness decision, or its blocked-claims honesty list.
    """
    resolved = artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path
    payload: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    aggregates = payload.get("decoder_aggregates")
    if not isinstance(aggregates, list) or not aggregates:
        raise ValueError(
            f"QEC readiness artefact has no decoder aggregates: {resolved.as_posix()}"
        )
    distance = payload.get("distance")
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 1:
        raise ValueError("QEC readiness artefact is missing its code distance")
    if not isinstance(payload.get("readiness_decision"), str) or not payload["readiness_decision"]:
        raise ValueError("QEC readiness artefact is missing its readiness decision")
    boundary = payload.get("claim_boundary")
    if (
        not isinstance(boundary, dict)
        or not isinstance(boundary.get("blocked"), list)
        or not boundary["blocked"]
        or not isinstance(boundary.get("supported"), list)
        or not boundary["supported"]
    ):
        raise ValueError("QEC readiness artefact is missing its supported/blocked claim boundary")
    return payload


def build_qec_readiness_bundle(
    *,
    artifact_path: Path = DEFAULT_QEC_READINESS_ARTIFACT_PATH,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build the schema-B ``studio.qec-readiness.v1`` bundle.

    Parameters
    ----------
    artifact_path
        Committed QEC readiness artefact; its content digest rides as a
        ``derived_from`` edge (a missing path fails closed).
    activity_timestamp
        Deterministic PROV timestamp for the emission.

    Returns
    -------
    EvidenceBundle
        The ``studio.qec-readiness.v1`` bundle, decoder logical-failure
        aggregates verbatim in ``cases[]`` at the artefact's code distance.

    Raises
    ------
    ValueError
        If the artefact fails its shape check.
    """
    payload = _load_readiness(artifact_path)
    aggregates: list[dict[str, Any]] = payload["decoder_aggregates"]
    boundary = payload["claim_boundary"]
    note = (
        f"decision: {payload['readiness_decision']}; "
        f"supported: {'; '.join(boundary['supported'])}; "
        f"blocked: {'; '.join(boundary['blocked'])}"
    )
    return EvidenceBundle(
        schema=QEC_READINESS_SCHEMA,
        entity=ProvEntity(
            entity_id=f"{STUDIO_ID}:qec-readiness:{QEC_READINESS_ARTIFACT_ID}",
            digest=_digest_payload(payload),
        ),
        activity=_studio_activity(
            verb=VALIDATE.name,
            command=_REGENERATED_BY,
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=note),
        ),
        substrate=Substrate.NUMERICAL_MODEL,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend="offline-surface-code-decoders",
            reference_backend="unencoded-physical",
        ),
        cases=tuple(
            CaseResult(
                operation_family=(
                    f"qec-decoder:{aggregate['family']}:{aggregate['label']}"
                    f":{aggregate['decoder']}:{aggregate['noise_model']}"
                ),
                dimension=int(payload["distance"]),
                status="simulated",
                error=float(aggregate["logical_failure_rate"]),
            )
            for aggregate in aggregates
        ),
        derived_from=(
            _committed_artifact_edge(
                artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path,
                label="QEC readiness artefact",
            ),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit the validated QEC-readiness bundle as JSON.

    Parameters
    ----------
    argv
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` when the bundle is admitted by the federation gate, ``1``
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_QEC_READINESS_ARTIFACT_PATH,
        help="committed QEC readiness artefact for the derivation edge",
    )
    args = parser.parse_args(argv)
    validated = validate_bundle(build_qec_readiness_bundle(artifact_path=args.artifact_path))
    print(json.dumps(validated.bundle.to_dict(), indent=2, sort_keys=True))
    if not validated.verdict.admitted:
        print(
            "QEC-readiness bundle rejected: " + "; ".join(validated.verdict.rejections),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "DEFAULT_QEC_READINESS_ARTIFACT_PATH",
    "QEC_READINESS_ARTIFACT_ID",
    "build_qec_readiness_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
