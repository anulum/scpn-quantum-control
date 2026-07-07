# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B baseline-scorecard bundle emission
"""Emit the differentiable baseline scorecard as a schema-B evidence bundle.

The scorecard (:mod:`scpn_quantum_control.differentiable_baseline_scorecard`) is the
governance surface of the differentiable lane: eleven external-baseline
categories, each with a verbatim status and explicit blockers. This module
federates that surface as one ``studio.differentiation-evidence.v1`` bundle —
the evidence family produced by the contract-reserved ``differentiate`` verb.

The mapping is deliberately honest. Category statuses ride in ``cases[]``
verbatim (today that is ``behind_baseline`` across the board, and that is what
the Hub renders); the bundle claim boundary is ``bounded-model`` with the
scorecard's own claim-boundary sentence as its validity note. Nothing here can
upgrade a category: promotion happens upstream in the scorecard/ledger, never
in the emitter. A scorecard that fails its own validation is refused outright.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    DerivedEdge,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    NumericProvenance,
    ProvEntity,
    Substrate,
    ValidityDomain,
)

from ..differentiable_baseline_scorecard import (
    DifferentiableBaselineScorecard,
    run_differentiable_baseline_scorecard,
    validate_differentiable_baseline_scorecard,
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
from .verbs import DIFFERENTIATE, DIFFERENTIATION_EVIDENCE_SCHEMA, STUDIO_ID

DEFAULT_SCORECARD_ARTIFACT_PATH = Path(
    "data/differentiable_phase_qnode/differentiable_baseline_scorecard_20260620.json"
)
"""Repository-relative path of the committed scorecard artefact."""

_REGENERATED_BY = "python -m scpn_quantum_control.studio.scorecard_bundle"


def _artifact_edge(artifact_path: Path) -> DerivedEdge:
    """Return the content-addressed derivation edge for the committed artefact.

    Parameters
    ----------
    artifact_path
        Path of the committed scorecard JSON artefact.

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
    return _committed_artifact_edge(artifact_path, label="scorecard artefact")


def build_scorecard_bundle(
    scorecard: DifferentiableBaselineScorecard | None = None,
    *,
    artifact_path: Path | None = None,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
    repo_root: Path = REPO_ROOT,
) -> EvidenceBundle:
    """Build the schema-B bundle for the differentiable baseline scorecard.

    Parameters
    ----------
    scorecard
        A scorecard to federate, or ``None`` to build the committed one via
        :func:`run_differentiable_baseline_scorecard`.
    artifact_path
        Optional committed-artefact path; when given, its content digest is
        attached as a ``derived_from`` edge (missing path fails closed).
    activity_timestamp
        Deterministic PROV timestamp for the source-tree emission.
    repo_root
        Repository root used by the scorecard's own path validation.

    Returns
    -------
    EvidenceBundle
        The ``studio.differentiation-evidence.v1`` bundle, category statuses
        verbatim in ``cases[]``.

    Raises
    ------
    ValueError
        If the scorecard fails :func:`validate_differentiable_baseline_scorecard`
        — an invalid scorecard is never federated.
    """
    resolved = scorecard if scorecard is not None else run_differentiable_baseline_scorecard()
    validation = validate_differentiable_baseline_scorecard(resolved, repo_root=repo_root)
    if not validation.passed:
        raise ValueError(
            "scorecard failed validation and is not federated: " + "; ".join(validation.errors)
        )
    payload = resolved.to_dict()
    edges = () if artifact_path is None else (_artifact_edge(artifact_path),)
    return EvidenceBundle(
        schema=DIFFERENTIATION_EVIDENCE_SCHEMA,
        entity=ProvEntity(
            entity_id=f"{STUDIO_ID}:baseline-scorecard:{resolved.artifact_id}",
            digest=_digest_payload(payload),
        ),
        activity=_studio_activity(
            verb=DIFFERENTIATE.name,
            command=_REGENERATED_BY,
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.TAXONOMY,
        evidence_kind=EvidenceKind.CURATED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=resolved.claim_boundary),
        ),
        substrate=Substrate.NUMERICAL_MODEL,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend="differentiable-baseline-scorecard",
            reference_backend="committed-ledger-json",
        ),
        cases=tuple(
            CaseResult(
                operation_family=f"baseline-category:{row.category}",
                dimension=index,
                status=row.status,
            )
            for index, row in enumerate(resolved.rows, start=1)
        ),
        derived_from=edges,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit the validated scorecard bundle as JSON.

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
        default=DEFAULT_SCORECARD_ARTIFACT_PATH,
        help="committed scorecard artefact for the derivation edge",
    )
    args = parser.parse_args(argv)
    validated = validate_bundle(build_scorecard_bundle(artifact_path=args.artifact_path))
    print(json.dumps(validated.bundle.to_dict(), indent=2, sort_keys=True))
    if not validated.verdict.admitted:
        print(
            "scorecard bundle rejected: " + "; ".join(validated.verdict.rejections),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "DEFAULT_SCORECARD_ARTIFACT_PATH",
    "build_scorecard_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
