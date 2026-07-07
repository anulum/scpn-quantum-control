# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B support-matrix bundle emission
"""Emit the transform-algebra support matrix as a schema-B evidence bundle.

The support matrix is generated from the executable transform-algebra audit
(:mod:`scpn_quantum_control.differentiable_transform_algebra`), so every row
mirrors an executed or fail-closed audit case instead of a hand-written
capability claim. This module federates that generated matrix as one
``studio.differentiation-evidence.v1`` bundle — the evidence family produced by
the contract-reserved ``differentiate`` verb.

The mapping is deliberately honest. Support rows ride in ``cases[]`` verbatim:
``passed`` rows carry their measured residual as the case error, ``blocked``
rows stay explicit fail-closed boundaries with no residual. The bundle claim
boundary is ``bounded-model`` with the support matrix's own claim-boundary
sentence as its validity note; the emitter can never upgrade a row, and an
audit that did not pass is never federated.
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
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    NumericProvenance,
    ProvEntity,
    Substrate,
    ValidityDomain,
)

from ..differentiable_transform_algebra import (
    TransformAlgebraAudit,
    run_transform_algebra_audit,
)
from ..differentiable_transform_support_matrix import (
    TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY,
)
from ..differentiable_transform_support_matrix_artifact import (
    DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH,
    TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID,
    build_transform_support_matrix_artifact,
)
from .evidence_bundle import (
    DEFAULT_ACTIVITY_TIMESTAMP,
    _committed_artifact_edge,
    _digest_payload,
    _studio_activity,
    _studio_agent,
    validate_bundle,
)
from .verbs import DIFFERENTIATE, DIFFERENTIATION_EVIDENCE_SCHEMA, STUDIO_ID

DEFAULT_SUPPORT_MATRIX_ARTIFACT_PATH = DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH
"""Repository-relative path of the committed support-matrix artefact."""

_REGENERATED_BY = "python -m scpn_quantum_control.studio.support_matrix_bundle"


def build_support_matrix_bundle(
    audit: TransformAlgebraAudit | None = None,
    *,
    artifact_path: Path | None = None,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build the schema-B bundle for the transform-algebra support matrix.

    Parameters
    ----------
    audit
        A transform-algebra audit to federate, or ``None`` to run
        :func:`run_transform_algebra_audit` on the current tree.
    artifact_path
        Optional committed-artefact path; when given, its content digest is
        attached as a ``derived_from`` edge (missing path fails closed).
    activity_timestamp
        Deterministic PROV timestamp for the source-tree emission.

    Returns
    -------
    EvidenceBundle
        The ``studio.differentiation-evidence.v1`` bundle, support rows
        verbatim in ``cases[]`` with measured residuals as case errors.

    Raises
    ------
    ValueError
        If the audit did not pass — failed cases, missing categories, or
        missing support rows are never federated.
    """
    resolved = run_transform_algebra_audit() if audit is None else audit
    payload = build_transform_support_matrix_artifact(resolved)
    edges = (
        ()
        if artifact_path is None
        else (_committed_artifact_edge(artifact_path, label="support-matrix artefact"),)
    )
    return EvidenceBundle(
        schema=DIFFERENTIATION_EVIDENCE_SCHEMA,
        entity=ProvEntity(
            entity_id=(
                f"{STUDIO_ID}:transform-support-matrix:{TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID}"
            ),
            digest=_digest_payload(payload),
        ),
        activity=_studio_activity(
            verb=DIFFERENTIATE.name,
            command=_REGENERATED_BY,
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY),
        ),
        substrate=Substrate.NUMERICAL_MODEL,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend="differentiable-transform-algebra-audit",
            reference_backend="analytic-and-adjoint-references",
        ),
        cases=tuple(
            CaseResult(
                operation_family=f"transform-support:{row.row_id}",
                dimension=index,
                status=row.status,
                error=row.residual,
            )
            for index, row in enumerate(resolved.support_matrix, start=1)
        ),
        derived_from=edges,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit the validated support-matrix bundle as JSON.

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
        default=DEFAULT_SUPPORT_MATRIX_ARTIFACT_PATH,
        help="committed support-matrix artefact for the derivation edge",
    )
    args = parser.parse_args(argv)
    validated = validate_bundle(build_support_matrix_bundle(artifact_path=args.artifact_path))
    print(json.dumps(validated.bundle.to_dict(), indent=2, sort_keys=True))
    if not validated.verdict.admitted:
        print(
            "support-matrix bundle rejected: " + "; ".join(validated.verdict.rejections),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "DEFAULT_SUPPORT_MATRIX_ARTIFACT_PATH",
    "build_support_matrix_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
