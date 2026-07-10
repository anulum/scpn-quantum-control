# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B readout-mitigation bundle emission
"""Federate the committed readout-mitigation evidence as a schema-B bundle.

The Phase 2 readout-mitigation summary
(``data/phase2_readout_mitigation/phase2_readout_mitigation_summary_2026-05-05.json``)
carries measured raw-versus-corrected parity-asymmetry pairs from real
``ibm_kingston`` runs, corrected through state-specific parity confusion
inversion. This module federates that artefact as one
``studio.mitigation.v1`` bundle — the evidence family produced by the
``mitigate`` verb.

The mapping is deliberately honest. Every measured pair rides in ``cases[]``
verbatim with its corrected relative asymmetry as the case error, and the
bundle claim boundary carries the artefact's own confusion-matrix caveat
verbatim: the correction is a state-specific parity inversion over selected
calibration states, **not** a full ``2^n x 2^n`` confusion-matrix inversion.
An artefact missing that caveat, its method, or its pairs is refused
outright — a weakened summary is never federated.
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
from .verbs import MITIGATE, MITIGATION_SCHEMA, STUDIO_ID

DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH = Path(
    "data/phase2_readout_mitigation/phase2_readout_mitigation_summary_2026-05-05.json"
)
"""Repository-relative path of the committed readout-mitigation summary."""

READOUT_MITIGATION_ARTIFACT_ID = "phase2-readout-mitigation-summary-2026-05-05"

_REGENERATED_BY = "python -m scpn_quantum_control.studio.readout_mitigation_bundle"


def _load_summary(artifact_path: Path) -> dict[str, Any]:
    """Load and shape-check the committed mitigation summary, failing closed.

    Parameters
    ----------
    artifact_path
        Committed summary path (relative paths resolve against the repo root).

    Returns
    -------
    dict[str, Any]
        The parsed summary payload.

    Raises
    ------
    ValueError
        If the summary is missing its method, its measured pairs, or the
        confusion-matrix honesty caveat.
    """
    resolved = artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path
    payload: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload.get("method"), str) or not payload["method"]:
        raise ValueError("readout-mitigation summary is missing its method")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"readout-mitigation summary has no pairs: {resolved.as_posix()}")
    note = payload.get("full_confusion_matrix_note")
    if not isinstance(note, str) or not note:
        raise ValueError("readout-mitigation summary is missing its confusion-matrix caveat")
    return payload


def build_readout_mitigation_bundle(
    *,
    artifact_path: Path = DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build the schema-B ``studio.mitigation.v1`` bundle.

    Parameters
    ----------
    artifact_path
        Committed readout-mitigation summary; its content digest rides as a
        ``derived_from`` edge (a missing path fails closed).
    activity_timestamp
        Deterministic PROV timestamp for the emission.

    Returns
    -------
    EvidenceBundle
        The ``studio.mitigation.v1`` bundle, measured raw-versus-corrected
        pairs verbatim in ``cases[]`` with corrected relative asymmetries as
        case errors.

    Raises
    ------
    ValueError
        If the summary fails its shape check.
    """
    payload = _load_summary(artifact_path)
    pairs: list[dict[str, Any]] = payload["pairs"]
    note = f"method: {payload['method']}; caveat: {payload['full_confusion_matrix_note']}"
    return EvidenceBundle(
        schema=MITIGATION_SCHEMA,
        entity=ProvEntity(
            entity_id=f"{STUDIO_ID}:readout-mitigation:{READOUT_MITIGATION_ARTIFACT_ID}",
            digest=_digest_payload(payload),
        ),
        activity=_studio_activity(
            verb=MITIGATE.name,
            command=_REGENERATED_BY,
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_SUPPORT,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=note),
        ),
        substrate=Substrate.HARDWARE_MITIGATED,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend="state-specific-parity-confusion-inversion",
            reference_backend="raw-counts",
        ),
        cases=tuple(
            CaseResult(
                operation_family=(f"readout-mitigation:{pair['dataset']}:{pair['comparison']}"),
                dimension=int(pair["depth"]),
                status="measured-corrected",
                error=float(pair["corrected_relative"]),
            )
            for pair in pairs
        ),
        derived_from=(
            _committed_artifact_edge(
                artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path,
                label="readout-mitigation summary artefact",
            ),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit the validated readout-mitigation bundle as JSON.

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
        default=DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH,
        help="committed readout-mitigation summary for the derivation edge",
    )
    args = parser.parse_args(argv)
    validated = validate_bundle(build_readout_mitigation_bundle(artifact_path=args.artifact_path))
    print(json.dumps(validated.bundle.to_dict(), indent=2, sort_keys=True))
    if not validated.verdict.admitted:
        print(
            "readout-mitigation bundle rejected: " + "; ".join(validated.verdict.rejections),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH",
    "READOUT_MITIGATION_ARTIFACT_ID",
    "build_readout_mitigation_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
