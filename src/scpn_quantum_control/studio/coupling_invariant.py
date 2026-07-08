# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B coupling-invariant bundle emission
"""Emit the effective-coupling invariant as a schema-B evidence bundle.

ST-14 federates the coupling-estimation surface needed by Studio without
promoting it beyond its inputs. The invariant is deliberately narrow:
``knm.kuramoto.effective-coupling`` from two committed estimator classes,
Hamiltonian learning and differentiable coupling learning with parameter-shift
gradient verification. It also requires two uncertainty sources:
shot-noise synchronisation uncertainty and zero-noise-extrapolation uncertainty.

The bundle does not use the DLA parity family and does not imply a DLA result.
It is a curated numerical-model contract proving that the estimator and UQ
classes are present, named, and wired to a Studio evidence schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

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

from ..analysis.hamiltonian_learning import HamiltonianLearningResult
from ..analysis.sync_uncertainty import UncertaintyInterval
from ..mitigation.zne_uncertainty import ZNEUncertaintyResult
from ..phase.coupling_learning import CouplingGradientVerificationResult
from .evidence_bundle import (
    DEFAULT_ACTIVITY_TIMESTAMP,
    _digest_payload,
    _studio_activity,
    _studio_agent,
    validate_bundle,
)
from .verbs import ANALYSE, COUPLING_INVARIANT_SCHEMA, STUDIO_ID

COUPLING_INVARIANT_ID: Final[str] = "knm.kuramoto.effective-coupling"
"""Logical invariant emitted by the ST-14 producer."""

COUPLING_INVARIANT_ARTIFACT_ID: Final[str] = "coupling-invariant-effective-20260708"
"""Stable Studio entity suffix for the effective-coupling invariant producer."""

_REGENERATED_BY: Final[str] = "python -m scpn_quantum_control.studio.coupling_invariant"


@dataclass(frozen=True)
class CouplingInvariantSource:
    """A committed estimator or uncertainty source behind the invariant.

    Parameters
    ----------
    source_id:
        Stable source identifier used on the Studio wire.
    module:
        Import path containing the committed source class.
    class_name:
        Source class name.
    role:
        Source role: ``"estimator"`` or ``"uncertainty"``.
    method:
        Human-readable method name that stays stable across refactors.
    """

    source_id: str
    module: str
    class_name: str
    role: str
    method: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready source descriptor."""
        return {
            "source_id": self.source_id,
            "module": self.module,
            "class_name": self.class_name,
            "role": self.role,
            "method": self.method,
        }


@dataclass(frozen=True)
class CouplingInvariantPayload:
    """JSON payload hashed into the Studio coupling-invariant bundle."""

    schema: str
    artifact_id: str
    invariant_id: str
    estimator_sources: tuple[CouplingInvariantSource, ...]
    uncertainty_sources: tuple[CouplingInvariantSource, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return the canonical payload represented by this producer."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "invariant_id": self.invariant_id,
            "estimator_sources": [source.to_dict() for source in self.estimator_sources],
            "uncertainty_sources": [source.to_dict() for source in self.uncertainty_sources],
            "claim_boundary": self.claim_boundary,
        }


def _source_from_class(
    *,
    source_id: str,
    role: str,
    method: str,
    cls: type[object],
) -> CouplingInvariantSource:
    """Build a source descriptor from a committed Python class."""
    return CouplingInvariantSource(
        source_id=source_id,
        module=cls.__module__,
        class_name=cls.__name__,
        role=role,
        method=method,
    )


def build_coupling_invariant_payload() -> CouplingInvariantPayload:
    """Build the ST-14 effective-coupling invariant payload.

    Returns
    -------
    CouplingInvariantPayload
        The canonical source inventory for
        ``knm.kuramoto.effective-coupling``.
    """
    return CouplingInvariantPayload(
        schema=COUPLING_INVARIANT_SCHEMA,
        artifact_id=COUPLING_INVARIANT_ARTIFACT_ID,
        invariant_id=COUPLING_INVARIANT_ID,
        estimator_sources=(
            _source_from_class(
                source_id="hamiltonian-learning",
                role="estimator",
                method="maximum-likelihood Hamiltonian correlator inversion",
                cls=HamiltonianLearningResult,
            ),
            _source_from_class(
                source_id="differentiable-coupling-learning.parameter-shift",
                role="estimator",
                method="parameter-shift gradient verification against central finite difference",
                cls=CouplingGradientVerificationResult,
            ),
        ),
        uncertainty_sources=(
            _source_from_class(
                source_id="sync_uncertainty",
                role="uncertainty",
                method="shot-noise synchronisation interval",
                cls=UncertaintyInterval,
            ),
            _source_from_class(
                source_id="zne_uncertainty",
                role="uncertainty",
                method="zero-noise extrapolation uncertainty propagation",
                cls=ZNEUncertaintyResult,
            ),
        ),
        claim_boundary=(
            "effective-coupling source inventory only; requires downstream numeric "
            "estimates before physics validation; not a DLA parity claim"
        ),
    )


def validate_coupling_invariant_payload(payload: CouplingInvariantPayload) -> bool:
    """Return whether ``payload`` satisfies the ST-14 coupling-invariant contract.

    Raises
    ------
    ValueError
        If the schema, invariant id, estimator set, or UQ set is malformed.
    """
    data = payload.to_dict()
    if payload.schema != COUPLING_INVARIANT_SCHEMA:
        raise ValueError("coupling invariant payload has the wrong schema")
    if "dla" in payload.invariant_id.lower():
        raise ValueError("coupling invariant must not be a DLA invariant")
    if payload.invariant_id != COUPLING_INVARIANT_ID:
        raise ValueError("coupling invariant id must be knm.kuramoto.effective-coupling")
    estimator_ids = {source.source_id for source in payload.estimator_sources}
    if estimator_ids != {
        "hamiltonian-learning",
        "differentiable-coupling-learning.parameter-shift",
    }:
        raise ValueError("coupling invariant must use exactly the two committed estimator classes")
    uncertainty_ids = {source.source_id for source in payload.uncertainty_sources}
    if uncertainty_ids != {"sync_uncertainty", "zne_uncertainty"}:
        raise ValueError("coupling invariant requires sync_uncertainty and zne_uncertainty")
    if any(source.role != "estimator" for source in payload.estimator_sources):
        raise ValueError("estimator sources must carry role=estimator")
    if any(source.role != "uncertainty" for source in payload.uncertainty_sources):
        raise ValueError("uncertainty sources must carry role=uncertainty")
    if not data["claim_boundary"]:
        raise ValueError("coupling invariant requires an explicit claim boundary")
    return True


def build_coupling_invariant_bundle(
    *,
    payload: CouplingInvariantPayload | None = None,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build the schema-B ``studio.coupling-invariant.v1`` bundle.

    Parameters
    ----------
    payload:
        Optional payload for tests; omitted means the committed source inventory.
    activity_timestamp:
        Deterministic PROV timestamp for source-tree emission.

    Returns
    -------
    EvidenceBundle
        The admitted bundle candidate for Studio federation.
    """
    resolved = payload or build_coupling_invariant_payload()
    validate_coupling_invariant_payload(resolved)
    wire_payload = resolved.to_dict()
    sources = (*resolved.estimator_sources, *resolved.uncertainty_sources)
    return EvidenceBundle(
        schema=COUPLING_INVARIANT_SCHEMA,
        entity=ProvEntity(
            entity_id=f"{STUDIO_ID}:coupling-invariant:{resolved.artifact_id}",
            digest=_digest_payload(wire_payload),
        ),
        activity=_studio_activity(
            verb=ANALYSE.name,
            command=_REGENERATED_BY,
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.CURATED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=resolved.claim_boundary),
        ),
        substrate=Substrate.NUMERICAL_MODEL,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend="hamiltonian-learning+differentiable-coupling-learning",
            reference_backend="sync_uncertainty+zne_uncertainty",
        ),
        cases=tuple(
            CaseResult(
                operation_family=f"{resolved.invariant_id}:{source.source_id}",
                dimension=index,
                status=source.role,
            )
            for index, source in enumerate(sources, start=1)
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit the validated coupling-invariant bundle as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    validated = validate_bundle(build_coupling_invariant_bundle())
    print(json.dumps(validated.bundle.to_dict(), indent=2, sort_keys=True))
    if not validated.verdict.admitted:
        print(
            "coupling-invariant bundle rejected: " + "; ".join(validated.verdict.rejections),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "COUPLING_INVARIANT_ARTIFACT_ID",
    "COUPLING_INVARIANT_ID",
    "CouplingInvariantPayload",
    "CouplingInvariantSource",
    "build_coupling_invariant_bundle",
    "build_coupling_invariant_payload",
    "main",
    "validate_coupling_invariant_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
