# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Artefacts
"""Serialisation and digest artefacts for topological optimisation traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .optimizers import TopologyOptimisationTrace


@dataclass(frozen=True)
class TopologyOptimisationArtifact:
    """Stable JSON artefact with embedded SHA-256 digest."""

    payload: dict[str, Any]
    sha256: str

    def to_json(self) -> str:
        """Return canonical JSON including the digest field."""
        data = dict(self.payload)
        data["sha256"] = self.sha256
        return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _trace_payload(trace: TopologyOptimisationTrace, claim_boundary: str) -> dict[str, Any]:
    return {
        "schema": "topological_optimisation_artifact_v1",
        "objective_schema": trace.objective_schema,
        "seed": trace.seed,
        "claim_boundary": claim_boundary,
        "initial_matrix": trace.initial_matrix.tolist(),
        "final_matrix": trace.final_matrix.tolist(),
        "steps": [
            {
                "index": step.index,
                "objective_total": step.objective.total,
                "terms": dict(sorted(step.objective.terms.items())),
                "degeneracy_mode": step.objective.degeneracy_mode.value,
                "p_h1": step.objective.h1_summary.p_h1,
                "n_h1_persistent": step.objective.h1_summary.n_h1_persistent,
                "gradient_norm": step.gradient_norm,
                "step_size": step.step_size,
                "perturbation": step.perturbation,
            }
            for step in trace.steps
        ],
    }


def export_topology_optimisation_artifact(
    trace: TopologyOptimisationTrace,
    *,
    claim_boundary: str,
) -> TopologyOptimisationArtifact:
    """Export a deterministic digest-bearing artefact."""
    payload = _trace_payload(trace, claim_boundary)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return TopologyOptimisationArtifact(payload=payload, sha256=digest)


__all__ = ["TopologyOptimisationArtifact", "export_topology_optimisation_artifact"]
