# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Hardware Integration
"""No-QPU hardware manifest gate for topology-control experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constraints import Edge, canonical_edge


@dataclass(frozen=True)
class TopologyHardwareManifest:
    """Preregistered hardware lane descriptor for topological control."""

    backend_name: str
    qubits: tuple[int, ...]
    coupling_edges: tuple[Edge, ...]
    shots: int
    qpu_minute_ceiling: float | None
    preregistration_id: str
    objective_sha256: str
    require_readout_calibration: bool = True
    live_submission_allowed: bool = False


def validate_topology_hardware_manifest(
    manifest: TopologyHardwareManifest,
    *,
    backend_descriptor: Any | None = None,
) -> TopologyHardwareManifest:
    """Validate a topology-control hardware manifest without submitting jobs."""

    errors: list[str] = []
    if not manifest.backend_name:
        errors.append("backend_name is required")
    if len(manifest.qubits) < 2:
        errors.append("at least two qubits are required")
    if len(set(manifest.qubits)) != len(manifest.qubits):
        errors.append("qubits must be unique")
    if manifest.shots <= 0:
        errors.append("shots must be positive")
    if manifest.qpu_minute_ceiling is None:
        errors.append("qpu_minute_ceiling is required")
    elif manifest.qpu_minute_ceiling < 0.0:
        errors.append("qpu_minute_ceiling must be non-negative")
    if not manifest.preregistration_id:
        errors.append("preregistration_id is required")
    if len(manifest.objective_sha256) != 64:
        errors.append("objective_sha256 must be a SHA-256 hex digest")
    if manifest.live_submission_allowed:
        errors.append(
            "live_submission_allowed must remain False for no-submit topology manifest validation"
        )
    if errors:
        raise ValueError("; ".join(errors))

    n = len(manifest.qubits)
    logical_edges = {canonical_edge(i, j) for i, j in manifest.coupling_edges}
    for i, j in logical_edges:
        if i >= n or j >= n:
            raise ValueError("coupling_edges must use logical qubit indices")

    if backend_descriptor is not None:
        available = getattr(getattr(backend_descriptor, "capabilities", None), "n_qubits", None)
        if available is not None and int(available) < n:
            raise ValueError("backend_descriptor does not expose enough qubits")

    return TopologyHardwareManifest(
        backend_name=manifest.backend_name,
        qubits=manifest.qubits,
        coupling_edges=tuple(sorted(logical_edges)),
        shots=manifest.shots,
        qpu_minute_ceiling=manifest.qpu_minute_ceiling,
        preregistration_id=manifest.preregistration_id,
        objective_sha256=manifest.objective_sha256,
        require_readout_calibration=manifest.require_readout_calibration,
        live_submission_allowed=manifest.live_submission_allowed,
    )


__all__ = ["TopologyHardwareManifest", "validate_topology_hardware_manifest"]
