# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio federation document (schema_a core + architecture_map extension)
"""The QUANTUM studio's federation document for STUDIO/Hub ingestion.

The federation document is one JSON with two blocks, per the locked fleet convention:

* ``schema_a`` — the platform :class:`~scpn_studio_platform.manifest.CapabilityManifest`
  (verbs, evidence schemas, content digest). This is the federation contract the Hub
  ingests; its vocabulary is the locked SDK enums emitted verbatim.
* ``architecture_map`` — an additive superset block the Hub ignores for federation but
  the architecture docs consume: the per-stage IO pipeline, the capability inventory,
  the backend/dispatch matrix, the interface surface, the cross-repo wire formats, and
  the honest scope boundaries. The field set is the fleet ``architecture-map.v2`` schema
  (peer-aligned with SC-NEUROCORE, 2026-06-24); it mirrors ``docs/architecture_map.md``.

This is emitted to a dedicated file so it never collides with the repository
inventory manifest (``docs/_generated/capability_manifest.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import build_manifest

#: Where the federation document is written, relative to the repository root.
STUDIO_MANIFEST_PATH = Path("docs/_generated/studio_manifest.json")

#: The fleet architecture-map extension schema version (peer-aligned with SC-NEUROCORE).
ARCHITECTURE_MAP_VERSION = "architecture-map.v2"


def _pipeline_stages() -> list[dict[str, Any]]:
    """Return the canonical data pipeline with per-stage IO contracts."""
    return [
        {
            "stage": "problem",
            "inputs": [
                "K_nm:(n,n) float64 square+finite+symmetric",
                "omega:(n,)",
                "metadata:json",
            ],
            "outputs": ["KuramotoProblem (frozen, diagonal zeroed)"],
            "processing_model": "validate + freeze; K symmetrised (K+K.T)/2",
        },
        {
            "stage": "hamiltonian",
            "inputs": ["KuramotoProblem"],
            "outputs": ["SparsePauliOp (XY)", "dense (2^n,2^n) complex128", "sparse CSC"],
            "processing_model": "Kuramoto->XY: K sin(theta_j-theta_i) -> -J(XX+YY), omega -> -h Z",
        },
        {
            "stage": "circuit",
            "inputs": ["KuramotoProblem", "time", "trotter_steps"],
            "outputs": ["QuantumCircuit", "Statevector", "trajectory"],
            "processing_model": "Lie/Suzuki Trotter synthesis",
        },
        {
            "stage": "execution",
            "inputs": ["circuit", "shots", "approval_id"],
            "outputs": ["raw counts", "job_id", "provenance dossier"],
            "processing_model": "SamplerV2 / provider HAL (approval-gated, fail-closed)",
        },
        {
            "stage": "mitigation",
            "inputs": ["counts", "per-scale estimates + standard errors"],
            "outputs": ["mitigated estimate", "uncertainty interval"],
            "processing_model": "ZNE/PEC/DD/readout/Z2 + propagated uncertainty",
        },
        {
            "stage": "analysis",
            "inputs": ["counts dict", "Statevector", "dense H"],
            "outputs": ["order parameter R", "witnesses", "OTOC", "DLA parity", "metrics"],
            "processing_model": "small-N exact diagonalisation gated by dense_budget",
        },
        {
            "stage": "ledger",
            "inputs": ["observables", "provenance"],
            "outputs": ["claim-classified, artefact-backed evidence"],
            "processing_model": "five-class hardware-status ledger",
        },
    ]


def _capabilities() -> list[dict[str, str]]:
    """Return the capability inventory with honest per-capability status."""
    return [
        {
            "name": "kuramoto-compilation",
            "domain": "Compilation",
            "tier": "core",
            "status": "wired",
        },
        {"name": "quantum-evolution", "domain": "Simulation", "tier": "core", "status": "wired"},
        {
            "name": "synchronisation-analysis",
            "domain": "Analysis",
            "tier": "core",
            "status": "wired",
        },
        {"name": "error-mitigation", "domain": "Mitigation", "tier": "core", "status": "wired"},
        {"name": "hardware-execution", "domain": "Hardware", "tier": "core", "status": "wired"},
        {"name": "evidence-ledger", "domain": "Provenance", "tier": "core", "status": "wired"},
        {
            "name": "whole-program-ad",
            "domain": "Differentiable",
            "tier": "extended",
            "status": "library-only",
        },
        {
            "name": "tensor-network-evolution",
            "domain": "Simulation",
            "tier": "extended",
            "status": "library-only",
        },
        {
            "name": "pulse-level-control",
            "domain": "Control",
            "tier": "research",
            "status": "feasibility-only",
        },
        {
            "name": "analog-execution",
            "domain": "Hardware",
            "tier": "research",
            "status": "feasibility-only",
        },
    ]


def _backends() -> list[dict[str, Any]]:
    """Return the backend/dispatch matrix with runtime-availability status."""
    return [
        {
            "name": "rust",
            "language": "Rust",
            "role": "hot kernels (scpn_quantum_engine, 151 PyO3)",
            "dispatch_order": 1,
            "status": "runtime-active",
        },
        {
            "name": "julia",
            "language": "Julia",
            "role": "order-parameter / mean-field (juliacall)",
            "dispatch_order": 2,
            "status": "build-available",
        },
        {
            "name": "python",
            "language": "Python",
            "role": "guaranteed numerical floor",
            "dispatch_order": 3,
            "status": "runtime-active",
        },
        {
            "name": "qiskit-runtime",
            "language": "Python",
            "role": "IBM Quantum execution",
            "dispatch_order": 4,
            "status": "declared",
        },
        {
            "name": "provider-hal",
            "language": "Python",
            "role": "16 approval-gated provider adapters",
            "dispatch_order": 5,
            "status": "declared",
        },
    ]


def _interfaces() -> list[dict[str, str]]:
    """Return the interface surface (CLI entry points, library, studio feed)."""
    return [
        {"kind": "cli", "entry": "scpn-bench = scpn_quantum_control.bench_cli:main"},
        {
            "kind": "cli",
            "entry": "scpn-verify-hardware-packs = scpn_quantum_control.hardware_result_packs:main",
        },
        {
            "kind": "cli",
            "entry": "scpn-generate-hardware-pack-evidence = scpn_quantum_control.hardware_result_pack_evidence:main",
        },
        {
            "kind": "cli",
            "entry": "scpn-provider-smoke = scpn_quantum_control.hardware.provider_smoke:main",
        },
        {
            "kind": "cli",
            "entry": "scpn-biological-qec-report = scpn_quantum_control.qec.biological_cli:main",
        },
        {
            "kind": "cli",
            "entry": "scpn-emit-studio-manifest = scpn_quantum_control.studio.federation:main",
        },
        {"kind": "library", "entry": "scpn_quantum_control"},
        {"kind": "studio_feed", "entry": STUDIO_MANIFEST_PATH.as_posix()},
    ]


def _wire_formats() -> list[dict[str, str]]:
    """Return the named cross-boundary wire formats with schema references."""
    return [
        {
            "name": "KuramotoProblem",
            "schema_ref": "scpn_quantum_control.bridge.kuramoto_problem (frozen K_nm/omega problem)",
        },
        {
            "name": "studio-evidence",
            "schema_ref": "studio.*.v1 (9 evidence schemas from the five-class hardware ledger)",
        },
        {
            "name": "UPDEPhaseArtifact",
            "schema_ref": "scpn_quantum_control.bridge.orchestrator_adapter (quantum R -> advance/hold/rollback)",
        },
        {
            "name": "spike-train<->Ry",
            "schema_ref": "scpn_quantum_control.bridge.snn_adapter (sc-neurocore spike trains <-> Ry angles)",
        },
    ]


def _cross_repo() -> list[dict[str, str]]:
    """Return the cross-repository sibling adapters and their wire formats."""
    return [
        {
            "sibling": "sc-neurocore",
            "adapter": "bridge.snn_adapter",
            "wire_format": "spike trains <-> Ry angles / quantum dense layer",
        },
        {
            "sibling": "scpn-control",
            "adapter": "bridge.control_plasma_knm",
            "wire_format": "plasma-native K/omega -> KuramotoProblem",
        },
        {
            "sibling": "scpn-fusion-core",
            "adapter": "bridge.fusion_core_frc",
            "wire_format": "FRC equilibrium -> pulsed-shot QAOA surrogate",
        },
        {
            "sibling": "scpn-phase-orchestrator",
            "adapter": "bridge.orchestrator_adapter",
            "wire_format": "orchestrator state <-> UPDEPhaseArtifact; quantum R -> advance/hold/rollback",
        },
    ]


def _boundaries() -> dict[str, list[str]]:
    """Return the honest scope boundaries (executed / bounded / feasibility-only / closed)."""
    return {
        "executed": [
            "Hamiltonian compilation",
            "Trotter/VQE/analysis on simulators",
            "IBM Runtime (approval-gated)",
            "error mitigation",
            "evidence ledger",
            "ML-DSA signing",
            "QRNG",
        ],
        "bounded": [
            "whole-program AD compiler (scalar + static dense-linalg, fail-closed)",
            "tensor-network evolution (nearest-neighbour only)",
            "analysis (small-N exact)",
        ],
        "feasibility_only": [
            "pulse-level optimal control",
            "analog/neutral-atom execution",
            "real-time intra-shot feedback",
            "FPGA/HLS deployment",
            "NV-magnetometry hardware",
        ],
        "closed": [
            "lab-control instrumentation",
            "broad quantum advantage (classical faster/more accurate at n<=16)",
        ],
    }


def build_architecture_map_extension() -> dict[str, Any]:
    """Return the architecture-map extension block (fleet ``architecture-map.v2`` schema).

    Additive superset over schema A: the pipeline, capability inventory, backend/dispatch
    matrix, interface surface, cross-repo wire formats, and honest scope boundaries. The
    field set is peer-aligned with SC-NEUROCORE (2026-06-24); the Hub ignores it for
    federation, the architecture docs consume it.
    """
    return {
        "version": ARCHITECTURE_MAP_VERSION,
        "pipeline_stages": _pipeline_stages(),
        "capabilities": _capabilities(),
        "backends": _backends(),
        "interfaces": _interfaces(),
        "wire_formats": _wire_formats(),
        "cross_repo": _cross_repo(),
        "boundaries": _boundaries(),
    }


def build_federation_document() -> dict[str, Any]:
    """Return the full federation document: schema_a core + architecture_map extension."""
    return {
        "schema_a": build_manifest().to_dict(),
        "architecture_map": build_architecture_map_extension(),
    }


def write_federation_document(repo_root: Path | None = None) -> Path:
    """Write the federation document to :data:`STUDIO_MANIFEST_PATH` and return the path.

    Parameters
    ----------
    repo_root
        Repository root; defaults to the current working directory.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    root = repo_root or Path.cwd()
    out = root / STUDIO_MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_federation_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    """CLI entry point: write the federation document and report its path + digest."""
    path = write_federation_document()
    digest = build_manifest().to_dict()["content_digest"]
    print(f"Wrote {path} (schema_a content_digest={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
