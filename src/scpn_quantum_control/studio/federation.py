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
  the architecture docs consume: the per-stage IO pipeline, the backend/dispatch
  matrix, the interface surface, the cross-repo wire formats, and the honest scope
  boundaries. It mirrors ``docs/architecture_map.md``.

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


def build_architecture_map_extension() -> dict[str, Any]:
    """Return the architecture-map extension block (additive superset over schema A).

    Mirrors ``docs/architecture_map.md``: the canonical data pipeline with per-stage
    IO contracts, the backend/dispatch matrix, the interface surface, the cross-repo
    wire formats, and the honest scope boundaries.
    """
    return {
        "pipeline": [
            {
                "stage": "problem",
                "inputs": [
                    "K_nm:(n,n) float64 square+finite+symmetric",
                    "omega:(n,)",
                    "metadata:json",
                ],
                "outputs": ["KuramotoProblem (frozen, diagonal zeroed)"],
                "processing_model": "validate + freeze; K symmetrised (K+K.T)/2",
                "backends": ["python"],
            },
            {
                "stage": "hamiltonian",
                "inputs": ["KuramotoProblem"],
                "outputs": ["SparsePauliOp (XY)", "dense (2^n,2^n) complex128", "sparse CSC"],
                "processing_model": "Kuramoto->XY: K sin(theta_j-theta_i) -> -J(XX+YY), omega -> -h Z",
                "backends": ["rust", "qiskit", "python"],
            },
            {
                "stage": "circuit",
                "inputs": ["KuramotoProblem", "time", "trotter_steps"],
                "outputs": ["QuantumCircuit", "Statevector", "trajectory"],
                "processing_model": "Lie/Suzuki Trotter synthesis",
                "backends": ["qiskit", "rust"],
            },
            {
                "stage": "execution",
                "inputs": ["circuit", "shots", "approval_id"],
                "outputs": ["raw counts", "job_id", "provenance dossier"],
                "processing_model": "SamplerV2 / provider HAL (approval-gated, fail-closed)",
                "backends": ["qiskit-runtime", "16 provider adapters"],
            },
            {
                "stage": "mitigation",
                "inputs": ["counts", "per-scale estimates + standard errors"],
                "outputs": ["mitigated estimate", "uncertainty interval"],
                "processing_model": "ZNE/PEC/DD/readout/Z2 + propagated uncertainty",
                "backends": ["numpy"],
            },
            {
                "stage": "analysis",
                "inputs": ["counts dict", "Statevector", "dense H"],
                "outputs": ["order parameter R", "witnesses", "OTOC", "DLA parity", "metrics"],
                "processing_model": "small-N exact diagonalisation gated by dense_budget",
                "backends": ["numpy", "rust"],
            },
            {
                "stage": "ledger",
                "inputs": ["observables", "provenance"],
                "outputs": ["claim-classified, artefact-backed evidence"],
                "processing_model": "five-class hardware-status ledger",
                "backends": ["python"],
            },
        ],
        "backends": [
            {
                "name": "rust",
                "language": "rust",
                "role": "hot kernels",
                "dispatch_order": 1,
                "runtime": "scpn_quantum_engine PyO3 0.29, 151 kernels",
                "build_vs_runtime": "build",
            },
            {
                "name": "julia",
                "language": "julia",
                "role": "order-parameter / mean-field",
                "dispatch_order": 2,
                "runtime": "juliacall",
                "build_vs_runtime": "runtime-jit",
            },
            {
                "name": "python",
                "language": "python",
                "role": "guaranteed floor",
                "dispatch_order": 3,
                "build_vs_runtime": "runtime",
            },
            {
                "name": "qiskit-runtime",
                "language": "python",
                "role": "IBM execution",
                "build_vs_runtime": "runtime",
            },
            {
                "name": "provider-hal",
                "language": "python",
                "role": "16 approval-gated providers",
                "build_vs_runtime": "runtime",
            },
        ],
        "interfaces": {
            "cli": [
                "scpn-bench",
                "scpn-verify-hardware-packs",
                "scpn-generate-hardware-pack-evidence",
                "scpn-provider-smoke",
                "scpn-biological-qec-report",
                "install-differentiable-framework-overlay",
            ],
            "library": "scpn_quantum_control",
            "rest": None,
            "grpc": None,
            "studio_feed": STUDIO_MANIFEST_PATH.as_posix(),
        },
        "cross_repo": [
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
        ],
        "boundaries": {
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
        },
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
