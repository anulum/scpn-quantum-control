# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological QEC Demonstration

"""Demonstration of Topological QEC on the SCPN 16-layer architecture.

Maps the stabilizers and syndromes of a surface code directly to the
hierarchical biological coupling matrix (K_nm) of the 16-layer framework.
This creates a native error correction code that 'lives' on the biological
topology, utilizing synaptic weights to optimize the decoding path.
"""

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.qec.biological_surface_code import (
    BiologicalMWPMDecoder,
    BiologicalSurfaceCode,
)


def main():
    print("=== SCPN 16-Layer Biological Surface Code ===\n")

    # 1. Build the physical SCPN coupling graph (N=16)
    K = build_knm_paper27(L=16)

    # 2. Construct the Biological Surface Code
    # Stabilizers are derived from nodes (X) and graph cycles (Z)
    print("Constructing code from 16-layer topology...")
    code = BiologicalSurfaceCode(K, threshold=0.01)

    summary = code.code_summary()
    print(f"  Data Qubits (Graph Edges): {summary['n_data_qubits']}")
    print(f"  X-Stabilisers (Graph Nodes): {summary['n_x_stabilisers']}")
    print(f"  Z-Stabilisers (Graph Cycles): {summary['n_z_stabilisers']}")
    print(f"  Estimated Logical Qubits: {summary['n_logical_qubits_estimate']}")
    print(f"  Commutation Check: {'PASS' if summary['css_commutes'] else 'FAIL'}")

    # 4. Error Correction Simulation
    decoder = BiologicalMWPMDecoder(code)

    # Simulate a single physical Z-error on a high-weight edge
    # We pick the edge with the strongest coupling (likely (0,1))
    edge_idx = 0
    edge_nodes = code.edges[edge_idx]
    print(
        f"\nSimulating Z-error on biological edge {edge_nodes} (Coupling={K[edge_nodes]:.4f})..."
    )

    error_z = np.zeros(code.num_data, dtype=np.int8)
    error_z[edge_idx] = 1

    # Compute the syndrome measured by the hardware
    syndrome_x = code.x_syndrome_from_z_errors(error_z)
    active_detectors = np.where(syndrome_x == 1)[0]
    print(f"  Detected syndrome at nodes: {list(active_detectors)}")

    # Run the Biological MWPM Decoder and apply correction
    correction, residual = decoder.decode_and_apply(error_z)
    success = np.all(residual == 0)

    print(f"  Correction successful? {success}")

    if success:
        print(
            "\n[RESULT] Biological QEC successfully mapped and corrected errors on the SCPN topology."
        )


if __name__ == "__main__":
    main()
