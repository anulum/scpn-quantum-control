# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Fault-tolerant UPDE demo: repetition code with syndrome extraction."""

from __future__ import annotations

from scpn_quantum_control.qec.fault_tolerant import FaultTolerantUPDE


def main() -> None:
    print("Fault-Tolerant UPDE Demo")
    print("=" * 50)

    for d in [3, 5]:
        ft = FaultTolerantUPDE(n_osc=4, code_distance=d)
        print(f"\nDistance d={d}:")
        print(f"  Physical qubits: {ft.physical_qubit_count()}")
        print(
            f"  Layout: {ft.n_osc} osc × ({ft.data_per_osc} data + {ft.ancilla_per_osc} ancilla)"
        )

        result = ft.step_with_qec(dt=0.1)
        print(f"  Errors detected: {result['errors_detected']}")
        for osc_idx, syn in enumerate(result["syndromes"]):
            if any(s != 0 for s in syn):
                print(f"  Osc {osc_idx} syndrome: {syn}")

    print("\nNote: fault-tolerant UPDE is a proof-of-concept.")
    print("Practical use requires post-2030 error-corrected hardware.")


if __name__ == "__main__":
    main()
