# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum SSGF descent demo
"""Variational geometry descent on the quantum synchronisation cost.

Optimises a latent vector parameterising the coupling geometry W(z) against
the quantum cost 1 - R_global (statevector Trotter evolution) and prints the
synchronisation gain of the descent. Small-system simulator demonstration.
"""

from __future__ import annotations

from scpn_quantum_control.ssgf.quantum_outer_cycle import quantum_outer_cycle


def main() -> None:
    print("Quantum SSGF descent demo")
    print("=" * 50)

    result = quantum_outer_cycle(
        n_osc=3,
        alpha=1.0,
        learning_rate=0.1,
        max_iterations=20,
        dt=1.0,
        seed=20260708,
    )

    print(f"iterations:        {result.n_iterations} (converged={result.converged})")
    print(f"R_global start:    {result.r_global_history[0]:.4f}")
    print(f"R_global final:    {result.final_r_global:.4f}")
    print(f"cost start/final:  {result.cost_history[0]:.4f} / {result.final_cost:.4f}")
    print("claim boundary: small-system statevector simulation; no hardware,")
    print("scaling, or SCPN-CODEBASE production claims.")


if __name__ == "__main__":
    main()
