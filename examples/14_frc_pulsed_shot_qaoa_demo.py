# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FRC pulsed-shot QAOA scheduling demo
"""Schedule FRC capacitor-bank firing with QAOA against a classical baseline.

Sister demo to ``examples/13_iter_disruption_demo.py``. Run with::

    python examples/14_frc_pulsed_shot_qaoa_demo.py
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.control.frc_pulsed_qaoa import (
    classical_sqp_schedule,
    optimal_schedule,
    solve_frc_pulsed_qaoa,
)
from scpn_quantum_control.control.qaoa_pulsed_cost import (
    FRCQAOAObjective,
    frc_pulsed_shot_cost,
)


def main() -> None:
    target_b_profile = np.linspace(0.5, 4.0, 8)  # desired external field ramp [T]
    available_energy = 1.0e6  # J
    objective = FRCQAOAObjective(
        target_s_parameter=2.5,
        bank_energy_budget_J=5.0e5,
        mrti_amplitude_max_m=1.0e-2,
        tilt_margin_required=0.3,
    )

    optimum = optimal_schedule(target_b_profile, available_energy, objective)
    classical = classical_sqp_schedule(target_b_profile, available_energy, objective, seed=0)
    quantum = solve_frc_pulsed_qaoa(
        target_b_profile, available_energy, objective, p_layers=4, restarts=8, seed=0
    )

    print("FRC pulsed-shot scheduling (8 capacitor banks)")
    print(f"  brute-force optimum : cost={optimum.cost:.5f}  schedule={optimum.schedule}")
    print(f"  classical SLSQP     : cost={classical.cost:.5f}  schedule={classical.schedule}")
    print(f"  QAOA (p=4)          : cost={quantum.cost:.5f}  schedule={quantum.schedule}")
    print(f"  QAOA / optimum      : {quantum.cost / optimum.cost:.4f}")

    _, components = frc_pulsed_shot_cost(
        np.array(quantum.schedule, dtype=float),
        target_b_profile,
        available_energy,
        objective,
        return_components=True,
    )
    print("  QAOA schedule physics:")
    print(f"    s-parameter   = {components['s_achieved']:.3f}")
    print(f"    energy used   = {components['energy_used_J']:.3e} J")
    print(f"    MRTI amplitude= {components['mrti_amplitude_m']:.3e} m")
    print(f"    tilt margin   = {components['tilt_margin']:.3f}")


if __name__ == "__main__":
    main()
