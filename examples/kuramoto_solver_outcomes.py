# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual solver timestamps and termination example
"""Run the public scientific factory and inspect existing SciPy outcomes."""

from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

import scpn_quantum_control as qc
from oscillatools import KuramotoParameters, KuramotoSystem, solve_kuramoto_ivp


class PhaseEvent:
    """Stop the original uncoupled flow when its unwrapped phase reaches 8.2."""

    terminal = True
    direction = 1.0

    def __call__(self, time: float, phases: NDArray[np.float64]) -> float:
        """Return the threshold residual used by SciPy's root interpolation."""
        return float(phases[0] - 8.2)


def singular_rule(
    phases: NDArray[np.float64], parameters: KuramotoParameters, time: float
) -> NDArray[np.float64]:
    """Demonstrate the finite-time singularity y'=1+y² at t=pi/2."""
    return 1 + phases * phases


def run_example() -> dict[str, object]:
    """Exercise finite scientific input binding and three solver outcomes.

    Returns
    -------
    dict[str, object]
        Actual fixed samples, event root and genuine failed-solver diagnostics.
        This analytic example performs no physical quantum submission.

    """
    problem = qc.build_kuramoto_problem(np.zeros((1, 1)), np.array([0.4]))
    design = qc.ScientificDesign(
        model="phase_kuramoto",
        normalisation="pairwise_sum",
        coordinate_space="logical",
        units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
        topology=(),
        initial_state=np.array([8.0]),
        observable="phase_order_parameter",
        observable_weights=np.ones(1),
        objective=qc.DesignObjective("simulate", None, "1"),
    )
    system = qc.build_scientific_phase_system(problem, design, dt=0.25)
    completed = solve_kuramoto_ivp(system, (0, 1), t_eval=[0, 0.5, 1])
    event = solve_kuramoto_ivp(system, (0, 1), events=PhaseEvent())
    times, phases = system.trajectory_with_times(4)
    singular = KuramotoSystem(
        singular_rule, np.zeros(1), KuramotoParameters(np.zeros(1), 0.0), dt=0.01
    )
    failed = solve_kuramoto_ivp(singular, (0, 2), rtol=1e-9, atol=1e-12)
    return {
        "fixed_times": times.tolist(),
        "fixed_phases": phases.tolist(),
        "completed": completed.termination,
        "event": event.termination,
        "event_times": [values.tolist() for values in event.event_times],
        "event_phases": [values.tolist() for values in event.event_phases],
        "failure": failed.termination,
        "failure_success": failed.success,
        "failure_last_time": float(failed.times[-1]),
        "failure_message": failed.message,
        "query_state_retained": singular.current_state.tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True))
