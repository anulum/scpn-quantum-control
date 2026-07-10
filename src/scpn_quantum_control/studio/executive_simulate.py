# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive simulate handler
"""The ``simulate`` executive action handler — bounded XY-Kuramoto evolution.

The ``simulate`` verb evolves a bounded ``K_nm``/``omega`` oscillator network on
a local dense-statevector simulator. The handler wraps the single public
:meth:`~scpn_quantum_control.phase.QuantumKuramotoSolver.run` entry point:
Trotterised time evolution of the XY spin Hamiltonian from ``t = 0`` to
``t_max`` in ``dt`` steps, measuring the Kuramoto synchronisation order parameter
``R(t)`` along the trajectory. It returns the trajectory summary
(``studio.quantum-evolution.v1``) and writes a standalone reproduction script.

The claim boundary is a *simulator estimate*: the reported order parameter is a
dense-statevector Trotter approximation at the stated step and Trotter
resolution, not a continuous-time exact solution, a physical claim, or QPU
execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from ..phase import QuantumKuramotoSolver
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .verbs import QUANTUM_EVOLUTION_SCHEMA

SIMULATE_VERB: Final[str] = "simulate"
_DEFAULT_BACKEND: Final[str] = "python"
_MAX_NODES: Final[int] = 12
_MAX_TROTTER_PER_STEP: Final[int] = 64
_MAX_TIME_STEPS: Final[int] = 256
_REPRODUCTION_TOLERANCE: Final[float] = 1e-9

SIMULATE_CLAIM_BOUNDARY: Final[str] = (
    "dense-statevector Trotter evolution of a bounded symmetric zero-diagonal "
    "XY-Kuramoto network on a local simulator; the reported synchronisation "
    "order parameter is a simulator estimate at the stated step/Trotter "
    "resolution, not a continuous-time exact solution, a physical claim, or QPU "
    "execution"
)


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_positive_int(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _as_row(row: object) -> Sequence[Any]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        raise ValueError("each K_nm row must be a sequence")
    return row


def _as_coupling_matrix(value: object) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("K_nm must be a square list of rows")
    rows = [[_as_float("K_nm row entry", entry) for entry in _as_row(row)] for row in value]
    size = len(rows)
    if not 2 <= size <= _MAX_NODES:
        raise ValueError(f"K_nm must have between 2 and {_MAX_NODES} nodes")
    if any(len(row) != size for row in rows):
        raise ValueError("K_nm must be square")
    for left in range(size):
        if rows[left][left] != 0.0:
            raise ValueError("K_nm diagonal must be zero")
        for right in range(left + 1, size):
            if rows[left][right] != rows[right][left]:
                raise ValueError("K_nm must be symmetric")
    return rows


def _normalise_simulate(parameters: Mapping[str, Any]) -> dict[str, Any]:
    k_nm = _as_coupling_matrix(parameters.get("K_nm"))
    size = len(k_nm)
    raw_omega = parameters.get("omega")
    if not isinstance(raw_omega, Sequence) or isinstance(raw_omega, (str, bytes)):
        raise ValueError("omega must be a sequence")
    if len(raw_omega) != size:
        raise ValueError("omega length must match the number of nodes")
    omega = [_as_float("omega entry", entry) for entry in raw_omega]
    t_max = _as_float("t_max", parameters.get("t_max"))
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    dt = _as_float("dt", parameters.get("dt"))
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if dt > t_max:
        raise ValueError("dt must not exceed t_max")
    if t_max / dt > _MAX_TIME_STEPS:
        raise ValueError(f"t_max/dt must resolve to at most {_MAX_TIME_STEPS} steps")
    trotter_per_step = _as_positive_int(
        "trotter_per_step", parameters.get("trotter_per_step"), maximum=_MAX_TROTTER_PER_STEP
    )
    trotter_order = parameters.get("trotter_order")
    if trotter_order not in (1, 2) or isinstance(trotter_order, bool):
        raise ValueError("trotter_order must be 1 or 2")
    return {
        "K_nm": k_nm,
        "omega": omega,
        "t_max": t_max,
        "dt": dt,
        "trotter_per_step": trotter_per_step,
        "trotter_order": int(trotter_order),
    }


class SimulateActionHandler(ActionHandler):
    """Executive handler for the read-only ``simulate`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"simulate"``."""
        return SIMULATE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the network and resolve a simulator evolution plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The simulate request; ``parameters`` must describe a bounded network
            (``K_nm``, ``omega``) and evolution schedule (``t_max``, ``dt``,
            ``trotter_per_step``, ``trotter_order``).
        contract : VerbContract
            The resolved ``simulate`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the simulate verb")
        simulate_spec = _normalise_simulate(request.parameters)
        steps = (
            f"validate the {len(simulate_spec['K_nm'])}-node K_nm/omega network",
            "build the XY spin Hamiltonian for the coupled oscillators",
            "Trotter-evolve the dense statevector from t=0 to t_max in dt steps",
            "measure the Kuramoto order parameter R(t) along the trajectory",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=SIMULATE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=simulate_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Evolve the network on the simulator and summarise the trajectory.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned evolution.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the order-parameter trajectory summary.
        """
        simulate_spec: dict[str, Any] = dict(plan.parameters)
        k_nm = np.asarray(simulate_spec["K_nm"], dtype=np.float64)
        omega = np.asarray(simulate_spec["omega"], dtype=np.float64)
        solver = QuantumKuramotoSolver(
            len(simulate_spec["K_nm"]),
            k_nm,
            omega,
            trotter_order=simulate_spec["trotter_order"],
        )
        trajectory = solver.run(
            simulate_spec["t_max"],
            simulate_spec["dt"],
            simulate_spec["trotter_per_step"],
        )
        order_parameter = np.asarray(trajectory.R, dtype=np.float64)
        times = np.asarray(trajectory.times, dtype=np.float64)
        outputs = {
            "backend": plan.backend,
            "n_nodes": len(simulate_spec["K_nm"]),
            "t_max": simulate_spec["t_max"],
            "dt": simulate_spec["dt"],
            "trotter_per_step": simulate_spec["trotter_per_step"],
            "trotter_order": simulate_spec["trotter_order"],
            "evolution_schema": QUANTUM_EVOLUTION_SCHEMA,
            "n_points": int(times.shape[0]),
            "order_parameter_initial": float(order_parameter[0]),
            "order_parameter_final": float(order_parameter[-1]),
            "order_parameter_max": float(order_parameter.max()),
            "order_parameter_mean": float(order_parameter.mean()),
            "order_parameter_delta": float(order_parameter[-1] - order_parameter[0]),
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the trajectory summary.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded simulate result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        simulate_spec: dict[str, Any] = dict(plan.parameters)
        source = _render_script(
            action_id=plan.action_id,
            simulate_spec=simulate_spec,
            order_parameter_final=float(result.outputs["order_parameter_final"]),
            order_parameter_mean=float(result.outputs["order_parameter_mean"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"simulate_{slug}.py",
            entrypoint=f"python simulate_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    simulate_spec: Mapping[str, Any],
    order_parameter_final: float,
    order_parameter_mean: float,
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio simulate action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Rebuilds the bounded K_nm/omega network, Trotter-evolves the dense\n"
        "statevector on the local simulator, and checks the Kuramoto order\n"
        "parameter summary the studio sealed. Values are simulator estimates at\n"
        "the stated step/Trotter resolution, agreeing to a numerical tolerance.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "from scpn_quantum_control.phase import QuantumKuramotoSolver\n\n"
        f"K_NM = {simulate_spec['K_nm']!r}\n"
        f"OMEGA = {simulate_spec['omega']!r}\n"
        f"T_MAX = {simulate_spec['t_max']!r}\n"
        f"DT = {simulate_spec['dt']!r}\n"
        f"TROTTER_PER_STEP = {simulate_spec['trotter_per_step']!r}\n"
        f"TROTTER_ORDER = {simulate_spec['trotter_order']!r}\n"
        f"EXPECTED_ORDER_PARAMETER_FINAL = {order_parameter_final!r}\n"
        f"EXPECTED_ORDER_PARAMETER_MEAN = {order_parameter_mean!r}\n"
        f"TOLERANCE = {_REPRODUCTION_TOLERANCE!r}\n\n\n"
        "def main() -> int:\n"
        '    """Re-evolve and verify the sealed order-parameter summary."""\n'
        "    solver = QuantumKuramotoSolver(\n"
        "        len(K_NM),\n"
        "        np.asarray(K_NM, dtype=np.float64),\n"
        "        np.asarray(OMEGA, dtype=np.float64),\n"
        "        trotter_order=TROTTER_ORDER,\n"
        "    )\n"
        "    trajectory = solver.run(T_MAX, DT, TROTTER_PER_STEP)\n"
        "    order_parameter = np.asarray(trajectory.R, dtype=np.float64)\n"
        "    final = float(order_parameter[-1])\n"
        "    mean = float(order_parameter.mean())\n"
        "    assert np.isclose(\n"
        "        final, EXPECTED_ORDER_PARAMETER_FINAL, atol=TOLERANCE, rtol=0.0\n"
        "    ), final\n"
        "    assert np.isclose(\n"
        "        mean, EXPECTED_ORDER_PARAMETER_MEAN, atol=TOLERANCE, rtol=0.0\n"
        "    ), mean\n"
        '    print(f"order_parameter_final={final} order_parameter_mean={mean} verified")\n'
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "SIMULATE_CLAIM_BOUNDARY",
    "SIMULATE_VERB",
    "SimulateActionHandler",
]
