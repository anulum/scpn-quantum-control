# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive compile handler
"""The ``compile`` executive action handler — bounded XY compile of a network.

The read-only ``compile`` verb compiles an arbitrary bounded ``K_nm``/``omega``
oscillator network into the studio's bit-exact XY compile unit
(:mod:`scpn_quantum_control.studio.recompute_kernel`). The handler validates the
network, builds the ``studio.xy-compile-recompute.v1`` unit, verifies it against
its own reference, and writes a standalone reproduction script.

The claim boundary is the bit-exact XY compile *decision path* only: the input
digest is recompute-verifiable (a browser can replay it through the WASM kernel).
It is not a physical ``K_nm`` claim, a continuous simulator value, or QPU
execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .recompute_kernel import (
    XY_COMPILE_RECOMPUTE_SCHEMA,
    build_xy_compile_recompute_unit,
    verify_xy_compile_recompute_unit,
)

COMPILE_VERB: Final[str] = "compile"
_DEFAULT_BACKEND: Final[str] = "python"
_MAX_NODES: Final[int] = 16
_MAX_TROTTER_STEPS: Final[int] = 64

COMPILE_CLAIM_BOUNDARY: Final[str] = (
    "bit-exact XY compile decision path for a bounded symmetric zero-diagonal "
    "K_nm/omega network; the input digest is recompute-verifiable in a browser, "
    "not a physical K_nm claim, a continuous simulator value, or QPU execution"
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


def _as_row(row: object) -> Sequence[Any]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        raise ValueError("each K_nm row must be a sequence")
    return row


def _normalise_compile(parameters: Mapping[str, Any]) -> dict[str, Any]:
    k_nm = _as_coupling_matrix(parameters.get("K_nm"))
    size = len(k_nm)
    raw_omega = parameters.get("omega")
    if not isinstance(raw_omega, Sequence) or isinstance(raw_omega, (str, bytes)):
        raise ValueError("omega must be a sequence")
    if len(raw_omega) != size:
        raise ValueError("omega length must match the number of nodes")
    omega = [_as_float("omega entry", entry) for entry in raw_omega]
    time = _as_float("time", parameters.get("time"))
    if time <= 0.0:
        raise ValueError("time must be positive")
    trotter_steps = _as_positive_int(
        "trotter_steps", parameters.get("trotter_steps"), maximum=_MAX_TROTTER_STEPS
    )
    trotter_order = parameters.get("trotter_order")
    if trotter_order not in (1, 2) or isinstance(trotter_order, bool):
        raise ValueError("trotter_order must be 1 or 2")
    return {
        "K_nm": k_nm,
        "omega": omega,
        "time": time,
        "trotter_steps": trotter_steps,
        "trotter_order": int(trotter_order),
    }


def _arrays(compile_spec: Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k_nm = np.asarray(compile_spec["K_nm"], dtype=np.float64)
    omega = np.asarray(compile_spec["omega"], dtype=np.float64)
    return k_nm, omega


class CompileActionHandler(ActionHandler):
    """Executive handler for the read-only ``compile`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"compile"``."""
        return COMPILE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the network and resolve a read-only compile plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The compile request; ``parameters`` must describe a bounded network
            (``K_nm``, ``omega``, ``time``, ``trotter_steps``, ``trotter_order``).
        contract : VerbContract
            The resolved ``compile`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the compile verb")
        compile_spec = _normalise_compile(request.parameters)
        steps = (
            f"validate the {len(compile_spec['K_nm'])}-node K_nm/omega network",
            "build the bit-exact XY compile recompute unit",
            "verify the unit against its own reference digest",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=COMPILE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=compile_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Compile the network into the bit-exact XY compile unit.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned compile network.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the input digest, recompute schema, and
            the self-verification verdict.
        """
        compile_spec: dict[str, Any] = dict(plan.parameters)
        k_nm, omega = _arrays(compile_spec)
        unit = build_xy_compile_recompute_unit(
            k_nm,
            omega,
            time=compile_spec["time"],
            trotter_steps=compile_spec["trotter_steps"],
            trotter_order=compile_spec["trotter_order"],
        )
        wire = unit.to_dict()
        verdict = verify_xy_compile_recompute_unit(unit)
        outputs = {
            "backend": plan.backend,
            "n_nodes": len(compile_spec["K_nm"]),
            "time": compile_spec["time"],
            "trotter_steps": compile_spec["trotter_steps"],
            "trotter_order": compile_spec["trotter_order"],
            "recompute_schema": XY_COMPILE_RECOMPUTE_SCHEMA,
            "verifiability_mode": wire["verifiability_mode"],
            "exactness_class": wire["exactness_class"],
            "input_sha256": wire["input_sha256"],
            "verified": verdict.value == "match",
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the XY compile unit.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded compile result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        compile_spec: dict[str, Any] = dict(plan.parameters)
        source = _render_script(
            action_id=plan.action_id,
            compile_spec=compile_spec,
            input_sha256=str(result.outputs["input_sha256"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"compile_{slug}.py",
            entrypoint=f"python compile_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(*, action_id: str, compile_spec: Mapping[str, Any], input_sha256: str) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio compile action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Rebuilds the bounded K_nm/omega network and recomputes the bit-exact XY\n"
        "compile unit, checking the input digest the studio sealed.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "from scpn_quantum_control.studio import (\n"
        "    build_xy_compile_recompute_unit,\n"
        "    verify_xy_compile_recompute_unit,\n"
        ")\n\n"
        f"K_NM = {compile_spec['K_nm']!r}\n"
        f"OMEGA = {compile_spec['omega']!r}\n"
        f"TIME = {compile_spec['time']!r}\n"
        f"TROTTER_STEPS = {compile_spec['trotter_steps']!r}\n"
        f"TROTTER_ORDER = {compile_spec['trotter_order']!r}\n"
        f"EXPECTED_INPUT_SHA256 = {input_sha256!r}\n\n\n"
        "def main() -> int:\n"
        '    """Recompute and verify the sealed XY compile unit."""\n'
        "    unit = build_xy_compile_recompute_unit(\n"
        "        np.asarray(K_NM, dtype=np.float64),\n"
        "        np.asarray(OMEGA, dtype=np.float64),\n"
        "        time=TIME,\n"
        "        trotter_steps=TROTTER_STEPS,\n"
        "        trotter_order=TROTTER_ORDER,\n"
        "    )\n"
        "    wire = unit.to_dict()\n"
        '    assert wire["input_sha256"] == EXPECTED_INPUT_SHA256, wire["input_sha256"]\n'
        '    assert verify_xy_compile_recompute_unit(unit).value == "match"\n'
        "    print(f\"input_sha256={wire['input_sha256']} verified\")\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "COMPILE_CLAIM_BOUNDARY",
    "COMPILE_VERB",
    "CompileActionHandler",
]
