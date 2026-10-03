# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive compile handler
"""Read-only supported program emission or bounded XY network compilation.

Requests containing only ``program_source`` emit the supported source IR through
the actual native Qiskit importer and an immutable source-bound plan. They retain
phase parameters, classical controls and readout and report emitted_not_executed.
Their reproduction scripts import source as data and never run imported Python.

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
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .compiler_trace import build_compiler_trace
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .program_authoring import compile_program_source
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
    trotter_order = _as_positive_int("trotter_order", parameters.get("trotter_order"), maximum=2)
    return {
        "K_nm": k_nm,
        "omega": omega,
        "time": time,
        "trotter_steps": trotter_steps,
        "trotter_order": trotter_order,
    }


def _arrays(compile_spec: Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k_nm = np.asarray(compile_spec["K_nm"], dtype=np.float64)
    omega = np.asarray(compile_spec["omega"], dtype=np.float64)
    return k_nm, omega


class CompileActionHandler(ActionHandler):
    """Executive handler for the read-only ``compile`` verb."""

    @property
    def verb(self) -> str:
        """The Studio verb owned by this handler."""
        return COMPILE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate a supported source or network and resolve its read-only plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The compile request; ``parameters`` must describe a bounded network
            (``K_nm``, ``omega``, ``time``, ``trotter_steps``, ``trotter_order``).
            Trotter steps and order must be integers, not booleans or floats;
            the supported orders are 1 and 2. Invalid input raises ValueError
            before a plan is returned; no order coercion is performed.
            Alternatively, the sole ``program_source`` parameter selects exact
            native source emission with the Python backend and no gate execution.
            Add ``compiler_trace=True`` and optional ``optimisation_level``
            (integer 0..3) for actual bounded static native pass qualification
            and textual MLIR. Effectful lowering is explicitly refused.
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
        if "program_source" in request.parameters:
            if set(request.parameters) - {
                "program_source",
                "compiler_trace",
                "optimisation_level",
            }:
                raise ValueError("program_source cannot be combined with network parameters")
            trace_mode = "compiler_trace" in request.parameters
            if trace_mode and request.parameters["compiler_trace"] is not True:
                raise ValueError(
                    "compiler_trace must be true when requesting native pass evidence"
                )
            level = request.parameters.get("optimisation_level", 2)
            if not trace_mode and "optimisation_level" in request.parameters:
                raise ValueError("optimisation_level requires compiler_trace")
            if type(level) is not int or not 0 <= level <= 3:
                raise ValueError("optimisation_level must be an integer between 0 and 3")
            if backend != "python":
                raise ValueError(
                    "program_source executive compilation requires the Python backend"
                )
            program = compile_program_source(request.parameters["program_source"])
            return ExecutionPlan(
                verb=self.verb,
                action_id=request.action_id,
                backend=backend,
                contract=contract,
                claim_boundary=(
                    "native static operator qualification with source-bound pass metadata and textual MLIR; emitted, not executed"
                    if trace_mode
                    else "supported source emission with exact operands, phases and readout; emitted, not executed"
                ),
                steps=(
                    "validate the bounded supported source",
                    "qualify the actual static native compiler pass and emit textual MLIR"
                    if trace_mode
                    else "emit immutable source-bound IR",
                    "write a reproducible trace export script"
                    if trace_mode
                    else "write a reproducible source import script",
                ),
                parameters=MappingProxyType(
                    {
                        "program_source": program.source,
                        "source_sha256": program.source_sha256,
                        **(
                            {"compiler_trace": True, "optimisation_level": level}
                            if trace_mode
                            else {}
                        ),
                    }
                ),
            )
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
        """Emit a sealed supported source record or build the XY compile unit.

        Parameters
        ----------
        plan : ExecutionPlan
            The sealed source identity or planned compile network.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the input digest, recompute schema, and
            the self-verification verdict.

        """
        if "program_source" in plan.parameters:
            program = compile_program_source(plan.parameters["program_source"])
            if program.source_sha256 != plan.parameters["source_sha256"]:
                raise ValueError("program source differs from its sealed compilation plan")
            if plan.parameters.get("compiler_trace") is True:
                trace = build_compiler_trace(
                    program.source, optimisation_level=plan.parameters["optimisation_level"]
                )
                return ExecutionResult(
                    status="succeeded",
                    outputs={
                        "backend": plan.backend,
                        "execution_status": "emitted_not_executed",
                        "source_sha256": program.source_sha256,
                        "compiler_trace": trace.to_dict(),
                    },
                )
            return ExecutionResult(
                status="succeeded",
                outputs={
                    "backend": plan.backend,
                    "execution_status": program.execution_status,
                    "source_sha256": program.source_sha256,
                    "program": program.to_dict(),
                },
            )
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
        """Write a standalone script reproducing source emission or the XY unit.

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
        if "program_source" in compile_spec:
            if compile_spec.get("compiler_trace") is True:
                source = (
                    '"""Reproduce native compiler evidence; emitted MLIR is not executed."""\n'
                    "from scpn_quantum_control.studio.compiler_trace import build_compiler_trace\n\n"
                    f"SOURCE = {compile_spec['program_source']!r}\n"
                    f"OPTIMISATION_LEVEL = {compile_spec['optimisation_level']!r}\n"
                    f"EXPECTED_SOURCE_SHA256 = {result.outputs['source_sha256']!r}\n\n"
                    f"EXPECTED_TRACE_SHA256 = {result.outputs['compiler_trace']['sha256']!r}\n\n"
                    "def main() -> int:\n"
                    '    """Emit actual native pass evidence for the original source."""\n'
                    "    trace = build_compiler_trace(SOURCE, optimisation_level=OPTIMISATION_LEVEL)\n"
                    "    assert trace.to_dict()['body']['source_sha256'] == EXPECTED_SOURCE_SHA256\n"
                    "    assert trace.to_dict()['sha256'] == EXPECTED_TRACE_SHA256\n"
                    "    print(trace.to_json())\n"
                    "    return 0\n\n"
                    "if __name__ == '__main__':\n"
                    "    raise SystemExit(main())\n"
                )
            else:
                source = (
                    '"""Reproduce supported source emission; this script does not execute gates."""\n'
                    "from scpn_quantum_control.studio.program_authoring import compile_program_source\n\n"
                    f"SOURCE = {compile_spec['program_source']!r}\n"
                    f"EXPECTED_SOURCE_SHA256 = {result.outputs['source_sha256']!r}\n\n"
                    "def main() -> int:\n"
                    '    """Emit the original program and verify its source identity."""\n'
                    "    program = compile_program_source(SOURCE)\n"
                    "    assert program.source_sha256 == EXPECTED_SOURCE_SHA256\n"
                    "    print(f'source_sha256={program.source_sha256} emitted_not_executed')\n"
                    "    return 0\n\n"
                    "if __name__ == '__main__':\n"
                    "    raise SystemExit(main())\n"
                )
        else:
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
