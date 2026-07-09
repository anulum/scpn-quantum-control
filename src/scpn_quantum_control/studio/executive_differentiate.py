# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive differentiate handler
"""The ``differentiate`` executive action handler — the first spine plugin.

Given a bounded *rational* scalar program (parameters plus ``mul``/``add``
operations, no transcendentals so value and gradient are exact and platform
reproducible), this handler:

1. plans the run under the read-only ``differentiate`` contract;
2. executes it through the compiled ``scpn_quantum_engine`` effect-IR replay,
   returning the reverse-mode value and gradient, and independently cross-checks
   that gradient against central finite differences of the same value function;
3. writes a standalone Python script that reconstructs the program and
   reproduces the value and gradient from the engine.

The claim boundary is deliberately narrow: exact reverse-mode differentiation of
a bounded rational program, cross-checked against finite differences — not
transcendental, linear-algebra, unbounded, provider, or hardware differentiation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .executive import (
    ActionHandler,
    ActionRegistry,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)

DIFFERENTIATE_VERB: Final[str] = "differentiate"
_DEFAULT_BACKEND: Final[str] = "python"
_FINITE_DIFFERENCE_STEP: Final[float] = 1.0e-5
_GRADIENT_TOLERANCE: Final[float] = 1.0e-6

DIFFERENTIATE_CLAIM_BOUNDARY: Final[str] = (
    "exact reverse-mode value and gradient of a bounded rational scalar program "
    "(mul/add over named parameters and numeric literals), cross-checked against "
    "central finite differences of the same value; not transcendental, "
    "linear-algebra, unbounded, provider, or hardware differentiation"
)

_Operation = Mapping[str, Any]
_NormalisedProgram = dict[str, Any]


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _normalise_program(parameters: Mapping[str, Any]) -> _NormalisedProgram:
    raw_inputs = parameters.get("inputs")
    raw_operations = parameters.get("operations")
    output = parameters.get("output")
    if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
        raise ValueError("inputs must be a sequence of [name, value] pairs")
    if not isinstance(raw_operations, Sequence) or isinstance(raw_operations, (str, bytes)):
        raise ValueError("operations must be a sequence of operation mappings")
    if not isinstance(output, str) or not output:
        raise ValueError("output must be a non-empty name")
    if len(raw_inputs) < 1:
        raise ValueError("at least one input parameter is required")
    if len(raw_operations) < 1:
        raise ValueError("at least one operation is required")

    names: dict[str, str] = {}
    inputs: list[tuple[str, float]] = []
    for index, pair in enumerate(raw_inputs):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError("each input must be a [name, value] pair")
        name = pair[0]
        if not isinstance(name, str) or not name:
            raise ValueError("input name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate input name {name!r}")
        names[name] = f"%{index}"
        inputs.append((name, _as_float(f"input {name!r}", pair[1])))

    operations: list[dict[str, Any]] = []
    for op_index, operation in enumerate(raw_operations):
        target = f"%{len(inputs) + op_index}"
        op, refs, into = _normalise_operation(operation, names)
        names[into] = target
        operations.append({"op": op, "inputs": refs, "into": into, "target": target})

    if output != operations[-1]["into"]:
        raise ValueError("output must name the result of the last operation")
    return {"inputs": inputs, "operations": operations, "output": output}


def _normalise_operation(
    operation: object, names: Mapping[str, str]
) -> tuple[str, list[str], str]:
    if not isinstance(operation, Mapping):
        raise ValueError("each operation must be a mapping")
    op = operation.get("op")
    refs = operation.get("inputs")
    into = operation.get("into")
    if op not in ("mul", "add"):
        raise ValueError("operation op must be 'mul' or 'add'")
    if not isinstance(into, str) or not into:
        raise ValueError("operation into must be a non-empty name")
    if into in names:
        raise ValueError(f"operation into {into!r} shadows an existing name")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or len(refs) != 2:
        raise ValueError("operation inputs must be two references")
    resolved: list[str] = []
    for ref in refs:
        if not isinstance(ref, str) or not ref:
            raise ValueError("operation input reference must be a non-empty string")
        resolved.append(_resolve_reference(ref, names))
    return op, resolved, into


def _resolve_reference(ref: str, names: Mapping[str, str]) -> str:
    if ref in names:
        return names[ref]
    try:
        float(ref)
    except ValueError as exc:
        raise ValueError(f"unknown operation reference {ref!r}") from exc
    return ref


def build_effect_ir(program: _NormalisedProgram) -> tuple[str, tuple[str, ...], list[float]]:
    """Build the canonical effect-IR, parameter targets, and input values.

    Parameters
    ----------
    program : dict
        A normalised program from :func:`_normalise_program`.

    Returns
    -------
    tuple
        ``(effect_ir_json, parameter_targets, input_values)`` — the serialised
        effect-IR the engine parses, the ordered parameter SSA targets, and the
        ordered input values.
    """
    inputs: list[tuple[str, float]] = program["inputs"]
    operations: list[dict[str, Any]] = program["operations"]
    effects: list[dict[str, Any]] = []
    for index, (name, _value) in enumerate(inputs):
        effects.append(
            {
                "index": index,
                "kind": "parameter",
                "target": f"%{index}",
                "inputs": [name],
                "version": 0,
                "ordering": index,
                "operation": "parameter",
            }
        )
    for op_index, operation in enumerate(operations):
        index = len(inputs) + op_index
        effects.append(
            {
                "index": index,
                "kind": "pure",
                "target": operation["target"],
                "inputs": list(operation["inputs"]),
                "version": 0,
                "ordering": index,
                "operation": operation["op"],
            }
        )
    ir = {
        "format": "program_ad_effect_ir.v1",
        "ssa_values": [
            {
                "name": f"%{index}",
                "producer": index,
                "version": 0,
                "shape": [],
                "dtype": "float64",
                "effect": index,
            }
            for index in range(len(effects))
        ],
        "effects": effects,
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": [index * 2 for index in range(len(effects))],
    }
    parameter_targets = tuple(f"%{index}" for index in range(len(inputs)))
    values = [value for _name, value in inputs]
    return json.dumps(ir, sort_keys=True, separators=(",", ":")), parameter_targets, values


def _engine_value_and_gradient(ir: str, values: Sequence[float]) -> dict[str, Any]:
    import scpn_quantum_engine as engine

    raw = engine.program_ad_effect_ir_interpret_value_and_gradient(ir, list(values))
    result: dict[str, Any] = json.loads(raw)
    if not result.get("supported") or result.get("value") is None:
        raise ValueError(f"program is not a supported bounded rational replay: {result}")
    return result


def _engine_value(ir: str, values: Sequence[float]) -> float:
    return float(_engine_value_and_gradient(ir, values)["value"])


def _finite_difference_gradient(ir: str, values: Sequence[float]) -> list[float]:
    gradient: list[float] = []
    for index in range(len(values)):
        plus = list(values)
        minus = list(values)
        plus[index] += _FINITE_DIFFERENCE_STEP
        minus[index] -= _FINITE_DIFFERENCE_STEP
        derivative = (_engine_value(ir, plus) - _engine_value(ir, minus)) / (
            2.0 * _FINITE_DIFFERENCE_STEP
        )
        gradient.append(derivative)
    return gradient


class DifferentiateActionHandler(ActionHandler):
    """Executive handler for the read-only ``differentiate`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"differentiate"``."""
        return DIFFERENTIATE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the rational program and resolve a read-only plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The differentiate request; ``parameters`` must describe a bounded
            rational program (``inputs``, ``operations``, ``output``).
        contract : VerbContract
            The resolved ``differentiate`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the differentiate verb")
        program = _normalise_program(request.parameters)
        steps = (
            f"build effect-IR for {len(program['operations'])} rational operations",
            f"interpret reverse-mode value and gradient on the {backend} backend",
            "cross-check the gradient against central finite differences",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=DIFFERENTIATE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=program,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Interpret the program and cross-check its gradient.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned rational program.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the value, reverse-mode gradient, the
            finite-difference gradient, the maximum absolute disagreement, and
            the verification verdict.
        """
        program: _NormalisedProgram = dict(plan.parameters)
        ir, parameter_targets, values = build_effect_ir(program)
        interpreted = _engine_value_and_gradient(ir, values)
        gradient = [float(component) for component in interpreted["gradient"]]
        finite_difference = _finite_difference_gradient(ir, values)
        max_abs_error = max(
            (abs(a - b) for a, b in zip(gradient, finite_difference, strict=True)),
            default=0.0,
        )
        outputs = {
            "backend": plan.backend,
            "value": float(interpreted["value"]),
            "gradient": gradient,
            "parameter_targets": list(parameter_targets),
            "finite_difference_gradient": finite_difference,
            "finite_difference_step": _FINITE_DIFFERENCE_STEP,
            "max_abs_error": max_abs_error,
            "tolerance": _GRADIENT_TOLERANCE,
            "verified": max_abs_error <= _GRADIENT_TOLERANCE,
            "effect_ir": ir,
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the value and gradient.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded execution result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        program: _NormalisedProgram = dict(plan.parameters)
        _ir, _targets, values = build_effect_ir(program)
        ir = str(result.outputs["effect_ir"])
        value = float(result.outputs["value"])
        gradient = [float(component) for component in result.outputs["gradient"]]
        source = _render_script(
            action_id=plan.action_id,
            ir=ir,
            values=values,
            value=value,
            gradient=gradient,
        )
        return build_generated_script(
            filename=f"reproduce_{_safe_slug(plan.action_id)}.py",
            entrypoint=f"python reproduce_{_safe_slug(plan.action_id)}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    ir: str,
    values: Sequence[float],
    value: float,
    gradient: Sequence[float],
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio differentiate action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Recomputes the reverse-mode value and gradient of a bounded rational program\n"
        "from the compiled scpn_quantum_engine effect-IR replay and checks them against\n"
        "the values the studio sealed.\n"
        '"""\n\n'
        "import json\n\n"
        "import scpn_quantum_engine as engine\n\n"
        f"EFFECT_IR = {ir!r}\n"
        f"INPUTS = {list(values)!r}\n"
        f"EXPECTED_VALUE = {value!r}\n"
        f"EXPECTED_GRADIENT = {list(gradient)!r}\n\n\n"
        "def main() -> int:\n"
        '    """Recompute and verify the sealed value and gradient."""\n'
        "    raw = engine.program_ad_effect_ir_interpret_value_and_gradient(EFFECT_IR, INPUTS)\n"
        "    result = json.loads(raw)\n"
        '    value = float(result["value"])\n'
        '    gradient = [float(component) for component in result["gradient"]]\n'
        "    assert value == EXPECTED_VALUE, (value, EXPECTED_VALUE)\n"
        "    assert gradient == EXPECTED_GRADIENT, (gradient, EXPECTED_GRADIENT)\n"
        '    print(f"value={value} gradient={gradient} verified")\n'
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


def default_registry() -> ActionRegistry:
    """Return an action registry with the differentiate handler registered.

    Returns
    -------
    ActionRegistry
        A fresh registry ready to run ``differentiate`` actions.
    """
    registry = ActionRegistry()
    registry.register(DifferentiateActionHandler())
    return registry


__all__ = [
    "DIFFERENTIATE_CLAIM_BOUNDARY",
    "DIFFERENTIATE_VERB",
    "DifferentiateActionHandler",
    "build_effect_ir",
    "default_registry",
]
