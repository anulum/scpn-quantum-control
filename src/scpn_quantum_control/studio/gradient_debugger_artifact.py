# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed units for the Studio gradient debugger
"""Build, write and check the committed units the Studio gradient debugger replays.

Every value in the artefact comes from an existing source owner:

* value, reverse-mode gradient and the central finite-difference cross-check
  come from the ``differentiate`` action (:class:`DifferentiateActionHandler`);
* the browser replay input is packed by :func:`encode_replay_input`, so the
  shipped WASM kernel recomputes the same program;
* gradients under each trainable mask come from
  :func:`parameter_shift_gradient`, which sets frozen parameters to zero;
* the RY/Z unit links the owner's identical-circuit reference row with its
  framework versions and circuit fingerprint.

The browser never computes a gradient itself; it shows these values and
recomputes the program through the WASM kernel.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ..differentiable_parameter_contracts import Parameter
from ..differentiable_parameter_shift import parameter_shift_gradient
from .committed_data import differences, run_committed_data_cli, serialise
from .executive import ExecutiveRequest, run_action
from .executive_differentiate import (
    DIFFERENTIATE_CLAIM_BOUNDARY,
    DIFFERENTIATE_VERB,
    default_registry,
)
from .program_ad_replay_artifact import encode_replay_input

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
GRADIENT_DEBUGGER_SCHEMA: Final[str] = "scpn_qc_studio_gradient_debugger_units_v1"
GRADIENT_DEBUGGER_ARTIFACT_ID: Final[str] = "studio-gradient-debugger-units-20261030"
DEFAULT_GRADIENT_DEBUGGER_JSON_PATH: Final[Path] = Path(
    "data/studio/gradient_debugger_units_20261030.json"
)
REFERENCE_PATH: Final[Path] = Path(
    "data/differentiable_phase_qnode/identical_circuit_gradient_comparison_20260616.json"
)
PLAN_PATH: Final[Path] = Path(
    "data/differentiable_phase_qnode/gradient_plan_explanations_20260709.json"
)
_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.studio.gradient_debugger_artifact --write"
)

#: How each committed unit is built; programs use the ``differentiate`` request shape.
_UNITS: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "unit_id": "ry-z-expectation",
        "title": "RY(θ) then measure Z",
        "objective": "⟨Z⟩ = cos θ",
        "cell_id": "ry::pauli_expectation::statevector::grad::native",
        "reference_case": "single_ry_z_expectation_exact_state",
        "program": {
            "inputs": [["theta", 0.4]],
            "operations": [{"op": "cos", "inputs": ["theta"], "into": "z"}],
            "output": "z",
        },
        "objective_fn": lambda values: math.cos(values[0]),
    },
    {
        "unit_id": "two-angle-mix",
        "title": "Two angles, weighted",
        "objective": "f = cos a + 0.25 sin b",
        "cell_id": None,
        "reference_case": None,
        "program": {
            "inputs": [["a", 0.3], ["b", -1.1]],
            "operations": [
                {"op": "cos", "inputs": ["a"], "into": "ca"},
                {"op": "sin", "inputs": ["b"], "into": "sb"},
                {"op": "mul", "inputs": ["sb", "0.25"], "into": "qb"},
                {"op": "add", "inputs": ["ca", "qb"], "into": "f"},
            ],
            "output": "f",
        },
        "objective_fn": lambda values: math.cos(values[0]) + 0.25 * math.sin(values[1]),
    },
)

GRADIENT_DEBUGGER_CLAIM_BOUNDARY: Final[str] = (
    "committed bounded scalar programs differentiated by the differentiate action "
    "and replayed by the browser WASM kernel; masked gradients from the "
    "parameter-shift rule; the RY/Z unit links an exact-state identical-circuit "
    "reference; no browser execution of JAX, Torch, provider or hardware runtimes"
)


def _differentiate(unit_id: str, program: Mapping[str, Any]) -> dict[str, Any]:
    request = ExecutiveRequest(
        verb=DIFFERENTIATE_VERB, action_id=f"gradient-debugger-{unit_id}", parameters=program
    )
    record = run_action(request, registry=default_registry())
    if record.result is None or record.result.status != "succeeded":
        raise ValueError(f"differentiate refused unit {unit_id!r}")
    outputs: dict[str, Any] = dict(record.result.outputs)
    if outputs.get("verified") is not True:
        raise ValueError(f"finite differences disagree for unit {unit_id!r}")
    return outputs


def _masked_gradients(
    objective: Callable[[Sequence[float]], float], names: Sequence[str], values: Sequence[float]
) -> list[dict[str, Any]]:
    def scalar(vector: NDArray[np.float64]) -> float:
        return float(objective([float(item) for item in vector]))

    rows: list[dict[str, Any]] = []
    for mask in itertools.product((True, False), repeat=len(names)):
        parameters = [
            Parameter(name, trainable) for name, trainable in zip(names, mask, strict=True)
        ]
        gradient = parameter_shift_gradient(scalar, list(values), parameters=parameters)
        rows.append({"trainable": list(mask), "gradient": [float(item) for item in gradient]})
    return rows


def _reference(case_id: str, root: Path) -> dict[str, Any]:
    payload = json.loads((root / REFERENCE_PATH).read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["case_id"] == case_id]
    if not rows:
        raise ValueError(f"reference case {case_id!r} is not in the comparison artefact")
    first = rows[0]
    return {
        "artifact_id": payload["artifact_id"],
        "case_id": case_id,
        "source_of_truth": payload["source_of_truth"],
        "parameter_values": list(first["parameter_values"]),
        "value": first["scpn_value"],
        "gradient": list(first["scpn_gradient"]),
        "frameworks": [
            {
                "backend": row["backend"],
                "dependency_versions": dict(row["dependency_versions"]),
                "circuit_fingerprint": row["circuit_fingerprint"],
                "evaluations": row["evaluations"],
                "value": row["backend_value"],
                "gradient": list(row["backend_gradient"]),
            }
            for row in rows
        ],
    }


def _plan_cells(root: Path) -> set[str]:
    payload = json.loads((root / PLAN_PATH).read_text(encoding="utf-8"))
    return {row["cell_id"] for row in payload["explanations"]}


def _unit(spec: Mapping[str, Any], root: Path, cells: set[str]) -> dict[str, Any]:
    unit_id = str(spec["unit_id"])
    program: Mapping[str, Any] = spec["program"]
    names = [str(pair[0]) for pair in program["inputs"]]
    values = [float(pair[1]) for pair in program["inputs"]]
    outputs = _differentiate(unit_id, program)
    if spec["cell_id"] is not None and spec["cell_id"] not in cells:
        raise ValueError(f"unit {unit_id!r} links an undeclared plan cell")
    replay = encode_replay_input(outputs["effect_ir"], values)
    reference = (
        None if spec["reference_case"] is None else _reference(spec["reference_case"], root)
    )
    return {
        "unit_id": unit_id,
        "title": spec["title"],
        "objective": spec["objective"],
        "cell_id": spec["cell_id"],
        "program": json.loads(json.dumps(program)),
        "parameters": [
            {"name": name, "target": target, "value": value}
            for name, target, value in zip(
                names, outputs["parameter_targets"], values, strict=True
            )
        ],
        "effect_ir": outputs["effect_ir"],
        "input_hex": replay.hex(),
        "input_sha256": f"sha256:{sha256(replay).hexdigest()}",
        "expected": {"value": outputs["value"], "gradient": list(outputs["gradient"])},
        "finite_difference": {
            "gradient": list(outputs["finite_difference_gradient"]),
            "step": outputs["finite_difference_step"],
            "max_abs_error": outputs["max_abs_error"],
            "tolerance": outputs["tolerance"],
            "verified": outputs["verified"],
        },
        "masks": _masked_gradients(spec["objective_fn"], names, values),
        "reference": reference,
        "local_runner": {
            "verb": DIFFERENTIATE_VERB,
            "action_id": f"gradient-debugger-{unit_id}",
            "params": json.loads(json.dumps(program)),
        },
    }


def build_gradient_debugger_artifact(root: Path = REPO_ROOT) -> dict[str, object]:
    """Build the committed gradient-debugger units from their source owners.

    Parameters
    ----------
    root
        Repository root holding the reference and plan artefacts.

    Returns
    -------
    dict[str, object]
        JSON-ready payload with one entry per unit.

    Raises
    ------
    ValueError
        If the differentiate action refuses a unit, its finite differences
        disagree, a linked plan cell is undeclared or a reference case is absent.

    """
    cells = _plan_cells(root)
    return {
        "schema": GRADIENT_DEBUGGER_SCHEMA,
        "artifact_id": GRADIENT_DEBUGGER_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "claim_boundary": GRADIENT_DEBUGGER_CLAIM_BOUNDARY,
        "differentiate_claim_boundary": DIFFERENTIATE_CLAIM_BOUNDARY,
        "plan_source": PLAN_PATH.as_posix(),
        "reference_source": REFERENCE_PATH.as_posix(),
        "units": [_unit(spec, root, cells) for spec in _UNITS],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed gradient-debugger units.

    Parameters
    ----------
    argv
        Optional argument vector, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the committed file differs from a fresh
        build, ``2`` when it cannot be read.

    """
    return run_committed_data_cli(
        argv,
        build=build_gradient_debugger_artifact,
        default_path=REPO_ROOT / DEFAULT_GRADIENT_DEBUGGER_JSON_PATH,
        label="gradient-debugger units",
        description=__doc__,
    )


__all__ = [
    "DEFAULT_GRADIENT_DEBUGGER_JSON_PATH",
    "GRADIENT_DEBUGGER_ARTIFACT_ID",
    "GRADIENT_DEBUGGER_CLAIM_BOUNDARY",
    "GRADIENT_DEBUGGER_SCHEMA",
    "build_gradient_debugger_artifact",
    "differences",
    "main",
    "serialise",
]


if __name__ == "__main__":
    raise SystemExit(main())
