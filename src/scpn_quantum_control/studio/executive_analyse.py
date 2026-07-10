# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive analyse handler
"""The ``analyse`` executive action handler — synchronisation witness of a phase cloud.

The read-only ``analyse`` verb runs the synchronisation witness
(:mod:`scpn_quantum_control.phase.synchronisation_witness`) over a bounded
phase cloud: harmonic Kuramoto order parameters, geodesic phase distances, and
exact Vietoris--Rips persistent homology (H0/H1 persistence pairs, Betti
curves, persistent component count, dominant loop lifetime). The handler
validates the cloud, computes the witness record, and writes a standalone
reproduction script.

The claim boundary is a classical phase-configuration analysis: exact
persistent homology of the given finite phase cloud over the given filtration
thresholds. It is not a quantum-state measurement, a dynamical-evolution claim,
or a statement about any generating model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from ..phase.synchronisation_witness import (
    SyncWitnessRecord,
    phase_cloud_synchronisation_witness,
)
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .verbs import SYNC_ANALYSIS_SCHEMA

ANALYSE_VERB: Final[str] = "analyse"
_DEFAULT_BACKEND: Final[str] = "numpy"
_MAX_NODES: Final[int] = 32
_MAX_THRESHOLDS: Final[int] = 64
_REPRODUCTION_TOLERANCE: Final[float] = 1e-12

ANALYSE_CLAIM_BOUNDARY: Final[str] = (
    "exact synchronisation witness (harmonic order parameters, geodesic "
    "distances, Vietoris-Rips persistent homology) of a bounded finite phase "
    "cloud over the given filtration thresholds; not a quantum-state "
    "measurement, a dynamical-evolution claim, or a statement about any "
    "generating model"
)


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_float_sequence(name: str, value: object, *, maximum: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of real numbers")
    entries = [_as_float(f"{name} entry", entry) for entry in value]
    if not 2 <= len(entries) <= maximum:
        raise ValueError(f"{name} must have between 2 and {maximum} entries")
    return entries


def _as_positive_int(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _normalise_analyse(parameters: Mapping[str, Any]) -> dict[str, Any]:
    phases = _as_float_sequence("phases", parameters.get("phases"), maximum=_MAX_NODES)
    thresholds = _as_float_sequence(
        "thresholds", parameters.get("thresholds"), maximum=_MAX_THRESHOLDS
    )
    if any(later <= earlier for earlier, later in zip(thresholds, thresholds[1:], strict=False)):
        raise ValueError("thresholds must be strictly increasing")
    if thresholds[0] < 0.0:
        raise ValueError("thresholds must be non-negative")
    reference_scale = _as_float("reference_scale", parameters.get("reference_scale"))
    if not thresholds[0] <= reference_scale <= thresholds[-1]:
        raise ValueError("reference_scale must lie within the threshold range")
    expected_components = _as_positive_int(
        "expected_components",
        parameters.get("expected_components", 1),
        maximum=_MAX_NODES,
    )
    return {
        "phases": phases,
        "thresholds": thresholds,
        "reference_scale": reference_scale,
        "expected_components": expected_components,
    }


def _witness(analyse_spec: Mapping[str, Any], action_id: str) -> SyncWitnessRecord:
    return phase_cloud_synchronisation_witness(
        np.asarray(analyse_spec["phases"], dtype=np.float64),
        thresholds=np.asarray(analyse_spec["thresholds"], dtype=np.float64),
        reference_scale=float(analyse_spec["reference_scale"]),
        case_id=action_id,
        expected_components=int(analyse_spec["expected_components"]),
    )


class AnalyseActionHandler(ActionHandler):
    """Executive handler for the read-only ``analyse`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"analyse"``."""
        return ANALYSE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the phase cloud and resolve a read-only analysis plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The analyse request; ``parameters`` must describe a bounded phase
            cloud (``phases``, ``thresholds``, ``reference_scale``, optional
            ``expected_components``).
        contract : VerbContract
            The resolved ``analyse`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the analyse verb")
        analyse_spec = _normalise_analyse(request.parameters)
        steps = (
            f"validate the {len(analyse_spec['phases'])}-node phase cloud",
            "measure the first and second harmonic Kuramoto order parameters",
            "build the geodesic phase-distance matrix",
            "reduce the Vietoris-Rips boundary matrix for H0/H1 persistence",
            "read the Betti curves and the persistent component count",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=ANALYSE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=analyse_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Run the synchronisation witness over the phase cloud.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned analysis.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the witness summary.
        """
        analyse_spec: dict[str, Any] = dict(plan.parameters)
        record = _witness(analyse_spec, plan.action_id)
        outputs = {
            "backend": plan.backend,
            "n_nodes": record.n_nodes,
            "analysis_schema": SYNC_ANALYSIS_SCHEMA,
            "order_parameter": record.order_parameter,
            "order_parameter_harmonic2": record.order_parameter_harmonic2,
            "persistent_component_count": record.persistent_component_count,
            "expected_components": analyse_spec["expected_components"],
            "dominant_h1_persistence": record.dominant_h1_persistence,
            "betti0_curve": [int(count) for count in record.betti0_curve],
            "betti1_curve": [int(count) for count in record.betti1_curve],
            "reference_scale": record.reference_scale,
            "witness_passed": record.passed,
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the witness summary.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded analyse result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        analyse_spec: dict[str, Any] = dict(plan.parameters)
        source = _render_script(
            action_id=plan.action_id,
            analyse_spec=analyse_spec,
            order_parameter=float(result.outputs["order_parameter"]),
            component_count=int(result.outputs["persistent_component_count"]),
            dominant_h1=float(result.outputs["dominant_h1_persistence"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"analyse_{slug}.py",
            entrypoint=f"python analyse_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    analyse_spec: Mapping[str, Any],
    order_parameter: float,
    component_count: int,
    dominant_h1: float,
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio analyse action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Recomputes the synchronisation witness (order parameter, persistent\n"
        "component count, dominant H1 lifetime) of the sealed phase cloud and\n"
        "checks it against the summary the studio sealed.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "from scpn_quantum_control.phase.synchronisation_witness import (\n"
        "    phase_cloud_synchronisation_witness,\n"
        ")\n\n"
        f"PHASES = {analyse_spec['phases']!r}\n"
        f"THRESHOLDS = {analyse_spec['thresholds']!r}\n"
        f"REFERENCE_SCALE = {analyse_spec['reference_scale']!r}\n"
        f"EXPECTED_COMPONENTS = {analyse_spec['expected_components']!r}\n"
        f"EXPECTED_ORDER_PARAMETER = {order_parameter!r}\n"
        f"EXPECTED_COMPONENT_COUNT = {component_count!r}\n"
        f"EXPECTED_DOMINANT_H1 = {dominant_h1!r}\n"
        f"TOLERANCE = {_REPRODUCTION_TOLERANCE!r}\n\n\n"
        "def main() -> int:\n"
        '    """Recompute and verify the sealed synchronisation witness."""\n'
        "    record = phase_cloud_synchronisation_witness(\n"
        "        np.asarray(PHASES, dtype=np.float64),\n"
        "        thresholds=np.asarray(THRESHOLDS, dtype=np.float64),\n"
        "        reference_scale=REFERENCE_SCALE,\n"
        "        expected_components=EXPECTED_COMPONENTS,\n"
        "    )\n"
        "    assert abs(record.order_parameter - EXPECTED_ORDER_PARAMETER) <= TOLERANCE\n"
        "    assert record.persistent_component_count == EXPECTED_COMPONENT_COUNT\n"
        "    assert abs(record.dominant_h1_persistence - EXPECTED_DOMINANT_H1) <= TOLERANCE\n"
        "    print(\n"
        '        f"order_parameter={record.order_parameter} "\n'
        '        f"components={record.persistent_component_count} verified"\n'
        "    )\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "ANALYSE_CLAIM_BOUNDARY",
    "ANALYSE_VERB",
    "AnalyseActionHandler",
]
