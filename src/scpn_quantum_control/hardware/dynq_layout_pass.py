# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DynQ Qiskit transpiler layout pass
"""Qiskit ``AnalysisPass`` adapter for the DynQ qubit mapper.

The framework-agnostic DynQ pipeline lives in :mod:`.qubit_mapper`; this module
is the thin Qiskit binding that reads calibration data from a
:class:`~qiskit.transpiler.Target`, runs :func:`.qubit_mapper.dynq_initial_layout`,
and publishes the chosen physical placement as ``property_set["layout"]`` so a
``PassManager`` can consume it exactly like ``TrivialLayout`` or ``SabreLayout``.

The pass is fail-closed: a target with no usable two-qubit gate error data, or a
circuit that no execution region can host, raises ``TranspilerError`` rather than
silently emitting a degraded layout.

Ref: Liu et al., arXiv:2601.19635 (2026) — DynQ.
"""

from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout, Target, TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass

from .qubit_mapper import dynq_initial_layout


def calibration_from_target(
    target: Target,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Extract DynQ calibration inputs from a Qiskit ``Target``.

    Two-qubit gate errors become the edge-error map; ``measure`` errors become
    the readout-error map. When several native operations report an error for
    the same qubit pair, the smallest (best-fidelity) value is kept. Instruction
    properties with a missing (``None``) error or global (``None``) qargs are
    skipped.

    Args:
        target: the device ``Target`` (e.g. ``backend.target``).

    Returns:
        ``(gate_errors, readout_errors)`` keyed as
        :func:`.qubit_mapper.dynq_initial_layout` expects — the gate-error keys
        are order-canonicalised ``(min, max)`` qubit pairs.
    """
    gate_errors: dict[tuple[int, int], float] = {}
    readout_errors: dict[int, float] = {}

    for name in target.operation_names:
        properties = target[name]
        for qargs, props in properties.items():
            if qargs is None or props is None:
                continue
            error = getattr(props, "error", None)
            if error is None:
                continue
            if len(qargs) == 2:
                pair = (min(qargs), max(qargs))
                previous = gate_errors.get(pair)
                if previous is None or error < previous:
                    gate_errors[pair] = float(error)
            elif len(qargs) == 1 and name == "measure":
                readout_errors[qargs[0]] = float(error)

    return gate_errors, readout_errors


class DynQLayoutPass(AnalysisPass):  # type: ignore[misc]  # qiskit is ignore_missing_imports, so AnalysisPass is Any; subclassing is intentional
    """Assign an initial layout with DynQ community-detected regions.

    Reads device calibration from a ``Target``, selects the highest-quality
    execution region that fits the circuit, and stores the mapping in
    ``property_set["layout"]`` (plus the full
    :class:`~.qubit_mapper.QubitMappingResult` under
    ``property_set["dynq_mapping_result"]`` for downstream inspection).

    Unlike Qiskit's per-circuit layout heuristics, the DynQ partition is a
    per-device property: the same ``Target`` yields a reproducible layout for
    any circuit width, so the pass is deterministic given ``seed``.
    """

    def __init__(
        self,
        target: Target,
        *,
        resolution: float = 1.0,
        min_qubits: int = 3,
        seed: int | None = None,
    ) -> None:
        """Configure the pass.

        Args:
            target: device ``Target`` supplying gate and readout errors.
            resolution: Louvain resolution (higher → smaller regions).
            min_qubits: minimum region size (DynQ uses 3).
            seed: Louvain seed for reproducibility.
        """
        super().__init__()
        self.target = target
        self.resolution = resolution
        self.min_qubits = min_qubits
        self.seed = seed

    @classmethod
    def from_backend(cls, backend: object, **kwargs: object) -> DynQLayoutPass:
        """Build the pass from a ``BackendV2`` by reading ``backend.target``.

        Args:
            backend: a ``BackendV2`` exposing a ``target`` attribute.
            **kwargs: forwarded to :class:`DynQLayoutPass`.
        """
        target = getattr(backend, "target", None)
        if not isinstance(target, Target):
            raise TranspilerError("DynQLayoutPass.from_backend: backend exposes no usable Target")
        return cls(target, **kwargs)  # type: ignore[arg-type]

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Compute and publish the DynQ layout for ``dag`` (analysis only)."""
        circuit_width = dag.num_qubits()

        gate_errors, readout_errors = calibration_from_target(self.target)
        if not gate_errors:
            raise TranspilerError("DynQLayoutPass: target reports no two-qubit gate error data")

        result = dynq_initial_layout(
            gate_errors,
            circuit_width,
            readout_errors=readout_errors or None,
            resolution=self.resolution,
            min_qubits=self.min_qubits,
            seed=self.seed,
        )
        if result is None:
            raise TranspilerError(
                f"DynQLayoutPass: no execution region can host a {circuit_width}-qubit circuit"
            )

        layout = Layout(
            {dag.qubits[index]: physical for index, physical in enumerate(result.initial_layout)}
        )
        self.property_set["layout"] = layout
        self.property_set["dynq_mapping_result"] = result
        return dag
