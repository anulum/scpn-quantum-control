# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Scientific problem parameter adapter
"""Bind original problems to immutable scientific identity and workspace schemas."""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp

from .kuramoto_core import KuramotoProblem, compile_hamiltonian, validate_scientific_design
from .scientific_design import ScientificDesign
from .studio_workspace.canonical import canonical_digest
from .studio_workspace.contracts import ParameterSpec, validate_parameter_binding


def _problem_snapshot(problem: KuramotoProblem) -> KuramotoProblem:
    """Copy the original owner into independent immutable numeric byte buffers."""
    snapshot = KuramotoProblem(problem.K_nm, problem.omega, problem.metadata)
    for key in ("K_nm", "omega"):
        values = getattr(snapshot, key)
        object.__setattr__(
            snapshot, key, np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
        )
    return snapshot


@dataclass(frozen=True, slots=True)
class ScientificProblemParameters:
    """Scientific companion over an owned snapshot of the original problem.

    Parameters
    ----------
    problem
        Existing validated coupling/frequency owner. Its raw codec is preserved.
    design
        Explicit immutable scientific model and supplied design declarations.

    Notes
    -----
    This adapter owns identity and input projection; it supplies no solver or
    unit conversion. Existing workspace schemas/encoding and numerical owners
    are reused. Physical coordinates do not imply calibrated hardware evidence.

    """

    problem: KuramotoProblem
    design: ScientificDesign
    schema: ClassVar[str] = "scientific_problem.v1"

    def __post_init__(self) -> None:
        """Validate the binding and capture independent original-owner arrays.

        Raises
        ------
        ValueError
            Model, state, topology, observable, units or objective are incompatible.

        """
        snapshot, design = self.problem, self.design
        validate_scientific_design(snapshot, design)
        object.__setattr__(self, "problem", snapshot)
        object.__setattr__(self, "design", design)

    def __getattribute__(self, name: str) -> object:
        """Read independent declarations without exposing stored NumPy headers.

        Parameters
        ----------
        name
            Attribute to read; problem/design return defensive snapshots.

        Returns
        -------
        object
            Independent problem/design or the original requested attribute.

        Notes
        -----
        Read-only arrays still allow shape/dtype header mutation. Copying these
        two public records prevents that mutation from rebinding saved identity.

        """
        value = object.__getattribute__(self, name)
        if name == "problem" and isinstance(value, KuramotoProblem):
            return _problem_snapshot(value)
        if name == "design" and isinstance(value, ScientificDesign):
            return replace(value)
        return value

    def _arrays(self) -> dict[str, tuple[NDArray[np.float64], str]]:
        """Return numeric parameters in declared producer order and exact units."""
        design = self.design
        arrays = {
            "omega": (self.problem.omega, design.units.frequency),
            "K_nm": (self.problem.K_nm, design.units.coupling),
            "initial_state_real": (
                np.asarray(design.initial_state.real, dtype=np.float64),
                design.units.state,
            ),
            "observable_weights": (design.observable_weights, design.units.observable),
        }
        if design.model == "quantum_xy":
            arrays["initial_state_imag"] = (
                np.asarray(design.initial_state.imag, dtype=np.float64),
                design.units.state,
            )
        if design.history_times is not None and design.history_states is not None:
            arrays["history_times"] = (design.history_times, design.units.time)
            arrays["history_states"] = (design.history_states, design.units.state)
        return arrays

    def parameter_values(self) -> dict[str, object]:
        """Export fresh exact float64 payloads accepted by workspace readers.

        Returns
        -------
        dict[str, object]
            Named dtype/shape/row-major binary64-hex payloads, without conversion.

        """
        return {
            key: {
                "dtype": "float64",
                "shape": list(array.shape),
                "values": [struct.pack(">d", float(value)).hex() for value in array.flat],
            }
            for key, (array, _) in self._arrays().items()
        }

    def to_dict(self) -> dict[str, object]:
        """Export a fresh versioned semantic companion, separate from raw data.

        Returns
        -------
        dict[str, object]
            Exact model, convention, topology, units, state/history and objective.

        """
        design = self.design
        return {
            "schema": self.schema,
            "body": {
                "model": design.model,
                "normalisation": design.normalisation,
                "coordinate_space": design.coordinate_space,
                "topology": [list(edge) for edge in design.topology],
                "units": {key: unit for key, (_, unit) in self._arrays().items()},
                "time_unit": design.units.time,
                "observable": design.observable,
                "objective": {
                    "kind": design.objective.kind,
                    "target": design.objective.target,
                    "unit": design.objective.unit,
                },
                "trainable": list(design.trainable),
                "parameters": self.parameter_values(),
            },
            "extensions": {},
        }

    @property
    def identity(self) -> str:
        """Return schema-bound semantic identity including all scientific inputs.

        Returns
        -------
        str
            Lowercase SHA-256 of the existing typed canonical encoding.

        """
        document = self.to_dict()
        return canonical_digest(
            self.schema, {"body": document["body"], "extensions": document["extensions"]}
        )

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        """Expose original workspace parameter schemas bound to this identity.

        Returns
        -------
        tuple[ParameterSpec, ...]
            Immutable float64 specifications in exact declared numeric order.

        """
        values = self.parameter_values()
        identity = self.identity
        specs = tuple(
            ParameterSpec(
                {
                    "key": key,
                    "dtype": "float64",
                    "shape": list(array.shape),
                    "unit": unit,
                    "domain": {"kind": "finite"},
                    "default_source": f"{self.schema}:{identity}#{key}",
                    "trainable": key in self.design.trainable,
                    "dependency_keys": [],
                }
            )
            for key, (array, unit) in self._arrays().items()
        )
        for spec in specs:
            key = str(spec.body["key"])
            validate_parameter_binding(spec, values[key], str(spec.body["unit"]))
        return specs

    def effective_problem(self) -> KuramotoProblem:
        """Project the explicit convention to the existing numerical problem.

        Returns
        -------
        KuramotoProblem
            Independent owner with pairwise coefficients and unchanged units.

        Notes
        -----
        Population-mean phase input weights become ``K/N``. This is an explicit
        input projection, not an automatic phase-to-spin transformation.

        """
        coupling = self.problem.K_nm
        if self.design.normalisation == "population_mean":
            coupling = coupling / self.problem.n_oscillators
        metadata = dict(self.problem.metadata)
        metadata.update(
            scientific_problem_identity=self.identity,
            model=self.design.model,
            normalisation=self.design.normalisation,
            coordinate_space=self.design.coordinate_space,
        )
        return KuramotoProblem(coupling, self.problem.omega, metadata)

    def compile_hamiltonian(self) -> SparsePauliOp:
        """Compile only an explicitly declared XY spin model with original code.

        Returns
        -------
        SparsePauliOp
            Existing compiler's Hamiltonian, with the declared coefficient units.

        Raises
        ------
        ValueError
            The declared model is phase dynamics, which has no implicit spin map.

        """
        if self.design.model != "quantum_xy":
            raise ValueError("phase_kuramoto has no implicit quantum-spin equivalence")
        return compile_hamiltonian(self.effective_problem())
