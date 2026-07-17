#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tensor-network baselines for the flagship workloads
"""Bounded-bond-dimension MPS baselines for the March flagship workloads.

The gPEPS-rebuttal defence (KIMI-11): instead of waiting for an external
reviewer to demonstrate that the flagship 16-qubit UPDE and 8-oscillator
Kuramoto workloads are classically simulable, this script quantifies it
ourselves and commits the numbers. For each workload it rebuilds the exact
logical Trotter body the hardware campaign executed, computes the exact
statevector reference for the order parameter R = |Σ⟨X⟩ + iΣ⟨Y⟩|/n, then
sweeps an Aer matrix-product-state simulation of the *identical circuit*
over bounded bond dimensions χ and reports R(χ), the absolute R error, the
worst per-qubit ⟨X⟩/⟨Y⟩ error, and wall time per χ.

The repository has never claimed quantum advantage for these workloads;
this artefact states their classical simulability with concrete numbers
(the smallest χ reproducing R to the preregistered tolerance), so the
honesty boundary is defended by committed arithmetic rather than by
assertion. Pure classical computation — no QPU access, no spend.

The Trotter body is expanded through its attached LieTrotter definition
before transpilation (re-synthesis would reorder the exponentials and
silently change the simulated unitary at O(t²)); the basis translation is
exact, and save instructions are appended after translation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scpn_quantum_control.bridge.knm_hamiltonian import (  # noqa: E402
    OMEGA_N_16,
    build_knm_paper27,
)
from scpn_quantum_control.hardware._experiment_helpers import _build_evo_base  # noqa: E402

SCHEMA = "scpn.tn_baseline_flagship_workloads.v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "tn_baseline_flagship"
DEFAULT_BOND_DIMENSIONS = (1, 2, 4, 8, 16, 32)
DEFAULT_R_TOLERANCE = 1e-3
BASIS_GATES = ["cx", "rz", "ry", "rx", "h", "x", "sdg", "id"]

CLAIM_BOUNDARY = (
    "Classical-simulability quantification only: the repository has never "
    "claimed quantum advantage for these workloads, and this artefact "
    "commits the concrete bounded-χ MPS numbers that pre-empt a "
    "gPEPS-style rebuttal. No statement about hardware quality or any "
    "other workload follows from it."
)


@dataclass(frozen=True)
class FlagshipWorkload:
    """One flagship logical workload, exactly as the March campaign built it."""

    name: str
    n_qubits: int
    evolution_time: float
    trotter_reps: int
    parent_artifact: str

    def body(self) -> QuantumCircuit:
        """Rebuild the logical Trotter body (ry init + LieTrotter evolution)."""
        coupling = build_knm_paper27(L=self.n_qubits)
        omega = OMEGA_N_16[: self.n_qubits]
        return _build_evo_base(
            self.n_qubits, coupling, omega, self.evolution_time, self.trotter_reps
        )


WORKLOADS: dict[str, FlagshipWorkload] = {
    "upde16": FlagshipWorkload(
        name="upde16",
        n_qubits=16,
        evolution_time=0.05,
        trotter_reps=1,
        parent_artifact="results/ibm_hardware_2026-03-28/upde_16_dd.json",
    ),
    "kuramoto8": FlagshipWorkload(
        name="kuramoto8",
        n_qubits=8,
        evolution_time=0.1,
        trotter_reps=2,
        parent_artifact="results/ibm_hardware_2026-03-28/kuramoto_8osc_zne.json",
    ),
}


def single_pauli(width: int, qubit: int, axis: str) -> SparsePauliOp:
    """Weight-one Pauli operator on one qubit (little-endian label order)."""
    label = ["I"] * width
    label[width - 1 - qubit] = axis
    return SparsePauliOp("".join(label))


def order_parameter(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Kuramoto order parameter R = |Σ⟨X⟩ + iΣ⟨Y⟩| / n."""
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("x and y expectation lists must be equal-length and non-empty")
    return math.hypot(float(np.sum(x_values)), float(np.sum(y_values))) / len(x_values)


def exact_reference(workload: FlagshipWorkload) -> dict[str, Any]:
    """Exact per-qubit ⟨X⟩/⟨Y⟩ and R of the Trotterised body via statevector.

    ``decompose()`` expands the evolution gate through its LieTrotter
    definition so the reference is the executed Trotter circuit, not the
    exact exponential.
    """
    state = Statevector(workload.body().decompose())
    width = workload.n_qubits
    x_values = [
        float(np.real(state.expectation_value(single_pauli(width, q, "X")))) for q in range(width)
    ]
    y_values = [
        float(np.real(state.expectation_value(single_pauli(width, q, "Y")))) for q in range(width)
    ]
    return {
        "x_expectations": x_values,
        "y_expectations": y_values,
        "order_parameter_r": order_parameter(x_values, y_values),
    }


def mps_expectations(
    workload: FlagshipWorkload, bond_dimension: int
) -> tuple[list[float], list[float], float]:
    """Per-qubit ⟨X⟩/⟨Y⟩ of the identical circuit via bounded-χ Aer MPS."""
    if bond_dimension < 1:
        raise ValueError("bond_dimension must be a positive integer")
    from qiskit_aer import AerSimulator

    width = workload.n_qubits
    simulation = transpile(
        workload.body().decompose(),
        basis_gates=BASIS_GATES,
        optimization_level=0,
    )
    for qubit in range(width):
        simulation.save_expectation_value(
            single_pauli(width, qubit, "X"), list(range(width)), label=f"x{qubit}"
        )
        simulation.save_expectation_value(
            single_pauli(width, qubit, "Y"), list(range(width)), label=f"y{qubit}"
        )
    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dimension,
    )
    started = time.perf_counter()
    data = simulator.run(simulation, shots=1).result().data(0)
    wall_ms = (time.perf_counter() - started) * 1000.0
    x_values = [float(data[f"x{qubit}"]) for qubit in range(width)]
    y_values = [float(data[f"y{qubit}"]) for qubit in range(width)]
    return x_values, y_values, wall_ms


def chi_sweep(
    workload: FlagshipWorkload,
    reference: Mapping[str, Any],
    bond_dimensions: Sequence[int],
) -> list[dict[str, Any]]:
    """Sweep bounded bond dimensions and compare against the exact reference."""
    rows: list[dict[str, Any]] = []
    reference_r = float(reference["order_parameter_r"])
    exact_x = np.asarray(reference["x_expectations"], dtype=np.float64)
    exact_y = np.asarray(reference["y_expectations"], dtype=np.float64)
    for bond_dimension in bond_dimensions:
        x_values, y_values, wall_ms = mps_expectations(workload, bond_dimension)
        r_value = order_parameter(x_values, y_values)
        component_error = max(
            float(np.max(np.abs(np.asarray(x_values) - exact_x))),
            float(np.max(np.abs(np.asarray(y_values) - exact_y))),
        )
        rows.append(
            {
                "bond_dimension": int(bond_dimension),
                "order_parameter_r": r_value,
                "r_abs_error": abs(r_value - reference_r),
                "max_component_abs_error": component_error,
                "wall_time_ms": wall_ms,
            }
        )
    return rows


def convergence_bond_dimension(rows: Sequence[Mapping[str, Any]], tolerance: float) -> int | None:
    """Smallest swept χ whose R error is within tolerance, if any."""
    within = [int(row["bond_dimension"]) for row in rows if float(row["r_abs_error"]) <= tolerance]
    return min(within) if within else None


def build_report(
    workload: FlagshipWorkload,
    reference: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Assemble the per-workload report block."""
    return {
        "workload": workload.name,
        "n_qubits": workload.n_qubits,
        "evolution_time": workload.evolution_time,
        "trotter_reps": workload.trotter_reps,
        "parent_artifact": workload.parent_artifact,
        "exact_reference": dict(reference),
        "r_tolerance": tolerance,
        "chi_sweep": [dict(row) for row in rows],
        "converged_bond_dimension": convergence_bond_dimension(rows, tolerance),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=[*sorted(WORKLOADS), "all"],
        default="all",
    )
    parser.add_argument(
        "--bond-dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_BOND_DIMENSIONS),
    )
    parser.add_argument("--r-tolerance", type=float, default=DEFAULT_R_TOLERANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = _parse_args(argv)
    if any(chi < 1 for chi in args.bond_dimensions):
        print("ERROR: bond dimensions must be positive integers", file=sys.stderr)
        return 2
    names = sorted(WORKLOADS) if args.workload == "all" else [args.workload]
    reports = []
    for name in names:
        workload = WORKLOADS[name]
        reference = exact_reference(workload)
        rows = chi_sweep(workload, reference, args.bond_dimensions)
        report = build_report(workload, reference, rows, args.r_tolerance)
        reports.append(report)
        print(
            f"{name}: exact R = {reference['order_parameter_r']:.6f}, "
            f"converged at chi = {report['converged_bond_dimension']}"
        )
        for row in rows:
            print(
                f"  chi={row['bond_dimension']:>3}  R={row['order_parameter_r']:.6f}  "
                f"|dR|={row['r_abs_error']:.2e}  "
                f"max|dXY|={row['max_component_abs_error']:.2e}  "
                f"{row['wall_time_ms']:.1f} ms"
            )
    payload = {
        "schema": SCHEMA,
        "generated_utc": _timestamp(),
        "basis_gates": BASIS_GATES,
        "claim_boundary": CLAIM_BOUNDARY,
        "workloads": reports,
    }
    output = args.output_dir / f"tn_baseline_flagship_{payload['generated_utc']}.json"
    digest = _write_json(output, payload)
    print(f"output: {output}")
    print(f"output sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
