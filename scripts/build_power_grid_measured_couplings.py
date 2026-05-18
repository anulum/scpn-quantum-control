# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Power-Grid Measured Coupling Builder
"""Build an IEEE 5-bus measured coupling artifact for K_nm validation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.applications.power_grid import (
    IEEE_5BUS_INERTIA,
    IEEE_5BUS_SUSCEPTANCE,
    IEEE_5BUS_VOLTAGE,
    IEEE_14BUS_BRANCH_REACTANCE,
    IEEE_14BUS_VOLTAGE,
    ieee_5bus_coupling_matrix,
    ieee_14bus_admittance_coupling_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings_power_grid_ieee5bus.json"
)
DEFAULT_14BUS_OUTPUT = (
    REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings_power_grid_ieee14bus.json"
)
OMEGA_0_RAD_S = 2.0 * np.pi * 60.0
SUSCEPTANCE_HALF_WIDTH = 0.005
VOLTAGE_HALF_WIDTH = 0.005
INERTIA_HALF_WIDTH = 0.05
REACTANCE_HALF_WIDTH = 0.000005


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _directed_coupling(
    *,
    susceptance: float,
    voltage_i: float,
    voltage_j: float,
    inertia_i: float,
) -> float:
    return voltage_i * voltage_j * susceptance / (2.0 * inertia_i * OMEGA_0_RAD_S)


def _directed_rounding_uncertainty(
    *,
    coupling: float,
    susceptance: float,
    voltage_i: float,
    voltage_j: float,
    inertia_i: float,
) -> float:
    if coupling == 0.0:
        return 0.0
    relative_terms = [
        SUSCEPTANCE_HALF_WIDTH / susceptance,
        VOLTAGE_HALF_WIDTH / voltage_i,
        VOLTAGE_HALF_WIDTH / voltage_j,
        INERTIA_HALF_WIDTH / inertia_i,
    ]
    return abs(coupling) * float(np.sqrt(np.sum(np.square(relative_terms))))


def symmetrised_edge_value_and_uncertainty(i: int, j: int) -> tuple[float, float]:
    """Return symmetrised K_ij and propagated rounding uncertainty."""
    susceptance = float(IEEE_5BUS_SUSCEPTANCE[i, j])
    if susceptance == 0.0:
        return 0.0, 0.0
    k_ij = _directed_coupling(
        susceptance=susceptance,
        voltage_i=float(IEEE_5BUS_VOLTAGE[i]),
        voltage_j=float(IEEE_5BUS_VOLTAGE[j]),
        inertia_i=float(IEEE_5BUS_INERTIA[i]),
    )
    k_ji = _directed_coupling(
        susceptance=susceptance,
        voltage_i=float(IEEE_5BUS_VOLTAGE[j]),
        voltage_j=float(IEEE_5BUS_VOLTAGE[i]),
        inertia_i=float(IEEE_5BUS_INERTIA[j]),
    )
    u_ij = _directed_rounding_uncertainty(
        coupling=k_ij,
        susceptance=susceptance,
        voltage_i=float(IEEE_5BUS_VOLTAGE[i]),
        voltage_j=float(IEEE_5BUS_VOLTAGE[j]),
        inertia_i=float(IEEE_5BUS_INERTIA[i]),
    )
    u_ji = _directed_rounding_uncertainty(
        coupling=k_ji,
        susceptance=susceptance,
        voltage_i=float(IEEE_5BUS_VOLTAGE[j]),
        voltage_j=float(IEEE_5BUS_VOLTAGE[i]),
        inertia_i=float(IEEE_5BUS_INERTIA[j]),
    )
    return (k_ij + k_ji) / 2.0, float(np.sqrt(u_ij**2 + u_ji**2) / 2.0)


def build_payload(*, command: list[str]) -> dict[str, Any]:
    coupling_matrix, omega = ieee_5bus_coupling_matrix(allow_builtin_reference=True)
    couplings = []
    for i in range(coupling_matrix.shape[0]):
        for j in range(i + 1, coupling_matrix.shape[1]):
            value, uncertainty = symmetrised_edge_value_and_uncertainty(i, j)
            has_line = bool(IEEE_5BUS_SUSCEPTANCE[i, j] > 0.0)
            couplings.append(
                {
                    "i": i + 1,
                    "j": j + 1,
                    "value": value,
                    "uncertainty": uncertainty,
                    "uncertainty_type": (
                        "rounded_input_half_width_first_order_propagation"
                        if has_line
                        else "topological_absence_in_public_benchmark"
                    ),
                    "source": "IEEE 5-bus public benchmark constants",
                    "raw_susceptance_per_unit": float(IEEE_5BUS_SUSCEPTANCE[i, j]),
                    "voltage_i_per_unit": float(IEEE_5BUS_VOLTAGE[i]),
                    "voltage_j_per_unit": float(IEEE_5BUS_VOLTAGE[j]),
                    "inertia_i_seconds": float(IEEE_5BUS_INERTIA[i]),
                    "inertia_j_seconds": float(IEEE_5BUS_INERTIA[j]),
                }
            )

    return {
        "schema_version": "scpn-quantum-control.measured-couplings.v1",
        "system": "IEEE 5-bus power-grid swing-equation coupling matrix",
        "unit": "dimensionless_swing_equation_coupling",
        "normalisation": (
            "K_ij = V_i V_j B_ij / (2 H_i omega_0), omega_0 = 2*pi*60 rad/s, "
            "then arithmetic symmetrisation of the two directed inertia terms."
        ),
        "normalisation_locked": True,
        "source_dataset": {
            "name": "IEEE 5-bus public benchmark constants",
            "source_reference": "IEEE PES public test feeder / Stagg-El-Abiad 5-bus constants",
            "source_mode": "curated_public_benchmark_constants",
            "raw_units": {
                "susceptance": "per_unit_on_100_MVA_base",
                "voltage": "per_unit",
                "inertia": "seconds",
                "omega_0": "radian_per_second",
                "frequency_deviation": "hertz_from_60_Hz_nominal",
            },
            "rounding_half_widths": {
                "susceptance_per_unit": SUSCEPTANCE_HALF_WIDTH,
                "voltage_per_unit": VOLTAGE_HALF_WIDTH,
                "inertia_seconds": INERTIA_HALF_WIDTH,
            },
        },
        "signal_processing": {
            "conversion": "public swing-equation constants to symmetric Kuramoto coupling",
            "nodes": ["bus_1", "bus_2", "bus_3", "bus_4", "bus_5"],
            "omega_hz_deviation": omega.tolist(),
            "susceptance_matrix_per_unit": IEEE_5BUS_SUSCEPTANCE.tolist(),
            "voltage_per_unit": IEEE_5BUS_VOLTAGE.tolist(),
            "inertia_seconds": IEEE_5BUS_INERTIA.tolist(),
            "converted_coupling_matrix": coupling_matrix.tolist(),
        },
        "couplings": couplings,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": command,
        },
    }


def _ieee14_reactance_lookup() -> dict[tuple[int, int], float]:
    return {
        (int(from_bus), int(to_bus)): float(reactance)
        for from_bus, to_bus, reactance in IEEE_14BUS_BRANCH_REACTANCE
    }


def _ieee14_rounding_uncertainty(
    *, value: float, reactance: float, voltage_i: float, voltage_j: float
) -> float:
    if value == 0.0:
        return 0.0
    relative_terms = [
        REACTANCE_HALF_WIDTH / reactance,
        VOLTAGE_HALF_WIDTH / voltage_i,
        VOLTAGE_HALF_WIDTH / voltage_j,
    ]
    return abs(value) * float(np.sqrt(np.sum(np.square(relative_terms))))


def build_ieee14_payload(*, command: list[str]) -> dict[str, Any]:
    """Build the IEEE 14-bus admittance candidate payload."""
    coupling_matrix, omega = ieee_14bus_admittance_coupling_matrix(allow_builtin_reference=True)
    reactance_lookup = _ieee14_reactance_lookup()
    couplings = []
    for i in range(coupling_matrix.shape[0]):
        for j in range(i + 1, coupling_matrix.shape[1]):
            edge = (i + 1, j + 1)
            reactance = reactance_lookup.get(edge)
            value = float(coupling_matrix[i, j])
            has_line = reactance is not None
            couplings.append(
                {
                    "i": i + 1,
                    "j": j + 1,
                    "value": value,
                    "uncertainty": (
                        _ieee14_rounding_uncertainty(
                            value=value,
                            reactance=float(reactance),
                            voltage_i=float(IEEE_14BUS_VOLTAGE[i]),
                            voltage_j=float(IEEE_14BUS_VOLTAGE[j]),
                        )
                        if has_line
                        else 0.0
                    ),
                    "uncertainty_type": (
                        "rounded_reactance_voltage_half_width_first_order_propagation"
                        if has_line
                        else "topological_absence_in_public_benchmark"
                    ),
                    "source": "IEEE 14-bus MATPOWER case14 public benchmark constants",
                    "raw_reactance_per_unit": reactance,
                    "raw_susceptance_per_unit": (1.0 / float(reactance)) if has_line else 0.0,
                    "voltage_i_per_unit": float(IEEE_14BUS_VOLTAGE[i]),
                    "voltage_j_per_unit": float(IEEE_14BUS_VOLTAGE[j]),
                }
            )

    return {
        "schema_version": "scpn-quantum-control.measured-couplings.v1",
        "system": "IEEE 14-bus power-grid voltage-weighted admittance coupling matrix",
        "unit": "per_unit_voltage_weighted_susceptance",
        "normalisation": (
            "K_ij = V_i V_j / X_ij for public case14 branches; absent branches are zero. "
            "This is a network-admittance candidate, not an inertia-normalised swing-equation "
            "candidate, because measured per-bus inertias are not supplied for every bus."
        ),
        "normalisation_locked": True,
        "source_dataset": {
            "name": "IEEE 14-bus MATPOWER case14 benchmark constants",
            "source_reference": "MATPOWER case14 / IEEE 14-bus public benchmark data",
            "source_mode": "curated_public_benchmark_constants",
            "raw_units": {
                "branch_reactance": "per_unit_on_100_MVA_base",
                "voltage": "per_unit",
                "frequency_deviation": "hertz_from_nominal",
            },
            "rounding_half_widths": {
                "reactance_per_unit": REACTANCE_HALF_WIDTH,
                "voltage_per_unit": VOLTAGE_HALF_WIDTH,
            },
            "known_limitations": [
                "no measured per-bus inertia constants for all load buses",
                "network-admittance control candidate only",
            ],
        },
        "signal_processing": {
            "conversion": "public branch reactance and voltage magnitudes to symmetric admittance coupling",
            "nodes": [f"bus_{index}" for index in range(1, 15)],
            "omega_hz_deviation": omega.tolist(),
            "branch_reactance_per_unit": IEEE_14BUS_BRANCH_REACTANCE.tolist(),
            "voltage_per_unit": IEEE_14BUS_VOLTAGE.tolist(),
            "converted_coupling_matrix": coupling_matrix.tolist(),
        },
        "couplings": couplings,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": command,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("ieee5", "ieee14"), default="ieee5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = [Path(sys.executable).name, *sys.argv]
    if args.case == "ieee14" and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_14BUS_OUTPUT
    payload = (
        build_ieee14_payload(command=command)
        if args.case == "ieee14"
        else build_payload(command=command)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {payload['system']}: {args.output}")
    print(f"Edges: {len(payload['couplings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
