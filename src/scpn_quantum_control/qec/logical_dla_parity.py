# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — logical DLA parity module
# scpn-quantum-control -- logical DLA parity roadmap model
"""Logical-level DLA parity resource roadmap.

This module deliberately does not claim that physical-qubit DLA parity
survives a surface-code encoding. It provides the deterministic resource and
noise rows needed before that theory question can be promoted to a simulation
or hardware target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.qec.error_budget import logical_error_rate
from scpn_quantum_control.qec.multiscale_qec import build_multiscale_qec

LOGICAL_DLA_PARITY_SCHEMA = "s7_logical_dla_parity_roadmap_v1"
CLAIM_BOUNDARY = (
    "roadmap and resource estimate only; no claim that DLA parity survives "
    "logical encoding and no hardware submission"
)


@dataclass(frozen=True)
class LogicalDLAParityRow:
    """Resource row for one logical-code distance."""

    n_oscillators: int
    code_distance: int
    physical_qubits_flat_surface_code: int
    physical_qubits_repetition_scaffold: int
    qec_rounds_per_trotter_step: int
    estimated_wall_clock_us_per_step: float
    p_physical: float
    logical_error_rate_per_round: float
    expected_step_fidelity: float
    parity_survival_claim_allowed: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible row data."""
        return asdict(self)


@dataclass(frozen=True)
class MultiscaleComparison:
    """Flat-vs-hierarchical QEC comparison for S7 planning."""

    n_oscillators: int
    flat_distance: int
    flat_surface_code_physical_qubits: int
    flat_logical_error_rate_per_round: float
    multiscale_total_physical_qubits: int
    multiscale_distances: tuple[int, ...]
    multiscale_effective_logical_rate: float
    multiscale_below_threshold: bool
    overhead_ratio_multiscale_to_flat: float
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible comparison data."""
        payload = asdict(self)
        payload["multiscale_distances"] = list(self.multiscale_distances)
        return payload


def surface_code_physical_qubits(n_oscillators: int, code_distance: int) -> int:
    """Return flat rotated-surface-code physical qubits for logical oscillators."""
    _validate_odd_distance(code_distance)
    _validate_n_oscillators(n_oscillators)
    return n_oscillators * (2 * code_distance * code_distance - 1)


def repetition_scaffold_physical_qubits(n_oscillators: int, code_distance: int) -> int:
    """Return existing repetition-code scaffold qubits for comparison only."""
    _validate_odd_distance(code_distance)
    _validate_n_oscillators(n_oscillators)
    return n_oscillators * (2 * code_distance - 1)


def estimate_logical_dla_parity_row(
    *,
    n_oscillators: int = 16,
    code_distance: int,
    p_physical: float = 0.003,
    syndrome_round_us: float = 1.0,
) -> LogicalDLAParityRow:
    """Estimate resources for a logical DLA-parity Trotter step.

    The fidelity model is a planning bound: all logical oscillators accrue one
    logical-error opportunity per QEC round in the step. It is intentionally
    conservative and does not promote DLA-parity survival.
    """
    _validate_probability(p_physical, "p_physical")
    if syndrome_round_us <= 0.0 or not np.isfinite(syndrome_round_us):
        raise ValueError("syndrome_round_us must be finite and positive")
    flat_qubits = surface_code_physical_qubits(n_oscillators, code_distance)
    repetition_qubits = repetition_scaffold_physical_qubits(n_oscillators, code_distance)
    p_logical = logical_error_rate(code_distance, p_physical)
    qec_rounds = code_distance
    failure_exposure = n_oscillators * qec_rounds * p_logical
    fidelity = float(np.exp(-failure_exposure))
    return LogicalDLAParityRow(
        n_oscillators=n_oscillators,
        code_distance=code_distance,
        physical_qubits_flat_surface_code=flat_qubits,
        physical_qubits_repetition_scaffold=repetition_qubits,
        qec_rounds_per_trotter_step=qec_rounds,
        estimated_wall_clock_us_per_step=float(qec_rounds * syndrome_round_us),
        p_physical=p_physical,
        logical_error_rate_per_round=float(p_logical),
        expected_step_fidelity=fidelity,
        parity_survival_claim_allowed=False,
        status="theory_required_before_simulation_or_hardware_promotion",
    )


def estimate_s7_resource_table(
    *,
    n_oscillators: int = 16,
    code_distances: tuple[int, ...] = (3, 5, 7),
    p_physical: float = 0.003,
    syndrome_round_us: float = 1.0,
) -> tuple[LogicalDLAParityRow, ...]:
    """Return the default S7 logical-DLA resource table."""
    if not code_distances:
        raise ValueError("code_distances must not be empty")
    return tuple(
        estimate_logical_dla_parity_row(
            n_oscillators=n_oscillators,
            code_distance=distance,
            p_physical=p_physical,
            syndrome_round_us=syndrome_round_us,
        )
        for distance in code_distances
    )


def compare_flat_surface_code_to_multiscale(
    *,
    n_oscillators: int = 16,
    flat_distance: int = 7,
    p_physical: float = 0.003,
    multiscale_distances: tuple[int, ...] = (3, 3, 3, 3, 3),
) -> MultiscaleComparison:
    """Compare flat surface-code resources with the existing MS-QEC hierarchy."""
    _validate_n_oscillators(n_oscillators)
    _validate_odd_distance(flat_distance)
    for distance in multiscale_distances:
        _validate_odd_distance(distance)
    flat_qubits = surface_code_physical_qubits(n_oscillators, flat_distance)
    flat_logical_rate = logical_error_rate(flat_distance, p_physical)
    K = build_knm_paper27(L=n_oscillators)
    multiscale = build_multiscale_qec(
        K,
        n_oscillators_per_level=n_oscillators,
        p_physical=p_physical,
        distances=list(multiscale_distances),
    )
    ratio = multiscale.total_physical_qubits / max(flat_qubits, 1)
    if (
        multiscale.total_physical_qubits < flat_qubits
        and multiscale.effective_logical_rate < flat_logical_rate
    ):
        conclusion = (
            "hierarchical_candidate_lower_qubit_overhead_and_error_rate_but_requires_theory_review"
        )
    elif multiscale.total_physical_qubits < flat_qubits:
        conclusion = "hierarchical_lower_qubit_overhead_but_logical_rate_not_viable"
    else:
        conclusion = "flat_surface_code_lower_qubit_overhead_for_this_distance_bundle"
    return MultiscaleComparison(
        n_oscillators=n_oscillators,
        flat_distance=flat_distance,
        flat_surface_code_physical_qubits=flat_qubits,
        flat_logical_error_rate_per_round=float(flat_logical_rate),
        multiscale_total_physical_qubits=multiscale.total_physical_qubits,
        multiscale_distances=tuple(multiscale_distances),
        multiscale_effective_logical_rate=float(multiscale.effective_logical_rate),
        multiscale_below_threshold=bool(multiscale.below_threshold),
        overhead_ratio_multiscale_to_flat=float(ratio),
        conclusion=conclusion,
    )


def logical_dla_parity_payload() -> dict[str, Any]:
    """Return the S7 logical-DLA parity roadmap payload."""
    rows = estimate_s7_resource_table()
    comparison = compare_flat_surface_code_to_multiscale()
    prerequisites = [
        "representation-theory review of XY Hamiltonian generators under the stabiliser group",
        "logical observable definition for DLA parity before Monte Carlo promotion",
        "noise-channel calibration separated from hardware-execution claims",
        "negative-result framing if the logical code destroys the physical parity signal",
    ]
    return {
        "schema": LOGICAL_DLA_PARITY_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "n_oscillators": 16,
        "code_distances": [row.code_distance for row in rows],
        "rows": [row.to_dict() for row in rows],
        "multiscale_comparison": comparison.to_dict(),
        "prerequisites": prerequisites,
        "no_qpu_submission": True,
        "hardware_submission_allowed": False,
        "parity_survival_claim_allowed": False,
    }


def logical_dla_parity_markdown(payload: dict[str, Any] | None = None) -> str:
    """Render the S7 logical-DLA parity roadmap as Markdown."""
    data = logical_dla_parity_payload() if payload is None else payload
    lines = [
        "# Logical DLA Parity Roadmap",
        "",
        "This note is a post-NISQ planning artefact. It estimates resources for",
        "logical-level DLA parity work and keeps the survival claim blocked until",
        "the theory and simulation prerequisites are closed.",
        "",
        "## Boundary",
        "",
        str(data["claim_boundary"]),
        "",
        "## Resource Table",
        "",
        "| N | d | Flat surface-code qubits | Repetition scaffold qubits | QEC rounds | Wall-clock us/step | p_L/round | Step fidelity | Status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["rows"]:
        lines.append(
            "| {n_oscillators} | {code_distance} | {physical_qubits_flat_surface_code} | "
            "{physical_qubits_repetition_scaffold} | {qec_rounds_per_trotter_step} | "
            "{estimated_wall_clock_us_per_step:.3f} | {logical_error_rate_per_round:.3e} | "
            "{expected_step_fidelity:.9f} | {status} |".format(**row)
        )
    comparison = data["multiscale_comparison"]
    lines.extend(
        [
            "",
            "## Multiscale QEC Cross-Check",
            "",
            f"- Flat d={comparison['flat_distance']} surface-code qubits: `{comparison['flat_surface_code_physical_qubits']}`",
            f"- Flat d={comparison['flat_distance']} logical error rate: `{comparison['flat_logical_error_rate_per_round']:.3e}`",
            f"- MS-QEC qubits for distances {comparison['multiscale_distances']}: `{comparison['multiscale_total_physical_qubits']}`",
            f"- MS-QEC effective logical rate: `{comparison['multiscale_effective_logical_rate']:.3e}`",
            f"- Overhead ratio MS-QEC/flat: `{comparison['overhead_ratio_multiscale_to_flat']:.6f}`",
            f"- Conclusion: `{comparison['conclusion']}`",
            "",
            "## Prerequisites",
        ]
    )
    lines.extend(f"- {item}" for item in data["prerequisites"])
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Regenerate and compare this roadmap with:",
            "",
            "```bash",
            "scpn-bench s7-logical-dla-roadmap",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_n_oscillators(n_oscillators: int) -> None:
    if n_oscillators < 2:
        raise ValueError("n_oscillators must be at least 2")


def _validate_odd_distance(code_distance: int) -> None:
    if code_distance < 3 or code_distance % 2 == 0:
        raise ValueError("code_distance must be an odd integer >= 3")


def _validate_probability(value: float, field: str) -> None:
    if value < 0.0 or value >= 1.0 or not np.isfinite(value):
        raise ValueError(f"{field} must be finite with 0 <= {field} < 1")


__all__ = [
    "CLAIM_BOUNDARY",
    "LOGICAL_DLA_PARITY_SCHEMA",
    "LogicalDLAParityRow",
    "MultiscaleComparison",
    "compare_flat_surface_code_to_multiscale",
    "estimate_logical_dla_parity_row",
    "estimate_s7_resource_table",
    "logical_dla_parity_markdown",
    "logical_dla_parity_payload",
    "repetition_scaffold_physical_qubits",
    "surface_code_physical_qubits",
]
