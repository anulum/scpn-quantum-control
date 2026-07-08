# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Josephson K_nm Magnitude Study
"""Josephson-array K_nm magnitude-study preregistration.

The study separates a high Josephson topology-correlation candidate from a
measured coupling-magnitude validation claim. It records the current rounded
``rho=0.990`` candidate, then keeps promotion blocked until a calibrated
Josephson or transmon coupling artifact supplies units, uncertainty, locked
normalisation, spectra, and null-model evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

from .josephson_array import JosephsonArrayParameters, josephson_benchmark

JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA = "scpn_qc_josephson_knm_magnitude_study_v1"
JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY = (
    "Topology-correlation candidate and magnitude-study preregistration only; "
    "this artifact does not validate K_nm physical coupling magnitudes."
)
DEFAULT_CANDIDATE_N = 14
DEFAULT_EXTENSION_TARGETS = (20, 30, 40)
DEFAULT_TOPOLOGY = "all_to_all"
TOPOLOGY_PROMOTION_FLOOR = 0.98
DIRECT_MAGNITUDE_RELATIVE_RMSE_GATE = 0.05
SPECTRAL_RELATIVE_DIFFERENCE_GATE = 0.05


@dataclass(frozen=True)
class JosephsonKnmCandidate:
    """One Josephson topology candidate for measured-magnitude follow-up."""

    n_junctions: int
    topology: str
    topology_correlation: float
    rounded_topology_correlation: float
    coupling_ratio: float
    parameter_source: str
    topology_source: str
    ej_ec_ratio: float
    is_transmon_regime: bool
    frequency_source: str
    claim_status: str

    def __post_init__(self) -> None:
        _require_positive_int(self.n_junctions, "n_junctions")
        _require_text(self.topology, "topology")
        _require_unit_interval(self.topology_correlation, "topology_correlation")
        _require_unit_interval(
            self.rounded_topology_correlation,
            "rounded_topology_correlation",
        )
        _require_positive_float(self.coupling_ratio, "coupling_ratio")
        _require_text(self.parameter_source, "parameter_source")
        _require_text(self.topology_source, "topology_source")
        _require_positive_float(self.ej_ec_ratio, "ej_ec_ratio")
        _require_text(self.frequency_source, "frequency_source")
        _require_text(self.claim_status, "claim_status")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible candidate data."""
        return {
            "n_junctions": self.n_junctions,
            "topology": self.topology,
            "topology_correlation": self.topology_correlation,
            "rounded_topology_correlation": self.rounded_topology_correlation,
            "coupling_ratio": self.coupling_ratio,
            "parameter_source": self.parameter_source,
            "topology_source": self.topology_source,
            "ej_ec_ratio": self.ej_ec_ratio,
            "is_transmon_regime": self.is_transmon_regime,
            "frequency_source": self.frequency_source,
            "claim_status": self.claim_status,
        }


@dataclass(frozen=True)
class JosephsonMagnitudeGate:
    """One fail-closed gate required before a Josephson magnitude claim."""

    name: str
    required: bool
    current_status: str
    evidence_required: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.current_status, "current_status")
        _require_nonempty_texts(self.evidence_required, "evidence_required")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible gate data."""
        return {
            "name": self.name,
            "required": self.required,
            "current_status": self.current_status,
            "evidence_required": list(self.evidence_required),
        }


@dataclass(frozen=True)
class JosephsonMagnitudeStudyDesign:
    """Complete Josephson K_nm magnitude-study design manifest."""

    schema: str
    candidate: JosephsonKnmCandidate
    calibration_artifact_schema: str
    extension_targets: tuple[int, ...]
    required_calibration_fields: tuple[str, ...]
    gates: tuple[JosephsonMagnitudeGate, ...]
    blocked_claims: tuple[str, ...]
    next_actions: tuple[str, ...]
    claim_boundary: str = JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY
    measured_calibration_available: bool = False
    magnitude_claim_allowed: bool = False
    hardware_submission_required: bool = False

    def __post_init__(self) -> None:
        _require_text(self.schema, "schema")
        _require_text(self.calibration_artifact_schema, "calibration_artifact_schema")
        _validate_extension_targets(self.extension_targets)
        _require_nonempty_texts(
            self.required_calibration_fields,
            "required_calibration_fields",
        )
        if not self.gates:
            raise ValueError("gates must be non-empty")
        _require_nonempty_texts(self.blocked_claims, "blocked_claims")
        _require_nonempty_texts(self.next_actions, "next_actions")
        _require_text(self.claim_boundary, "claim_boundary")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible design data."""
        return {
            "schema": self.schema,
            "candidate": self.candidate.to_dict(),
            "calibration_artifact_schema": self.calibration_artifact_schema,
            "extension_targets": list(self.extension_targets),
            "required_calibration_fields": list(self.required_calibration_fields),
            "gates": [gate.to_dict() for gate in self.gates],
            "blocked_claims": list(self.blocked_claims),
            "next_actions": list(self.next_actions),
            "claim_boundary": self.claim_boundary,
            "measured_calibration_available": self.measured_calibration_available,
            "magnitude_claim_allowed": self.magnitude_claim_allowed,
            "hardware_submission_required": self.hardware_submission_required,
        }


def build_josephson_knm_magnitude_study_design(
    *,
    n_junctions: int = DEFAULT_CANDIDATE_N,
    topology: str = DEFAULT_TOPOLOGY,
    parameters: JosephsonArrayParameters | None = None,
    extension_targets: Sequence[int] = DEFAULT_EXTENSION_TARGETS,
) -> JosephsonMagnitudeStudyDesign:
    """Build the Josephson K_nm magnitude-study preregistration manifest.

    Parameters
    ----------
    n_junctions:
        Number of Josephson-array nodes used for the topology candidate.
    topology:
        Josephson topology model passed to :func:`josephson_benchmark`.
    parameters:
        Parameter set used to evaluate the topology candidate. When omitted,
        explicitly labelled nominal transmon literature parameters are used.
    extension_targets:
        Larger node counts to preregister for the same study after calibrated
        Josephson coupling artifacts exist.

    Returns
    -------
    JosephsonMagnitudeStudyDesign
        Design manifest with topology evidence, required calibration fields,
        and fail-closed promotion gates.
    """
    _require_positive_int(n_junctions, "n_junctions")
    _require_text(topology, "topology")
    targets = _validate_extension_targets(tuple(int(size) for size in extension_targets))
    active_parameters = parameters or JosephsonArrayParameters.nominal_transmon()
    k_matrix = build_knm_paper27(L=n_junctions)
    omega, frequency_source = _omega_for_candidate(n_junctions, active_parameters)
    benchmark = josephson_benchmark(
        k_matrix,
        omega,
        topology=topology,
        parameters=active_parameters,
        allow_illustrative_topology=True,
    )
    candidate = JosephsonKnmCandidate(
        n_junctions=benchmark.n_junctions,
        topology=topology,
        topology_correlation=benchmark.topology_correlation,
        rounded_topology_correlation=round(benchmark.topology_correlation, 3),
        coupling_ratio=benchmark.coupling_ratio,
        parameter_source=benchmark.parameter_source,
        topology_source=benchmark.topology_source,
        ej_ec_ratio=benchmark.ej_ec_ratio,
        is_transmon_regime=benchmark.is_transmon_regime,
        frequency_source=frequency_source,
        claim_status=_candidate_status(benchmark.topology_correlation),
    )
    return JosephsonMagnitudeStudyDesign(
        schema=JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA,
        candidate=candidate,
        calibration_artifact_schema="scpn_qc_josephson_calibrated_couplings_v1",
        extension_targets=targets,
        required_calibration_fields=(
            "system_id",
            "device_or_array_source",
            "coupling_edges_0_indexed",
            "coupling_unit",
            "normalisation",
            "normalisation_locked",
            "value",
            "uncertainty",
            "calibration_timestamp",
            "source_reference",
        ),
        gates=_default_gates(),
        blocked_claims=(
            "No K_nm measured-magnitude validation from topology correlation alone.",
            "No physical-unit Josephson coupling claim from nominal literature parameters.",
            "No hardware-device coupling-map claim without backend calibration provenance.",
            "No promotion over EEG or power-grid controls until the same gates pass.",
        ),
        next_actions=(
            "Collect or derive a calibrated Josephson/transmon coupling artifact in the declared schema.",
            "Run the existing K_nm physical-validation audit on the calibrated artifact at the candidate and extension sizes.",
            "Require direct magnitude, spectral response, uncertainty, and null-model gates before promotion.",
        ),
    )


def render_josephson_knm_magnitude_study_markdown(
    design: JosephsonMagnitudeStudyDesign,
) -> str:
    """Render a human-reviewable Josephson K_nm magnitude-study report."""
    candidate = design.candidate
    lines = [
        "# Josephson K_nm Magnitude Study",
        "",
        "This QWC-5.2 artifact records the Josephson topology-correlation",
        "candidate and the measured-magnitude gates required before any",
        "physical K_nm coupling claim.",
        "",
        "## Boundary",
        "",
        design.claim_boundary,
        "",
        "## Candidate",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| N | {candidate.n_junctions} |",
        f"| topology | {candidate.topology} |",
        f"| topology source | {candidate.topology_source} |",
        f"| topology correlation | {candidate.topology_correlation:.6f} |",
        f"| rounded topology correlation | {candidate.rounded_topology_correlation:.3f} |",
        f"| coupling ratio | {candidate.coupling_ratio:.6f} |",
        f"| parameter source | {candidate.parameter_source} |",
        f"| frequency source | {candidate.frequency_source} |",
        f"| E_J/E_C | {candidate.ej_ec_ratio:.1f} |",
        f"| transmon regime | {candidate.is_transmon_regime} |",
        f"| claim status | {candidate.claim_status} |",
        "",
        "## Required Calibration Fields",
        "",
    ]
    lines.extend(f"- `{field}`" for field in design.required_calibration_fields)
    lines.extend(["", "## Extension Targets", ""])
    lines.extend(f"- N={target}" for target in design.extension_targets)
    lines.extend(
        [
            "",
            "## Promotion Gates",
            "",
            "| gate | current status | evidence required |",
            "| --- | --- | --- |",
        ]
    )
    for gate in design.gates:
        lines.append(
            f"| {gate.name} | {gate.current_status} | {'; '.join(gate.evidence_required)} |"
        )
    lines.extend(["", "## Blocked Claims", ""])
    lines.extend(f"- {claim}" for claim in design.blocked_claims)
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in design.next_actions)
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```bash",
            "scpn-bench knm-josephson-magnitude-study",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _candidate_status(topology_correlation: float) -> str:
    if topology_correlation >= TOPOLOGY_PROMOTION_FLOOR:
        return "topology_candidate_magnitude_blocked"
    return "below_topology_candidate_floor"


def _default_gates() -> tuple[JosephsonMagnitudeGate, ...]:
    return (
        JosephsonMagnitudeGate(
            name="calibrated_coupling_units",
            required=True,
            current_status="blocked_nominal_parameters_only",
            evidence_required=(
                "coupling_unit must denote calibrated Hz, rad/s, GHz, or derived Josephson energy",
                "every coupling edge must carry source metadata",
            ),
        ),
        JosephsonMagnitudeGate(
            name="locked_normalisation_and_uncertainty",
            required=True,
            current_status="blocked_no_measured_uncertainty",
            evidence_required=(
                "normalisation_locked=true",
                "per-edge uncertainty is finite and non-negative",
            ),
        ),
        JosephsonMagnitudeGate(
            name="direct_magnitude_fit",
            required=True,
            current_status="blocked_no_calibrated_artifact",
            evidence_required=(
                f"relative RMSE <= {DIRECT_MAGNITUDE_RELATIVE_RMSE_GATE:.2f}",
                "all reported edges participate in the fit",
            ),
        ),
        JosephsonMagnitudeGate(
            name="spectral_response",
            required=True,
            current_status="blocked_no_calibrated_artifact",
            evidence_required=(
                f"critical-coupling proxy relative difference <= {SPECTRAL_RELATIVE_DIFFERENCE_GATE:.2f}",
                "weighted adjacency and Laplacian spectra are recorded",
            ),
        ),
        JosephsonMagnitudeGate(
            name="null_models",
            required=True,
            current_status="blocked_no_calibrated_artifact",
            evidence_required=(
                "node-label null model gate passes",
                "edge-value null model gate passes",
            ),
        ),
    )


def _omega_for_candidate(
    n_junctions: int,
    parameters: JosephsonArrayParameters,
) -> tuple[NDArray[np.float64], str]:
    if n_junctions <= len(OMEGA_N_16):
        return OMEGA_N_16[:n_junctions], "canonical_OMEGA_N_16_prefix"
    return (
        np.full(n_junctions, parameters.ec_ghz, dtype=np.float64),
        "uniform_josephson_charging_energy_placeholder",
    )


def _validate_extension_targets(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("extension_targets must be non-empty")
    for value in values:
        _require_positive_int(value, "extension_targets")
    if tuple(sorted(values)) != values:
        raise ValueError("extension_targets must be sorted")
    if len(set(values)) != len(values):
        raise ValueError("extension_targets must be unique")
    return values


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_positive_int(value: int, field_name: str) -> None:
    if value <= 1:
        raise ValueError(f"{field_name} must be >= 2")


def _require_positive_float(value: float, field_name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive")


def _require_unit_interval(value: float, field_name: str) -> None:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [-1, 1]")


def _require_nonempty_texts(values: Sequence[str], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must be non-empty")
    for value in values:
        _require_text(value, field_name)
