# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping analog mapping evidence
"""Deterministic analog-mapping evidence bundle builder, renderer, and writer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibrate import CalibrationSensitivity, calibration_sensitivity
from .compare import AnalogDigitalComparison, compare_analog_model_to_trotter
from .contracts import ANALOG_MAPPING_CLAIM_BOUNDARY, FeasibilityReport, MappingRequest
from .feasibility import assess_mapping_feasibility
from .platforms import platform_profile

ANALOG_MAPPING_EVIDENCE_SCHEMA = "analog_mapping_evidence.v1"


@dataclass(frozen=True, slots=True)
class AnalogMappingEvidenceBundle:
    """Deterministic feasibility, comparison, and sensitivity evidence."""

    schema: str
    request: MappingRequest
    report: FeasibilityReport
    comparison: AnalogDigitalComparison | None
    calibration: CalibrationSensitivity | None
    profile_ledger_ref: str
    claim_boundary: str = ANALOG_MAPPING_CLAIM_BOUNDARY
    no_provider_contact: bool = True
    hardware_submission_allowed: bool = False
    hardware_support_claim_allowed: bool = False
    analog_advantage_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Keep detailed evidence conditional on an admitted compiler model."""
        details_present = self.comparison is not None and self.calibration is not None
        if details_present != self.report.supported:
            raise ValueError("comparison and calibration evidence must match report support")
        if (
            not self.no_provider_contact
            or self.hardware_submission_allowed
            or self.hardware_support_claim_allowed
            or self.analog_advantage_claim_allowed
        ):
            raise ValueError("analog-mapping evidence must remain local and non-promotional")

    @property
    def digest(self) -> str:
        """Return SHA-256 over the bundle payload excluding the digest field."""
        encoded = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-ready bundle with deterministic digest."""
        payload = self._payload()
        payload["digest"] = self.digest
        return payload

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request": self.request.to_dict(),
            "request_digest": self.request.digest,
            "report": self.report.to_dict(),
            "comparison": self.comparison.to_dict() if self.comparison is not None else None,
            "calibration": self.calibration.to_dict() if self.calibration is not None else None,
            "profile_ledger_ref": self.profile_ledger_ref,
            "claim_boundary": self.claim_boundary,
            "no_provider_contact": self.no_provider_contact,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "hardware_support_claim_allowed": self.hardware_support_claim_allowed,
            "analog_advantage_claim_allowed": self.analog_advantage_claim_allowed,
        }


def build_analog_mapping_evidence(
    request: MappingRequest,
    profile_id: str,
    *,
    trotter_steps: int = 32,
    relative_drift: float = 0.05,
) -> AnalogMappingEvidenceBundle:
    """Build local evidence for one request and static platform profile."""
    profile = platform_profile(profile_id)
    report = assess_mapping_feasibility(request, profile)
    comparison: AnalogDigitalComparison | None = None
    sensitivity: CalibrationSensitivity | None = None
    if report.supported:
        compiler_platform = profile.compiler_platform
        if compiler_platform is None:
            raise RuntimeError("supported profile did not resolve a compiler platform")
        comparison = compare_analog_model_to_trotter(
            request,
            compiler_platform=compiler_platform,
            trotter_steps=trotter_steps,
        )
        sensitivity = calibration_sensitivity(
            request.coupling_matrix,
            request.coupling_scale * request.coupling_matrix,
            nominal_scale=request.coupling_scale,
            relative_drift=relative_drift,
        )
    return AnalogMappingEvidenceBundle(
        schema=ANALOG_MAPPING_EVIDENCE_SCHEMA,
        request=request,
        report=report,
        comparison=comparison,
        calibration=sensitivity,
        profile_ledger_ref=profile.ledger_ref,
    )


def write_analog_mapping_evidence(
    path: str | Path,
    bundle: AnalogMappingEvidenceBundle,
) -> Path:
    """Write a canonical JSON evidence artifact and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def analog_mapping_markdown(bundle: AnalogMappingEvidenceBundle) -> str:
    """Render a concise non-promotional evidence note."""
    report = bundle.report
    lines = [
        "# Analog Mapping Feasibility Evidence",
        "",
        f"- Schema: `{bundle.schema}`",
        f"- Digest: `{bundle.digest}`",
        f"- Profile: `{report.profile_id}`",
        f"- Observed topology: `{report.observed_topology}`",
        f"- Internal compiler-model mapping supported: `{report.supported}`",
        "- Hardware submission allowed: `False`",
        "- Hardware support claim allowed: `False`",
        "- Analog advantage claim allowed: `False`",
        "",
        "## Boundary",
        "",
        bundle.claim_boundary,
        "",
        "## Diagnostics",
        "",
    ]
    lines.extend(
        f"- `{item.severity}` `{item.code}`: {item.message}" for item in report.diagnostics
    )
    if bundle.comparison is not None:
        lines.extend(
            [
                "",
                "## Bounded Model Comparison",
                "",
                f"- N: `{bundle.comparison.n_nodes}`",
                f"- Trotter steps: `{bundle.comparison.trotter_steps}`",
                f"- Compiler parameter RMSE: `{bundle.comparison.parameter_rmse:.6g}`",
                "- Digital Trotter state fidelity: "
                f"`{bundle.comparison.digital_trotter_state_fidelity:.12g}`",
                f"- Within declared tolerance: `{bundle.comparison.within_declared_tolerance}`",
            ]
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ANALOG_MAPPING_EVIDENCE_SCHEMA",
    "AnalogMappingEvidenceBundle",
    "analog_mapping_markdown",
    "build_analog_mapping_evidence",
    "write_analog_mapping_evidence",
]
