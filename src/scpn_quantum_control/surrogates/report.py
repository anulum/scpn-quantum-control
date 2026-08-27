# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-reservoir evidence reporting
"""Deterministic JSON and Markdown evidence for the quantum-reservoir product."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from ..applications.quantum_reservoir_product import ReservoirTrainingCertificate
from .fidelity import SurrogateFidelityCertificate, SurrogateGradientCertificate
from .hybrid import ExactValidatedSurrogateProposal
from .models import GaussianRBFSurrogate

QUANTUM_RESERVOIR_EVIDENCE_SCHEMA = "scpn.quantum_reservoir_surrogates.v1"
QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY = (
    "Synthetic local exact-statevector and classical-reference evidence only. "
    "No hardware QRC, provider execution, unseen-domain generalisation, closed-loop "
    "control, optimisation advantage, publication, or deployment claim."
)
_NUMERIC_CUSTODY_DECIMALS = 6


def _canonicalise_evidence_numbers(value: object) -> object:
    """Normalise sub-precision runtime drift before evidence serialisation."""
    if isinstance(value, float):
        rounded = round(value, _NUMERIC_CUSTODY_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {str(key): _canonicalise_evidence_numbers(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise_evidence_numbers(child) for child in value]
    return value


@dataclass(frozen=True, slots=True)
class SurrogateSupportRow:
    """One executable or explicitly blocked surrogate support-matrix row."""

    surface: str
    status: Literal["local_exact_supported", "bounded_supported", "blocked_dependency"]
    evidence: str
    boundary: str

    def __post_init__(self) -> None:
        """Require complete non-empty support metadata."""
        if not all(value.strip() for value in (self.surface, self.evidence, self.boundary)):
            raise ValueError("support rows require non-empty surface, evidence, and boundary.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready support row."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuantumReservoirSurrogateEvidence:
    """Complete deterministic quantum-reservoir and surrogate evidence bundle."""

    reservoir_certificates: tuple[ReservoirTrainingCertificate, ...]
    surrogate_model: GaussianRBFSurrogate
    value_fidelity: SurrogateFidelityCertificate
    gradient_fidelity: SurrogateGradientCertificate
    exact_validated_proposal: ExactValidatedSurrogateProposal
    support_rows: tuple[SurrogateSupportRow, ...]
    schema: str = QUANTUM_RESERVOIR_EVIDENCE_SCHEMA
    claim_boundary: str = QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY

    def __post_init__(self) -> None:
        """Require both task families and a complete support matrix."""
        task_kinds = {certificate.task_kind for certificate in self.reservoir_certificates}
        if task_kinds != {"classification", "forecast"}:
            raise ValueError(
                "reservoir certificates must cover classification and forecast tasks."
            )
        if not self.value_fidelity.passed or not self.gradient_fidelity.passed:
            raise ValueError("surrogate value and gradient fidelity certificates must pass.")
        required_surfaces = {
            "qrc_heldout_certificates",
            "matched_esn_comparator",
            "gaussian_rbf_value_fidelity",
            "analytic_rbf_gradient_fidelity",
            "codesign_exact_validated_proposal",
            "multimodal_forecasting_adapter",
            "differentiable_notebook_curriculum",
        }
        if {row.surface for row in self.support_rows} != required_surfaces:
            raise ValueError(
                "support rows must cover the complete bounded quantum-reservoir surface."
            )

    def to_dict(self) -> dict[str, object]:
        """Return a canonical digest-bound evidence mapping."""
        payload: dict[str, object] = {
            "schema": self.schema,
            "reservoir_certificates": [
                certificate.to_dict() for certificate in self.reservoir_certificates
            ],
            "surrogate_model": self.surrogate_model.to_dict(),
            "value_fidelity": self.value_fidelity.to_dict(),
            "gradient_fidelity": self.gradient_fidelity.to_dict(),
            "exact_validated_proposal": self.exact_validated_proposal.to_dict(),
            "support_rows": [row.to_dict() for row in self.support_rows],
            "primary_sources": [
                {
                    "doi": "10.1103/PhysRevApplied.8.024030",
                    "role": "fixed quantum reservoir and trained classical readout",
                },
                {
                    "doi": "10.1103/PhysRevLett.131.100803",
                    "role": "classical-surrogate fidelity and honesty benchmark",
                },
                {
                    "doi": "10.1038/s42005-025-02423-4",
                    "role": "RBF proposal followed by true quantum-objective query",
                },
            ],
            "claim_boundary": self.claim_boundary,
        }
        payload = cast(dict[str, object], _canonicalise_evidence_numbers(payload))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["content_digest"] = hashlib.sha256(canonical).hexdigest()
        return payload


def render_quantum_reservoir_surrogate_markdown(
    evidence: QuantumReservoirSurrogateEvidence,
) -> str:
    """Render a concise human-readable view of the deterministic evidence."""
    payload = evidence.to_dict()
    lines = [
        "# Quantum Reservoir and Surrogate Evidence",
        "",
        f"Schema: `{evidence.schema}`",
        f"Content digest: `{payload['content_digest']}`",
        "",
        "## Held-out reservoir certificates",
        "",
        "| Task | Train / validation | QRC / ESN features | QRC validation MSE | ESN validation MSE | Lower MSE |",
        "|---|---:|---:|---:|---:|---|",
    ]
    certificates = cast(list[dict[str, object]], payload["reservoir_certificates"])
    for certificate in certificates:
        task_kind = cast(str, certificate["task_kind"])
        n_train = cast(int, certificate["n_train"])
        n_validation = cast(int, certificate["n_validation"])
        n_quantum_features = cast(int, certificate["n_quantum_features"])
        n_esn_features = cast(int, certificate["n_esn_features"])
        quantum_validation_mse = cast(float, certificate["quantum_validation_mse"])
        esn_validation_mse = cast(float, certificate["esn_validation_mse"])
        lower_validation_mse = cast(str, certificate["lower_validation_mse"])
        lines.append(
            f"| `{task_kind}` | {n_train} / {n_validation} "
            f"| {n_quantum_features} / {n_esn_features} "
            f"| {quantum_validation_mse:.9g} "
            f"| {esn_validation_mse:.9g} "
            f"| `{lower_validation_mse}` |"
        )
    value_fidelity = cast(dict[str, object], payload["value_fidelity"])
    gradient_fidelity = cast(dict[str, object], payload["gradient_fidelity"])
    lines.extend(
        [
            "",
            "## Classical surrogate fidelity",
            "",
            f"- Held-out value fidelity: `passed={value_fidelity['passed']}`, "
            f"RMSE `{cast(float, value_fidelity['rmse']):.9g}`, maximum error "
            f"`{cast(float, value_fidelity['max_absolute_error']):.9g}`, R² "
            f"`{cast(float, value_fidelity['r_squared']):.9g}`.",
            f"- Analytic-gradient fidelity: `passed={gradient_fidelity['passed']}`, "
            f"RMSE `{cast(float, gradient_fidelity['rmse']):.9g}`, maximum error "
            f"`{cast(float, gradient_fidelity['max_absolute_error']):.9g}` against exact local "
            "central differences.",
            f"- Exact proposal validation: `{evidence.exact_validated_proposal.reason}`; "
            "the controller proposal remains unapplied.",
            "",
            "## Support matrix",
            "",
            "| Surface | Status | Evidence / boundary |",
            "|---|---|---|",
        ]
    )
    for row in evidence.support_rows:
        lines.append(f"| `{row.surface}` | `{row.status}` | {row.evidence} {row.boundary} |")
    lines.extend(["", "## Claim boundary", "", evidence.claim_boundary, ""])
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_quantum_reservoir_surrogate_evidence(
    evidence: QuantumReservoirSurrogateEvidence,
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[str, str]:
    """Write deterministic JSON and Markdown evidence and return file digests."""
    json_text = json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
    markdown_text = render_quantum_reservoir_surrogate_markdown(evidence)
    _atomic_write(json_path, json_text)
    _atomic_write(markdown_path, markdown_text)
    return (
        hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
        hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY",
    "QUANTUM_RESERVOIR_EVIDENCE_SCHEMA",
    "QuantumReservoirSurrogateEvidence",
    "SurrogateSupportRow",
    "render_quantum_reservoir_surrogate_markdown",
    "write_quantum_reservoir_surrogate_evidence",
]
