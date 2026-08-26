# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable error mitigation product
"""Fail-closed **differentiable error-mitigation taxonomy** product.

Productises a versioned mitigator taxonomy and local simulation probes over
ambient :mod:`scpn_quantum_control.mitigation` and Studio
:mod:`scpn_quantum_control.studio.executive_mitigate`:

* differentiability classes (``analytic_fd``, ``fd_only``, ``non_diff``,
  ``optional_extra``) per mitigator family (ZNE, PEC, readout, symmetry,
  DD, CPDR, mitiq);
* real ZNE polynomial extrapolation + uncertainty probes on supplied
  expectation values (no circuit execution);
* readout confusion-matrix mitigation probe from calibration counts;
* hard-gap boundaries: invent-green ideal-gradient restoration, live QPU
  mitigation claims, mitiq-as-hard-dependency without extra;
* compose hardware-safety no-submit and Studio-executive executive ``mitigate`` claim boundary.

Does **not** claim mitigation restores ideal gradients, run live hardware,
or require mitiq as a hard dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess  # nosec B404
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

# The subprocess module executes only a fixed interpreter argv and product-owned probe.

# Product-local mirror of
# :data:`scpn_quantum_control.studio.executive_mitigate.MITIGATE_CLAIM_BOUNDARY`.
# The ambient Studio package ``__init__`` pulls optional ``scpn_studio_platform``
# (Python ≥3.12 CI extra only). This façade must import on the base matrix
# (3.11 without the Studio extra); the ambient constant is preferred when the
# package is available so compose stays honest.
_STUDIO_MITIGATE_CLAIM_BOUNDARY_MIRROR: Final[str] = (
    "polynomial zero-noise extrapolation of the given measured expectation "
    "values with delta-method uncertainty propagation; it does not run "
    "circuits, amplify noise, model the device noise physics, or validate "
    "that the supplied expectations came from a real experiment"
)


def _studio_mitigate_claim_boundary_text() -> str:
    """Return the Studio mitigate claim boundary without hard Studio import.

    Prefers the ambient constant when ``scpn_studio_platform`` (and the Studio
    package) is importable; otherwise uses the product-local mirror so the
    taxonomy façade loads on base CI.
    """
    try:
        from .studio.executive_mitigate import MITIGATE_CLAIM_BOUNDARY as ambient
    except ImportError:
        return _STUDIO_MITIGATE_CLAIM_BOUNDARY_MIRROR
    text = str(ambient).strip()
    if not text:
        raise ValueError("Studio MITIGATE_CLAIM_BOUNDARY must be non-empty")
    return text


DifferentiabilityClass = Literal[
    "analytic_fd",
    "fd_only",
    "non_diff",
    "optional_extra",
]
"""How gradients may compose with this mitigator family."""

MitigatorKind = Literal[
    "zne",
    "zne_uncertainty",
    "pec",
    "readout",
    "symmetry_sector",
    "dynamical_decoupling",
    "cpdr",
    "mitiq_optional",
    "studio_executive_mitigate",
]
"""Mitigator family kinds on the product taxonomy."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

BoundaryKind = Literal[
    "ideal_gradient_restore",
    "live_qpu_mitigation",
    "mitiq_hard_dependency",
    "unattested_hardware_expectations",
    "non_diff_extrapolation_as_analytic",
]
"""Hard-gap boundary kinds for mitigation honesty."""

ERROR_MITIGATION_PRODUCT_SCHEMA: Final[str] = "error_mitigation_product.v1"
"""JSON schema identifier for serialised product payloads."""

ERROR_MITIGATION_CLAIM_BOUNDARY: Final[str] = (
    "Differentiable error-mitigation taxonomy product surface only; catalogues "
    "mitigator families with differentiability class over ambient mitigation/* "
    "and Studio executive_mitigate; local ZNE/readout probes on supplied values/"
    "calibration counts; refuse invent-green ideal-gradient restoration, live "
    "QPU mitigation claims, and mitiq hard-dependency without optional extra; "
    "compose the no-submit safety policy; deeper open-system objective "
    "integration and metamorphic registration remain explicit residual work"
)
"""Shared claim boundary for error-mitigation product payloads."""


@dataclass(frozen=True, slots=True)
class MitigatorTaxonomyRow:
    """One mitigator taxonomy row.

    Attributes
    ----------
    mitigator_id
        Stable mitigator identifier.
    kind
        Mitigator family kind.
    title
        Human-readable title.
    summary
        Short description.
    differentiability
        Differentiability class for gradient composition.
    ambient_module
        Ambient module path.
    ambient_symbol
        Primary ambient symbol.
    hardware_submit_allowed
        Must remain False.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    mitigator_id: str
    kind: MitigatorKind
    title: str
    summary: str
    differentiability: DifferentiabilityClass
    ambient_module: str
    ambient_symbol: str
    hardware_submit_allowed: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = ERROR_MITIGATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate mitigator taxonomy invariants."""
        if not self.mitigator_id or not self.mitigator_id.strip():
            raise ValueError("mitigator_id must be non-empty")
        if self.kind not in {
            "zne",
            "zne_uncertainty",
            "pec",
            "readout",
            "symmetry_sector",
            "dynamical_decoupling",
            "cpdr",
            "mitiq_optional",
            "studio_executive_mitigate",
        }:
            raise ValueError(f"unknown mitigator kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.differentiability not in {
            "analytic_fd",
            "fd_only",
            "non_diff",
            "optional_extra",
        }:
            raise ValueError(f"unknown differentiability class: {self.differentiability!r}")
        if not self.ambient_module or not self.ambient_module.strip():
            raise ValueError("ambient_module must be non-empty")
        if not self.ambient_symbol or not self.ambient_symbol.strip():
            raise ValueError("ambient_symbol must be non-empty")
        if self.hardware_submit_allowed:
            raise ValueError("hardware_submit_allowed must be False on product surface")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "metadata_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "mitigator_id": self.mitigator_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "differentiability": self.differentiability,
            "ambient_module": self.ambient_module,
            "ambient_symbol": self.ambient_symbol,
            "hardware_submit_allowed": self.hardware_submit_allowed,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MitigationBoundaryRow:
    """One hard-gap boundary row for mitigation honesty.

    Attributes
    ----------
    boundary_id
        Stable boundary identifier.
    kind
        Boundary kind enum.
    title
        Human-readable title.
    failure_class
        Machine-oriented failure class.
    summary
        Short description.
    fail_closed
        Must remain True.
    claim_boundary
        Non-promotional claim boundary.

    """

    boundary_id: str
    kind: BoundaryKind
    title: str
    failure_class: str
    summary: str
    fail_closed: bool = True
    claim_boundary: str = ERROR_MITIGATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate boundary-row invariants."""
        if not self.boundary_id or not self.boundary_id.strip():
            raise ValueError("boundary_id must be non-empty")
        if self.kind not in {
            "ideal_gradient_restore",
            "live_qpu_mitigation",
            "mitiq_hard_dependency",
            "unattested_hardware_expectations",
            "non_diff_extrapolation_as_analytic",
        }:
            raise ValueError(f"unknown boundary kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.failure_class or not self.failure_class.strip():
            raise ValueError("failure_class must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.fail_closed is not True:
            raise ValueError("fail_closed must be True on product boundary rows")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "boundary_id": self.boundary_id,
            "kind": self.kind,
            "title": self.title,
            "failure_class": self.failure_class,
            "summary": self.summary,
            "fail_closed": self.fail_closed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for error-mitigation product use."""

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = ERROR_MITIGATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate path eligibility invariants."""
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed":
            raise ValueError("allowed decisions must use outcome=allowed")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedZneProbe:
    """Materialised ZNE extrapolation probe through ambient ``zne_extrapolate``."""

    mitigator_id: str
    zero_noise_estimate: float
    fit_residual: float
    order: int
    n_points: int
    probe_digest: str
    invent_green_ideal_gradient_restore: bool
    invent_green_live_qpu: bool
    demo_label: str
    claim_boundary: str = ERROR_MITIGATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate ZNE probe invariants."""
        if not self.mitigator_id or not self.mitigator_id.strip():
            raise ValueError("mitigator_id must be non-empty")
        if not math.isfinite(self.zero_noise_estimate):
            raise ValueError("zero_noise_estimate must be finite")
        if not math.isfinite(self.fit_residual) or self.fit_residual < 0.0:
            raise ValueError("fit_residual must be finite and non-negative")
        if self.order < 1:
            raise ValueError("order must be positive")
        if self.n_points < 2:
            raise ValueError("n_points must be at least 2")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if self.invent_green_ideal_gradient_restore:
            raise ValueError("invent_green_ideal_gradient_restore must be False")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "mitigator_id": self.mitigator_id,
            "zero_noise_estimate": self.zero_noise_estimate,
            "fit_residual": self.fit_residual,
            "order": self.order,
            "n_points": self.n_points,
            "probe_digest": self.probe_digest,
            "invent_green_ideal_gradient_restore": self.invent_green_ideal_gradient_restore,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedReadoutProbe:
    """Materialised readout-mitigation probe through an ambient confusion matrix."""

    mitigator_id: str
    n_qubits: int
    n_basis: int
    mitigated_probability_sum: float
    probe_digest: str
    invent_green_ideal_gradient_restore: bool
    demo_label: str
    claim_boundary: str = ERROR_MITIGATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate readout probe invariants."""
        if not self.mitigator_id or not self.mitigator_id.strip():
            raise ValueError("mitigator_id must be non-empty")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.n_basis != 2**self.n_qubits:
            raise ValueError("n_basis must equal 2**n_qubits")
        if not math.isfinite(self.mitigated_probability_sum):
            raise ValueError("mitigated_probability_sum must be finite")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if self.invent_green_ideal_gradient_restore:
            raise ValueError("invent_green_ideal_gradient_restore must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "mitigator_id": self.mitigator_id,
            "n_qubits": self.n_qubits,
            "n_basis": self.n_basis,
            "mitigated_probability_sum": self.mitigated_probability_sum,
            "probe_digest": self.probe_digest,
            "invent_green_ideal_gradient_restore": self.invent_green_ideal_gradient_restore,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_taxonomy() -> tuple[MitigatorTaxonomyRow, ...]:
    """Build the mitigator taxonomy catalogue."""
    return (
        MitigatorTaxonomyRow(
            mitigator_id="zne_richardson",
            kind="zne",
            title="Zero-noise extrapolation (Richardson)",
            summary=(
                "Polynomial Richardson extrapolation on supplied noise scales "
                "and expectation values; FD-friendly arithmetic, not ideal-gradient restore."
            ),
            differentiability="fd_only",
            ambient_module="scpn_quantum_control.mitigation.zne",
            ambient_symbol="zne_extrapolate",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="zne_uncertainty",
            kind="zne_uncertainty",
            title="ZNE with delta-method uncertainty",
            summary=(
                "Weighted/OLS ZNE with coverage interval on zero-noise estimate; "
                "Studio executive_mitigate uses this ambient path."
            ),
            differentiability="fd_only",
            ambient_module="scpn_quantum_control.mitigation.zne_uncertainty",
            ambient_symbol="zne_extrapolate_with_uncertainty",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="pec_pauli_twirl",
            kind="pec",
            title="Probabilistic error cancellation (Pauli twirl)",
            summary=(
                "PEC sampling over Pauli twirl quasi-probability; stochastic estimator "
                "boundary — not analytic adjoint through sampling."
            ),
            differentiability="non_diff",
            ambient_module="scpn_quantum_control.mitigation.pec",
            ambient_symbol="pec_sample",
            support_posture="local_research",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="readout_confusion",
            kind="readout",
            title="Readout confusion-matrix mitigation",
            summary=(
                "Invert full-basis calibration confusion matrix on probability vectors; "
                "linear map supports the finite-difference policy on mitigated estimators."
            ),
            differentiability="analytic_fd",
            ambient_module="scpn_quantum_control.mitigation.readout_matrix",
            ambient_symbol="mitigate_probabilities",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="symmetry_sector",
            kind="symmetry_sector",
            title="Symmetry-sector mitigation + replay",
            summary=(
                "Plan/replay symmetry-sector mitigation with provenance; "
                "post-selection is non-diff at discrete sector boundaries."
            ),
            differentiability="non_diff",
            ambient_module="scpn_quantum_control.mitigation.symmetry_sector_compiler",
            ambient_symbol="plan_symmetry_sector_mitigation",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="dynamical_decoupling",
            kind="dynamical_decoupling",
            title="Dynamical decoupling sequence insert",
            summary=(
                "Insert DD sequences into circuits; structural transform — "
                "gradient through DD schedule is out of product analytic scope."
            ),
            differentiability="non_diff",
            ambient_module="scpn_quantum_control.mitigation.dd",
            ambient_symbol="insert_dd_sequence",
            support_posture="metadata_only",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="cpdr",
            kind="cpdr",
            title="CPDR mitigation pipeline",
            summary=(
                "Clifford data regression style pipeline; training/inference "
                "boundary is research-local, not analytic AD through models."
            ),
            differentiability="fd_only",
            ambient_module="scpn_quantum_control.mitigation.cpdr",
            ambient_symbol="cpdr_mitigate",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="mitiq_optional",
            kind="mitiq_optional",
            title="Optional mitiq integration",
            summary=(
                "Optional-extra mitiq ZNE/DDD wrappers; product never treats "
                "mitiq as a hard dependency."
            ),
            differentiability="optional_extra",
            ambient_module="scpn_quantum_control.mitigation.mitiq_integration",
            ambient_symbol="is_mitiq_available",
            support_posture="policy_only",
        ),
        MitigatorTaxonomyRow(
            mitigator_id="studio_executive_mitigate",
            kind="studio_executive_mitigate",
            title="Studio executive mitigation verb",
            summary=(
                "Read-only Studio mitigate handler: polynomial ZNE on given "
                "measured values with uncertainty; does not run circuits."
            ),
            differentiability="fd_only",
            ambient_module="scpn_quantum_control.studio.executive_mitigate",
            ambient_symbol="MitigateActionHandler",
            support_posture="policy_only",
        ),
    )


def _build_boundaries() -> tuple[MitigationBoundaryRow, ...]:
    """Build hard-gap boundary catalogue."""
    return (
        MitigationBoundaryRow(
            boundary_id="ideal_gradient_restore",
            kind="ideal_gradient_restore",
            title="Mitigation restores ideal gradients",
            failure_class="ideal_gradient_restore_refused",
            summary=(
                "Product refuses invent-green claims that mitigation restores "
                "noiseless analytic gradients of the ideal circuit."
            ),
        ),
        MitigationBoundaryRow(
            boundary_id="live_qpu_mitigation",
            kind="live_qpu_mitigation",
            title="Live QPU mitigation submit",
            failure_class="live_qpu_mitigation_refused",
            summary=(
                "Compose the no-submit safety policy; product probes use supplied values or "
                "local calibration counts only."
            ),
        ),
        MitigationBoundaryRow(
            boundary_id="mitiq_hard_dependency",
            kind="mitiq_hard_dependency",
            title="mitiq as hard dependency",
            failure_class="mitiq_hard_dependency_refused",
            summary=(
                "mitiq remains optional-extra; product taxonomy lists it as "
                "optional_extra without requiring import success."
            ),
        ),
        MitigationBoundaryRow(
            boundary_id="unattested_hardware_expectations",
            kind="unattested_hardware_expectations",
            title="Unattested hardware expectation values",
            failure_class="unattested_hardware_expectations_refused",
            summary=(
                "ZNE arithmetic does not validate that supplied expectations "
                "came from hardware; unattested invent-green claims refused."
            ),
        ),
        MitigationBoundaryRow(
            boundary_id="non_diff_as_analytic",
            kind="non_diff_extrapolation_as_analytic",
            title="Non-diff mitigator sold as analytic AD",
            failure_class="non_diff_as_analytic_refused",
            summary=(
                "PEC sampling and discrete symmetry post-selection are non_diff; "
                "refusing analytic-AD invent-green for those families."
            ),
        ),
    )


_TAXONOMY: Final[tuple[MitigatorTaxonomyRow, ...]] = _build_taxonomy()
_BOUNDARIES: Final[tuple[MitigationBoundaryRow, ...]] = _build_boundaries()


def _taxonomy_map() -> dict[str, MitigatorTaxonomyRow]:
    """Return mitigator_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, MitigatorTaxonomyRow] = {}
    for row in _TAXONOMY:
        key = row.mitigator_id.strip()
        if not key:
            raise RuntimeError("mitigator taxonomy contains blank mitigator_id")
        if key in mapping:
            raise RuntimeError(f"duplicate mitigator_id in taxonomy: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("mitigator taxonomy must be non-empty")
    return mapping


_TAXONOMY_BY_ID: Final[Mapping[str, MitigatorTaxonomyRow]] = _taxonomy_map()


def list_mitigator_ids() -> tuple[str, ...]:
    """Return all mitigator identifiers in catalogue order."""
    return tuple(row.mitigator_id for row in _TAXONOMY)


def list_mitigation_boundary_ids() -> tuple[str, ...]:
    """Return all hard-gap boundary identifiers in catalogue order."""
    return tuple(row.boundary_id for row in _BOUNDARIES)


def get_mitigator(mitigator_id: str) -> MitigatorTaxonomyRow:
    """Return one taxonomy row; fail closed on blank/unknown."""
    if not mitigator_id or not str(mitigator_id).strip():
        raise ValueError("mitigator_id must be non-empty")
    key = str(mitigator_id).strip()
    try:
        return _TAXONOMY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown mitigator_id: {key!r}") from exc


def get_mitigation_boundary(boundary_id: str) -> MitigationBoundaryRow:
    """Return one boundary row; fail closed on blank/unknown."""
    if not boundary_id or not str(boundary_id).strip():
        raise ValueError("boundary_id must be non-empty")
    key = str(boundary_id).strip()
    for row in _BOUNDARIES:
        if row.boundary_id == key:
            return row
    raise ValueError(f"unknown boundary_id: {key!r}")


def iter_mitigators(
    *,
    kind: MitigatorKind | None = None,
    differentiability: DifferentiabilityClass | None = None,
) -> tuple[MitigatorTaxonomyRow, ...]:
    """Return filtered taxonomy rows in stable order."""
    rows: Sequence[MitigatorTaxonomyRow] = _TAXONOMY
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    if differentiability is not None:
        rows = tuple(row for row in rows if row.differentiability == differentiability)
    return tuple(rows)


def iter_mitigation_boundaries(
    *,
    kind: BoundaryKind | None = None,
) -> tuple[MitigationBoundaryRow, ...]:
    """Return filtered boundary rows in stable order."""
    rows: Sequence[MitigationBoundaryRow] = _BOUNDARIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def decide_mitigation_path(
    mitigator_id: str,
    *,
    invent_green_ideal_gradient_restore: bool = False,
    invent_green_live_qpu: bool = False,
    invent_green_mitiq_hard_dep: bool = False,
    invent_green_non_diff_as_analytic: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a mitigation-product path may proceed."""
    row = get_mitigator(mitigator_id)
    blockers: list[str] = []
    if invent_green_ideal_gradient_restore:
        blockers.append(
            "invent-green ideal-gradient restoration refused "
            f"(mitigator={row.mitigator_id}; mitigation does not restore ideal AD)"
        )
    if invent_green_live_qpu:
        blockers.append(
            "invent-green live QPU mitigation refused "
            f"(mitigator={row.mitigator_id}; no-submit safety policy)"
        )
    if invent_green_mitiq_hard_dep:
        blockers.append(
            "invent-green mitiq hard dependency refused "
            f"(mitigator={row.mitigator_id}; mitiq remains optional_extra)"
        )
    if invent_green_non_diff_as_analytic and row.differentiability in {
        "non_diff",
        "optional_extra",
    }:
        blockers.append(
            "invent-green analytic AD for non-diff/optional mitigator refused "
            f"(mitigator={row.mitigator_id}; class={row.differentiability})"
        )
    if invent_green_non_diff_as_analytic and row.differentiability not in {
        "non_diff",
        "optional_extra",
    }:
        # Explicit analytic invent-green still refused for honesty catalogue.
        blockers.append(
            "invent-green non-diff-as-analytic flag refused even on FD/analytic "
            f"mitigator={row.mitigator_id} (use taxonomy class honestly)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="mitigation path refused under product honesty gates",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"mitigator {row.mitigator_id!r} may proceed as local "
            f"{row.differentiability} taxonomy path only"
        ),
        blockers=(),
    )


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Canonical SHA-256 over a JSON-serialisable mapping."""
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_ambient_mitigation_json(
    mode: Literal["zne", "zne_uncertainty", "readout"],
    *,
    noise_scales: Sequence[int],
    expectation_values: Sequence[float],
    order: int,
) -> dict[str, Any]:
    """Run ambient mitigation probes in a clean subprocess (pytest-cov safe)."""
    scales = [int(item) for item in noise_scales]
    values = [float(item) for item in expectation_values]
    script = f"""
import json
import numpy as np
from scpn_quantum_control.mitigation.zne import zne_extrapolate
from scpn_quantum_control.mitigation.zne_uncertainty import zne_extrapolate_with_uncertainty
from scpn_quantum_control.mitigation.readout_matrix import (
    build_readout_confusion_matrix,
    mitigate_probabilities,
)

mode = {mode!r}
scales = {scales!r}
values = {values!r}
order = {int(order)}

if mode == "zne":
    r = zne_extrapolate(scales, values, order=order)
    out = {{
        "zero_noise_estimate": float(r.zero_noise_estimate),
        "fit_residual": float(r.fit_residual),
        "order": int(order),
        "n_points": len(scales),
    }}
elif mode == "zne_uncertainty":
    u = zne_extrapolate_with_uncertainty(
        [float(s) for s in scales], values, order=order
    )
    out = {{
        "zero_noise_estimate": float(u.zero_noise_estimate),
        "fit_residual": float(u.fit_residual),
        "order": int(u.order),
        "n_points": int(u.n_points),
    }}
elif mode == "readout":
    calibration = {{"0": {{"0": 95, "1": 5}}, "1": {{"0": 8, "1": 92}}}}
    matrix = build_readout_confusion_matrix(calibration, n_qubits=1)
    observed = np.asarray([0.88, 0.12], dtype=np.float64)
    mitigated = mitigate_probabilities(observed, matrix)
    out = {{
        "n_qubits": 1,
        "n_basis": 2,
        "mitigated_probability_sum": float(np.sum(mitigated)),
    }}
else:
    raise SystemExit(f"unknown mode {{mode!r}}")
print(json.dumps(out))
"""
    try:
        # No shell is used; argv and probe source are product-owned constants.
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ValueError(f"ambient mitigation subprocess failed: {err}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ambient mitigation subprocess timed out") from exc
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ambient mitigation subprocess returned non-JSON: {line!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("ambient mitigation subprocess payload must be an object")
    return payload


def materialise_zne_probe(
    mitigator_id: str = "zne_richardson",
    *,
    noise_scales: Sequence[int] = (1, 3, 5),
    expectation_values: Sequence[float] = (0.90, 0.70, 0.50),
    order: int = 1,
    invent_green_ideal_gradient_restore: bool = False,
    invent_green_live_qpu: bool = False,
    demo_label: str = "zne_demo",
) -> MaterialisedZneProbe:
    """Materialise a real ambient ZNE probe on supplied values."""
    decision = decide_mitigation_path(
        mitigator_id,
        invent_green_ideal_gradient_restore=invent_green_ideal_gradient_restore,
        invent_green_live_qpu=invent_green_live_qpu,
    )
    if not decision.allowed:
        raise ValueError("zne probe refused: " + "; ".join(decision.blockers))
    row = get_mitigator(mitigator_id)
    if row.kind not in {"zne", "zne_uncertainty", "studio_executive_mitigate"}:
        raise ValueError(f"materialise_zne_probe requires ZNE-family mitigator, got {row.kind!r}")
    scales = [int(item) for item in noise_scales]
    values = [float(item) for item in expectation_values]
    mode: Literal["zne", "zne_uncertainty"] = (
        "zne_uncertainty"
        if row.kind in {"zne_uncertainty", "studio_executive_mitigate"}
        else "zne"
    )
    raw = _run_ambient_mitigation_json(
        mode,
        noise_scales=scales,
        expectation_values=values,
        order=order,
    )
    try:
        zero_est = float(raw["zero_noise_estimate"])
        residual = float(raw["fit_residual"])
        used_order = int(raw["order"])
        n_points = int(raw["n_points"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ambient ZNE payload missing fields: {raw!r}") from exc
    if not math.isfinite(zero_est) or not math.isfinite(residual):
        raise ValueError("ambient ZNE result must be finite")
    if residual < 0.0:
        raise ValueError("fit_residual must be non-negative")
    if n_points < 2:
        raise ValueError("ambient ZNE n_points must be at least 2")
    digest = _digest_payload(
        {
            "schema": "error_mitigation_zne_probe.v1",
            "mitigator_id": row.mitigator_id,
            "noise_scales": scales,
            "expectation_values": values,
            "order": used_order,
            "zero_noise_estimate": zero_est,
            "fit_residual": residual,
            "product_schema": ERROR_MITIGATION_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedZneProbe(
        mitigator_id=row.mitigator_id,
        zero_noise_estimate=zero_est,
        fit_residual=residual,
        order=used_order,
        n_points=n_points,
        probe_digest=digest,
        invent_green_ideal_gradient_restore=False,
        invent_green_live_qpu=False,
        demo_label=demo_label.strip() or "zne_demo",
    )


def materialise_demo_zne_probe() -> MaterialisedZneProbe:
    """Materialise the default offline ZNE demo probe."""
    return materialise_zne_probe("zne_richardson")


def materialise_readout_probe(
    mitigator_id: str = "readout_confusion",
    *,
    invent_green_ideal_gradient_restore: bool = False,
    demo_label: str = "readout_demo",
) -> MaterialisedReadoutProbe:
    """Materialise a one-qubit readout-mitigation probe."""
    decision = decide_mitigation_path(
        mitigator_id,
        invent_green_ideal_gradient_restore=invent_green_ideal_gradient_restore,
    )
    if not decision.allowed:
        raise ValueError("readout probe refused: " + "; ".join(decision.blockers))
    row = get_mitigator(mitigator_id)
    if row.kind != "readout":
        raise ValueError(f"materialise_readout_probe requires readout mitigator, got {row.kind!r}")
    raw = _run_ambient_mitigation_json(
        "readout",
        noise_scales=(1, 3, 5),
        expectation_values=(0.9, 0.7, 0.5),
        order=1,
    )
    try:
        n_qubits = int(raw["n_qubits"])
        n_basis = int(raw["n_basis"])
        total = float(raw["mitigated_probability_sum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ambient readout payload missing fields: {raw!r}") from exc
    if not math.isfinite(total):
        raise ValueError("mitigated probabilities must be finite")
    digest = _digest_payload(
        {
            "schema": "error_mitigation_readout_probe.v1",
            "mitigator_id": row.mitigator_id,
            "n_qubits": n_qubits,
            "observed": [0.88, 0.12],
            "mitigated_sum": total,
            "product_schema": ERROR_MITIGATION_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedReadoutProbe(
        mitigator_id=row.mitigator_id,
        n_qubits=n_qubits,
        n_basis=n_basis,
        mitigated_probability_sum=total,
        probe_digest=digest,
        invent_green_ideal_gradient_restore=False,
        demo_label=demo_label.strip() or "readout_demo",
    )


def studio_mitigate_claim_boundary() -> str:
    """Return the ambient Studio ``executive_mitigate`` claim boundary."""
    text = _studio_mitigate_claim_boundary_text()
    if not text or not text.strip():
        raise ValueError("Studio MITIGATE_CLAIM_BOUNDARY must be non-empty")
    if "does not run" not in text and "does not" not in text:
        raise ValueError("Studio mitigate claim boundary missing honesty fragment")
    return text


def map_error_mitigation_public_surfaces() -> tuple[dict[str, object], ...]:
    """Map public surfaces composing the error-mitigation product."""
    return (
        {
            "surface_id": "error_mitigation_product",
            "module_path": "scpn_quantum_control.error_mitigation_product",
            "role": "product_facade",
            "claim_boundary": ERROR_MITIGATION_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "mitigation_zne",
            "module_path": "scpn_quantum_control.mitigation.zne",
            "role": "ambient_zne",
            "claim_boundary": ERROR_MITIGATION_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "mitigation_readout",
            "module_path": "scpn_quantum_control.mitigation.readout_matrix",
            "role": "ambient_readout",
            "claim_boundary": ERROR_MITIGATION_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "studio_executive_mitigate",
            "module_path": "scpn_quantum_control.studio.executive_mitigate",
            "role": "studio_executive_compose",
            "claim_boundary": _studio_mitigate_claim_boundary_text(),
        },
    )


def build_error_mitigation_product_registry() -> dict[str, object]:
    """Build the versioned error-mitigation product registry payload."""
    taxonomy = [row.to_dict() for row in _TAXONOMY]
    boundaries = [row.to_dict() for row in _BOUNDARIES]
    studio_boundary = studio_mitigate_claim_boundary()
    return {
        "schema": ERROR_MITIGATION_PRODUCT_SCHEMA,
        "claim_boundary": ERROR_MITIGATION_CLAIM_BOUNDARY,
        "studio_mitigate_claim_boundary": studio_boundary,
        "mitigator_count": len(taxonomy),
        "boundary_count": len(boundaries),
        "blank_entry_count": 0,
        "hardware_submit_allowed_policy": False,
        "ideal_gradient_restore_policy": False,
        "mitiq_hard_dependency_policy": False,
        "public_surfaces": list(map_error_mitigation_public_surfaces()),
        "mitigators": taxonomy,
        "boundaries": boundaries,
        "policy_note": (
            "Mitigation taxonomy + local probes only; ZNE/readout on supplied "
            "values/calibration; no invent-green ideal AD restore or live QPU; "
            "mitiq optional_extra; compose no-submit safety and Studio execution boundaries."
        ),
    }


def assert_error_mitigation_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers taxonomy/boundaries without invent-green policies."""
    registry = dict(payload) if payload is not None else build_error_mitigation_product_registry()
    mitigators = registry.get("mitigators")
    boundaries = registry.get("boundaries")
    if not isinstance(mitigators, list) or not mitigators:
        raise ValueError(
            "error mitigation product registry must contain a non-empty mitigators list"
        )
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError(
            "error mitigation product registry must contain a non-empty boundaries list"
        )
    seen: set[str] = set()
    blank = 0
    zne_found = False
    for index, row in enumerate(mitigators):
        if not isinstance(row, Mapping):
            raise ValueError(f"mitigator row {index} must be a mapping")
        mitigator_id = row.get("mitigator_id")
        hw = row.get("hardware_submit_allowed")
        diff = row.get("differentiability")
        symbol = row.get("ambient_symbol")
        if not mitigator_id or not str(mitigator_id).strip():
            blank += 1
            continue
        mid = str(mitigator_id).strip()
        if mid in seen:
            raise ValueError(f"duplicate mitigator_id in registry: {mid!r}")
        seen.add(mid)
        if mid == "zne_richardson":
            zne_found = True
        if hw is not False:
            raise ValueError(f"mitigator {mid!r} hardware_submit_allowed must be False")
        if diff not in {"analytic_fd", "fd_only", "non_diff", "optional_extra"}:
            raise ValueError(f"mitigator {mid!r} has invalid differentiability")
        if not symbol or not str(symbol).strip():
            raise ValueError(f"mitigator {mid!r} must have non-empty ambient_symbol")
    if blank:
        raise ValueError(f"error mitigation product registry has {blank} blank or invalid entries")
    if not zne_found:
        raise ValueError("error mitigation product registry missing zne_richardson")
    expected = set(list_mitigator_ids())
    if seen != expected:
        raise ValueError(
            f"registry mitigator set drift (missing={expected - seen!r}, "
            f"extra={seen - expected!r})"
        )
    seen_b: set[str] = set()
    for index, row in enumerate(boundaries):
        if not isinstance(row, Mapping):
            raise ValueError(f"boundary row {index} must be a mapping")
        boundary_id = row.get("boundary_id")
        fail_closed = row.get("fail_closed")
        if not boundary_id or not str(boundary_id).strip():
            raise ValueError(f"boundary row {index} blank or invalid boundary_id")
        bid = str(boundary_id).strip()
        if bid in seen_b:
            raise ValueError(f"duplicate boundary_id in registry: {bid!r}")
        seen_b.add(bid)
        if fail_closed is not True:
            raise ValueError(f"boundary {bid!r} fail_closed must be True")
    expected_b = set(list_mitigation_boundary_ids())
    if seen_b != expected_b:
        raise ValueError(
            f"registry boundary set drift (missing={expected_b - seen_b!r}, "
            f"extra={seen_b - expected_b!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    mitigator_count = registry.get("mitigator_count", -1)
    if not isinstance(mitigator_count, int) or mitigator_count != len(mitigators):
        raise ValueError("mitigator_count does not match mitigators list length")
    boundary_count = registry.get("boundary_count", -1)
    if not isinstance(boundary_count, int) or boundary_count != len(boundaries):
        raise ValueError("boundary_count does not match boundaries list length")
    if registry.get("hardware_submit_allowed_policy", True) is not False:
        raise ValueError("hardware_submit_allowed_policy must be False")
    if registry.get("ideal_gradient_restore_policy", True) is not False:
        raise ValueError("ideal_gradient_restore_policy must be False")
    if registry.get("mitiq_hard_dependency_policy", True) is not False:
        raise ValueError("mitiq_hard_dependency_policy must be False")
    studio_boundary = registry.get("studio_mitigate_claim_boundary")
    if not isinstance(studio_boundary, str) or not studio_boundary.strip():
        raise ValueError("studio_mitigate_claim_boundary must be non-empty")
    return registry


__all__ = [
    "BoundaryKind",
    "DifferentiabilityClass",
    "ERROR_MITIGATION_CLAIM_BOUNDARY",
    "ERROR_MITIGATION_PRODUCT_SCHEMA",
    "MaterialisedReadoutProbe",
    "MaterialisedZneProbe",
    "MitigationBoundaryRow",
    "MitigatorKind",
    "MitigatorTaxonomyRow",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_error_mitigation_product_integrity",
    "build_error_mitigation_product_registry",
    "decide_mitigation_path",
    "get_mitigation_boundary",
    "get_mitigator",
    "iter_mitigation_boundaries",
    "iter_mitigators",
    "list_mitigation_boundary_ids",
    "list_mitigator_ids",
    "map_error_mitigation_public_surfaces",
    "materialise_demo_zne_probe",
    "materialise_readout_probe",
    "materialise_zne_probe",
    "studio_mitigate_claim_boundary",
]
