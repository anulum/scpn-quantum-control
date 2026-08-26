# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Geometric quantum control product
"""Fail-closed **geometric quantum control** product surface.

Productises a geometry glossary + inventory of QFI / McLachlan metric / QNG
surfaces and local diagnostic probes over ambient
:mod:`scpn_quantum_control.phase.variational_metric` and
:mod:`scpn_quantum_control.phase.natural_gradient`:

* versioned geometry capability catalogue (QFI spectral, McLachlan metric,
  QNG regularisation, criticality diagnostics, ambient inventory);
* metric spectrum probe (rank, nullity, condition, eigenvalues) via ambient
  :func:`~scpn_quantum_control.phase.variational_metric.mclachlan_metric`;
* QNG direction probe via ambient
  :func:`~scpn_quantum_control.phase.natural_gradient.solve_natural_gradient_direction`
  (natural-gradient singular-metric regularisation compose);
* hard-gap boundaries: invent-green experimental advantage at criticality,
  live QPU geometry claims, silent repair of indefinite metrics;
* refuse invent-green.

Does **not** claim experimental quantum advantage at criticality, submit to
QPU hardware, or replace natural-gradient regularisation policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from .phase.natural_gradient import NaturalGradientDirection, solve_natural_gradient_direction
from .phase.variational_metric import mclachlan_metric

GeometryCapabilityKind = Literal[
    "qfi_spectral",
    "mclachlan_metric",
    "qng_regularised",
    "criticality_diagnostics",
    "ambient_inventory",
]
"""Geometry capability kinds on the product catalogue."""

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
    "experimental_advantage_criticality",
    "live_qpu_geometry",
    "indefinite_metric_silent_repair",
    "full_information_geometry_textbook",
]
"""Hard-gap boundary kinds for geometric control honesty."""

GEOMETRIC_CONTROL_PRODUCT_SCHEMA: Final[str] = "geometric_control_product.v1"
"""JSON schema identifier for serialised product payloads."""

GEOMETRIC_CONTROL_CLAIM_BOUNDARY: Final[str] = (
    "Geometric quantum control product surface only; catalogues QFI/"
    "McLachlan/QNG/criticality capabilities over ambient variational_metric "
    "and natural_gradient; local metric spectrum and QNG direction probes; "
    "refuse invent-green experimental advantage at criticality, live QPU "
    "geometry claims, and silent indefinite-metric repair; compose BL-13 "
    "regularisation; residual S50.5 dashboard panel depth and S50.6 notebook "
    "depth open honestly"
)
"""Shared claim boundary for geometric control product payloads."""

# Glossary constants (S50.0) — machine-readable labels, not marketing claims.
GEOMETRY_GLOSSARY: Final[Mapping[str, str]] = {
    "QFI": (
        "Quantum Fisher information: Braunstein–Caves multiparameter bound "
        "from spectral sum over energy eigenstates of a Hamiltonian family."
    ),
    "Fubini_Study_McLachlan": (
        "McLachlan metric G_ij = Re(⟨∂_iψ|∂_jψ⟩) is the real quantum geometric "
        "tensor of a parametrised ansatz (shared by VarQRTE/VarQITE)."
    ),
    "QNG": (
        "Quantum natural gradient: precondition Euclidean gradient by a "
        "regularised metric solve; singular metrics use BL-13 damping policy."
    ),
    "criticality": (
        "Geometry criticality diagnostics: small spectral gap / large metric "
        "condition / nullity — research probes, not experimental advantage."
    ),
}


@dataclass(frozen=True, slots=True)
class GeometryCapabilityRow:
    """One geometric control capability catalogue row (S50.0 / S50.1).

    Attributes
    ----------
    capability_id
        Stable capability identifier.
    kind
        Capability kind enum.
    title
        Human-readable title.
    summary
        Short description.
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

    capability_id: str
    kind: GeometryCapabilityKind
    title: str
    summary: str
    ambient_module: str
    ambient_symbol: str
    hardware_submit_allowed: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = GEOMETRIC_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate geometry capability invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.kind not in {
            "qfi_spectral",
            "mclachlan_metric",
            "qng_regularised",
            "criticality_diagnostics",
            "ambient_inventory",
        }:
            raise ValueError(f"unknown capability kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
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
            "capability_id": self.capability_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "ambient_module": self.ambient_module,
            "ambient_symbol": self.ambient_symbol,
            "hardware_submit_allowed": self.hardware_submit_allowed,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class GeometryBoundaryRow:
    """One hard-gap boundary row for geometric control (S50.0 out-of-scope)."""

    boundary_id: str
    kind: BoundaryKind
    title: str
    failure_class: str
    summary: str
    fail_closed: bool = True
    claim_boundary: str = GEOMETRIC_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate boundary-row invariants."""
        if not self.boundary_id or not self.boundary_id.strip():
            raise ValueError("boundary_id must be non-empty")
        if self.kind not in {
            "experimental_advantage_criticality",
            "live_qpu_geometry",
            "indefinite_metric_silent_repair",
            "full_information_geometry_textbook",
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
    """Fail-closed path eligibility for geometric control product use."""

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = GEOMETRIC_CONTROL_CLAIM_BOUNDARY

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
class MaterialisedMetricDiagnosticsProbe:
    """Materialised McLachlan metric spectrum diagnostics (S50.2 / S50.3)."""

    capability_id: str
    n_parameters: int
    metric_rank: int
    metric_nullity: int
    condition_number: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    eigenvalues: tuple[float, ...]
    probe_digest: str
    invent_green_advantage: bool
    invent_green_live_qpu: bool
    demo_label: str
    claim_boundary: str = GEOMETRIC_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate metric diagnostics probe invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.n_parameters < 1:
            raise ValueError("n_parameters must be positive")
        if self.metric_rank < 0 or self.metric_rank > self.n_parameters:
            raise ValueError("metric_rank must be in [0, n_parameters]")
        if self.metric_nullity != self.n_parameters - self.metric_rank:
            raise ValueError("metric_nullity must equal n_parameters - metric_rank")
        if not math.isfinite(self.condition_number) or self.condition_number <= 0.0:
            raise ValueError("condition_number must be finite and positive")
        if not math.isfinite(self.minimum_eigenvalue):
            raise ValueError("minimum_eigenvalue must be finite")
        if not math.isfinite(self.maximum_eigenvalue):
            raise ValueError("maximum_eigenvalue must be finite")
        if len(self.eigenvalues) != self.n_parameters:
            raise ValueError("eigenvalues length must equal n_parameters")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if self.invent_green_advantage:
            raise ValueError("invent_green_advantage must be False")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "capability_id": self.capability_id,
            "n_parameters": self.n_parameters,
            "metric_rank": self.metric_rank,
            "metric_nullity": self.metric_nullity,
            "condition_number": self.condition_number,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "eigenvalues": list(self.eigenvalues),
            "probe_digest": self.probe_digest,
            "invent_green_advantage": self.invent_green_advantage,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedQngDirectionProbe:
    """Materialised QNG direction probe via ambient regularised solve (S50.2)."""

    capability_id: str
    metric_rank: int
    metric_nullity: int
    condition_number: float
    natural_gradient_norm: float
    euclidean_gradient_norm: float
    regularization_reason: str
    direction: tuple[float, ...]
    probe_digest: str
    invent_green_advantage: bool
    invent_green_live_qpu: bool
    demo_label: str
    claim_boundary: str = GEOMETRIC_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate QNG direction probe invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.metric_rank < 0:
            raise ValueError("metric_rank must be non-negative")
        if self.metric_nullity < 0:
            raise ValueError("metric_nullity must be non-negative")
        if not math.isfinite(self.condition_number) or self.condition_number <= 0.0:
            raise ValueError("condition_number must be finite and positive")
        if not math.isfinite(self.natural_gradient_norm) or self.natural_gradient_norm < 0.0:
            raise ValueError("natural_gradient_norm must be finite and non-negative")
        if not math.isfinite(self.euclidean_gradient_norm) or self.euclidean_gradient_norm < 0.0:
            raise ValueError("euclidean_gradient_norm must be finite and non-negative")
        if not self.regularization_reason or not self.regularization_reason.strip():
            raise ValueError("regularization_reason must be non-empty")
        if not self.direction:
            raise ValueError("direction must be non-empty")
        if any(not math.isfinite(item) for item in self.direction):
            raise ValueError("direction entries must be finite")
        if not self.probe_digest or len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if self.invent_green_advantage:
            raise ValueError("invent_green_advantage must be False")
        if self.invent_green_live_qpu:
            raise ValueError("invent_green_live_qpu must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "capability_id": self.capability_id,
            "metric_rank": self.metric_rank,
            "metric_nullity": self.metric_nullity,
            "condition_number": self.condition_number,
            "natural_gradient_norm": self.natural_gradient_norm,
            "euclidean_gradient_norm": self.euclidean_gradient_norm,
            "regularization_reason": self.regularization_reason,
            "direction": list(self.direction),
            "probe_digest": self.probe_digest,
            "invent_green_advantage": self.invent_green_advantage,
            "invent_green_live_qpu": self.invent_green_live_qpu,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_capabilities() -> tuple[GeometryCapabilityRow, ...]:
    """Build geometry capability catalogue (S50.0 / S50.1)."""
    return (
        GeometryCapabilityRow(
            capability_id="qfi_spectral",
            kind="qfi_spectral",
            title="Spectral QFI for coupling parameters",
            summary=(
                "Ambient analysis.qfi.compute_qfi: dense spectral QFI for small "
                "XY Hamiltonians; Cramér–Rao bounds — simulation only."
            ),
            ambient_module="scpn_quantum_control.analysis.qfi",
            ambient_symbol="compute_qfi",
            support_posture="local_research",
        ),
        GeometryCapabilityRow(
            capability_id="mclachlan_metric",
            kind="mclachlan_metric",
            title="McLachlan quantum geometric metric",
            summary=(
                "Ambient variational_metric.mclachlan_metric: G_ij = Re(⟨∂_iψ|∂_jψ⟩) "
                "from state derivatives (exact π-shift path when ansatz allows)."
            ),
            ambient_module="scpn_quantum_control.phase.variational_metric",
            ambient_symbol="mclachlan_metric",
        ),
        GeometryCapabilityRow(
            capability_id="qng_regularised",
            kind="qng_regularised",
            title="Regularised quantum natural gradient",
            summary=(
                "Ambient natural_gradient.solve_natural_gradient_direction with "
                "BL-13 damping / eigenvalue-floor / condition policy."
            ),
            ambient_module="scpn_quantum_control.phase.natural_gradient",
            ambient_symbol="solve_natural_gradient_direction",
        ),
        GeometryCapabilityRow(
            capability_id="criticality_diagnostics",
            kind="criticality_diagnostics",
            title="Metric criticality diagnostics",
            summary=(
                "Rank, nullity, condition number, and eigenvalue spectrum of the "
                "McLachlan metric — probes only, not experimental advantage."
            ),
            ambient_module="scpn_quantum_control.phase.variational_metric",
            ambient_symbol="mclachlan_metric",
            support_posture="policy_only",
        ),
        GeometryCapabilityRow(
            capability_id="ambient_inventory",
            kind="ambient_inventory",
            title="Geometry ambient inventory map",
            summary=(
                "Catalogue of fisher / metric / QNG modules (differentiable_fisher, "
                "differentiable_natural_gradient, qfi, variational_metric)."
            ),
            ambient_module="scpn_quantum_control.geometric_control_product",
            ambient_symbol="list_geometry_ambient_inventory",
            support_posture="metadata_only",
        ),
    )


def _build_boundaries() -> tuple[GeometryBoundaryRow, ...]:
    """Build hard-gap boundary catalogue."""
    return (
        GeometryBoundaryRow(
            boundary_id="experimental_advantage_criticality",
            kind="experimental_advantage_criticality",
            title="Experimental advantage at criticality",
            failure_class="experimental_advantage_criticality_refused",
            summary=(
                "Product refuses invent-green claims of experimental quantum "
                "advantage at geometric criticality (pack §2 Out)."
            ),
        ),
        GeometryBoundaryRow(
            boundary_id="live_qpu_geometry",
            kind="live_qpu_geometry",
            title="Live QPU geometry submit",
            failure_class="live_qpu_geometry_refused",
            summary=(
                "Compose BL-47 no-submit; geometry probes use local synthetic "
                "state derivatives / metrics only."
            ),
        ),
        GeometryBoundaryRow(
            boundary_id="indefinite_metric_silent_repair",
            kind="indefinite_metric_silent_repair",
            title="Silent repair of indefinite metrics",
            failure_class="indefinite_metric_silent_repair_refused",
            summary=(
                "Ambient QNG fails closed on indefinite metrics instead of "
                "silently repairing them (BL-13 policy)."
            ),
        ),
        GeometryBoundaryRow(
            boundary_id="full_information_geometry_textbook",
            kind="full_information_geometry_textbook",
            title="Full information-geometry textbook claim",
            failure_class="information_geometry_textbook_refused",
            summary=(
                "Product is a control-geometry evidence surface, not a complete "
                "information-geometry textbook (pack §2 Out)."
            ),
        ),
    )


_CAPABILITIES: Final[tuple[GeometryCapabilityRow, ...]] = _build_capabilities()
_BOUNDARIES: Final[tuple[GeometryBoundaryRow, ...]] = _build_boundaries()


def _capability_map() -> dict[str, GeometryCapabilityRow]:
    """Return capability_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, GeometryCapabilityRow] = {}
    for row in _CAPABILITIES:
        key = row.capability_id.strip()
        if not key:
            raise RuntimeError("geometry capability catalogue contains blank capability_id")
        if key in mapping:
            raise RuntimeError(f"duplicate capability_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("geometry capability catalogue must be non-empty")
    return mapping


_CAPABILITY_BY_ID: Final[Mapping[str, GeometryCapabilityRow]] = _capability_map()


def list_geometry_capability_ids() -> tuple[str, ...]:
    """Return all geometry capability identifiers in catalogue order."""
    return tuple(row.capability_id for row in _CAPABILITIES)


def list_geometry_boundary_ids() -> tuple[str, ...]:
    """Return all hard-gap boundary identifiers in catalogue order."""
    return tuple(row.boundary_id for row in _BOUNDARIES)


def list_geometry_glossary_keys() -> tuple[str, ...]:
    """Return geometry glossary keys (S50.0) in stable order."""
    return tuple(GEOMETRY_GLOSSARY.keys())


def get_geometry_glossary_entry(key: str) -> str:
    """Return one glossary definition; fail closed on blank/unknown."""
    if not key or not str(key).strip():
        raise ValueError("glossary key must be non-empty")
    k = str(key).strip()
    try:
        return GEOMETRY_GLOSSARY[k]
    except KeyError as exc:
        raise ValueError(f"unknown glossary key: {k!r}") from exc


def get_geometry_capability(capability_id: str) -> GeometryCapabilityRow:
    """Return one capability row; fail closed on blank/unknown."""
    if not capability_id or not str(capability_id).strip():
        raise ValueError("capability_id must be non-empty")
    key = str(capability_id).strip()
    try:
        return _CAPABILITY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown capability_id: {key!r}") from exc


def get_geometry_boundary(boundary_id: str) -> GeometryBoundaryRow:
    """Return one boundary row; fail closed on blank/unknown."""
    if not boundary_id or not str(boundary_id).strip():
        raise ValueError("boundary_id must be non-empty")
    key = str(boundary_id).strip()
    for row in _BOUNDARIES:
        if row.boundary_id == key:
            return row
    raise ValueError(f"unknown boundary_id: {key!r}")


def iter_geometry_capabilities(
    *,
    kind: GeometryCapabilityKind | None = None,
) -> tuple[GeometryCapabilityRow, ...]:
    """Return filtered capability rows in stable order."""
    rows: Sequence[GeometryCapabilityRow] = _CAPABILITIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def iter_geometry_boundaries(
    *,
    kind: BoundaryKind | None = None,
) -> tuple[GeometryBoundaryRow, ...]:
    """Return filtered boundary rows in stable order."""
    rows: Sequence[GeometryBoundaryRow] = _BOUNDARIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def list_geometry_ambient_inventory() -> tuple[dict[str, object], ...]:
    """Return ambient inventory map for fisher/metric/QNG modules (S50.1)."""
    return (
        {
            "module_path": "scpn_quantum_control.analysis.qfi",
            "symbols": ("compute_qfi", "qfi_gap_tradeoff", "QFIResult"),
            "role": "spectral_qfi_small_system",
        },
        {
            "module_path": "scpn_quantum_control.phase.variational_metric",
            "symbols": ("mclachlan_metric", "analytic_state_derivatives"),
            "role": "mclachlan_ansatz_geometry",
        },
        {
            "module_path": "scpn_quantum_control.phase.natural_gradient",
            "symbols": (
                "solve_natural_gradient_direction",
                "parameter_shift_natural_gradient_descent",
            ),
            "role": "qng_regularised_solve_trainability_boundary",
        },
        {
            "module_path": "scpn_quantum_control.differentiable_fisher",
            "symbols": ("empirical_fisher_vector_product", "empirical_fisher_conjugate_gradient"),
            "role": "empirical_fisher_ops",
        },
        {
            "module_path": "scpn_quantum_control.differentiable_natural_gradient",
            "symbols": ("natural_gradient", "NaturalGradientOptimizer"),
            "role": "diff_ng_optimizer",
        },
    )


def decide_geometry_path(
    capability_id: str,
    *,
    invent_green_advantage: bool = False,
    invent_green_live_qpu: bool = False,
    invent_green_indefinite_silent_repair: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a geometric control product path may proceed."""
    row = get_geometry_capability(capability_id)
    blockers: list[str] = []
    if invent_green_advantage:
        blockers.append(
            "invent-green experimental advantage at criticality refused "
            f"(capability={row.capability_id}; geometry probes only)"
        )
    if invent_green_live_qpu:
        blockers.append(
            "invent-green live QPU geometry refused "
            f"(capability={row.capability_id}; BL-47 no-submit compose)"
        )
    if invent_green_indefinite_silent_repair:
        blockers.append(
            "invent-green silent repair of indefinite metrics refused "
            f"(capability={row.capability_id}; ambient QNG fails closed)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="geometry path refused under product honesty gates",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"capability {row.capability_id!r} may proceed as local geometry "
            "research / diagnostics only"
        ),
        blockers=(),
    )


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Canonical SHA-256 over a JSON-serialisable mapping."""
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _demo_state_derivatives(
    n_parameters: int = 3,
    dim: int = 4,
    seed: int = 50,
) -> NDArray[np.complex128]:
    """Return a deterministic synthetic state-derivative batch for demos."""
    if n_parameters < 1:
        raise ValueError("n_parameters must be positive")
    if dim < 2:
        raise ValueError("dim must be at least 2")
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(n_parameters, dim))
    imag = rng.normal(size=(n_parameters, dim))
    return (real + 1j * imag).astype(np.complex128)


def materialise_metric_diagnostics_probe(
    capability_id: str = "criticality_diagnostics",
    *,
    n_parameters: int = 3,
    dim: int = 4,
    seed: int = 50,
    eigenvalue_floor: float = 1e-12,
    invent_green_advantage: bool = False,
    invent_green_live_qpu: bool = False,
    demo_label: str = "metric_diagnostics_demo",
) -> MaterialisedMetricDiagnosticsProbe:
    """Materialise McLachlan metric spectrum diagnostics (S50.2 / S50.3)."""
    decision = decide_geometry_path(
        capability_id,
        invent_green_advantage=invent_green_advantage,
        invent_green_live_qpu=invent_green_live_qpu,
    )
    if not decision.allowed:
        raise ValueError("metric diagnostics probe refused: " + "; ".join(decision.blockers))
    row = get_geometry_capability(capability_id)
    if row.kind not in {"criticality_diagnostics", "mclachlan_metric"}:
        raise ValueError(
            f"materialise_metric_diagnostics_probe requires metric-family capability, "
            f"got {row.kind!r}"
        )
    if eigenvalue_floor < 0.0 or not math.isfinite(eigenvalue_floor):
        raise ValueError("eigenvalue_floor must be finite and non-negative")
    derivs = _demo_state_derivatives(n_parameters=n_parameters, dim=dim, seed=seed)
    metric = mclachlan_metric(derivs)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("ambient McLachlan metric must be square")
    if metric.shape[0] != n_parameters:
        raise ValueError("ambient metric size must match n_parameters")
    if not np.all(np.isfinite(metric)):
        raise ValueError("ambient McLachlan metric must be finite")
    evals = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    evals_list = tuple(float(v) for v in evals)
    rank = int(np.sum(evals > eigenvalue_floor))
    nullity = int(n_parameters - rank)
    min_ev = float(evals[0])
    max_ev = float(evals[-1])
    if max_ev <= eigenvalue_floor:
        condition = float("inf")
    else:
        positive = evals[evals > eigenvalue_floor]
        condition = float(positive[-1] / positive[0]) if positive.size else float("inf")
    if not math.isfinite(condition) or condition <= 0.0:
        # Fail closed rather than invent a green finite condition on singular metrics.
        raise ValueError(
            "metric condition number is non-finite or non-positive; "
            "singular metrics require BL-13 QNG regularisation, not invent-green condition"
        )
    digest = _digest_payload(
        {
            "schema": "geometric_control_metric_diagnostics.v1",
            "capability_id": row.capability_id,
            "n_parameters": n_parameters,
            "seed": seed,
            "eigenvalues": list(evals_list),
            "metric_rank": rank,
            "metric_nullity": nullity,
            "condition_number": condition,
            "product_schema": GEOMETRIC_CONTROL_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedMetricDiagnosticsProbe(
        capability_id=row.capability_id,
        n_parameters=n_parameters,
        metric_rank=rank,
        metric_nullity=nullity,
        condition_number=condition,
        minimum_eigenvalue=min_ev,
        maximum_eigenvalue=max_ev,
        eigenvalues=evals_list,
        probe_digest=digest,
        invent_green_advantage=False,
        invent_green_live_qpu=False,
        demo_label=demo_label.strip() or "metric_diagnostics_demo",
    )


def materialise_demo_metric_diagnostics_probe() -> MaterialisedMetricDiagnosticsProbe:
    """Materialise the default offline metric diagnostics demo probe."""
    return materialise_metric_diagnostics_probe("criticality_diagnostics")


def materialise_qng_direction_probe(
    capability_id: str = "qng_regularised",
    *,
    n_parameters: int = 3,
    dim: int = 4,
    seed: int = 50,
    invent_green_advantage: bool = False,
    invent_green_live_qpu: bool = False,
    demo_label: str = "qng_direction_demo",
) -> MaterialisedQngDirectionProbe:
    """Materialise a regularised QNG direction via ambient solver (S50.2)."""
    decision = decide_geometry_path(
        capability_id,
        invent_green_advantage=invent_green_advantage,
        invent_green_live_qpu=invent_green_live_qpu,
    )
    if not decision.allowed:
        raise ValueError("qng direction probe refused: " + "; ".join(decision.blockers))
    row = get_geometry_capability(capability_id)
    if row.kind != "qng_regularised":
        raise ValueError(
            f"materialise_qng_direction_probe requires qng_regularised, got {row.kind!r}"
        )
    derivs = _demo_state_derivatives(n_parameters=n_parameters, dim=dim, seed=seed)
    metric = mclachlan_metric(derivs)
    rng = np.random.default_rng(seed + 1)
    gradient = rng.normal(size=n_parameters).astype(np.float64)
    direction: NaturalGradientDirection = solve_natural_gradient_direction(gradient, metric)
    direction_tuple = tuple(float(x) for x in direction.direction)
    digest = _digest_payload(
        {
            "schema": "geometric_control_qng_direction.v1",
            "capability_id": row.capability_id,
            "n_parameters": n_parameters,
            "seed": seed,
            "metric_rank": int(direction.metric_rank),
            "metric_nullity": int(direction.metric_nullity),
            "condition_number": float(direction.condition_number),
            "natural_gradient_norm": float(direction.natural_gradient_norm),
            "direction": list(direction_tuple),
            "regularization_reason": direction.regularization_reason,
            "product_schema": GEOMETRIC_CONTROL_PRODUCT_SCHEMA,
        }
    )
    return MaterialisedQngDirectionProbe(
        capability_id=row.capability_id,
        metric_rank=int(direction.metric_rank),
        metric_nullity=int(direction.metric_nullity),
        condition_number=float(direction.condition_number),
        natural_gradient_norm=float(direction.natural_gradient_norm),
        euclidean_gradient_norm=float(direction.euclidean_gradient_norm),
        regularization_reason=str(direction.regularization_reason),
        direction=direction_tuple,
        probe_digest=digest,
        invent_green_advantage=False,
        invent_green_live_qpu=False,
        demo_label=demo_label.strip() or "qng_direction_demo",
    )


def map_geometric_control_public_surfaces() -> tuple[dict[str, object], ...]:
    """Map public surfaces composing the geometric control product."""
    return (
        {
            "surface_id": "geometric_control_product",
            "module_path": "scpn_quantum_control.geometric_control_product",
            "role": "product_facade",
            "claim_boundary": GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "variational_metric",
            "module_path": "scpn_quantum_control.phase.variational_metric",
            "role": "ambient_mclachlan",
            "claim_boundary": GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "natural_gradient",
            "module_path": "scpn_quantum_control.phase.natural_gradient",
            "role": "ambient_qng_trainability_boundary",
            "claim_boundary": GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "analysis_qfi",
            "module_path": "scpn_quantum_control.analysis.qfi",
            "role": "ambient_spectral_qfi",
            "claim_boundary": GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
        },
    )


def build_geometric_control_product_registry() -> dict[str, object]:
    """Build the versioned geometric control product registry payload."""
    capabilities = [row.to_dict() for row in _CAPABILITIES]
    boundaries = [row.to_dict() for row in _BOUNDARIES]
    inventory = list(list_geometry_ambient_inventory())
    glossary = {key: GEOMETRY_GLOSSARY[key] for key in GEOMETRY_GLOSSARY}
    return {
        "schema": GEOMETRIC_CONTROL_PRODUCT_SCHEMA,
        "claim_boundary": GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
        "capability_count": len(capabilities),
        "boundary_count": len(boundaries),
        "blank_entry_count": 0,
        "hardware_submit_allowed_policy": False,
        "experimental_advantage_criticality_policy": False,
        "indefinite_metric_silent_repair_policy": False,
        "public_surfaces": list(map_geometric_control_public_surfaces()),
        "capabilities": capabilities,
        "boundaries": boundaries,
        "ambient_inventory": inventory,
        "glossary": glossary,
        "policy_note": (
            "Geometry catalogue + local McLachlan/QNG probes only; no invent-green "
            "advantage at criticality or live QPU; BL-13 regularisation compose; "
            "S50.5/S50.6 residual dashboard/notebook depth honest."
        ),
    }


def assert_geometric_control_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers capabilities/boundaries without invent-green policies."""
    registry = dict(payload) if payload is not None else build_geometric_control_product_registry()
    capabilities = registry.get("capabilities")
    boundaries = registry.get("boundaries")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError(
            "geometric control product registry must contain a non-empty capabilities list"
        )
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError(
            "geometric control product registry must contain a non-empty boundaries list"
        )
    seen: set[str] = set()
    blank = 0
    mclachlan_found = False
    for index, row in enumerate(capabilities):
        if not isinstance(row, Mapping):
            raise ValueError(f"capability row {index} must be a mapping")
        capability_id = row.get("capability_id")
        hw = row.get("hardware_submit_allowed")
        symbol = row.get("ambient_symbol")
        if not capability_id or not str(capability_id).strip():
            blank += 1
            continue
        cid = str(capability_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate capability_id in registry: {cid!r}")
        seen.add(cid)
        if cid == "mclachlan_metric":
            mclachlan_found = True
        if hw is not False:
            raise ValueError(f"capability {cid!r} hardware_submit_allowed must be False")
        if not symbol or not str(symbol).strip():
            raise ValueError(f"capability {cid!r} must have non-empty ambient_symbol")
    if blank:
        raise ValueError(
            f"geometric control product registry has {blank} blank or invalid entries"
        )
    if not mclachlan_found:
        raise ValueError("geometric control product registry missing mclachlan_metric")
    expected = set(list_geometry_capability_ids())
    if seen != expected:
        raise ValueError(
            f"registry capability set drift (missing={expected - seen!r}, "
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
    expected_b = set(list_geometry_boundary_ids())
    if seen_b != expected_b:
        raise ValueError(
            f"registry boundary set drift (missing={expected_b - seen_b!r}, "
            f"extra={seen_b - expected_b!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    capability_count = registry.get("capability_count", -1)
    if not isinstance(capability_count, int) or capability_count != len(capabilities):
        raise ValueError("capability_count does not match capabilities list length")
    boundary_count = registry.get("boundary_count", -1)
    if not isinstance(boundary_count, int) or boundary_count != len(boundaries):
        raise ValueError("boundary_count does not match boundaries list length")
    if registry.get("hardware_submit_allowed_policy", True) is not False:
        raise ValueError("hardware_submit_allowed_policy must be False")
    if registry.get("experimental_advantage_criticality_policy", True) is not False:
        raise ValueError("experimental_advantage_criticality_policy must be False")
    if registry.get("indefinite_metric_silent_repair_policy", True) is not False:
        raise ValueError("indefinite_metric_silent_repair_policy must be False")
    glossary = registry.get("glossary")
    if not isinstance(glossary, Mapping) or not glossary:
        raise ValueError("glossary must be a non-empty mapping")
    for key in list_geometry_glossary_keys():
        if key not in glossary:
            raise ValueError(f"glossary missing key {key!r}")
    inventory = registry.get("ambient_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("ambient_inventory must be a non-empty list")
    return registry


__all__ = [
    "BoundaryKind",
    "GEOMETRIC_CONTROL_CLAIM_BOUNDARY",
    "GEOMETRIC_CONTROL_PRODUCT_SCHEMA",
    "GEOMETRY_GLOSSARY",
    "GeometryBoundaryRow",
    "GeometryCapabilityKind",
    "GeometryCapabilityRow",
    "MaterialisedMetricDiagnosticsProbe",
    "MaterialisedQngDirectionProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_geometric_control_product_integrity",
    "build_geometric_control_product_registry",
    "decide_geometry_path",
    "get_geometry_boundary",
    "get_geometry_capability",
    "get_geometry_glossary_entry",
    "iter_geometry_boundaries",
    "iter_geometry_capabilities",
    "list_geometry_ambient_inventory",
    "list_geometry_boundary_ids",
    "list_geometry_capability_ids",
    "list_geometry_glossary_keys",
    "map_geometric_control_public_surfaces",
    "materialise_demo_metric_diagnostics_probe",
    "materialise_metric_diagnostics_probe",
    "materialise_qng_direction_probe",
]
