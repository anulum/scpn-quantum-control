# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Thermodynamics readiness product
"""Fail-closed **thermodynamics readiness** product surface.

Productises the honest quantum-thermodynamics readiness model over ambient
:mod:`scpn_quantum_control.thermodynamics.readiness` and inventories Free Energy
Principle (FEP) modules as **research-only** (research-lane / tier C) until a concrete
sync-control hook is proven:

* versioned readiness-capability catalogue (K-sweep, entropy production, work
  identity, heat dissipation) with machine-checked claim boundary;
* FEP inventory rows (predictive coding, variational free energy) marked
  ``research_only`` with research-lane pointers — never product peak claims;
* K-sweep probe via ambient
  :func:`~scpn_quantum_control.thermodynamics.readiness.run_k_sweep_protocol`;
* refuse invent-green thermodynamic peak claims and hardware submission;
* fail-closed blanks/unknowns.

Does **not** submit to QPU hardware, promote FEP to product status, or assert
thermodynamic peak advantage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .thermodynamics.readiness import (
    CLAIM_BOUNDARY as AMBIENT_CLAIM_BOUNDARY,
)
from .thermodynamics.readiness import (
    QUANTUM_THERMO_SCHEMA,
    ThermodynamicSweepConfig,
    ThermodynamicSweepResult,
    quantum_thermo_payload,
    run_k_sweep_protocol,
)

ReadinessCapabilityKind = Literal[
    "k_sweep_protocol",
    "entropy_production",
    "work_identity",
    "heat_dissipation",
    "claim_boundary_gate",
]
"""Readiness capability kinds on the product catalogue."""

FepInventoryStatus = Literal[
    "research_only",
    "product_hook_open",
    "permanent_boundary",
]
"""FEP research-inventory status badges."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
    "research_only",
]
"""Support posture badges for thermo readiness product rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

THERMO_READINESS_PRODUCT_SCHEMA: Final[str] = "thermo_readiness_product.v1"
"""JSON schema identifier for serialised product payloads."""

THERMO_READINESS_CLAIM_BOUNDARY: Final[str] = (
    "Thermodynamics readiness product surface only; catalogues no-submit "
    "readiness capabilities over ambient thermodynamics.readiness; inventories "
    "FEP modules as research-only; refuse invent-green thermodynamic peak claims "
    "and hardware submission; optional future sync-control FEP hook design remains "
    "open honestly without implementation"
)
"""Shared claim boundary for thermo readiness product payloads."""

_EXPECTED_AMBIENT_BOUNDARY_FRAGMENT: Final[str] = "no thermodynamic peak"
"""Substring required in the ambient claim boundary."""


@dataclass(frozen=True, slots=True)
class ReadinessCapabilityRow:
    """One thermodynamics-readiness capability catalogue row.

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
    ambient_symbol
        Ambient module symbol exercised for this capability.
    hardware_submission_allowed
        Must remain False on product surface.
    thermodynamic_peak_claim_allowed
        Must remain False on product surface.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    capability_id: str
    kind: ReadinessCapabilityKind
    title: str
    summary: str
    ambient_symbol: str
    hardware_submission_allowed: bool = False
    thermodynamic_peak_claim_allowed: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = THERMO_READINESS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate readiness capability invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if self.kind not in {
            "k_sweep_protocol",
            "entropy_production",
            "work_identity",
            "heat_dissipation",
            "claim_boundary_gate",
        }:
            raise ValueError(f"unknown capability kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.ambient_symbol or not self.ambient_symbol.strip():
            raise ValueError("ambient_symbol must be non-empty")
        if self.hardware_submission_allowed:
            raise ValueError("hardware_submission_allowed must be False on product surface")
        if self.thermodynamic_peak_claim_allowed:
            raise ValueError("thermodynamic_peak_claim_allowed must be False on product surface")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "metadata_only",
            "research_only",
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
            "ambient_symbol": self.ambient_symbol,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "thermodynamic_peak_claim_allowed": self.thermodynamic_peak_claim_allowed,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class FepInventoryRow:
    """One FEP research-only inventory row.

    Attributes
    ----------
    module_id
        Stable FEP module identifier.
    module_path
        Import path of the ambient research module.
    title
        Human-readable title.
    summary
        Short description.
    status
        Inventory status (research_only by default).
    research_lane_pointer
        Deep-analysis research-lane pointer.
    product_hook_proven
        Must remain False until owner-approved promotion.
    claim_boundary
        Non-promotional claim boundary.

    """

    module_id: str
    module_path: str
    title: str
    summary: str
    status: FepInventoryStatus = "research_only"
    research_lane_pointer: str = "deep_analysis_research_lane_registry.fep"
    product_hook_proven: bool = False
    claim_boundary: str = THERMO_READINESS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate FEP inventory invariants."""
        if not self.module_id or not self.module_id.strip():
            raise ValueError("module_id must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.status not in {"research_only", "product_hook_open", "permanent_boundary"}:
            raise ValueError(f"unknown FEP inventory status: {self.status!r}")
        if not self.research_lane_pointer or not self.research_lane_pointer.strip():
            raise ValueError("research_lane_pointer must be non-empty")
        # Specific research_only refusal first so both messages remain reachable.
        if self.status == "research_only" and self.product_hook_proven:
            raise ValueError("research_only rows cannot set product_hook_proven")
        if self.product_hook_proven:
            raise ValueError(
                "product_hook_proven must be False on product surface "
                "(research-only inventory; no invent-green FEP promotion)"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "module_id": self.module_id,
            "module_path": self.module_path,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "research_lane_pointer": self.research_lane_pointer,
            "product_hook_proven": self.product_hook_proven,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for thermo readiness product use.

    Attributes
    ----------
    outcome
        Allowed or refused.
    allowed
        Whether the path may proceed under this product.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    claim_boundary
        Non-promotional claim boundary.

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = THERMO_READINESS_CLAIM_BOUNDARY

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
class MaterialisedKSweepProbe:
    """Materialised K-sweep readiness probe via ambient run_k_sweep_protocol.

    Attributes
    ----------
    capability_id
        Capability used for the probe (typically k_sweep_protocol).
    schema
        Ambient sweep result schema.
    peak_k
        Peak K candidate from the calibrated protocol (not a peak claim).
    row_count
        Number of sweep rows.
    hardware_submission_allowed
        Always False.
    thermodynamic_peak_claim_allowed
        Always False.
    ambient_claim_boundary
        Ambient CLAIM_BOUNDARY string.
    falsifier
        Ambient falsifier text.
    probe_digest
        Canonical SHA-256 over key sweep fields (anti-cheat).
    demo_label
        Demo fixture label.
    claim_boundary
        Product claim boundary.

    """

    capability_id: str
    schema: str
    peak_k: float
    row_count: int
    hardware_submission_allowed: bool
    thermodynamic_peak_claim_allowed: bool
    ambient_claim_boundary: str
    falsifier: str
    probe_digest: str
    demo_label: str
    claim_boundary: str = THERMO_READINESS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate K-sweep probe invariants."""
        if not self.capability_id or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if not self.schema or not self.schema.strip():
            raise ValueError("schema must be non-empty")
        if self.row_count < 3:
            raise ValueError("row_count must be at least 3 (K-sweep grid minimum)")
        if self.hardware_submission_allowed:
            raise ValueError("hardware_submission_allowed must be False")
        if self.thermodynamic_peak_claim_allowed:
            raise ValueError("thermodynamic_peak_claim_allowed must be False")
        if not self.ambient_claim_boundary or not self.ambient_claim_boundary.strip():
            raise ValueError("ambient_claim_boundary must be non-empty")
        if _EXPECTED_AMBIENT_BOUNDARY_FRAGMENT not in self.ambient_claim_boundary:
            raise ValueError("ambient_claim_boundary must retain no-thermodynamic-peak honesty")
        if not self.falsifier or not self.falsifier.strip():
            raise ValueError("falsifier must be non-empty")
        if not self.probe_digest or not self.probe_digest.strip():
            raise ValueError("probe_digest must be non-empty")
        if len(self.probe_digest) != 64:
            raise ValueError("probe_digest must be a 64-char hex SHA-256")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "capability_id": self.capability_id,
            "schema": self.schema,
            "peak_k": self.peak_k,
            "row_count": self.row_count,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "thermodynamic_peak_claim_allowed": self.thermodynamic_peak_claim_allowed,
            "ambient_claim_boundary": self.ambient_claim_boundary,
            "falsifier": self.falsifier,
            "probe_digest": self.probe_digest,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_capabilities() -> tuple[ReadinessCapabilityRow, ...]:
    """Build the thermodynamics-readiness capability catalogue."""
    return (
        ReadinessCapabilityRow(
            capability_id="k_sweep_protocol",
            kind="k_sweep_protocol",
            title="K-sweep thermodynamics-readiness protocol",
            summary=(
                "Deterministic no-submit K-sweep over calibrated entropy/heat/"
                "work observables; peak_k is a candidate only, never a peak claim."
            ),
            ambient_symbol="run_k_sweep_protocol",
            support_posture="local_research",
        ),
        ReadinessCapabilityRow(
            capability_id="entropy_production",
            kind="entropy_production",
            title="Entropy-production rate accounting",
            summary=(
                "Finite-rate entropy production for calibrated protocol points; "
                "negative budgets refuse promotion."
            ),
            ambient_symbol="entropy_production_rate",
            support_posture="local_research",
        ),
        ReadinessCapabilityRow(
            capability_id="work_identity",
            kind="work_identity",
            title="Calibrated Jarzynski work identity",
            summary=(
                "Jarzynski free-energy estimate from calibrated work samples "
                "with irreversibility residual diagnostics."
            ),
            ambient_symbol="calibrated_work_identity",
            support_posture="local_research",
        ),
        ReadinessCapabilityRow(
            capability_id="heat_dissipation",
            kind="heat_dissipation",
            title="Heat dissipation from jump statistics",
            summary=(
                "Heat-current estimate from Lindblad jump counts; protocol "
                "estimate only, not hardware evidence."
            ),
            ambient_symbol="heat_dissipation_rate",
            support_posture="local_research",
        ),
        ReadinessCapabilityRow(
            capability_id="claim_boundary_gate",
            kind="claim_boundary_gate",
            title="Machine-checked claim boundary",
            summary=(
                "Ambient CLAIM_BOUNDARY must retain no-peak-claim and "
                "no-hardware-submission honesty language."
            ),
            ambient_symbol="CLAIM_BOUNDARY",
            support_posture="policy_only",
        ),
    )


def _build_fep_inventory() -> tuple[FepInventoryRow, ...]:
    """Build the FEP research-only inventory."""
    return (
        FepInventoryRow(
            module_id="predictive_coding",
            module_path="scpn_quantum_control.fep.predictive_coding",
            title="Hierarchical predictive coding",
            summary=(
                "Friston-style hierarchical prediction-error message passing "
                "mapped onto SCPN layers; research-only until sync-control hook "
                "is proven."
            ),
            status="research_only",
            research_lane_pointer="deep_analysis_research_lane_registry.fep.predictive_coding",
        ),
        FepInventoryRow(
            module_id="variational_free_energy",
            module_path="scpn_quantum_control.fep.variational_free_energy",
            title="Variational free energy",
            summary=(
                "Gaussian variational free-energy decomposition (complexity + "
                "accuracy); research-only inventory, not a product peak claim."
            ),
            status="research_only",
            research_lane_pointer="deep_analysis_research_lane_registry.fep.variational_free_energy",
        ),
    )


_CAPABILITIES: Final[tuple[ReadinessCapabilityRow, ...]] = _build_capabilities()
_FEP_INVENTORY: Final[tuple[FepInventoryRow, ...]] = _build_fep_inventory()


def _capability_map() -> dict[str, ReadinessCapabilityRow]:
    """Return capability_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, ReadinessCapabilityRow] = {}
    for row in _CAPABILITIES:
        key = row.capability_id.strip()
        if not key:
            raise RuntimeError("readiness capability catalogue contains blank capability_id")
        if key in mapping:
            raise RuntimeError(f"duplicate capability_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("readiness capability catalogue must be non-empty")
    return mapping


_CAPABILITY_BY_ID: Final[Mapping[str, ReadinessCapabilityRow]] = _capability_map()


def _fep_map() -> dict[str, FepInventoryRow]:
    """Return module_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, FepInventoryRow] = {}
    for row in _FEP_INVENTORY:
        key = row.module_id.strip()
        if not key:
            raise RuntimeError("FEP inventory contains blank module_id")
        if key in mapping:
            raise RuntimeError(f"duplicate module_id in FEP inventory: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("FEP inventory must be non-empty")
    return mapping


_FEP_BY_ID: Final[Mapping[str, FepInventoryRow]] = _fep_map()


def verify_ambient_claim_boundary() -> str:
    """Return the machine-checked ambient claim boundary.

    Returns
    -------
    str
        Ambient claim boundary string.

    Raises
    ------
    ValueError
        If ambient boundary is blank or missing required honesty fragments.

    """
    boundary = AMBIENT_CLAIM_BOUNDARY
    if not boundary or not str(boundary).strip():
        raise ValueError("ambient CLAIM_BOUNDARY must be non-empty")
    text = str(boundary).strip()
    if _EXPECTED_AMBIENT_BOUNDARY_FRAGMENT not in text:
        raise ValueError("ambient CLAIM_BOUNDARY must state no thermodynamic peak claim")
    if "no hardware submission" not in text:
        raise ValueError("ambient CLAIM_BOUNDARY must state no hardware submission")
    return text


def list_readiness_capability_ids() -> tuple[str, ...]:
    """Return all readiness capability identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable capability ids.

    """
    return tuple(row.capability_id for row in _CAPABILITIES)


def list_fep_module_ids() -> tuple[str, ...]:
    """Return all FEP inventory module identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable FEP module ids.

    """
    return tuple(row.module_id for row in _FEP_INVENTORY)


def get_readiness_capability(capability_id: str) -> ReadinessCapabilityRow:
    """Return one readiness capability row; fail closed on blank/unknown.

    Parameters
    ----------
    capability_id
        Capability identifier.

    Returns
    -------
    ReadinessCapabilityRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not capability_id or not str(capability_id).strip():
        raise ValueError("capability_id must be non-empty")
    key = str(capability_id).strip()
    try:
        return _CAPABILITY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown capability_id: {key!r}") from exc


def get_fep_inventory_row(module_id: str) -> FepInventoryRow:
    """Return one FEP inventory row; fail closed on blank/unknown.

    Parameters
    ----------
    module_id
        FEP module identifier.

    Returns
    -------
    FepInventoryRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not module_id or not str(module_id).strip():
        raise ValueError("module_id must be non-empty")
    key = str(module_id).strip()
    try:
        return _FEP_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown module_id: {key!r}") from exc


def iter_readiness_capabilities(
    *,
    kind: ReadinessCapabilityKind | None = None,
) -> tuple[ReadinessCapabilityRow, ...]:
    """Return filtered readiness capability rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.

    Returns
    -------
    tuple[ReadinessCapabilityRow, ...]
        Matching rows.

    """
    rows: Sequence[ReadinessCapabilityRow] = _CAPABILITIES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def iter_fep_inventory(
    *,
    status: FepInventoryStatus | None = None,
) -> tuple[FepInventoryRow, ...]:
    """Return filtered FEP inventory rows in stable order.

    Parameters
    ----------
    status
        Optional status filter.

    Returns
    -------
    tuple[FepInventoryRow, ...]
        Matching rows.

    """
    rows: Sequence[FepInventoryRow] = _FEP_INVENTORY
    if status is not None:
        rows = tuple(row for row in rows if row.status == status)
    return tuple(rows)


def decide_readiness_path(
    capability_id: str,
    *,
    invent_green_peak_claim: bool = False,
    invent_green_hardware_submit: bool = False,
    invent_green_fep_product: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a thermodynamics-readiness path may proceed.

    Parameters
    ----------
    capability_id
        Readiness capability identifier.
    invent_green_peak_claim
        If true, refuse (no thermodynamic peak claim).
    invent_green_hardware_submit
        If true, refuse (no hardware submission).
    invent_green_fep_product
        If true, refuse because FEP remains research-only.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    row = get_readiness_capability(capability_id)
    # The ambient boundary must remain honest before any allowed path.
    verify_ambient_claim_boundary()
    blockers: list[str] = []
    if invent_green_peak_claim:
        blockers.append(
            "invent-green thermodynamic peak claim refused "
            f"(capability={row.capability_id}; ambient readiness estimate only)"
        )
    if invent_green_hardware_submit:
        blockers.append(
            "invent-green hardware submission refused "
            f"(capability={row.capability_id}; no-submit readiness only)"
        )
    if invent_green_fep_product:
        blockers.append(
            "invent-green FEP product promotion refused "
            f"(capability={row.capability_id}; FEP inventory is research_only)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="thermo readiness path refused under product honesty gates",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"capability {row.capability_id!r} may proceed as no-submit readiness "
            "estimate only (no peak claim, no FEP product promotion)"
        ),
        blockers=(),
    )


def _digest_sweep_result(result: ThermodynamicSweepResult) -> str:
    """Canonical SHA-256 over key K-sweep fields."""
    payload = {
        "schema": result.schema,
        "k_values": list(result.k_values),
        "peak_k": result.peak_k,
        "falsifier": result.falsifier,
        "hardware_submission_allowed": result.hardware_submission_allowed,
        "hardware_claim_allowed": result.hardware_claim_allowed,
        "rows": [
            {
                "k_value": row.k_value,
                "entropy_production_nat_per_s": row.entropy_production_nat_per_s,
                "heat_current_joule_per_s": row.heat_current_joule_per_s,
                "hardware_submission_allowed": row.hardware_submission_allowed,
                "hardware_claim_allowed": row.hardware_claim_allowed,
            }
            for row in result.rows
        ],
        "product_schema": THERMO_READINESS_PRODUCT_SCHEMA,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialise_k_sweep_probe(
    capability_id: str = "k_sweep_protocol",
    *,
    config: ThermodynamicSweepConfig | None = None,
    invent_green_peak_claim: bool = False,
    invent_green_hardware_submit: bool = False,
    demo_label: str = "s9_k_sweep_demo",
) -> MaterialisedKSweepProbe:
    """Materialise a real ambient K-sweep probe for a readiness capability.

    Parameters
    ----------
    capability_id
        Must be ``k_sweep_protocol`` (other kinds refuse).
    config
        Optional ambient :class:`ThermodynamicSweepConfig`.
    invent_green_peak_claim
        If true, refuse.
    invent_green_hardware_submit
        If true, refuse.
    demo_label
        Demo fixture label.

    Returns
    -------
    MaterialisedKSweepProbe
        Probe with ambient peak_k candidate and anti-cheat digest.

    Raises
    ------
    ValueError
        If path refused, wrong capability, or ambient honesty broken.

    """
    decision = decide_readiness_path(
        capability_id,
        invent_green_peak_claim=invent_green_peak_claim,
        invent_green_hardware_submit=invent_green_hardware_submit,
    )
    if not decision.allowed:
        raise ValueError("k-sweep probe refused: " + "; ".join(decision.blockers))
    row = get_readiness_capability(capability_id)
    if row.kind != "k_sweep_protocol":
        raise ValueError(
            f"materialise_k_sweep_probe requires kind=k_sweep_protocol, got {row.kind!r}"
        )
    ambient_boundary = verify_ambient_claim_boundary()
    result = run_k_sweep_protocol(config)
    if result.hardware_submission_allowed or result.hardware_claim_allowed:
        raise ValueError("ambient K-sweep must keep hardware flags False")
    if not result.rows:
        raise ValueError("ambient K-sweep returned empty rows")
    digest = _digest_sweep_result(result)
    return MaterialisedKSweepProbe(
        capability_id=row.capability_id,
        schema=result.schema,
        peak_k=float(result.peak_k),
        row_count=len(result.rows),
        hardware_submission_allowed=False,
        thermodynamic_peak_claim_allowed=False,
        ambient_claim_boundary=ambient_boundary,
        falsifier=result.falsifier,
        probe_digest=digest,
        demo_label=demo_label.strip() or "s9_k_sweep_demo",
    )


def materialise_demo_k_sweep_probe() -> MaterialisedKSweepProbe:
    """Materialise the default offline K-sweep demo probe.

    Returns
    -------
    MaterialisedKSweepProbe
        Default demo probe over ambient :func:`run_k_sweep_protocol`.

    """
    return materialise_k_sweep_probe("k_sweep_protocol")


def materialise_quantum_thermo_payload_probe() -> dict[str, object]:
    """Return ambient :func:`quantum_thermo_payload` with product honesty checks.

    Returns
    -------
    dict[str, object]
        Ambient payload plus product schema/boundary annotations.

    Raises
    ------
    ValueError
        If ambient invent-green flags or boundary honesty fail.

    """
    ambient_boundary = verify_ambient_claim_boundary()
    raw = quantum_thermo_payload()
    if raw.get("hardware_submission_allowed") is not False:
        raise ValueError("ambient payload hardware_submission_allowed must be False")
    if raw.get("thermodynamic_peak_claim_allowed") is not False:
        raise ValueError("ambient payload thermodynamic_peak_claim_allowed must be False")
    if raw.get("no_qpu_submission") is not True:
        raise ValueError("ambient payload no_qpu_submission must be True")
    claim = raw.get("claim_boundary")
    if not isinstance(claim, str) or _EXPECTED_AMBIENT_BOUNDARY_FRAGMENT not in claim:
        raise ValueError("ambient payload claim_boundary must retain peak honesty")
    schema = raw.get("schema")
    if schema != QUANTUM_THERMO_SCHEMA:
        raise ValueError(
            f"ambient payload schema mismatch: expected {QUANTUM_THERMO_SCHEMA!r}, got {schema!r}"
        )
    return {
        "product_schema": THERMO_READINESS_PRODUCT_SCHEMA,
        "product_claim_boundary": THERMO_READINESS_CLAIM_BOUNDARY,
        "ambient_claim_boundary": ambient_boundary,
        "ambient_schema": QUANTUM_THERMO_SCHEMA,
        "hardware_submission_allowed": False,
        "thermodynamic_peak_claim_allowed": False,
        "payload": raw,
    }


def map_thermo_readiness_public_surfaces() -> tuple[dict[str, object], ...]:
    """Map public surfaces composing the thermo readiness product.

    Returns
    -------
    tuple[dict[str, object], ...]
        Surface descriptors with module paths and roles.

    """
    return (
        {
            "surface_id": "thermo_readiness_product",
            "module_path": "scpn_quantum_control.thermo_readiness_product",
            "role": "product_facade",
            "claim_boundary": THERMO_READINESS_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "thermodynamics_readiness",
            "module_path": "scpn_quantum_control.thermodynamics.readiness",
            "role": "ambient_s9_readiness",
            "claim_boundary": AMBIENT_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "fep_predictive_coding",
            "module_path": "scpn_quantum_control.fep.predictive_coding",
            "role": "fep_research_inventory",
            "claim_boundary": THERMO_READINESS_CLAIM_BOUNDARY,
        },
        {
            "surface_id": "fep_variational_free_energy",
            "module_path": "scpn_quantum_control.fep.variational_free_energy",
            "role": "fep_research_inventory",
            "claim_boundary": THERMO_READINESS_CLAIM_BOUNDARY,
        },
    )


def build_thermo_readiness_product_registry() -> dict[str, object]:
    """Build the versioned thermo readiness product registry payload.

    Returns
    -------
    dict[str, object]
        Schema v1 registry with capabilities, FEP inventory, and policy flags.

    """
    ambient_boundary = verify_ambient_claim_boundary()
    capabilities = [row.to_dict() for row in _CAPABILITIES]
    fep_rows = [row.to_dict() for row in _FEP_INVENTORY]
    return {
        "schema": THERMO_READINESS_PRODUCT_SCHEMA,
        "claim_boundary": THERMO_READINESS_CLAIM_BOUNDARY,
        "ambient_claim_boundary": ambient_boundary,
        "ambient_schema": QUANTUM_THERMO_SCHEMA,
        "capability_count": len(capabilities),
        "fep_inventory_count": len(fep_rows),
        "blank_entry_count": 0,
        "hardware_submission_allowed_policy": False,
        "thermodynamic_peak_claim_allowed_policy": False,
        "fep_product_promotion_allowed_policy": False,
        "public_surfaces": list(map_thermo_readiness_public_surfaces()),
        "capabilities": capabilities,
        "fep_inventory": fep_rows,
        "policy_note": (
            "Readiness estimate only via thermodynamics.readiness; FEP modules "
            "inventoried as research_only; no invent-green peak claim or hardware "
            "submit; future FEP sync-control hook design remains open honestly."
        ),
    }


def assert_thermo_readiness_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers capabilities/FEP without invent-green peak/QPU.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_thermo_readiness_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green policies appear.

    """
    registry = dict(payload) if payload is not None else build_thermo_readiness_product_registry()
    capabilities = registry.get("capabilities")
    fep_inventory = registry.get("fep_inventory")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError(
            "thermo readiness product registry must contain a non-empty capabilities list"
        )
    if not isinstance(fep_inventory, list) or not fep_inventory:
        raise ValueError(
            "thermo readiness product registry must contain a non-empty fep_inventory list"
        )
    seen: set[str] = set()
    blank = 0
    k_sweep_found = False
    for index, row in enumerate(capabilities):
        if not isinstance(row, Mapping):
            raise ValueError(f"capability row {index} must be a mapping")
        capability_id = row.get("capability_id")
        hw = row.get("hardware_submission_allowed")
        peak = row.get("thermodynamic_peak_claim_allowed")
        ambient_symbol = row.get("ambient_symbol")
        if not capability_id or not str(capability_id).strip():
            blank += 1
            continue
        cid = str(capability_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate capability_id in registry: {cid!r}")
        seen.add(cid)
        if cid == "k_sweep_protocol":
            k_sweep_found = True
        if hw is not False:
            raise ValueError(f"capability {cid!r} hardware_submission_allowed must be False")
        if peak is not False:
            raise ValueError(f"capability {cid!r} thermodynamic_peak_claim_allowed must be False")
        if not ambient_symbol or not str(ambient_symbol).strip():
            raise ValueError(f"capability {cid!r} must have non-empty ambient_symbol")
    if blank:
        raise ValueError(f"thermo readiness product registry has {blank} blank or invalid entries")
    if not k_sweep_found:
        raise ValueError("thermo readiness product registry missing k_sweep_protocol")
    expected = set(list_readiness_capability_ids())
    if seen != expected:
        raise ValueError(
            f"registry capability set drift (missing={expected - seen!r}, "
            f"extra={seen - expected!r})"
        )
    seen_fep: set[str] = set()
    for index, row in enumerate(fep_inventory):
        if not isinstance(row, Mapping):
            raise ValueError(f"FEP inventory row {index} must be a mapping")
        module_id = row.get("module_id")
        status = row.get("status")
        proven = row.get("product_hook_proven")
        if not module_id or not str(module_id).strip():
            raise ValueError(f"FEP inventory row {index} blank or invalid module_id")
        mid = str(module_id).strip()
        if mid in seen_fep:
            raise ValueError(f"duplicate module_id in FEP inventory: {mid!r}")
        seen_fep.add(mid)
        if status != "research_only":
            raise ValueError(f"FEP module {mid!r} status must be research_only on product surface")
        if proven is not False:
            raise ValueError(f"FEP module {mid!r} product_hook_proven must be False")
    expected_fep = set(list_fep_module_ids())
    if seen_fep != expected_fep:
        raise ValueError(
            f"registry FEP set drift (missing={expected_fep - seen_fep!r}, "
            f"extra={seen_fep - expected_fep!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    capability_count = registry.get("capability_count", -1)
    if not isinstance(capability_count, int) or capability_count != len(capabilities):
        raise ValueError("capability_count does not match capabilities list length")
    fep_count = registry.get("fep_inventory_count", -1)
    if not isinstance(fep_count, int) or fep_count != len(fep_inventory):
        raise ValueError("fep_inventory_count does not match fep_inventory list length")
    if registry.get("hardware_submission_allowed_policy", True) is not False:
        raise ValueError("hardware_submission_allowed_policy must be False")
    if registry.get("thermodynamic_peak_claim_allowed_policy", True) is not False:
        raise ValueError("thermodynamic_peak_claim_allowed_policy must be False")
    if registry.get("fep_product_promotion_allowed_policy", True) is not False:
        raise ValueError("fep_product_promotion_allowed_policy must be False")
    ambient_boundary = registry.get("ambient_claim_boundary")
    if not isinstance(ambient_boundary, str) or (
        _EXPECTED_AMBIENT_BOUNDARY_FRAGMENT not in ambient_boundary
    ):
        raise ValueError("registry ambient_claim_boundary must retain peak honesty")
    return registry


def compute_k_sweep_request_digest(
    *,
    k_values: Sequence[float],
    transition_k: float,
    capability_id: str = "k_sweep_protocol",
) -> str:
    """Compute a canonical digest for a K-sweep readiness request (anti-cheat).

    Parameters
    ----------
    k_values
        K grid values.
    transition_k
        Transition centre K.
    capability_id
        Capability identifier.

    Returns
    -------
    str
        Hex SHA-256 digest.

    Raises
    ------
    ValueError
        If inputs are empty/invalid.

    """
    if not capability_id or not str(capability_id).strip():
        raise ValueError("capability_id must be non-empty")
    if not k_values or len(tuple(k_values)) < 3:
        raise ValueError("k_values must contain at least three values")
    values = [float(item) for item in k_values]
    if any(not math.isfinite(item) for item in values):
        raise ValueError("k_values must be finite")
    if not math.isfinite(float(transition_k)):
        raise ValueError("transition_k must be finite")
    if sorted(values) != values or len(set(values)) != len(values):
        raise ValueError("k_values must be strictly increasing")
    if float(transition_k) not in values:
        raise ValueError("transition_k must be one of k_values")
    get_readiness_capability(capability_id)
    payload = {
        "schema": "thermo_readiness_k_sweep_request.v1",
        "capability_id": str(capability_id).strip(),
        "k_values": values,
        "transition_k": float(transition_k),
        "product_schema": THERMO_READINESS_PRODUCT_SCHEMA,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AMBIENT_CLAIM_BOUNDARY",
    "FepInventoryRow",
    "FepInventoryStatus",
    "MaterialisedKSweepProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "ReadinessCapabilityKind",
    "ReadinessCapabilityRow",
    "SupportPosture",
    "THERMO_READINESS_CLAIM_BOUNDARY",
    "THERMO_READINESS_PRODUCT_SCHEMA",
    "assert_thermo_readiness_product_integrity",
    "build_thermo_readiness_product_registry",
    "compute_k_sweep_request_digest",
    "decide_readiness_path",
    "get_fep_inventory_row",
    "get_readiness_capability",
    "iter_fep_inventory",
    "iter_readiness_capabilities",
    "list_fep_module_ids",
    "list_readiness_capability_ids",
    "map_thermo_readiness_public_surfaces",
    "materialise_demo_k_sweep_probe",
    "materialise_k_sweep_probe",
    "materialise_quantum_thermo_payload_probe",
    "verify_ambient_claim_boundary",
]
