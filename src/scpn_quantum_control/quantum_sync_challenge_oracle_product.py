# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Sync Challenge oracle product (BL-32 / P1)
"""Fail-closed **Quantum Sync Challenge oracle** product surface (BL-32).

Productises a claim-governed synchronisation challenge oracle façade over ambient
sync witnesses, objectives, and coupling recovery — not a second solver stack:

* versioned problem-family catalogue F1–F4 (synthetic first; hardware schema-only);
* metric and baseline catalogue rows with support badges (BL-52 compose);
* materialised oracle probe via ambient
  :func:`~scpn_quantum_control.phase.synchronisation_witness.run_sync_witness_suite`
  and order-parameter metric on deterministic fixtures;
* anti-cheat digests (family + seed + schema) and fail-closed path decisions;
* refuse invent-green quantum advantage claims, live hardware without ticket,
  and ranking unvalidated submissions.

Does **not** ship full ``challenge/`` package depth (S32.4–S32.12 residual),
live leaderboard SaaS, or remote code execution for submissions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from .phase.synchronisation_objectives import kuramoto_order_parameter
from .phase.synchronisation_witness import (
    SYNC_WITNESS_CLAIM_BOUNDARY,
    run_sync_witness_suite,
)

FamilySupportStatus = Literal[
    "synthetic_deterministic",
    "noisy_sim",
    "hardware_gated",
]
"""Support status for challenge problem families."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges for oracle rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

BaselineKind = Literal[
    "classical_numpy",
    "quantum_simulator",
    "hardware_schema_only",
]
"""Baseline kinds on the product catalogue."""

MetricKind = Literal[
    "order_parameter",
    "witness_suite",
    "coupling_recovery",
    "gradient_certificate",
    "convergence_certificate",
]
"""Metric kinds exposed by the oracle product."""

QUANTUM_SYNC_CHALLENGE_ORACLE_PRODUCT_SCHEMA: Final[str] = (
    "quantum_sync_challenge_oracle_product.v1"
)
"""JSON schema identifier for serialised product payloads."""

QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY: Final[str] = (
    "Quantum Sync Challenge oracle product surface only; catalogues synthetic "
    "problem families F1–F4, metrics, and baselines with fail-closed support "
    "status; materialises ambient synchronisation_witness suite probes and "
    "order-parameter fixtures; refuse invent-green quantum advantage, live "
    "hardware without owner ticket, and ranking unvalidated submissions; does "
    "not claim full challenge package depth or live leaderboard SaaS "
    "(S32.4–S32.12 residual)"
)
"""Shared claim boundary for sync challenge oracle product payloads."""


@dataclass(frozen=True, slots=True)
class ProblemFamilyRow:
    """One challenge problem family catalogue row (S32.0 / S32.2).

    Attributes
    ----------
    family_id
        Stable family identifier (F1–F4).
    title
        Human-readable title.
    summary
        Short description.
    support_status
        Synthetic / noisy_sim / hardware_gated.
    default_seed
        Deterministic generator seed for fixtures.
    n_nodes
        Default node count for smoke fixtures.
    ambient_pointer
        Ambient module pointer for composition.
    bl52_route_pointer
        BL-52 governed-route matrix pointer.
    bl53_pointer
        BL-53 unsuitable / anti-silent-wrong pointer.
    invent_green_advantage
        Must remain False.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.
    """

    family_id: str
    title: str
    summary: str
    support_status: FamilySupportStatus
    default_seed: int
    n_nodes: int
    ambient_pointer: str
    bl52_route_pointer: str
    bl53_pointer: str
    invent_green_advantage: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate problem family invariants."""
        if not self.family_id or not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.support_status not in {
            "synthetic_deterministic",
            "noisy_sim",
            "hardware_gated",
        }:
            raise ValueError(f"unknown support_status: {self.support_status!r}")
        if self.default_seed < 0:
            raise ValueError("default_seed must be non-negative")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be >= 2")
        if not self.ambient_pointer or not self.ambient_pointer.strip():
            raise ValueError("ambient_pointer must be non-empty")
        if not self.bl52_route_pointer or not self.bl52_route_pointer.strip():
            raise ValueError("bl52_route_pointer must be non-empty")
        if not self.bl53_pointer or not self.bl53_pointer.strip():
            raise ValueError("bl53_pointer must be non-empty")
        if self.invent_green_advantage:
            raise ValueError("invent_green_advantage must be False")
        if self.support_status == "hardware_gated" and self.support_posture != (
            "live_hardware_gated"
        ):
            raise ValueError(
                "hardware_gated families must use support_posture=live_hardware_gated"
            )
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
            "family_id": self.family_id,
            "title": self.title,
            "summary": self.summary,
            "support_status": self.support_status,
            "default_seed": self.default_seed,
            "n_nodes": self.n_nodes,
            "ambient_pointer": self.ambient_pointer,
            "bl52_route_pointer": self.bl52_route_pointer,
            "bl53_pointer": self.bl53_pointer,
            "invent_green_advantage": self.invent_green_advantage,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MetricCatalogueRow:
    """One oracle metric catalogue row (S32.3).

    Attributes
    ----------
    metric_id
        Stable metric identifier.
    kind
        Metric kind enum.
    title
        Human-readable title.
    ambient_pointer
        Ambient implementation pointer.
    required_for_leaderboard
        Whether validation requires this metric for ranking.
    support_posture
        Support posture badge.
    claim_boundary
        Non-promotional claim boundary.
    """

    metric_id: str
    kind: MetricKind
    title: str
    ambient_pointer: str
    required_for_leaderboard: bool
    support_posture: SupportPosture = "local_research"
    claim_boundary: str = QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate metric row invariants."""
        if not self.metric_id or not self.metric_id.strip():
            raise ValueError("metric_id must be non-empty")
        if self.kind not in {
            "order_parameter",
            "witness_suite",
            "coupling_recovery",
            "gradient_certificate",
            "convergence_certificate",
        }:
            raise ValueError(f"unknown metric kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.ambient_pointer or not self.ambient_pointer.strip():
            raise ValueError("ambient_pointer must be non-empty")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "metadata_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "metric_id": self.metric_id,
            "kind": self.kind,
            "title": self.title,
            "ambient_pointer": self.ambient_pointer,
            "required_for_leaderboard": self.required_for_leaderboard,
            "support_posture": self.support_posture,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class BaselineCatalogueRow:
    """One challenge baseline catalogue row (S32.4 / S32.5 / S32.10).

    Attributes
    ----------
    baseline_id
        Stable baseline identifier.
    kind
        Baseline kind enum.
    title
        Human-readable title.
    no_submit
        Always True on product surface for non-ticketed paths.
    owner_ticket_required
        Whether live execution requires owner ticket.
    support_posture
        Support posture badge.
    claim_boundary
        Non-promotional claim boundary.
    """

    baseline_id: str
    kind: BaselineKind
    title: str
    no_submit: bool = True
    owner_ticket_required: bool = False
    support_posture: SupportPosture = "local_research"
    claim_boundary: str = QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate baseline row invariants."""
        if not self.baseline_id or not self.baseline_id.strip():
            raise ValueError("baseline_id must be non-empty")
        if self.kind not in {
            "classical_numpy",
            "quantum_simulator",
            "hardware_schema_only",
        }:
            raise ValueError(f"unknown baseline kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if self.no_submit is not True:
            raise ValueError("no_submit must be True on product baseline catalogue")
        if self.kind == "hardware_schema_only" and not self.owner_ticket_required:
            raise ValueError("hardware_schema_only baselines must set owner_ticket_required=True")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "metadata_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "baseline_id": self.baseline_id,
            "kind": self.kind,
            "title": self.title,
            "no_submit": self.no_submit,
            "owner_ticket_required": self.owner_ticket_required,
            "support_posture": self.support_posture,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for challenge oracle product use.

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
    claim_boundary: str = QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY

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
class MaterialisedOracleProbe:
    """Materialised oracle probe from ambient witness suite + order parameter.

    Attributes
    ----------
    family_id
        Family used for fixture digest context.
    instance_digest
        Anti-cheat digest of family/seed/schema.
    witness_case_count
        Ambient suite record count.
    witness_all_passed
        Whether all ambient witness records passed.
    order_parameter
        Kuramoto order parameter on synchronised fixture phases.
    invent_green_advantage
        Always False.
    invent_green_hardware
        Always False.
    ambient_witness_claim_boundary
        Ambient SYNC_WITNESS_CLAIM_BOUNDARY excerpt pointer.
    demo_label
        Demo fixture label.
    claim_boundary
        Product claim boundary.
    """

    family_id: str
    instance_digest: str
    witness_case_count: int
    witness_all_passed: bool
    order_parameter: float
    invent_green_advantage: bool
    invent_green_hardware: bool
    ambient_witness_claim_boundary: str
    demo_label: str
    claim_boundary: str = QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate oracle probe invariants."""
        if not self.family_id or not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not self.instance_digest or not self.instance_digest.strip():
            raise ValueError("instance_digest must be non-empty")
        if self.witness_case_count < 1:
            raise ValueError("witness_case_count must be positive")
        if not 0.0 <= self.order_parameter <= 1.0 + 1e-9:
            raise ValueError("order_parameter must be in [0, 1]")
        if self.invent_green_advantage:
            raise ValueError("invent_green_advantage must be False")
        if self.invent_green_hardware:
            raise ValueError("invent_green_hardware must be False")
        if not self.ambient_witness_claim_boundary or not (
            self.ambient_witness_claim_boundary.strip()
        ):
            raise ValueError("ambient_witness_claim_boundary must be non-empty")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "family_id": self.family_id,
            "instance_digest": self.instance_digest,
            "witness_case_count": self.witness_case_count,
            "witness_all_passed": self.witness_all_passed,
            "order_parameter": self.order_parameter,
            "invent_green_advantage": self.invent_green_advantage,
            "invent_green_hardware": self.invent_green_hardware,
            "ambient_witness_claim_boundary": self.ambient_witness_claim_boundary,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_problem_families() -> tuple[ProblemFamilyRow, ...]:
    """Build F1–F4 problem family catalogue (S32.2)."""
    return (
        ProblemFamilyRow(
            family_id="F1_all_to_all_kuramoto",
            title="All-to-all Kuramoto synthetic",
            summary="Dense mean-field Kuramoto synthetic; deterministic smoke fixtures.",
            support_status="synthetic_deterministic",
            default_seed=3201,
            n_nodes=8,
            ambient_pointer=(
                "scpn_quantum_control.phase.synchronisation_objectives.kuramoto_order_parameter"
            ),
            bl52_route_pointer="governed_route:challenge.F1.synthetic",
            bl53_pointer="unsuitable_scenario_registry.challenge_f1",
            support_posture="local_research",
        ),
        ProblemFamilyRow(
            family_id="F2_sparse_ring_xy",
            title="Sparse ring / XY synthetic",
            summary="Ring topology XY / sparse graph synthetic family.",
            support_status="synthetic_deterministic",
            default_seed=3202,
            n_nodes=8,
            ambient_pointer=("scpn_quantum_control.phase.coupling_time_series_recovery"),
            bl52_route_pointer="governed_route:challenge.F2.synthetic",
            bl53_pointer="unsuitable_scenario_registry.challenge_f2",
            support_posture="local_research",
        ),
        ProblemFamilyRow(
            family_id="F3_cluster_sync",
            title="Cluster synchronisation synthetic",
            summary="Multi-cluster phase clouds; ambient witness clustered regime.",
            support_status="synthetic_deterministic",
            default_seed=3203,
            n_nodes=9,
            ambient_pointer=(
                "scpn_quantum_control.phase.synchronisation_witness.run_sync_witness_suite"
            ),
            bl52_route_pointer="governed_route:challenge.F3.synthetic",
            bl53_pointer="unsuitable_scenario_registry.challenge_f3",
            support_posture="local_research",
        ),
        ProblemFamilyRow(
            family_id="F4_noisy_finite_shot",
            title="Noisy finite-shot synthetic",
            summary=(
                "Noisy sim family schema; uncertainty mandatory; no invent-green hardware shots."
            ),
            support_status="noisy_sim",
            default_seed=3204,
            n_nodes=6,
            ambient_pointer="scpn_quantum_control.phase.synchronisation_witness",
            bl52_route_pointer="governed_route:challenge.F4.noisy_sim",
            bl53_pointer="unsuitable_scenario_registry.challenge_f4",
            support_posture="local_research",
        ),
        ProblemFamilyRow(
            family_id="FH_hardware_gated",
            title="Hardware-gated family (schema only)",
            summary=(
                "Hardware family schema row with empty execution; owner ticket "
                "required (S32.10 residual)."
            ),
            support_status="hardware_gated",
            default_seed=3299,
            n_nodes=4,
            ambient_pointer="BL-47 hardware_safe_execution (no_submit default)",
            bl52_route_pointer="governed_route:challenge.FH.hardware_gated",
            bl53_pointer="unsuitable_scenario_registry.challenge_fh",
            support_posture="live_hardware_gated",
        ),
    )


def _build_metrics() -> tuple[MetricCatalogueRow, ...]:
    """Build metric catalogue (S32.3)."""
    return (
        MetricCatalogueRow(
            metric_id="order_parameter_r1",
            kind="order_parameter",
            title="Kuramoto order parameter r₁",
            ambient_pointer=(
                "scpn_quantum_control.phase.synchronisation_objectives.kuramoto_order_parameter"
            ),
            required_for_leaderboard=True,
        ),
        MetricCatalogueRow(
            metric_id="witness_suite",
            kind="witness_suite",
            title="Sync witness suite certificates",
            ambient_pointer=(
                "scpn_quantum_control.phase.synchronisation_witness.run_sync_witness_suite"
            ),
            required_for_leaderboard=True,
        ),
        MetricCatalogueRow(
            metric_id="coupling_recovery",
            kind="coupling_recovery",
            title="Coupling recovery error metrics",
            ambient_pointer=("scpn_quantum_control.phase.coupling_time_series_recovery"),
            required_for_leaderboard=False,
        ),
        MetricCatalogueRow(
            metric_id="gradient_certificate",
            kind="gradient_certificate",
            title="Gradient correctness certificate wrapper",
            ambient_pointer="ambient FD/analytic gates (compose residual)",
            required_for_leaderboard=False,
            support_posture="metadata_only",
        ),
        MetricCatalogueRow(
            metric_id="convergence_certificate",
            kind="convergence_certificate",
            title="Optimiser convergence certificate wrapper",
            ambient_pointer=("scpn_quantum_control.phase.optimizer_convergence_suite"),
            required_for_leaderboard=False,
            support_posture="metadata_only",
        ),
    )


def _build_baselines() -> tuple[BaselineCatalogueRow, ...]:
    """Build baseline catalogue (S32.4 / S32.5 / S32.10)."""
    return (
        BaselineCatalogueRow(
            baseline_id="classical_numpy_cpu",
            kind="classical_numpy",
            title="Classical NumPy/SciPy CPU baseline",
            no_submit=True,
            owner_ticket_required=False,
            support_posture="local_research",
        ),
        BaselineCatalogueRow(
            baseline_id="quantum_simulator_phase_qnode",
            kind="quantum_simulator",
            title="Quantum simulator phase-QNode / program-AD path",
            no_submit=True,
            owner_ticket_required=False,
            support_posture="local_research",
        ),
        BaselineCatalogueRow(
            baseline_id="hardware_schema_only",
            kind="hardware_schema_only",
            title="Hardware row schema only (no execution)",
            no_submit=True,
            owner_ticket_required=True,
            support_posture="live_hardware_gated",
        ),
    )


_FAMILIES: Final[tuple[ProblemFamilyRow, ...]] = _build_problem_families()
_METRICS: Final[tuple[MetricCatalogueRow, ...]] = _build_metrics()
_BASELINES: Final[tuple[BaselineCatalogueRow, ...]] = _build_baselines()


def _family_map() -> dict[str, ProblemFamilyRow]:
    """Return family_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, ProblemFamilyRow] = {}
    for row in _FAMILIES:
        key = row.family_id.strip()
        if not key:
            raise RuntimeError("problem family catalogue contains blank family_id")
        if key in mapping:
            raise RuntimeError(f"duplicate family_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("problem family catalogue must be non-empty")
    return mapping


_FAMILY_BY_ID: Final[Mapping[str, ProblemFamilyRow]] = _family_map()


def list_problem_family_ids() -> tuple[str, ...]:
    """Return all problem family identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable family ids.
    """
    return tuple(row.family_id for row in _FAMILIES)


def list_metric_ids() -> tuple[str, ...]:
    """Return all metric identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable metric ids.
    """
    return tuple(row.metric_id for row in _METRICS)


def list_baseline_ids() -> tuple[str, ...]:
    """Return all baseline identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable baseline ids.
    """
    return tuple(row.baseline_id for row in _BASELINES)


def get_problem_family(family_id: str) -> ProblemFamilyRow:
    """Return one problem family row; fail closed on blank/unknown.

    Parameters
    ----------
    family_id
        Family identifier.

    Returns
    -------
    ProblemFamilyRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.
    """
    if not family_id or not str(family_id).strip():
        raise ValueError("family_id must be non-empty")
    key = str(family_id).strip()
    try:
        return _FAMILY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown family_id: {key!r}") from exc


def iter_problem_families(
    *,
    support_status: FamilySupportStatus | None = None,
) -> tuple[ProblemFamilyRow, ...]:
    """Return filtered problem family rows in stable order.

    Parameters
    ----------
    support_status
        Optional support status filter.

    Returns
    -------
    tuple[ProblemFamilyRow, ...]
        Matching rows.
    """
    rows: Sequence[ProblemFamilyRow] = _FAMILIES
    if support_status is not None:
        rows = tuple(row for row in rows if row.support_status == support_status)
    return tuple(rows)


def compute_instance_digest(
    family_id: str,
    *,
    seed: int | None = None,
    schema_version: str = "challenge_instance.v1",
) -> str:
    """Compute anti-cheat content digest for a problem instance (H1).

    Parameters
    ----------
    family_id
        Known family id.
    seed
        Optional seed override (defaults to family default_seed).
    schema_version
        Instance schema version label.

    Returns
    -------
    str
        Hex SHA-256 digest of canonical JSON payload.

    Raises
    ------
    ValueError
        If family unknown or seed negative.
    """
    row = get_problem_family(family_id)
    use_seed = row.default_seed if seed is None else seed
    if use_seed < 0:
        raise ValueError("seed must be non-negative")
    if not schema_version or not schema_version.strip():
        raise ValueError("schema_version must be non-empty")
    payload = {
        "schema_version": schema_version.strip(),
        "family_id": row.family_id,
        "seed": use_seed,
        "n_nodes": row.n_nodes,
        "support_status": row.support_status,
        "product_schema": QUANTUM_SYNC_CHALLENGE_ORACLE_PRODUCT_SCHEMA,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def decide_challenge_path(
    family_id: str,
    *,
    request_leaderboard_rank: bool = False,
    submission_validated: bool = False,
    invent_green_advantage: bool = False,
    owner_ticket_present: bool = False,
    request_hardware_execution: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a challenge oracle path may proceed (S32.6 / S32.7).

    Parameters
    ----------
    family_id
        Problem family identifier.
    request_leaderboard_rank
        Whether the caller requests ranking on the leaderboard.
    submission_validated
        Whether anti-cheat / schema validation passed.
    invent_green_advantage
        If true, refuse.
    owner_ticket_present
        Required for hardware execution.
    request_hardware_execution
        Whether live hardware execution is requested.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.
    """
    row = get_problem_family(family_id)
    blockers: list[str] = []
    if invent_green_advantage:
        blockers.append(
            f"invent-green quantum advantage claim refused (family={row.family_id}; BL-32 honesty)"
        )
    if request_leaderboard_rank and not submission_validated:
        blockers.append(
            "leaderboard ranking refuses unvalidated submissions "
            f"(family={row.family_id}; S32.7 scoring purity)"
        )
    if request_hardware_execution:
        if row.support_status != "hardware_gated":
            blockers.append(
                f"family {row.family_id!r} is not hardware_gated "
                f"(support_status={row.support_status})"
            )
        if not owner_ticket_present:
            blockers.append(
                "owner ticket required for hardware execution "
                f"(family={row.family_id}; BL-47 no_submit default)"
            )
        # Product surface never auto-executes hardware even with ticket.
        blockers.append(
            "hardware execution residual on product surface (schema-only FH row; S32.10)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="challenge path refused under fail-closed oracle product policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"challenge path allowed for family {row.family_id!r} "
            f"(support_status={row.support_status}; invent_green_advantage=False)"
        ),
        blockers=(),
    )


def materialise_oracle_probe(
    family_id: str = "F1_all_to_all_kuramoto",
) -> MaterialisedOracleProbe:
    """Materialise oracle probe via ambient witness suite + order parameter.

    Parameters
    ----------
    family_id
        Known synthetic family for digest context.

    Returns
    -------
    MaterialisedOracleProbe
        Finite primary observables with invent-green flags False.

    Raises
    ------
    ValueError
        If family unknown or ambient suite empty.
    """
    row = get_problem_family(family_id)
    if row.support_status == "hardware_gated":
        raise ValueError(
            "cannot materialise execution probe for hardware_gated family "
            "(schema-only; use synthetic families)"
        )
    digest = compute_instance_digest(family_id)
    suite = run_sync_witness_suite()
    records = suite.records
    if not records:
        raise RuntimeError("ambient sync witness suite returned no records")
    all_passed = all(bool(rec.passed) for rec in records)
    # Deterministic synchronised fixture phases (n=8) for order-parameter metric.
    phases = np.array(
        [0.01, -0.01, 0.02, -0.02, 0.0, 0.015, -0.015, 0.005],
        dtype=np.float64,
    )
    order = float(kuramoto_order_parameter(phases))
    return MaterialisedOracleProbe(
        family_id=row.family_id,
        instance_digest=digest,
        witness_case_count=len(records),
        witness_all_passed=all_passed,
        order_parameter=order,
        invent_green_advantage=False,
        invent_green_hardware=False,
        ambient_witness_claim_boundary=str(SYNC_WITNESS_CLAIM_BOUNDARY),
        demo_label="ambient_sync_witness_suite_and_order_parameter",
    )


def materialise_demo_oracle_probe() -> MaterialisedOracleProbe:
    """Materialise the deterministic F1 demo oracle probe.

    Returns
    -------
    MaterialisedOracleProbe
        Ambient composition probe.
    """
    return materialise_oracle_probe("F1_all_to_all_kuramoto")


def map_quantum_sync_challenge_oracle_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of challenge oracle product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.
    """
    return (
        {
            "module_path": ("scpn_quantum_control.quantum_sync_challenge_oracle_product"),
            "role": "quantum_sync_challenge_oracle_product_surface",
            "support_posture": "local_research",
            "family_ids": list(list_problem_family_ids()),
            "metric_ids": list(list_metric_ids()),
            "baseline_ids": list(list_baseline_ids()),
            "invent_green_advantage": False,
            "claim_boundary": QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.phase.synchronisation_witness",
            "role": "ambient_sync_witness_suite",
            "support_posture": "local_research",
            "symbol_name": "run_sync_witness_suite",
            "claim_boundary": QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.phase.synchronisation_objectives",
            "role": "ambient_order_parameter_metric",
            "support_posture": "local_research",
            "symbol_name": "kuramoto_order_parameter",
            "claim_boundary": QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.phase.coupling_time_series_recovery",
            "role": "ambient_coupling_recovery_compose",
            "support_posture": "local_research",
            "symbol_name": "default_coupling_recovery_cases",
            "claim_boundary": QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
        },
    )


def build_quantum_sync_challenge_oracle_product_registry() -> dict[str, object]:
    """Build the full serialisable challenge oracle product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with families/metrics/baselines (no blanks).
    """
    families = [row.to_dict() for row in _FAMILIES]
    metrics = [row.to_dict() for row in _METRICS]
    baselines = [row.to_dict() for row in _BASELINES]
    return {
        "schema": QUANTUM_SYNC_CHALLENGE_ORACLE_PRODUCT_SCHEMA,
        "claim_boundary": QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY,
        "family_count": len(families),
        "metric_count": len(metrics),
        "baseline_count": len(baselines),
        "blank_entry_count": 0,
        "invent_green_advantage_policy": False,
        "invent_green_hardware_policy": False,
        "public_surfaces": list(map_quantum_sync_challenge_oracle_public_surfaces()),
        "families": families,
        "metrics": metrics,
        "baselines": baselines,
        "policy_note": (
            "Challenge oracle product façade only; ambient witnesses/objectives/"
            "coupling recovery remain the spine; no live leaderboard SaaS; "
            "S32.4–S32.12 residual full challenge package depth open honestly."
        ),
    }


def assert_quantum_sync_challenge_oracle_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers families/metrics without invent-green advantage.

    Parameters
    ----------
    payload
        Optional payload from
        :func:`build_quantum_sync_challenge_oracle_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green policies appear.
    """
    registry = (
        dict(payload)
        if payload is not None
        else build_quantum_sync_challenge_oracle_product_registry()
    )
    families = registry.get("families")
    metrics = registry.get("metrics")
    baselines = registry.get("baselines")
    if not isinstance(families, list) or not families:
        raise ValueError(
            "challenge oracle product registry must contain a non-empty families list"
        )
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("challenge oracle product registry must contain a non-empty metrics list")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError(
            "challenge oracle product registry must contain a non-empty baselines list"
        )
    seen: set[str] = set()
    blank = 0
    f1_found = False
    hw_found = False
    for index, row in enumerate(families):
        if not isinstance(row, Mapping):
            raise ValueError(f"family row {index} must be a mapping")
        family_id = row.get("family_id")
        invent = row.get("invent_green_advantage")
        support_status = row.get("support_status")
        bl52 = row.get("bl52_route_pointer")
        if not family_id or not str(family_id).strip():
            blank += 1
            continue
        fid = str(family_id).strip()
        if fid in seen:
            raise ValueError(f"duplicate family_id in registry: {fid!r}")
        seen.add(fid)
        if fid == "F1_all_to_all_kuramoto":
            f1_found = True
        if support_status == "hardware_gated":
            hw_found = True
        if invent is not False:
            raise ValueError(f"family {fid!r} invent_green_advantage must be False")
        if not bl52 or not str(bl52).strip():
            raise ValueError(f"family {fid!r} must have bl52_route_pointer")
        if support_status not in {
            "synthetic_deterministic",
            "noisy_sim",
            "hardware_gated",
        }:
            raise ValueError(f"family {fid!r} has unknown support_status: {support_status!r}")
    if blank:
        raise ValueError(f"challenge oracle product registry has {blank} blank or invalid entries")
    if not f1_found:
        raise ValueError("challenge oracle product registry missing F1_all_to_all_kuramoto")
    if not hw_found:
        raise ValueError("challenge oracle product registry missing hardware_gated family row")
    expected = set(list_problem_family_ids())
    if seen != expected:
        raise ValueError(
            f"registry family set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    seen_metrics: set[str] = set()
    for index, row in enumerate(metrics):
        if not isinstance(row, Mapping):
            raise ValueError(f"metric row {index} must be a mapping")
        metric_id = row.get("metric_id")
        if not metric_id or not str(metric_id).strip():
            raise ValueError(f"metric row {index} blank or invalid metric_id")
        mid = str(metric_id).strip()
        if mid in seen_metrics:
            raise ValueError(f"duplicate metric_id in registry: {mid!r}")
        seen_metrics.add(mid)
    expected_metrics = set(list_metric_ids())
    if seen_metrics != expected_metrics:
        raise ValueError(
            f"registry metric set drift (missing={expected_metrics - seen_metrics!r}, "
            f"extra={seen_metrics - expected_metrics!r})"
        )
    for index, row in enumerate(baselines):
        if not isinstance(row, Mapping):
            raise ValueError(f"baseline row {index} must be a mapping")
        no_submit = row.get("no_submit")
        if no_submit is not True:
            raise ValueError(f"baseline row {index} no_submit must be True")
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    family_count = registry.get("family_count", -1)
    if not isinstance(family_count, int) or family_count != len(families):
        raise ValueError("family_count does not match families list length")
    metric_count = registry.get("metric_count", -1)
    if not isinstance(metric_count, int) or metric_count != len(metrics):
        raise ValueError("metric_count does not match metrics list length")
    baseline_count = registry.get("baseline_count", -1)
    if not isinstance(baseline_count, int) or baseline_count != len(baselines):
        raise ValueError("baseline_count does not match baselines list length")
    invent_adv = registry.get("invent_green_advantage_policy", True)
    if invent_adv is not False:
        raise ValueError("invent_green_advantage_policy must be False")
    invent_hw = registry.get("invent_green_hardware_policy", True)
    if invent_hw is not False:
        raise ValueError("invent_green_hardware_policy must be False")
    return registry


__all__ = [
    "QUANTUM_SYNC_CHALLENGE_ORACLE_CLAIM_BOUNDARY",
    "QUANTUM_SYNC_CHALLENGE_ORACLE_PRODUCT_SCHEMA",
    "BaselineCatalogueRow",
    "BaselineKind",
    "FamilySupportStatus",
    "MaterialisedOracleProbe",
    "MetricCatalogueRow",
    "MetricKind",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "ProblemFamilyRow",
    "SupportPosture",
    "assert_quantum_sync_challenge_oracle_product_integrity",
    "build_quantum_sync_challenge_oracle_product_registry",
    "compute_instance_digest",
    "decide_challenge_path",
    "get_problem_family",
    "iter_problem_families",
    "list_baseline_ids",
    "list_metric_ids",
    "list_problem_family_ids",
    "map_quantum_sync_challenge_oracle_public_surfaces",
    "materialise_demo_oracle_probe",
    "materialise_oracle_probe",
]
