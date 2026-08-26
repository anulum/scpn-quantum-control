# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Campaign harness productisation
"""Fail-closed **campaign harness productisation** surface.

Productises reusable hardware-campaign harness templates with prereg hooks,
digests, and hardware-safety / advantage-language integration over ambient campaign modules:

* versioned harness catalogue (AppQSim, IQM layout-transfer, closed-loop
  publication, ambient ``benchmark_harness`` registry);
* no-submit default and owner-ticket gates for live hardware paths;
* dry-run materialised probes via ambient
  :func:`~scpn_quantum_control.benchmarks.appqsim_protocol.appqsim_benchmark`,
  :func:`~scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark.build_layout_transfer_plan`,
  and
  :func:`~scpn_quantum_control.benchmarks.closed_loop_publication_run.run_closed_loop_publication`;
* content digests for config / synthetic calibration payloads;
* refuse invent-green live QPU submit, unattested claim promotion, and
  post-hoc prereg mutation.

Does **not** complete full reproduction-kit hermetic kit export or attested-result attestation
sealing; both integrations remain explicit residual work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from .benchmark_harness.registry import list_benchmark_families
from .benchmarks.appqsim_protocol import appqsim_benchmark
from .benchmarks.closed_loop_publication_run import (
    ClosedLoopRunConfig,
    run_closed_loop_publication,
)
from .benchmarks.iqm_layout_transfer_benchmark import build_layout_transfer_plan
from .hardware.iqm_lattice_calibration import LatticeCalibration

HarnessKind = Literal[
    "appqsim_protocol",
    "iqm_layout_transfer",
    "closed_loop_publication",
    "benchmark_harness_registry",
]
"""Productised campaign-harness kind vocabulary."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges for campaign harness rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

CAMPAIGN_HARNESS_PRODUCT_SCHEMA: Final[str] = "campaign_harness_product.v1"
"""JSON schema identifier for serialised product payloads."""

CAMPAIGN_HARNESS_CLAIM_BOUNDARY: Final[str] = (
    "Campaign harness productisation surface only; catalogues reusable AppQSim, "
    "IQM layout-transfer, closed-loop publication, and ambient benchmark_harness "
    "registry templates with preregistration digests and a no-submit default; "
    "dry-run probes only; refuse unsupported live QPU submission and unattested "
    "claim promotion; hermetic reproduction and attestation slots remain open"
)
"""Shared claim boundary for campaign harness product payloads."""


@dataclass(frozen=True, slots=True)
class CampaignHarnessRow:
    """One campaign-harness catalogue row.

    Attributes
    ----------
    harness_id
        Stable harness identifier.
    kind
        Harness kind enum.
    title
        Human-readable title.
    summary
        Short description.
    ambient_pointer
        Ambient module / entrypoint pointer.
    hardware_safety_pointer
        Hardware-safe no-submit policy pointer.
    advantage_protocol_pointer
        Advantage-language policy pointer.
    no_submit_default
        Always True on product surface.
    owner_ticket_required_for_live
        Whether live hardware requires owner ticket.
    invent_green_live_submit
        Must remain False.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    harness_id: str
    kind: HarnessKind
    title: str
    summary: str
    ambient_pointer: str
    hardware_safety_pointer: str
    advantage_protocol_pointer: str
    no_submit_default: bool = True
    owner_ticket_required_for_live: bool = True
    invent_green_live_submit: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = CAMPAIGN_HARNESS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate campaign harness row invariants."""
        if not self.harness_id or not self.harness_id.strip():
            raise ValueError("harness_id must be non-empty")
        if self.kind not in {
            "appqsim_protocol",
            "iqm_layout_transfer",
            "closed_loop_publication",
            "benchmark_harness_registry",
        }:
            raise ValueError(f"unknown harness kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.ambient_pointer or not self.ambient_pointer.strip():
            raise ValueError("ambient_pointer must be non-empty")
        if not self.hardware_safety_pointer or not self.hardware_safety_pointer.strip():
            raise ValueError("hardware_safety_pointer must be non-empty")
        if not self.advantage_protocol_pointer or not self.advantage_protocol_pointer.strip():
            raise ValueError("advantage_protocol_pointer must be non-empty")
        if self.no_submit_default is not True:
            raise ValueError("no_submit_default must be True on product surface")
        if self.invent_green_live_submit:
            raise ValueError("invent_green_live_submit must be False")
        if self.kind in {"iqm_layout_transfer", "closed_loop_publication"} and (
            not self.owner_ticket_required_for_live
        ):
            raise ValueError(
                "live-capable harness kinds must set owner_ticket_required_for_live=True"
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
            "harness_id": self.harness_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "ambient_pointer": self.ambient_pointer,
            "hardware_safety_pointer": self.hardware_safety_pointer,
            "advantage_protocol_pointer": self.advantage_protocol_pointer,
            "no_submit_default": self.no_submit_default,
            "owner_ticket_required_for_live": self.owner_ticket_required_for_live,
            "invent_green_live_submit": self.invent_green_live_submit,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for campaign harness product use.

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
    claim_boundary: str = CAMPAIGN_HARNESS_CLAIM_BOUNDARY

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
class MaterialisedCampaignProbe:
    """Materialised dry-run campaign probe.

    Attributes
    ----------
    harness_id
        Harness exercised.
    probe_kind
        Kind of dry-run probe.
    config_digest
        Anti-cheat digest of probe configuration.
    primary_metric
        Primary finite observable label.
    primary_value
        Primary finite observable value.
    no_submit
        Always True for product probes.
    invent_green_live_submit
        Always False.
    attestation_slot_present
        False until attestation integration lands.
    hermetic_kit_slot_present
        False until hermetic reproduction integration lands.
    demo_label
        Demo fixture label.
    claim_boundary
        Non-promotional claim boundary.

    """

    harness_id: str
    probe_kind: str
    config_digest: str
    primary_metric: str
    primary_value: float
    no_submit: bool
    invent_green_live_submit: bool
    attestation_slot_present: bool
    hermetic_kit_slot_present: bool
    demo_label: str
    claim_boundary: str = CAMPAIGN_HARNESS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate campaign probe invariants."""
        if not self.harness_id or not self.harness_id.strip():
            raise ValueError("harness_id must be non-empty")
        if not self.probe_kind or not self.probe_kind.strip():
            raise ValueError("probe_kind must be non-empty")
        if not self.config_digest or not self.config_digest.strip():
            raise ValueError("config_digest must be non-empty")
        if len(self.config_digest) != 64:
            raise ValueError("config_digest must be a 64-char hex SHA-256")
        if not self.primary_metric or not self.primary_metric.strip():
            raise ValueError("primary_metric must be non-empty")
        if not np.isfinite(self.primary_value):
            raise ValueError("primary_value must be finite")
        if self.no_submit is not True:
            raise ValueError("no_submit must be True on product probes")
        if self.invent_green_live_submit:
            raise ValueError("invent_green_live_submit must be False")
        if self.attestation_slot_present:
            raise ValueError(
                "attestation_slot_present must be False until attestation integration lands"
            )
        if self.hermetic_kit_slot_present:
            raise ValueError(
                "hermetic_kit_slot_present must be False until hermetic integration lands"
            )
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "harness_id": self.harness_id,
            "probe_kind": self.probe_kind,
            "config_digest": self.config_digest,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "no_submit": self.no_submit,
            "invent_green_live_submit": self.invent_green_live_submit,
            "attestation_slot_present": self.attestation_slot_present,
            "hermetic_kit_slot_present": self.hermetic_kit_slot_present,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Return hex SHA-256 of canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _build_harness_catalogue() -> tuple[CampaignHarnessRow, ...]:
    """Build the campaign-harness catalogue from ambient modules."""
    return (
        CampaignHarnessRow(
            harness_id="appqsim_protocol",
            kind="appqsim_protocol",
            title="AppQSim protocol governance harness",
            summary=(
                "AppQSim metrics vs classical exact diagonalisation; simulator "
                "dry-run; no live QPU default."
            ),
            ambient_pointer=("scpn_quantum_control.benchmarks.appqsim_protocol.appqsim_benchmark"),
            hardware_safety_pointer="hardware_safe_execution.no_submit_appqsim",
            advantage_protocol_pointer="advantage_protocol.appqsim_requires_promoted_ledger",
            support_posture="local_research",
            owner_ticket_required_for_live=True,
        ),
        CampaignHarnessRow(
            harness_id="iqm_layout_transfer",
            kind="iqm_layout_transfer",
            title="IQM layout-transfer harness template",
            summary=(
                "Layout transfer plan builder over lattice calibration; dry-run "
                "plan materialisation; live submit residual ticketed."
            ),
            ambient_pointer=(
                "scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark."
                "build_layout_transfer_plan"
            ),
            hardware_safety_pointer="hardware_safe_execution.no_submit_iqm_layout",
            advantage_protocol_pointer="advantage_protocol.layout_transfer_requires_promoted_ledger",
            support_posture="live_hardware_gated",
            owner_ticket_required_for_live=True,
        ),
        CampaignHarnessRow(
            harness_id="closed_loop_publication",
            kind="closed_loop_publication",
            title="Closed-loop publication run template",
            summary=(
                "Closed-loop latency + publication package dry-run; shared-host "
                "timing advisory; live residual ticketed."
            ),
            ambient_pointer=(
                "scpn_quantum_control.benchmarks.closed_loop_publication_run."
                "run_closed_loop_publication"
            ),
            hardware_safety_pointer="hardware_safe_execution.no_submit_closed_loop",
            advantage_protocol_pointer="advantage_protocol.closed_loop_requires_promoted_ledger",
            support_posture="live_hardware_gated",
            owner_ticket_required_for_live=True,
        ),
        CampaignHarnessRow(
            harness_id="benchmark_harness_registry",
            kind="benchmark_harness_registry",
            title="Ambient benchmark_harness family registry",
            summary=(
                "Typed benchmark family registry (implemented + planned) with "
                "fail-closed planned blockers."
            ),
            ambient_pointer=(
                "scpn_quantum_control.benchmark_harness.registry.list_benchmark_families"
            ),
            hardware_safety_pointer="hardware_safe_execution.registry_no_submit",
            advantage_protocol_pointer="advantage_protocol.registry_planned_not_promoted",
            support_posture="metadata_only",
            owner_ticket_required_for_live=True,
        ),
    )


_HARNESSES: Final[tuple[CampaignHarnessRow, ...]] = _build_harness_catalogue()


def _harness_map() -> dict[str, CampaignHarnessRow]:
    """Return harness_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, CampaignHarnessRow] = {}
    for row in _HARNESSES:
        key = row.harness_id.strip()
        if not key:
            raise RuntimeError("campaign harness catalogue contains blank harness_id")
        if key in mapping:
            raise RuntimeError(f"duplicate harness_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("campaign harness catalogue must be non-empty")
    return mapping


_HARNESS_BY_ID: Final[Mapping[str, CampaignHarnessRow]] = _harness_map()


def list_campaign_harness_ids() -> tuple[str, ...]:
    """Return all campaign harness identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable harness ids.

    """
    return tuple(row.harness_id for row in _HARNESSES)


def get_campaign_harness(harness_id: str) -> CampaignHarnessRow:
    """Return one harness row; fail closed on blank/unknown.

    Parameters
    ----------
    harness_id
        Harness identifier.

    Returns
    -------
    CampaignHarnessRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not harness_id or not str(harness_id).strip():
        raise ValueError("harness_id must be non-empty")
    key = str(harness_id).strip()
    try:
        return _HARNESS_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown harness_id: {key!r}") from exc


def iter_campaign_harnesses(
    *,
    kind: HarnessKind | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[CampaignHarnessRow, ...]:
    """Return filtered harness rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[CampaignHarnessRow, ...]
        Matching rows.

    """
    rows: Sequence[CampaignHarnessRow] = _HARNESSES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def list_ambient_benchmark_family_ids() -> tuple[str, ...]:
    """Return ambient benchmark_harness family ids (implemented + planned).

    Returns
    -------
    tuple[str, ...]
        Stable ambient family identifiers.

    """
    return tuple(family.benchmark_id for family in list_benchmark_families())


def decide_campaign_path(
    harness_id: str,
    *,
    mode: Literal["dry_run", "ticketed_live", "would_live"] = "dry_run",
    owner_ticket_present: bool = False,
    invent_green_live_submit: bool = False,
    invent_green_unattested_claim: bool = False,
    mutate_prereg_after_freeze: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a campaign-harness path may proceed.

    Parameters
    ----------
    harness_id
        Harness identifier.
    mode
        dry_run (default), ticketed_live, or would_live.
    owner_ticket_present
        Required for ticketed_live / would_live.
    invent_green_live_submit
        If true, refuse.
    invent_green_unattested_claim
        If true, refuse an unattested promotion claim.
    mutate_prereg_after_freeze
        If true, refuse post-hoc prereg mutation.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    row = get_campaign_harness(harness_id)
    blockers: list[str] = []
    if invent_green_live_submit:
        blockers.append(
            f"invent-green live QPU submit refused (harness={row.harness_id}; no-submit default)"
        )
    if invent_green_unattested_claim:
        blockers.append(
            "invent-green unattested claim promotion refused "
            f"(harness={row.harness_id}; attestation integration remains residual)"
        )
    if mutate_prereg_after_freeze:
        blockers.append(
            f"post-hoc prereg mutation refused (harness={row.harness_id}; campaign harness freeze)"
        )
    if mode not in {"dry_run", "ticketed_live", "would_live"}:
        blockers.append(f"unknown campaign mode: {mode!r}")
    if mode in {"ticketed_live", "would_live"}:
        if row.owner_ticket_required_for_live and not owner_ticket_present:
            blockers.append(
                f"owner ticket required for ticketed_live/would_live (harness={row.harness_id})"
            )
        if mode == "would_live":
            blockers.append(
                "would_live auto-submit refused on product surface "
                f"(harness={row.harness_id}; use ticketed residual)"
            )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="campaign path refused under fail-closed harness product policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"campaign path allowed for harness {row.harness_id!r} "
            f"(mode={mode!r}; no_submit_default=True)"
        ),
        blockers=(),
    )


def materialise_appqsim_probe(
    *,
    n_oscillators: int = 3,
    coupling: float = 0.5,
    seed: int = 0,
) -> MaterialisedCampaignProbe:
    """Materialise an AppQSim dry-run probe through the ambient benchmark.

    Parameters
    ----------
    n_oscillators
        Ring size for synthetic K/omega.
    coupling
        Nearest-neighbour coupling strength.
    seed
        Deterministic frequency seed.

    Returns
    -------
    MaterialisedCampaignProbe
        Finite primary observables with invent_green_live_submit=False.

    Raises
    ------
    ValueError
        If inputs are invalid.

    """
    if n_oscillators < 2:
        raise ValueError("n_oscillators must be >= 2")
    if coupling <= 0.0:
        raise ValueError("coupling must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    k_mat = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    for index in range(n_oscillators):
        neighbour = (index + 1) % n_oscillators
        k_mat[index, neighbour] = coupling
        k_mat[neighbour, index] = coupling
    omega = rng.normal(0.0, 0.1, size=n_oscillators).astype(np.float64)
    metrics = appqsim_benchmark(k_mat, omega)
    payload = {
        "schema": "campaign_appqsim_probe.v1",
        "n_oscillators": n_oscillators,
        "coupling": coupling,
        "seed": seed,
        "n_qubits": int(metrics.n_qubits),
    }
    return MaterialisedCampaignProbe(
        harness_id="appqsim_protocol",
        probe_kind="appqsim_dry_run",
        config_digest=_digest_payload(payload),
        primary_metric="order_parameter_error",
        primary_value=float(metrics.order_parameter_error),
        no_submit=True,
        invent_green_live_submit=False,
        attestation_slot_present=False,
        hermetic_kit_slot_present=False,
        demo_label="ambient_appqsim_benchmark_ring",
    )


def materialise_iqm_layout_probe(
    *,
    num_qubits: int = 8,
    seed: int = 20260721,
) -> MaterialisedCampaignProbe:
    """Materialise an IQM layout-transfer plan probe.

    Parameters
    ----------
    num_qubits
        Synthetic lattice size.
    seed
        Transpiler / plan seed.

    Returns
    -------
    MaterialisedCampaignProbe
        Finite primary observables with invent_green_live_submit=False.

    Raises
    ------
    ValueError
        If inputs are invalid.

    """
    if num_qubits < 4:
        raise ValueError("num_qubits must be >= 4")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    edges = tuple((index, index + 1) for index in range(num_qubits - 1))
    calibration = LatticeCalibration(
        num_qubits=num_qubits,
        edges=edges,
        edge_fidelity={(a, b): 0.99 for a, b in edges},
        readout_error={index: 0.01 for index in range(num_qubits)},
    )
    plan = build_layout_transfer_plan(
        calibration,
        sizes=(num_qubits,),
        depth=2,
        seed=seed,
    )
    parity_pass = 1.0 if plan.blocks and plan.blocks[0].depth_parity.passes else 0.0
    payload = {
        "schema": "campaign_iqm_layout_probe.v1",
        "num_qubits": num_qubits,
        "seed": seed,
        "main_shots": int(plan.main_shots),
        "block_count": len(plan.blocks),
    }
    return MaterialisedCampaignProbe(
        harness_id="iqm_layout_transfer",
        probe_kind="iqm_layout_plan_dry_run",
        config_digest=_digest_payload(payload),
        primary_metric="depth_parity_pass",
        primary_value=float(parity_pass),
        no_submit=True,
        invent_green_live_submit=False,
        attestation_slot_present=False,
        hermetic_kit_slot_present=False,
        demo_label="ambient_iqm_layout_transfer_plan_synthetic_lattice",
    )


def materialise_closed_loop_probe(
    *,
    n_oscillators: int = 3,
    n_rounds: int = 3,
    seed: int = 0,
) -> MaterialisedCampaignProbe:
    """Materialise a closed-loop publication dry-run probe.

    Parameters
    ----------
    n_oscillators
        Feedback network size.
    n_rounds
        Control rounds.
    seed
        Run seed.

    Returns
    -------
    MaterialisedCampaignProbe
        Finite primary observables with invent_green_live_submit=False.

    Raises
    ------
    ValueError
        If inputs are invalid.

    """
    if n_oscillators < 2:
        raise ValueError("n_oscillators must be >= 2")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    config = ClosedLoopRunConfig(
        n_oscillators=n_oscillators,
        n_rounds=n_rounds,
        dynamic_circuit_rounds=max(1, min(2, n_rounds)),
        seed=seed,
    )
    artifact = run_closed_loop_publication(config)
    latency = artifact.latency_report
    max_round = 0.0
    if isinstance(latency, Mapping):
        raw = latency.get("max_round_latency_s", 0.0)
        max_round = float(raw) if raw is not None else 0.0
    payload = {
        "schema": "campaign_closed_loop_probe.v1",
        "n_oscillators": n_oscillators,
        "n_rounds": n_rounds,
        "seed": seed,
        "schema_version": str(artifact.schema_version),
        "timing_grade": str(artifact.timing_grade),
    }
    return MaterialisedCampaignProbe(
        harness_id="closed_loop_publication",
        probe_kind="closed_loop_publication_dry_run",
        config_digest=_digest_payload(payload),
        primary_metric="max_round_latency_s",
        primary_value=float(max_round),
        no_submit=True,
        invent_green_live_submit=False,
        attestation_slot_present=False,
        hermetic_kit_slot_present=False,
        demo_label="ambient_closed_loop_publication_run",
    )


def materialise_demo_campaign_probe() -> MaterialisedCampaignProbe:
    """Materialise the deterministic closed-loop demo probe.

    Returns
    -------
    MaterialisedCampaignProbe
        Closed-loop publication dry-run probe.

    """
    return materialise_closed_loop_probe(n_oscillators=3, n_rounds=3, seed=0)


def map_campaign_harness_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of campaign harness product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.campaign_harness_product",
            "role": "campaign_harness_product_surface",
            "support_posture": "local_research",
            "harness_ids": list(list_campaign_harness_ids()),
            "no_submit_default": True,
            "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.benchmarks.appqsim_protocol",
            "role": "ambient_appqsim_protocol",
            "support_posture": "local_research",
            "symbol_name": "appqsim_benchmark",
            "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        },
        {
            "module_path": ("scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark"),
            "role": "ambient_iqm_layout_transfer",
            "support_posture": "live_hardware_gated",
            "symbol_name": "build_layout_transfer_plan",
            "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        },
        {
            "module_path": ("scpn_quantum_control.benchmarks.closed_loop_publication_run"),
            "role": "ambient_closed_loop_publication",
            "support_posture": "live_hardware_gated",
            "symbol_name": "run_closed_loop_publication",
            "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.benchmark_harness.registry",
            "role": "ambient_benchmark_family_registry",
            "support_posture": "metadata_only",
            "symbol_name": "list_benchmark_families",
            "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        },
    )


def build_campaign_harness_product_registry() -> dict[str, object]:
    """Build the full serialisable campaign harness product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with harnesses (no blanks).

    """
    harnesses = [row.to_dict() for row in _HARNESSES]
    ambient_families = list(list_ambient_benchmark_family_ids())
    return {
        "schema": CAMPAIGN_HARNESS_PRODUCT_SCHEMA,
        "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        "harness_count": len(harnesses),
        "blank_entry_count": 0,
        "no_submit_default_policy": True,
        "invent_green_live_submit_policy": False,
        "attestation_slot_policy": False,
        "hermetic_kit_slot_policy": False,
        "ambient_benchmark_family_ids": ambient_families,
        "public_surfaces": list(map_campaign_harness_public_surfaces()),
        "harnesses": harnesses,
        "policy_note": (
            "Reusable campaign harness templates only; dry-run default; "
            "live submission remains ticketed; hermetic reproduction and "
            "attestation slots remain explicit residual work."
        ),
    }


def assert_campaign_harness_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers harnesses without invent-green live submit.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_campaign_harness_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green policies appear.

    """
    registry = dict(payload) if payload is not None else build_campaign_harness_product_registry()
    harnesses = registry.get("harnesses")
    if not isinstance(harnesses, list) or not harnesses:
        raise ValueError(
            "campaign harness product registry must contain a non-empty harnesses list"
        )
    seen: set[str] = set()
    blank = 0
    appqsim_found = False
    closed_loop_found = False
    for index, row in enumerate(harnesses):
        if not isinstance(row, Mapping):
            raise ValueError(f"harness row {index} must be a mapping")
        harness_id = row.get("harness_id")
        invent = row.get("invent_green_live_submit")
        no_submit = row.get("no_submit_default")
        ambient = row.get("ambient_pointer")
        if not harness_id or not str(harness_id).strip():
            blank += 1
            continue
        hid = str(harness_id).strip()
        if hid in seen:
            raise ValueError(f"duplicate harness_id in registry: {hid!r}")
        seen.add(hid)
        if hid == "appqsim_protocol":
            appqsim_found = True
        if hid == "closed_loop_publication":
            closed_loop_found = True
        if invent is not False:
            raise ValueError(f"harness {hid!r} invent_green_live_submit must be False")
        if no_submit is not True:
            raise ValueError(f"harness {hid!r} no_submit_default must be True")
        if not ambient or not str(ambient).strip():
            raise ValueError(f"harness {hid!r} must have ambient_pointer")
    if blank:
        raise ValueError(f"campaign harness product registry has {blank} blank or invalid entries")
    if not appqsim_found:
        raise ValueError("campaign harness product registry missing appqsim_protocol")
    if not closed_loop_found:
        raise ValueError("campaign harness product registry missing closed_loop_publication")
    expected = set(list_campaign_harness_ids())
    if seen != expected:
        raise ValueError(
            f"registry harness set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    harness_count = registry.get("harness_count", -1)
    if not isinstance(harness_count, int) or harness_count != len(harnesses):
        raise ValueError("harness_count does not match harnesses list length")
    no_submit_policy = registry.get("no_submit_default_policy", False)
    if no_submit_policy is not True:
        raise ValueError("no_submit_default_policy must be True")
    invent_policy = registry.get("invent_green_live_submit_policy", True)
    if invent_policy is not False:
        raise ValueError("invent_green_live_submit_policy must be False")
    attestation = registry.get("attestation_slot_policy", True)
    if attestation is not False:
        raise ValueError(
            "attestation_slot_policy must be False until attestation integration lands"
        )
    hermetic = registry.get("hermetic_kit_slot_policy", True)
    if hermetic is not False:
        raise ValueError("hermetic_kit_slot_policy must be False until hermetic integration lands")
    return registry


__all__ = [
    "CAMPAIGN_HARNESS_CLAIM_BOUNDARY",
    "CAMPAIGN_HARNESS_PRODUCT_SCHEMA",
    "CampaignHarnessRow",
    "HarnessKind",
    "MaterialisedCampaignProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_campaign_harness_product_integrity",
    "build_campaign_harness_product_registry",
    "decide_campaign_path",
    "get_campaign_harness",
    "iter_campaign_harnesses",
    "list_ambient_benchmark_family_ids",
    "list_campaign_harness_ids",
    "map_campaign_harness_public_surfaces",
    "materialise_appqsim_probe",
    "materialise_closed_loop_probe",
    "materialise_demo_campaign_probe",
    "materialise_iqm_layout_probe",
]
