# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Compose existing control/* stack product
"""Fail-closed **compose existing control/*** product surface.

Productises a typed adapter / ownership map over ambient production control
modules so quantum-classical co-design does **not** reinvent a second control
stack:

* versioned ownership catalogue for ``control/*`` and hardware feedback ports;
* protocol adapter rows (realtime feedback, closed-loop telemetry, cosim,
  hardware dry-run) with hardware-safety policy composition pointers;
* fail-closed path decision: refuse evaluate / run without
  :class:`~scpn_quantum_control.control.closed_loop_analysis.ClosedLoopExecutionPolicy`;
* materialised closed-loop telemetry probe via ambient
  :func:`~scpn_quantum_control.control.closed_loop_analysis.evaluate_closed_loop_policy`
  (simulation-only by default; invent-green live hardware refused without ticket);
* executable policy-gated realtime, QAOA-MPC, and quantum/classical partition
  adapters in :mod:`scpn_quantum_control.control_stack_runtime_adapters`;
* refuse invent-green PCS integration and rewrite claims of ``realtime_runtime``.

Does **not** rewrite ``realtime_runtime`` or invent PCS. Pulse execution remains
an explicit fail-closed hand-off to the optional pulse-execution boundary rather
than a control-stack responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .control.closed_loop_analysis import (
    ClosedLoopExecutionDecision,
    ClosedLoopExecutionPolicy,
    evaluate_closed_loop_policy,
)

OwnerKind = Literal[
    "control_realtime",
    "control_closed_loop",
    "control_qaoa_mpc",
    "control_adaptive",
    "control_runtime",
    "cosimulation",
    "hardware_feedback",
    "policy_compose",
]
"""Ownership kinds for compose catalogue rows."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "adapter_only",
]
"""Support posture badges for compose ownership rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

AdapterPort = Literal[
    "realtime_feedback",
    "closed_loop_telemetry",
    "qaoa_mpc_optional",
    "cosimulation_partition",
    "hardware_feedback_dryrun",
    "execution_policy_gate",
]
"""Typed adapter ports exposed by this product (not a second stack)."""

CONTROL_STACK_COMPOSE_PRODUCT_SCHEMA: Final[str] = "control_stack_compose_product.v1"
"""JSON schema identifier for serialised product payloads."""

CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY: Final[str] = (
    "Compose existing control/* stack product surface only; typed adapters and "
    "ownership map over ambient control/realtime_feedback, closed_loop_analysis, "
    "realtime_runtime, qaoa_mpc, adaptive_branching, cosimulation, and hardware "
    "feedback_* ports; refuse evaluate without ClosedLoopExecutionPolicy; refuse "
    "invent-green PCS integration and second-stack rewrites; QAOA-MPC and "
    "cosimulation adapters are local-only; pulse execution fails closed to the "
    "optional pulse-execution boundary"
)
"""Shared claim boundary for control-stack compose product payloads."""


@dataclass(frozen=True, slots=True)
class OwnershipRow:
    """One row in the existing control-stack ownership catalogue.

    Attributes
    ----------
    module_id
        Stable module identifier.
    module_path
        Import path of the ambient module (never a second stack).
    owner_kind
        Ownership kind badge.
    title
        Human-readable title.
    summary
        Short description of ownership responsibility.
    adapter_port
        Primary adapter port this module feeds (if any).
    support_posture
        Support posture badge.
    rewrites_forbidden
        Whether rewrites of this module are out of product scope.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    module_id: str
    module_path: str
    owner_kind: OwnerKind
    title: str
    summary: str
    adapter_port: AdapterPort | None
    support_posture: SupportPosture
    rewrites_forbidden: bool
    as_of: str = "2026-07-24"
    claim_boundary: str = CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate ownership row invariants."""
        if not self.module_id or not self.module_id.strip():
            raise ValueError("module_id must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.owner_kind not in {
            "control_realtime",
            "control_closed_loop",
            "control_qaoa_mpc",
            "control_adaptive",
            "control_runtime",
            "cosimulation",
            "hardware_feedback",
            "policy_compose",
        }:
            raise ValueError(f"unknown owner_kind: {self.owner_kind!r}")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "adapter_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if self.adapter_port is not None and self.adapter_port not in {
            "realtime_feedback",
            "closed_loop_telemetry",
            "qaoa_mpc_optional",
            "cosimulation_partition",
            "hardware_feedback_dryrun",
            "execution_policy_gate",
        }:
            raise ValueError(f"unknown adapter_port: {self.adapter_port!r}")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "module_id": self.module_id,
            "module_path": self.module_path,
            "owner_kind": self.owner_kind,
            "title": self.title,
            "summary": self.summary,
            "adapter_port": self.adapter_port,
            "support_posture": self.support_posture,
            "rewrites_forbidden": self.rewrites_forbidden,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class AdapterPortRow:
    """One typed adapter port over ambient control modules.

    Attributes
    ----------
    port_id
        Adapter port identifier.
    title
        Human-readable title.
    ambient_modules
        Ambient module paths this port adapts (compose, do not rewrite).
    hardware_safety_pointer
        Hardware-safe, no-submit policy pointer.
    support_posture
        Support posture badge.
    requires_execution_policy
        Whether evaluate/run must pass ClosedLoopExecutionPolicy first.
    invent_green_pcs
        Must remain False (PCS integration not claimed).
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    port_id: AdapterPort
    title: str
    ambient_modules: tuple[str, ...]
    hardware_safety_pointer: str
    support_posture: SupportPosture
    requires_execution_policy: bool
    invent_green_pcs: bool = False
    as_of: str = "2026-07-24"
    claim_boundary: str = CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate adapter port invariants."""
        if self.port_id not in {
            "realtime_feedback",
            "closed_loop_telemetry",
            "qaoa_mpc_optional",
            "cosimulation_partition",
            "hardware_feedback_dryrun",
            "execution_policy_gate",
        }:
            raise ValueError(f"unknown port_id: {self.port_id!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.ambient_modules:
            raise ValueError("ambient_modules must be non-empty")
        if any(not item or not str(item).strip() for item in self.ambient_modules):
            raise ValueError("ambient_modules entries must be non-empty")
        if not self.hardware_safety_pointer or not self.hardware_safety_pointer.strip():
            raise ValueError("hardware_safety_pointer must be non-empty")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "adapter_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if self.invent_green_pcs:
            raise ValueError("invent_green_pcs must be False (no PCS invent-green)")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this port."""
        return {
            "port_id": self.port_id,
            "title": self.title,
            "ambient_modules": list(self.ambient_modules),
            "hardware_safety_pointer": self.hardware_safety_pointer,
            "support_posture": self.support_posture,
            "requires_execution_policy": self.requires_execution_policy,
            "invent_green_pcs": self.invent_green_pcs,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for control-stack compose use.

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
    claim_boundary: str = CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY

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
class MaterialisedClosedLoopTelemetryProbe:
    """Materialised telemetry probe from ambient closed-loop policy evaluation.

    Attributes
    ----------
    authorised
        Whether ambient policy authorised the request.
    mode
        Ambient execution mode string (simulation|hardware).
    reason
        Ambient decision reason.
    requested_rounds
        Rounds requested at product boundary.
    invent_green_pcs
        Always False — PCS not claimed.
    allow_hardware
        Policy allow_hardware flag used.
    live_ticket_present
        Whether a non-empty live ticket was supplied.
    demo_label
        Demo fixture label.
    claim_boundary
        Non-promotional claim boundary.

    """

    authorised: bool
    mode: str
    reason: str
    requested_rounds: int
    invent_green_pcs: bool
    allow_hardware: bool
    live_ticket_present: bool
    demo_label: str
    claim_boundary: str = CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate telemetry probe invariants."""
        if self.requested_rounds < 1:
            raise ValueError("requested_rounds must be positive")
        if not self.mode or not self.mode.strip():
            raise ValueError("mode must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.invent_green_pcs:
            raise ValueError("invent_green_pcs must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")
        if self.mode not in {"simulation", "hardware"}:
            raise ValueError(f"unknown mode: {self.mode!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "authorised": self.authorised,
            "mode": self.mode,
            "reason": self.reason,
            "requested_rounds": self.requested_rounds,
            "invent_green_pcs": self.invent_green_pcs,
            "allow_hardware": self.allow_hardware,
            "live_ticket_present": self.live_ticket_present,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_ownership_catalogue() -> tuple[OwnershipRow, ...]:
    """Build the fixed ownership map (control/* vs adapters; no second stack)."""
    return (
        OwnershipRow(
            module_id="realtime_feedback",
            module_path="scpn_quantum_control.control.realtime_feedback",
            owner_kind="control_realtime",
            title="Realtime sync feedback controller",
            summary="Ambient Kuramoto-XY feedback controller; adapter port only.",
            adapter_port="realtime_feedback",
            support_posture="local_research",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="realtime_runtime",
            module_path="scpn_quantum_control.control.realtime_runtime",
            owner_kind="control_runtime",
            title="Realtime runtime + SLA",
            summary="Production realtime loop/SLA; compose adapters cannot rewrite it.",
            adapter_port=None,
            support_posture="local_research",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="closed_loop_analysis",
            module_path="scpn_quantum_control.control.closed_loop_analysis",
            owner_kind="control_closed_loop",
            title="Closed-loop analysis + ExecutionPolicy",
            summary="Policy gate and telemetry source for co-design adapters.",
            adapter_port="closed_loop_telemetry",
            support_posture="policy_only",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="qaoa_mpc",
            module_path="scpn_quantum_control.control.qaoa_mpc",
            owner_kind="control_qaoa_mpc",
            title="QAOA-MPC schedule control",
            summary=(
                "Abstract QAOA-MPC adapter; pulse execution fails closed to the "
                "optional pulse-execution boundary."
            ),
            adapter_port="qaoa_mpc_optional",
            support_posture="local_research",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="adaptive_branching",
            module_path="scpn_quantum_control.control.adaptive_branching",
            owner_kind="control_adaptive",
            title="Adaptive branching readiness",
            summary="Branch tables and readiness; adapter-only exposure.",
            adapter_port=None,
            support_posture="adapter_only",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="cosimulation_quantum_classical",
            module_path="scpn_quantum_control.cosimulation.quantum_classical",
            owner_kind="cosimulation",
            title="Quantum-classical cosimulation partition",
            summary="Executable local Knm partition bridge with mapped telemetry.",
            adapter_port="cosimulation_partition",
            support_posture="local_research",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="hardware_feedback_dryrun",
            module_path="scpn_quantum_control.hardware.feedback_dryrun",
            owner_kind="hardware_feedback",
            title="Hardware feedback dry-run",
            summary="Hardware-side adapter under same ports; no second stack.",
            adapter_port="hardware_feedback_dryrun",
            support_posture="live_hardware_gated",
            rewrites_forbidden=True,
        ),
        OwnershipRow(
            module_id="execution_policy_gate",
            module_path="scpn_quantum_control.control.closed_loop_analysis",
            owner_kind="policy_compose",
            title="ExecutionPolicy gate (hardware-safe compose)",
            summary="Refuse evaluate/run without ClosedLoopExecutionPolicy.",
            adapter_port="execution_policy_gate",
            support_posture="policy_only",
            rewrites_forbidden=True,
        ),
    )


def _build_adapter_ports() -> tuple[AdapterPortRow, ...]:
    """Build the typed adapter-port catalogue."""
    return (
        AdapterPortRow(
            port_id="execution_policy_gate",
            title="ExecutionPolicy gate",
            ambient_modules=("scpn_quantum_control.control.closed_loop_analysis",),
            hardware_safety_pointer="hardware_safe_execution.closed_loop_execution_policy",
            support_posture="policy_only",
            requires_execution_policy=True,
        ),
        AdapterPortRow(
            port_id="realtime_feedback",
            title="Realtime feedback adapter",
            ambient_modules=("scpn_quantum_control.control.realtime_feedback",),
            hardware_safety_pointer="hardware_safe_execution.no_submit_feedback",
            support_posture="local_research",
            requires_execution_policy=True,
        ),
        AdapterPortRow(
            port_id="closed_loop_telemetry",
            title="Closed-loop telemetry adapter",
            ambient_modules=("scpn_quantum_control.control.closed_loop_analysis",),
            hardware_safety_pointer="hardware_safe_execution.closed_loop_telemetry",
            support_posture="policy_only",
            requires_execution_policy=True,
        ),
        AdapterPortRow(
            port_id="qaoa_mpc_optional",
            title="QAOA-MPC optional adapter",
            ambient_modules=("scpn_quantum_control.control.qaoa_mpc",),
            hardware_safety_pointer="hardware_safe_execution.qaoa_mpc_optional",
            support_posture="local_research",
            requires_execution_policy=True,
        ),
        AdapterPortRow(
            port_id="cosimulation_partition",
            title="Cosimulation partition adapter",
            ambient_modules=("scpn_quantum_control.cosimulation.quantum_classical",),
            hardware_safety_pointer="hardware_safe_execution.cosim_partition",
            support_posture="local_research",
            requires_execution_policy=True,
        ),
        AdapterPortRow(
            port_id="hardware_feedback_dryrun",
            title="Hardware feedback dry-run adapter",
            ambient_modules=(
                "scpn_quantum_control.hardware.feedback_dryrun",
                "scpn_quantum_control.hardware.feedback_loop",
            ),
            hardware_safety_pointer="hardware_safe_execution.feedback_dryrun",
            support_posture="live_hardware_gated",
            requires_execution_policy=True,
        ),
    )


_OWNERSHIP: Final[tuple[OwnershipRow, ...]] = _build_ownership_catalogue()
_PORTS: Final[tuple[AdapterPortRow, ...]] = _build_adapter_ports()


def _ownership_map() -> dict[str, OwnershipRow]:
    """Return module_id → ownership row map; refuse blanks/duplicates."""
    mapping: dict[str, OwnershipRow] = {}
    for row in _OWNERSHIP:
        key = row.module_id.strip()
        if not key:
            raise RuntimeError("ownership catalogue contains blank module_id")
        if key in mapping:
            raise RuntimeError(f"duplicate module_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("ownership catalogue must be non-empty")
    return mapping


def _port_map() -> dict[str, AdapterPortRow]:
    """Return port_id → adapter port row map; refuse blanks/duplicates."""
    mapping: dict[str, AdapterPortRow] = {}
    for row in _PORTS:
        key = str(row.port_id).strip()
        if not key:
            raise RuntimeError("adapter port catalogue contains blank port_id")
        if key in mapping:
            raise RuntimeError(f"duplicate port_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("adapter port catalogue must be non-empty")
    return mapping


_OWNERSHIP_BY_ID: Final[Mapping[str, OwnershipRow]] = _ownership_map()
_PORTS_BY_ID: Final[Mapping[str, AdapterPortRow]] = _port_map()


def list_ownership_module_ids() -> tuple[str, ...]:
    """Return all ownership module identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable module ids.

    """
    return tuple(row.module_id for row in _OWNERSHIP)


def list_adapter_port_ids() -> tuple[str, ...]:
    """Return all adapter port identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable port ids.

    """
    return tuple(row.port_id for row in _PORTS)


def get_ownership_row(module_id: str) -> OwnershipRow:
    """Return one ownership row by module id; fail closed on blank/unknown.

    Parameters
    ----------
    module_id
        Ownership module identifier.

    Returns
    -------
    OwnershipRow
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
        return _OWNERSHIP_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown module_id: {key!r}") from exc


def get_adapter_port(port_id: str) -> AdapterPortRow:
    """Return one adapter port by id; fail closed on blank/unknown.

    Parameters
    ----------
    port_id
        Adapter port identifier.

    Returns
    -------
    AdapterPortRow
        Matching port.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not port_id or not str(port_id).strip():
        raise ValueError("port_id must be non-empty")
    key = str(port_id).strip()
    try:
        return _PORTS_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown port_id: {key!r}") from exc


def iter_ownership_rows(
    *,
    support_posture: SupportPosture | None = None,
) -> tuple[OwnershipRow, ...]:
    """Return filtered ownership rows in stable order.

    Parameters
    ----------
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[OwnershipRow, ...]
        Matching rows.

    """
    rows: Sequence[OwnershipRow] = _OWNERSHIP
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def decide_control_compose_path(
    port_id: str,
    *,
    policy_present: bool = False,
    invent_green_pcs: bool = False,
    rewrite_realtime_runtime: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a compose adapter path may proceed.

    Parameters
    ----------
    port_id
        Adapter port identifier.
    policy_present
        Whether a ClosedLoopExecutionPolicy instance is supplied by the caller.
    invent_green_pcs
        If true, refuse (PCS invent-green forbidden).
    rewrite_realtime_runtime
        If true, refuse (rewrites forbidden).

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    port = get_adapter_port(port_id)
    blockers: list[str] = []
    if invent_green_pcs:
        blockers.append(
            f"invent-green PCS integration refused (pointer={port.hardware_safety_pointer}; claim_boundary)"
        )
    if rewrite_realtime_runtime:
        blockers.append(
            "rewrite of realtime_runtime refused (compose adapters preserve ambient "
            "runtime ownership; rewrites_forbidden)"
        )
    if port.requires_execution_policy and not policy_present:
        blockers.append(
            "ClosedLoopExecutionPolicy required before evaluate/run "
            f"(port={port.port_id}; hardware-safe compose gate)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="control compose path refused under fail-closed product policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"adapter port {port.port_id!r} allowed under ExecutionPolicy "
            f"(ambient={','.join(port.ambient_modules)})"
        ),
        blockers=(),
    )


def materialise_closed_loop_telemetry_probe(
    *,
    allow_hardware: bool = False,
    live_ticket: str | None = None,
    backend_allowlist: tuple[str, ...] = ("sim",),
    round_budget: int = 8,
    requested_rounds: int = 1,
) -> MaterialisedClosedLoopTelemetryProbe:
    """Materialise telemetry via ambient ``evaluate_closed_loop_policy``.

    Always sets ``invent_green_pcs=False``. Live hardware requires ambient
    policy authorisation (ticket + allow_hardware).

    Parameters
    ----------
    allow_hardware
        Policy allow_hardware flag.
    live_ticket
        Optional live ticket string.
    backend_allowlist
        Ambient backend allowlist.
    round_budget
        Ambient round budget.
    requested_rounds
        Rounds requested for evaluate.

    Returns
    -------
    MaterialisedClosedLoopTelemetryProbe
        Finite primary observables from ambient decision.

    Raises
    ------
    ValueError
        If requested_rounds invalid or ambient evaluate fails validation.

    """
    if requested_rounds < 1:
        raise ValueError("requested_rounds must be positive")
    policy = ClosedLoopExecutionPolicy(
        allow_hardware=allow_hardware,
        live_ticket=live_ticket,
        backend_allowlist=backend_allowlist,
        round_budget=round_budget,
    )
    decision: ClosedLoopExecutionDecision = evaluate_closed_loop_policy(
        policy,
        requested_rounds=requested_rounds,
    )
    mode = str(decision.mode.value if hasattr(decision.mode, "value") else decision.mode)
    return MaterialisedClosedLoopTelemetryProbe(
        authorised=bool(decision.authorised),
        mode=mode,
        reason=str(decision.reason),
        requested_rounds=requested_rounds,
        invent_green_pcs=False,
        allow_hardware=allow_hardware,
        live_ticket_present=bool(live_ticket and str(live_ticket).strip()),
        demo_label="ambient_closed_loop_execution_policy_probe",
    )


def materialise_demo_closed_loop_telemetry_probe() -> MaterialisedClosedLoopTelemetryProbe:
    """Materialise the deterministic simulation-only demo probe.

    Returns
    -------
    MaterialisedClosedLoopTelemetryProbe
        Authorised simulation decision with invent_green_pcs=False.

    """
    return materialise_closed_loop_telemetry_probe(
        allow_hardware=False,
        live_ticket=None,
        backend_allowlist=("sim",),
        round_budget=8,
        requested_rounds=1,
    )


def map_control_stack_compose_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of control-stack compose product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.control_stack_compose_product",
            "role": "control_stack_compose_product_surface",
            "support_posture": "adapter_only",
            "ownership_module_ids": list(list_ownership_module_ids()),
            "adapter_port_ids": list(list_adapter_port_ids()),
            "invent_green_pcs": False,
            "claim_boundary": CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.control.closed_loop_analysis",
            "role": "ambient_execution_policy_and_telemetry",
            "support_posture": "policy_only",
            "symbol_name": "evaluate_closed_loop_policy",
            "claim_boundary": CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.control.realtime_feedback",
            "role": "ambient_realtime_feedback_controller",
            "support_posture": "local_research",
            "symbol_name": "RealtimeSyncFeedbackController",
            "claim_boundary": CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.control.realtime_runtime",
            "role": "ambient_realtime_runtime_no_rewrite",
            "support_posture": "local_research",
            "rewrites_forbidden": True,
            "claim_boundary": CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
        },
    )


def build_control_stack_compose_product_registry() -> dict[str, object]:
    """Build the full serialisable control-stack compose product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with ownership + ports (no blanks).

    """
    ownership = [row.to_dict() for row in _OWNERSHIP]
    ports = [row.to_dict() for row in _PORTS]
    return {
        "schema": CONTROL_STACK_COMPOSE_PRODUCT_SCHEMA,
        "claim_boundary": CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY,
        "ownership_count": len(ownership),
        "port_count": len(ports),
        "blank_entry_count": 0,
        "invent_green_pcs_policy": False,
        "rewrites_forbidden_policy": True,
        "public_surfaces": list(map_control_stack_compose_public_surfaces()),
        "ownership": ownership,
        "adapter_ports": ports,
        "policy_note": (
            "Compose adapters over ambient control/* only; no second stack; "
            "ClosedLoopExecutionPolicy required before evaluate/run; "
            "PCS invent-green forbidden; local QAOA-MPC and cosimulation adapters "
            "are executable; pulse execution fails closed to the optional "
            "pulse-execution boundary."
        ),
    }


def assert_control_stack_compose_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers ownership/ports without blanks or invent-green PCS.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_control_stack_compose_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, invent-green PCS, or rewrite policy drift.

    """
    registry = (
        dict(payload) if payload is not None else build_control_stack_compose_product_registry()
    )
    ownership = registry.get("ownership")
    ports = registry.get("adapter_ports")
    if not isinstance(ownership, list) or not ownership:
        raise ValueError(
            "control stack compose product registry must contain a non-empty ownership list"
        )
    if not isinstance(ports, list) or not ports:
        raise ValueError(
            "control stack compose product registry must contain a non-empty adapter_ports list"
        )
    seen_modules: set[str] = set()
    blank = 0
    policy_gate_found = False
    for index, row in enumerate(ownership):
        if not isinstance(row, Mapping):
            raise ValueError(f"ownership row {index} must be a mapping")
        module_id = row.get("module_id")
        module_path = row.get("module_path")
        rewrites = row.get("rewrites_forbidden")
        if not module_id or not str(module_id).strip():
            blank += 1
            continue
        mid = str(module_id).strip()
        if mid in seen_modules:
            raise ValueError(f"duplicate module_id in registry: {mid!r}")
        seen_modules.add(mid)
        if mid == "execution_policy_gate":
            policy_gate_found = True
        if not module_path or not str(module_path).strip():
            raise ValueError(f"module {mid!r} must have module_path")
        if rewrites is not True:
            raise ValueError(f"module {mid!r} must set rewrites_forbidden=True")
    if blank:
        raise ValueError(
            f"control stack compose product registry has {blank} blank or invalid entries"
        )
    if not policy_gate_found:
        raise ValueError("control stack compose product registry missing execution_policy_gate")
    expected = set(list_ownership_module_ids())
    if seen_modules != expected:
        raise ValueError(
            f"registry ownership set drift (missing={expected - seen_modules!r}, "
            f"extra={seen_modules - expected!r})"
        )
    seen_ports: set[str] = set()
    for index, row in enumerate(ports):
        if not isinstance(row, Mapping):
            raise ValueError(f"adapter port row {index} must be a mapping")
        port_id = row.get("port_id")
        requires = row.get("requires_execution_policy")
        invent_pcs = row.get("invent_green_pcs")
        if not port_id or not str(port_id).strip():
            raise ValueError(f"adapter port row {index} blank or invalid port_id")
        pid = str(port_id).strip()
        if pid in seen_ports:
            raise ValueError(f"duplicate port_id in registry: {pid!r}")
        seen_ports.add(pid)
        if requires is not True:
            raise ValueError(f"port {pid!r} must require_execution_policy=True")
        if invent_pcs is not False:
            raise ValueError(f"port {pid!r} invent_green_pcs must be False")
    expected_ports = set(list_adapter_port_ids())
    if seen_ports != expected_ports:
        raise ValueError(
            f"registry port set drift (missing={expected_ports - seen_ports!r}, "
            f"extra={seen_ports - expected_ports!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    ownership_count = registry.get("ownership_count", -1)
    if not isinstance(ownership_count, int) or ownership_count != len(ownership):
        raise ValueError("ownership_count does not match ownership list length")
    port_count = registry.get("port_count", -1)
    if not isinstance(port_count, int) or port_count != len(ports):
        raise ValueError("port_count does not match adapter_ports list length")
    invent_policy = registry.get("invent_green_pcs_policy", True)
    if invent_policy is not False:
        raise ValueError("invent_green_pcs_policy must be False")
    rewrites_policy = registry.get("rewrites_forbidden_policy", False)
    if rewrites_policy is not True:
        raise ValueError("rewrites_forbidden_policy must be True")
    return registry


__all__ = [
    "CONTROL_STACK_COMPOSE_CLAIM_BOUNDARY",
    "CONTROL_STACK_COMPOSE_PRODUCT_SCHEMA",
    "AdapterPort",
    "AdapterPortRow",
    "MaterialisedClosedLoopTelemetryProbe",
    "OwnerKind",
    "OwnershipRow",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_control_stack_compose_product_integrity",
    "build_control_stack_compose_product_registry",
    "decide_control_compose_path",
    "get_adapter_port",
    "get_ownership_row",
    "iter_ownership_rows",
    "list_adapter_port_ids",
    "list_ownership_module_ids",
    "map_control_stack_compose_public_surfaces",
    "materialise_closed_loop_telemetry_probe",
    "materialise_demo_closed_loop_telemetry_probe",
]
