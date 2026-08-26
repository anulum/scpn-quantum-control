# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-HAL provider federation product
"""Fail-closed **Multi-HAL provider federation** product surface.

Productises a **capability-true federation matrix** over ambient HAL adapters
(``hardware/hal_*.py``, ``backends.list_hal_backend_descriptors``,
``hal.built_in_backend_profiles``) with hardware-safety no-submit default:

* versioned HAL inventory from live ambient descriptors and profiles;
* capability records for shots, mid-circuit measurement, pulse, approval, and IR;
* federation-matrix generation that rejects blank or unknown backend ids;
* offline dry-run path decisions without network submission;
* refuse invent-green live submit without owner ticket (hardware-safety compose).

Does **not** submit QPU jobs, invent online calibration, complete the feedback
adapter integration, or automate competitive-baseline monitoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .hardware.backends import list_hal_backend_descriptors
from .hardware.hal import BackendProfile, built_in_backend_profiles
from .hardware.provider_capability_core import (
    ProviderCapabilityDecision,
    ProviderCapabilitySnapshot,
    assess_provider_capability_snapshot,
)

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges for federation rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

FederationRouteMode = Literal["dry_run", "ticketed_prep", "would_live"]
"""Product-level federation route modes (surface never auto-submits)."""

MULTI_HAL_FEDERATION_PRODUCT_SCHEMA: Final[str] = "multi_hal_federation_product.v2"
"""JSON schema identifier for serialised product payloads."""

MULTI_HAL_FEDERATION_CLAIM_BOUNDARY: Final[str] = (
    "Multi-HAL provider federation product surface only; capability-true matrix "
    "over ambient hardware/hal_* adapters and backend descriptors; default "
    "hardware-safe no-submit dry-run; refuse invent-green live submit without owner "
    "ticket and refuse blank/unknown backend ids; does not claim live queue "
    "depth, complete feedback-adapter integration, or automated competitive-baseline monitoring"
)
"""Shared claim boundary for multi-HAL federation product payloads."""


@dataclass(frozen=True, slots=True)
class HalCapabilityRecord:
    """One capability-true HAL row in the versioned federation matrix.

    Attributes
    ----------
    backend_id
        Stable backend identifier (matches ambient profile/descriptor).
    provider
        Provider name.
    broker
        Aggregator / broker name when present.
    adapter_module
        Ambient HAL adapter module path.
    modality
        Hardware modality label from ambient profile.
    supports_shots
        Whether shot-based execution is declared.
    supports_mid_circuit_measurement
        Mid-circuit measurement declaration.
    supports_pulse
        Pulse / OpenPulse declaration.
    supports_statevector
        Statevector capability declaration.
    submit_requires_approval
        Whether live submit requires approval.
    can_submit
        Ambient can_submit flag (product still no-submit by default).
    is_cloud
        Whether the target is cloud-backed.
    ir_formats
        Supported IR format tokens.
    max_qubits
        Declared max qubits when known (None = unknown, not invent-green).
    no_submit_default
        Product default no-submit posture (always True on this surface).
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    backend_id: str
    provider: str
    broker: str
    adapter_module: str
    modality: str
    supports_shots: bool
    supports_mid_circuit_measurement: bool
    supports_pulse: bool
    supports_statevector: bool
    submit_requires_approval: bool
    can_submit: bool
    is_cloud: bool
    ir_formats: tuple[str, ...]
    max_qubits: int | None
    no_submit_default: bool = True
    support_posture: SupportPosture = "metadata_only"
    as_of: str = "2026-07-24"
    claim_boundary: str = MULTI_HAL_FEDERATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate capability record invariants."""
        if not self.backend_id or not self.backend_id.strip():
            raise ValueError("backend_id must be non-empty")
        if not self.provider or not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.broker or not self.broker.strip():
            raise ValueError("broker must be non-empty")
        if not self.adapter_module or not self.adapter_module.strip():
            raise ValueError("adapter_module must be non-empty")
        if not self.modality or not self.modality.strip():
            raise ValueError("modality must be non-empty")
        if not self.ir_formats:
            raise ValueError("ir_formats must be non-empty")
        if any(not item or not str(item).strip() for item in self.ir_formats):
            raise ValueError("ir_formats entries must be non-empty")
        if self.max_qubits is not None and self.max_qubits < 1:
            raise ValueError("max_qubits must be positive when set")
        if self.no_submit_default is not True:
            raise ValueError("no_submit_default must be True on product surface")
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
        """Return a JSON-ready mapping for this record."""
        return {
            "backend_id": self.backend_id,
            "provider": self.provider,
            "broker": self.broker,
            "adapter_module": self.adapter_module,
            "modality": self.modality,
            "supports_shots": self.supports_shots,
            "supports_mid_circuit_measurement": self.supports_mid_circuit_measurement,
            "supports_pulse": self.supports_pulse,
            "supports_statevector": self.supports_statevector,
            "submit_requires_approval": self.submit_requires_approval,
            "can_submit": self.can_submit,
            "is_cloud": self.is_cloud,
            "ir_formats": list(self.ir_formats),
            "max_qubits": self.max_qubits,
            "no_submit_default": self.no_submit_default,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for multi-HAL federation product use.

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
    claim_boundary: str = MULTI_HAL_FEDERATION_CLAIM_BOUNDARY

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
class MaterialisedFederationDryRunProbe:
    """Materialised offline dry-run federation probe for one HAL row.

    Attributes
    ----------
    backend_id
        Backend assessed.
    provider
        Provider name.
    status
        Ambient assess status (ready|blocked|unknown).
    no_submit
        Always True on product probes.
    invent_green_live_submit
        Always False.
    blockers
        Ambient / product blockers.
    warnings
        Ambient warnings.
    demo_label
        Demo fixture label.
    claim_boundary
        Non-promotional claim boundary.

    """

    backend_id: str
    provider: str
    status: str
    no_submit: bool
    invent_green_live_submit: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    demo_label: str
    claim_boundary: str = MULTI_HAL_FEDERATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate dry-run probe invariants."""
        if not self.backend_id or not self.backend_id.strip():
            raise ValueError("backend_id must be non-empty")
        if not self.provider or not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if self.status not in {"ready", "blocked", "unknown"}:
            raise ValueError(f"unknown status: {self.status!r}")
        if self.no_submit is not True:
            raise ValueError("no_submit must be True on product dry-run probe")
        if self.invent_green_live_submit:
            raise ValueError("invent_green_live_submit must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if any(not item or not item.strip() for item in self.warnings):
            raise ValueError("warnings entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "backend_id": self.backend_id,
            "provider": self.provider,
            "status": self.status,
            "no_submit": self.no_submit,
            "invent_green_live_submit": self.invent_green_live_submit,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _profile_by_id() -> dict[str, BackendProfile]:
    """Index ambient backend profiles by backend_id."""
    mapping: dict[str, BackendProfile] = {}
    for profile in built_in_backend_profiles():
        key = str(profile.backend_id).strip()
        if not key:
            raise RuntimeError("ambient backend profile has blank backend_id")
        if key in mapping:
            raise RuntimeError(f"duplicate ambient backend profile: {key!r}")
        mapping[key] = profile
    if not mapping:
        raise RuntimeError("ambient backend profiles must be non-empty")
    return mapping


def _build_hal_capability_catalogue() -> tuple[HalCapabilityRecord, ...]:
    """Build capability catalogue from ambient HAL descriptors + profiles."""
    profiles = _profile_by_id()
    rows: list[HalCapabilityRecord] = []
    seen: set[str] = set()
    for descriptor in list_hal_backend_descriptors():
        backend_id = str(descriptor.name).strip()
        if not backend_id:
            raise RuntimeError("ambient HAL descriptor has blank name")
        if backend_id in seen:
            raise RuntimeError(f"duplicate ambient HAL descriptor: {backend_id!r}")
        seen.add(backend_id)
        profile = profiles.get(backend_id)
        broker = (
            str(profile.broker).strip()
            if profile is not None and profile.broker
            else str(descriptor.provider).strip() or "direct"
        )
        modality = (
            str(profile.modality)
            if profile is not None and profile.modality
            else str(descriptor.execution_mode)
        )
        caps = profile.capabilities if profile is not None else None
        ir_formats = (
            tuple(str(item) for item in profile.ir_formats)
            if profile is not None and profile.ir_formats
            else tuple(str(item) for item in descriptor.workloads)
        )
        if not ir_formats:
            ir_formats = ("metadata_only",)
        max_qubits = caps.max_qubits if caps is not None else descriptor.max_qubits
        support: SupportPosture = (
            "live_hardware_gated"
            if descriptor.can_submit and descriptor.submit_requires_approval
            else "metadata_only"
        )
        rows.append(
            HalCapabilityRecord(
                backend_id=backend_id,
                provider=str(descriptor.provider),
                broker=broker or "direct",
                adapter_module=str(descriptor.adapter_module),
                modality=modality,
                supports_shots=bool(
                    caps.supports_shots if caps is not None else descriptor.supports_shots
                ),
                supports_mid_circuit_measurement=bool(
                    caps.supports_mid_circuit_measurement
                    if caps is not None
                    else descriptor.supports_mid_circuit_measurement
                ),
                supports_pulse=bool(
                    caps.supports_pulse if caps is not None else descriptor.supports_pulse
                ),
                supports_statevector=bool(
                    caps.supports_statevector
                    if caps is not None
                    else descriptor.supports_statevector
                ),
                submit_requires_approval=bool(descriptor.submit_requires_approval),
                can_submit=bool(descriptor.can_submit),
                is_cloud=bool(profile.is_cloud) if profile is not None else True,
                ir_formats=ir_formats,
                max_qubits=int(max_qubits) if max_qubits is not None else None,
                support_posture=support,
            )
        )
    if not rows:
        raise RuntimeError("multi-HAL federation catalogue must be non-empty")
    return tuple(rows)


_CANONICAL_HALS: Final[tuple[HalCapabilityRecord, ...]] = _build_hal_capability_catalogue()


def _catalogue_map() -> dict[str, HalCapabilityRecord]:
    """Return backend_id → capability row map; refuse blanks/duplicates."""
    mapping: dict[str, HalCapabilityRecord] = {}
    for row in _CANONICAL_HALS:
        key = row.backend_id.strip()
        if not key:
            raise RuntimeError("federation catalogue contains blank backend_id")
        if key in mapping:
            raise RuntimeError(f"duplicate backend_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("federation catalogue must be non-empty")
    return mapping


_HAL_BY_ID: Final[Mapping[str, HalCapabilityRecord]] = _catalogue_map()


def list_hal_backend_ids() -> tuple[str, ...]:
    """Return all federated HAL backend identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable backend ids from ambient inventory.

    """
    return tuple(row.backend_id for row in _CANONICAL_HALS)


def list_hal_providers() -> tuple[str, ...]:
    """Return unique provider names in stable first-seen order.

    Returns
    -------
    tuple[str, ...]
        Provider tokens.

    """
    seen: list[str] = []
    for row in _CANONICAL_HALS:
        if row.provider not in seen:
            seen.append(row.provider)
    return tuple(seen)


def get_hal_capability(backend_id: str) -> HalCapabilityRecord:
    """Return one HAL capability row; fail closed on blank/unknown.

    Parameters
    ----------
    backend_id
        Backend identifier.

    Returns
    -------
    HalCapabilityRecord
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not backend_id or not str(backend_id).strip():
        raise ValueError("backend_id must be non-empty")
    key = str(backend_id).strip()
    try:
        return _HAL_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown backend_id: {key!r}") from exc


def iter_hal_capabilities(
    *,
    provider: str | None = None,
    support_posture: SupportPosture | None = None,
    supports_pulse: bool | None = None,
) -> tuple[HalCapabilityRecord, ...]:
    """Return filtered HAL capability rows in stable order.

    Parameters
    ----------
    provider
        Optional provider filter.
    support_posture
        Optional posture filter.
    supports_pulse
        Optional pulse capability filter.

    Returns
    -------
    tuple[HalCapabilityRecord, ...]
        Matching rows.

    """
    rows: Sequence[HalCapabilityRecord] = _CANONICAL_HALS
    if provider is not None:
        rows = tuple(row for row in rows if row.provider == provider)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    if supports_pulse is not None:
        rows = tuple(row for row in rows if row.supports_pulse is supports_pulse)
    return tuple(rows)


def build_federation_matrix() -> tuple[dict[str, object], ...]:
    """Build the serialisable federation matrix from the validated catalogue.

    Returns
    -------
    tuple[dict[str, object], ...]
        One row per ambient HAL backend (no blanks).

    """
    return tuple(row.to_dict() for row in _CANONICAL_HALS)


def decide_federation_route(
    backend_id: str,
    *,
    mode: FederationRouteMode = "dry_run",
    owner_ticket_present: bool = False,
    invent_green_live_submit: bool = False,
    allow_network: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a hardware-safe federation route may proceed.

    Parameters
    ----------
    backend_id
        Target backend id (must be known).
    mode
        dry_run (default), ticketed_prep, or would_live.
    owner_ticket_present
        Required for ticketed_prep / would_live.
    invent_green_live_submit
        If true, refuse.
    allow_network
        Product dry-run refuses network by default; if true with dry_run, refuse
        invent-green network probing claims.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    row = get_hal_capability(backend_id)
    blockers: list[str] = []
    if invent_green_live_submit:
        blockers.append(
            f"invent-green live submit refused (backend={row.backend_id}; no-submit default)"
        )
    if mode not in {"dry_run", "ticketed_prep", "would_live"}:
        blockers.append(f"unknown federation mode: {mode!r}")
    if mode == "dry_run" and allow_network:
        blockers.append(
            "dry_run federation path refuses network probes "
            f"(backend={row.backend_id}; metadata-only matrix)"
        )
    if mode in {"ticketed_prep", "would_live"} and not owner_ticket_present:
        blockers.append(
            "owner ticket required for ticketed_prep/would_live "
            f"(backend={row.backend_id}; submit_requires_approval="
            f"{row.submit_requires_approval})"
        )
    if mode == "would_live" and not row.can_submit:
        blockers.append(f"backend {row.backend_id!r} cannot submit (can_submit=False)")
    if mode == "would_live":
        # Product surface never auto-authorises live submit even with ticket —
        # ticketed prep is the honest residual path.
        blockers.append(
            "would_live auto-submit refused on product surface "
            "(use the ticketed_prep residual under hardware-safe execution policy)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="federation route refused under fail-closed multi-HAL product policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"federation route {mode!r} allowed for backend {row.backend_id!r} "
            f"(provider={row.provider}; no_submit_default=True)"
        ),
        blockers=(),
    )


def _snapshot_from_record(row: HalCapabilityRecord) -> ProviderCapabilitySnapshot:
    """Build a no-submit offline metadata snapshot for dry-run assess."""
    # n_qubits floor=1 is schema-only when max_qubits unknown — not invent-green hardware size.
    n_qubits = row.max_qubits if row.max_qubits is not None else 1
    return ProviderCapabilitySnapshot(
        route_id=f"product_dry_run:{row.backend_id}",
        aggregator=row.broker,
        provider=row.provider,
        backend_id=row.backend_id,
        target_name=row.backend_id,
        n_qubits=n_qubits,
        supported_ir_formats=row.ir_formats,
        basis_gates=(),
        native_features=(),
        online=False,
        simulator=not row.is_cloud,
        no_submit=True,
        max_shots=None,
        max_circuits=None,
        queue_depth=None,
        calibration_timestamp=None,
    )


def materialise_federation_dry_run_probe(
    backend_id: str,
    *,
    required_ir_format: str | None = None,
    min_qubits: int | None = None,
) -> MaterialisedFederationDryRunProbe:
    """Materialise an offline dry-run capability probe via ambient assessment.

    Offline / no-submit snapshot; does not open network connections.

    Parameters
    ----------
    backend_id
        Known backend id.
    required_ir_format
        Optional IR requirement for assess.
    min_qubits
        Optional qubit floor for assess.

    Returns
    -------
    MaterialisedFederationDryRunProbe
        Finite primary observables with invent_green_live_submit=False.

    """
    row = get_hal_capability(backend_id)
    snapshot = _snapshot_from_record(row)
    decision: ProviderCapabilityDecision = assess_provider_capability_snapshot(
        snapshot,
        aggregator=row.broker,
        provider=row.provider,
        backend_id=row.backend_id,
        route_id=snapshot.route_id,
        required_ir_format=required_ir_format,
        min_qubits=min_qubits,
    )
    return MaterialisedFederationDryRunProbe(
        backend_id=row.backend_id,
        provider=row.provider,
        status=str(decision.status),
        no_submit=True,
        invent_green_live_submit=False,
        blockers=tuple(str(item) for item in decision.blockers),
        warnings=tuple(str(item) for item in decision.warnings),
        demo_label="ambient_provider_capability_dry_run",
    )


def materialise_demo_federation_dry_run_probe() -> MaterialisedFederationDryRunProbe:
    """Materialise a deterministic demo dry-run probe on the first catalogue row.

    Returns
    -------
    MaterialisedFederationDryRunProbe
        Offline assess with invent_green_live_submit=False.

    """
    first = list_hal_backend_ids()[0]
    return materialise_federation_dry_run_probe(first)


def map_multi_hal_federation_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of multi-HAL federation product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.multi_hal_federation_product",
            "role": "multi_hal_federation_product_surface",
            "support_posture": "metadata_only",
            "backend_count": len(list_hal_backend_ids()),
            "provider_count": len(list_hal_providers()),
            "no_submit_default": True,
            "claim_boundary": MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.hardware.backends",
            "role": "ambient_hal_descriptor_inventory",
            "support_posture": "metadata_only",
            "symbol_name": "list_hal_backend_descriptors",
            "claim_boundary": MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.hardware.hal",
            "role": "ambient_backend_profiles",
            "support_posture": "metadata_only",
            "symbol_name": "built_in_backend_profiles",
            "claim_boundary": MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.hardware.provider_capability_core",
            "role": "ambient_capability_assess_no_submit",
            "support_posture": "policy_only",
            "symbol_name": "assess_provider_capability_snapshot",
            "claim_boundary": MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
        },
    )


def build_multi_hal_federation_product_registry() -> dict[str, object]:
    """Build the full serialisable multi-HAL federation product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with matrix rows (no blanks).

    """
    matrix = list(build_federation_matrix())
    return {
        "schema": MULTI_HAL_FEDERATION_PRODUCT_SCHEMA,
        "claim_boundary": MULTI_HAL_FEDERATION_CLAIM_BOUNDARY,
        "backend_count": len(matrix),
        "provider_count": len(list_hal_providers()),
        "blank_entry_count": 0,
        "no_submit_default_policy": True,
        "invent_green_live_submit_policy": False,
        "public_surfaces": list(map_multi_hal_federation_public_surfaces()),
        "federation_matrix": matrix,
        "providers": list(list_hal_providers()),
        "policy_note": (
            "Capability-true multi-HAL matrix over ambient adapters only; "
            "hardware-safe no-submit dry-run default; live submit remains ticketed; "
            "feedback-adapter integration and competitive-baseline monitoring remain open."
        ),
    }


def assert_multi_hal_federation_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers HAL matrix without blanks or invent-green submit.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_multi_hal_federation_product_registry`.

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
        dict(payload) if payload is not None else build_multi_hal_federation_product_registry()
    )
    if registry.get("schema") != MULTI_HAL_FEDERATION_PRODUCT_SCHEMA:
        raise ValueError("multi-HAL federation product schema mismatch")
    if registry.get("claim_boundary") != MULTI_HAL_FEDERATION_CLAIM_BOUNDARY:
        raise ValueError("multi-HAL federation product claim boundary mismatch")
    matrix = registry.get("federation_matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(
            "multi-HAL federation product registry must contain a non-empty federation_matrix"
        )
    seen: set[str] = set()
    blank = 0
    for index, row in enumerate(matrix):
        if not isinstance(row, Mapping):
            raise ValueError(f"federation matrix row {index} must be a mapping")
        backend_id = row.get("backend_id")
        provider = row.get("provider")
        adapter = row.get("adapter_module")
        no_submit = row.get("no_submit_default")
        ir_formats = row.get("ir_formats")
        if not backend_id or not str(backend_id).strip():
            blank += 1
            continue
        bid = str(backend_id).strip()
        if bid in seen:
            raise ValueError(f"duplicate backend_id in registry: {bid!r}")
        seen.add(bid)
        if not provider or not str(provider).strip():
            raise ValueError(f"backend {bid!r} must have provider")
        if not adapter or not str(adapter).strip():
            raise ValueError(f"backend {bid!r} must have adapter_module")
        if no_submit is not True:
            raise ValueError(f"backend {bid!r} no_submit_default must be True")
        if not isinstance(ir_formats, list) or not ir_formats:
            raise ValueError(f"backend {bid!r} must have non-empty ir_formats list")
    if blank:
        raise ValueError(
            f"multi-HAL federation product registry has {blank} blank or invalid entries"
        )
    expected = set(list_hal_backend_ids())
    if seen != expected:
        raise ValueError(
            f"registry backend set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    backend_count = registry.get("backend_count", -1)
    if not isinstance(backend_count, int) or backend_count != len(matrix):
        raise ValueError("backend_count does not match federation_matrix length")
    no_submit_policy = registry.get("no_submit_default_policy", False)
    if no_submit_policy is not True:
        raise ValueError("no_submit_default_policy must be True")
    invent_policy = registry.get("invent_green_live_submit_policy", True)
    if invent_policy is not False:
        raise ValueError("invent_green_live_submit_policy must be False")
    return registry


__all__ = [
    "MULTI_HAL_FEDERATION_CLAIM_BOUNDARY",
    "MULTI_HAL_FEDERATION_PRODUCT_SCHEMA",
    "FederationRouteMode",
    "HalCapabilityRecord",
    "MaterialisedFederationDryRunProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_multi_hal_federation_product_integrity",
    "build_federation_matrix",
    "build_multi_hal_federation_product_registry",
    "decide_federation_route",
    "get_hal_capability",
    "iter_hal_capabilities",
    "list_hal_backend_ids",
    "list_hal_providers",
    "map_multi_hal_federation_public_surfaces",
    "materialise_demo_federation_dry_run_probe",
    "materialise_federation_dry_run_probe",
]
