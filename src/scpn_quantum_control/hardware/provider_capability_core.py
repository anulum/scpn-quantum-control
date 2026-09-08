# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Core
"""Provider-neutral no-submit capability contracts and readiness decisions."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, get_args

from .aggregators import (
    AggregatorProviderRoute,
    ResolvedAggregatorProviderRoute,
    built_in_aggregator_provider_routes,
    resolve_aggregator_provider_route,
)
from .openpulse_control import (
    OpenPulseCalibrationWorkflow,
    build_rabi_amplitude_calibration_workflow,
)

CapabilityDecisionStatus = Literal["ready", "blocked", "unknown"]

RouteVerb = Literal["metadata", "compile", "submit", "retrieve", "cancel", "result_formats"]
"""Operations a provider route can advertise, inventoried independently."""

ROUTE_VERBS: tuple[RouteVerb, ...] = get_args(RouteVerb)

DIRECT_AGGREGATOR = "direct"
"""Aggregator label meaning the provider is reached without a broker."""

ROUTE_CATALOGUE_CONTRACT = "provider_route_catalogue.v1"

_OBSERVATION_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


ProviderMetadataProbe = Callable[[ResolvedAggregatorProviderRoute], "ProviderCapabilitySnapshot"]


@dataclass(frozen=True)
class ProviderCapabilitySnapshot:
    """Provider target metadata collected without submitting a workload."""

    route_id: str
    aggregator: str
    provider: str
    backend_id: str
    target_name: str
    n_qubits: int
    supported_ir_formats: tuple[str, ...]
    basis_gates: tuple[str, ...] = field(default_factory=tuple)
    native_features: tuple[str, ...] = field(default_factory=tuple)
    online: bool | None = None
    simulator: bool = False
    no_submit: bool = True
    max_shots: int | None = None
    max_circuits: int | None = None
    queue_depth: int | None = None
    calibration_timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the no-submit target metadata contract."""
        for field_name in ("route_id", "aggregator", "provider", "backend_id", "target_name"):
            _require_text(getattr(self, field_name), field_name)
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if not self.supported_ir_formats:
            raise ValueError("supported_ir_formats must not be empty")
        _require_string_tuple(self.supported_ir_formats, "supported_ir_formats")
        _require_string_tuple(self.basis_gates, "basis_gates")
        _require_string_tuple(self.native_features, "native_features")
        if self.no_submit is not True:
            raise ValueError("capability snapshots must be no-submit metadata")
        for field_name in ("max_shots", "max_circuits"):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be positive when provided")
        if self.queue_depth is not None and self.queue_depth < 0:
            raise ValueError("queue_depth must be non-negative when provided")
        if self.calibration_timestamp is not None:
            _require_text(self.calibration_timestamp, "calibration_timestamp")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")


@dataclass(frozen=True)
class ProviderCapabilityDecision:
    """Readiness decision for one no-submit provider capability snapshot."""

    snapshot: ProviderCapabilitySnapshot
    status: CapabilityDecisionStatus
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    required_ir_format: str | None
    min_qubits: int | None
    no_submit: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise the provider capability decision."""
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "required_ir_format": self.required_ir_format,
            "min_qubits": self.min_qubits,
            "no_submit": self.no_submit,
            "snapshot": {
                "route_id": self.snapshot.route_id,
                "aggregator": self.snapshot.aggregator,
                "provider": self.snapshot.provider,
                "backend_id": self.snapshot.backend_id,
                "target_name": self.snapshot.target_name,
                "n_qubits": self.snapshot.n_qubits,
                "supported_ir_formats": list(self.snapshot.supported_ir_formats),
                "basis_gates": list(self.snapshot.basis_gates),
                "native_features": list(self.snapshot.native_features),
                "online": self.snapshot.online,
                "simulator": self.snapshot.simulator,
                "max_shots": self.snapshot.max_shots,
                "max_circuits": self.snapshot.max_circuits,
                "queue_depth": self.snapshot.queue_depth,
                "calibration_timestamp": self.snapshot.calibration_timestamp,
                "metadata": dict(self.snapshot.metadata),
            },
        }


@dataclass(frozen=True)
class OpenPulseControlReadiness:
    """No-submit readiness surface for pulse-level OpenPulse calibration lanes."""

    snapshot: ProviderCapabilitySnapshot
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    workflow: OpenPulseCalibrationWorkflow | None
    required_ir_formats: tuple[str, ...]
    required_native_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the OpenPulse readiness decision."""
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "required_ir_formats": list(self.required_ir_formats),
            "required_native_features": list(self.required_native_features),
            "snapshot": {
                "route_id": self.snapshot.route_id,
                "provider": self.snapshot.provider,
                "backend_id": self.snapshot.backend_id,
                "target_name": self.snapshot.target_name,
                "supported_ir_formats": list(self.snapshot.supported_ir_formats),
                "native_features": list(self.snapshot.native_features),
                "n_qubits": self.snapshot.n_qubits,
                "online": self.snapshot.online,
                "simulator": self.snapshot.simulator,
                "calibration_timestamp": self.snapshot.calibration_timestamp,
            },
            "workflow": self.workflow.to_payload() if self.workflow is not None else None,
            "hardware_submission": False,
        }


def build_openpulse_control_readiness(
    snapshot: ProviderCapabilitySnapshot,
    *,
    qubit: int,
    dt: float,
    amplitude_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
    shots: int = 4096,
) -> OpenPulseControlReadiness:
    """Build no-submit readiness for OpenPulse control and calibration workflows."""
    blockers: list[str] = []
    warnings: list[str] = []
    required_ir_formats = ("openpulse", "qiskit_qpy", "qiskit")
    required_native_features = ("pulse_control", "drive_channel_access")

    if snapshot.online is False:
        blockers.append("target is offline")
    if snapshot.online is None:
        warnings.append("target online status is unknown")
    if snapshot.n_qubits <= qubit:
        blockers.append(
            f"target exposes {snapshot.n_qubits} qubits, requested qubit index {qubit}"
        )
    if not any(fmt in snapshot.supported_ir_formats for fmt in required_ir_formats):
        blockers.append(
            "target does not advertise an OpenPulse-compatible IR route "
            f"(required one of: {', '.join(required_ir_formats)})"
        )

    feature_set = set(snapshot.native_features)
    if "dynamic_circuits" not in feature_set:
        warnings.append("target does not advertise dynamic_circuits")
    missing_features = [
        feature for feature in required_native_features if feature not in feature_set
    ]
    if missing_features:
        blockers.append(
            "target is missing pulse native features: " + ", ".join(sorted(missing_features))
        )

    workflow: OpenPulseCalibrationWorkflow | None = None
    if not blockers:
        workflow = build_rabi_amplitude_calibration_workflow(
            backend_name=snapshot.target_name,
            qubit=qubit,
            amplitude_grid=amplitude_grid,
            shots=shots,
            dt=dt,
        )

    return OpenPulseControlReadiness(
        snapshot=snapshot,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        workflow=workflow,
        required_ir_formats=required_ir_formats,
        required_native_features=required_native_features,
    )


def probe_aggregator_provider_capability(
    *,
    aggregator: str,
    provider: str,
    metadata_probe: ProviderMetadataProbe,
    ir_format: str | None = None,
    route_id: str | None = None,
    min_qubits: int | None = None,
) -> ProviderCapabilityDecision:
    """Resolve a route, collect provider metadata, and assess it without submission."""
    resolved = resolve_aggregator_provider_route(
        aggregator=aggregator,
        provider=provider,
        ir_format=ir_format,
        route_id=route_id,
    )
    snapshot = metadata_probe(resolved)
    return assess_provider_capability_snapshot(
        snapshot,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        route_id=resolved.route.route_id,
        required_ir_format=ir_format,
        min_qubits=min_qubits,
    )


def assess_provider_capability_snapshot(
    snapshot: ProviderCapabilitySnapshot,
    *,
    aggregator: str,
    provider: str,
    backend_id: str,
    route_id: str | None = None,
    required_ir_format: str | None = None,
    min_qubits: int | None = None,
) -> ProviderCapabilityDecision:
    """Assess route-level provider metadata without submitting work."""
    blockers: list[str] = []
    warnings: list[str] = []
    if route_id is not None and snapshot.route_id != route_id:
        blockers.append(f"route mismatch: expected {route_id}, got {snapshot.route_id}")
    if snapshot.aggregator != aggregator:
        blockers.append(f"aggregator mismatch: expected {aggregator}, got {snapshot.aggregator}")
    if snapshot.provider != provider:
        blockers.append(f"provider mismatch: expected {provider}, got {snapshot.provider}")
    if snapshot.backend_id != backend_id:
        blockers.append(f"backend mismatch: expected {backend_id}, got {snapshot.backend_id}")
    if snapshot.online is False:
        blockers.append("provider target is offline")
    if snapshot.online is None:
        warnings.append("provider target online status was not reported")
    if min_qubits is not None and snapshot.n_qubits < min_qubits:
        blockers.append(
            f"target has {snapshot.n_qubits} qubits but route requires at least {min_qubits}"
        )
    if required_ir_format is not None and required_ir_format not in snapshot.supported_ir_formats:
        blockers.append(f"target does not support required IR format: {required_ir_format}")

    if blockers:
        status: CapabilityDecisionStatus = "blocked"
    elif warnings:
        status = "unknown"
    else:
        status = "ready"
    return ProviderCapabilityDecision(
        snapshot=snapshot,
        status=status,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        required_ir_format=required_ir_format,
        min_qubits=min_qubits,
    )


@dataclass(frozen=True)
class RouteVerbSupport:
    """Support evidence for one operation on one provider route.

    Source-declared support and dated observed support are held in separate
    fields so a documented capability is never mistaken for a demonstrated one.
    ``None`` means unknown and is preserved as unknown; it is never narrowed to
    ``True`` or ``False`` to make an inventory row look complete.

    Parameters
    ----------
    verb
        Operation this record describes.
    declared
        Whether the route's own source material advertises the operation.
        ``None`` when the source says nothing about it.
    declared_source
        Where the declaration was read, required when ``declared`` is ``True``.
    declared_on
        ``YYYY-MM-DD`` date the declaration was read, required when ``declared``
        is ``True``.
    observed
        Whether conformance evidence demonstrated the operation. ``None`` when
        nothing has been observed.
    observed_on
        ``YYYY-MM-DD`` date of the observation, required when ``observed`` is
        ``True``.
    conformance_owner
        Repository-relative ``tests/test_*`` path that owns the demonstration,
        required when ``observed`` is ``True``.

    Raises
    ------
    ValueError
        If the verb is unknown, a date is not a valid Gregorian calendar date
        in ASCII ``YYYY-MM-DD`` format, a required
        provenance field is missing for an advertised or observed operation, or
        an observation contradicts an explicit non-declaration.

    """

    verb: RouteVerb
    declared: bool | None = None
    declared_source: str | None = None
    declared_on: str | None = None
    observed: bool | None = None
    observed_on: str | None = None
    conformance_owner: str | None = None

    def __post_init__(self) -> None:
        """Validate provenance completeness without inventing support values."""
        if self.verb not in ROUTE_VERBS:
            raise ValueError(f"unknown route verb: {self.verb!r}")
        for field_name in ("declared", "observed"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be True, False or None")
        if self.declared is True:
            _require_text(self.declared_source, "declared_source")
            _require_observation_date(self.declared_on, "declared_on")
        if self.observed is True:
            _require_observation_date(self.observed_on, "observed_on")
            _require_text(self.conformance_owner, "conformance_owner")
            owner = str(self.conformance_owner)
            if not owner.startswith("tests/test_"):
                raise ValueError("conformance_owner must be a tests/test_* path")
            if self.declared is False:
                raise ValueError(
                    "observed support contradicts an explicit non-declaration; "
                    "resolve the source before recording the observation"
                )
        for field_name in ("declared_on", "observed_on"):
            value = getattr(self, field_name)
            if value is not None:
                _require_observation_date(value, field_name)

    def to_dict(self) -> dict[str, Any]:
        """Return the verb record with unknown support preserved as null."""
        return {
            "verb": self.verb,
            "declared": self.declared,
            "declared_source": self.declared_source,
            "declared_on": self.declared_on,
            "observed": self.observed,
            "observed_on": self.observed_on,
            "conformance_owner": self.conformance_owner,
        }


@dataclass(frozen=True)
class ProviderRouteCatalogueEntry:
    """One inventory row keyed by provider, broker, device, modality and time.

    ``broker`` is ``None`` for a direct provider route, which keeps provider and
    broker identities distinct: the same provider and device reached through a
    broker is a different row, never a merged one.

    Parameters
    ----------
    route_id
        Identifier of the declared route this row inventories.
    provider
        Provider identity, independent of who hosts the access path.
    broker
        Aggregator hosting the route, or ``None`` for a direct route.
    device
        Backend identifier the route resolves to.
    modality
        Target family of the device.
    observed_at
        Valid Gregorian calendar date this row's evidence was assembled,
        in zero-padded ASCII ``YYYY-MM-DD`` format (years 0001 through 9999).
    verbs
        One record per operation, in ``ROUTE_VERBS`` order.
    submit_requires_approval
        Whether the declared route marks submission as approval-gated.

    Raises
    ------
    ValueError
        If an identity field is empty, the date is malformed or impossible, or the verb
        records are not exactly one per operation in canonical order.

    """

    route_id: str
    provider: str
    broker: str | None
    device: str
    modality: str
    observed_at: str
    verbs: tuple[RouteVerbSupport, ...]
    submit_requires_approval: bool = True

    def __post_init__(self) -> None:
        """Validate inventory identity and canonical verb coverage."""
        for field_name in ("route_id", "provider", "device", "modality"):
            _require_text(getattr(self, field_name), field_name)
        if self.broker is not None:
            _require_text(self.broker, "broker")
            if self.broker == DIRECT_AGGREGATOR:
                raise ValueError("a direct route must record broker as None")
        _require_observation_date(self.observed_at, "observed_at")
        if tuple(record.verb for record in self.verbs) != ROUTE_VERBS:
            raise ValueError(
                f"verbs must hold exactly one record per operation in {ROUTE_VERBS} order"
            )

    @property
    def inventory_key(self) -> tuple[str, str | None, str, str, str]:
        """The provider/broker/device/modality/time inventory key."""
        return (self.provider, self.broker, self.device, self.modality, self.observed_at)

    @property
    def is_direct(self) -> bool:
        """Whether the provider is reached without a broker."""
        return self.broker is None

    def support(self, verb: RouteVerb) -> RouteVerbSupport:
        """Return the record for one operation.

        Parameters
        ----------
        verb
            Operation to look up.

        Returns
        -------
        RouteVerbSupport
            The stored record, including unknown support as ``None``.

        Raises
        ------
        KeyError
            If the verb is not part of the canonical operation set.

        """
        for record in self.verbs:
            if record.verb == verb:
                return record
        raise KeyError(verb)

    @property
    def observed_verbs(self) -> tuple[RouteVerb, ...]:
        """Operations with demonstrated support, in canonical order."""
        return tuple(record.verb for record in self.verbs if record.observed is True)

    @property
    def unverified(self) -> bool:
        """Whether no operation on this route has been demonstrated."""
        return not self.observed_verbs

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready inventory row with its contract label."""
        return {
            "contract": ROUTE_CATALOGUE_CONTRACT,
            "route_id": self.route_id,
            "provider": self.provider,
            "broker": self.broker,
            "device": self.device,
            "modality": self.modality,
            "observed_at": self.observed_at,
            "submit_requires_approval": self.submit_requires_approval,
            "unverified": self.unverified,
            "verbs": [record.to_dict() for record in self.verbs],
        }


def build_provider_route_catalogue(
    *,
    observed_at: str,
    routes: Sequence[AggregatorProviderRoute] | None = None,
    evidence: Mapping[str, Sequence[RouteVerbSupport]] | None = None,
) -> tuple[ProviderRouteCatalogueEntry, ...]:
    """Inventory declared provider routes without contacting any provider.

    Identity comes from the declared route table. Support comes only from
    explicitly supplied evidence: a route with no evidence yields a row whose
    every operation is unknown, never one that is assumed available. No network
    call, credential read or submission occurs.

    Parameters
    ----------
    observed_at
        ``YYYY-MM-DD`` date recorded on every row of this inventory.
    routes
        Declared routes to inventory. Defaults to the built-in route table.
    evidence
        Per-route support records, keyed by ``route_id``. Records for an unknown
        route or a duplicated verb are rejected rather than ignored.

    Returns
    -------
    tuple of ProviderRouteCatalogueEntry
        One row per declared route, in the order the routes were supplied.

    Raises
    ------
    ValueError
        If the date is malformed, the routes contain a duplicate ``route_id``,
        or the evidence names an unknown route or repeats a verb.

    """
    _require_observation_date(observed_at, "observed_at")
    declared_routes = (
        tuple(routes) if routes is not None else built_in_aggregator_provider_routes()
    )
    route_ids = [route.route_id for route in declared_routes]
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("routes must not repeat a route_id")
    supplied = dict(evidence or {})
    unknown_routes = sorted(set(supplied) - set(route_ids))
    if unknown_routes:
        raise ValueError(f"evidence names routes absent from the inventory: {unknown_routes}")

    entries: list[ProviderRouteCatalogueEntry] = []
    for route in declared_routes:
        records = tuple(supplied.get(route.route_id, ()))
        seen = [record.verb for record in records]
        if len(set(seen)) != len(seen):
            raise ValueError(f"{route.route_id}: evidence repeats a verb")
        by_verb = {record.verb: record for record in records}
        entries.append(
            ProviderRouteCatalogueEntry(
                route_id=route.route_id,
                provider=route.provider,
                broker=None if route.aggregator == DIRECT_AGGREGATOR else route.aggregator,
                device=route.backend_id,
                modality=route.target_family,
                observed_at=observed_at,
                verbs=tuple(
                    by_verb.get(verb, RouteVerbSupport(verb=verb)) for verb in ROUTE_VERBS
                ),
                submit_requires_approval=route.submit_requires_approval,
            )
        )
    return tuple(entries)


def _require_observation_date(value: Any, field_name: str) -> None:
    _require_text(value, field_name)
    if not _OBSERVATION_DATE_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be an ISO YYYY-MM-DD observation date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid calendar date") from exc


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty text")


def _require_string_tuple(value: tuple[str, ...], field_name: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field_name} must be a tuple of non-empty strings")


__all__ = [
    "DIRECT_AGGREGATOR",
    "ROUTE_CATALOGUE_CONTRACT",
    "ROUTE_VERBS",
    "CapabilityDecisionStatus",
    "OpenPulseControlReadiness",
    "ProviderCapabilityDecision",
    "ProviderCapabilitySnapshot",
    "ProviderMetadataProbe",
    "ProviderRouteCatalogueEntry",
    "RouteVerb",
    "RouteVerbSupport",
    "assess_provider_capability_snapshot",
    "build_openpulse_control_readiness",
    "build_provider_route_catalogue",
    "probe_aggregator_provider_capability",
]
