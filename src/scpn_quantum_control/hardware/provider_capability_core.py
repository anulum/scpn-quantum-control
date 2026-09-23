# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Core
"""Provider-neutral no-submit capability contracts and readiness decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, get_args

from .aggregators import (
    AggregatorProviderRoute,
    ResolvedAggregatorProviderRoute,
    built_in_aggregator_provider_routes,
    resolve_aggregator_provider_route,
)
from .hal import BackendProfile, built_in_backend_profiles
from .openpulse_control import (
    OpenPulseCalibrationWorkflow,
    build_rabi_amplitude_calibration_workflow,
)
from .provider_route_configuration import provider_route_credential_refs

CapabilityDecisionStatus = Literal["ready", "blocked", "unknown"]

RouteVerb = Literal["metadata", "compile", "submit", "retrieve", "cancel", "result_formats"]
"""Operations a provider route can advertise, inventoried independently."""

ROUTE_VERBS: tuple[RouteVerb, ...] = get_args(RouteVerb)

DIRECT_AGGREGATOR = "direct"
"""Aggregator label meaning the provider is reached without a broker."""

ROUTE_CATALOGUE_CONTRACT = "provider_route_catalogue.v2"

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
    max_calibration_age_seconds: float | None = None
    calibration_age_seconds: float | None = None
    freshness_as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the provider capability decision."""
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "required_ir_format": self.required_ir_format,
            "min_qubits": self.min_qubits,
            "no_submit": self.no_submit,
            "calibration_freshness": {
                "evaluated": self.max_calibration_age_seconds is not None,
                "max_age_seconds": self.max_calibration_age_seconds,
                "observed_age_seconds": self.calibration_age_seconds,
                "as_of": self.freshness_as_of,
            },
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
    max_calibration_age_seconds: float | None = None,
    as_of: datetime | None = None,
) -> ProviderCapabilityDecision:
    """Resolve and assess metadata without submission, with optional freshness gate."""
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
        max_calibration_age_seconds=max_calibration_age_seconds,
        as_of=as_of,
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
    max_calibration_age_seconds: float | None = None,
    as_of: datetime | None = None,
) -> ProviderCapabilityDecision:
    """Assess route metadata, optionally requiring a fresh calibration timestamp.

    The caller supplies an explicit timezone-aware ``as_of`` with a finite,
    non-negative maximum age. A readiness result without this gate does not
    assert calibration freshness. This function never submits provider work.
    """
    if max_calibration_age_seconds is None:
        if as_of is not None:
            raise ValueError("as_of requires max_calibration_age_seconds")
    else:
        if (
            isinstance(max_calibration_age_seconds, bool)
            or not isinstance(max_calibration_age_seconds, int | float)
            or not math.isfinite(max_calibration_age_seconds)
            or max_calibration_age_seconds < 0
        ):
            raise ValueError("max_calibration_age_seconds must be finite and non-negative")
        if not isinstance(as_of, datetime) or as_of.utcoffset() is None:
            raise ValueError("as_of must be an explicit timezone-aware datetime")
    blockers: list[str] = []
    warnings: list[str] = []
    calibration_age_seconds: float | None = None
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
    if max_calibration_age_seconds is not None:
        assert as_of is not None
        timestamp = snapshot.calibration_timestamp
        if timestamp is None:
            blockers.append("calibration timestamp is missing")
        else:
            try:
                calibration_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                blockers.append("calibration timestamp is invalid")
            else:
                if calibration_at.utcoffset() is None:
                    blockers.append("calibration timestamp must include a timezone")
                else:
                    calibration_age_seconds = (as_of - calibration_at).total_seconds()
                    if calibration_age_seconds < 0:
                        blockers.append("calibration timestamp is in the future")
                    elif calibration_age_seconds > max_calibration_age_seconds:
                        blockers.append("calibration timestamp is stale")

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
        max_calibration_age_seconds=max_calibration_age_seconds,
        calibration_age_seconds=calibration_age_seconds,
        freshness_as_of=as_of.astimezone(UTC).isoformat() if as_of is not None else None,
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
        Canonical repository-relative ``tests/test_*.py`` path, required for
        declared or observed positive support. This record validates its shape;
        catalogue construction resolves it against explicit route/verb owners.

    Raises
    ------
    ValueError
        If the verb is unknown, a date is not a valid Gregorian calendar date
        in ASCII ``YYYY-MM-DD`` format, a required
        provenance field is missing for an advertised or observed operation, or
        an observation contradicts an explicit non-declaration.

    Notes
    -----
    This is a supplied assertion, not a test execution receipt. Neither a valid
    record nor a resolvable owner certifies a provider run or authorises hardware.

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
            if self.declared is False:
                raise ValueError(
                    "observed support contradicts an explicit non-declaration; "
                    "resolve the source before recording the observation"
                )
        for field_name in ("declared_on", "observed_on"):
            value = getattr(self, field_name)
            if value is not None:
                _require_observation_date(value, field_name)
        if self.declared is True or self.observed is True:
            _require_text(self.conformance_owner, "conformance_owner")
        if self.conformance_owner is not None and not re.fullmatch(
            r"tests/test_[A-Za-z0-9_]+\.py", self.conformance_owner
        ):
            raise ValueError("conformance_owner must be a tests/test_* path ending in .py")

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
        Execution modality from the HAL backend profile, not a provider name.
    observed_at
        Valid Gregorian calendar date this row's evidence was assembled,
        in zero-padded ASCII ``YYYY-MM-DD`` format (years 0001 through 9999).
    verbs
        One record per operation, in ``ROUTE_VERBS`` order.
    submit_requires_approval
        Whether the declared route marks submission as approval-gated.
    sdk_package
        Declared adapter dependency; None means unknown, not installed.
    adapter_module
        Declared adapter owner, without importing or instantiating it.
    credential_configuration_refs
        Source locators for configured client/factory/credential parameters.
        None means unknown; no credential values or authentication status.
    target_family
        Route's target-family label, retained separately from modality.
    conformance_root
        Absolute source checkout root used to resolve positive support owners.
        Required only for positive support; never inferred from the working
        directory. Consumed at construction, not retained or exported.
    conformance_owners
        Trusted caller's registry mapping ``(route_id, verb)`` to its test owner.
        Every positive record must match this registry and a regular test file
        under ``conformance_root``. Consumed at construction, not exported.

    Raises
    ------
    ValueError
        If an identity field is empty, the date is malformed or impossible, or the verb
        records are not exactly one per operation in canonical order, or a
        positive record lacks an explicitly registered, resolvable owner.
        Mutable verb containers and non-boolean approval flags are refused.

    Notes
    -----
    Owner resolution is a construction-time provenance check, not a test runner
    or hardware readiness verdict. The caller must supply a reviewed registry
    independently of the evidence being checked. Do not derive it from the
    incoming claims. Reconstructing a positive row requires this context again.

    """

    route_id: str
    provider: str
    broker: str | None
    device: str
    modality: str
    observed_at: str
    verbs: tuple[RouteVerbSupport, ...]
    submit_requires_approval: bool = True
    sdk_package: str | None = field(default=None, kw_only=True)
    adapter_module: str | None = field(default=None, kw_only=True)
    credential_configuration_refs: tuple[str, ...] | None = field(default=None, kw_only=True)
    target_family: str | None = field(default=None, kw_only=True)
    conformance_root: InitVar[Path | None] = None
    conformance_owners: InitVar[Mapping[tuple[str, str], str] | None] = None

    def __post_init__(
        self,
        conformance_root: Path | None,
        conformance_owners: Mapping[tuple[str, str], str] | None,
    ) -> None:
        """Validate inventory identity and canonical verb coverage."""
        for field_name in ("route_id", "provider", "device", "modality"):
            _require_text(getattr(self, field_name), field_name)
        for field_name in ("sdk_package", "adapter_module", "target_family"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if self.credential_configuration_refs is not None:
            _require_string_tuple(
                self.credential_configuration_refs, "credential_configuration_refs"
            )
        if self.broker is not None:
            _require_text(self.broker, "broker")
            if self.broker == DIRECT_AGGREGATOR:
                raise ValueError("a direct route must record broker as None")
        _require_observation_date(self.observed_at, "observed_at")
        if not isinstance(self.submit_requires_approval, bool):
            raise ValueError("submit_requires_approval must be boolean")
        if not isinstance(self.verbs, tuple) or any(
            not isinstance(record, RouteVerbSupport) for record in self.verbs
        ):
            raise ValueError("verbs must be an immutable tuple of RouteVerbSupport records")
        if tuple(record.verb for record in self.verbs) != ROUTE_VERBS:
            raise ValueError(
                f"verbs must hold exactly one record per operation in {ROUTE_VERBS} order"
            )
        for record in self.verbs:
            if record.declared is True or record.observed is True:
                _resolve_conformance_owner(
                    self.route_id, record, conformance_root, conformance_owners
                )

    @property
    def inventory_key(self) -> tuple[str, str | None, str, str, str]:
        """Provider/broker/device/modality/time grouping; aliases may share it."""
        return (self.provider, self.broker, self.device, self.modality, self.observed_at)

    @property
    def route_key(self) -> tuple[str, str, str | None, str, str, str]:
        """Unambiguous row identity retaining the declared access route.

        Two route aliases may share the same backend/profile inventory key.
        Prefixing that key with route_id preserves separate evidence bindings
        without inventing a distinct physical device or merging observations.

        """
        return (self.route_id, *self.inventory_key)

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
        """Recorded observations with resolved owners, not certified device runs."""
        return tuple(record.verb for record in self.verbs if record.observed is True)

    @property
    def unverified(self) -> bool:
        """Whether the row lacks observations; False is not hardware readiness."""
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
            "sdk_package": self.sdk_package,
            "adapter_module": self.adapter_module,
            "credential_configuration_refs": (
                list(self.credential_configuration_refs)
                if self.credential_configuration_refs is not None
                else None
            ),
            "target_family": self.target_family,
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
    conformance_root: Path | None = None,
    conformance_owners: Mapping[tuple[str, str], str] | None = None,
    profiles: Sequence[BackendProfile] | None = None,
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
    conformance_root
        Absolute source root for positive-support owner resolution. No default
        checkout is assumed; unknown-only inventories need no filesystem access.
    conformance_owners
        Independently reviewed ``(route_id, verb)`` to test-path registry. Positive
        claims must match its binding and resolve under ``conformance_root``.
        Resolution does not run tests, certify a provider, or grant approval.
    profiles
        Authoritative HAL profiles, defaulting to built-ins. Every route must
        resolve a unique backend profile with the same SDK. Supply explicit
        profiles for custom backends; modality is never inferred from a name.

    Returns
    -------
    tuple of ProviderRouteCatalogueEntry
        One row per declared route, in the order the routes were supplied.

    Raises
    ------
    ValueError
        If the date is malformed, the routes contain a duplicate ``route_id``,
        or the evidence names an unknown route, repeats a verb, or has a positive
        claim without a registered and resolvable conformance owner.
        Also raised for missing/duplicate backend profiles or SDK disagreement.

    """
    _require_observation_date(observed_at, "observed_at")
    profile_rows = tuple(profiles) if profiles is not None else built_in_backend_profiles()
    profile_by_id = {profile.backend_id: profile for profile in profile_rows}
    if len(profile_by_id) != len(profile_rows):
        raise ValueError("profiles must not repeat a backend_id")
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
        profile = profile_by_id.get(route.backend_id)
        if profile is None:
            raise ValueError(f"{route.route_id}: missing backend profile {route.backend_id}")
        if route.sdk_package != profile.sdk_package:
            raise ValueError(f"{route.route_id}: SDK disagrees with backend profile")
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
                modality=profile.modality,
                sdk_package=route.sdk_package,
                adapter_module=route.adapter_module,
                credential_configuration_refs=provider_route_credential_refs(route.adapter_module),
                target_family=route.target_family,
                observed_at=observed_at,
                verbs=tuple(
                    by_verb.get(verb, RouteVerbSupport(verb=verb)) for verb in ROUTE_VERBS
                ),
                submit_requires_approval=route.submit_requires_approval,
                conformance_root=conformance_root,
                conformance_owners=conformance_owners,
            )
        )
    return tuple(entries)


def _resolve_conformance_owner(
    route_id: str,
    record: RouteVerbSupport,
    root: Path | None,
    owners: Mapping[tuple[str, str], str] | None,
) -> None:
    if root is None or not root.is_absolute() or not root.is_dir():
        raise ValueError("conformance_root must be an explicit absolute source directory")
    owner = record.conformance_owner
    if owner is None or owners is None or owners.get((route_id, record.verb)) != owner:
        raise ValueError(f"{route_id}/{record.verb}: conformance_owner is not registered")
    candidate = root.resolve() / owner
    if not candidate.is_file() or candidate.resolve() != candidate:
        raise ValueError("conformance_owner must resolve to a regular non-symlink test file")


def _require_observation_date(value: Any, field_name: str) -> None:
    _require_text(value, field_name)
    if not _OBSERVATION_DATE_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be an ISO YYYY-MM-DD observation date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid calendar date") from exc


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
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
