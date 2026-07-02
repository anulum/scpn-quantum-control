# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PennyLane Provider-Plugin Evidence
"""PennyLane provider-plugin gradient artefacts and fail-closed route matrix."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

PENNYLANE_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC = "2026-06-27T00:00:00Z"


@dataclass(frozen=True)
class PennyLanePluginMatrixRoute:
    """One route in the bounded PennyLane plugin/provider parity matrix."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate route metadata for a fail-closed plugin matrix."""
        object.__setattr__(self, "name", _normalise_metadata_text("route name", self.name))
        status = _normalise_metadata_text("route status", self.status)
        if status not in {"passed", "blocked", "failed"}:
            raise ValueError("route status must be one of passed, blocked, or failed")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "reason",
            _normalise_metadata_text("route reason", self.reason),
        )
        object.__setattr__(
            self,
            "requires",
            tuple(_normalise_metadata_text("route requirement", item) for item in self.requires),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PennyLane plugin route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PennyLaneProviderPluginExecutionArtifact:
    """Validated PennyLane provider-plugin execution evidence.

    Provider evidence records the PennyLane interface and differentiation
    method used for the captured provider route. Those fields are part of the
    evidence chain, so provider-gradient parity must cite the same values before
    the route can pass.
    """

    artifact_id: str
    plugin_name: str
    provider_name: str
    device_name: str
    backend_name: str
    circuit_fingerprint: str
    execution_mode: str
    interface: str
    diff_method: str
    shots: int | None
    result_digest: str
    metadata_digest: str
    hardware_execution: bool = False
    raw_result_replay_artifact_id: str | None = None

    def __post_init__(self) -> None:
        """Validate provider-plugin execution metadata without hardware claims."""
        for field_name in (
            "artifact_id",
            "plugin_name",
            "provider_name",
            "device_name",
            "backend_name",
            "circuit_fingerprint",
            "execution_mode",
            "interface",
            "diff_method",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_metadata_text(field_name, getattr(self, field_name)),
            )
        if self.shots is not None and (
            isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0
        ):
            raise ValueError("shots must be positive when provided")
        if self.hardware_execution:
            raise ValueError(
                "provider-plugin execution artefacts must not claim hardware execution"
            )
        _validate_provider_plugin_execution_mode(self.execution_mode)
        if self.raw_result_replay_artifact_id is not None:
            object.__setattr__(
                self,
                "raw_result_replay_artifact_id",
                _normalise_metadata_text(
                    "raw_result_replay_artifact_id",
                    self.raw_result_replay_artifact_id,
                ),
            )
        for field_name in ("result_digest", "metadata_digest"):
            object.__setattr__(
                self,
                field_name,
                _normalise_sha256_digest(field_name, getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready provider-plugin execution metadata."""
        return {
            "artifact_id": self.artifact_id,
            "plugin_name": self.plugin_name,
            "provider_name": self.provider_name,
            "device_name": self.device_name,
            "backend_name": self.backend_name,
            "circuit_fingerprint": self.circuit_fingerprint,
            "execution_mode": self.execution_mode,
            "interface": self.interface,
            "diff_method": self.diff_method,
            "shots": self.shots,
            "result_digest": self.result_digest,
            "metadata_digest": self.metadata_digest,
            "hardware_execution": self.hardware_execution,
            "raw_result_replay_artifact_id": self.raw_result_replay_artifact_id,
        }


@dataclass(frozen=True)
class PennyLaneProviderGradientParityArtifact:
    """Validated PennyLane provider-plugin gradient parity evidence.

    Parity artefacts must match the referenced provider execution artefact on
    interface, differentiation method, device identity, circuit fingerprint, and
    shot policy before the provider-gradient route can pass.
    """

    artifact_id: str
    provider_execution_artifact_id: str
    plugin_name: str
    provider_name: str
    device_name: str
    backend_name: str
    circuit_fingerprint: str
    interface: str
    diff_method: str
    gradient_digest: str
    reference_gradient_digest: str
    max_abs_error: float
    l2_error: float
    tolerance: float
    shots: int | None
    replay_artifact_id: str
    hardware_execution: bool = False

    def __post_init__(self) -> None:
        """Validate same-circuit provider-plugin gradient parity metadata."""
        for field_name in (
            "artifact_id",
            "provider_execution_artifact_id",
            "plugin_name",
            "provider_name",
            "device_name",
            "backend_name",
            "circuit_fingerprint",
            "interface",
            "diff_method",
            "replay_artifact_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_metadata_text(field_name, getattr(self, field_name)),
            )
        if self.shots is not None and (
            isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0
        ):
            raise ValueError("shots must be positive when provided")
        if self.hardware_execution:
            raise ValueError(
                "provider-gradient parity artefacts must not claim hardware execution"
            )
        for field_name in ("gradient_digest", "reference_gradient_digest"):
            object.__setattr__(
                self,
                field_name,
                _normalise_sha256_digest(field_name, getattr(self, field_name)),
            )
        max_abs_error = _as_non_negative_finite_metric("max_abs_error", self.max_abs_error)
        l2_error = _as_non_negative_finite_metric("l2_error", self.l2_error)
        tolerance = _as_non_negative_finite_metric("tolerance", self.tolerance)
        if max_abs_error > tolerance:
            raise ValueError("max_abs_error must not exceed tolerance")
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "l2_error", l2_error)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready provider-plugin gradient parity metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_execution_artifact_id": self.provider_execution_artifact_id,
            "plugin_name": self.plugin_name,
            "provider_name": self.provider_name,
            "device_name": self.device_name,
            "backend_name": self.backend_name,
            "circuit_fingerprint": self.circuit_fingerprint,
            "interface": self.interface,
            "diff_method": self.diff_method,
            "gradient_digest": self.gradient_digest,
            "reference_gradient_digest": self.reference_gradient_digest,
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "shots": self.shots,
            "replay_artifact_id": self.replay_artifact_id,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class PennyLaneHardwarePluginExecutionArtifact:
    """Ticketed PennyLane hardware-plugin execution evidence.

    The artefact records a captured live hardware-plugin run without promoting
    benchmark or provider-exceedance claims. Calibration freshness is expressed
    by UTC capture and expiry timestamps; construction rejects inverted windows,
    and ``run_pennylane_plugin_matrix`` rejects stale calibration at the review
    cutoff before opening the hardware-plugin route.
    """

    artifact_id: str
    plugin_name: str
    provider_name: str
    device_name: str
    backend_name: str
    circuit_fingerprint: str
    execution_mode: str
    shots: int
    live_execution_ticket: str
    provider_allowlist_id: str
    shot_budget_id: str
    hardware_evidence_id: str
    result_digest: str
    raw_counts_digest: str
    calibration_snapshot_digest: str
    calibration_captured_at_utc: str
    calibration_valid_until_utc: str
    metadata_digest: str
    hardware_execution: bool = True

    def __post_init__(self) -> None:
        """Validate ticketed hardware-plugin execution evidence metadata."""
        for field_name in (
            "artifact_id",
            "plugin_name",
            "provider_name",
            "device_name",
            "backend_name",
            "circuit_fingerprint",
            "execution_mode",
            "live_execution_ticket",
            "provider_allowlist_id",
            "shot_budget_id",
            "hardware_evidence_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_metadata_text(field_name, getattr(self, field_name)),
            )
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        if not self.hardware_execution:
            raise ValueError("hardware-plugin execution artefacts must claim hardware execution")
        _validate_hardware_plugin_execution_mode(self.execution_mode)
        for field_name in (
            "result_digest",
            "raw_counts_digest",
            "calibration_snapshot_digest",
            "metadata_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_sha256_digest(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "calibration_captured_at_utc",
            _normalise_utc_timestamp(
                "calibration_captured_at_utc",
                self.calibration_captured_at_utc,
            ),
        )
        object.__setattr__(
            self,
            "calibration_valid_until_utc",
            _normalise_utc_timestamp(
                "calibration_valid_until_utc",
                self.calibration_valid_until_utc,
            ),
        )
        if _utc_timestamp(
            "calibration_valid_until_utc",
            self.calibration_valid_until_utc,
        ) <= _utc_timestamp(
            "calibration_captured_at_utc",
            self.calibration_captured_at_utc,
        ):
            raise ValueError(
                "calibration_valid_until_utc must be after calibration_captured_at_utc"
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready hardware-plugin execution metadata."""
        return {
            "artifact_id": self.artifact_id,
            "plugin_name": self.plugin_name,
            "provider_name": self.provider_name,
            "device_name": self.device_name,
            "backend_name": self.backend_name,
            "circuit_fingerprint": self.circuit_fingerprint,
            "execution_mode": self.execution_mode,
            "shots": self.shots,
            "live_execution_ticket": self.live_execution_ticket,
            "provider_allowlist_id": self.provider_allowlist_id,
            "shot_budget_id": self.shot_budget_id,
            "hardware_evidence_id": self.hardware_evidence_id,
            "result_digest": self.result_digest,
            "raw_counts_digest": self.raw_counts_digest,
            "calibration_snapshot_digest": self.calibration_snapshot_digest,
            "calibration_captured_at_utc": self.calibration_captured_at_utc,
            "calibration_valid_until_utc": self.calibration_valid_until_utc,
            "metadata_digest": self.metadata_digest,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class PennyLaneProviderEvidenceBundle:
    """Validated PennyLane provider execution, parity, and hardware evidence bundle."""

    artifact_id: str
    provider_execution_artifact: PennyLaneProviderPluginExecutionArtifact
    captured_at_utc: str
    valid_until_utc: str
    provider_gradient_parity_artifact: PennyLaneProviderGradientParityArtifact | None = None
    hardware_execution_artifact: PennyLaneHardwarePluginExecutionArtifact | None = None
    claim_boundary: str = "pennylane_provider_evidence_bundle"

    def __post_init__(self) -> None:
        """Validate bundle identity, evidence chain, and freshness metadata."""
        object.__setattr__(
            self,
            "artifact_id",
            _normalise_metadata_text("artifact_id", self.artifact_id),
        )
        object.__setattr__(
            self,
            "captured_at_utc",
            _normalise_utc_timestamp("captured_at_utc", self.captured_at_utc),
        )
        object.__setattr__(
            self,
            "valid_until_utc",
            _normalise_utc_timestamp("valid_until_utc", self.valid_until_utc),
        )
        object.__setattr__(
            self,
            "claim_boundary",
            _normalise_metadata_text("claim_boundary", self.claim_boundary),
        )
        if _utc_timestamp("valid_until_utc", self.valid_until_utc) <= _utc_timestamp(
            "captured_at_utc",
            self.captured_at_utc,
        ):
            raise ValueError("valid_until_utc must be after captured_at_utc")
        _validate_provider_gradient_parity_pair(
            self.provider_execution_artifact,
            self.provider_gradient_parity_artifact,
        )
        _validate_provider_bundle_hardware_chain(
            self.provider_execution_artifact,
            self.hardware_execution_artifact,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PennyLane provider evidence bundle metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_execution_artifact": self.provider_execution_artifact.to_dict(),
            "provider_gradient_parity_artifact": (
                None
                if self.provider_gradient_parity_artifact is None
                else self.provider_gradient_parity_artifact.to_dict()
            ),
            "hardware_execution_artifact": (
                None
                if self.hardware_execution_artifact is None
                else self.hardware_execution_artifact.to_dict()
            ),
            "captured_at_utc": self.captured_at_utc,
            "valid_until_utc": self.valid_until_utc,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PennyLanePluginMatrixResult:
    """Fail-closed PennyLane plugin/provider parity matrix."""

    routes: tuple[PennyLanePluginMatrixRoute, ...]
    provider_execution_artifact: PennyLaneProviderPluginExecutionArtifact | None = None
    provider_gradient_parity_artifact: PennyLaneProviderGradientParityArtifact | None = None
    hardware_execution_artifact: PennyLaneHardwarePluginExecutionArtifact | None = None
    provider_evidence_bundle: PennyLaneProviderEvidenceBundle | None = None
    claim_boundary: str = "bounded_pennylane_plugin_matrix"

    @property
    def local_plugin_parity_ready(self) -> bool:
        """Return whether bounded local/default-qubit PennyLane routes pass."""
        return all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith(("default_qubit_", "phase_qnode_export_"))
        )

    @property
    def provider_plugin_execution_ready(self) -> bool:
        """Return whether provider-plugin execution artefacts are attached."""
        return self.route_status("provider_plugin_execution") == "passed"

    @property
    def hardware_plugin_execution_ready(self) -> bool:
        """Return whether live hardware-plugin execution artefacts are attached."""
        return all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith("hardware_plugin_")
        )

    @property
    def provider_plugin_gradient_parity_ready(self) -> bool:
        """Return whether provider-plugin gradient parity artefacts are attached."""
        return self.route_status("provider_plugin_gradient_parity") == "passed"

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether the matrix permits PennyLane provider-exceedance claims."""
        return all(route.status == "passed" for route in self.routes)

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return routes that still block PennyLane provider-exceedance claims."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named route, failing closed on unknown names."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PennyLane plugin route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PennyLane plugin/provider parity metadata."""
        return {
            "local_plugin_parity_ready": self.local_plugin_parity_ready,
            "provider_plugin_execution_ready": self.provider_plugin_execution_ready,
            "hardware_plugin_execution_ready": self.hardware_plugin_execution_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "provider_execution_artifact": (
                None
                if self.provider_execution_artifact is None
                else self.provider_execution_artifact.to_dict()
            ),
            "provider_gradient_parity_artifact": (
                None
                if self.provider_gradient_parity_artifact is None
                else self.provider_gradient_parity_artifact.to_dict()
            ),
            "hardware_execution_artifact": (
                None
                if self.hardware_execution_artifact is None
                else self.hardware_execution_artifact.to_dict()
            ),
            "provider_evidence_bundle": (
                None
                if self.provider_evidence_bundle is None
                else self.provider_evidence_bundle.to_dict()
            ),
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


def run_pennylane_plugin_matrix(
    *,
    provider_execution_artifact: PennyLaneProviderPluginExecutionArtifact | None = None,
    provider_gradient_parity_artifact: PennyLaneProviderGradientParityArtifact | None = None,
    hardware_execution_artifact: PennyLaneHardwarePluginExecutionArtifact | None = None,
    provider_evidence_bundle: PennyLaneProviderEvidenceBundle | None = None,
    evidence_freshness_as_of_utc: str = PENNYLANE_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC,
) -> PennyLanePluginMatrixResult:
    """Return a fail-closed PennyLane plugin/provider parity matrix.

    The current evidence covers bounded local ``default.qubit`` exact-state
    parity, metadata-preserving shot policy records, and registered
    Phase-QNode export through the local PennyLane route. Provider plugin
    execution, live hardware execution, and promotion evidence remain blocked
    until concrete artefacts are attached. Ticketed hardware artefacts must
    carry fresh calibration metadata for the supplied review cutoff.
    """
    if provider_evidence_bundle is not None:
        if (
            provider_execution_artifact is not None
            or provider_gradient_parity_artifact is not None
            or hardware_execution_artifact is not None
        ):
            raise ValueError(
                "provider_evidence_bundle cannot be combined with individual "
                "PennyLane provider artefacts"
            )
        _validate_provider_evidence_bundle_freshness(
            provider_evidence_bundle,
            as_of_utc=evidence_freshness_as_of_utc,
        )
        provider_execution_artifact = provider_evidence_bundle.provider_execution_artifact
        provider_gradient_parity_artifact = (
            provider_evidence_bundle.provider_gradient_parity_artifact
        )
        hardware_execution_artifact = provider_evidence_bundle.hardware_execution_artifact
    _validate_provider_gradient_parity_pair(
        provider_execution_artifact,
        provider_gradient_parity_artifact,
    )
    _validate_hardware_calibration_freshness(
        hardware_execution_artifact,
        as_of_utc=evidence_freshness_as_of_utc,
    )
    provider_execution_status = "passed" if provider_execution_artifact is not None else "blocked"
    provider_execution_reason = (
        "validated PennyLane provider-plugin execution artefact is attached"
        if provider_execution_artifact is not None
        else "PennyLane provider-plugin execution artefacts are not attached"
    )
    provider_gradient_status = (
        "passed" if provider_gradient_parity_artifact is not None else "blocked"
    )
    provider_gradient_reason = (
        "validated same-circuit provider-plugin gradient parity artefact is attached"
        if provider_gradient_parity_artifact is not None
        else "provider-plugin gradients need same-circuit provider execution and replay artefacts"
    )
    hardware_execution_status = "passed" if hardware_execution_artifact is not None else "blocked"
    hardware_execution_reason = (
        "validated ticketed PennyLane hardware-plugin execution artefact is attached"
        if hardware_execution_artifact is not None
        else "live PennyLane hardware plugin execution requires ticketed hardware evidence"
    )
    routes = (
        PennyLanePluginMatrixRoute(
            name="default_qubit_exact_state",
            status="passed",
            reason="bounded identical-circuit exact-state parity is available for local default.qubit",
        ),
        PennyLanePluginMatrixRoute(
            name="default_qubit_shot_policy_metadata",
            status="passed",
            reason="bounded local conversion records exact-state versus finite-shot shot policy metadata",
        ),
        PennyLanePluginMatrixRoute(
            name="phase_qnode_export_default_qubit",
            status="passed",
            reason="registered local Phase-QNode circuits export to PennyLane default.qubit QNodes",
        ),
        PennyLanePluginMatrixRoute(
            name="phase_qnode_import_supported_tapes",
            status="passed",
            reason="supported PennyLane QuantumScript imports round-trip through the bounded importer",
        ),
        PennyLanePluginMatrixRoute(
            name="provider_plugin_execution",
            status=provider_execution_status,
            reason=provider_execution_reason,
            requires=(
                "plugin_inventory",
                "provider_plugin_adapter",
                "provider_execution_artifact",
                "device_metadata_artifact",
                "interface_metadata",
                "diff_method_metadata",
            ),
        ),
        PennyLanePluginMatrixRoute(
            name="hardware_plugin_execution",
            status=hardware_execution_status,
            reason=hardware_execution_reason,
            requires=(
                "live_ticket",
                "provider_allowlist",
                "shot_budget",
                "hardware_evidence_id",
            ),
        ),
        PennyLanePluginMatrixRoute(
            name="provider_plugin_gradient_parity",
            status=provider_gradient_status,
            reason=provider_gradient_reason,
            requires=(
                "same_circuit_provider_artifact",
                "matching_interface_metadata",
                "matching_diff_method_metadata",
                "raw_result_replay",
                "gradient_parity_artifact",
            ),
        ),
        PennyLanePluginMatrixRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion requires isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PennyLanePluginMatrixResult(
        routes=routes,
        provider_execution_artifact=provider_execution_artifact,
        provider_gradient_parity_artifact=provider_gradient_parity_artifact,
        hardware_execution_artifact=hardware_execution_artifact,
        provider_evidence_bundle=provider_evidence_bundle,
    )


def _normalise_metadata_text(field_name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{field_name} must not contain control characters")
    return text


def _normalise_sha256_digest(field_name: str, value: object) -> str:
    digest = str(value)
    hex_digest = digest.removeprefix("sha256:")
    if not (
        digest.startswith("sha256:")
        and len(hex_digest) == 64
        and all(char in "0123456789abcdefABCDEF" for char in hex_digest)
    ):
        raise ValueError(f"{field_name} must be a sha256:<64-hex> digest")
    return f"sha256:{hex_digest.lower()}"


def _as_non_negative_finite_metric(name: str, value: float) -> float:
    metric = float(value)
    if metric < 0.0 or not np.isfinite(metric):
        raise ValueError(f"{name} must be finite and non-negative")
    return metric


def _validate_provider_gradient_parity_pair(
    provider_execution_artifact: PennyLaneProviderPluginExecutionArtifact | None,
    provider_gradient_parity_artifact: PennyLaneProviderGradientParityArtifact | None,
) -> None:
    if provider_gradient_parity_artifact is None:
        return
    if provider_execution_artifact is None:
        raise ValueError(
            "provider gradient parity artefacts require a provider execution artefact"
        )
    expected = {
        "provider_execution_artifact_id": provider_execution_artifact.artifact_id,
        "plugin_name": provider_execution_artifact.plugin_name,
        "provider_name": provider_execution_artifact.provider_name,
        "device_name": provider_execution_artifact.device_name,
        "backend_name": provider_execution_artifact.backend_name,
        "circuit_fingerprint": provider_execution_artifact.circuit_fingerprint,
        "interface": provider_execution_artifact.interface,
        "diff_method": provider_execution_artifact.diff_method,
        "shots": provider_execution_artifact.shots,
    }
    for field_name, expected_value in expected.items():
        actual_value = getattr(provider_gradient_parity_artifact, field_name)
        if actual_value != expected_value:
            raise ValueError(
                "provider gradient parity artefact does not match provider execution "
                f"artefact field {field_name}"
            )


def _validate_provider_bundle_hardware_chain(
    provider_execution_artifact: PennyLaneProviderPluginExecutionArtifact,
    hardware_execution_artifact: PennyLaneHardwarePluginExecutionArtifact | None,
) -> None:
    if hardware_execution_artifact is None:
        return
    expected = {
        "provider_name": provider_execution_artifact.provider_name,
        "circuit_fingerprint": provider_execution_artifact.circuit_fingerprint,
        "shots": provider_execution_artifact.shots,
    }
    for field_name, expected_value in expected.items():
        actual_value = getattr(hardware_execution_artifact, field_name)
        if actual_value != expected_value:
            raise ValueError(
                "hardware execution artefact does not match provider execution "
                f"artefact field hardware_execution_artifact.{field_name}"
            )


def _normalise_utc_timestamp(field_name: str, value: object) -> str:
    timestamp = _utc_timestamp(field_name, value)
    return timestamp.isoformat().replace("+00:00", "Z")


def _utc_timestamp(field_name: str, value: object) -> datetime:
    text = _normalise_metadata_text(field_name, value)
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    return timestamp.astimezone(timezone.utc).replace(microsecond=0)


def _validate_provider_evidence_bundle_freshness(
    provider_evidence_bundle: PennyLaneProviderEvidenceBundle,
    *,
    as_of_utc: str,
) -> None:
    valid_until = _utc_timestamp(
        "provider_evidence_bundle.valid_until_utc",
        provider_evidence_bundle.valid_until_utc,
    )
    if valid_until <= _utc_timestamp("evidence_freshness_as_of_utc", as_of_utc):
        raise ValueError("provider_evidence_bundle.valid_until_utc is stale for the review cutoff")


def _validate_hardware_calibration_freshness(
    hardware_execution_artifact: PennyLaneHardwarePluginExecutionArtifact | None,
    *,
    as_of_utc: str,
) -> None:
    if hardware_execution_artifact is None:
        return
    valid_until = _utc_timestamp(
        "hardware_execution_artifact.calibration_valid_until_utc",
        hardware_execution_artifact.calibration_valid_until_utc,
    )
    if valid_until <= _utc_timestamp("evidence_freshness_as_of_utc", as_of_utc):
        raise ValueError(
            "hardware_execution_artifact.calibration_valid_until_utc is stale "
            "for the review cutoff"
        )


def _validate_provider_plugin_execution_mode(execution_mode: str) -> None:
    mode = execution_mode.lower()
    if mode in {"hardware", "qpu", "live_qpu", "provider_qpu"}:
        raise ValueError("execution_mode must not imply hardware execution")
    if "provider" not in mode:
        raise ValueError("execution_mode must identify provider-plugin execution")


def _validate_hardware_plugin_execution_mode(execution_mode: str) -> None:
    mode = execution_mode.lower()
    if not any(token in mode for token in ("hardware", "qpu", "live")):
        raise ValueError("execution_mode must identify live hardware execution")
    if "simulator" in mode or "offline" in mode or "replay" in mode:
        raise ValueError("execution_mode must not identify simulator or replay execution")


__all__ = [
    "PennyLaneHardwarePluginExecutionArtifact",
    "PennyLaneProviderEvidenceBundle",
    "PennyLanePluginMatrixResult",
    "PennyLanePluginMatrixRoute",
    "PennyLaneProviderGradientParityArtifact",
    "PennyLaneProviderPluginExecutionArtifact",
    "run_pennylane_plugin_matrix",
]
