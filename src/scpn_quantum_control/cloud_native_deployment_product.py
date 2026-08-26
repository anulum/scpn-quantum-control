# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cloud-native deployment boundary product
"""Fail-closed **cloud-native deployment boundary** product surface.

Productises documented, fail-closed cloud deploy patterns for workers/batch over
ambient :mod:`scpn_quantum_control.deployment.cloud_native`:

* versioned deployment-pattern catalogue (batch worker, stable-core gate,
  offline research) with hardware-safety / QPU-compute no always-on QPU posture;
* threat-model rows (secret leakage, cost, live cluster create);
* dry-run manifest generation via ambient
  :func:`~scpn_quantum_control.deployment.cloud_native.generate_cloud_manifests`;
* refuse secret-like env injection, always-on QPU deploy claims, and live
  cluster/API contact invent-green.

Does **not** create clusters, load credentials, contact cloud APIs, or submit
QPU jobs (ambient claim boundary preserved).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .deployment.cloud_native import (
    CloudDeploymentSpec,
    ContainerResources,
    generate_cloud_manifests,
)

DeploymentPatternKind = Literal[
    "batch_worker",
    "stable_core_gate",
    "offline_research",
]
"""Cloud-native deployment pattern kinds on the product catalogue."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges for deployment product rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

ThreatKind = Literal[
    "secret_leakage",
    "always_on_qpu",
    "live_cluster_create",
    "credential_loading",
    "unbounded_cost",
]
"""Threat-model kinds for cloud deploy boundary."""

CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA: Final[str] = "cloud_native_deployment_product.v2"
"""JSON schema identifier for serialised product payloads."""

CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY: Final[str] = (
    "Cloud-native deployment boundary product surface only; catalogues batch/"
    "worker deploy patterns and threat-model rows; dry-run manifest generation "
    "via ambient deployment.cloud_native; refuse secret-like env, always-on QPU "
    "deploy claims, live cluster create, and credential loading; composes "
    "hardware-safe execution with dry-run compute planning; fuller enterprise "
    "packaging and operations runbooks remain open"
)
"""Shared claim boundary for cloud-native deployment product payloads."""

# Default offline image reference for dry-run demos (not a live pull claim).
_DEMO_IMAGE: Final[str] = "ghcr.io/anulum/scpn-quantum-control:local-research"
_DEMO_NAME: Final[str] = "scpn-batch-worker"


@dataclass(frozen=True, slots=True)
class DeploymentPatternRow:
    """One cloud-native deployment pattern catalogue row.

    Attributes
    ----------
    pattern_id
        Stable pattern identifier.
    kind
        Pattern kind enum.
    title
        Human-readable title.
    summary
        Short description.
    default_command
        Default container command tokens.
    allows_always_on_qpu
        Must remain False on product surface.
    secret_env_allowed
        Must remain False.
    live_cluster_create
        Must remain False.
    hardware_safety_pointer
        Hardware-safe no-submit policy pointer.
    compute_plan_pointer
        Dry-run QPU compute-plan posture pointer.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    pattern_id: str
    kind: DeploymentPatternKind
    title: str
    summary: str
    default_command: tuple[str, ...]
    allows_always_on_qpu: bool = False
    secret_env_allowed: bool = False
    live_cluster_create: bool = False
    hardware_safety_pointer: str = "hardware_safe_execution.cloud_deploy_no_submit"
    compute_plan_pointer: str = "qpu_compute.dry_run_default"
    support_posture: SupportPosture = "policy_only"
    as_of: str = "2026-07-24"
    claim_boundary: str = CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate deployment pattern invariants."""
        if not self.pattern_id or not self.pattern_id.strip():
            raise ValueError("pattern_id must be non-empty")
        if self.kind not in {"batch_worker", "stable_core_gate", "offline_research"}:
            raise ValueError(f"unknown pattern kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.default_command or any(not str(item).strip() for item in self.default_command):
            raise ValueError("default_command must contain non-empty items")
        if self.allows_always_on_qpu:
            raise ValueError("allows_always_on_qpu must be False on product surface")
        if self.secret_env_allowed:
            raise ValueError("secret_env_allowed must be False on product surface")
        if self.live_cluster_create:
            raise ValueError("live_cluster_create must be False on product surface")
        if not self.hardware_safety_pointer or not self.hardware_safety_pointer.strip():
            raise ValueError("hardware_safety_pointer must be non-empty")
        if not self.compute_plan_pointer or not self.compute_plan_pointer.strip():
            raise ValueError("compute_plan_pointer must be non-empty")
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
            "pattern_id": self.pattern_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "default_command": list(self.default_command),
            "allows_always_on_qpu": self.allows_always_on_qpu,
            "secret_env_allowed": self.secret_env_allowed,
            "live_cluster_create": self.live_cluster_create,
            "hardware_safety_pointer": self.hardware_safety_pointer,
            "compute_plan_pointer": self.compute_plan_pointer,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ThreatModelRow:
    """One threat-model row for the cloud deployment boundary.

    Attributes
    ----------
    threat_id
        Stable threat identifier.
    kind
        Threat kind enum.
    title
        Human-readable title.
    mitigation
        Product mitigation summary.
    fail_closed
        Must remain True.
    claim_boundary
        Non-promotional claim boundary.

    """

    threat_id: str
    kind: ThreatKind
    title: str
    mitigation: str
    fail_closed: bool = True
    claim_boundary: str = CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate threat-model row invariants."""
        if not self.threat_id or not self.threat_id.strip():
            raise ValueError("threat_id must be non-empty")
        if self.kind not in {
            "secret_leakage",
            "always_on_qpu",
            "live_cluster_create",
            "credential_loading",
            "unbounded_cost",
        }:
            raise ValueError(f"unknown threat kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.mitigation or not self.mitigation.strip():
            raise ValueError("mitigation must be non-empty")
        if self.fail_closed is not True:
            raise ValueError("fail_closed must be True on product threat model")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "threat_id": self.threat_id,
            "kind": self.kind,
            "title": self.title,
            "mitigation": self.mitigation,
            "fail_closed": self.fail_closed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for cloud deploy product use.

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
    claim_boundary: str = CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY

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
class MaterialisedDeployDryRunProbe:
    """Materialised dry-run deploy probe via ambient generate_cloud_manifests.

    Attributes
    ----------
    pattern_id
        Pattern used for the probe.
    manifest_sha256
        Ambient bundle sha256 of generated files.
    file_names
        Generated manifest file names.
    ambient_claim_boundary
        Ambient CloudManifestBundle claim boundary.
    invent_green_live_cluster
        Always False.
    invent_green_always_on_qpu
        Always False.
    secret_env_present
        Always False on successful product probes.
    demo_label
        Demo fixture label.
    claim_boundary
        Product claim boundary.

    """

    pattern_id: str
    manifest_sha256: str
    file_names: tuple[str, ...]
    ambient_claim_boundary: str
    invent_green_live_cluster: bool
    invent_green_always_on_qpu: bool
    secret_env_present: bool
    demo_label: str
    claim_boundary: str = CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate dry-run probe invariants."""
        if not self.pattern_id or not self.pattern_id.strip():
            raise ValueError("pattern_id must be non-empty")
        if not self.manifest_sha256 or not self.manifest_sha256.strip():
            raise ValueError("manifest_sha256 must be non-empty")
        if len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a 64-char hex SHA-256")
        if not self.file_names:
            raise ValueError("file_names must be non-empty")
        if any(not item or not str(item).strip() for item in self.file_names):
            raise ValueError("file_names entries must be non-empty")
        if not self.ambient_claim_boundary or not self.ambient_claim_boundary.strip():
            raise ValueError("ambient_claim_boundary must be non-empty")
        if self.invent_green_live_cluster:
            raise ValueError("invent_green_live_cluster must be False")
        if self.invent_green_always_on_qpu:
            raise ValueError("invent_green_always_on_qpu must be False")
        if self.secret_env_present:
            raise ValueError("secret_env_present must be False on product probes")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "pattern_id": self.pattern_id,
            "manifest_sha256": self.manifest_sha256,
            "file_names": list(self.file_names),
            "ambient_claim_boundary": self.ambient_claim_boundary,
            "invent_green_live_cluster": self.invent_green_live_cluster,
            "invent_green_always_on_qpu": self.invent_green_always_on_qpu,
            "secret_env_present": self.secret_env_present,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_patterns() -> tuple[DeploymentPatternRow, ...]:
    """Build the deployment pattern catalogue."""
    return (
        DeploymentPatternRow(
            pattern_id="batch_worker",
            kind="batch_worker",
            title="Batch worker (offline)",
            summary=(
                "Kubernetes/Compose batch worker for offline SCPN workloads; "
                "no always-on QPU; no secret env."
            ),
            default_command=("scpn-bench", "stable-core-contract-gate"),
            support_posture="policy_only",
        ),
        DeploymentPatternRow(
            pattern_id="stable_core_gate",
            kind="stable_core_gate",
            title="Stable-core contract gate job",
            summary=(
                "One-shot stable-core contract gate container; offline research "
                "posture; hardware-safe no-submit and dry-run compute planning."
            ),
            default_command=("scpn-bench", "stable-core-contract-gate"),
            support_posture="local_research",
        ),
        DeploymentPatternRow(
            pattern_id="offline_research",
            kind="offline_research",
            title="Offline research worker",
            summary=(
                "Low-replica research worker for local/offline packaging demos; "
                "never invent-green live cluster create."
            ),
            default_command=("python", "-m", "scpn_quantum_control"),
            support_posture="metadata_only",
        ),
    )


def _build_threats() -> tuple[ThreatModelRow, ...]:
    """Build the fail-closed threat-model catalogue."""
    return (
        ThreatModelRow(
            threat_id="secret_leakage",
            kind="secret_leakage",
            title="Secret-like environment variables in manifests",
            mitigation=(
                "Ambient CloudDeploymentSpec rejects TOKEN/SECRET/PASSWORD/"
                "API_KEY-like env keys; product path refuses secret injection."
            ),
        ),
        ThreatModelRow(
            threat_id="always_on_qpu",
            kind="always_on_qpu",
            title="Always-on QPU deploy claims",
            mitigation=(
                "Product patterns set allows_always_on_qpu=False and compose "
                "hardware-safe no-submit with dry-run compute planning."
            ),
        ),
        ThreatModelRow(
            threat_id="live_cluster_create",
            kind="live_cluster_create",
            title="Live cluster create / cloud API contact",
            mitigation=(
                "Ambient generator emits files only; product refuses invent-green "
                "live cluster create and credential loading."
            ),
        ),
        ThreatModelRow(
            threat_id="credential_loading",
            kind="credential_loading",
            title="Local credential / kubeconfig loading",
            mitigation=(
                "Ambient module does not read credentials; product path refuses "
                "credential-loading invent-green."
            ),
        ),
        ThreatModelRow(
            threat_id="unbounded_cost",
            kind="unbounded_cost",
            title="Unbounded cloud cost from always-on replicas",
            mitigation=(
                "Dry-run defaults to replicas=1 and offline image labels; cost "
                "threat documented fail-closed."
            ),
        ),
    )


_PATTERNS: Final[tuple[DeploymentPatternRow, ...]] = _build_patterns()
_THREATS: Final[tuple[ThreatModelRow, ...]] = _build_threats()


def _pattern_map() -> dict[str, DeploymentPatternRow]:
    """Return pattern_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, DeploymentPatternRow] = {}
    for row in _PATTERNS:
        key = row.pattern_id.strip()
        if not key:
            raise RuntimeError("deployment pattern catalogue contains blank pattern_id")
        if key in mapping:
            raise RuntimeError(f"duplicate pattern_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("deployment pattern catalogue must be non-empty")
    return mapping


_PATTERN_BY_ID: Final[Mapping[str, DeploymentPatternRow]] = _pattern_map()


def list_deployment_pattern_ids() -> tuple[str, ...]:
    """Return all deployment pattern identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable pattern ids.

    """
    return tuple(row.pattern_id for row in _PATTERNS)


def list_threat_ids() -> tuple[str, ...]:
    """Return all threat-model identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable threat ids.

    """
    return tuple(row.threat_id for row in _THREATS)


def get_deployment_pattern(pattern_id: str) -> DeploymentPatternRow:
    """Return one deployment pattern row; fail closed on blank/unknown.

    Parameters
    ----------
    pattern_id
        Pattern identifier.

    Returns
    -------
    DeploymentPatternRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not pattern_id or not str(pattern_id).strip():
        raise ValueError("pattern_id must be non-empty")
    key = str(pattern_id).strip()
    try:
        return _PATTERN_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown pattern_id: {key!r}") from exc


def iter_deployment_patterns(
    *,
    kind: DeploymentPatternKind | None = None,
) -> tuple[DeploymentPatternRow, ...]:
    """Return filtered deployment pattern rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.

    Returns
    -------
    tuple[DeploymentPatternRow, ...]
        Matching rows.

    """
    rows: Sequence[DeploymentPatternRow] = _PATTERNS
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def decide_deploy_path(
    pattern_id: str,
    *,
    invent_green_live_cluster: bool = False,
    invent_green_always_on_qpu: bool = False,
    inject_secret_env: bool = False,
    load_credentials: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a cloud deployment path may proceed.

    Parameters
    ----------
    pattern_id
        Deployment pattern identifier.
    invent_green_live_cluster
        If true, refuse.
    invent_green_always_on_qpu
        If true, refuse.
    inject_secret_env
        If true, refuse.
    load_credentials
        If true, refuse.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    row = get_deployment_pattern(pattern_id)
    blockers: list[str] = []
    if invent_green_live_cluster:
        blockers.append(
            "invent-green live cluster create refused "
            f"(pattern={row.pattern_id}; ambient generates files only)"
        )
    if invent_green_always_on_qpu:
        blockers.append(
            "invent-green always-on QPU deploy refused "
            f"(pattern={row.pattern_id}; hardware-safe no-submit posture)"
        )
    if inject_secret_env:
        blockers.append(
            "secret-like environment injection refused "
            f"(pattern={row.pattern_id}; ambient CloudDeploymentSpec rejects secrets)"
        )
    if load_credentials:
        blockers.append(
            "credential / kubeconfig loading refused "
            f"(pattern={row.pattern_id}; product dry-run only)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="deploy path refused under fail-closed cloud-native product policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"deploy dry-run path allowed for pattern {row.pattern_id!r} "
            f"(always_on_qpu=False; secret_env=False)"
        ),
        blockers=(),
    )


def materialise_deploy_dry_run_probe(
    pattern_id: str = "batch_worker",
    *,
    name: str = _DEMO_NAME,
    image: str = _DEMO_IMAGE,
    replicas: int = 1,
    env: Mapping[str, str] | None = None,
) -> MaterialisedDeployDryRunProbe:
    """Materialise dry-run manifests via the ambient manifest generator.

    Parameters
    ----------
    pattern_id
        Known deployment pattern.
    name
        Workload name (Kubernetes DNS-1123).
    image
        Container image reference (no whitespace).
    replicas
        Replica count (default 1 for cost honesty).
    env
        Optional non-secret environment mapping.

    Returns
    -------
    MaterialisedDeployDryRunProbe
        Finite primary observables with invent-green flags False.

    Raises
    ------
    ValueError
        If pattern unknown or ambient validation fails (e.g. secret env).

    """
    row = get_deployment_pattern(pattern_id)
    env_map = dict(env) if env is not None else {}
    # Product-level pre-check mirrors ambient secret policy for clearer errors.
    for key in env_map:
        upper = key.upper()
        if any(
            token in upper
            for token in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "PRIVATE")
        ):
            raise ValueError(
                "secret-like environment variables are not allowed in product dry-run"
            )
    spec = CloudDeploymentSpec(
        name=name,
        image=image,
        command=row.default_command,
        replicas=replicas,
        resources=ContainerResources(),
        env=env_map,
    )
    bundle = generate_cloud_manifests(spec)
    return MaterialisedDeployDryRunProbe(
        pattern_id=row.pattern_id,
        manifest_sha256=str(bundle.sha256),
        file_names=tuple(sorted(bundle.files.keys())),
        ambient_claim_boundary=str(bundle.claim_boundary),
        invent_green_live_cluster=False,
        invent_green_always_on_qpu=False,
        secret_env_present=False,
        demo_label="ambient_generate_cloud_manifests_dry_run",
    )


def materialise_demo_deploy_dry_run_probe() -> MaterialisedDeployDryRunProbe:
    """Materialise the deterministic batch_worker dry-run demo probe.

    Returns
    -------
    MaterialisedDeployDryRunProbe
        Ambient manifest dry-run probe.

    """
    return materialise_deploy_dry_run_probe("batch_worker")


def map_cloud_native_deployment_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of cloud-native deployment product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.cloud_native_deployment_product",
            "role": "cloud_native_deployment_product_surface",
            "support_posture": "policy_only",
            "pattern_ids": list(list_deployment_pattern_ids()),
            "threat_ids": list(list_threat_ids()),
            "allows_always_on_qpu": False,
            "claim_boundary": CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.deployment.cloud_native",
            "role": "ambient_manifest_generator",
            "support_posture": "policy_only",
            "symbol_name": "generate_cloud_manifests",
            "claim_boundary": CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY,
        },
    )


def build_cloud_native_deployment_product_registry() -> dict[str, object]:
    """Build the full serialisable cloud-native deployment product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with patterns + threats (no blanks).

    """
    patterns = [row.to_dict() for row in _PATTERNS]
    threats = [row.to_dict() for row in _THREATS]
    return {
        "schema": CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA,
        "claim_boundary": CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY,
        "pattern_count": len(patterns),
        "threat_count": len(threats),
        "blank_entry_count": 0,
        "allows_always_on_qpu_policy": False,
        "secret_env_allowed_policy": False,
        "live_cluster_create_policy": False,
        "public_surfaces": list(map_cloud_native_deployment_public_surfaces()),
        "patterns": patterns,
        "threats": threats,
        "policy_note": (
            "Cloud-native dry-run packaging only; ambient deployment.cloud_native "
            "generates K8s/Compose files without cluster create or secrets; "
            "hardware-safe execution and dry-run compute planning prohibit an "
            "always-on QPU; enterprise operations runbooks remain open."
        ),
    }


def assert_cloud_native_deployment_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers patterns/threats without invent-green QPU/cluster.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_cloud_native_deployment_product_registry`.

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
        dict(payload) if payload is not None else build_cloud_native_deployment_product_registry()
    )
    if registry.get("schema") != CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA:
        raise ValueError("cloud-native deployment product schema mismatch")
    patterns = registry.get("patterns")
    threats = registry.get("threats")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(
            "cloud-native deployment product registry must contain a non-empty patterns list"
        )
    if not isinstance(threats, list) or not threats:
        raise ValueError(
            "cloud-native deployment product registry must contain a non-empty threats list"
        )
    seen: set[str] = set()
    blank = 0
    batch_found = False
    for index, row in enumerate(patterns):
        if not isinstance(row, Mapping):
            raise ValueError(f"pattern row {index} must be a mapping")
        pattern_id = row.get("pattern_id")
        always_qpu = row.get("allows_always_on_qpu")
        secret_env = row.get("secret_env_allowed")
        live = row.get("live_cluster_create")
        command = row.get("default_command")
        if not pattern_id or not str(pattern_id).strip():
            blank += 1
            continue
        pid = str(pattern_id).strip()
        if pid in seen:
            raise ValueError(f"duplicate pattern_id in registry: {pid!r}")
        seen.add(pid)
        if pid == "batch_worker":
            batch_found = True
        if always_qpu is not False:
            raise ValueError(f"pattern {pid!r} allows_always_on_qpu must be False")
        if secret_env is not False:
            raise ValueError(f"pattern {pid!r} secret_env_allowed must be False")
        if live is not False:
            raise ValueError(f"pattern {pid!r} live_cluster_create must be False")
        if not isinstance(command, list) or not command:
            raise ValueError(f"pattern {pid!r} must have non-empty default_command list")
    if blank:
        raise ValueError(
            f"cloud-native deployment product registry has {blank} blank or invalid entries"
        )
    if not batch_found:
        raise ValueError("cloud-native deployment product registry missing batch_worker")
    expected = set(list_deployment_pattern_ids())
    if seen != expected:
        raise ValueError(
            f"registry pattern set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    seen_threats: set[str] = set()
    for index, row in enumerate(threats):
        if not isinstance(row, Mapping):
            raise ValueError(f"threat row {index} must be a mapping")
        threat_id = row.get("threat_id")
        fail_closed = row.get("fail_closed")
        if not threat_id or not str(threat_id).strip():
            raise ValueError(f"threat row {index} blank or invalid threat_id")
        tid = str(threat_id).strip()
        if tid in seen_threats:
            raise ValueError(f"duplicate threat_id in registry: {tid!r}")
        seen_threats.add(tid)
        if fail_closed is not True:
            raise ValueError(f"threat {tid!r} fail_closed must be True")
    expected_threats = set(list_threat_ids())
    if seen_threats != expected_threats:
        raise ValueError(
            f"registry threat set drift (missing={expected_threats - seen_threats!r}, "
            f"extra={seen_threats - expected_threats!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    pattern_count = registry.get("pattern_count", -1)
    if not isinstance(pattern_count, int) or pattern_count != len(patterns):
        raise ValueError("pattern_count does not match patterns list length")
    threat_count = registry.get("threat_count", -1)
    if not isinstance(threat_count, int) or threat_count != len(threats):
        raise ValueError("threat_count does not match threats list length")
    if registry.get("allows_always_on_qpu_policy", True) is not False:
        raise ValueError("allows_always_on_qpu_policy must be False")
    if registry.get("secret_env_allowed_policy", True) is not False:
        raise ValueError("secret_env_allowed_policy must be False")
    if registry.get("live_cluster_create_policy", True) is not False:
        raise ValueError("live_cluster_create_policy must be False")
    return registry


def compute_spec_digest(
    *,
    name: str,
    image: str,
    command: Sequence[str],
    replicas: int = 1,
) -> str:
    """Compute a canonical digest for a dry-run deploy request (anti-cheat).

    Parameters
    ----------
    name
        Workload name.
    image
        Image reference.
    command
        Command tokens.
    replicas
        Replica count.

    Returns
    -------
    str
        Hex SHA-256 digest.

    Raises
    ------
    ValueError
        If inputs are empty/invalid.

    """
    if not name or not name.strip():
        raise ValueError("name must be non-empty")
    if not image or not image.strip():
        raise ValueError("image must be non-empty")
    if not command or any(not str(item).strip() for item in command):
        raise ValueError("command must contain non-empty items")
    if replicas < 1:
        raise ValueError("replicas must be positive")
    payload = {
        "schema": "cloud_native_deploy_request.v1",
        "name": name.strip(),
        "image": image.strip(),
        "command": [str(item) for item in command],
        "replicas": replicas,
        "product_schema": CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY",
    "CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA",
    "DeploymentPatternKind",
    "DeploymentPatternRow",
    "MaterialisedDeployDryRunProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "ThreatKind",
    "ThreatModelRow",
    "assert_cloud_native_deployment_product_integrity",
    "build_cloud_native_deployment_product_registry",
    "compute_spec_digest",
    "decide_deploy_path",
    "get_deployment_pattern",
    "iter_deployment_patterns",
    "list_deployment_pattern_ids",
    "list_threat_ids",
    "map_cloud_native_deployment_public_surfaces",
    "materialise_demo_deploy_dry_run_probe",
    "materialise_deploy_dry_run_probe",
]
