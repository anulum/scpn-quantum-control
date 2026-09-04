# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Domain application honesty kits
"""Fail-closed claim and data boundaries for domain-facing applications.

The objects in this module do not certify domain validity.  They make the
opposite boundary explicit: each kit identifies the small software route that
is supported, the data origin admitted by that route, and the claims that
remain forbidden.  The built-in registry covers the application-honesty power-grid,
Josephson, EEG-like, and ITER-inspired application families.

All returned records are immutable and JSON-ready.  The audit functions are
local and deterministic; they do not read credentials, contact providers,
submit hardware work, or inspect private datasets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from .dataset_catalog import (
    ApplicationBenchmarkPrivacyAudit,
    audit_application_benchmark_privacy,
)

APPLICATION_HONESTY_SCHEMA = "scpn.application-honesty.v1"
APPLICATION_HONESTY_CLAIM_BOUNDARY = (
    "software-contract and synthetic-or-curated benchmark evidence only; not domain "
    "validation, operational control, clinical use, facility prediction, hardware "
    "performance, or quantum advantage"
)
ForecastingDomainTag = Literal[
    "synthetic",
    "grid_like_sim",
    "eeg_like_sim",
    "plasma_like_sim",
]
FORECASTING_DOMAIN_TAGS: frozenset[str] = frozenset(
    {"synthetic", "grid_like_sim", "eeg_like_sim", "plasma_like_sim"}
)


class ApplicationSupportStatus(str, Enum):
    """Public support grade for a domain-facing application route.

    ``BOUNDED_RESEARCH`` means that the named software path is tested for its
    documented small benchmark, while ``SIMULATION_ONLY`` requires generated
    inputs and forbids measured-domain interpretation.
    """

    BOUNDED_RESEARCH = "bounded_research"
    SIMULATION_ONLY = "simulation_only"


class ApplicationDataOrigin(str, Enum):
    """Admitted input provenance for an honesty kit."""

    SYNTHETIC = "synthetic"
    CURATED_PUBLIC = "curated_public"


@dataclass(frozen=True, slots=True)
class DomainApplicationHonestyKit:
    """Immutable claim boundary for one domain-facing application family.

    Parameters
    ----------
    kit_id
        Stable machine identifier for the kit.
    domain_tag
        Non-promotional domain label used in reports and user interfaces.
    title
        Human-readable kit name.
    support_status
        Whether the route is a bounded research benchmark or simulation-only.
    data_origin
        Provenance class admitted by this kit.
    synthetic_only
        ``True`` when measured or curated domain data must not enter the route.
    dataset_ids
        Packaged public catalogue identifiers governed by the kit.  An empty
        tuple means that the route generates its inputs in code.
    source_modules
        Import paths implementing the bounded route.
    allowed_uses
        Positive, narrowly worded descriptions of supported software use.
    caveats
        Scientific and operational limitations that callers must preserve.
    claims_forbidden
        Explicit claims that this kit never authorises.
    forecasting_tags
        Simulation-only forecasting tags that may be cross-referenced. These
        tags do not convert a synthetic forecast into domain evidence.

    Notes
    -----
    Construction validates the internal policy relationships.  In particular,
    a synthetic-only kit cannot declare curated input data or packaged dataset
    identifiers, and every kit must retain at least one forbidden claim.

    """

    kit_id: str
    domain_tag: str
    title: str
    support_status: ApplicationSupportStatus
    data_origin: ApplicationDataOrigin
    synthetic_only: bool
    dataset_ids: tuple[str, ...]
    source_modules: tuple[str, ...]
    allowed_uses: tuple[str, ...]
    caveats: tuple[str, ...]
    claims_forbidden: tuple[str, ...]
    forecasting_tags: tuple[ForecastingDomainTag, ...] = ()

    def __post_init__(self) -> None:
        """Validate the fail-closed relationships between policy fields."""
        for name, value in (
            ("kit_id", self.kit_id),
            ("domain_tag", self.domain_tag),
            ("title", self.title),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.support_status, ApplicationSupportStatus):
            raise TypeError("support_status must be an ApplicationSupportStatus")
        if not isinstance(self.data_origin, ApplicationDataOrigin):
            raise TypeError("data_origin must be an ApplicationDataOrigin")
        if self.synthetic_only is not (self.data_origin is ApplicationDataOrigin.SYNTHETIC):
            raise ValueError("synthetic_only must match a synthetic data_origin")
        if self.synthetic_only and self.dataset_ids:
            raise ValueError("synthetic-only kits cannot name packaged datasets")
        _require_non_empty_strings(self.source_modules, name="source_modules")
        _require_non_empty_strings(self.allowed_uses, name="allowed_uses")
        _require_non_empty_strings(self.caveats, name="caveats")
        _require_non_empty_strings(self.claims_forbidden, name="claims_forbidden")
        _require_unique_strings(self.dataset_ids, name="dataset_ids")
        if any(tag not in FORECASTING_DOMAIN_TAGS for tag in self.forecasting_tags):
            raise ValueError("forecasting_tags must use the simulation-only vocabulary")
        if len(set(self.forecasting_tags)) != len(self.forecasting_tags):
            raise ValueError("forecasting_tags must be unique")

    @property
    def publication_safe(self) -> bool:
        """Always false because a kit is not domain-publication evidence."""
        return False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation with explicit non-claim fields."""
        return {
            "kit_id": self.kit_id,
            "domain_tag": self.domain_tag,
            "title": self.title,
            "support_status": self.support_status.value,
            "data_origin": self.data_origin.value,
            "synthetic_only": self.synthetic_only,
            "publication_safe": self.publication_safe,
            "dataset_ids": list(self.dataset_ids),
            "source_modules": list(self.source_modules),
            "allowed_uses": list(self.allowed_uses),
            "caveats": list(self.caveats),
            "claims_forbidden": list(self.claims_forbidden),
            "forecasting_tags": list(self.forecasting_tags),
        }


@dataclass(frozen=True, slots=True)
class ApplicationHonestyAuditReport:
    """Deterministic aggregate of honesty kits and dataset privacy checks.

    Parameters
    ----------
    kits
        Validated built-in honesty-kit records.
    dataset_privacy
        Catalogue privacy audit rows.  Each row has already loaded and
        validated the corresponding packaged ``QPUDataArtifact``.

    """

    kits: tuple[DomainApplicationHonestyKit, ...]
    dataset_privacy: tuple[ApplicationBenchmarkPrivacyAudit, ...]

    def __post_init__(self) -> None:
        """Reject empty, duplicate, or incomplete aggregate reports."""
        if not self.kits:
            raise ValueError("honesty report requires at least one kit")
        if not self.dataset_privacy:
            raise ValueError("honesty report requires dataset privacy evidence")
        _require_unique_strings(tuple(kit.kit_id for kit in self.kits), name="kit ids")
        governed_ids = {dataset_id for kit in self.kits for dataset_id in kit.dataset_ids}
        audited_ids = {row.dataset_id for row in self.dataset_privacy}
        missing = governed_ids - audited_ids
        if missing:
            raise ValueError(f"kit datasets missing privacy evidence: {sorted(missing)!r}")

    @property
    def passed(self) -> bool:
        """Whether every built-in kit and dataset privacy row is valid."""
        return bool(self.kits) and all(row.passed for row in self.dataset_privacy)

    def content_digest(self) -> str:
        """Return a SHA-256 digest of the canonical report payload."""
        payload = self._payload_without_digest()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON evidence payload including its digest."""
        payload = self._payload_without_digest()
        payload["content_digest"] = self.content_digest()
        return payload

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": APPLICATION_HONESTY_SCHEMA,
            "claim_boundary": APPLICATION_HONESTY_CLAIM_BOUNDARY,
            "passed": self.passed,
            "kit_count": len(self.kits),
            "dataset_privacy_count": len(self.dataset_privacy),
            "kits": [kit.as_dict() for kit in self.kits],
            "dataset_privacy": [row.as_dict() for row in self.dataset_privacy],
        }


def _require_non_empty_strings(values: tuple[str, ...], *, name: str) -> None:
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    _require_unique_strings(values, name=name)


def _require_unique_strings(values: tuple[str, ...], *, name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


_BUILTIN_KITS = (
    DomainApplicationHonestyKit(
        kit_id="power_grid_public_benchmark",
        domain_tag="power_grid_benchmark",
        title="Public power-grid benchmark kit",
        support_status=ApplicationSupportStatus.BOUNDED_RESEARCH,
        data_origin=ApplicationDataOrigin.CURATED_PUBLIC,
        synthetic_only=False,
        dataset_ids=("ieee5bus_power_grid",),
        source_modules=(
            "scpn_quantum_control.applications.power_grid",
            "scpn_quantum_control.applications.app_plugins",
        ),
        allowed_uses=(
            "compile public IEEE benchmark constants into a bounded Kuramoto problem",
            "exercise topology-similarity and artifact-custody software contracts",
        ),
        caveats=(
            "the packaged case is a compact public benchmark, not live SCADA data",
            "topology similarity is not transient-stability or control validation",
        ),
        claims_forbidden=(
            "live-grid control",
            "operational stability prediction",
            "utility deployment readiness",
            "quantum advantage",
        ),
        forecasting_tags=("grid_like_sim",),
    ),
    DomainApplicationHonestyKit(
        kit_id="josephson_illustrative_simulation",
        domain_tag="josephson_like_sim",
        title="Illustrative Josephson-array simulation kit",
        support_status=ApplicationSupportStatus.SIMULATION_ONLY,
        data_origin=ApplicationDataOrigin.SYNTHETIC,
        synthetic_only=True,
        dataset_ids=(),
        source_modules=("scpn_quantum_control.applications.josephson_array",),
        allowed_uses=(
            "construct labelled illustrative coupling topologies",
            "compare finite synthetic coupling structures",
        ),
        caveats=(
            "nominal transmon values are literature illustrations, not calibration records",
            "an XY structural analogy is not a device-level reproduction",
        ),
        claims_forbidden=(
            "measured-device validation",
            "hardware self-simulation",
            "fabrication or calibration guidance",
            "quantum advantage",
        ),
    ),
    DomainApplicationHonestyKit(
        kit_id="eeg_like_synthetic",
        domain_tag="eeg_like_sim",
        title="Synthetic EEG-like software kit",
        support_status=ApplicationSupportStatus.SIMULATION_ONLY,
        data_origin=ApplicationDataOrigin.SYNTHETIC,
        synthetic_only=True,
        dataset_ids=(),
        source_modules=(
            "scpn_quantum_control.applications.eeg_classification",
            "scpn_quantum_control.forecasting.synthetic_multimodal",
        ),
        allowed_uses=(
            "exercise PLV-shaped matrix validation with generated inputs",
            "evaluate EEG-like simulation-only forecasting tags",
        ),
        caveats=(
            "EEG-like labels describe generator shape only",
            "no participant recording, diagnosis, or clinical endpoint is admitted",
        ),
        claims_forbidden=(
            "clinical classification",
            "diagnostic or therapeutic use",
            "neural-dynamics reproduction",
            "human-subject generalisation",
        ),
        forecasting_tags=("eeg_like_sim",),
    ),
    DomainApplicationHonestyKit(
        kit_id="iter_disruption_inspired_simulation",
        domain_tag="plasma_like_sim",
        title="ITER-inspired disruption simulation kit",
        support_status=ApplicationSupportStatus.SIMULATION_ONLY,
        data_origin=ApplicationDataOrigin.SYNTHETIC,
        synthetic_only=True,
        dataset_ids=(),
        source_modules=(
            "scpn_quantum_control.applications.disruption_classifier",
            "scpn_quantum_control.control.q_disruption_iter",
        ),
        allowed_uses=(
            "generate opt-in synthetic disruption-feature fixtures",
            "exercise advisory classifier and dependency contracts locally",
        ),
        caveats=(
            "feature ranges and generated labels are simulation fixtures",
            "facility data and downstream control admission require independent evidence",
        ),
        claims_forbidden=(
            "ITER disruption prediction",
            "facility validation",
            "real-time controller admission",
            "plasma-operation guidance",
        ),
        forecasting_tags=("plasma_like_sim",),
    ),
)


def list_domain_application_honesty_kits() -> tuple[DomainApplicationHonestyKit, ...]:
    """Return all built-in domain application honesty kits in stable order."""
    return _BUILTIN_KITS


def get_domain_application_honesty_kit(kit_id: str) -> DomainApplicationHonestyKit:
    """Return one built-in honesty kit by stable identifier.

    Parameters
    ----------
    kit_id
        Exact identifier returned by :func:`list_domain_application_honesty_kits`.

    Raises
    ------
    KeyError
        If ``kit_id`` is unknown.  The error includes the known identifiers.

    """
    for kit in _BUILTIN_KITS:
        if kit.kit_id == kit_id:
            return kit
    known = ", ".join(kit.kit_id for kit in _BUILTIN_KITS)
    raise KeyError(f"unknown application honesty kit {kit_id!r}; known: {known}")


def get_domain_application_honesty_kit_for_dataset(
    dataset_id: str,
) -> DomainApplicationHonestyKit:
    """Return the unique kit governing a packaged dataset identifier.

    Synthetic-only kits intentionally have no packaged dataset identifiers and
    therefore cannot be resolved through this function.

    Raises
    ------
    KeyError
        If no built-in kit governs ``dataset_id``.
    RuntimeError
        If registry corruption assigns the same dataset to multiple kits.

    """
    matches = tuple(kit for kit in _BUILTIN_KITS if dataset_id in kit.dataset_ids)
    if not matches:
        raise KeyError(f"no application honesty kit governs dataset {dataset_id!r}")
    if len(matches) != 1:
        raise RuntimeError(f"multiple application honesty kits govern dataset {dataset_id!r}")
    return matches[0]


def build_application_honesty_audit_report() -> ApplicationHonestyAuditReport:
    """Build deterministic local evidence for every kit and catalogue row.

    Returns
    -------
    ApplicationHonestyAuditReport
        Immutable report with a canonical content digest.

    Notes
    -----
    The audit loads only versioned packaged application artifacts.  It performs
    no network access and never opens a user-supplied or private dataset.

    """
    return ApplicationHonestyAuditReport(
        kits=list_domain_application_honesty_kits(),
        dataset_privacy=audit_application_benchmark_privacy(),
    )


def render_application_honesty_audit_markdown(
    report: ApplicationHonestyAuditReport,
) -> str:
    """Render a human-readable Markdown evidence report.

    Parameters
    ----------
    report
        Validated report returned by
        :func:`build_application_honesty_audit_report`.

    """
    lines = [
        "# Domain Application Honesty Evidence",
        "",
        f"- Schema: `{APPLICATION_HONESTY_SCHEMA}`",
        f"- Result: `{'PASS' if report.passed else 'FAIL'}`",
        f"- Content digest: `{report.content_digest()}`",
        f"- Claim boundary: {APPLICATION_HONESTY_CLAIM_BOUNDARY}.",
        "",
        "## Honesty kits",
        "",
        "| Kit | Support | Data origin | Synthetic only | Forecasting tags |",
        "|---|---|---|:---:|---|",
    ]
    for kit in report.kits:
        tags = ", ".join(kit.forecasting_tags) or "none"
        lines.append(
            f"| `{kit.kit_id}` | `{kit.support_status.value}` | "
            f"`{kit.data_origin.value}` | `{str(kit.synthetic_only).lower()}` | `{tags}` |"
        )
    lines.extend(
        [
            "",
            "## Packaged dataset privacy audit",
            "",
            "| Dataset | Source mode | Privacy class | Personal data | Result |",
            "|---|---|---|:---:|:---:|",
        ]
    )
    for row in report.dataset_privacy:
        lines.append(
            f"| `{row.dataset_id}` | `{row.source_mode}` | "
            f"`{row.privacy_classification}` | "
            f"`{str(row.contains_personal_data).lower()}` | "
            f"`{'PASS' if row.passed else 'FAIL'}` |"
        )
    lines.extend(
        [
            "",
            "This evidence validates software metadata and packaged-artifact privacy "
            "boundaries only. It is not domain, clinical, facility, hardware, or "
            "advantage evidence.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "APPLICATION_HONESTY_CLAIM_BOUNDARY",
    "APPLICATION_HONESTY_SCHEMA",
    "FORECASTING_DOMAIN_TAGS",
    "ApplicationDataOrigin",
    "ApplicationHonestyAuditReport",
    "ApplicationSupportStatus",
    "DomainApplicationHonestyKit",
    "ForecastingDomainTag",
    "build_application_honesty_audit_report",
    "get_domain_application_honesty_kit",
    "get_domain_application_honesty_kit_for_dataset",
    "list_domain_application_honesty_kits",
    "render_application_honesty_audit_markdown",
]
