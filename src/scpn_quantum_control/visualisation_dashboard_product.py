# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — fixture-driven visualisation dashboard product
"""Fail-closed **fixture-driven visualisation dashboard** product.

Productises static panel catalogues and materialised report probes from
allowed fixtures only:

* versioned panel registry (order-parameter/energy-loss, gradient-norm,
  coupling, witness summary, bitstring-from-saved-pack);
* secrets/token refuse on export text;
* materialised static bundle probe with digests and ``live_qpu=false`` honesty;
* refuse invent-green live QPU streaming and always-on SaaS dashboard claims.

Composes ambient :func:`differentiable_dashboard_status` only as a status
pointer — does **not** grow ``differentiable_dashboard.py`` into a god-file or
claim a full multi-panel command-line or SaaS product.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

PanelKind = Literal[
    "order_parameter_energy_loss",
    "gradient_norm",
    "coupling_heatmap",
    "witness_summary",
    "bitstring_saved_pack",
]
"""Panel family kinds for the product catalogue."""

SupportPosture = Literal[
    "fixture_materialised",
    "catalogue_only",
    "refuse_only",
]
"""Support posture badges for visualisation panels."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

VISUALISATION_DASHBOARD_PRODUCT_SCHEMA: Final[str] = "visualisation_dashboard_product.v2"
"""JSON schema identifier for serialised product payloads."""

_VISUALISATION_DEMO_FIXTURE_SCHEMA: Final[str] = "visualisation_demo_fixture.v2"
"""Schema identifier for the deterministic local demonstration fixture."""

VISUALISATION_DASHBOARD_CLAIM_BOUNDARY: Final[str] = (
    "This fixture-driven visualisation dashboard product catalogues static panel "
    "families and materialises local report probes from synthetic or explicitly "
    "allowed fixtures. It sets live_qpu=false and refuses live QPU streaming and "
    "always-on SaaS dashboard claims. Remaining panel bodies, a command-line "
    "bundle writer, challenge-result embeds, and notebook widgets remain outside "
    "the current product."
)
"""Shared claim boundary for visualisation product payloads."""

_VISUALISATION_DASHBOARD_POLICY_NOTE: Final[str] = (
    "Use only fixture-driven static panels. live_qpu remains false; live QPU "
    "streaming and always-on SaaS are refused. Additional panel bodies, a "
    "command-line bundle writer, challenge-result embeds, and notebook widgets "
    "remain outside this product. differentiable_dashboard remains a status-only "
    "facade."
)
"""Canonical product-registry policy note."""


def _require_exact_claim_boundary(claim_boundary: str) -> None:
    """Reject records whose claim boundary differs from the governed contract."""
    if claim_boundary != VISUALISATION_DASHBOARD_CLAIM_BOUNDARY:
        raise ValueError(
            "claim_boundary must match VISUALISATION_DASHBOARD_CLAIM_BOUNDARY exactly"
        )


def _is_sha256_digest(value: str) -> bool:
    """Return whether a string is exactly one lowercase SHA-256 hex digest."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*\S+"),
    re.compile(r"(?i)token\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
)
"""Fail-closed secret/token patterns for export scanning."""


@dataclass(frozen=True, slots=True)
class VisualisationPanelRow:
    """One product catalogue row for a static visualisation panel.

    Attributes
    ----------
    panel_id
        Stable panel identifier.
    kind
        Panel family kind.
    title
        Human-readable title.
    summary
        Short description.
    module_path
        Primary ambient or product module path.
    symbol_name
        Primary symbol.
    support_posture
        Support posture badge.
    live_qpu
        Must be False (no invent-green live QPU streaming).
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    panel_id: str
    kind: PanelKind
    title: str
    summary: str
    module_path: str
    symbol_name: str
    support_posture: SupportPosture
    live_qpu: bool = False
    as_of: str = "2026-07-24"
    claim_boundary: str = VISUALISATION_DASHBOARD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate panel catalogue row invariants."""
        if not self.panel_id or not self.panel_id.strip():
            raise ValueError("panel_id must be non-empty")
        if self.kind not in {
            "order_parameter_energy_loss",
            "gradient_norm",
            "coupling_heatmap",
            "witness_summary",
            "bitstring_saved_pack",
        }:
            raise ValueError(f"unknown panel kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.symbol_name or not self.symbol_name.strip():
            raise ValueError("symbol_name must be non-empty")
        if self.support_posture not in {
            "fixture_materialised",
            "catalogue_only",
            "refuse_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if self.live_qpu:
            raise ValueError(
                "product panels must set live_qpu=False (no invent-green live QPU streaming UI)"
            )
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "panel_id": self.panel_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "support_posture": self.support_posture,
            "live_qpu": self.live_qpu,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for visualisation product use.

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

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = VISUALISATION_DASHBOARD_CLAIM_BOUNDARY

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
        _require_exact_claim_boundary(self.claim_boundary)

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
class SecretsScanResult:
    """Result of scanning export text for secrets or tokens.

    Attributes
    ----------
    clean
        Whether no secret patterns matched.
    findings
        Matched pattern labels (empty when clean).

    """

    clean: bool
    findings: tuple[str, ...]
    claim_boundary: str = VISUALISATION_DASHBOARD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate secrets scan result invariants."""
        if self.clean and self.findings:
            raise ValueError("clean scans cannot list findings")
        if not self.clean and not self.findings:
            raise ValueError("dirty scans require findings")
        if any(not item or not item.strip() for item in self.findings):
            raise ValueError("findings entries must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this scan result."""
        return {
            "clean": self.clean,
            "findings": list(self.findings),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedStaticReportProbe:
    """Materialised static fixture-driven report probe.

    Attributes
    ----------
    panel_ids
        Panels included in the probe bundle.
    series_point_count
        Number of series points materialised (order-parameter/energy).
    gradient_norm_count
        Number of gradient-norm samples materialised.
    fixture_digest_sha256
        SHA-256 hex digest of the canonical fixture JSON.
    live_qpu
        Must be False.
    secrets_clean
        Whether export text passed the secrets scanner.
    demo_label
        Demo fixture label.

    """

    panel_ids: tuple[str, ...]
    series_point_count: int
    gradient_norm_count: int
    fixture_digest_sha256: str
    live_qpu: bool
    secrets_clean: bool
    demo_label: str
    claim_boundary: str = VISUALISATION_DASHBOARD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised static report probe invariants."""
        if not self.panel_ids:
            raise ValueError("panel_ids must be non-empty")
        if any(not item or not item.strip() for item in self.panel_ids):
            raise ValueError("panel_ids entries must be non-empty")
        if self.series_point_count <= 0:
            raise ValueError("series_point_count must be positive")
        if self.gradient_norm_count <= 0:
            raise ValueError("gradient_norm_count must be positive")
        if not _is_sha256_digest(self.fixture_digest_sha256):
            raise ValueError("fixture_digest_sha256 must be a lowercase SHA-256 hex digest")
        if self.live_qpu:
            raise ValueError("static report probe must set live_qpu=False")
        if not self.secrets_clean:
            raise ValueError("static report probe requires secrets_clean=True")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "panel_ids": list(self.panel_ids),
            "series_point_count": self.series_point_count,
            "gradient_norm_count": self.gradient_norm_count,
            "fixture_digest_sha256": self.fixture_digest_sha256,
            "live_qpu": self.live_qpu,
            "secrets_clean": self.secrets_clean,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    panel_id: str,
    *,
    kind: PanelKind,
    title: str,
    summary: str,
    module_path: str,
    symbol_name: str,
    support_posture: SupportPosture,
) -> VisualisationPanelRow:
    """Build one panel catalogue row."""
    return VisualisationPanelRow(
        panel_id=panel_id,
        kind=kind,
        title=title,
        summary=summary,
        module_path=module_path,
        symbol_name=symbol_name,
        support_posture=support_posture,
    )


_CANONICAL_PANELS: Final[tuple[VisualisationPanelRow, ...]] = (
    _row(
        "order_parameter_energy_loss",
        kind="order_parameter_energy_loss",
        title="Order-parameter and energy/loss time series",
        summary=(
            "Static series panel for order-parameter and energy/loss curves from "
            "optimiser or witness fixtures (no live stream)."
        ),
        module_path="scpn_quantum_control.visualisation_dashboard_product",
        symbol_name="materialise_demo_static_report_probe",
        support_posture="fixture_materialised",
    ),
    _row(
        "gradient_norm",
        kind="gradient_norm",
        title="Gradient-norm / trainability panel",
        summary=(
            "Static gradient-norm series from saved optimiser traces or synthetic "
            "fixtures (fixture-driven only)."
        ),
        module_path="scpn_quantum_control.visualisation_dashboard_product",
        symbol_name="materialise_demo_static_report_probe",
        support_posture="fixture_materialised",
    ),
    _row(
        "coupling_heatmap",
        kind="coupling_heatmap",
        title="Coupling heatmap / graph panel",
        summary=(
            "Catalogue row for coupling heatmap/graph panels from saved matrices "
            "(materialisation residual beyond demo probe)."
        ),
        module_path="scpn_quantum_control.visualisation_dashboard_product",
        symbol_name="get_visualisation_panel",
        support_posture="catalogue_only",
    ),
    _row(
        "witness_summary",
        kind="witness_summary",
        title="Witness Betti / persistence summary",
        summary=(
            "Catalogue row for witness persistence/Betti summaries from saved "
            "witness suite fixtures."
        ),
        module_path="scpn_quantum_control.visualisation_dashboard_product",
        symbol_name="get_visualisation_panel",
        support_posture="catalogue_only",
    ),
    _row(
        "bitstring_saved_pack",
        kind="bitstring_saved_pack",
        title="Bitstring histogram from saved hardware pack",
        summary=(
            "Catalogue row for bitstring histograms from saved hardware packs only "
            "(no live fetch)."
        ),
        module_path="scpn_quantum_control.visualisation_dashboard_product",
        symbol_name="get_visualisation_panel",
        support_posture="catalogue_only",
    ),
    _row(
        "refuse_live_qpu_stream",
        kind="order_parameter_energy_loss",
        title="Refuse live QPU streaming / always-on SaaS",
        summary=(
            "Product refuse path for invent-green live QPU streaming UI and "
            "always-on SaaS dashboard claims."
        ),
        module_path="scpn_quantum_control.visualisation_dashboard_product",
        symbol_name="decide_visualisation_path",
        support_posture="refuse_only",
    ),
)


def _catalogue_map() -> dict[str, VisualisationPanelRow]:
    """Return panel_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, VisualisationPanelRow] = {}
    for row in _CANONICAL_PANELS:
        key = row.panel_id.strip()
        if not key:
            raise RuntimeError("visualisation catalogue contains blank panel_id")
        if key in mapping:
            raise RuntimeError(f"duplicate panel_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("visualisation catalogue must be non-empty")
    return mapping


_PANEL_BY_ID: Final[Mapping[str, VisualisationPanelRow]] = _catalogue_map()


def list_visualisation_panel_ids() -> tuple[str, ...]:
    """Return all product panel identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered panel identifiers.

    """
    return tuple(row.panel_id for row in _CANONICAL_PANELS)


def get_visualisation_panel(panel_id: str) -> VisualisationPanelRow:
    """Return one panel row or raise for blank/unknown identifiers.

    Parameters
    ----------
    panel_id
        Catalogue panel key.

    Returns
    -------
    VisualisationPanelRow
        Matching row.

    Raises
    ------
    ValueError
        If ``panel_id`` is blank or unknown (fail closed).

    """
    if not panel_id or not str(panel_id).strip():
        raise ValueError("panel_id must be a non-empty string")
    key = str(panel_id).strip()
    try:
        return _PANEL_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown panel_id {key!r}; refuse invent-green visualisation "
            f"product claim (known_count={len(_PANEL_BY_ID)})"
        ) from exc


def iter_visualisation_panels(
    *,
    kind: PanelKind | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[VisualisationPanelRow, ...]:
    """Return filtered panel rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[VisualisationPanelRow, ...]
        Matching rows.

    """
    rows: Sequence[VisualisationPanelRow] = _CANONICAL_PANELS
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def scan_export_for_secrets(text: str) -> SecretsScanResult:
    """Scan export text for secret or token patterns.

    Parameters
    ----------
    text
        Export payload text to scan.

    Returns
    -------
    SecretsScanResult
        Clean or dirty scan result with findings.

    Raises
    ------
    ValueError
        If ``text`` is not a string.

    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    findings: list[str] = []
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(pattern.pattern)
    if findings:
        return SecretsScanResult(clean=False, findings=tuple(dict.fromkeys(findings)))
    return SecretsScanResult(clean=True, findings=())


def decide_visualisation_path(
    *,
    request_live_qpu_stream: bool = False,
    request_saas_dashboard: bool = False,
    fixture_driven: bool = True,
) -> PathEligibilityDecision:
    """Decide whether a visualisation product path may proceed.

    Parameters
    ----------
    request_live_qpu_stream
        When true, refuse invent-green live QPU streaming UI.
    request_saas_dashboard
        When true, refuse invent-green always-on SaaS dashboard.
    fixture_driven
        Whether a fixture-driven static path is declared.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused decision with blockers.

    """
    blockers: list[str] = []
    if request_live_qpu_stream:
        blockers.append(
            "live QPU streaming UI refused on visualisation product "
            "(fixture-driven static reports only; live_qpu=false)"
        )
    if request_saas_dashboard:
        blockers.append(
            "always-on SaaS dashboard service refused "
            "(static HTML/JSON bundles only; out of product scope)"
        )
    if not fixture_driven:
        blockers.append(
            "non-fixture-driven visualisation path refused "
            "(product requires allowed fixtures / synthetic demo fixtures)"
        )
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="visualisation product refuse: " + "; ".join(unique),
            blockers=unique,
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            "visualisation product path allowed for fixture-driven static panels "
            "(live_qpu=false; no SaaS claim)"
        ),
        blockers=(),
    )


def _demo_fixture_payload() -> dict[str, object]:
    """Return a deterministic synthetic fixture for static demos."""
    return {
        "schema": _VISUALISATION_DEMO_FIXTURE_SCHEMA,
        "order_parameter": [0.1, 0.35, 0.62, 0.81, 0.9],
        "energy_loss": [1.2, 0.95, 0.7, 0.55, 0.48],
        "gradient_norms": [0.8, 0.45, 0.3, 0.18, 0.1],
        "live_qpu": False,
        "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
    }


def materialise_demo_static_report_probe() -> MaterialisedStaticReportProbe:
    """Materialise a deterministic static report probe from demo fixtures.

    Builds order-parameter/energy and gradient-norm series from a synthetic
    fixture, digests the fixture JSON, and scans the export text for secrets.

    Returns
    -------
    MaterialisedStaticReportProbe
        Non-empty panel ids, series counts, digest, and honesty flags.

    Raises
    ------
    ValueError
        If path is refused or secrets scan fails.

    """
    decision = decide_visualisation_path(fixture_driven=True)
    if not decision.allowed:
        raise ValueError(f"static report probe refused: {decision.reason}")

    fixture = _demo_fixture_payload()
    if fixture.get("schema") != _VISUALISATION_DEMO_FIXTURE_SCHEMA:
        raise ValueError("demo fixture schema mismatch")
    if fixture.get("claim_boundary") != VISUALISATION_DASHBOARD_CLAIM_BOUNDARY:
        raise ValueError("demo fixture claim_boundary mismatch")
    if fixture.get("live_qpu") is not False:
        raise ValueError("demo fixture must set live_qpu=False")
    order = fixture["order_parameter"]
    energy = fixture["energy_loss"]
    grads = fixture["gradient_norms"]
    if not isinstance(order, list) or not isinstance(energy, list) or not isinstance(grads, list):
        raise ValueError("demo fixture series must be lists")
    if len(order) != len(energy) or not order:
        raise ValueError("order_parameter and energy_loss must be non-empty equal length")
    if not grads:
        raise ValueError("gradient_norms must be non-empty")

    canonical = json.dumps(fixture, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    scan = scan_export_for_secrets(canonical)
    if not scan.clean:
        raise ValueError("demo fixture failed secrets scan: " + ", ".join(scan.findings))

    return MaterialisedStaticReportProbe(
        panel_ids=("order_parameter_energy_loss", "gradient_norm"),
        series_point_count=len(order),
        gradient_norm_count=len(grads),
        fixture_digest_sha256=digest,
        live_qpu=False,
        secrets_clean=True,
        demo_label="synthetic_order_energy_gradient_fixture",
    )


def map_visualisation_dashboard_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of visualisation product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.visualisation_dashboard_product",
            "role": "visualisation_dashboard_product_surface",
            "support_posture": "fixture_materialised",
            "panel_ids": list(list_visualisation_panel_ids()),
            "live_qpu": False,
            "ambient_status_pointer": "scpn_quantum_control.differentiable_dashboard",
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.differentiable_dashboard",
            "role": "ambient_capability_status_facade",
            "support_posture": "catalogue_only",
            "symbol_name": "differentiable_dashboard_status",
            "live_qpu": False,
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        },
    )


def build_visualisation_dashboard_product_registry() -> dict[str, object]:
    """Build the full serialisable visualisation product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with panels (no blanks).

    """
    panels = [row.to_dict() for row in _CANONICAL_PANELS]
    return {
        "schema": VISUALISATION_DASHBOARD_PRODUCT_SCHEMA,
        "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        "panel_count": len(panels),
        "blank_entry_count": 0,
        "default_panel_id": "order_parameter_energy_loss",
        "live_qpu_policy": False,
        "public_surfaces": list(map_visualisation_dashboard_public_surfaces()),
        "panels": panels,
        "policy_note": _VISUALISATION_DASHBOARD_POLICY_NOTE,
    }


def assert_visualisation_dashboard_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers panels without blanks or invent-live-QPU.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_visualisation_dashboard_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green live_qpu flags appear.

    """
    registry = (
        dict(payload) if payload is not None else build_visualisation_dashboard_product_registry()
    )
    panels = registry.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("visualisation product registry must contain a non-empty panels list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    refuse_found = False
    for index, row in enumerate(panels):
        if not isinstance(row, Mapping):
            raise ValueError(f"panel row {index} must be a mapping")
        panel_id = row.get("panel_id")
        kind = row.get("kind")
        live_qpu = row.get("live_qpu")
        symbol_name = row.get("symbol_name")
        if not panel_id or not str(panel_id).strip():
            blank += 1
            continue
        pid = str(panel_id).strip()
        if pid in seen:
            raise ValueError(f"duplicate panel_id in registry: {pid!r}")
        seen.add(pid)
        if pid == "order_parameter_energy_loss":
            default_found = True
        if pid == "refuse_live_qpu_stream":
            refuse_found = True
        if kind not in {
            "order_parameter_energy_loss",
            "gradient_norm",
            "coupling_heatmap",
            "witness_summary",
            "bitstring_saved_pack",
        }:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"panel {pid!r} must have symbol_name")
        if live_qpu is True:
            raise ValueError(
                f"panel {pid!r} invent-green live_qpu: product rows must set live_qpu=False"
            )
    if blank:
        raise ValueError(f"visualisation product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("visualisation product registry missing order_parameter_energy_loss")
    if not refuse_found:
        raise ValueError("visualisation product registry missing refuse_live_qpu_stream")
    expected = set(list_visualisation_panel_ids())
    if seen != expected:
        raise ValueError(
            f"registry panel set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    panel_count = registry.get("panel_count", -1)
    if not isinstance(panel_count, int) or panel_count != len(panels):
        raise ValueError("panel_count does not match panels list length")
    live_policy = registry.get("live_qpu_policy", True)
    if live_policy is not False:
        raise ValueError("live_qpu_policy must be False")
    if registry.get("schema") != VISUALISATION_DASHBOARD_PRODUCT_SCHEMA:
        raise ValueError("product schema mismatch")
    if registry.get("claim_boundary") != VISUALISATION_DASHBOARD_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary mismatch")
    if registry.get("policy_note") != _VISUALISATION_DASHBOARD_POLICY_NOTE:
        raise ValueError("policy_note mismatch")
    if registry.get("default_panel_id") != "order_parameter_energy_loss":
        raise ValueError("default_panel_id mismatch")
    expected_rows = {row.panel_id: row.to_dict() for row in _CANONICAL_PANELS}
    for index, row in enumerate(panels):
        panel_id = str(row["panel_id"]).strip()
        if dict(row) != expected_rows[panel_id]:
            raise ValueError(f"panel row {index} drift for {panel_id!r}")
    if registry.get("public_surfaces") != list(map_visualisation_dashboard_public_surfaces()):
        raise ValueError("public_surfaces mismatch")
    return registry


__all__ = [
    "VISUALISATION_DASHBOARD_CLAIM_BOUNDARY",
    "VISUALISATION_DASHBOARD_PRODUCT_SCHEMA",
    "MaterialisedStaticReportProbe",
    "PanelKind",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SecretsScanResult",
    "SupportPosture",
    "VisualisationPanelRow",
    "assert_visualisation_dashboard_product_integrity",
    "build_visualisation_dashboard_product_registry",
    "decide_visualisation_path",
    "get_visualisation_panel",
    "iter_visualisation_panels",
    "list_visualisation_panel_ids",
    "map_visualisation_dashboard_public_surfaces",
    "materialise_demo_static_report_probe",
    "scan_export_for_secrets",
]
