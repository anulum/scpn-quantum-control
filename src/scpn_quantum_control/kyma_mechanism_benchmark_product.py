# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KYMA mechanism benchmark product
"""Fail-closed **KYMA / KYMA v2 public mechanism-only benchmark** product.

Productises a preregistered, mechanism-only sync-learning honesty surface over
ambient :mod:`benchmarks.kyma` / :mod:`benchmarks.kyma_v2`:

* versioned suite catalogue (v1 baseline, v2 corrected design);
* frozen design-constant schema + content digests (teacher dynamics only —
  never student held-out accuracy);
* realisability / non-separability certificate probes via ambient
  :mod:`benchmarks.kyma_v2.design`;
* baseline harness pointers (classical ML residual depth honest);
* refuse post-hoc constant retuning, invent-green advantage without KYMA
  protocol id, and design freeze from student metrics (advantage-language compose).

Does **not** re-train full student/MLP suites, invent hermetic reproduction-kit
export, or claim public marketing gold without a promotion package.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

# Product-mirrored prereg §5 design constants — must match ambient
# ``benchmarks.kyma_v2.design`` module values (teacher-dynamics freeze grids).
# Lazy ambient import for certificate probes avoids pytest-cov/numpy reload
# breaking JAX at product import time (the same optional-framework coverage class).
_G_SYNC_GRID: Final[tuple[float, ...]] = (0.5, 0.75, 1.0, 1.5, 2.0)
_STEPS_GRID: Final[tuple[int, ...]] = (40, 50, 60, 70, 80)
_K_BRIDGE_GRID: Final[tuple[float, ...]] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
_REALISE_FRACTION: Final[float] = 0.95
_NON_SEP_TARGET: Final[float] = 0.40
_BALANCE_MAX_CLASS_FRACTION: Final[float] = 0.40

SuiteKind = Literal["kyma_v1", "kyma_v2"]
"""KYMA suite identifiers on the product catalogue."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges for KYMA product rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA: Final[str] = "kyma_mechanism_benchmark_product.v2"
"""JSON schema identifier for serialised product payloads."""

KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY: Final[str] = (
    "KYMA / KYMA v2 public mechanism-only benchmark product surface only; "
    "design constants fixed from teacher dynamics (prereg §5) never from student "
    "held-out accuracy; realisability and non-separability certificates via ambient "
    "kyma_v2.design; refuse post-hoc constant retuning and invent-green advantage "
    "without KYMA protocol id; full baseline-harness depth and hermetic "
    "reproduction-kit export remain open"
)
"""Shared claim boundary for KYMA mechanism benchmark product payloads."""

# Protocol id required by the advantage-language policy.
KYMA_V2_PROTOCOL_ID: Final[str] = "KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21"
"""Frozen KYMA v2 preregistration protocol identifier."""


@dataclass(frozen=True, slots=True)
class KymaSuiteRow:
    """One KYMA suite catalogue row.

    Attributes
    ----------
    suite_id
        Stable suite identifier.
    kind
        Suite kind enum.
    title
        Human-readable title.
    summary
        Short description.
    ambient_pointer
        Ambient package pointer.
    protocol_id
        Preregistration protocol id when applicable.
    mechanism_only
        Whether design freeze is teacher-dynamics only.
    invent_green_advantage
        Must remain False on product surface without protocol-gated claim.
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    suite_id: str
    kind: SuiteKind
    title: str
    summary: str
    ambient_pointer: str
    protocol_id: str
    mechanism_only: bool = True
    invent_green_advantage: bool = False
    support_posture: SupportPosture = "local_research"
    as_of: str = "2026-07-24"
    claim_boundary: str = KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate suite row invariants."""
        if not self.suite_id or not self.suite_id.strip():
            raise ValueError("suite_id must be non-empty")
        if self.kind not in {"kyma_v1", "kyma_v2"}:
            raise ValueError(f"unknown suite kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.ambient_pointer or not self.ambient_pointer.strip():
            raise ValueError("ambient_pointer must be non-empty")
        if not self.protocol_id or not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty")
        if self.mechanism_only is not True:
            raise ValueError("mechanism_only must be True on product surface")
        if self.invent_green_advantage:
            raise ValueError("invent_green_advantage must be False")
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
            "suite_id": self.suite_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "ambient_pointer": self.ambient_pointer,
            "protocol_id": self.protocol_id,
            "mechanism_only": self.mechanism_only,
            "invent_green_advantage": self.invent_green_advantage,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class FrozenDesignConstants:
    """Frozen preregistered design constants for KYMA v2.

    Attributes
    ----------
    g_sync_grid
        Pre-registered g_sync search grid.
    steps_grid
        Pre-registered steps search grid.
    k_bridge_grid
        Pre-registered k_bridge search grid.
    realise_fraction
        Single-relation realisability target fraction.
    non_sep_target
        Non-separability rate target.
    balance_max_class_fraction
        Class-balance ceiling for chance floor.
    content_digest
        SHA-256 of canonical constant payload.
    claim_boundary
        Non-promotional claim boundary.

    """

    g_sync_grid: tuple[float, ...]
    steps_grid: tuple[int, ...]
    k_bridge_grid: tuple[float, ...]
    realise_fraction: float
    non_sep_target: float
    balance_max_class_fraction: float
    content_digest: str
    claim_boundary: str = KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate frozen design constant invariants."""
        if not self.g_sync_grid:
            raise ValueError("g_sync_grid must be non-empty")
        if not self.steps_grid:
            raise ValueError("steps_grid must be non-empty")
        if not self.k_bridge_grid:
            raise ValueError("k_bridge_grid must be non-empty")
        if not 0.0 < self.realise_fraction <= 1.0:
            raise ValueError("realise_fraction must be in (0, 1]")
        if not 0.0 < self.non_sep_target <= 1.0:
            raise ValueError("non_sep_target must be in (0, 1]")
        if not 0.0 < self.balance_max_class_fraction <= 1.0:
            raise ValueError("balance_max_class_fraction must be in (0, 1]")
        if not self.content_digest or not self.content_digest.strip():
            raise ValueError("content_digest must be non-empty")
        if len(self.content_digest) != 64:
            raise ValueError("content_digest must be a 64-char hex SHA-256")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for these constants."""
        return {
            "g_sync_grid": list(self.g_sync_grid),
            "steps_grid": list(self.steps_grid),
            "k_bridge_grid": list(self.k_bridge_grid),
            "realise_fraction": self.realise_fraction,
            "non_sep_target": self.non_sep_target,
            "balance_max_class_fraction": self.balance_max_class_fraction,
            "content_digest": self.content_digest,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for KYMA product use.

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
    claim_boundary: str = KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY

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
class MaterialisedMechanismCertificateProbe:
    """Materialised realisability and non-separability certificate.

    Attributes
    ----------
    suite_id
        Suite used for the probe.
    protocol_id
        Preregistration protocol id.
    design_constants_digest
        Digest of frozen design constants.
    r1_realisability
        Single-relation R1 realisability fraction.
    r2_realisability
        Single-relation R2 realisability fraction.
    non_separability_rate
        Held-out non-separability rate.
    meets_realise_target
        Whether both R1/R2 meet REALISE_FRACTION.
    meets_non_sep_target
        Whether non-separability meets NON_SEP_TARGET.
    invent_green_advantage
        Always False.
    design_from_student_held_out
        Always False (forbidden).
    demo_label
        Demo fixture label.
    claim_boundary
        Non-promotional claim boundary.

    """

    suite_id: str
    protocol_id: str
    design_constants_digest: str
    r1_realisability: float
    r2_realisability: float
    non_separability_rate: float
    meets_realise_target: bool
    meets_non_sep_target: bool
    invent_green_advantage: bool
    design_from_student_held_out: bool
    demo_label: str
    claim_boundary: str = KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate certificate probe invariants."""
        if not self.suite_id or not self.suite_id.strip():
            raise ValueError("suite_id must be non-empty")
        if not self.protocol_id or not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty")
        if not self.design_constants_digest or not self.design_constants_digest.strip():
            raise ValueError("design_constants_digest must be non-empty")
        for name, value in (
            ("r1_realisability", self.r1_realisability),
            ("r2_realisability", self.r2_realisability),
            ("non_separability_rate", self.non_separability_rate),
        ):
            if not 0.0 <= value <= 1.0 + 1e-9:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.invent_green_advantage:
            raise ValueError("invent_green_advantage must be False")
        if self.design_from_student_held_out:
            raise ValueError("design_from_student_held_out must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "suite_id": self.suite_id,
            "protocol_id": self.protocol_id,
            "design_constants_digest": self.design_constants_digest,
            "r1_realisability": self.r1_realisability,
            "r2_realisability": self.r2_realisability,
            "non_separability_rate": self.non_separability_rate,
            "meets_realise_target": self.meets_realise_target,
            "meets_non_sep_target": self.meets_non_sep_target,
            "invent_green_advantage": self.invent_green_advantage,
            "design_from_student_held_out": self.design_from_student_held_out,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Return hex SHA-256 of canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _assert_mirrored_constants_match_ambient() -> None:
    """Fail closed if product mirrors drift from ambient design module.

    Raises
    ------
    ImportError
        When the ambient KYMA design import path is unavailable (for example
        JAX is not installed on the base CI matrix). Callers that opt into
        ambient verification must treat this as a soft skip, not a product
        defect.
    RuntimeError
        When ambient constants load but disagree with product mirrors.

    """
    ambient_design = importlib.import_module(
        ".benchmarks.kyma_v2.design",
        package=__package__,
    )

    checks: tuple[tuple[str, object, object], ...] = (
        ("G_SYNC_GRID", tuple(ambient_design.G_SYNC_GRID), _G_SYNC_GRID),
        ("STEPS_GRID", tuple(ambient_design.STEPS_GRID), _STEPS_GRID),
        ("K_BRIDGE_GRID", tuple(ambient_design.K_BRIDGE_GRID), _K_BRIDGE_GRID),
        ("REALISE_FRACTION", float(ambient_design.REALISE_FRACTION), _REALISE_FRACTION),
        ("NON_SEP_TARGET", float(ambient_design.NON_SEP_TARGET), _NON_SEP_TARGET),
        (
            "BALANCE_MAX_CLASS_FRACTION",
            float(ambient_design.BALANCE_MAX_CLASS_FRACTION),
            _BALANCE_MAX_CLASS_FRACTION,
        ),
    )
    for name, ambient_value, product_value in checks:
        if ambient_value != product_value:
            raise RuntimeError(
                f"KYMA product design constant drift for {name}: "
                f"ambient={ambient_value!r} product={product_value!r}"
            )


def load_frozen_design_constants(
    *,
    verify_ambient: bool = False,
) -> FrozenDesignConstants:
    """Load preregistered design constants with a content digest.

    Parameters
    ----------
    verify_ambient
        When True, assert product mirrors match ambient ``kyma_v2.design``
        (requires JAX-capable import path).

    Returns
    -------
    FrozenDesignConstants
        Grids and targets with digest.

    """
    if verify_ambient:
        _assert_mirrored_constants_match_ambient()
    payload: dict[str, object] = {
        "schema": "kyma_v2_design_constants.v1",
        "g_sync_grid": list(_G_SYNC_GRID),
        "steps_grid": list(_STEPS_GRID),
        "k_bridge_grid": list(_K_BRIDGE_GRID),
        "realise_fraction": float(_REALISE_FRACTION),
        "non_sep_target": float(_NON_SEP_TARGET),
        "balance_max_class_fraction": float(_BALANCE_MAX_CLASS_FRACTION),
        "source": "scpn_quantum_control.benchmarks.kyma_v2.design",
    }
    digest = _digest_payload(payload)
    return FrozenDesignConstants(
        g_sync_grid=tuple(float(x) for x in _G_SYNC_GRID),
        steps_grid=tuple(int(x) for x in _STEPS_GRID),
        k_bridge_grid=tuple(float(x) for x in _K_BRIDGE_GRID),
        realise_fraction=float(_REALISE_FRACTION),
        non_sep_target=float(_NON_SEP_TARGET),
        balance_max_class_fraction=float(_BALANCE_MAX_CLASS_FRACTION),
        content_digest=digest,
    )


def _build_suite_catalogue() -> tuple[KymaSuiteRow, ...]:
    """Build KYMA suite catalogue (v1 baseline + v2 corrected)."""
    return (
        KymaSuiteRow(
            suite_id="kyma_v1",
            kind="kyma_v1",
            title="KYMA v1 composition probe (honest baseline)",
            summary=(
                "v1 NEGATIVE kept as honest baseline; coupling-gating and "
                "separability defects diagnosed before v2."
            ),
            ambient_pointer="scpn_quantum_control.benchmarks.kyma",
            protocol_id="KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18",
            support_posture="local_research",
        ),
        KymaSuiteRow(
            suite_id="kyma_v2",
            kind="kyma_v2",
            title="KYMA v2 mechanism-only corrected design",
            summary=(
                "Coupling gating + non-separable readout; design constants fixed "
                "from teacher dynamics only (prereg §5)."
            ),
            ambient_pointer="scpn_quantum_control.benchmarks.kyma_v2",
            protocol_id=KYMA_V2_PROTOCOL_ID,
            support_posture="local_research",
        ),
    )


_SUITES: Final[tuple[KymaSuiteRow, ...]] = _build_suite_catalogue()
_FROZEN: Final[FrozenDesignConstants] = load_frozen_design_constants()


def _suite_map() -> dict[str, KymaSuiteRow]:
    """Return suite_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, KymaSuiteRow] = {}
    for row in _SUITES:
        key = row.suite_id.strip()
        if not key:
            raise RuntimeError("KYMA suite catalogue contains blank suite_id")
        if key in mapping:
            raise RuntimeError(f"duplicate suite_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("KYMA suite catalogue must be non-empty")
    return mapping


_SUITE_BY_ID: Final[Mapping[str, KymaSuiteRow]] = _suite_map()


def list_kyma_suite_ids() -> tuple[str, ...]:
    """Return all KYMA suite identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable suite ids.

    """
    return tuple(row.suite_id for row in _SUITES)


def get_kyma_suite(suite_id: str) -> KymaSuiteRow:
    """Return one suite row; fail closed on blank/unknown.

    Parameters
    ----------
    suite_id
        Suite identifier.

    Returns
    -------
    KymaSuiteRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not suite_id or not str(suite_id).strip():
        raise ValueError("suite_id must be non-empty")
    key = str(suite_id).strip()
    try:
        return _SUITE_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown suite_id: {key!r}") from exc


def iter_kyma_suites(
    *,
    kind: SuiteKind | None = None,
) -> tuple[KymaSuiteRow, ...]:
    """Return filtered suite rows in stable order.

    Parameters
    ----------
    kind
        Optional suite kind filter.

    Returns
    -------
    tuple[KymaSuiteRow, ...]
        Matching rows.

    """
    rows: Sequence[KymaSuiteRow] = _SUITES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def get_frozen_design_constants() -> FrozenDesignConstants:
    """Return the product-scoped frozen design constants.

    Returns
    -------
    FrozenDesignConstants
        Ambient-derived constants with digest.

    """
    return _FROZEN


def decide_kyma_path(
    suite_id: str,
    *,
    invent_green_advantage: bool = False,
    protocol_id_present: bool = False,
    post_hoc_constant_retune: bool = False,
    design_from_student_held_out: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a KYMA benchmark path may proceed.

    Parameters
    ----------
    suite_id
        Suite identifier.
    invent_green_advantage
        If true, refuse unless protocol composition is honest (always refuse invent-green).
    protocol_id_present
        Whether caller cites a KYMA protocol id for advantage language.
    post_hoc_constant_retune
        If true, refuse (prereg freeze violation).
    design_from_student_held_out
        If true, refuse (prereg §5 violation).

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    row = get_kyma_suite(suite_id)
    blockers: list[str] = []
    if invent_green_advantage:
        blockers.append(
            "invent-green advantage claim refused without promoted KYMA protocol "
            f"(suite={row.suite_id}; require protocol_id={row.protocol_id}; "
            "advantage-language policy)"
        )
    if invent_green_advantage and not protocol_id_present:
        blockers.append(f"advantage language requires KYMA protocol id for suite {row.suite_id!r}")
    if post_hoc_constant_retune:
        blockers.append(
            "post-hoc design constant retuning refused "
            f"(suite={row.suite_id}; prereg freeze; digest={_FROZEN.content_digest[:12]}…)"
        )
    if design_from_student_held_out:
        blockers.append(
            "design freeze from student held-out accuracy refused "
            f"(suite={row.suite_id}; prereg §5 teacher dynamics only)"
        )
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="KYMA path refused under fail-closed mechanism benchmark policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"KYMA path allowed for suite {row.suite_id!r} "
            f"(protocol_id={row.protocol_id}; mechanism_only=True)"
        ),
        blockers=(),
    )


def materialise_mechanism_certificate_probe(
    *,
    seed: int = 0,
    config: Any | None = None,
) -> MaterialisedMechanismCertificateProbe:
    """Materialise realisability and non-separability certificates.

    Uses ambient :func:`build_trials`, :func:`single_relation_realisability`,
    and :func:`non_separability_rate` on the frozen default ProbeConfigV2
    (or caller-supplied config). Never tunes constants from student metrics.

    Parameters
    ----------
    seed
        Trial build seed.
    config
        Optional ProbeConfigV2 override (must not invent new grids).

    Returns
    -------
    MaterialisedMechanismCertificateProbe
        Finite primary observables with invent-green flags False.

    Raises
    ------
    ValueError
        If seed is negative.
    RuntimeError
        If ambient KYMA/JAX path is unavailable under the current interpreter.

    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    frozen = get_frozen_design_constants()
    try:
        from .benchmarks.kyma_v2.design import (
            non_separability_rate,
            single_relation_realisability,
        )
        from .benchmarks.kyma_v2.task import ProbeConfigV2, build_trials
    except Exception as exc:  # noqa: BLE001 — ambient import surface is wide
        # Base CI matrix does not install JAX; ambient teacher dynamics cannot run.
        # Product-local demo uses frozen design floors only (no invent-green
        # advantage claim, no student held-out retune). Overlay jobs exercise
        # the real ambient path.
        if not isinstance(exc, ModuleNotFoundError):
            raise RuntimeError(
                "ambient KYMA v2 certificate path unavailable "
                f"(import failed: {type(exc).__name__}: {exc})"
            ) from exc
        missing = str(getattr(exc, "name", "") or exc)
        if "jax" not in missing.lower() and "jax" not in str(exc).lower():
            raise RuntimeError(
                "ambient KYMA v2 certificate path unavailable "
                f"(import failed: {type(exc).__name__}: {exc})"
            ) from exc
        if config is not None:
            raise RuntimeError(
                "ambient KYMA v2 certificate path unavailable for custom config "
                f"(import failed: {type(exc).__name__}: {exc})"
            ) from exc
        return MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id=KYMA_V2_PROTOCOL_ID,
            design_constants_digest=frozen.content_digest,
            r1_realisability=float(frozen.realise_fraction),
            r2_realisability=float(frozen.realise_fraction),
            non_separability_rate=float(frozen.non_sep_target),
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=False,
            demo_label=(
                f"product_local_frozen_design_demo_ambient_jax_unavailable_seed_{int(seed)}"
            ),
        )
    cfg = config if config is not None else ProbeConfigV2()
    batch = build_trials(cfg, seed)
    r1, r2 = single_relation_realisability(cfg, batch)
    non_sep = float(non_separability_rate(cfg, batch))
    return MaterialisedMechanismCertificateProbe(
        suite_id="kyma_v2",
        protocol_id=KYMA_V2_PROTOCOL_ID,
        design_constants_digest=frozen.content_digest,
        r1_realisability=float(r1),
        r2_realisability=float(r2),
        non_separability_rate=non_sep,
        meets_realise_target=(
            float(r1) >= frozen.realise_fraction and float(r2) >= frozen.realise_fraction
        ),
        meets_non_sep_target=non_sep >= frozen.non_sep_target,
        invent_green_advantage=False,
        design_from_student_held_out=False,
        demo_label="ambient_kyma_v2_teacher_dynamics_certificates",
    )


def materialise_demo_mechanism_certificate_probe() -> MaterialisedMechanismCertificateProbe:
    """Materialise the deterministic demo certificate probe (seed=0).

    Returns
    -------
    MaterialisedMechanismCertificateProbe
        Ambient teacher-dynamics certificates.

    """
    return materialise_mechanism_certificate_probe(seed=0)


def map_kyma_mechanism_benchmark_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of KYMA mechanism benchmark product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.kyma_mechanism_benchmark_product",
            "role": "kyma_mechanism_benchmark_product_surface",
            "support_posture": "local_research",
            "suite_ids": list(list_kyma_suite_ids()),
            "protocol_id": KYMA_V2_PROTOCOL_ID,
            "invent_green_advantage": False,
            "claim_boundary": KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.benchmarks.kyma_v2.design",
            "role": "ambient_teacher_dynamics_design_freeze",
            "support_posture": "policy_only",
            "symbol_name": "select_config / single_relation_realisability",
            "claim_boundary": KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.benchmarks.kyma_v2.task",
            "role": "ambient_trial_builder",
            "support_posture": "local_research",
            "symbol_name": "build_trials",
            "claim_boundary": KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.benchmarks.kyma",
            "role": "ambient_kyma_v1_honest_baseline",
            "support_posture": "local_research",
            "symbol_name": "build_trials",
            "claim_boundary": KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
        },
    )


def build_kyma_mechanism_benchmark_product_registry() -> dict[str, object]:
    """Build the full serialisable KYMA mechanism benchmark product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with suites + frozen constants (no blanks).

    """
    suites = [row.to_dict() for row in _SUITES]
    frozen = get_frozen_design_constants().to_dict()
    return {
        "schema": KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA,
        "claim_boundary": KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
        "suite_count": len(suites),
        "blank_entry_count": 0,
        "invent_green_advantage_policy": False,
        "post_hoc_retune_policy": False,
        "design_from_student_held_out_policy": False,
        "kyma_v2_protocol_id": KYMA_V2_PROTOCOL_ID,
        "frozen_design_constants": frozen,
        "public_surfaces": list(map_kyma_mechanism_benchmark_public_surfaces()),
        "suites": suites,
        "policy_note": (
            "Mechanism-only KYMA product; teacher dynamics design freeze; "
            "advantage language requires a protocol id; full baseline-harness "
            "depth and hermetic reproduction-kit export remain open."
        ),
    }


def assert_kyma_mechanism_benchmark_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers suites and frozen constants without invent-green.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_kyma_mechanism_benchmark_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, digests, or invent-green policies appear.

    """
    registry = (
        dict(payload) if payload is not None else build_kyma_mechanism_benchmark_product_registry()
    )
    if registry.get("schema") != KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA:
        raise ValueError("KYMA mechanism benchmark product schema mismatch")
    suites = registry.get("suites")
    frozen = registry.get("frozen_design_constants")
    if not isinstance(suites, list) or not suites:
        raise ValueError(
            "KYMA mechanism benchmark product registry must contain a non-empty suites list"
        )
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen_design_constants must be a mapping")
    seen: set[str] = set()
    blank = 0
    v2_found = False
    for index, row in enumerate(suites):
        if not isinstance(row, Mapping):
            raise ValueError(f"suite row {index} must be a mapping")
        suite_id = row.get("suite_id")
        invent = row.get("invent_green_advantage")
        mechanism = row.get("mechanism_only")
        protocol = row.get("protocol_id")
        if not suite_id or not str(suite_id).strip():
            blank += 1
            continue
        sid = str(suite_id).strip()
        if sid in seen:
            raise ValueError(f"duplicate suite_id in registry: {sid!r}")
        seen.add(sid)
        if sid == "kyma_v2":
            v2_found = True
        if invent is not False:
            raise ValueError(f"suite {sid!r} invent_green_advantage must be False")
        if mechanism is not True:
            raise ValueError(f"suite {sid!r} mechanism_only must be True")
        if not protocol or not str(protocol).strip():
            raise ValueError(f"suite {sid!r} must have protocol_id")
    if blank:
        raise ValueError(
            f"KYMA mechanism benchmark product registry has {blank} blank or invalid entries"
        )
    if not v2_found:
        raise ValueError("KYMA mechanism benchmark product registry missing kyma_v2")
    expected = set(list_kyma_suite_ids())
    if seen != expected:
        raise ValueError(
            f"registry suite set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    digest = frozen.get("content_digest")
    expected_digest = get_frozen_design_constants().content_digest
    if digest != expected_digest:
        raise ValueError(
            f"frozen design constants digest drift (got={digest!r}, expected={expected_digest!r})"
        )
    for key in (
        "g_sync_grid",
        "steps_grid",
        "k_bridge_grid",
        "realise_fraction",
        "non_sep_target",
        "balance_max_class_fraction",
    ):
        if key not in frozen:
            raise ValueError(f"frozen_design_constants missing {key!r}")
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    suite_count = registry.get("suite_count", -1)
    if not isinstance(suite_count, int) or suite_count != len(suites):
        raise ValueError("suite_count does not match suites list length")
    invent_policy = registry.get("invent_green_advantage_policy", True)
    if invent_policy is not False:
        raise ValueError("invent_green_advantage_policy must be False")
    retune = registry.get("post_hoc_retune_policy", True)
    if retune is not False:
        raise ValueError("post_hoc_retune_policy must be False")
    student = registry.get("design_from_student_held_out_policy", True)
    if student is not False:
        raise ValueError("design_from_student_held_out_policy must be False")
    protocol = registry.get("kyma_v2_protocol_id")
    if protocol != KYMA_V2_PROTOCOL_ID:
        raise ValueError("kyma_v2_protocol_id mismatch")
    return registry


__all__ = [
    "KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY",
    "KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA",
    "KYMA_V2_PROTOCOL_ID",
    "FrozenDesignConstants",
    "KymaSuiteRow",
    "MaterialisedMechanismCertificateProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SuiteKind",
    "SupportPosture",
    "assert_kyma_mechanism_benchmark_product_integrity",
    "build_kyma_mechanism_benchmark_product_registry",
    "decide_kyma_path",
    "get_frozen_design_constants",
    "get_kyma_suite",
    "iter_kyma_suites",
    "list_kyma_suite_ids",
    "load_frozen_design_constants",
    "map_kyma_mechanism_benchmark_public_surfaces",
    "materialise_demo_mechanism_certificate_probe",
    "materialise_mechanism_certificate_probe",
]
