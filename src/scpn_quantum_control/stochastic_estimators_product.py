# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stochastic estimators product surface
"""Fail-closed **stochastic estimators & policies** product surface.

Productises finite-shot / stochastic gradient estimators as a first-class
product: versioned estimator catalogue (SPSA, score-function, parameter-shift
shot allocation), confidence/failure policy contracts composing hardware-safety honesty,
and dry-run helpers that refuse invent-green live QPU shot runs.

Composes ambient
:mod:`scpn_quantum_control.differentiable_stochastic_estimators` and
:mod:`scpn_quantum_control.differentiable_stochastic_policy` — does **not**
re-architect estimator engines or invent-green full variance/bias experiment
campaigns, which remain an open capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_stochastic_policy import GradientFailurePolicy

EstimatorKind = Literal[
    "spsa",
    "score_function",
    "parameter_shift_shot_allocation",
    "confidence_policy",
]
"""Estimator / policy catalogue kinds."""

SupportPosture = Literal[
    "local_materialised",
    "finite_shot_materialised",
    "policy_only",
    "hardware_refused",
]
"""Support posture badges for product rows."""

DryRunOutcome = Literal["allowed_dry_run", "refused"]
"""Structured dry-run outcomes."""

STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA: Final[str] = "stochastic_estimators_product.v2"
"""JSON schema identifier for serialised product payloads."""

STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY: Final[str] = (
    "Stochastic estimators product surface only; catalogues SPSA, "
    "score-function, and shot-allocation helpers with confidence-policy "
    "contracts; materialised finite-shot uncertainty only; composes the "
    "hardware-safe no-submit and shot-budget policy; does not invent-green live "
    "QPU shot runs or full variance/bias experiment campaigns"
)
"""Shared claim boundary for estimators, policies, and dry-run decisions."""


@dataclass(frozen=True, slots=True)
class StochasticEstimatorRow:
    """One product catalogue row for a stochastic estimator or policy.

    Attributes
    ----------
    estimator_id
        Stable catalogue identifier.
    kind
        Estimator / policy kind.
    title
        Human-readable title.
    summary
        Short description.
    module_path
        Primary ambient module path.
    symbol_name
        Primary ambient symbol.
    support_posture
        Support posture badge.
    allows_hardware_shots
        Whether this product row claims live hardware shots (must be False).
    hardware_safety_pointer
        Hardware-safe no-submit and shot-budget policy pointer.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    estimator_id: str
    kind: EstimatorKind
    title: str
    summary: str
    module_path: str
    symbol_name: str
    support_posture: SupportPosture
    allows_hardware_shots: bool = False
    hardware_safety_pointer: str = "hardware_safe_execution.no_submit_shot_budget"
    as_of: str = "2026-07-24"
    claim_boundary: str = STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate catalogue row invariants."""
        if not self.estimator_id or not self.estimator_id.strip():
            raise ValueError("estimator_id must be non-empty")
        if self.kind not in {
            "spsa",
            "score_function",
            "parameter_shift_shot_allocation",
            "confidence_policy",
        }:
            raise ValueError(f"unknown estimator kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.symbol_name or not self.symbol_name.strip():
            raise ValueError("symbol_name must be non-empty")
        if self.support_posture not in {
            "local_materialised",
            "finite_shot_materialised",
            "policy_only",
            "hardware_refused",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if self.allows_hardware_shots:
            raise ValueError(
                "product estimators must set allows_hardware_shots=False "
                "under the hardware-safe no-submit policy"
            )
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if not self.hardware_safety_pointer or not self.hardware_safety_pointer.strip():
            raise ValueError("hardware_safety_pointer must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "estimator_id": self.estimator_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "support_posture": self.support_posture,
            "allows_hardware_shots": self.allows_hardware_shots,
            "hardware_safety_pointer": self.hardware_safety_pointer,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class EstimatorDryRunDecision:
    """Fail-closed dry-run decision for a product estimator path.

    Attributes
    ----------
    estimator_id
        Estimator validated.
    outcome
        Allowed dry-run or refused.
    allowed
        Whether the dry-run plan may proceed (never means QPU shots ran).
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    planned_shots
        Acknowledged shot budget for the dry-run plan (0 when refused).

    """

    estimator_id: str
    outcome: DryRunOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    planned_shots: int
    claim_boundary: str = STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate dry-run decision invariants."""
        if not self.estimator_id or not self.estimator_id.strip():
            raise ValueError("estimator_id must be non-empty")
        if self.outcome not in {"allowed_dry_run", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed_dry_run":
            raise ValueError("allowed decisions must use outcome=allowed_dry_run")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.planned_shots < 0:
            raise ValueError("planned_shots must be non-negative")
        if self.allowed and self.planned_shots <= 0:
            raise ValueError("allowed decisions require positive planned_shots")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "estimator_id": self.estimator_id,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "planned_shots": self.planned_shots,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedSPSAProbe:
    """Materialised local SPSA probe result for product contracts.

    Attributes
    ----------
    gradient
        Estimated gradient vector.
    seed
        Seed used for the deterministic materialised probe.
    repetitions
        Number of SPSA repetition pairs.
    shots
        Finite-shot count used (None when infinite/analytic materialisation).
    max_abs_gradient
        Maximum absolute gradient entry (primary observable).

    """

    gradient: tuple[float, ...]
    seed: int
    repetitions: int
    shots: int | None
    max_abs_gradient: float
    claim_boundary: str = STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised probe invariants."""
        if not self.gradient:
            raise ValueError("gradient must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")
        if self.max_abs_gradient < 0.0:
            raise ValueError("max_abs_gradient must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "gradient": list(self.gradient),
            "seed": self.seed,
            "repetitions": self.repetitions,
            "shots": self.shots,
            "max_abs_gradient": self.max_abs_gradient,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    estimator_id: str,
    *,
    kind: EstimatorKind,
    title: str,
    summary: str,
    module_path: str,
    symbol_name: str,
    support_posture: SupportPosture,
) -> StochasticEstimatorRow:
    """Build one catalogue row."""
    return StochasticEstimatorRow(
        estimator_id=estimator_id,
        kind=kind,
        title=title,
        summary=summary,
        module_path=module_path,
        symbol_name=symbol_name,
        support_posture=support_posture,
    )


_CANONICAL_ESTIMATORS: Final[tuple[StochasticEstimatorRow, ...]] = (
    _row(
        "spsa_gradient",
        kind="spsa",
        title="SPSA gradient estimate",
        summary=(
            "Seeded simultaneous-perturbation stochastic approximation gradient "
            "with optional finite-shot variance metadata."
        ),
        module_path="scpn_quantum_control.differentiable_stochastic_estimators",
        symbol_name="spsa_gradient_estimate",
        support_posture="finite_shot_materialised",
    ),
    _row(
        "score_function_gradient",
        kind="score_function",
        title="Score-function gradient estimate",
        summary=(
            "Score-function (REINFORCE-style) gradient estimate with materialised "
            "sample records and uncertainty metadata."
        ),
        module_path="scpn_quantum_control.differentiable_stochastic_estimators",
        symbol_name="score_function_gradient_estimate",
        support_posture="finite_shot_materialised",
    ),
    _row(
        "parameter_shift_shot_allocation",
        kind="parameter_shift_shot_allocation",
        title="Parameter-shift shot allocation",
        summary=(
            "Allocate finite shots across parameter-shift evaluations under an "
            "explicit budget (materialised plan; no hardware submission)."
        ),
        module_path="scpn_quantum_control.differentiable_stochastic_estimators",
        symbol_name="allocate_parameter_shift_shots",
        support_posture="local_materialised",
    ),
    _row(
        "gradient_failure_policy",
        kind="confidence_policy",
        title="Gradient failure / confidence policy",
        summary=(
            "Fail-closed uncertainty policy contracts (max SE / confidence radius) "
            "composing hardware-safe shot-budget honesty for materialised "
            "stochastic gradients."
        ),
        module_path="scpn_quantum_control.differentiable_stochastic_policy",
        symbol_name="GradientFailurePolicy",
        support_posture="policy_only",
    ),
)


def _catalogue_map() -> dict[str, StochasticEstimatorRow]:
    """Return estimator_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, StochasticEstimatorRow] = {}
    for row in _CANONICAL_ESTIMATORS:
        key = row.estimator_id.strip()
        if not key:
            raise RuntimeError("stochastic estimators catalogue contains blank estimator_id")
        if key in mapping:
            raise RuntimeError(f"duplicate estimator_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("stochastic estimators catalogue must be non-empty")
    return mapping


_ESTIMATOR_BY_ID: Final[Mapping[str, StochasticEstimatorRow]] = _catalogue_map()


def list_stochastic_estimator_ids() -> tuple[str, ...]:
    """Return all product estimator identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered estimator identifiers.

    """
    return tuple(row.estimator_id for row in _CANONICAL_ESTIMATORS)


def get_stochastic_estimator(estimator_id: str) -> StochasticEstimatorRow:
    """Return one estimator row or raise for blank/unknown identifiers.

    Parameters
    ----------
    estimator_id
        Catalogue estimator key.

    Returns
    -------
    StochasticEstimatorRow
        Matching row.

    Raises
    ------
    ValueError
        If ``estimator_id`` is blank or unknown (fail closed).

    """
    if not estimator_id or not str(estimator_id).strip():
        raise ValueError("estimator_id must be a non-empty string")
    key = str(estimator_id).strip()
    try:
        return _ESTIMATOR_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown estimator_id {key!r}; refuse invent-green stochastic "
            f"estimator product claim (known_count={len(_ESTIMATOR_BY_ID)})"
        ) from exc


def iter_stochastic_estimators(
    *,
    kind: EstimatorKind | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[StochasticEstimatorRow, ...]:
    """Return filtered estimator rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[StochasticEstimatorRow, ...]
        Matching rows.

    """
    rows: Sequence[StochasticEstimatorRow] = _CANONICAL_ESTIMATORS
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def build_product_failure_policy(
    *,
    max_standard_error: float | None = None,
    max_confidence_radius: float | None = None,
    require_trainable: bool = True,
) -> GradientFailurePolicy:
    """Build a product-scoped gradient failure policy.

    Parameters
    ----------
    max_standard_error
        Optional positive max standard-error threshold.
    max_confidence_radius
        Optional positive max confidence-radius threshold.
    require_trainable
        Whether non-trainable parameters must be present/consistent.

    Returns
    -------
    GradientFailurePolicy
        Ambient fail-closed policy object.

    """
    return GradientFailurePolicy(
        max_standard_error=max_standard_error,
        max_confidence_radius=max_confidence_radius,
        require_trainable=require_trainable,
    )


def dry_run_stochastic_estimator(
    estimator_id: str,
    *,
    planned_shots: int = 100,
    request_hardware_shots: bool = False,
) -> EstimatorDryRunDecision:
    """Acknowledge a materialised finite-shot dry-run plan (no QPU execution).

    Parameters
    ----------
    estimator_id
        Catalogue estimator key.
    planned_shots
        Positive planned shot budget for the dry-run plan.
    request_hardware_shots
        When true, refuse under the hardware-safe no-submit policy.

    Returns
    -------
    EstimatorDryRunDecision
        Allowed dry-run or refused decision.

    Raises
    ------
    ValueError
        If ``estimator_id`` is blank/unknown or ``planned_shots`` is invalid
        when not refused for hardware first.

    """
    row = get_stochastic_estimator(estimator_id)
    blockers: list[str] = []
    if request_hardware_shots or row.allows_hardware_shots:
        blockers.append(
            "hardware/QPU shot request refused on stochastic estimators product "
            f"(hardware-safe policy pointer={row.hardware_safety_pointer})"
        )
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return EstimatorDryRunDecision(
            estimator_id=row.estimator_id,
            outcome="refused",
            allowed=False,
            reason="stochastic estimators product refuse: " + "; ".join(unique),
            blockers=unique,
            planned_shots=0,
        )

    if not isinstance(planned_shots, int) or isinstance(planned_shots, bool) or planned_shots <= 0:
        raise ValueError("planned_shots must be a positive integer")

    return EstimatorDryRunDecision(
        estimator_id=row.estimator_id,
        outcome="allowed_dry_run",
        allowed=True,
        reason=(
            f"materialised dry-run plan for {row.estimator_id!r} allowed "
            f"(planned_shots={planned_shots}, posture={row.support_posture}, "
            f"module={row.module_path}); no QPU submission occurred"
        ),
        blockers=(),
        planned_shots=planned_shots,
    )


def materialise_demo_spsa_probe(
    *,
    values: ArrayLike | None = None,
    seed: int = 0,
    repetitions: int = 2,
    perturbation_radius: float = 0.1,
) -> MaterialisedSPSAProbe:
    """Run a deterministic local SPSA probe on a quadratic demo objective.

    Uses ambient :func:`spsa_gradient_estimate` on ``f(x) = sum(x_i^2)`` so the
    true gradient is ``2x``. No hardware shots; infinite/analytic materialisation.

    Parameters
    ----------
    values
        Parameter vector (default ``[0.5, -0.25]``).
    seed
        Non-negative SPSA seed.
    repetitions
        Positive SPSA repetition count.
    perturbation_radius
        Positive SPSA radius.

    Returns
    -------
    MaterialisedSPSAProbe
        Gradient and metadata with non-empty primary observables.

    Raises
    ------
    ValueError
        If ambient SPSA validation fails.

    """
    from .differentiable_stochastic_estimators import spsa_gradient_estimate

    x = np.asarray([0.5, -0.25] if values is None else values, dtype=np.float64)

    def objective(params: NDArray[np.float64]) -> float:
        arr = np.asarray(params, dtype=np.float64)
        return float(np.sum(arr * arr))

    result = spsa_gradient_estimate(
        objective,
        x,
        perturbation_radius=perturbation_radius,
        repetitions=repetitions,
        seed=seed,
        shots=None,
    )
    gradient = tuple(float(v) for v in np.asarray(result.gradient, dtype=np.float64).ravel())
    if not gradient:
        raise ValueError("SPSA probe returned empty gradient")
    max_abs = float(max(abs(v) for v in gradient))
    return MaterialisedSPSAProbe(
        gradient=gradient,
        seed=seed,
        repetitions=repetitions,
        shots=None,
        max_abs_gradient=max_abs,
    )


def map_stochastic_estimators_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of stochastic estimator product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for estimator in _CANONICAL_ESTIMATORS:
        path = estimator.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "stochastic_estimators_product_surface",
                "support_posture": estimator.support_posture,
                "estimator_ids": [
                    e.estimator_id for e in _CANONICAL_ESTIMATORS if e.module_path == path
                ],
                "hardware_safety_pointer": estimator.hardware_safety_pointer,
                "claim_boundary": STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_stochastic_estimators_product_registry() -> dict[str, object]:
    """Build the full serialisable stochastic estimators product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with estimators (no blanks).

    """
    estimators = [row.to_dict() for row in _CANONICAL_ESTIMATORS]
    return {
        "schema": STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA,
        "claim_boundary": STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY,
        "estimator_count": len(estimators),
        "blank_entry_count": 0,
        "default_estimator_id": "spsa_gradient",
        "public_surfaces": list(map_stochastic_estimators_public_surfaces()),
        "estimators": estimators,
        "policy_note": (
            "Stochastic estimators product catalogue only; ambient SPSA / "
            "score-function / shot-allocation engines remain the implementation; "
            "full variance/bias experiment campaigns remain open; no invent-green "
            "live QPU shot runs under the hardware-safe no-submit policy."
        ),
    }


def assert_stochastic_estimators_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers estimators without blanks or invent-hardware.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_stochastic_estimators_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-hardware rows appear.

    """
    registry = (
        dict(payload) if payload is not None else build_stochastic_estimators_product_registry()
    )
    if registry.get("schema") != STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA:
        raise ValueError("stochastic estimators product schema mismatch")
    estimators = registry.get("estimators")
    if not isinstance(estimators, list) or not estimators:
        raise ValueError(
            "stochastic estimators product registry must contain a non-empty estimators list"
        )
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(estimators):
        if not isinstance(row, Mapping):
            raise ValueError(f"estimator row {index} must be a mapping")
        estimator_id = row.get("estimator_id")
        kind = row.get("kind")
        allows_hardware = row.get("allows_hardware_shots")
        symbol_name = row.get("symbol_name")
        if not estimator_id or not str(estimator_id).strip():
            blank += 1
            continue
        eid = str(estimator_id).strip()
        if eid in seen:
            raise ValueError(f"duplicate estimator_id in registry: {eid!r}")
        seen.add(eid)
        if eid == "spsa_gradient":
            default_found = True
        if kind not in {
            "spsa",
            "score_function",
            "parameter_shift_shot_allocation",
            "confidence_policy",
        }:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"estimator {eid!r} must have symbol_name")
        if allows_hardware is True:
            raise ValueError(
                f"estimator {eid!r} invent-green hardware shots: product rows "
                "must set allows_hardware_shots=False"
            )
    if blank:
        raise ValueError(
            f"stochastic estimators product registry has {blank} blank or invalid entries"
        )
    if not default_found:
        raise ValueError("stochastic estimators product registry missing spsa_gradient")
    expected = set(list_stochastic_estimator_ids())
    if seen != expected:
        raise ValueError(
            f"registry estimator set drift (missing={expected - seen!r}, "
            f"extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    estimator_count = registry.get("estimator_count", -1)
    if not isinstance(estimator_count, int) or estimator_count != len(estimators):
        raise ValueError("estimator_count does not match estimators list length")
    return registry


__all__ = [
    "STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY",
    "STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA",
    "DryRunOutcome",
    "EstimatorDryRunDecision",
    "EstimatorKind",
    "MaterialisedSPSAProbe",
    "StochasticEstimatorRow",
    "SupportPosture",
    "assert_stochastic_estimators_product_integrity",
    "build_product_failure_policy",
    "build_stochastic_estimators_product_registry",
    "dry_run_stochastic_estimator",
    "get_stochastic_estimator",
    "iter_stochastic_estimators",
    "list_stochastic_estimator_ids",
    "map_stochastic_estimators_public_surfaces",
    "materialise_demo_spsa_probe",
]
