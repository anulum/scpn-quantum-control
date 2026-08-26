# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adjoint reversible replay product surface
"""Fail-closed **adjoint differentiation via reversible replay** product.

Productises reverse-mode adjoint-via-replay contracts over ambient Program AD
adjoint generation and executable replay:

* versioned surface catalogue (reversibility, checkpoint policy, reverse grad,
  executable replay, irreversible refuse);
* serialisable checkpoint-policy objects;
* materialised unitary-style scalar demos via ambient
  :func:`program_adjoint_value_and_grad` /
  :func:`program_adjoint_replay_gradient`;
* fail-closed refuse for mid-circuit measurement, irreversible ops, invent-green
  Catalyst parity, and hardware adjoint claims.

Does **not** re-architect ambient engines, invent Catalyst-class reverse
simulators, or claim hardware adjoint; the reporting, planner-registration,
usage-guide, and open-system-reverse capabilities remain open.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import ArrayLike

SurfaceKind = Literal[
    "reversibility_conditions",
    "checkpoint_policy",
    "reverse_adjoint_grad",
    "executable_replay",
    "irreversible_refuse",
    "catalyst_hardware_refuse",
]
"""Catalogue kinds for adjoint-replay product rows."""

SupportPosture = Literal[
    "local_materialised",
    "policy_only",
    "refuse_only",
]
"""Support posture badges for product rows."""

CheckpointSchedule = Literal["every_k", "binomial", "fixed_budget"]
"""Supported checkpoint schedule kinds (contract only)."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

ADJOINT_REPLAY_PRODUCT_SCHEMA: Final[str] = "adjoint_replay_product.v2"
"""JSON schema identifier for serialised product payloads."""

ADJOINT_REPLAY_CLAIM_BOUNDARY: Final[str] = (
    "Adjoint reversible-replay product surface only; catalogues reversibility "
    "conditions, checkpoint policies, reverse-mode Program AD gradient, and "
    "executable adjoint step replay over ambient program_ad_adjoint; "
    "materialised local scalar demos only; refuses mid-circuit measurement / "
    "irreversible ops invent-green, Catalyst parity, and hardware adjoint; "
    "does not invent full automatic checkpointing, open-system reverse, or "
    "planner-registration coverage"
)
"""Shared claim boundary for adjoint-replay product payloads."""


def _demo_quadratic_objective(values: Any) -> object:
    """Module-level scalar objective ``x**2 + y**2`` for materialised demos.

    Defined at module scope so whole-program AD source frontend can recover
    source regions (interactive lambdas fail ``source_frontend_missing``).
    """
    x, y = values
    return x**2 + y**2


@dataclass(frozen=True, slots=True)
class AdjointReplaySurfaceRow:
    """One product catalogue row for an adjoint-replay surface.

    Attributes
    ----------
    surface_id
        Stable catalogue identifier.
    kind
        Surface kind.
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
    allows_catalyst_parity
        Must be False (no invent-green Catalyst parity).
    allows_hardware_adjoint
        Must be False (no invent-green hardware adjoint).
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    surface_id: str
    kind: SurfaceKind
    title: str
    summary: str
    module_path: str
    symbol_name: str
    support_posture: SupportPosture
    allows_catalyst_parity: bool = False
    allows_hardware_adjoint: bool = False
    as_of: str = "2026-07-24"
    claim_boundary: str = ADJOINT_REPLAY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate catalogue row invariants."""
        if not self.surface_id or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if self.kind not in {
            "reversibility_conditions",
            "checkpoint_policy",
            "reverse_adjoint_grad",
            "executable_replay",
            "irreversible_refuse",
            "catalyst_hardware_refuse",
        }:
            raise ValueError(f"unknown surface kind: {self.kind!r}")
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
            "policy_only",
            "refuse_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if self.allows_catalyst_parity:
            raise ValueError(
                "product surfaces must set allows_catalyst_parity=False "
                "(no invent-green Catalyst parity)"
            )
        if self.allows_hardware_adjoint:
            raise ValueError(
                "product surfaces must set allows_hardware_adjoint=False "
                "(no invent-green hardware adjoint)"
            )
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "support_posture": self.support_posture,
            "allows_catalyst_parity": self.allows_catalyst_parity,
            "allows_hardware_adjoint": self.allows_hardware_adjoint,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Checkpoint schedule contract for reverse-mode memory trade-offs.

    Attributes
    ----------
    schedule
        Schedule kind (every-k, binomial, or fixed budget).
    interval_k
        Positive interval for ``every_k`` (ignored otherwise).
    max_checkpoints
        Positive max checkpoint count for ``fixed_budget`` / binomial cap.
    recompute_allowed
        Whether recompute segments between checkpoints are allowed.

    """

    schedule: CheckpointSchedule
    interval_k: int = 1
    max_checkpoints: int = 8
    recompute_allowed: bool = True
    claim_boundary: str = ADJOINT_REPLAY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate checkpoint policy invariants."""
        if self.schedule not in {"every_k", "binomial", "fixed_budget"}:
            raise ValueError(f"unknown checkpoint schedule: {self.schedule!r}")
        if self.interval_k <= 0:
            raise ValueError("interval_k must be positive")
        if self.max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this policy."""
        return {
            "schedule": self.schedule,
            "interval_k": self.interval_k,
            "max_checkpoints": self.max_checkpoints,
            "recompute_allowed": self.recompute_allowed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ReversibilityReport:
    """Reversibility predicate report for a candidate reverse path.

    Attributes
    ----------
    reversible
        Whether the path is accepted for adjoint replay product use.
    supported_ops
        Supported op labels (may be empty when refused).
    blockers
        Non-empty when not reversible.
    reason
        Human-readable reason.

    """

    reversible: bool
    supported_ops: tuple[str, ...]
    blockers: tuple[str, ...]
    reason: str
    claim_boundary: str = ADJOINT_REPLAY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate reversibility report invariants."""
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.reversible and self.blockers:
            raise ValueError("reversible reports cannot list blockers")
        if not self.reversible and not self.blockers:
            raise ValueError("non-reversible reports require blockers")
        if any(not item or not str(item).strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if any(not item or not str(item).strip() for item in self.supported_ops):
            raise ValueError("supported_ops entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this report."""
        return {
            "reversible": self.reversible,
            "supported_ops": list(self.supported_ops),
            "blockers": list(self.blockers),
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for adjoint-replay product use.

    Attributes
    ----------
    outcome
        Allowed or refused.
    allowed
        Whether reverse adjoint replay may proceed under this product.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = ADJOINT_REPLAY_CLAIM_BOUNDARY

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
class MaterialisedAdjointReplayProbe:
    """Materialised local reverse-adjoint + executable replay probe.

    Attributes
    ----------
    values
        Evaluation parameter vector.
    value
        Objective value.
    adjoint_gradient
        Reverse-mode adjoint generation gradient.
    replay_gradient
        Gradient from executable adjoint step replay.
    agreement_max_abs
        ``max|adjoint - replay|`` residual.
    replay_node_count
        Ambient replay node count certificate field.
    supported
        Whether ambient adjoint generation reported support.
    demo_label
        Which demo was materialised.

    """

    values: tuple[float, ...]
    value: float
    adjoint_gradient: tuple[float, ...]
    replay_gradient: tuple[float, ...]
    agreement_max_abs: float
    replay_node_count: int
    supported: bool
    demo_label: str
    claim_boundary: str = ADJOINT_REPLAY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised probe invariants."""
        if not self.values:
            raise ValueError("values must be non-empty")
        if not self.adjoint_gradient:
            raise ValueError("adjoint_gradient must be non-empty")
        if len(self.adjoint_gradient) != len(self.values):
            raise ValueError("adjoint_gradient length must match values")
        if len(self.replay_gradient) != len(self.values):
            raise ValueError("replay_gradient length must match values")
        if not np.isfinite(self.value):
            raise ValueError("value must be finite")
        if self.agreement_max_abs < 0.0 or not np.isfinite(self.agreement_max_abs):
            raise ValueError("agreement_max_abs must be finite and non-negative")
        if self.replay_node_count < 0:
            raise ValueError("replay_node_count must be non-negative")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")
        if not self.supported:
            raise ValueError("materialised probe requires supported ambient adjoint")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "values": list(self.values),
            "value": self.value,
            "adjoint_gradient": list(self.adjoint_gradient),
            "replay_gradient": list(self.replay_gradient),
            "agreement_max_abs": self.agreement_max_abs,
            "replay_node_count": self.replay_node_count,
            "supported": self.supported,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    surface_id: str,
    *,
    kind: SurfaceKind,
    title: str,
    summary: str,
    module_path: str,
    symbol_name: str,
    support_posture: SupportPosture,
) -> AdjointReplaySurfaceRow:
    """Build one catalogue row."""
    return AdjointReplaySurfaceRow(
        surface_id=surface_id,
        kind=kind,
        title=title,
        summary=summary,
        module_path=module_path,
        symbol_name=symbol_name,
        support_posture=support_posture,
    )


_CANONICAL_SURFACES: Final[tuple[AdjointReplaySurfaceRow, ...]] = (
    _row(
        "reversibility_conditions",
        kind="reversibility_conditions",
        title="Reversibility condition predicates",
        summary=(
            "Product predicates that accept unitary/supported Program AD IR and "
            "refuse mid-circuit measurement and irreversible ops."
        ),
        module_path="scpn_quantum_control.adjoint_replay_product",
        symbol_name="assess_reversibility",
        support_posture="policy_only",
    ),
    _row(
        "checkpoint_policy",
        kind="checkpoint_policy",
        title="Checkpoint schedule policy",
        summary=(
            "Serialisable checkpoint schedule contracts (every-k, binomial, "
            "fixed budget) for memory/recompute trade-offs."
        ),
        module_path="scpn_quantum_control.adjoint_replay_product",
        symbol_name="CheckpointPolicy",
        support_posture="policy_only",
    ),
    _row(
        "reverse_adjoint_grad",
        kind="reverse_adjoint_grad",
        title="Reverse-mode Program AD adjoint gradient",
        summary=(
            "Ambient reverse-mode adjoint generation gradient for supported "
            "captured Program AD IR."
        ),
        module_path="scpn_quantum_control.program_ad_adjoint",
        symbol_name="program_adjoint_value_and_grad",
        support_posture="local_materialised",
    ),
    _row(
        "executable_adjoint_replay",
        kind="executable_replay",
        title="Executable adjoint step replay",
        summary=(
            "Execute generated ProgramADAdjointStep stream and return the "
            "replayed gradient with agreement certificate against attached gradient."
        ),
        module_path="scpn_quantum_control.program_ad_adjoint",
        symbol_name="program_adjoint_replay_gradient",
        support_posture="local_materialised",
    ),
    _row(
        "irreversible_mid_circuit_refuse",
        kind="irreversible_refuse",
        title="Refuse mid-circuit measurement / irreversible ops",
        summary=(
            "Fail-closed refuse when mid-circuit measurement or irreversible "
            "ops are present — silent real-mode substitution forbidden."
        ),
        module_path="scpn_quantum_control.adjoint_replay_product",
        symbol_name="decide_adjoint_replay_path",
        support_posture="refuse_only",
    ),
    _row(
        "catalyst_hardware_adjoint_refuse",
        kind="catalyst_hardware_refuse",
        title="Refuse Catalyst parity / hardware adjoint claims",
        summary=(
            "Explicit refuse for invent-green Catalyst adjoint-jacobian parity "
            "and hardware/QPU adjoint execution claims."
        ),
        module_path="scpn_quantum_control.adjoint_replay_product",
        symbol_name="decide_adjoint_replay_path",
        support_posture="refuse_only",
    ),
)


def _catalogue_map() -> dict[str, AdjointReplaySurfaceRow]:
    """Return surface_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, AdjointReplaySurfaceRow] = {}
    for row in _CANONICAL_SURFACES:
        key = row.surface_id.strip()
        if not key:
            raise RuntimeError("adjoint-replay catalogue contains blank surface_id")
        if key in mapping:
            raise RuntimeError(f"duplicate surface_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("adjoint-replay catalogue must be non-empty")
    return mapping


_SURFACE_BY_ID: Final[Mapping[str, AdjointReplaySurfaceRow]] = _catalogue_map()


def list_adjoint_replay_surface_ids() -> tuple[str, ...]:
    """Return all product surface identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered surface identifiers.

    """
    return tuple(row.surface_id for row in _CANONICAL_SURFACES)


def get_adjoint_replay_surface(surface_id: str) -> AdjointReplaySurfaceRow:
    """Return one surface row or raise for blank/unknown identifiers.

    Parameters
    ----------
    surface_id
        Catalogue surface key.

    Returns
    -------
    AdjointReplaySurfaceRow
        Matching row.

    Raises
    ------
    ValueError
        If ``surface_id`` is blank or unknown (fail closed).

    """
    if not surface_id or not str(surface_id).strip():
        raise ValueError("surface_id must be a non-empty string")
    key = str(surface_id).strip()
    try:
        return _SURFACE_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown surface_id {key!r}; refuse invent-green adjoint-replay "
            f"product claim (known_count={len(_SURFACE_BY_ID)})"
        ) from exc


def iter_adjoint_replay_surfaces(
    *,
    kind: SurfaceKind | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[AdjointReplaySurfaceRow, ...]:
    """Return filtered surface rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[AdjointReplaySurfaceRow, ...]
        Matching rows.

    """
    rows: Sequence[AdjointReplaySurfaceRow] = _CANONICAL_SURFACES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def build_checkpoint_policy(
    *,
    schedule: CheckpointSchedule = "every_k",
    interval_k: int = 1,
    max_checkpoints: int = 8,
    recompute_allowed: bool = True,
) -> CheckpointPolicy:
    """Build a product-scoped checkpoint policy contract.

    Parameters
    ----------
    schedule
        Checkpoint schedule kind.
    interval_k
        Positive interval for ``every_k``.
    max_checkpoints
        Positive max checkpoint count.
    recompute_allowed
        Whether recompute between checkpoints is allowed.

    Returns
    -------
    CheckpointPolicy
        Validated policy object.

    """
    return CheckpointPolicy(
        schedule=schedule,
        interval_k=interval_k,
        max_checkpoints=max_checkpoints,
        recompute_allowed=recompute_allowed,
    )


def assess_reversibility(
    *,
    has_mid_circuit_measurement: bool = False,
    has_irreversible_ops: bool = False,
    has_supported_unitary_ir: bool = True,
) -> ReversibilityReport:
    """Assess whether a path is reversible under product predicates.

    Parameters
    ----------
    has_mid_circuit_measurement
        When true, refuse (irreversible measurement).
    has_irreversible_ops
        When true, refuse (noise/reset/etc. without reverse).
    has_supported_unitary_ir
        Whether a supported unitary / Program AD IR path is declared.

    Returns
    -------
    ReversibilityReport
        Reversible report or refuse with blockers.

    """
    blockers: list[str] = []
    if has_mid_circuit_measurement:
        blockers.append(
            "mid-circuit measurement present; reverse adjoint replay refused "
            "(fail-closed irreversible boundary)"
        )
    if has_irreversible_ops:
        blockers.append("irreversible ops present without reverse map; adjoint replay refused")
    if not has_supported_unitary_ir:
        blockers.append("no supported unitary/Program AD IR declared for reverse adjoint replay")
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return ReversibilityReport(
            reversible=False,
            supported_ops=(),
            blockers=unique,
            reason="path not reversible under product predicates: " + "; ".join(unique),
        )
    return ReversibilityReport(
        reversible=True,
        supported_ops=(
            "program_ad_supported_ir",
            "unitary_scalar_objective",
            "executable_adjoint_step_replay",
        ),
        blockers=(),
        reason=(
            "path reversible under product predicates for supported Program AD "
            "IR / unitary scalar objectives"
        ),
    )


def decide_adjoint_replay_path(
    *,
    has_mid_circuit_measurement: bool = False,
    has_irreversible_ops: bool = False,
    has_supported_unitary_ir: bool = True,
    request_catalyst_parity: bool = False,
    request_hardware_adjoint: bool = False,
) -> PathEligibilityDecision:
    """Decide whether adjoint-replay product path may proceed.

    Parameters
    ----------
    has_mid_circuit_measurement
        Mid-circuit measurement flag.
    has_irreversible_ops
        Irreversible ops flag.
    has_supported_unitary_ir
        Supported IR declaration flag.
    request_catalyst_parity
        When true, refuse invent-green Catalyst parity.
    request_hardware_adjoint
        When true, refuse invent-green hardware adjoint.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused decision with blockers.

    """
    blockers: list[str] = []
    report = assess_reversibility(
        has_mid_circuit_measurement=has_mid_circuit_measurement,
        has_irreversible_ops=has_irreversible_ops,
        has_supported_unitary_ir=has_supported_unitary_ir,
    )
    if not report.reversible:
        blockers.extend(report.blockers)
    if request_catalyst_parity:
        blockers.append(
            "Catalyst adjoint-jacobian parity claim refused "
            "(out of product scope; no invent-green Catalyst parity)"
        )
    if request_hardware_adjoint:
        blockers.append(
            "hardware/QPU adjoint execution claim refused (no invent-green hardware adjoint)"
        )
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="adjoint-replay product refuse: " + "; ".join(unique),
            blockers=unique,
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            "adjoint-replay product path allowed for local materialised "
            "Program AD reverse/replay (no Catalyst/hardware claim)"
        ),
        blockers=(),
    )


def materialise_demo_adjoint_replay_probe(
    *,
    values: ArrayLike | None = None,
) -> MaterialisedAdjointReplayProbe:
    """Materialise reverse adjoint + executable replay on ``x**2+y**2``.

    Uses ambient whole-program AD + :func:`program_adjoint_gradient` and
    :func:`program_adjoint_replay_gradient`. True gradient at default
    ``[0.5, -0.25]`` is ``[1.0, -0.5]``.

    Parameters
    ----------
    values
        Parameter vector (default ``[0.5, -0.25]``).

    Returns
    -------
    MaterialisedAdjointReplayProbe
        Value, adjoint and replay gradients with agreement residual.

    Raises
    ------
    ValueError
        If ambient adjoint generation/replay fails or path is refused.

    """
    from .differentiable import whole_program_value_and_grad
    from .program_ad_adjoint import (
        program_adjoint_gradient,
        program_adjoint_replay_gradient,
    )

    decision = decide_adjoint_replay_path(has_supported_unitary_ir=True)
    if not decision.allowed:
        raise ValueError(f"demo path refused: {decision.reason}")

    x = np.asarray([0.5, -0.25] if values is None else values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("values must be a one-dimensional vector of length >= 2")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")

    result = whole_program_value_and_grad(
        _demo_quadratic_objective,
        x,
        trace=False,
    )
    if result.adjoint_result is None:
        raise ValueError("whole-program result missing adjoint metadata")
    if not result.adjoint_result.supported:
        raise ValueError(
            "ambient adjoint generation unsupported for demo objective: "
            + ", ".join(result.adjoint_result.unsupported_ops)
        )

    adjoint_grad = program_adjoint_gradient(result)
    replay_grad = program_adjoint_replay_gradient(result)
    adjoint_t = tuple(float(v) for v in np.asarray(adjoint_grad, dtype=np.float64).ravel())
    replay_t = tuple(float(v) for v in np.asarray(replay_grad, dtype=np.float64).ravel())
    agreement = float(np.max(np.abs(np.asarray(adjoint_t) - np.asarray(replay_t))))
    return MaterialisedAdjointReplayProbe(
        values=tuple(float(v) for v in x.ravel()),
        value=float(result.value),
        adjoint_gradient=adjoint_t,
        replay_gradient=replay_t,
        agreement_max_abs=agreement,
        replay_node_count=int(result.adjoint_result.replay_node_count),
        supported=True,
        demo_label="quadratic_sum_of_squares",
    )


def map_adjoint_replay_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of adjoint-replay product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for surface in _CANONICAL_SURFACES:
        path = surface.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "adjoint_replay_product_surface",
                "support_posture": surface.support_posture,
                "surface_ids": [
                    s.surface_id for s in _CANONICAL_SURFACES if s.module_path == path
                ],
                "allows_catalyst_parity": False,
                "allows_hardware_adjoint": False,
                "claim_boundary": ADJOINT_REPLAY_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_adjoint_replay_product_registry() -> dict[str, object]:
    """Build the full serialisable adjoint-replay product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with surfaces (no blanks).

    """
    surfaces = [row.to_dict() for row in _CANONICAL_SURFACES]
    return {
        "schema": ADJOINT_REPLAY_PRODUCT_SCHEMA,
        "claim_boundary": ADJOINT_REPLAY_CLAIM_BOUNDARY,
        "surface_count": len(surfaces),
        "blank_entry_count": 0,
        "default_surface_id": "reverse_adjoint_grad",
        "public_surfaces": list(map_adjoint_replay_public_surfaces()),
        "surfaces": surfaces,
        "policy_note": (
            "Adjoint reversible-replay product catalogue only; ambient "
            "program_ad_adjoint remains the reverse/replay implementation; "
            "full checkpoint-schedule campaign, reporting, planner registration, "
            "usage guide, and open-system reverse remain open; no invent-green "
            "Catalyst/hardware."
        ),
    }


def assert_adjoint_replay_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers surfaces without blanks or invent-green.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_adjoint_replay_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green flags appear.

    """
    registry = dict(payload) if payload is not None else build_adjoint_replay_product_registry()
    if registry.get("schema") != ADJOINT_REPLAY_PRODUCT_SCHEMA:
        raise ValueError("adjoint-replay product schema mismatch")
    surfaces = registry.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("adjoint-replay product registry must contain a non-empty surfaces list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    refuse_found = False
    for index, row in enumerate(surfaces):
        if not isinstance(row, Mapping):
            raise ValueError(f"surface row {index} must be a mapping")
        surface_id = row.get("surface_id")
        kind = row.get("kind")
        symbol_name = row.get("symbol_name")
        allows_catalyst = row.get("allows_catalyst_parity")
        allows_hw = row.get("allows_hardware_adjoint")
        if not surface_id or not str(surface_id).strip():
            blank += 1
            continue
        sid = str(surface_id).strip()
        if sid in seen:
            raise ValueError(f"duplicate surface_id in registry: {sid!r}")
        seen.add(sid)
        if sid == "reverse_adjoint_grad":
            default_found = True
        if sid == "irreversible_mid_circuit_refuse":
            refuse_found = True
        if kind not in {
            "reversibility_conditions",
            "checkpoint_policy",
            "reverse_adjoint_grad",
            "executable_replay",
            "irreversible_refuse",
            "catalyst_hardware_refuse",
        }:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"surface {sid!r} must have symbol_name")
        if allows_catalyst is True:
            raise ValueError(
                f"surface {sid!r} invent-green Catalyst parity: "
                "allows_catalyst_parity must be False"
            )
        if allows_hw is True:
            raise ValueError(
                f"surface {sid!r} invent-green hardware adjoint: "
                "allows_hardware_adjoint must be False"
            )
    if blank:
        raise ValueError(f"adjoint-replay product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("adjoint-replay product registry missing reverse_adjoint_grad")
    if not refuse_found:
        raise ValueError("adjoint-replay product registry missing irreversible_mid_circuit_refuse")
    expected = set(list_adjoint_replay_surface_ids())
    if seen != expected:
        raise ValueError(
            f"registry surface set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    surface_count = registry.get("surface_count", -1)
    if not isinstance(surface_count, int) or surface_count != len(surfaces):
        raise ValueError("surface_count does not match surfaces list length")
    return registry


__all__ = [
    "ADJOINT_REPLAY_CLAIM_BOUNDARY",
    "ADJOINT_REPLAY_PRODUCT_SCHEMA",
    "AdjointReplaySurfaceRow",
    "CheckpointPolicy",
    "CheckpointSchedule",
    "MaterialisedAdjointReplayProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "ReversibilityReport",
    "SupportPosture",
    "SurfaceKind",
    "assess_reversibility",
    "assert_adjoint_replay_product_integrity",
    "build_adjoint_replay_product_registry",
    "build_checkpoint_policy",
    "decide_adjoint_replay_path",
    "get_adjoint_replay_surface",
    "iter_adjoint_replay_surfaces",
    "list_adjoint_replay_surface_ids",
    "map_adjoint_replay_public_surfaces",
    "materialise_demo_adjoint_replay_probe",
]
