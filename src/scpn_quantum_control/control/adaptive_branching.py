# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- adaptive branching readiness
"""S8 mid-circuit adaptive branching readiness model.

The S8 track needs dynamic-circuit support before live execution. This module
locks the branch policies, prerequisite checks, and no-submit artefact boundary
without promoting adaptive advantage or hardware evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

ADAPTIVE_BRANCHING_SCHEMA = "s8_adaptive_branching_readiness_v1"
BRANCH_ROW_SCHEMA = "s8_adaptive_branch_row_v1"
CLAIM_BOUNDARY = (
    "readiness and branch-planning artifact only; no adaptive-advantage claim "
    "and no hardware submission"
)
BRANCH_DECISION_BOUNDARY = "branch-planning decision only; not hardware evidence"


@dataclass(frozen=True)
class AdaptiveBranchingConfig:
    """Configuration for S8 adaptive-branching planning."""

    n_oscillators: int = 4
    n_rounds: int = 3
    target_r: float = 0.75
    deadband: float = 0.05
    max_parity_leakage: float = 0.08
    chimera_imbalance_threshold: float = 0.30
    correction_gain: float = 0.30
    max_correction_angle: float = 0.20

    def __post_init__(self) -> None:
        if not isinstance(self.n_oscillators, int) or self.n_oscillators < 2:
            raise ValueError("n_oscillators must be an integer >= 2")
        if not isinstance(self.n_rounds, int) or self.n_rounds < 1:
            raise ValueError("n_rounds must be a positive integer")
        _require_range(self.target_r, 0.0, 1.0, "target_r")
        _require_range(self.deadband, 0.0, 1.0, "deadband")
        _require_range(self.max_parity_leakage, 0.0, 1.0, "max_parity_leakage")
        _require_range(self.chimera_imbalance_threshold, 0.0, 1.0, "chimera_imbalance_threshold")
        _require_positive(self.correction_gain, "correction_gain")
        _require_range(self.max_correction_angle, 0.0, np.pi, "max_correction_angle")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible config data."""
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveBranchDecision:
    """Decision for one measured adaptive-branching state."""

    local_r: float
    parity_leakage: float
    cluster_imbalance: float
    triggered_policy: str
    action: str
    correction_angle: float
    claim_boundary: str = BRANCH_DECISION_BOUNDARY
    hardware_submission_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible decision data."""
        return {
            "schema": BRANCH_ROW_SCHEMA,
            **asdict(self),
        }


@dataclass(frozen=True)
class AdaptiveBranchingReadiness:
    """No-submit readiness result for an S8 target backend."""

    ready: bool
    required_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    n_oscillators: int
    n_rounds: int
    hardware_submission_allowed: bool = False
    adaptive_advantage_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible readiness data."""
        payload = asdict(self)
        payload["required_features"] = list(self.required_features)
        payload["missing_features"] = list(self.missing_features)
        return payload


def classify_branch_state(
    *,
    local_r: float,
    parity_leakage: float,
    cluster_imbalance: float,
    config: AdaptiveBranchingConfig | None = None,
) -> AdaptiveBranchDecision:
    """Classify one mid-circuit observation into an adaptive branch action."""
    cfg = config or AdaptiveBranchingConfig()
    _require_range(local_r, 0.0, 1.0, "local_r")
    _require_range(parity_leakage, 0.0, 1.0, "parity_leakage")
    _require_range(cluster_imbalance, 0.0, 1.0, "cluster_imbalance")

    if parity_leakage > cfg.max_parity_leakage:
        return _decision(
            local_r,
            parity_leakage,
            cluster_imbalance,
            "dla_parity_leakage",
            "sector_rebalance",
            cfg.max_correction_angle,
        )
    if (
        cluster_imbalance > cfg.chimera_imbalance_threshold
        and local_r < cfg.target_r - cfg.deadband
    ):
        deficit = cfg.target_r - local_r
        return _decision(
            local_r,
            parity_leakage,
            cluster_imbalance,
            "chimera_cluster_detector",
            "topology_aware_pulse",
            _correction_angle(deficit, cfg),
        )
    if local_r < cfg.target_r - cfg.deadband:
        deficit = cfg.target_r - local_r
        return _decision(
            local_r,
            parity_leakage,
            cluster_imbalance,
            "local_order_threshold",
            "corrective_kick",
            _correction_angle(deficit, cfg),
        )
    return _decision(
        local_r,
        parity_leakage,
        cluster_imbalance,
        "none",
        "hold",
        0.0,
    )


def build_adaptive_branch_table(
    config: AdaptiveBranchingConfig | None = None,
    *,
    local_r_grid: tuple[float, ...] = (0.55, 0.70, 0.78),
    parity_leakage_grid: tuple[float, ...] = (0.0, 0.12),
    cluster_imbalance_grid: tuple[float, ...] = (0.0, 0.35),
) -> tuple[AdaptiveBranchDecision, ...]:
    """Build a deterministic branch table for S8 readiness review."""
    cfg = config or AdaptiveBranchingConfig()
    if not local_r_grid or not parity_leakage_grid or not cluster_imbalance_grid:
        raise ValueError("branch-table grids must not be empty")
    return tuple(
        classify_branch_state(
            local_r=float(local_r),
            parity_leakage=float(parity_leakage),
            cluster_imbalance=float(cluster_imbalance),
            config=cfg,
        )
        for local_r in local_r_grid
        for parity_leakage in parity_leakage_grid
        for cluster_imbalance in cluster_imbalance_grid
    )


def required_s8_dynamic_features() -> tuple[str, ...]:
    """Return backend features required before S8 live execution."""
    return (
        "mid_circuit_measurement",
        "conditional_control",
        "conditional_reset",
        "cross_shot_batches",
    )


def estimate_branching_readiness(
    config: AdaptiveBranchingConfig | None = None,
    *,
    backend_features: tuple[str, ...] = (),
) -> AdaptiveBranchingReadiness:
    """Estimate S8 execution readiness from declared backend features."""
    cfg = config or AdaptiveBranchingConfig()
    required = required_s8_dynamic_features()
    supported = set(backend_features)
    missing = tuple(feature for feature in required if feature not in supported)
    return AdaptiveBranchingReadiness(
        ready=not missing,
        required_features=required,
        missing_features=missing,
        n_oscillators=cfg.n_oscillators,
        n_rounds=cfg.n_rounds,
    )


def s8_adaptive_branching_payload() -> dict[str, Any]:
    """Return the S8 adaptive-branching readiness payload."""
    config = AdaptiveBranchingConfig()
    branch_table = build_adaptive_branch_table(config)
    readiness = estimate_branching_readiness(
        config,
        backend_features=("mid_circuit_measurement", "conditional_control"),
    )
    policies = [
        {
            "name": "local_order_threshold",
            "observable": "partial local Kuramoto order parameter",
            "trigger": "local_r < target_r - deadband",
            "action": "corrective_kick",
        },
        {
            "name": "dla_parity_leakage",
            "observable": "sector leakage estimate",
            "trigger": "parity_leakage > max_parity_leakage",
            "action": "sector_rebalance",
        },
        {
            "name": "chimera_cluster_detector",
            "observable": "cluster imbalance with low local order",
            "trigger": "cluster_imbalance > threshold and local_r below target window",
            "action": "topology_aware_pulse",
        },
    ]
    return {
        "schema": ADAPTIVE_BRANCHING_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config.to_dict(),
        "policies": policies,
        "branch_table": [row.to_dict() for row in branch_table],
        "branch_table_count": len(branch_table),
        "readiness": readiness.to_dict(),
        "prerequisites": [
            "S1 cross-shot feedback plumbing remains the runner-level dependency",
            "target backend must declare mid-circuit measurement and conditional control",
            "Rust branch-table generation is a performance follow-up, not a correctness prerequisite",
            "hardware execution requires explicit preregistration and approval",
        ],
        "falsifier": "win-rate <= 50% on preregistered equal-depth open-loop comparison",
        "no_qpu_submission": True,
        "hardware_submission_allowed": False,
        "adaptive_advantage_claim_allowed": False,
    }


def s8_adaptive_branching_markdown(payload: dict[str, Any] | None = None) -> str:
    """Render the S8 adaptive-branching readiness note."""
    data = s8_adaptive_branching_payload() if payload is None else payload
    lines = [
        "# Adaptive Branching Readiness",
        "",
        "This is the S8 no-submit readiness surface for mid-circuit adaptive",
        "branching. It records branch policies and backend prerequisites, with",
        "no hardware submission and no adaptive-advantage claim.",
        "",
        "## Boundary",
        "",
        str(data["claim_boundary"]),
        "",
        "## Branch Policies",
    ]
    lines.extend(
        "- `{name}`: observe {observable}; trigger `{trigger}`; action `{action}`.".format(
            **policy
        )
        for policy in data["policies"]
    )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- Ready: `{data['readiness']['ready']}`",
            f"- Required features: `{data['readiness']['required_features']}`",
            f"- Missing features: `{data['readiness']['missing_features']}`",
            f"- Branch table rows: `{data['branch_table_count']}`",
            "",
            "## Falsifier",
            "",
            str(data["falsifier"]),
            "",
            "## Prerequisites",
        ]
    )
    lines.extend(f"- {item}" for item in data["prerequisites"])
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Regenerate and compare this readiness artefact with:",
            "",
            "```bash",
            "scpn-bench s8-adaptive-branching-readiness",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision(
    local_r: float,
    parity_leakage: float,
    cluster_imbalance: float,
    triggered_policy: str,
    action: str,
    correction_angle: float,
) -> AdaptiveBranchDecision:
    return AdaptiveBranchDecision(
        local_r=float(local_r),
        parity_leakage=float(parity_leakage),
        cluster_imbalance=float(cluster_imbalance),
        triggered_policy=triggered_policy,
        action=action,
        correction_angle=float(correction_angle),
    )


def _correction_angle(deficit: float, config: AdaptiveBranchingConfig) -> float:
    return float(np.clip(deficit * config.correction_gain, 0.0, config.max_correction_angle))


def _require_range(value: float, lower: float, upper: float, name: str) -> None:
    if not np.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "ADAPTIVE_BRANCHING_SCHEMA",
    "AdaptiveBranchDecision",
    "AdaptiveBranchingConfig",
    "AdaptiveBranchingReadiness",
    "build_adaptive_branch_table",
    "classify_branch_state",
    "estimate_branching_readiness",
    "required_s8_dynamic_features",
    "s8_adaptive_branching_markdown",
    "s8_adaptive_branching_payload",
]
