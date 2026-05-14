# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 opening foundation fixtures
"""Source-bounded fixtures for Paper 0 opening foundation records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_opening_foundation_validation_spec

CLAIM_BOUNDARY = "source-bounded opening foundation; not empirical validation evidence"
HARDWARE_STATUS = "source_foundation_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00001", "P0R00017")


@dataclass(frozen=True, slots=True)
class OpeningFoundationConfig:
    """Finite settings for the opening foundation fixture."""

    expected_boundary_set_size: int = 4
    expected_terminal_count: int = 7
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.expected_boundary_set_size < 1:
            raise ValueError("expected_boundary_set_size must be at least 1")
        if self.expected_terminal_count < 1:
            raise ValueError("expected_terminal_count must be at least 1")


@dataclass(frozen=True, slots=True)
class OpeningFoundationFixtureResult:
    """Combined opening foundation fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    boundary_set: tuple[str, str, str, str]
    terminal_set: tuple[str, str, str, str, str, str, str]
    boundary_set_size: int
    terminal_count: int
    expected_boundary_set_size: int
    expected_terminal_count: int
    spec_count: int
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def boundary_set_c0() -> tuple[str, str, str, str]:
    """Return the source-stated C_0 boundary set."""
    return ("G_local", "F_env", "G_cosmic", "O_state")


def terminal_set_f0() -> tuple[str, str, str, str, str, str, str]:
    """Return the source-stated F_0 terminal set."""
    return ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def beta0_boundary_assertion(
    *,
    boundary_members: tuple[str, ...],
    active_terminals: tuple[str, ...],
) -> bool:
    """Validate the global no-free-boundary beta_0(E,A) membership assertion."""
    allowed_boundaries = set(boundary_set_c0())
    allowed_terminals = set(terminal_set_f0())
    unknown_boundaries = tuple(item for item in boundary_members if item not in allowed_boundaries)
    if unknown_boundaries:
        raise ValueError(f"unknown boundary members: {unknown_boundaries}")
    if not active_terminals:
        raise ValueError("at least one active terminal is required")
    unknown_terminals = tuple(item for item in active_terminals if item not in allowed_terminals)
    if unknown_terminals:
        raise ValueError(f"unknown terminal ids: {unknown_terminals}")
    return True


def validate_opening_foundation_fixture(
    config: OpeningFoundationConfig | None = None,
) -> OpeningFoundationFixtureResult:
    """Run the opening foundation and global-boundary axiom fixture."""
    cfg = config or OpeningFoundationConfig()
    keys = (
        "opening_foundation.book_identity",
        "opening_foundation.quasicritical_ms_qec",
        "opening_foundation.recursive_optimisation",
        "opening_foundation.ebs_terminal_anchor",
        "opening_foundation.global_boundary_axiom",
    )
    specs = tuple(
        load_opening_foundation_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    controls = {
        "free_boundary_rejection_label": _free_boundary_rejection_label(),
        "unknown_terminal_rejection_label": _unknown_terminal_rejection_label(),
        "empty_terminal_subset_rejection_label": _empty_terminal_subset_rejection_label(),
    }
    c0 = boundary_set_c0()
    f0 = terminal_set_f0()
    return OpeningFoundationFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        boundary_set=c0,
        terminal_set=f0,
        boundary_set_size=len(c0),
        terminal_count=len(f0),
        expected_boundary_set_size=cfg.expected_boundary_set_size,
        expected_terminal_count=cfg.expected_terminal_count,
        spec_count=len(keys),
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "source_foundation_only_no_experiment",
            }
        ),
    )


def _free_boundary_rejection_label() -> float:
    try:
        beta0_boundary_assertion(
            boundary_members=("G_local", "free_boundary"),
            active_terminals=("T1",),
        )
    except ValueError:
        return 1.0
    return 0.0


def _unknown_terminal_rejection_label() -> float:
    try:
        beta0_boundary_assertion(boundary_members=("G_local",), active_terminals=("T8",))
    except ValueError:
        return 1.0
    return 0.0


def _empty_terminal_subset_rejection_label() -> float:
    try:
        beta0_boundary_assertion(boundary_members=("G_local",), active_terminals=())
    except ValueError:
        return 1.0
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "OpeningFoundationConfig",
    "OpeningFoundationFixtureResult",
    "beta0_boundary_assertion",
    "boundary_set_c0",
    "terminal_set_f0",
    "validate_opening_foundation_fixture",
]
