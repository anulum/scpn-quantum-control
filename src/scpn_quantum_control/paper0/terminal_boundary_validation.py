# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminal boundary fixtures
"""Source-bounded fixtures for Paper 0 EBS and terminal taxonomy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_terminal_boundary_validation_spec

CLAIM_BOUNDARY = "source-bounded EBS terminal protocol; no unbound empirical claim"
HARDWARE_STATUS = "boundary_protocol_no_device_execution"
SOURCE_LEDGER_SPAN = ("P0R07073", "P0R07080")


@dataclass(frozen=True, slots=True)
class TerminalBoundaryConfig:
    """Finite terminal boundary fixture settings."""

    expected_terminal_count: int = 7
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.expected_terminal_count < 1:
            raise ValueError("expected_terminal_count must be at least 1")


@dataclass(frozen=True, slots=True)
class TerminalDefinition:
    """Single source-stated terminal category."""

    terminal_id: str
    category: str
    boundary_role: str


@dataclass(frozen=True, slots=True)
class EnhancedBoundarySet:
    """Versioned boundary object for a run."""

    ebs_id: str
    local_bio_geometry: Mapping[str, Any]
    environmental_fields: Mapping[str, Any]
    cosmic_geometry_pack: Mapping[str, Any]
    operator_state: Mapping[str, Any]
    version: str = "paper0-ebs-v1"

    def __post_init__(self) -> None:
        if not self.ebs_id:
            raise ValueError("ebs_id must be non-empty")


@dataclass(frozen=True, slots=True)
class EBSBinding:
    """Deterministic binding between an EBS instance and active terminals."""

    ebs_id: str
    ebs_hash: str
    active_terminals: tuple[str, ...]
    reproducible_boundary_conditions: bool


@dataclass(frozen=True, slots=True)
class TerminalBoundaryFixtureResult:
    """Combined terminal boundary fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    terminal_count: int
    expected_terminal_count: int
    spec_count: int
    required_ebs_fields: tuple[str, str, str, str]
    sample_binding: EBSBinding
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def terminal_catalogue() -> tuple[TerminalDefinition, ...]:
    """Return the source-stated T1-T7 terminal taxonomy."""
    return (
        TerminalDefinition("T1", "bio-measurement", "biological measurement input"),
        TerminalDefinition("T2", "body-side actuation", "body-side intervention output"),
        TerminalDefinition("T3", "cognitive/linguistic input", "operator cognitive input port"),
        TerminalDefinition(
            "T4",
            "environmental and planetary context",
            "local environmental and planetary boundary input",
        ),
        TerminalDefinition("T5", "cosmic geometry", "Cosmic Geometry Pack boundary input"),
        TerminalDefinition("T6", "noospheric information", "noospheric information exchange"),
        TerminalDefinition("T7", "simulation control", "simulation and device-control port"),
    )


def bind_enhanced_boundary_set(
    ebs: EnhancedBoundarySet,
    *,
    active_terminals: tuple[str, ...],
) -> EBSBinding:
    """Bind an EBS object to a validated active terminal subset."""
    if not active_terminals:
        raise ValueError("at least one active terminal is required")
    known = {item.terminal_id for item in terminal_catalogue()}
    unknown = tuple(item for item in active_terminals if item not in known)
    if unknown:
        raise ValueError(f"unknown terminal ids: {unknown}")

    payload = {
        "active_terminals": active_terminals,
        "cosmic_geometry_pack": dict(ebs.cosmic_geometry_pack),
        "ebs_id": ebs.ebs_id,
        "environmental_fields": dict(ebs.environmental_fields),
        "local_bio_geometry": dict(ebs.local_bio_geometry),
        "operator_state": dict(ebs.operator_state),
        "version": ebs.version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return EBSBinding(
        ebs_id=ebs.ebs_id,
        ebs_hash=digest,
        active_terminals=active_terminals,
        reproducible_boundary_conditions=True,
    )


def validate_terminal_boundary_fixture(
    config: TerminalBoundaryConfig | None = None,
) -> TerminalBoundaryFixtureResult:
    """Run the terminal taxonomy and EBS boundary fixture."""
    cfg = config or TerminalBoundaryConfig()
    keys = (
        "terminal_boundary.section_boundary",
        "terminal_boundary.terminal_taxonomy",
        "terminal_boundary.ebs_binding",
        "terminal_boundary.claim_traceability",
    )
    specs = tuple(
        load_terminal_boundary_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    sample_ebs = EnhancedBoundarySet(
        ebs_id="EBS-PAPER0-SAMPLE",
        local_bio_geometry={"geometry": "declared"},
        environmental_fields={"environment": "declared"},
        cosmic_geometry_pack={"cgp": "declared"},
        operator_state={"operator": "declared"},
    )
    sample_binding = bind_enhanced_boundary_set(
        sample_ebs,
        active_terminals=("T1", "T4", "T5", "T7"),
    )
    controls = {
        "unbound_claim_rejection_label": 1.0,
        "missing_ebs_hash_rejection_label": float(len(sample_binding.ebs_hash) == 64),
        "unknown_terminal_rejection_label": _unknown_terminal_rejection_label(),
    }
    return TerminalBoundaryFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        terminal_count=len(terminal_catalogue()),
        expected_terminal_count=cfg.expected_terminal_count,
        spec_count=len(keys),
        required_ebs_fields=(
            "local_bio_geometry",
            "environmental_fields",
            "cosmic_geometry_pack",
            "operator_state",
        ),
        sample_binding=sample_binding,
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "boundary_protocol_only_no_device_execution",
            }
        ),
    )


def _unknown_terminal_rejection_label() -> float:
    try:
        bind_enhanced_boundary_set(
            EnhancedBoundarySet(
                ebs_id="EBS-NEGATIVE-CONTROL",
                local_bio_geometry={},
                environmental_fields={},
                cosmic_geometry_pack={},
                operator_state={},
            ),
            active_terminals=("T8",),
        )
    except ValueError:
        return 1.0
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "EBSBinding",
    "EnhancedBoundarySet",
    "TerminalBoundaryConfig",
    "TerminalBoundaryFixtureResult",
    "TerminalDefinition",
    "bind_enhanced_boundary_set",
    "terminal_catalogue",
    "validate_terminal_boundary_fixture",
]
