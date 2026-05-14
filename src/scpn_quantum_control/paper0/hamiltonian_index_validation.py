# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Hamiltonian index fixtures
"""Catalogue fixtures for the Paper 0 Appendix C Hamiltonian/operator index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_hamiltonian_index_validation_spec

CLAIM_BOUNDARY = "source-bounded Hamiltonian/operator index; not empirical evidence"
HARDWARE_STATUS = "operator_index_no_execution"
SOURCE_LEDGER_SPAN = ("P0R06878", "P0R06915")


@dataclass(frozen=True, slots=True)
class HamiltonianIndexConfig:
    """Finite Hamiltonian index fixture settings."""

    expected_operator_count: int = 9
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.expected_operator_count < 1:
            raise ValueError("expected_operator_count must be at least 1")


@dataclass(frozen=True, slots=True)
class OperatorIndexEntry:
    """Single indexed Hamiltonian or operator reference."""

    symbol: str
    label: str
    layer_group: str
    equation: str | None
    source_ledger_id: str
    location: str


@dataclass(frozen=True, slots=True)
class HamiltonianIndexFixtureResult:
    """Combined Hamiltonian/operator index fixture result."""

    spec_keys: tuple[str, str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    operator_count: int
    expected_operator_count: int
    layer_groups: tuple[str, ...]
    location_coverage_valid: bool
    unresolved_equation_symbols: tuple[str, ...]
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def operator_catalogue() -> tuple[OperatorIndexEntry, ...]:
    """Return the source-stated Appendix C operator catalogue."""
    return (
        OperatorIndexEntry(
            "L_Anulum",
            "Master Lagrangian",
            "fundamental_meta_universal",
            "sqrt(-g)(R - 2 Lambda_Psi) + L_SM + |D_mu Psi|^2 - V(Psi) + L_int",
            "P0R06885",
            "Paper 13 / Paper 16",
        ),
        OperatorIndexEntry(
            "H_MT",
            "Microtubule Frohlich Hamiltonian",
            "microscopic_layer1",
            "sum hbar omega a^dagger a + S(a^dagger + a) + H_bath",
            "P0R06889",
            "Paper 1 Section 4.10",
        ),
        OperatorIndexEntry(
            "H_PQT",
            "Piezo-Quantum Transduction",
            "microscopic_layer1",
            "hbar g_pz (b^dagger a + b a^dagger)",
            "P0R06892",
            "Paper 1 Section 4.23",
        ),
        OperatorIndexEntry(
            "H_iso",
            "Isotopic Spin Interaction",
            "microscopic_layer1",
            "sum gamma I dot (B_loc + B_Psi)",
            "P0R06895",
            "Paper 1 Protocol E",
        ),
        OperatorIndexEntry(
            "H_NI",
            "Neuro-Immune Hamiltonian",
            "mesoscopic_layers2_4",
            "H_neural + H_immune + H_tunnel",
            "P0R06899",
            "Paper 2 / Paper 23",
        ),
        OperatorIndexEntry(
            "H_syn",
            "Quantum Synaptic Hamiltonian",
            "mesoscopic_layers2_4",
            None,
            "P0R06901",
            "Paper 2 Section 22",
        ),
        OperatorIndexEntry(
            "H_RP",
            "Radical Pair Hamiltonian",
            "macroscopic_layers6_8",
            "sum g mu_B B dot S + J(S_1 dot S_2)",
            "P0R06905",
            "Paper 6 / Paper 31",
        ),
        OperatorIndexEntry(
            "R_Psi",
            "Phase-Curvature Tensor",
            "informational_operator",
            None,
            "P0R06910",
            "Paper 16 UPDE",
        ),
        OperatorIndexEntry(
            "O_sem",
            "Semiotic Operator",
            "informational_operator",
            None,
            "P0R06912",
            "Paper 7",
        ),
    )


def classify_operator_layer(symbol: str) -> str:
    """Return the source layer group for an indexed operator symbol."""
    for entry in operator_catalogue():
        if entry.symbol == symbol:
            return entry.layer_group
    raise ValueError("operator symbol is not in Appendix C catalogue")


def validate_operator_locations(entries: tuple[OperatorIndexEntry, ...]) -> bool:
    """Validate that every catalogue entry has a non-empty location reference."""
    for entry in entries:
        if not entry.location:
            raise ValueError("operator location must be non-empty")
    return True


def validate_hamiltonian_index_fixture(
    config: HamiltonianIndexConfig | None = None,
) -> HamiltonianIndexFixtureResult:
    """Run the Hamiltonian/operator index boundary fixture."""
    cfg = config or HamiltonianIndexConfig()
    keys = (
        "appendix_c.hamiltonian_index.appendix_boundary",
        "appendix_c.hamiltonian_index.master_lagrangian",
        "appendix_c.hamiltonian_index.microtubule_layer1",
        "appendix_c.hamiltonian_index.neuroimmune_mesoscopic",
        "appendix_c.hamiltonian_index.radical_pair_macro",
        "appendix_c.hamiltonian_index.informational_operators",
        "appendix_c.hamiltonian_index.structural_separators",
    )
    specs = tuple(
        load_hamiltonian_index_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    entries = operator_catalogue()
    layer_groups = tuple(dict.fromkeys(entry.layer_group for entry in entries))
    controls = {
        "unknown_operator_rejection_label": _unknown_operator_rejection_label(),
        "missing_location_rejection_label": _missing_location_rejection_label(),
        "unsupported_executed_validation_rejection_label": 1.0,
    }
    return HamiltonianIndexFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        operator_count=len(entries),
        expected_operator_count=cfg.expected_operator_count,
        layer_groups=layer_groups,
        location_coverage_valid=validate_operator_locations(entries),
        unresolved_equation_symbols=tuple(
            entry.symbol for entry in entries if entry.equation is None
        ),
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "operator_index_only_no_execution",
            }
        ),
    )


def _unknown_operator_rejection_label() -> float:
    try:
        classify_operator_layer("H_missing")
    except ValueError as exc:
        return float("operator symbol is not in Appendix C catalogue" in str(exc))
    return 0.0


def _missing_location_rejection_label() -> float:
    entries = list(operator_catalogue())
    first = entries[0]
    entries[0] = OperatorIndexEntry(
        symbol=first.symbol,
        label=first.label,
        layer_group=first.layer_group,
        equation=first.equation,
        source_ledger_id=first.source_ledger_id,
        location="",
    )
    try:
        validate_operator_locations(tuple(entries))
    except ValueError as exc:
        return float("operator location must be non-empty" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "HamiltonianIndexConfig",
    "HamiltonianIndexFixtureResult",
    "OperatorIndexEntry",
    "classify_operator_layer",
    "operator_catalogue",
    "validate_hamiltonian_index_fixture",
    "validate_operator_locations",
]
