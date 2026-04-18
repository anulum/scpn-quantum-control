# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Benchmark harness — schema
"""Typed dataclasses for the Phase 1 DLA-parity campaign dataset.

Types only — no I/O, no computation. Read by :mod:`.dataset` when
loading the four sub-phase JSON files, and by :mod:`.reproduce`
and :mod:`.baselines` when consuming the loaded records.

The JSON schema these types mirror is documented in
``data/phase1_dla_parity/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Sector = Literal["even", "odd"]
SubphaseName = Literal[
    "phase1_bench",
    "phase1_5_reinforce",
    "phase2_exhaust",
    "phase2_5_final_burn",
]


@dataclass(frozen=True, slots=True)
class PhaseOneCircuitMeta:
    """Metadata attached to every circuit in the Phase 1 dataset."""

    experiment: str
    n_qubits: int
    depth: int
    sector: Sector
    initial: str
    rep: int
    shots: int
    t_step: float


@dataclass(frozen=True, slots=True)
class PhaseOneCircuit:
    """A single Phase 1 circuit: metadata + measured counts."""

    meta: PhaseOneCircuitMeta
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class PhaseOneSubphase:
    """A Phase 1 sub-phase file (one of four).

    Fields mirror the JSON top level:
    ``experiment``, ``timestamp_utc``, ``backend``, ``job_ids``,
    ``wall_time_s``, ``n_circuits``, ``t_step``, ``circuits``.
    Any fields not critical to the reproducer are kept as typed
    primitives; free-form extensions land in :attr:`extra`.
    """

    experiment: str
    timestamp_utc: str
    backend: str
    job_ids: tuple[str, ...]
    wall_time_s: float
    n_circuits: int
    t_step: float
    circuits: tuple[PhaseOneCircuit, ...]
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> SubphaseName:
        """Infer the sub-phase name from the ``experiment`` field.

        The Phase 1 campaign uses four sub-phase experiment names:
        ``phase1_dla_parity_mini_bench``,
        ``phase1_5_reinforce``,
        ``phase2_exhaust``,
        ``phase2_5_final_burn``. Map the first to the shorter
        canonical name :data:`SubphaseName` uses elsewhere.
        """
        mapping: dict[str, SubphaseName] = {
            "phase1_dla_parity_mini_bench": "phase1_bench",
            "phase1_5_reinforce": "phase1_5_reinforce",
            "phase2_exhaust": "phase2_exhaust",
            "phase2_5_final_burn": "phase2_5_final_burn",
        }
        try:
            return mapping[self.experiment]
        except KeyError as exc:
            raise ValueError(
                f"Unknown sub-phase experiment name: {self.experiment!r}. "
                "If a new sub-phase was added, extend the mapping in "
                "PhaseOneSubphase.name.",
            ) from exc


@dataclass(frozen=True, slots=True)
class PhaseOneDataset:
    """Full Phase 1 dataset — four sub-phases composed together."""

    subphases: tuple[PhaseOneSubphase, ...]

    @property
    def circuits(self) -> tuple[PhaseOneCircuit, ...]:
        """Every circuit across every sub-phase."""
        return tuple(c for sp in self.subphases for c in sp.circuits)

    @property
    def n_circuits_total(self) -> int:
        return len(self.circuits)

    @property
    def backends(self) -> frozenset[str]:
        """Distinct hardware backends named in the dataset.

        Phase 1 is a single-backend campaign (`ibm_kingston`); this
        property returns a set for forward-compatibility with a
        multi-backend Phase 2.
        """
        return frozenset(sp.backend for sp in self.subphases)


@dataclass(frozen=True, slots=True)
class StatisticalSummary:
    """Per-depth Welch statistics derived from the circuits.

    Matches the shape the existing
    ``scripts/analyse_phase1_dla_parity.py`` emits so the harness
    can cross-check the two pathways.
    """

    depth: int
    leakage_even: float
    leakage_even_sem: float
    leakage_odd: float
    leakage_odd_sem: float
    asymmetry_relative: float
    welch_t: float
    welch_p: float
    n_reps_even: int
    n_reps_odd: int
