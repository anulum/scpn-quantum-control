# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Benchmark harness — schema
"""Typed dataclasses for the DLA-parity benchmark dataset.

Types only — no I/O, no computation. Read by :mod:`.dataset` when
loading the sub-phase JSON files, and by :mod:`.reproduce` and
:mod:`.baselines` when consuming the loaded records.

The JSON schema these types mirror is documented in
``data/phase1_dla_parity/README.md`` (directory name kept for the
published dataset; new code is named by scientific content, not
campaign timing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Sector = Literal["even", "odd", "baseline"]
BenchmarkRunName = Literal[
    "bench",
    "reinforce",
    "exhaust",
    "final_burn",
]


@dataclass(frozen=True, slots=True)
class BenchmarkCircuitMeta:
    """Metadata attached to every circuit in the benchmark dataset."""

    experiment: str
    n_qubits: int
    depth: int
    sector: Sector
    initial: str
    rep: int
    shots: int
    t_step: float


@dataclass(frozen=True, slots=True)
class BenchmarkCircuit:
    """A single benchmark circuit: metadata + measured counts."""

    meta: BenchmarkCircuitMeta
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One published sub-phase file (one of four for the DLA-parity campaign).

    Fields mirror the JSON top level: ``experiment``,
    ``timestamp_utc``, ``backend``, ``job_ids``, ``wall_time_s``,
    ``n_circuits``, ``t_step``, ``circuits``. Any top-level JSON keys
    not required by the reproducer land in :attr:`extra`.
    """

    experiment: str
    timestamp_utc: str
    backend: str
    job_ids: tuple[str, ...]
    wall_time_s: float
    n_circuits: int
    t_step: float
    circuits: tuple[BenchmarkCircuit, ...]
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> BenchmarkRunName:
        """Infer the canonical short run name from the ``experiment`` field.

        The published DLA-parity campaign uses four ``experiment``
        labels: ``phase1_dla_parity_mini_bench``,
        ``phase1_5_reinforce``, ``phase2_exhaust``,
        ``phase2_5_final_burn``. Map each to a short, content-named
        :data:`BenchmarkRunName` value.
        """
        mapping: dict[str, BenchmarkRunName] = {
            "phase1_dla_parity_mini_bench": "bench",
            "phase1_5_reinforce": "reinforce",
            "phase2_exhaust_cycle": "exhaust",
            "phase2_5_final_burn": "final_burn",
        }
        try:
            return mapping[self.experiment]
        except KeyError as exc:
            raise ValueError(
                f"Unknown benchmark-run experiment name: {self.experiment!r}. "
                "If a new run was added, extend the mapping in "
                "BenchmarkRun.name.",
            ) from exc


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """Full benchmark dataset — the sub-phase runs composed together."""

    subphases: tuple[BenchmarkRun, ...]

    @property
    def circuits(self) -> tuple[BenchmarkCircuit, ...]:
        """Every circuit across every run."""
        return tuple(c for sp in self.subphases for c in sp.circuits)

    @property
    def n_circuits_total(self) -> int:
        return len(self.circuits)

    @property
    def backends(self) -> frozenset[str]:
        """Distinct hardware backends named in the dataset.

        The published DLA-parity dataset is single-backend
        (``ibm_kingston``); this property returns a set for
        forward-compatibility with a multi-backend re-run.
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
