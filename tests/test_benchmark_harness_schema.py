# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark_harness.schema tests
"""Tests for `scpn_quantum_control.benchmark_harness.schema`.

Types only; no I/O to test. Cover:

* Happy-path construction of each dataclass.
* Immutability (frozen=True) — construction is the only mutation.
* `PhaseOneSubphase.name` — known experiment names map to the
  canonical sub-phase label; unknown names raise `ValueError`.
* `PhaseOneDataset.circuits` / `n_circuits_total` / `backends`
  aggregate correctly across sub-phases.
* Dataclass equality (slots=True does not break eq).
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.benchmark_harness.schema import (
    PhaseOneCircuit,
    PhaseOneCircuitMeta,
    PhaseOneDataset,
    PhaseOneSubphase,
    StatisticalSummary,
)


def _make_meta(depth: int = 2, sector: str = "even") -> PhaseOneCircuitMeta:
    return PhaseOneCircuitMeta(
        experiment="A_dla_parity_n4",
        n_qubits=4,
        depth=depth,
        sector=sector,  # type: ignore[arg-type]
        initial="0011" if sector == "even" else "0001",
        rep=0,
        shots=2048,
        t_step=0.3,
    )


def _make_circuit(depth: int = 2, sector: str = "even") -> PhaseOneCircuit:
    return PhaseOneCircuit(meta=_make_meta(depth, sector), counts={"1100": 1500})


def _make_subphase(
    experiment: str = "phase1_dla_parity_mini_bench",
    *,
    n_circuits: int = 2,
) -> PhaseOneSubphase:
    return PhaseOneSubphase(
        experiment=experiment,
        timestamp_utc="2026-04-10T183728Z",
        backend="ibm_kingston",
        job_ids=("d7ck79m5nvhs73a4nr10",),
        wall_time_s=44.11,
        n_circuits=n_circuits,
        t_step=0.3,
        circuits=tuple(_make_circuit(depth=2 + i, sector="even") for i in range(n_circuits)),
    )


class TestPhaseOneCircuitMeta:
    def test_happy_path(self) -> None:
        meta = _make_meta()
        assert meta.n_qubits == 4
        assert meta.depth == 2
        assert meta.sector == "even"
        assert meta.initial == "0011"
        assert meta.shots == 2048
        assert meta.t_step == 0.3

    def test_frozen(self) -> None:
        meta = _make_meta()
        with pytest.raises(AttributeError):
            meta.depth = 99  # type: ignore[misc]


class TestPhaseOneCircuit:
    def test_happy_path(self) -> None:
        c = _make_circuit(depth=6, sector="odd")
        assert c.meta.depth == 6
        assert c.meta.sector == "odd"
        assert c.counts == {"1100": 1500}

    def test_frozen(self) -> None:
        c = _make_circuit()
        with pytest.raises(AttributeError):
            c.counts = {}  # type: ignore[misc]


class TestPhaseOneSubphase:
    def test_happy_path(self) -> None:
        sp = _make_subphase()
        assert sp.backend == "ibm_kingston"
        assert sp.n_circuits == 2
        assert len(sp.circuits) == 2

    def test_name_resolution_each_known_experiment(self) -> None:
        cases = {
            "phase1_dla_parity_mini_bench": "phase1_bench",
            "phase1_5_reinforce": "phase1_5_reinforce",
            "phase2_exhaust": "phase2_exhaust",
            "phase2_5_final_burn": "phase2_5_final_burn",
        }
        for experiment, canonical in cases.items():
            sp = _make_subphase(experiment=experiment, n_circuits=1)
            assert sp.name == canonical

    def test_name_raises_on_unknown_experiment(self) -> None:
        sp = _make_subphase(experiment="phase3_future_unknown", n_circuits=1)
        with pytest.raises(ValueError, match="Unknown sub-phase"):
            _ = sp.name


class TestPhaseOneDataset:
    def test_empty(self) -> None:
        ds = PhaseOneDataset(subphases=())
        assert ds.n_circuits_total == 0
        assert ds.backends == frozenset()
        assert ds.circuits == ()

    def test_aggregates_across_subphases(self) -> None:
        sp1 = _make_subphase(experiment="phase1_dla_parity_mini_bench", n_circuits=3)
        sp2 = _make_subphase(experiment="phase1_5_reinforce", n_circuits=4)
        ds = PhaseOneDataset(subphases=(sp1, sp2))
        assert ds.n_circuits_total == 7
        assert ds.backends == frozenset({"ibm_kingston"})
        assert len(ds.circuits) == 7

    def test_multi_backend_reported_as_set(self) -> None:
        sp1 = _make_subphase()
        sp2 = PhaseOneSubphase(
            experiment="phase2_exhaust",
            timestamp_utc="2026-04-10T185634Z",
            backend="ibm_marrakesh",
            job_ids=("x",),
            wall_time_s=1.0,
            n_circuits=0,
            t_step=0.3,
            circuits=(),
        )
        ds = PhaseOneDataset(subphases=(sp1, sp2))
        assert ds.backends == frozenset({"ibm_kingston", "ibm_marrakesh"})


class TestStatisticalSummary:
    def test_happy_path(self) -> None:
        s = StatisticalSummary(
            depth=6,
            leakage_even=0.1291,
            leakage_even_sem=0.0031,
            leakage_odd=0.1099,
            leakage_odd_sem=0.0018,
            asymmetry_relative=0.175,
            welch_t=5.37,
            welch_p=6.6e-6,
            n_reps_even=21,
            n_reps_odd=21,
        )
        assert s.depth == 6
        assert pytest.approx(s.asymmetry_relative, rel=1e-6) == 0.175
        assert s.n_reps_even == s.n_reps_odd == 21

    def test_frozen(self) -> None:
        s = StatisticalSummary(
            depth=2,
            leakage_even=0.0,
            leakage_even_sem=0.0,
            leakage_odd=0.0,
            leakage_odd_sem=0.0,
            asymmetry_relative=0.0,
            welch_t=0.0,
            welch_p=1.0,
            n_reps_even=0,
            n_reps_odd=0,
        )
        with pytest.raises(AttributeError):
            s.depth = 99  # type: ignore[misc]
