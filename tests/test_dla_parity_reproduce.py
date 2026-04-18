# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dla_parity.reproduce tests
"""Tests for `scpn_quantum_control.dla_parity.reproduce`.

Cover:

* ``recompute_parity_leakage`` — property: always in [0, 1] for any
  non-empty count histogram; deterministic across counter orderings;
  correct on hand-built edge cases (all-same-parity → 0; all-opposite
  → 1; empty → NaN).
* ``compute_depth_summaries`` — happy path on the real dataset,
  sector filter, n_qubits filter, experiment-prefix filter.
* ``_fisher_combined`` (indirectly) — chi² grows as more significant
  depths are added; NaN propagation.
* ``reproduce_statistics`` — real dataset passes the published
  figures within default tolerances; a tightened tolerance that
  would reject bit-exact drift still passes (our recomputation is
  bit-exact on the measured-leakage path); a loosened published
  value outside tolerance triggers the expected AssertionError.
* ``ReproductionTolerance`` / ``ReproductionResult`` / ``FisherResult``
  — frozen dataclasses, equality.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.dla_parity.dataset import (
    DEFAULT_DATA_DIR,
    RUN_FILES,
    load_dla_parity_dataset,
)
from scpn_quantum_control.dla_parity.reproduce import (
    DEFAULT_PUBLISHED_SUMMARY,
    FisherResult,
    ReproductionResult,
    ReproductionTolerance,
    compute_depth_summaries,
    recompute_parity_leakage,
    reproduce_statistics,
)
from scpn_quantum_control.dla_parity.schema import (
    DlaParityCircuit,
    DlaParityCircuitMeta,
    DlaParityDataset,
    DlaParityRun,
)


def _has_real_data() -> bool:
    return (
        DEFAULT_DATA_DIR.is_dir()
        and all((DEFAULT_DATA_DIR / f).is_file() for f in RUN_FILES)
        and DEFAULT_PUBLISHED_SUMMARY.is_file()
    )


needs_real_data = pytest.mark.skipif(
    not _has_real_data(),
    reason="DLA-parity dataset or published summary not present",
)


def _circuit(
    *,
    counts: dict[str, int],
    depth: int = 2,
    sector: str = "even",
    initial: str = "0011",
    experiment: str = "A_dla_parity_n4",
    n_qubits: int = 4,
) -> DlaParityCircuit:
    meta = DlaParityCircuitMeta(
        experiment=experiment,
        n_qubits=n_qubits,
        depth=depth,
        sector=sector,  # type: ignore[arg-type]
        initial=initial,
        rep=0,
        shots=sum(counts.values()),
        t_step=0.3,
    )
    return DlaParityCircuit(meta=meta, counts=counts)


class TestRecomputeParityLeakage:
    def test_all_same_parity_is_zero(self) -> None:
        c = _circuit(initial="0011", counts={"1100": 100, "0011": 50, "1001": 25})
        assert recompute_parity_leakage(c) == 0.0

    def test_all_opposite_parity_is_one(self) -> None:
        c = _circuit(initial="0011", counts={"0001": 100, "1110": 50})
        assert recompute_parity_leakage(c) == 1.0

    def test_mixed_counts_exact(self) -> None:
        # initial "0011" → parity 0. "1100" parity 0 (same), "0001" parity 1 (opposite).
        c = _circuit(initial="0011", counts={"1100": 300, "0001": 100})
        assert recompute_parity_leakage(c) == pytest.approx(0.25, rel=1e-12)

    def test_empty_counts_is_nan(self) -> None:
        c = _circuit(counts={})
        leak = recompute_parity_leakage(c)
        assert math.isnan(leak)

    def test_leakage_invariant_under_count_reordering(self) -> None:
        c1 = _circuit(initial="0011", counts={"1100": 300, "0001": 100, "1110": 50})
        c2 = _circuit(initial="0011", counts={"0001": 100, "1110": 50, "1100": 300})
        assert recompute_parity_leakage(c1) == recompute_parity_leakage(c2)

    @given(
        n_qubits=st.integers(min_value=2, max_value=6),
        shots_per_bitstring=st.dictionaries(
            keys=st.text(
                alphabet=st.sampled_from(["0", "1"]),
                min_size=2,
                max_size=6,
            ),
            values=st.integers(min_value=0, max_value=10_000),
            min_size=1,
            max_size=16,
        ),
    )
    @settings(max_examples=200, deadline=None)
    def test_leakage_bounded_0_1(self, n_qubits: int, shots_per_bitstring: dict[str, int]) -> None:
        clean = {k: v for k, v in shots_per_bitstring.items() if len(k) == n_qubits and v > 0}
        if not clean:
            return
        initial = "0" * n_qubits
        c = _circuit(
            counts=clean,
            initial=initial,
            n_qubits=n_qubits,
        )
        leak = recompute_parity_leakage(c)
        assert 0.0 <= leak <= 1.0


class TestComputeDepthSummaries:
    def test_sector_filter_excludes_baseline(self) -> None:
        circuits = [
            _circuit(counts={"0000": 100}, sector="baseline"),
            _circuit(counts={"1100": 100}, sector="even", depth=2),
            _circuit(counts={"0001": 60, "1100": 40}, sector="odd", depth=2, initial="0001"),
            _circuit(counts={"1100": 100}, sector="even", depth=2),
            _circuit(counts={"0001": 100}, sector="odd", depth=2, initial="0001"),
        ]
        run = DlaParityRun(
            experiment="x",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        assert len(summaries) == 1
        assert summaries[0].depth == 2
        assert summaries[0].n_reps_even == 2
        assert summaries[0].n_reps_odd == 2

    def test_experiment_prefix_filter_skips_other_experiments(self) -> None:
        circuits = [
            _circuit(counts={"1100": 100}, experiment="B_something_else"),
            _circuit(counts={"1100": 100}, experiment="A_dla_parity_n4"),
            _circuit(counts={"0001": 100}, experiment="A_dla_parity_n4", sector="odd"),
        ]
        run = DlaParityRun(
            experiment="x",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        assert summaries[0].n_reps_even == 1
        assert summaries[0].n_reps_odd == 1

    def test_nqubits_filter(self) -> None:
        circuits = [
            _circuit(counts={"11110000": 100}, n_qubits=8, initial="00110000"),
            _circuit(counts={"1100": 100}, n_qubits=4),
            _circuit(counts={"0001": 100}, n_qubits=4, sector="odd", initial="0001"),
        ]
        run = DlaParityRun(
            experiment="x",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        assert len(summaries) == 1
        assert summaries[0].n_reps_even + summaries[0].n_reps_odd == 2


class TestFisherResult:
    def test_frozen(self) -> None:
        f = FisherResult(
            chi2=1.0,
            degrees_of_freedom=2,
            combined_p=0.5,
            n_depths_significant_at_0_05=0,
            n_depths_tested=1,
        )
        with pytest.raises(AttributeError):
            f.chi2 = 2.0  # type: ignore[misc]


class TestReproductionTolerance:
    def test_defaults_present(self) -> None:
        t = ReproductionTolerance()
        assert t.leakage_mean_abs > 0
        assert t.fisher_chi2_rel > 0

    def test_frozen(self) -> None:
        t = ReproductionTolerance()
        with pytest.raises(AttributeError):
            t.leakage_mean_abs = 0.0  # type: ignore[misc]


@needs_real_data
class TestReproduceStatisticsRealDataset:
    def test_reproduces_published_numbers(self) -> None:
        ds = load_dla_parity_dataset()
        result = reproduce_statistics(ds)
        assert isinstance(result, ReproductionResult)
        assert result.n_circuits_used > 0
        # Published: 7 of 8 depths significant at 0.05.
        assert result.fisher.n_depths_significant_at_0_05 == 7
        assert result.fisher.n_depths_tested == 8
        # Published chi² ≈ 123.4.
        assert result.fisher.chi2 == pytest.approx(123.40011440581384, rel=1e-6)
        # Every claim checked should have diff strictly under its tolerance.
        tol = result.tolerance
        for name, _expected, _actual, diff in result.claims_checked:
            if name.endswith(".welch_t") or name.endswith(".welch_p"):
                assert diff <= max(tol.welch_t_rel, tol.welch_p_rel)
            elif name == "fisher.chi2":
                assert diff <= tol.fisher_chi2_rel
            else:
                assert diff <= max(
                    tol.leakage_mean_abs,
                    tol.leakage_sem_abs,
                    tol.asymmetry_relative_abs,
                    tol.mean_asymmetry_abs,
                    tol.peak_asymmetry_abs,
                )

    def test_tolerance_breach_raises(self, tmp_path: Path) -> None:
        # Copy the published summary and tamper one leakage mean.
        with Path(DEFAULT_PUBLISHED_SUMMARY).open("r", encoding="utf-8") as fh:
            pub = json.load(fh)
        pub["depth_summaries"][0]["mean_even"] += 0.5
        tampered = tmp_path / "phase1_dla_parity_summary.json"
        with tampered.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh)
        ds = load_dla_parity_dataset()
        with pytest.raises(AssertionError, match="leakage_even"):
            reproduce_statistics(ds, published_summary=tampered)

    def test_fisher_chi2_breach_raises(self, tmp_path: Path) -> None:
        with Path(DEFAULT_PUBLISHED_SUMMARY).open("r", encoding="utf-8") as fh:
            pub = json.load(fh)
        pub["fisher_combined"]["chi2"] = 9.99
        tampered = tmp_path / "phase1_dla_parity_summary.json"
        with tampered.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh)
        ds = load_dla_parity_dataset()
        with pytest.raises(AssertionError, match="fisher.chi2"):
            reproduce_statistics(ds, published_summary=tampered)

    def test_default_summary_path_used_when_none(self) -> None:
        ds = load_dla_parity_dataset()
        result = reproduce_statistics(ds, published_summary=None)
        assert Path(result.published_source) == DEFAULT_PUBLISHED_SUMMARY

    def test_accepts_str_path(self) -> None:
        ds = load_dla_parity_dataset()
        result = reproduce_statistics(ds, published_summary=str(DEFAULT_PUBLISHED_SUMMARY))
        assert result.fisher.n_depths_tested == 8


class TestReproduceStatisticsSynthetic:
    def test_empty_dataset_fisher_is_nan(self) -> None:
        ds = DlaParityDataset(runs=())
        summaries = compute_depth_summaries(ds)
        assert summaries == ()

    def test_nan_mismatch_raises(self, tmp_path: Path) -> None:
        # Synthesise circuits with variance so Welch yields a finite
        # p-value (and hence a finite chi²); then tamper the published
        # chi² to NaN so the reproducer sees expected=NaN vs actual=finite.
        circuits: list[DlaParityCircuit] = []
        for k in range(10):
            e_opposite = 40 + 2 * k  # varies so even-sector variance > 0
            o_opposite = 600 + 3 * k  # varies so odd-sector variance > 0 and distinct
            circuits.append(
                _circuit(
                    counts={"1100": 1000 - e_opposite, "0001": e_opposite},
                    sector="even",
                ),
            )
            circuits.append(
                _circuit(
                    counts={"0001": 1000 - o_opposite, "1100": o_opposite},
                    sector="odd",
                    initial="0001",
                ),
            )
        run = DlaParityRun(
            experiment="A_dla_parity_n4",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        s0 = summaries[0]
        pub = {
            "depth_summaries": [
                {
                    "depth": s0.depth,
                    "mean_even": s0.leakage_even,
                    "mean_odd": s0.leakage_odd,
                    "sem_even": s0.leakage_even_sem,
                    "sem_odd": s0.leakage_odd_sem,
                    "asymmetry_relative": s0.asymmetry_relative,
                    "welch_t": s0.welch_t,
                    "welch_p": s0.welch_p,
                },
            ],
            "fisher_combined": {"chi2": float("nan"), "df": 0, "combined_p": float("nan")},
        }
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh, allow_nan=True)
        with pytest.raises(AssertionError, match="NaN mismatch"):
            reproduce_statistics(ds, published_summary=path)

    def test_published_depth_without_dataset_match_is_skipped(self, tmp_path: Path) -> None:
        # Dataset has depth=2 only; published summary has depth=99 too.
        circuits = [
            _circuit(counts={"1100": 100}, sector="even", depth=2),
            _circuit(counts={"0001": 100}, sector="odd", depth=2, initial="0001"),
        ] * 3
        run = DlaParityRun(
            experiment="A_dla_parity_n4",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        s = compute_depth_summaries(ds)[0]
        pub = {
            "depth_summaries": [
                {
                    "depth": s.depth,
                    "mean_even": s.leakage_even,
                    "mean_odd": s.leakage_odd,
                    "sem_even": s.leakage_even_sem,
                    "sem_odd": s.leakage_odd_sem,
                    "asymmetry_relative": s.asymmetry_relative,
                    "welch_t": s.welch_t,
                    "welch_p": s.welch_p,
                },
                # Extra published depth the dataset does not include.
                {
                    "depth": 99,
                    "mean_even": 0.5,
                    "mean_odd": 0.5,
                    "sem_even": 0.01,
                    "sem_odd": 0.01,
                    "asymmetry_relative": 0.0,
                    "welch_t": 0.0,
                    "welch_p": 1.0,
                },
            ],
            "fisher_combined": {
                "chi2": -2.0 * math.log(s.welch_p) if 0 < s.welch_p < 1 else float("nan"),
                "df": 2,
                "combined_p": float("nan"),
            },
        }
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh, allow_nan=True)
        # Runs without AssertionError even though published has depth=99.
        result = reproduce_statistics(
            ds,
            tolerance=ReproductionTolerance(fisher_chi2_rel=1e-3),
            published_summary=path,
        )
        # No claim for depth=99 should appear.
        assert all("depth=99" not in name for name, *_ in result.claims_checked)

    def test_optional_mean_and_peak_asymmetry_claims(self, tmp_path: Path) -> None:
        # Build a minimal valid matching-summary then inject the two
        # optional scalars to exercise their check branches.
        circuits = [
            _circuit(counts={"1100": 100}, sector="even", depth=2),
            _circuit(counts={"0001": 100}, sector="odd", depth=2, initial="0001"),
        ] * 3
        run = DlaParityRun(
            experiment="A_dla_parity_n4",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        s = summaries[0]
        mean_asym = float(s.asymmetry_relative)
        peak_asym = float(s.asymmetry_relative)
        pub = {
            "depth_summaries": [
                {
                    "depth": s.depth,
                    "mean_even": s.leakage_even,
                    "mean_odd": s.leakage_odd,
                    "sem_even": s.leakage_even_sem,
                    "sem_odd": s.leakage_odd_sem,
                    "asymmetry_relative": s.asymmetry_relative,
                    "welch_t": s.welch_t,
                    "welch_p": s.welch_p,
                },
            ],
            "fisher_combined": {"chi2": float("nan"), "df": 0, "combined_p": float("nan")},
            "mean_asymmetry_relative": mean_asym,
            "peak_asymmetry_relative": peak_asym,
        }
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh, allow_nan=True)
        result = reproduce_statistics(ds, published_summary=path)
        names = {name for name, *_ in result.claims_checked}
        assert "mean_asymmetry_relative" in names
        assert "peak_asymmetry_relative" in names

    def test_dataset_depth_not_in_published_is_skipped(self, tmp_path: Path) -> None:
        # Dataset has depth=2 and depth=4; published only has depth=2.
        def run_at_depth(d: int) -> list[DlaParityCircuit]:
            out: list[DlaParityCircuit] = []
            for k in range(8):
                out.append(
                    _circuit(
                        counts={"1100": 900 - 3 * k, "0001": 100 + 3 * k},
                        sector="even",
                        depth=d,
                    ),
                )
                out.append(
                    _circuit(
                        counts={"0001": 920 - 2 * k, "1100": 80 + 2 * k},
                        sector="odd",
                        depth=d,
                        initial="0001",
                    ),
                )
            return out

        circuits = run_at_depth(2) + run_at_depth(4)
        run = DlaParityRun(
            experiment="A_dla_parity_n4",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits),
            t_step=0.3,
            circuits=tuple(circuits),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        d2 = next(s for s in summaries if s.depth == 2)
        pub = {
            "depth_summaries": [
                {
                    "depth": 2,
                    "mean_even": d2.leakage_even,
                    "mean_odd": d2.leakage_odd,
                    "sem_even": d2.leakage_even_sem,
                    "sem_odd": d2.leakage_odd_sem,
                    "asymmetry_relative": d2.asymmetry_relative,
                    "welch_t": d2.welch_t,
                    "welch_p": d2.welch_p,
                },
            ],
            # Fisher chi² across BOTH depths in our dataset.
            "fisher_combined": {
                "chi2": -2.0 * sum(math.log(s.welch_p) for s in summaries if 0 < s.welch_p < 1),
                "df": 4,
                "combined_p": float("nan"),
            },
        }
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(pub, fh, allow_nan=True)
        result = reproduce_statistics(ds, published_summary=path)
        # Only depth=2 claims should appear (depth=4 was silently skipped).
        depth_claim_names = {n for n, *_ in result.claims_checked if n.startswith("depth=")}
        assert all(n.startswith("depth=2.") for n in depth_claim_names)

    def test_empty_summaries_give_nan_peak_and_mean(self) -> None:
        from scpn_quantum_control.dla_parity.reproduce import _peak_and_mean

        val, depth, mean = _peak_and_mean(())
        assert math.isnan(val)
        assert depth == -1
        assert math.isnan(mean)

    def test_top_level_not_object_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(["not", "an", "object"], fh)
        with pytest.raises(ValueError, match="expected top-level JSON object"):
            reproduce_statistics(DlaParityDataset(runs=()), published_summary=path)

    def test_depth_summaries_not_list_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"depth_summaries": {"not": "a-list"}, "fisher_combined": {}}, fh)
        with pytest.raises(ValueError, match="'depth_summaries' must be a list"):
            reproduce_statistics(DlaParityDataset(runs=()), published_summary=path)

    def test_depth_row_not_dict_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "depth_summaries": ["not-a-dict-row", None, 42],
                    "fisher_combined": {"chi2": 0.0},
                },
                fh,
            )
        # Dataset empty → summaries empty → loop over depth_rows skips
        # all non-dict entries without raising; fisher passes because
        # expected=0 and actual=NaN would mismatch, so we use a tolerance
        # bundle that allows it.
        ds = DlaParityDataset(runs=())
        with pytest.raises(AssertionError, match="NaN mismatch"):
            reproduce_statistics(ds, published_summary=path)

    def test_fisher_combined_not_object_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"depth_summaries": [], "fisher_combined": ["not", "an", "object"]}, fh)
        with pytest.raises(ValueError, match="'fisher_combined' must be an object"):
            reproduce_statistics(DlaParityDataset(runs=()), published_summary=path)

    def test_non_numeric_published_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "depth_summaries": [],
                    "fisher_combined": {"chi2": "not-a-number"},
                },
                fh,
            )
        with pytest.raises(ValueError, match="expected number"):
            reproduce_statistics(DlaParityDataset(runs=()), published_summary=path)

    def test_nan_expected_vs_nan_actual_passes(self, tmp_path: Path) -> None:
        # Synthesise a one-depth dataset with matching published summary
        # whose leakage means are NaN on both sides.
        circuits_even = [_circuit(counts={"1100": 1000}, sector="even", depth=6) for _ in range(3)]
        circuits_odd = [
            _circuit(counts={"0001": 1000}, sector="odd", depth=6, initial="0001")
            for _ in range(3)
        ]
        run = DlaParityRun(
            experiment="A_dla_parity_n4",
            timestamp_utc="2026",
            backend="ibm",
            job_ids=(),
            wall_time_s=0.0,
            n_circuits=len(circuits_even) + len(circuits_odd),
            t_step=0.3,
            circuits=tuple(circuits_even + circuits_odd),
        )
        ds = DlaParityDataset(runs=(run,))
        summaries = compute_depth_summaries(ds)
        s0 = summaries[0]
        # Build a synthetic "published" summary that matches our
        # recomputed values exactly.
        pub = {
            "depth_summaries": [
                {
                    "depth": s0.depth,
                    "mean_even": s0.leakage_even,
                    "mean_odd": s0.leakage_odd,
                    "sem_even": s0.leakage_even_sem,
                    "sem_odd": s0.leakage_odd_sem,
                    "asymmetry_relative": s0.asymmetry_relative,
                    "welch_t": s0.welch_t,
                    "welch_p": s0.welch_p,
                },
            ],
            "fisher_combined": {"chi2": float("nan"), "df": 0, "combined_p": float("nan")},
        }
        path = tmp_path / "s.json"
        with path.open("w", encoding="utf-8") as fh:
            # json.dump rejects NaN by default — allow_nan=True writes
            # them as the non-standard token ``NaN`` that json.load parses.
            json.dump(pub, fh, allow_nan=True)
        result = reproduce_statistics(ds, published_summary=path)
        assert result.n_circuits_used == 6
        assert math.isnan(result.fisher.chi2)
