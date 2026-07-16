# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the mitigated Bell re-run submitter
"""Tests for scripts/submit_bell_rerun_mitigated.py.

The circuit tests pin the preregistered analyser geometry against exact
statevector arithmetic (ideal S = 2√2 with the minus sign on setting 1, the
recompute-script convention). Layout selection, fail-closed count coercion,
mitigation arithmetic, the preregistered decision rule, readiness gating,
and the CLI (dry-run, blocked, and fully mocked submission paths) are all
exercised without any hardware access.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.mitigation.readout_matrix import (
    build_readout_confusion_matrix,
    computational_basis_labels,
)
from scripts import submit_bell_rerun_mitigated as script
from scripts.recompute_chsh_bell_test import QUBIT_PAIRS, chsh_for_pair

SHOTS = 4096
CAL_SHOTS = 8192
LABELS = computational_basis_labels(script.N_QUBITS)


def ideal_setting_counts(shots: int = SHOTS) -> list[dict[str, int]]:
    """Exact statevector probabilities scaled to an integer count table.

    Rounding residue is absorbed into the most likely bitstring so shot
    conservation holds exactly.
    """
    tables: list[dict[str, int]] = []
    for circuit in script.chsh_setting_circuits():
        body = circuit.remove_final_measurements(inplace=False)
        probabilities = Statevector(body).probabilities_dict()
        counts = {key: int(round(value * shots)) for key, value in probabilities.items()}
        counts = {key: value for key, value in counts.items() if value > 0}
        drift = shots - sum(counts.values())
        top = max(counts, key=lambda key: counts[key])
        counts[top] += drift
        tables.append(counts)
    return tables


def perfect_calibration_counts(shots: int = CAL_SHOTS) -> dict[str, dict[str, int]]:
    """Noise-free calibration: every prepared state reads back exactly."""
    return {label: {label: shots} for label in LABELS}


class FakeProps:
    def __init__(self, error: float | None) -> None:
        self.error = error


class FakeTarget:
    """Minimal stand-in for a qiskit Target: names + per-op property maps."""

    def __init__(self, op_maps: dict[str, dict[tuple[int, ...] | None, FakeProps]]) -> None:
        self._op_maps = op_maps
        self.operation_names = tuple(op_maps)

    def __getitem__(self, name: str) -> dict[tuple[int, ...] | None, FakeProps]:
        return self._op_maps[name]


def good_target() -> FakeTarget:
    """Four calibrated edges; (0,1) and (2,3) are the two best disjoint ones."""
    return FakeTarget(
        {
            "cz": {
                (0, 1): FakeProps(0.002),
                (1, 2): FakeProps(0.001),
                (2, 3): FakeProps(0.004),
                (4, 5): FakeProps(0.030),
            },
            "measure": {
                (0,): FakeProps(0.010),
                (1,): FakeProps(0.012),
                (2,): FakeProps(0.011),
                (3,): FakeProps(0.013),
                (4,): FakeProps(0.020),
                (5,): FakeProps(0.020),
            },
        }
    )


class TestChshSettingCircuits:
    def test_four_settings_with_pinned_names(self) -> None:
        circuits = script.chsh_setting_circuits()
        assert [circuit.name for circuit in circuits] == [
            "bell_rerun_s0",
            "bell_rerun_s1",
            "bell_rerun_s2",
            "bell_rerun_s3",
        ]
        assert all(circuit.num_qubits == script.N_QUBITS for circuit in circuits)

    def test_ideal_statistics_reach_tsirelson_on_both_pairs(self) -> None:
        entries = [{"counts": counts} for counts in ideal_setting_counts()]
        for pair in QUBIT_PAIRS:
            statistics = chsh_for_pair(entries, pair)
            assert statistics.s_value == pytest.approx(2.0 * math.sqrt(2.0), abs=2e-3)
            assert statistics.settings_e[1] == pytest.approx(-1.0 / math.sqrt(2.0), abs=2e-3)

    def test_setting_angles_follow_the_sign_convention(self) -> None:
        (a0, b0), (a1, b1), (a2, b2), (a3, b3) = script.SETTING_ANGLES
        assert (a0, a1) == (0.0, 0.0)
        assert (a2, a3) == (math.pi / 2.0, math.pi / 2.0)
        assert (b0, b2) == (math.pi / 4.0, math.pi / 4.0)
        assert (b1, b3) == (3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0)


class TestCalibrationCircuits:
    def test_sixteen_circuits_prepare_their_labels_exactly(self) -> None:
        circuits = script.calibration_circuits()
        assert len(circuits) == 2**script.N_QUBITS
        for label, circuit in zip(LABELS, circuits, strict=True):
            assert circuit.name == f"bell_rerun_cal_{label}"
            body = circuit.remove_final_measurements(inplace=False)
            probabilities = Statevector(body).probabilities_dict()
            assert probabilities == pytest.approx({label: 1.0})


class TestLayoutSelection:
    def test_selects_two_best_disjoint_edges(self) -> None:
        layout = script.select_physical_qubits(good_target())
        # (1,2) scores 0.001 + 0.0115 = 0.0125 — the best edge. (0,1) and
        # (2,3) both overlap it, so the best disjoint partner is (4,5).
        assert layout.physical_qubits == (1, 2, 4, 5)
        assert layout.edge_errors == (0.001, 0.030)
        assert layout.readout_errors == (0.012, 0.011, 0.020, 0.020)
        assert layout.worst_readout_error == 0.020

    def test_no_calibrated_edges_fails_closed(self) -> None:
        target = FakeTarget({"measure": {(0,): FakeProps(0.01)}})
        with pytest.raises(ValueError, match="no fully calibrated two-qubit edges"):
            script.select_physical_qubits(target)

    def test_no_disjoint_second_edge_fails_closed(self) -> None:
        target = FakeTarget(
            {
                "cz": {(0, 1): FakeProps(0.002), (1, 2): FakeProps(0.003)},
                "measure": {
                    (0,): FakeProps(0.01),
                    (1,): FakeProps(0.01),
                    (2,): FakeProps(0.01),
                },
            }
        )
        with pytest.raises(ValueError, match="no two disjoint"):
            script.select_physical_qubits(target)

    def test_edges_without_readout_are_skipped(self) -> None:
        target = FakeTarget(
            {
                "cz": {
                    (0, 1): FakeProps(0.0001),
                    (2, 3): FakeProps(0.002),
                    (4, 5): FakeProps(0.003),
                },
                "measure": {
                    (2,): FakeProps(0.01),
                    (3,): FakeProps(0.01),
                    (4,): FakeProps(0.01),
                    (5,): FakeProps(0.01),
                },
            }
        )
        layout = script.select_physical_qubits(target)
        assert layout.physical_qubits == (2, 3, 4, 5)

    def test_none_qargs_and_missing_errors_are_ignored(self) -> None:
        edge_maps: dict[str, dict[tuple[int, ...] | None, FakeProps]] = {
            "cz": {
                None: FakeProps(0.5),
                (0,): FakeProps(0.5),
                (0, 1): FakeProps(None),
                (2, 3): FakeProps(0.002),
                (4, 5): FakeProps(0.003),
            },
        }
        edges = script._two_qubit_edge_errors(FakeTarget(edge_maps))
        assert edges == {(2, 3): 0.002, (4, 5): 0.003}
        readout_maps: dict[str, dict[tuple[int, ...] | None, FakeProps]] = {
            "measure": {
                None: FakeProps(0.5),
                (0, 1): FakeProps(0.5),
                (2,): FakeProps(0.01),
                (3,): FakeProps(None),
                (4,): FakeProps(0.01),
                (5,): FakeProps(0.01),
            },
        }
        readout = script._readout_errors(FakeTarget(readout_maps))
        assert readout == {2: 0.01, 4: 0.01, 5: 0.01}

    def test_duplicate_edge_keeps_the_smaller_error(self) -> None:
        op_maps: dict[str, dict[tuple[int, ...] | None, FakeProps]] = {
            "cz": {(0, 1): FakeProps(0.005)},
            "ecr": {(1, 0): FakeProps(0.003)},
        }
        edges = script._two_qubit_edge_errors(FakeTarget(op_maps))
        assert edges == {(0, 1): 0.003}

    def test_duplicate_edge_ignores_the_larger_error(self) -> None:
        op_maps: dict[str, dict[tuple[int, ...] | None, FakeProps]] = {
            "cz": {(0, 1): FakeProps(0.003)},
            "ecr": {(1, 0): FakeProps(0.005)},
        }
        edges = script._two_qubit_edge_errors(FakeTarget(op_maps))
        assert edges == {(0, 1): 0.003}

    def test_none_property_map_is_skipped(self) -> None:
        op_maps: dict[str, Any] = {
            "delay": None,
            "cz": {(0, 1): FakeProps(0.002)},
        }
        edges = script._two_qubit_edge_errors(FakeTarget(op_maps))
        assert edges == {(0, 1): 0.002}


class TestCoerceCounts:
    def test_valid_counts_pass_and_pad_to_width(self) -> None:
        coerced = script.coerce_counts({"11": 3, "0000": 5}, expected_shots=8, field_name="test")
        assert coerced == {"0011": 3, "0000": 5}

    def test_width_overflow_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="exactly 4 bits"):
            script.coerce_counts({"00000": 8}, expected_shots=8, field_name="test")

    def test_negative_count_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="test value"):
            script.coerce_counts({"0000": -1}, expected_shots=8, field_name="test")

    def test_duplicate_bitstring_after_padding_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="duplicate bitstring"):
            script.coerce_counts({"11": 4, "0011": 4}, expected_shots=8, field_name="test")

    def test_shot_conservation_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="expected 9, observed 8"):
            script.coerce_counts({"0000": 8}, expected_shots=9, field_name="test")


class TestProbabilityPairCorrelator:
    def test_matches_parity_convention(self) -> None:
        probabilities = [0.0] * len(LABELS)
        probabilities[LABELS.index("0000")] = 0.5
        probabilities[LABELS.index("0010")] = 0.5
        # "0010" flips qubit 1 only: pair (0, 1) decorrelates, pair (2, 3)
        # stays perfectly correlated.
        assert script.probability_pair_correlator(probabilities, LABELS, (0, 1)) == pytest.approx(
            0.0
        )
        assert script.probability_pair_correlator(probabilities, LABELS, (2, 3)) == pytest.approx(
            1.0
        )
        probabilities = [0.0] * len(LABELS)
        probabilities[LABELS.index("0011")] = 1.0
        assert script.probability_pair_correlator(probabilities, LABELS, (0, 1)) == pytest.approx(
            1.0
        )
        assert script.probability_pair_correlator(probabilities, LABELS, (2, 3)) == pytest.approx(
            1.0
        )

    def test_negative_quasi_probability_enters_unclipped(self) -> None:
        probabilities = [0.0] * len(LABELS)
        probabilities[LABELS.index("0000")] = 1.1
        probabilities[LABELS.index("0001")] = -0.1
        assert script.probability_pair_correlator(probabilities, LABELS, (0, 1)) == pytest.approx(
            1.2
        )


class TestMitigatedStatistics:
    def test_identity_confusion_matrix_reproduces_unmitigated(self) -> None:
        setting_counts = ideal_setting_counts()
        confusion = build_readout_confusion_matrix(perfect_calibration_counts(), script.N_QUBITS)
        entries = [{"counts": counts} for counts in setting_counts]
        for pair in QUBIT_PAIRS:
            mitigated = script.mitigated_pair_statistics(
                setting_counts, confusion, pair, shots=SHOTS
            )
            unmitigated = chsh_for_pair(entries, pair)
            assert mitigated["settings_e"] == pytest.approx(list(unmitigated.settings_e))
            assert mitigated["s_value"] == pytest.approx(unmitigated.s_value)
            assert mitigated["sigma_multinomial_approximation"] == pytest.approx(unmitigated.sigma)
            assert "inversion" in mitigated["sigma_note"]


class TestSetting1BandDecision:
    def test_within_band_reads_as_artefact(self) -> None:
        decision = script.setting1_band_decision([0.8, -0.79, 0.82, 0.81], shots=SHOTS)
        assert decision["within_2_sigma_of_band"] is True
        assert "artefact" in decision["preregistered_reading"]
        assert decision["other_settings_abs_e_band"] == [
            pytest.approx(0.8),
            pytest.approx(0.82),
        ]

    def test_far_below_band_reads_as_persistent_asymmetry(self) -> None:
        decision = script.setting1_band_decision([0.8, 0.30, 0.82, 0.81], shots=SHOTS)
        assert decision["within_2_sigma_of_band"] is False
        assert "persists" in decision["preregistered_reading"]

    def test_saturated_correlator_clamps_sigma_at_zero(self) -> None:
        decision = script.setting1_band_decision([1.0, -1.0, 1.0, 1.0], shots=SHOTS)
        assert decision["sigma_setting1"] == 0.0
        assert decision["within_2_sigma_of_band"] is True


class TestBuildAnalysis:
    def test_full_analysis_shape_and_values(self) -> None:
        analysis = script.build_analysis(
            ideal_setting_counts(), perfect_calibration_counts(), shots=SHOTS
        )
        assert analysis["schema"] == "scpn_bell_rerun_mitigated_analysis_v1"
        assert analysis["experiment_id"] == script.EXPERIMENT_ID
        assert analysis["preregistration"] == script.PREREGISTRATION_PATH
        assert len(analysis["pairs"]) == 2
        for pair_row in analysis["pairs"]:
            unmitigated = pair_row["unmitigated"]
            assert unmitigated["s_value"] == pytest.approx(2.0 * math.sqrt(2.0), abs=2e-3)
            assert unmitigated["sigma"] > 0.0
            assert unmitigated["significance"] > 0.0
            assert pair_row["mitigated"]["s_value"] == pytest.approx(unmitigated["s_value"])
            assert pair_row["decision_rule"]["within_2_sigma_of_band"] is True
        assert "loophole-free" in analysis["claim_boundary"]


class TestReadiness:
    def _layout(self) -> script.LayoutSelection:
        return script.LayoutSelection(
            physical_qubits=(1, 2, 4, 5),
            edge_errors=(0.001, 0.030),
            readout_errors=(0.012, 0.011, 0.020, 0.020),
        )

    def _args(self, **overrides: Any) -> Any:
        defaults: dict[str, Any] = {
            "shots": SHOTS,
            "cal_shots": CAL_SHOTS,
            "max_qpu_seconds": 60.0,
            "max_depth": 60,
            "readout_error_abort": 0.15,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_ready_when_all_gates_pass(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=self._layout(),
            isa_depths=[12, 14, 13, 15] + [3] * 16,
            args=self._args(),
        )
        assert readiness["status"] == "ready_for_submission"
        assert readiness["n_circuits"] == 20
        assert readiness["estimated_qpu_seconds"] == pytest.approx(22.0)
        assert readiness["layout"]["physical_qubits"] == [1, 2, 4, 5]
        assert "full-basis" in readiness["mitigation_deviation_note"]

    def test_blocked_by_readout_error(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=self._layout(),
            isa_depths=[12] * 20,
            args=self._args(readout_error_abort=0.015),
        )
        assert readiness["status"] == "blocked"
        assert any("readout-error abort" in reason for reason in readiness["reasons"])

    def test_blocked_by_depth(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=self._layout(),
            isa_depths=[12] * 19 + [999],
            args=self._args(),
        )
        assert readiness["status"] == "blocked"
        assert any("depth ceiling" in reason for reason in readiness["reasons"])

    def test_blocked_by_budget(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=self._layout(),
            isa_depths=[12] * 20,
            args=self._args(max_qpu_seconds=10.0),
        )
        assert readiness["status"] == "blocked"
        assert any("QPU-second cap" in reason for reason in readiness["reasons"])


class TestEstimate:
    def test_uses_the_empirical_anchor(self) -> None:
        assert script.estimate_qpu_seconds(20) == pytest.approx(22.0)
        assert pytest.approx(1.1) == script.ESTIMATED_SECONDS_PER_CIRCUIT


class TestIoHelpers:
    def test_write_json_roundtrip_with_matching_digest(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "payload.json"
        digest = script._write_json(target, {"a": 1})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        assert digest == script._sha256(target)
        assert len(digest) == 64

    def test_timestamp_shape(self) -> None:
        stamp = script._timestamp()
        assert len(stamp) == 16
        assert stamp.endswith("Z")


class FakeRegister:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


class FakePubResult:
    def __init__(self, counts: dict[str, int], register: str = "meas") -> None:
        self.data = types.SimpleNamespace(**{register: FakeRegister(counts)})


class TestPubCounts:
    def test_reads_meas_register(self) -> None:
        assert script._pub_counts(FakePubResult({"0000": 8})) == {"0000": 8}

    def test_reads_alternate_register_names(self) -> None:
        assert script._pub_counts(FakePubResult({"0000": 8}, register="c0")) == {"0000": 8}

    def test_no_register_fails_closed(self) -> None:
        empty = types.SimpleNamespace(data=types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="no classical register"):
            script._pub_counts(empty)


class FakeJob:
    def __init__(self, results: list[FakePubResult]) -> None:
        self._results = results

    def job_id(self) -> str:
        return "test-bell-rerun-job"

    def result(self, timeout: float) -> list[FakePubResult]:
        assert timeout > 0
        return self._results


class FakeSampler:
    last_pubs: list[tuple[Any, Any, int]] = []

    def __init__(self, mode: Any) -> None:
        self.mode = mode

    def run(self, pubs: list[tuple[Any, Any, int]]) -> FakeJob:
        FakeSampler.last_pubs = pubs
        results = [FakePubResult(counts) for counts in ideal_setting_counts()]
        results += [FakePubResult({label: CAL_SHOTS}) for label in LABELS]
        return FakeJob(results)


@pytest.fixture()
def mocked_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route backend loading, transpilation, and the sampler to fakes."""
    import scripts.prepare_s1_ibm_live_readiness as readiness_module

    fake_backend = types.SimpleNamespace(target=good_target())
    monkeypatch.setattr(readiness_module, "load_authenticated_backend", lambda *args: fake_backend)
    monkeypatch.setattr(script, "transpile", lambda circuits, **kwargs: circuits)
    runtime = types.ModuleType("qiskit_ibm_runtime")
    runtime.SamplerV2 = FakeSampler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", runtime)


class TestMain:
    def test_submit_without_confirm_budget_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert script.main(["--submit"]) == 2
        assert "requires --confirm-budget" in capsys.readouterr().err

    def test_dry_run_writes_readiness_only(
        self,
        mocked_hardware: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert script.main(["--out-dir", str(tmp_path)]) == 0
        output = capsys.readouterr().out
        assert "readiness=ready_for_submission" in output
        assert "hardware_submission=false" in output
        written = sorted(path.name for path in tmp_path.iterdir())
        assert len(written) == 1
        assert written[0].startswith("bell_rerun_readiness_")

    def test_blocked_readiness_exits_three(
        self,
        mocked_hardware: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = script.main(["--out-dir", str(tmp_path), "--readout-error-abort", "0.001"])
        assert exit_code == 3
        assert "readiness=blocked" in capsys.readouterr().out

    def test_full_mocked_submission_lands_all_documents(
        self,
        mocked_hardware: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = script.main(["--out-dir", str(tmp_path), "--submit", "--confirm-budget"])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "hardware_submission=true" in output
        assert "job_id=test-bell-rerun-job" in output
        assert "S_unmitigated=2.82" in output

        shots_by_position = [shots for _, _, shots in FakeSampler.last_pubs]
        assert shots_by_position == [SHOTS] * 4 + [CAL_SHOTS] * 16

        names = sorted(path.name for path in tmp_path.iterdir())
        assert len(names) == 3
        raw_path = next(path for path in tmp_path.iterdir() if "raw_counts" in path.name)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert raw["schema"] == "scpn_bell_rerun_mitigated_raw_counts_v1"
        assert raw["job_id"] == "test-bell-rerun-job"
        assert raw["parent_artifact"] == script.PARENT_ARTIFACT
        assert len(raw["results"]) == 4
        assert len(raw["calibration_counts"]) == 16
        assert raw["approval"]["allowed_provider"] == "ibm_runtime"
        analysis_path = next(path for path in tmp_path.iterdir() if "analysis" in path.name)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        for pair_row in analysis["pairs"]:
            assert pair_row["unmitigated"]["s_value"] == pytest.approx(
                2.0 * math.sqrt(2.0), abs=2e-3
            )

    def test_parse_args_defaults(self) -> None:
        args = script._parse_args([])
        assert args.backend == script.DEFAULT_BACKEND
        assert args.shots == SHOTS
        assert args.cal_shots == CAL_SHOTS
        assert args.max_qpu_seconds == 60.0
        assert not args.submit


class TestModuleBootstrap:
    def test_reload_inserts_repo_root_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        stripped = [entry for entry in sys.path if entry != str(script.REPO_ROOT)]
        monkeypatch.setattr(sys, "path", stripped)
        reloaded = importlib.reload(script)
        assert sys.path[0] == str(reloaded.REPO_ROOT)
