# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the on-QPU dynamic feedback submitter
"""Tests for scripts/submit_onqpu_dynamic_feedback.py.

Arm construction is pinned to the committed feedback templates (conditional
blocks present in the ON arm and absent in the matched OFF arm), the star
layout selection, capability discovery, TVD arithmetic with its multinomial
error label, monitor trigger marginals, fail-closed count coercion, and the
CLI (dry-run, blocked, and fully mocked submission paths) are all exercised
without hardware access.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scripts import submit_onqpu_dynamic_feedback as script

SHOTS = 4096
CAL_SHOTS = 8192


class TestArmCircuits:
    def test_on_arm_carries_conditionals_and_off_arm_does_not(self) -> None:
        on_circuit = script.arm_circuit("feedback_on", 2)
        off_circuit = script.arm_circuit("feedback_off", 2)
        on_ops = dict(on_circuit.count_ops())
        off_ops = dict(off_circuit.count_ops())
        assert on_ops.get("if_else", 0) == 2
        assert "if_else" not in off_ops
        # The matched control keeps the monitor interaction and measurement.
        assert off_ops["measure"] == on_ops["measure"]
        assert on_circuit.name == "rc1_feedback_on_rounds2"
        assert off_circuit.name == "rc1_feedback_off_rounds2"

    def test_register_shapes_follow_the_template(self) -> None:
        circuit = script.arm_circuit("feedback_on", 3)
        register_names = {register.name: register.size for register in circuit.cregs}
        assert register_names == {"monitor_bit": 3, "readout": 3}
        assert circuit.num_qubits == script.TOTAL_QUBITS

    def test_unknown_arm_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown campaign arm"):
            script.arm_circuit("feedback_maybe", 2)

    def test_calibration_circuits_prepare_extremes(self) -> None:
        zeros, ones = script.calibration_circuits()
        assert zeros.name == "rc1_cal0" and ones.name == "rc1_cal1"
        assert zeros.num_qubits == script.TOTAL_QUBITS
        assert dict(ones.count_ops())["x"] == script.TOTAL_QUBITS


class FakeProps:
    def __init__(self, error: float | None) -> None:
        self.error = error


class FakeTarget:
    def __init__(self, op_maps: dict[str, Any], extra_operations: tuple[str, ...] = ()) -> None:
        self._op_maps = op_maps
        self.operation_names = tuple(op_maps) + extra_operations

    def __getitem__(self, name: str) -> Any:
        return self._op_maps.get(name, {})


DYNAMIC_OPS = ("if_else", "reset")


def star_target(extra_operations: tuple[str, ...] = DYNAMIC_OPS) -> FakeTarget:
    """Qubit 1 is a degree-3 centre; qubit 5 hangs off qubit 4."""
    return FakeTarget(
        {
            "cz": {
                (0, 1): FakeProps(0.002),
                (1, 2): FakeProps(0.003),
                (1, 3): FakeProps(0.004),
                (4, 5): FakeProps(0.001),
                (1, 4): FakeProps(0.02),
            },
            "measure": {(index,): FakeProps(0.01) for index in range(6)},
        },
        extra_operations=extra_operations,
    )


class TestCapabilityAndLayout:
    def test_dynamic_support_missing_names_the_gaps(self) -> None:
        target = star_target(extra_operations=())
        assert script.dynamic_support_missing(target) == ["if_else", "reset"]
        assert script.dynamic_support_missing(star_target()) == []

    def test_star_layout_selects_the_best_centre(self) -> None:
        star = script.select_star_layout(star_target())
        assert star.monitor == 1
        assert set(star.system) == {0, 2, 3}
        assert star.initial_layout == [*star.system, 1]
        assert len(star.readout_errors) == 4

    def test_two_centres_keep_the_cheaper_one(self) -> None:
        target = FakeTarget(
            {
                "cz": {
                    (0, 1): FakeProps(0.001),
                    (1, 2): FakeProps(0.001),
                    (1, 3): FakeProps(0.001),
                    (4, 5): FakeProps(0.05),
                    (5, 6): FakeProps(0.05),
                    (5, 7): FakeProps(0.05),
                },
                "measure": {(index,): FakeProps(0.01) for index in range(8)},
            },
            extra_operations=DYNAMIC_OPS,
        )
        star = script.select_star_layout(target)
        assert star.monitor == 1

    def test_no_degree_three_centre_fails_closed(self) -> None:
        line = FakeTarget(
            {
                "cz": {(0, 1): FakeProps(0.002), (1, 2): FakeProps(0.003)},
                "measure": {(index,): FakeProps(0.01) for index in range(3)},
            }
        )
        with pytest.raises(ValueError, match="three calibrated neighbours"):
            script.select_star_layout(line)

    def test_uncalibrated_readout_excludes_the_edge(self) -> None:
        target = FakeTarget(
            {
                "cz": {
                    (0, 1): FakeProps(0.002),
                    (1, 2): FakeProps(0.003),
                    (1, 3): FakeProps(0.004),
                },
                "measure": {(0,): FakeProps(0.01), (1,): FakeProps(0.01), (2,): FakeProps(0.01)},
            }
        )
        with pytest.raises(ValueError, match="three calibrated neighbours"):
            script.select_star_layout(target)


class TestTvdArithmetic:
    def test_identical_distributions_give_zero(self) -> None:
        counts = {"000": 2048, "111": 2048}
        row = script.total_variation_distance(counts, counts)
        assert row["tvd"] == pytest.approx(0.0)
        assert "multinomial" in row["sigma_note"]

    def test_disjoint_distributions_give_one(self) -> None:
        row = script.total_variation_distance({"000": 100}, {"111": 100})
        assert row["tvd"] == pytest.approx(1.0)

    def test_known_half_shift(self) -> None:
        row = script.total_variation_distance({"000": 50, "111": 50}, {"000": 100})
        assert row["tvd"] == pytest.approx(0.5)
        bins = {entry["bitstring"]: entry for entry in row["per_bin"]}
        assert bins["111"]["delta"] == pytest.approx(0.5)
        assert bins["111"]["z_score"] > 0.0

    def test_zero_shot_table_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="at least one shot"):
            script.total_variation_distance({}, {"000": 1})

    def test_saturated_bin_has_zero_sigma_and_zero_z(self) -> None:
        row = script.total_variation_distance({"000": 10}, {"000": 10})
        assert row["per_bin"][0]["sigma"] == 0.0
        assert row["per_bin"][0]["z_score"] == 0.0


class TestMonitorRates:
    def test_marginals_are_little_endian_per_round(self) -> None:
        counts = {"01": 60, "10": 40}
        rates = script.monitor_trigger_rates(counts, 2)
        assert rates[0] == pytest.approx(0.6)  # round 0 = last character
        assert rates[1] == pytest.approx(0.4)

    def test_zero_shots_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="at least one shot"):
            script.monitor_trigger_rates({}, 2)


class TestCoerceAndEstimate:
    def test_coerce_pads_and_conserves(self) -> None:
        coerced = script.coerce_counts(
            {"1": 3, "000": 5}, width=3, expected_shots=8, field_name="test"
        )
        assert coerced == {"001": 3, "000": 5}

    def test_coerce_duplicate_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="duplicate bitstring"):
            script.coerce_counts({"1": 4, "001": 4}, width=3, expected_shots=8, field_name="test")

    def test_coerce_shot_mismatch_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="expected 9"):
            script.coerce_counts({"000": 8}, width=3, expected_shots=9, field_name="test")

    def test_estimate_applies_the_dynamic_multiplier(self) -> None:
        # 4 dynamic mains at 1.1 × 3 plus 2 calibrations at 1.1.
        assert script.estimate_qpu_seconds(4, 2) == pytest.approx(15.4)


class TestIoHelpers:
    def test_write_json_roundtrip(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.json"
        digest = script._write_json(target, {"a": 1})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        assert digest == script._sha256(target)

    def test_timestamp_shape(self) -> None:
        stamp = script._timestamp()
        assert len(stamp) == 16 and stamp.endswith("Z")


class FakeRegister:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


class FakePubResult:
    def __init__(self, registers: dict[str, dict[str, int]]) -> None:
        self.data = types.SimpleNamespace(
            **{name: FakeRegister(counts) for name, counts in registers.items()}
        )


class TestRegisterCounts:
    def test_reads_the_named_register(self) -> None:
        pub = FakePubResult({"readout": {"000": 8}, "monitor_bit": {"01": 8}})
        assert script._register_counts(pub, "readout") == {"000": 8}
        assert script._register_counts(pub, "monitor_bit") == {"01": 8}

    def test_missing_register_fails_closed(self) -> None:
        pub = FakePubResult({"readout": {"000": 8}})
        with pytest.raises(RuntimeError, match="monitor_bit"):
            script._register_counts(pub, "monitor_bit")


class FakeJob:
    def __init__(self, results: list[FakePubResult]) -> None:
        self._results = results

    def job_id(self) -> str:
        return "test-rc1-job"

    def result(self, timeout: float) -> list[FakePubResult]:
        assert timeout > 0
        return self._results


class FakeSampler:
    last_pubs: list[tuple[Any, Any, int]] = []

    def __init__(self, mode: Any) -> None:
        self.mode = mode

    def run(self, pubs: list[tuple[Any, Any, int]]) -> FakeJob:
        FakeSampler.last_pubs = pubs
        results: list[FakePubResult] = []
        for circuit, _, shots in pubs:
            if circuit.name.startswith("rc1_cal"):
                bit = "1" if circuit.name.endswith("1") else "0"
                results.append(FakePubResult({"meas": {bit * script.TOTAL_QUBITS: shots}}))
                continue
            n_rounds = int(circuit.name.rsplit("rounds", 1)[1])
            if "feedback_on" in circuit.name:
                readout = {"000": shots // 2, "011": shots - shots // 2}
            else:
                readout = {"000": shots}
            results.append(
                FakePubResult(
                    {
                        "readout": readout,
                        "monitor_bit": {
                            "0" * n_rounds: shots // 2,
                            "1" * n_rounds: shots - shots // 2,
                        },
                    }
                )
            )
        return FakeJob(results)


@pytest.fixture()
def mocked_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route backend loading, transpilation, and the sampler to fakes."""
    import scripts.prepare_s1_ibm_live_readiness as readiness_module

    fake_backend = types.SimpleNamespace(target=star_target())
    monkeypatch.setattr(readiness_module, "load_authenticated_backend", lambda *args: fake_backend)
    # Intercept both transpile calls: the routed (backend=...) pass and the
    # unrouted basis-translation reference pass return the circuits as-is,
    # giving depth ratios of exactly 1.0.
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
        assert len(list(tmp_path.iterdir())) == 1

    def test_missing_dynamic_support_blocks(
        self,
        mocked_hardware: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.prepare_s1_ibm_live_readiness as readiness_module

        fake_backend = types.SimpleNamespace(target=star_target(extra_operations=()))
        monkeypatch.setattr(
            readiness_module, "load_authenticated_backend", lambda *args: fake_backend
        )
        assert script.main(["--out-dir", str(tmp_path)]) == 3
        assert "readiness=blocked" in capsys.readouterr().out

    def test_depth_ratio_ceiling_blocks(
        self,
        mocked_hardware: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = script.main(["--out-dir", str(tmp_path), "--max-depth-ratio", "0.1"])
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
        assert "job_id=test-rc1-job" in output

        shots_by_position = [shots for _, _, shots in FakeSampler.last_pubs]
        assert shots_by_position == [SHOTS] * 4 + [CAL_SHOTS] * 2

        raw_path = next(path for path in tmp_path.iterdir() if "raw_counts" in path.name)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert raw["schema"] == "scpn_onqpu_dynamic_feedback_raw_counts_v1"
        assert len(raw["results"]) == 6
        main_rows = [row for row in raw["results"] if "readout_counts" in row]
        assert len(main_rows) == 4
        assert all("monitor_counts" in row for row in main_rows)

        analysis_path = next(path for path in tmp_path.iterdir() if "analysis" in path.name)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert len(analysis["round_settings"]) == 2
        for row in analysis["round_settings"]:
            # The fake ON arm moves half the mass to "011": TVD = 0.5.
            assert row["comparison"]["tvd"] == pytest.approx(0.5)
            rates = row["monitor_trigger_rates"]
            assert all(rate == pytest.approx(0.5) for rate in rates["feedback_on"])
        assert "latency" in analysis["claim_boundary"]

    def test_parse_args_defaults(self) -> None:
        args = script._parse_args([])
        assert args.backend == script.DEFAULT_BACKEND
        assert args.shots == SHOTS
        assert args.max_qpu_seconds == 120.0
        assert args.max_depth_ratio == pytest.approx(2.0)
        assert not args.submit


class TestModuleBootstrap:
    def test_reload_inserts_repo_root_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        stripped = [entry for entry in sys.path if entry != str(script.REPO_ROOT)]
        monkeypatch.setattr(sys, "path", stripped)
        reloaded = importlib.reload(script)
        assert sys.path[0] == str(reloaded.REPO_ROOT)
