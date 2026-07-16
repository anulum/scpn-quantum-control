# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the maximum-width Kuramoto-XY submitter
"""Tests for scripts/submit_max_width_kuramoto_xy.py.

The physics is pinned against exact statevector arithmetic on small widths
(the Aer MPS baseline must match the Trotterised circuit to 1e-9, and the
measurement route must recover the same per-qubit expectations). Chain
planning, per-qubit readout mitigation, the order parameter, readiness
gating, fail-closed count coercion, and the CLI (dry-run, blocked, and fully
mocked submission paths) are exercised without hardware access.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from scripts import submit_max_width_kuramoto_xy as script

SHOTS = 4096
CAL_SHOTS = 8192


def statevector_expectations(width: int, reps: int) -> tuple[list[float], list[float]]:
    """Per-qubit X/Y expectations of the Trotterised circuit, exactly."""
    state = Statevector(script.evolution_body(width, reps).decompose())
    x_values = []
    y_values = []
    for qubit in range(width):
        label_x = ["I"] * width
        label_x[width - 1 - qubit] = "X"
        label_y = ["I"] * width
        label_y[width - 1 - qubit] = "Y"
        x_values.append(float(state.expectation_value(SparsePauliOp("".join(label_x))).real))
        y_values.append(float(state.expectation_value(SparsePauliOp("".join(label_y))).real))
    return x_values, y_values


def counts_from_circuit(circuit: Any, width: int, shots: int) -> dict[str, int]:
    """Exact measurement distribution scaled to an integer count table."""
    body = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector(body.decompose()).probabilities_dict()
    counts = {key: int(round(value * shots)) for key, value in probabilities.items()}
    counts = {key: value for key, value in counts.items() if value > 0}
    drift = shots - sum(counts.values())
    top = max(counts, key=lambda key: counts[key])
    counts[top] += drift
    return {key.zfill(width): value for key, value in counts.items()}


class TestWorkloadFamily:
    def test_chain_coupling_is_nearest_neighbour_and_symmetric(self) -> None:
        coupling = script.chain_coupling(6)
        assert np.allclose(coupling, coupling.T)
        assert coupling[0, 2] == 0.0
        assert coupling[2, 4] == 0.0
        assert coupling[0, 1] == pytest.approx(0.302)  # Paper 27 Table 2 anchor
        assert coupling[4, 5] > 0.0

    def test_evolution_body_shape(self) -> None:
        body = script.evolution_body(5, 2)
        assert body.num_qubits == 5
        assert body.name == "width5_reps2"

    def test_setting_circuits_rotate_and_measure(self) -> None:
        body = script.evolution_body(3, 1)
        x_circuit = script.measured_setting_circuit(body, "x")
        y_circuit = script.measured_setting_circuit(body, "y")
        assert x_circuit.name.endswith("_x")
        assert y_circuit.name.endswith("_y")
        x_ops = dict(x_circuit.count_ops())
        y_ops = dict(y_circuit.count_ops())
        assert x_ops["h"] == 3 and "sdg" not in x_ops
        assert y_ops["h"] == 3 and y_ops["sdg"] == 3

    def test_unknown_setting_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown measurement setting"):
            script.measured_setting_circuit(script.evolution_body(2, 1), "z")

    def test_calibration_circuits_prepare_extremes(self) -> None:
        zeros, ones = script.width_calibration_circuits(3)
        zero_probs = Statevector(
            zeros.remove_final_measurements(inplace=False)
        ).probabilities_dict()
        one_probs = Statevector(ones.remove_final_measurements(inplace=False)).probabilities_dict()
        assert zero_probs == pytest.approx({"000": 1.0})
        assert one_probs == pytest.approx({"111": 1.0})


class TestExpectationArithmetic:
    def test_measurement_route_matches_statevector(self) -> None:
        width, reps = 3, 1
        x_ref, y_ref = statevector_expectations(width, reps)
        body = script.evolution_body(width, reps)
        big = 10**7
        x_counts = counts_from_circuit(script.measured_setting_circuit(body, "x"), width, big)
        y_counts = counts_from_circuit(script.measured_setting_circuit(body, "y"), width, big)
        assert script.per_qubit_z_expectations(x_counts, width) == pytest.approx(x_ref, abs=1e-3)
        assert script.per_qubit_z_expectations(y_counts, width) == pytest.approx(y_ref, abs=1e-3)

    def test_mps_baseline_matches_statevector_exactly(self) -> None:
        width, reps = 3, 2
        x_ref, y_ref = statevector_expectations(width, reps)
        x_mps, y_mps = script.exact_baseline_expectations(width, reps)
        assert x_mps == pytest.approx(x_ref, abs=1e-9)
        assert y_mps == pytest.approx(y_ref, abs=1e-9)

    def test_zero_shot_table_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="no shots"):
            script.per_qubit_z_expectations({}, 2)

    def test_order_parameter_known_values(self) -> None:
        assert script.order_parameter([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
        assert script.order_parameter([0.6, 0.6], [0.8, 0.8]) == pytest.approx(1.0)
        assert script.order_parameter([1.0, -1.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_order_parameter_shape_mismatch_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            script.order_parameter([1.0], [1.0, 0.0])
        with pytest.raises(ValueError, match="equal-length"):
            script.order_parameter([], [])


class TestReadoutMitigation:
    def test_perfect_calibration_gives_zero_epsilon(self) -> None:
        cal0 = {"00": CAL_SHOTS}
        cal1 = {"11": CAL_SHOTS}
        pairs = script.readout_error_pairs(cal0, cal1, 2)
        assert pairs == [(0.0, 0.0), (0.0, 0.0)]

    def test_known_flip_rates_are_recovered(self) -> None:
        # Qubit 0 flips 0→1 a quarter of the time; qubit 1 flips 1→0 half.
        cal0 = {"00": 6144, "01": 2048}
        cal1 = {"11": 4096, "01": 4096}
        pairs = script.readout_error_pairs(cal0, cal1, 2)
        assert pairs[0][0] == pytest.approx(0.25)
        assert pairs[0][1] == pytest.approx(0.0)
        assert pairs[1][0] == pytest.approx(0.0)
        assert pairs[1][1] == pytest.approx(0.5)

    def test_identity_channel_leaves_expectation_unchanged(self) -> None:
        assert script.mitigate_z_expectation(0.42, 0.0, 0.0) == pytest.approx(0.42)

    def test_symmetric_channel_inverts_exactly(self) -> None:
        # ε₀ = ε₁ = 0.1 shrinks <Z> by (1 − 2ε); the inversion restores it.
        true_z = 0.6
        observed = true_z * (1.0 - 2.0 * 0.1)
        assert script.mitigate_z_expectation(observed, 0.1, 0.1) == pytest.approx(true_z)

    def test_non_invertible_calibration_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="not invertible"):
            script.mitigate_z_expectation(0.0, 0.6, 0.5)


class TestAnalyseWidth:
    def test_ideal_counts_reproduce_the_baseline(self) -> None:
        width, reps = 3, 1
        body = script.evolution_body(width, reps)
        big = 10**7
        x_counts = counts_from_circuit(script.measured_setting_circuit(body, "x"), width, big)
        y_counts = counts_from_circuit(script.measured_setting_circuit(body, "y"), width, big)
        cal0 = {"0" * width: CAL_SHOTS}
        cal1 = {"1" * width: CAL_SHOTS}
        row = script.analyse_width(width, reps, x_counts, y_counts, cal0, cal1)
        assert row["r_unmitigated"] == pytest.approx(row["r_exact_mps"], abs=1e-3)
        assert row["r_mitigated"] == pytest.approx(row["r_unmitigated"], abs=1e-9)
        assert len(row["per_qubit"]["x_unmitigated"]) == width

    def test_baseline_can_be_deferred(self) -> None:
        cal = ({"00": CAL_SHOTS}, {"11": CAL_SHOTS})
        row = script.analyse_width(
            2, 1, {"00": SHOTS}, {"00": SHOTS}, *cal, include_baseline=False
        )
        assert "r_exact_mps" not in row


class TestCoerceCounts:
    def test_valid_counts_pass_and_pad(self) -> None:
        coerced = script.coerce_counts(
            {"11": 3, "000": 5}, width=3, expected_shots=8, field_name="test"
        )
        assert coerced == {"011": 3, "000": 5}

    def test_shot_mismatch_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="expected 9, observed 8"):
            script.coerce_counts({"000": 8}, width=3, expected_shots=9, field_name="test")

    def test_duplicate_after_padding_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="duplicate bitstring"):
            script.coerce_counts({"11": 4, "011": 4}, width=3, expected_shots=8, field_name="test")

    def test_negative_count_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="test value"):
            script.coerce_counts({"000": -1}, width=3, expected_shots=8, field_name="test")


class FakeProps:
    def __init__(self, error: float | None) -> None:
        self.error = error


class FakeTarget:
    def __init__(self, op_maps: dict[str, Any]) -> None:
        self._op_maps = op_maps
        self.operation_names = tuple(op_maps)

    def __getitem__(self, name: str) -> Any:
        return self._op_maps[name]


def line_target(n: int, gate: float = 0.005, readout: float = 0.01) -> FakeTarget:
    """A calibrated 1-D line of ``n`` qubits."""
    return FakeTarget(
        {
            "cz": {(i, i + 1): FakeProps(gate) for i in range(n - 1)},
            "measure": {(i,): FakeProps(readout) for i in range(n)},
        }
    )


class TestPlanWidths:
    def test_line_of_40_resolves_32_and_device_max(self) -> None:
        resolved, skipped = script.plan_widths(line_target(40), median_edge_error_abort=0.03)
        widths = [width for width, _ in resolved]
        assert widths == [32, 40]
        skipped_widths = {entry["width"] for entry in skipped}
        assert skipped_widths == {64, 104}
        assert all("device-max" in entry["reason"] for entry in skipped)

    def test_noisy_chain_is_skipped_by_the_median_threshold(self) -> None:
        resolved, skipped = script.plan_widths(
            line_target(40, gate=0.08), median_edge_error_abort=0.03
        )
        assert resolved == []
        assert any("abort threshold" in entry["reason"] for entry in skipped)

    def test_no_calibrated_chain_fails_closed(self) -> None:
        empty = FakeTarget({"measure": {(0,): FakeProps(0.01)}})
        with pytest.raises(ValueError, match="no usable calibrated chain"):
            script.plan_widths(empty, median_edge_error_abort=0.03)

    def test_unreachable_intermediate_width_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defensive branch: a preregistered width below device-max that no
        # seed reaches. Greedy growth cannot produce this on a static graph
        # (any prefix of the longest chain is reachable), so it is forced.
        monkeypatch.setattr(script, "select_error_aware_chain", lambda *args, **kwargs: None)
        resolved, skipped = script.plan_widths(line_target(40), median_edge_error_abort=0.03)
        widths = [width for width, _ in resolved]
        assert widths == [40]
        unreachable = next(entry for entry in skipped if entry["width"] == 32)
        assert "no chain of this width is reachable" in unreachable["reason"]

    def test_device_max_equal_to_preregistered_width_is_not_duplicated(self) -> None:
        resolved, skipped = script.plan_widths(line_target(32), median_edge_error_abort=0.03)
        widths = [width for width, _ in resolved]
        assert widths == [32]
        assert {entry["width"] for entry in skipped} == {32, 64, 104}
        equal_entry = next(entry for entry in skipped if entry["width"] == 32)
        assert "covered by the device-max point" in equal_entry["reason"]


class TestEstimateAndIo:
    def test_estimate_uses_the_anchor(self) -> None:
        assert script.estimate_qpu_seconds(24) == pytest.approx(26.4)

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
    def __init__(self, counts: dict[str, int], register: str = "meas") -> None:
        self.data = types.SimpleNamespace(**{register: FakeRegister(counts)})


class TestPubCounts:
    def test_reads_registers(self) -> None:
        assert script._pub_counts(FakePubResult({"0": 1})) == {"0": 1}
        assert script._pub_counts(FakePubResult({"0": 1}, register="c")) == {"0": 1}

    def test_no_register_fails_closed(self) -> None:
        empty = types.SimpleNamespace(data=types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="no classical register"):
            script._pub_counts(empty)


class FakeJob:
    def __init__(self, results: list[FakePubResult]) -> None:
        self._results = results

    def job_id(self) -> str:
        return "test-max-width-job"

    def result(self, timeout: float) -> list[FakePubResult]:
        assert timeout > 0
        return self._results


class FakeSampler:
    last_pubs: list[tuple[Any, Any, int]] = []

    def __init__(self, mode: Any) -> None:
        self.mode = mode

    def run(self, pubs: list[tuple[Any, Any, int]]) -> FakeJob:
        FakeSampler.last_pubs = pubs
        results = []
        for circuit, _, shots in pubs:
            width = circuit.num_qubits
            name = circuit.name
            if name.endswith("cal0"):
                results.append(FakePubResult({"0" * width: shots}))
            elif name.endswith("cal1"):
                results.append(FakePubResult({"1" * width: shots}))
            else:
                results.append(FakePubResult(counts_from_circuit(circuit, width, shots)))
        return FakeJob(results)


@pytest.fixture()
def mocked_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route backend loading, transpilation, and the sampler to fakes.

    The fake device is a 5-qubit line, so the preregistered widths are all
    skipped and the campaign runs at the device-max width of 5 — exercising
    the partial-campaign paths the preregistration requires.
    """
    import scripts.prepare_s1_ibm_live_readiness as readiness_module

    fake_backend = types.SimpleNamespace(target=line_target(5))
    monkeypatch.setattr(readiness_module, "load_authenticated_backend", lambda *args: fake_backend)

    from qiskit import transpile as real_transpile

    def fake_transpile(circuits: Any, **kwargs: Any) -> Any:
        if "backend" in kwargs:
            return circuits
        return real_transpile(circuits, **kwargs)

    monkeypatch.setattr(script, "transpile", fake_transpile)
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
        assert "width=5" in output
        assert "skipped_width=32" in output
        assert "hardware_submission=false" in output
        assert len(list(tmp_path.iterdir())) == 1

    def test_blocked_readiness_exits_three(
        self,
        mocked_hardware: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = script.main(
            ["--out-dir", str(tmp_path), "--median-edge-error-abort", "0.0001"]
        )
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
        assert "job_id=test-max-width-job" in output

        # 6 pubs for the single width-5 point: 4 main at SHOTS, 2 cal at CAL_SHOTS.
        shots_by_position = [shots for _, _, shots in FakeSampler.last_pubs]
        assert shots_by_position == [SHOTS] * 4 + [CAL_SHOTS] * 2

        raw_path = next(path for path in tmp_path.iterdir() if "raw_counts" in path.name)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert raw["schema"] == "scpn_max_width_kuramoto_xy_raw_counts_v1"
        assert len(raw["width_layouts"]) == 1
        assert raw["width_layouts"][0]["width"] == 5
        assert len(raw["width_layouts"][0]["chain_qubits"]) == 5
        assert len(raw["results"]) == 6
        assert {entry["width"] for entry in raw["skipped_widths"]} == {32, 64, 104}

        analysis_path = next(path for path in tmp_path.iterdir() if "analysis" in path.name)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert len(analysis["width_rows"]) == 2  # reps 1 and 2
        for row in analysis["width_rows"]:
            # Ideal fake counts + perfect calibration: hardware R equals the
            # exact MPS baseline within count-rounding noise.
            assert row["r_unmitigated"] == pytest.approx(row["r_exact_mps"], abs=2e-2)
            assert row["r_mitigated"] == pytest.approx(row["r_unmitigated"], abs=1e-9)
        assert "advantage claim" in analysis["claim_boundary"]

    def test_skip_baseline_defers_the_mps_run(
        self,
        mocked_hardware: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = script.main(
            ["--out-dir", str(tmp_path), "--submit", "--confirm-budget", "--skip-baseline"]
        )
        assert exit_code == 0
        analysis_path = next(path for path in tmp_path.iterdir() if "analysis" in path.name)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert all("r_exact_mps" not in row for row in analysis["width_rows"])
        assert "deferred" in analysis["baseline"]

    def test_parse_args_defaults(self) -> None:
        args = script._parse_args([])
        assert args.backend == script.DEFAULT_BACKEND
        assert args.shots == SHOTS
        assert args.max_qpu_seconds == 120.0
        assert args.median_edge_error_abort == pytest.approx(0.03)
        assert not args.submit


class TestModuleBootstrap:
    def test_reload_inserts_repo_root_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        stripped = [entry for entry in sys.path if entry != str(script.REPO_ROOT)]
        monkeypatch.setattr(sys, "path", stripped)
        reloaded = importlib.reload(script)
        assert sys.path[0] == str(reloaded.REPO_ROOT)
