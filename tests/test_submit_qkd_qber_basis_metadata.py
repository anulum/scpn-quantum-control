# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the QBER basis-metadata re-run submitter
"""Tests for scripts/submit_qkd_qber_basis_metadata.py.

Circuit tests pin the matched-basis geometry against exact statevector
arithmetic (ideal |Φ+⟩ gives zero mismatch in both matched bases, and an
unmatched basis gives 1/2). Basis metadata, mismatch arithmetic, the March
naive-sift derivation (synthetic and against the real committed artefact),
the preregistered decision rule, mitigation-on/off analysis, readiness
gating, and the CLI (dry-run, blocked, and fully mocked submission paths)
are all exercised without any hardware access.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.mitigation.readout_matrix import computational_basis_labels
from scripts import submit_qkd_qber_basis_metadata as script
from scripts.recompute_chsh_bell_test import QUBIT_PAIRS

SHOTS = 4096
CAL_SHOTS = 8192
LABELS = computational_basis_labels(script.N_QUBITS)


def statevector_probabilities(circuit: QuantumCircuit) -> list[float]:
    bare = circuit.remove_final_measurements(inplace=False)
    return [float(p) for p in Statevector(bare).probabilities()]


def perfect_counts(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    probabilities = statevector_probabilities(circuit)
    counts: dict[str, int] = {}
    remaining = shots
    nonzero = [(label, p) for label, p in zip(LABELS, probabilities, strict=True) if p > 1e-12]
    for index, (label, probability) in enumerate(nonzero):
        value = remaining if index == len(nonzero) - 1 else round(shots * probability)
        counts[label] = value
        remaining -= value
    return counts


def perfect_calibration_counts() -> dict[str, dict[str, int]]:
    return {label: {label: CAL_SHOTS} for label in LABELS}


class TestCircuits:
    def test_two_matched_basis_circuits_with_names(self) -> None:
        circuits = script.matched_basis_circuits()
        assert [c.name for c in circuits] == ["qkd_qber_zz", "qkd_qber_xx"]

    @pytest.mark.parametrize("index", [0, 1])
    def test_matched_bases_give_zero_mismatch_ideally(self, index: int) -> None:
        circuit = script.matched_basis_circuits()[index]
        probabilities = statevector_probabilities(circuit)
        for pair in QUBIT_PAIRS:
            mismatch = script.quasi_probability_mismatch(probabilities, LABELS, pair)
            assert mismatch == pytest.approx(0.0, abs=1e-12)

    def test_unmatched_basis_gives_half_mismatch(self) -> None:
        circuit = QuantumCircuit(script.N_QUBITS)
        for pair in QUBIT_PAIRS:
            circuit.h(pair[0])
            circuit.cx(pair[0], pair[1])
        circuit.h(0)
        circuit.h(2)
        probabilities = [float(p) for p in Statevector(circuit).probabilities()]
        for pair in QUBIT_PAIRS:
            mismatch = script.quasi_probability_mismatch(probabilities, LABELS, pair)
            assert mismatch == pytest.approx(0.5, abs=1e-12)

    def test_calibration_circuits_prepare_their_labels(self) -> None:
        circuits = script.calibration_circuits()
        assert len(circuits) == 16
        for label, circuit in zip(LABELS, circuits, strict=True):
            assert circuit.name == f"qkd_qber_cal_{label}"
            probabilities = statevector_probabilities(circuit)
            assert probabilities[int(label, 2)] == pytest.approx(1.0)


class TestBasisMetadata:
    def test_metadata_records_bases_and_pairs(self) -> None:
        metadata = script.basis_metadata()
        assert [m["alice_basis"] for m in metadata] == ["Z", "X"]
        assert all(m["alice_basis"] == m["bob_basis"] for m in metadata)
        assert all(m["pairs"] == [list(p) for p in QUBIT_PAIRS] for m in metadata)

    def test_metadata_declares_no_per_shot_randomness(self) -> None:
        assert all(m["per_shot_random_basis"] is False for m in script.basis_metadata())


class TestMismatchArithmetic:
    def test_pair_mismatch_from_counts(self) -> None:
        counts = {"0000": 900, "0001": 50, "0010": 50}
        assert script.pair_mismatch_probability(counts, (0, 1)) == pytest.approx(0.1)
        assert script.pair_mismatch_probability(counts, (2, 3)) == pytest.approx(0.0)

    def test_pair_mismatch_rejects_empty_counts(self) -> None:
        with pytest.raises(ValueError, match="no shots"):
            script.pair_mismatch_probability({"0000": 0}, (0, 1))

    def test_quasi_probability_mismatch_keeps_negative_terms(self) -> None:
        probabilities = [0.0] * 16
        probabilities[int("0001", 2)] = 0.06
        probabilities[int("0010", 2)] = -0.01
        assert script.quasi_probability_mismatch(probabilities, LABELS, (0, 1)) == pytest.approx(
            0.05
        )

    def test_binomial_sigma_matches_direct_formula(self) -> None:
        assert script.binomial_sigma(0.05, 4096) == pytest.approx(math.sqrt(0.05 * 0.95 / 4096))

    def test_binomial_sigma_clamps_out_of_range_rates(self) -> None:
        assert script.binomial_sigma(-0.2, 100) == 0.0
        assert script.binomial_sigma(1.7, 100) == 0.0


class TestMarchNaiveSift:
    def make_artifact(self, tmp_path: Path) -> Path:
        payload = {
            "backend": "ibm_fez",
            "results": [
                {"pub_index": 0, "counts": {"0000": 3900, "0001": 100}},
                {"pub_index": 1, "counts": {"0000": 3800, "1000": 200}},
                {"pub_index": 2, "counts": {"0000": 4000}},
            ],
        }
        path = tmp_path / "qkd_qber_4q.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_synthetic_sift_rates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = self.make_artifact(tmp_path)
        monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
        sift = script.march_naive_sift(path)
        assert sift["assumed_pub_bases"] == ["Z", "X"]
        assert len(sift["entries"]) == 4
        by_key = {(e["pub_index"], tuple(e["pair"])): e for e in sift["entries"]}
        assert by_key[(0, (0, 1))]["mismatch_rate"] == pytest.approx(100 / 4000)
        assert by_key[(1, (0, 1))]["mismatch_rate"] == pytest.approx(0.0)
        assert by_key[(1, (2, 3))]["mismatch_rate"] == pytest.approx(200 / 4000)

    def test_real_committed_artifact_stays_in_naive_band(self) -> None:
        sift = script.march_naive_sift(script.REPO_ROOT / script.PARENT_ARTIFACT)
        rates = [entry["mismatch_rate"] for entry in sift["entries"]]
        assert all(0.0 <= rate <= 0.1 for rate in rates)
        assert max(rates) < 0.055
        assert sift["published_qber"] == {"zz": 0.055, "xx": 0.058}


class TestDecisionRule:
    def sift(self) -> dict[str, Any]:
        return {"entries": [{"mismatch_rate": 0.02}, {"mismatch_rate": 0.037}]}

    def measurement(self, rate: float) -> dict[str, Any]:
        return {"mitigated_mismatch_rate": rate, "sigma_binomial": 0.003}

    def test_rates_inside_band_support_overstatement_reading(self) -> None:
        outcome = script.decision_rule([self.measurement(0.03)], self.sift())
        assert outcome["all_mitigated_rates_within_2_sigma_of_band"] is True
        assert "overstate" in outcome["preregistered_reading"]

    def test_rate_above_band_flips_the_reading(self) -> None:
        outcome = script.decision_rule([self.measurement(0.055)], self.sift())
        assert outcome["all_mitigated_rates_within_2_sigma_of_band"] is False
        assert "outside" in outcome["preregistered_reading"]

    def test_band_edges_stretch_by_two_sigma(self) -> None:
        outcome = script.decision_rule([self.measurement(0.042)], self.sift())
        assert outcome["all_mitigated_rates_within_2_sigma_of_band"] is True


class TestBuildAnalysis:
    def test_perfect_counts_give_zero_rates(self, tmp_path: Path) -> None:
        circuits = script.matched_basis_circuits()
        basis_counts = [perfect_counts(circuit, SHOTS) for circuit in circuits]
        artifact = TestMarchNaiveSift().make_artifact(tmp_path)
        analysis = script.build_analysis(
            basis_counts,
            perfect_calibration_counts(),
            shots=SHOTS,
            march_sift=script.march_naive_sift(artifact),
        )
        assert analysis["schema"] == "scpn_qkd_qber_basis_metadata_analysis_v1"
        assert len(analysis["measurements"]) == 4
        for measurement in analysis["measurements"]:
            assert measurement["mismatch_rate"] == pytest.approx(0.0)
            assert measurement["mitigated_mismatch_rate"] == pytest.approx(0.0, abs=1e-9)
        assert analysis["decision_rule"]["all_mitigated_rates_within_2_sigma_of_band"] is True
        assert "no QKD" in analysis["claim_boundary"]

    def test_metadata_travels_into_the_analysis(self, tmp_path: Path) -> None:
        circuits = script.matched_basis_circuits()
        basis_counts = [perfect_counts(circuit, SHOTS) for circuit in circuits]
        artifact = TestMarchNaiveSift().make_artifact(tmp_path)
        analysis = script.build_analysis(
            basis_counts,
            perfect_calibration_counts(),
            shots=SHOTS,
            march_sift=script.march_naive_sift(artifact),
        )
        assert analysis["basis_metadata"] == script.basis_metadata()
        assert analysis["march_naive_sift"]["assumed_pub_bases"] == ["Z", "X"]


def make_args(**overrides: Any) -> Any:
    import argparse

    defaults = {
        "shots": SHOTS,
        "cal_shots": CAL_SHOTS,
        "max_qpu_seconds": script.DEFAULT_MAX_QPU_SECONDS,
        "max_depth": script.DEFAULT_MAX_DEPTH,
        "readout_error_abort": script.DEFAULT_READOUT_ERROR_ABORT,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_layout(readout: float = 0.01) -> Any:
    from scripts.submit_bell_rerun_mitigated import LayoutSelection

    return LayoutSelection(
        physical_qubits=(0, 1, 2, 3),
        edge_errors=(0.004, 0.005),
        readout_errors=(readout, readout, readout, readout),
    )


class TestReadiness:
    def test_ready_when_all_gates_pass(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=make_layout(),
            isa_depths=[12, 14, 3, 3],
            args=make_args(),
        )
        assert readiness["status"] == "ready_for_submission"
        assert readiness["n_circuits"] == 18
        assert readiness["estimated_qpu_seconds"] == pytest.approx(18 * 1.1)
        assert readiness["basis_metadata"] == script.basis_metadata()

    def test_blocked_on_readout_error(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=make_layout(readout=0.5),
            isa_depths=[12],
            args=make_args(),
        )
        assert readiness["status"] == "blocked"
        assert any("readout-error abort" in reason for reason in readiness["reasons"])

    def test_blocked_on_depth(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=make_layout(),
            isa_depths=[500],
            args=make_args(),
        )
        assert readiness["status"] == "blocked"

    def test_blocked_on_budget(self) -> None:
        readiness = script.build_readiness(
            backend_name="ibm_fez",
            layout=make_layout(),
            isa_depths=[12],
            args=make_args(max_qpu_seconds=1.0),
        )
        assert readiness["status"] == "blocked"

    def test_estimate_uses_the_empirical_anchor(self) -> None:
        assert script.estimate_qpu_seconds(18) == pytest.approx(19.8)


class TestWriteJson:
    def test_returns_sha256_of_file(self, tmp_path: Path) -> None:
        import hashlib

        path = tmp_path / "x.json"
        digest = script._write_json(path, {"k": 1})
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


class FakeRegister:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


class FakeData:
    def __init__(self, counts: dict[str, int]) -> None:
        self.meas = FakeRegister(counts)


class FakePubResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self.data = FakeData(counts)


class TestPubCounts:
    def test_extracts_from_meas_register(self) -> None:
        assert script._pub_counts(FakePubResult({"0000": 4})) == {"0000": 4}

    def test_fails_without_a_counts_register(self) -> None:
        class Empty:
            data = object()

        with pytest.raises(RuntimeError, match="no classical register"):
            script._pub_counts(Empty())


class FakeJob:
    def __init__(
        self,
        pubs: list[FakePubResult],
        *,
        metrics_error: bool = False,
        metrics_non_mapping: bool = False,
    ) -> None:
        self._pubs = pubs
        self._metrics_error = metrics_error
        self._metrics_non_mapping = metrics_non_mapping

    def job_id(self) -> str:
        return "testjobid0000000001"

    def result(self, timeout: float) -> list[FakePubResult]:
        return self._pubs

    def metrics(self) -> Any:
        if self._metrics_error:
            raise RuntimeError("metrics endpoint down")
        if self._metrics_non_mapping:
            return ["not-a-mapping"]
        return {"usage": {"seconds": 19.5}}


class TestMain:
    def prepare(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        metrics_error: bool = False,
        metrics_non_mapping: bool = False,
    ) -> dict[str, Any]:
        import types as _fake_types

        from scripts import prepare_s1_ibm_live_readiness as readiness_module

        monkeypatch.setattr(
            readiness_module,
            "load_authenticated_backend",
            lambda *a, **k: _fake_types.SimpleNamespace(target=None),
        )
        monkeypatch.setattr(script, "select_physical_qubits", lambda target: make_layout())

        def fake_transpile(circuits: Any, **kwargs: Any) -> Any:
            return circuits

        monkeypatch.setattr(script, "transpile", fake_transpile)

        mains = script.matched_basis_circuits()
        calibrations = script.calibration_circuits()
        pubs = [FakePubResult(perfect_counts(c, SHOTS)) for c in mains]
        pubs += [
            FakePubResult({label: CAL_SHOTS})
            for label in computational_basis_labels(script.N_QUBITS)
        ]
        assert len(pubs) == len(mains) + len(calibrations)

        captured: dict[str, Any] = {}

        class FakeSampler:
            def __init__(self, mode: Any) -> None:
                captured["mode"] = mode

            def run(self, submitted: list[Any]) -> FakeJob:
                captured["pubs"] = submitted
                return FakeJob(
                    pubs,
                    metrics_error=metrics_error,
                    metrics_non_mapping=metrics_non_mapping,
                )

        import sys as _sys
        import types as _types

        module = _types.ModuleType("qiskit_ibm_runtime")
        module.SamplerV2 = FakeSampler  # type: ignore[attr-defined]
        monkeypatch.setitem(_sys.modules, "qiskit_ibm_runtime", module)
        return captured

    def march_artifact(self, tmp_path: Path) -> Path:
        return TestMarchNaiveSift().make_artifact(tmp_path)

    def test_submit_requires_confirm_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert script.main(["--submit"]) == 2
        assert "requires --confirm-budget" in capsys.readouterr().err

    def test_dry_run_writes_readiness_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.prepare(tmp_path, monkeypatch)
        out_dir = tmp_path / "pack"
        code = script.main(
            ["--out-dir", str(out_dir), "--credentials-vault", str(tmp_path / "vault.md")]
        )
        assert code == 0
        readiness_files = list(out_dir.glob("qkd_qber_readiness_*.json"))
        assert len(readiness_files) == 1
        assert not list(out_dir.glob("qkd_qber_raw_counts_*.json"))
        out = capsys.readouterr().out
        assert "readiness=ready_for_submission" in out
        assert "hardware_submission=false" in out

    def test_blocked_readiness_returns_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.prepare(tmp_path, monkeypatch)
        code = script.main(
            [
                "--out-dir",
                str(tmp_path / "pack"),
                "--credentials-vault",
                str(tmp_path / "vault.md"),
                "--max-qpu-seconds",
                "1.0",
            ]
        )
        assert code == 3

    def run_submission(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        metrics_error: bool = False,
        metrics_non_mapping: bool = False,
    ) -> tuple[int, Path, dict[str, Any]]:
        captured = self.prepare(
            tmp_path,
            monkeypatch,
            metrics_error=metrics_error,
            metrics_non_mapping=metrics_non_mapping,
        )
        out_dir = tmp_path / "pack"
        code = script.main(
            [
                "--out-dir",
                str(out_dir),
                "--credentials-vault",
                str(tmp_path / "vault.md"),
                "--march-artifact",
                str(self.march_artifact(tmp_path)),
                "--submit",
                "--confirm-budget",
            ]
        )
        return code, out_dir, captured

    def test_full_submission_writes_hash_bound_packs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out_dir, captured = self.run_submission(tmp_path, monkeypatch)
        assert code == 0
        raw = json.loads(
            next(iter(out_dir.glob("qkd_qber_raw_counts_*.json"))).read_text(encoding="utf-8")
        )
        analysis = json.loads(
            next(iter(out_dir.glob("qkd_qber_analysis_*.json"))).read_text(encoding="utf-8")
        )
        assert raw["job_id"] == "testjobid0000000001"
        assert raw["basis_metadata"] == script.basis_metadata()
        assert raw["usage"] == {"seconds": 19.5}
        assert len(captured["pubs"]) == 18
        assert captured["pubs"][0][2] == SHOTS
        assert captured["pubs"][-1][2] == CAL_SHOTS
        for measurement in analysis["measurements"]:
            assert measurement["mismatch_rate"] == pytest.approx(0.0)
        assert analysis["raw_counts_sha256"]
        out = capsys.readouterr().out
        assert "job_id=testjobid0000000001" in out
        assert "pair q0q1 ZZ" in out

    def test_metrics_failure_is_recorded_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out_dir, _ = self.run_submission(tmp_path, monkeypatch, metrics_error=True)
        assert code == 0
        raw = json.loads(
            next(iter(out_dir.glob("qkd_qber_raw_counts_*.json"))).read_text(encoding="utf-8")
        )
        assert raw["usage"] == {"note": "job metrics unavailable at retrieval time"}

    def test_non_mapping_metrics_leave_usage_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out_dir, _ = self.run_submission(tmp_path, monkeypatch, metrics_non_mapping=True)
        assert code == 0
        raw = json.loads(
            next(iter(out_dir.glob("qkd_qber_raw_counts_*.json"))).read_text(encoding="utf-8")
        )
        assert raw["usage"] == {}
