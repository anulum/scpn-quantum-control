# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for HardwareRunner reproducibility hooks
"""Tests for the AUD-4 reproducibility hooks on HardwareRunner.

Covers the deterministic ``seed_transpiler`` (default + override, and that it
reaches the connected pass manager) and ``calibration_snapshot`` (the
not-connected guard, the no-properties simulator record, and the populated
snapshot from a backend that exposes median calibration).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult


class TestSeedTranspiler:
    def test_default_seed(self) -> None:
        assert HardwareRunner(use_simulator=True).seed_transpiler == 20260718

    def test_custom_seed_stored(self) -> None:
        assert HardwareRunner(use_simulator=True, seed_transpiler=42).seed_transpiler == 42

    def test_seed_reaches_pass_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}
        import scpn_quantum_control.hardware.runner as runner_mod

        real = runner_mod.generate_preset_pass_manager

        def _spy(*args: object, **kwargs: object):
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(runner_mod, "generate_preset_pass_manager", _spy)
        runner = HardwareRunner(use_simulator=True, seed_transpiler=777)
        runner.connect()
        assert captured.get("seed_transpiler") == 777


class _Props:
    def __init__(self, t1: float, t2: float, ro: float) -> None:
        self._t1, self._t2, self._ro = t1, t2, ro
        self.last_update_date = "2026-03-29"

    def t1(self, _q: int) -> float:
        return self._t1

    def t2(self, _q: int) -> float:
        return self._t2

    def readout_error(self, _q: int) -> float:
        return self._ro


class TestCalibrationSnapshot:
    def test_requires_connect(self) -> None:
        runner = HardwareRunner(use_simulator=True)
        with pytest.raises(RuntimeError, match="connect"):
            runner.calibration_snapshot()

    def test_simulator_reports_unavailable(self) -> None:
        runner = HardwareRunner(use_simulator=True)
        runner.connect()  # Aer/basic simulator: no calibration properties
        snap = runner.calibration_snapshot()
        assert snap["available"] is False
        assert snap["seed_transpiler"] == 20260718

    def test_populated_snapshot_from_backend_properties(self) -> None:
        runner = HardwareRunner(use_simulator=True, seed_transpiler=5)
        runner._backend = SimpleNamespace(
            name="ibm_fez",
            num_qubits=3,
            properties=lambda: _Props(t1=146.7e-6, t2=109.3e-6, ro=0.01508),
        )
        snap = runner.calibration_snapshot()
        assert snap["available"] is True
        assert snap["backend"] == "ibm_fez"
        assert snap["num_qubits"] == 3
        assert snap["t1_median_us"] == pytest.approx(146.7, abs=0.1)
        assert snap["t2_median_us"] == pytest.approx(109.3, abs=0.1)
        assert snap["readout_error_median"] == pytest.approx(0.01508)
        assert snap["last_update_date"] == "2026-03-29"

    def test_backend_without_properties_callable(self) -> None:
        runner = HardwareRunner(use_simulator=True)
        runner._backend = SimpleNamespace(name="stub", num_qubits=1, properties=lambda: None)
        assert runner.calibration_snapshot()["available"] is False


class TestSaveResultEmbedsCalibration:
    """AUD-4b: every saved pack carries a ``calibration`` snapshot block."""

    def test_single_result_carries_calibration(self, tmp_path) -> None:
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        path = runner.save_result(
            JobResult(job_id="j1", backend_name="aer", experiment_name="e", counts={"0": 1})
        )
        data = json.loads(path.read_text())
        assert "calibration" in data
        assert data["calibration"]["seed_transpiler"] == 20260718

    def test_list_result_carries_calibration(self, tmp_path) -> None:
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        path = runner.save_result(
            [JobResult(job_id="j1", backend_name="aer", experiment_name="e")],
            filename="batch.json",
        )
        data = json.loads(path.read_text())
        assert data["calibration"]["backend"] == runner.backend_name

    def test_calibration_fail_open_when_not_connected(self, tmp_path) -> None:
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        # No connect(): save must still succeed with a fail-open record.
        record = runner._calibration_for_pack()
        assert record == {"available": False, "reason": "not connected"}
