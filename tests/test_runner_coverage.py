# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Runner Coverage Tests
"""Coverage tests for hardware.runner — simulator paths only."""

from scpn_quantum_control.hardware.runner import HardwareRunner


class TestRunnerInit:
    def test_simulator_init(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.use_simulator is True
        assert runner.results_dir.exists()

    def test_default_resilience_level(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.resilience_level == 2

    def test_fractional_gates_default(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.use_fractional_gates is True

    def test_optimization_level(self, tmp_path):
        runner = HardwareRunner(
            use_simulator=True,
            optimization_level=3,
            results_dir=str(tmp_path / "res"),
        )
        assert runner.optimization_level == 3

    def test_results_dir_fallback(self):
        # Non-writable path should fallback to tempdir
        runner = HardwareRunner(use_simulator=True, results_dir="/nonexistent/deep/path")
        assert runner.results_dir.exists()


class TestRunnerConnect:
    def test_connect_simulator(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert runner._backend is not None
