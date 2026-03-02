import pytest

from scpn_quantum_control.hardware.runner import HardwareRunner


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


@pytest.fixture
def sim_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    return runner
