import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.runner import HardwareRunner


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


@pytest.fixture
def sim_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    return runner


@pytest.fixture
def knm_4q():
    return build_knm_paper27(L=4), OMEGA_N_16[:4]


@pytest.fixture
def knm_8q():
    return build_knm_paper27(L=8), OMEGA_N_16[:8]


@pytest.fixture
def rng():
    return np.random.default_rng(42)
