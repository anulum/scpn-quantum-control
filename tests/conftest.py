# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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
