# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Advantage Figure Reproducer
"""Reproduce the quantum-advantage crossover figure from committed data."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_quantum_advantage_crossover.py"


@pytest.fixture(scope="module")
def crossover_module():
    spec = importlib.util.spec_from_file_location("_plot_quantum_advantage_crossover", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hardware_points_cover_full_committed_ibm_fez_scaling(crossover_module):
    points = crossover_module.load_hardware_points()
    assert [p.n_qubits for p in points] == [4, 6, 8, 10, 12, 14, 16]
    assert {p.backend for p in points} == {"ibm_fez"}
    assert all(p.job_id.startswith("ibm-run-") for p in points)
    assert sum(p.shots for p in points) == 130_000
    assert points[-1].source_file == "results/hw_upde_16_snapshot.json"
    assert points[-1].depth == 770
    assert points[-1].qpu_budget_ms == 60_000.0


def test_classical_points_keep_exact_and_ode_boundaries(crossover_module):
    points = crossover_module.load_classical_points()
    by_n = {p.n_qubits: p for p in points}
    assert by_n[4].exact_diag_ms == 0.1
    assert by_n[12].exact_diag_ms == 26_812.1
    assert by_n[14].exact_diag_ms is None
    assert "OOM" in by_n[16].note
    assert by_n[16].ode_ms < 20.0


def test_exponential_fit_and_crossover_are_conservative(crossover_module):
    hardware = crossover_module.load_hardware_points()
    classical = crossover_module.load_classical_points()
    exact_fit, hardware_fit, crossover = crossover_module.build_crossover_model(
        hardware,
        classical,
    )
    assert exact_fit.slope > 0.5
    assert exact_fit.r_squared > 0.90
    assert hardware_fit.slope >= 0.0
    assert 10.0 <= crossover <= 16.0

    classical_at_16 = exact_fit.predict(np.array([16.0]))[0]
    hardware_at_16 = hardware_fit.predict(np.array([16.0]))[0]
    assert classical_at_16 > hardware_at_16


def test_plot_generation_writes_png_and_pdf(crossover_module, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    png_path, pdf_path, crossover = crossover_module.plot_quantum_advantage_crossover(
        output_dir=tmp_path,
        docs_output_dir=None,
    )

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 20_000
    assert pdf_path.stat().st_size > 5_000
    assert 10.0 <= crossover <= 16.0
