# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark harness public tests
# scpn-quantum-control -- public benchmark harness tests
"""Tests for the S5 public benchmark harness facade."""

from __future__ import annotations

from scpn_quantum_control.benchmark_harness import (
    available_baselines,
    load_phase1_dataset,
    reproduce_phase1_statistics,
    run_phase1_benchmark,
)


def test_load_phase1_dataset_public_facade() -> None:
    dataset = load_phase1_dataset()

    assert dataset.n_circuits_total == 342
    assert dataset.backends == frozenset({"ibm_kingston"})


def test_reproduce_phase1_statistics_public_facade() -> None:
    result = reproduce_phase1_statistics()

    assert result.n_circuits_used > 0
    assert result.fisher.degrees_of_freedom == 16
    assert result.peak_asymmetry_depth == 6
    assert result.claims_checked


def test_run_phase1_benchmark_includes_classical_reference() -> None:
    result = run_phase1_benchmark(baselines_backend="numpy")

    assert result.classical_reference.backend == "numpy"
    assert result.classical_reference.is_zero_within_tolerance is True
    assert available_baselines()["numpy"] is True
