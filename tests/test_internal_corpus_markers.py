# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- internal corpus marker tests
"""Tests for CI-safe classification of private-corpus test modules."""

from __future__ import annotations

from pathlib import Path

from _internal_corpus_markers import is_performance_gate, requires_internal_paper0_corpus


def test_paper0_source_extraction_tests_require_internal_corpus() -> None:
    """Paper 0 promotion tests are real but need ignored extraction artefacts."""
    assert requires_internal_paper0_corpus(
        Path("tests/test_build_paper0_lorentz_eft_resolution_specs.py")
    )
    assert requires_internal_paper0_corpus(
        Path("tests/test_paper0_lorentz_eft_resolution_validation.py")
    )
    assert requires_internal_paper0_corpus(
        Path("tests/test_run_paper0_lorentz_eft_resolution_fixture.py")
    )
    assert requires_internal_paper0_corpus(
        Path("tests/test_reconcile_paper0_validation_coverage.py")
    )


def test_public_tests_do_not_require_internal_corpus() -> None:
    """Public package tests stay in default CI and Docker runs."""
    assert not requires_internal_paper0_corpus(Path("tests/test_knm_hamiltonian.py"))
    assert not requires_internal_paper0_corpus(Path("tests/test_public_claim_boundaries.py"))


def test_wall_clock_performance_gates_are_explicitly_classified() -> None:
    """Machine-dependent timing gates are not default correctness tests."""
    assert is_performance_gate(Path("tests/test_perf_regression.py"))
    assert is_performance_gate(Path("tests/test_pipeline_wiring_performance.py"))
    assert not is_performance_gate(Path("tests/test_rust_path_benchmarks.py"))
