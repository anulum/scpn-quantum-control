# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Witness Tests
"""Tests for the BL-18 synchronisation-witness suite."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase.synchronisation_witness import (
    SYNC_WITNESS_CLAIM_BOUNDARY,
    SYNC_WITNESS_EVIDENCE_CLASS,
    SyncWitnessCase,
    SyncWitnessRecord,
    SyncWitnessSuiteResult,
    _as_persistence_pairs,
    _bootstrap_order_parameter_std,
    _cluster_phases,
    _dominant_persistence,
    _persistence_to_list,
    _run_case,
    betti_curve,
    default_sync_witness_cases,
    geodesic_phase_distance_matrix,
    harmonic_order_parameter,
    phase_cloud_synchronisation_witness,
    run_sync_witness_suite,
    sync_witness_boundary_rows,
    vietoris_rips_persistence,
)

PI = float(np.pi)


def _default_thresholds() -> NDArray[np.float64]:
    return np.linspace(0.05, PI, 24, dtype=np.float64)


def _valid_record() -> SyncWitnessRecord:
    return phase_cloud_synchronisation_witness(
        np.array([0.0, 0.0, 0.01, -0.01], dtype=np.float64),
        thresholds=_default_thresholds(),
        reference_scale=0.5,
        case_id="unit",
        regime="synchronised",
        min_order_parameter=0.99,
        max_order_parameter=1.0,
        expected_components=1,
        min_dominant_h1=0.0,
        max_dominant_h1=1.0e-6,
    )


# --------------------------------------------------------------------------- #
# harmonic_order_parameter
# --------------------------------------------------------------------------- #
def test_harmonic_order_parameter_synchronised_is_one() -> None:
    value = harmonic_order_parameter(np.zeros(5))
    assert value == pytest.approx(1.0)


def test_harmonic_order_parameter_uniform_is_zero() -> None:
    phases = np.linspace(0.0, 2.0 * PI, 8, endpoint=False)
    assert harmonic_order_parameter(phases) == pytest.approx(0.0, abs=1e-12)


def test_harmonic_order_parameter_second_harmonic_detects_antiphase() -> None:
    antiphase = np.array([0.0, PI])
    assert harmonic_order_parameter(antiphase, harmonic=1) == pytest.approx(0.0, abs=1e-12)
    assert harmonic_order_parameter(antiphase, harmonic=2) == pytest.approx(1.0)


def test_harmonic_order_parameter_rejects_nonpositive_harmonic() -> None:
    with pytest.raises(ValueError, match="harmonic must be a positive integer"):
        harmonic_order_parameter(np.zeros(3), harmonic=0)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros(1), np.array([0.0, np.nan])])
def test_harmonic_order_parameter_rejects_bad_phases(bad: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        harmonic_order_parameter(bad)


# --------------------------------------------------------------------------- #
# geodesic_phase_distance_matrix
# --------------------------------------------------------------------------- #
def test_geodesic_distance_known_values() -> None:
    matrix = geodesic_phase_distance_matrix(np.array([0.0, PI / 2.0, PI]))
    expected = np.array([[0.0, PI / 2.0, PI], [PI / 2.0, 0.0, PI / 2.0], [PI, PI / 2.0, 0.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_geodesic_distance_wraps_around_circle() -> None:
    matrix = geodesic_phase_distance_matrix(np.array([0.0, 3.0 * PI / 2.0]))
    assert matrix[0, 1] == pytest.approx(PI / 2.0)


def test_geodesic_distance_is_symmetric_zero_diagonal() -> None:
    matrix = geodesic_phase_distance_matrix(np.array([0.3, 1.1, 2.7, 4.9]))
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# vietoris_rips_persistence
# --------------------------------------------------------------------------- #
def test_persistence_square_has_expected_h1_loop() -> None:
    square = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    persistence = vietoris_rips_persistence(geodesic_phase_distance_matrix(square))
    h1 = persistence[1]
    lifetimes = h1[:, 1] - h1[:, 0]
    dominant = h1[np.argmax(lifetimes)]
    assert dominant[0] == pytest.approx(PI / 2.0)
    assert dominant[1] == pytest.approx(PI)


def test_persistence_h0_has_single_essential_class() -> None:
    square = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    h0 = vietoris_rips_persistence(geodesic_phase_distance_matrix(square))[0]
    essential = h0[~np.isfinite(h0[:, 1])]
    assert essential.shape[0] == 1
    assert essential[0, 0] == pytest.approx(0.0)


def test_persistence_equilateral_triangle_has_no_persistent_loop() -> None:
    triangle = np.deg2rad([0.0, 120.0, 240.0])
    h1 = vietoris_rips_persistence(geodesic_phase_distance_matrix(triangle))[1]
    assert _dominant_persistence(h1) == pytest.approx(0.0, abs=1e-12)


def test_persistence_max_dimension_zero_omits_h1() -> None:
    triangle = np.deg2rad([0.0, 120.0, 240.0])
    persistence = vietoris_rips_persistence(
        geodesic_phase_distance_matrix(triangle), max_dimension=0
    )
    assert set(persistence) == {0}


def test_persistence_rejects_bad_max_dimension() -> None:
    with pytest.raises(ValueError, match="max_dimension must be 0 or 1"):
        vietoris_rips_persistence(np.zeros((2, 2)), max_dimension=2)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3)),
        np.zeros((1, 1)),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
        np.zeros(3),
    ],
)
def test_persistence_rejects_bad_distance(bad: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        vietoris_rips_persistence(bad)


# --------------------------------------------------------------------------- #
# betti_curve
# --------------------------------------------------------------------------- #
def test_betti_curve_counts_live_classes() -> None:
    pairs = np.array([[0.0, 1.0], [0.5, np.inf]])
    thresholds = np.array([0.25, 0.75, 1.5])
    np.testing.assert_array_equal(betti_curve(pairs, thresholds), np.array([1, 2, 1]))


def test_betti_curve_empty_pairs_is_zero() -> None:
    np.testing.assert_array_equal(
        betti_curve(np.empty((0, 2)), np.array([0.1, 0.2])), np.array([0, 0])
    )


@pytest.mark.parametrize(
    "bad_thresholds",
    [
        np.array([]),
        np.array([[0.1, 0.2]]),
        np.array([0.2, 0.1]),
        np.array([-0.1, 0.2]),
        np.array([0.1, np.inf]),
    ],
)
def test_betti_curve_rejects_bad_thresholds(bad_thresholds: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        betti_curve(np.array([[0.0, 1.0]]), bad_thresholds)


def test_betti_curve_rejects_bad_pairs_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        betti_curve(np.array([[0.0, 1.0, 2.0]]), np.array([0.5]))


def test_betti_curve_rejects_negative_or_nan_pairs() -> None:
    with pytest.raises(ValueError):
        betti_curve(np.array([[-1.0, 1.0]]), np.array([0.5]))
    with pytest.raises(ValueError):
        betti_curve(np.array([[np.nan, 1.0]]), np.array([0.5]))


# --------------------------------------------------------------------------- #
# private helpers
# --------------------------------------------------------------------------- #
def test_dominant_persistence_branches() -> None:
    assert _dominant_persistence(np.empty((0, 2))) == 0.0
    assert _dominant_persistence(np.array([[0.5, np.inf]])) == 0.0
    assert _dominant_persistence(np.array([[0.5, 1.5], [0.0, 0.2]])) == pytest.approx(1.0)


def test_bootstrap_std_zero_paths() -> None:
    phases = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    assert _bootstrap_order_parameter_std(phases, noise_std=0.1, n_bootstrap=0, seed=1) == 0.0
    assert _bootstrap_order_parameter_std(phases, noise_std=0.0, n_bootstrap=8, seed=1) == 0.0


def test_bootstrap_std_is_deterministic_and_positive() -> None:
    phases = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    first = _bootstrap_order_parameter_std(phases, noise_std=0.1, n_bootstrap=16, seed=7)
    second = _bootstrap_order_parameter_std(phases, noise_std=0.1, n_bootstrap=16, seed=7)
    assert first == second
    assert first > 0.0


def test_cluster_phases_shape() -> None:
    phases = _cluster_phases((0.0, 1.0), spread=0.1, per_cluster=3)
    assert phases.shape == (6,)


def test_as_persistence_pairs_empty_returns_shape() -> None:
    assert _as_persistence_pairs("h", np.empty((0, 2))).shape == (0, 2)


def test_persistence_to_list_handles_infinite_death() -> None:
    listed = _persistence_to_list(np.array([[0.0, np.inf], [0.1, 0.5]]))
    assert listed[0][1] == float("inf")
    assert listed[1] == [0.1, 0.5]


def test_run_case_matches_direct_witness() -> None:
    case = default_sync_witness_cases()[0]
    record = _run_case(case)
    assert record.case_id == case.case_id
    assert record.passed


# --------------------------------------------------------------------------- #
# SyncWitnessCase
# --------------------------------------------------------------------------- #
def _valid_case() -> SyncWitnessCase:
    return default_sync_witness_cases()[0]


def test_case_to_dict_round_trips_fields() -> None:
    case = _valid_case()
    payload = case.to_dict()
    assert payload["case_id"] == case.case_id
    assert payload["regime"] == "synchronised"
    assert len(cast(list[float], payload["phases"])) == case.phases.size


@pytest.mark.parametrize(
    "overrides",
    [
        {"case_id": ""},
        {"regime": "unknown"},
        {"phases": np.zeros(1)},
        {"thresholds": np.array([0.2, 0.1])},
        {"reference_scale": 0.0},
        {"reference_scale": 100.0},
        {"reference_scale": float("nan")},
        {"reference_scale": np.array([1.0, 2.0])},
        {"noise_std": -1.0},
        {"n_bootstrap": -1},
        {"n_bootstrap": 1.5},
        {"seed": 1.5},
        {"min_order_parameter": 0.9, "max_order_parameter": 0.1},
        {"max_order_parameter": 1.5},
        {"expected_components": 0},
        {"min_dominant_h1": 1.0, "max_dominant_h1": 0.5},
    ],
)
def test_case_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    case = _valid_case()
    with pytest.raises(ValueError):
        dataclasses.replace(case, **overrides)


# --------------------------------------------------------------------------- #
# SyncWitnessRecord
# --------------------------------------------------------------------------- #
def test_record_to_dict_contains_curves_and_boundary() -> None:
    record = _valid_record()
    payload = record.to_dict()
    assert payload["claim_boundary"] == SYNC_WITNESS_CLAIM_BOUNDARY
    assert len(cast(list[int], payload["betti0_curve"])) == record.thresholds.size
    assert payload["persistent_component_count"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"case_id": ""},
        {"regime": "unknown"},
        {"n_nodes": 1},
        {"order_parameter": 2.0},
        {"order_parameter_harmonic2": -0.1},
        {"order_parameter_std": -1.0},
        {"persistent_component_count": 0},
        {"dominant_h1_persistence": -1.0},
        {"reference_scale": 0.0},
        {"n_bootstrap": -1},
        {"noise_std": -1.0},
        {"claim_boundary": ""},
    ],
)
def test_record_rejects_invalid_scalar_fields(overrides: dict[str, Any]) -> None:
    record = _valid_record()
    with pytest.raises(ValueError):
        dataclasses.replace(record, **overrides)


def test_record_rejects_mismatched_betti_length() -> None:
    record = _valid_record()
    with pytest.raises(ValueError, match="Betti curves must match"):
        dataclasses.replace(record, betti0_curve=np.array([1, 2], dtype=np.int64))


def test_record_rejects_negative_betti() -> None:
    record = _valid_record()
    bad = np.full(record.thresholds.size, -1, dtype=np.int64)
    with pytest.raises(ValueError, match="Betti curves must be non-negative"):
        dataclasses.replace(record, betti0_curve=bad)


def test_record_rejects_non_bool_passed() -> None:
    record = _valid_record()
    with pytest.raises(ValueError, match="passed must be a boolean"):
        dataclasses.replace(record, passed=cast(bool, 1))


@pytest.mark.parametrize(
    "bad_pairs",
    [
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[np.nan, 1.0]]),
        np.array([[-1.0, 1.0]]),
        np.array([[1.0, 0.5]]),
    ],
)
def test_record_rejects_invalid_persistence_pairs(bad_pairs: NDArray[np.float64]) -> None:
    record = _valid_record()
    with pytest.raises(ValueError):
        dataclasses.replace(record, h1_persistence=bad_pairs)


# --------------------------------------------------------------------------- #
# SyncWitnessBoundaryRow
# --------------------------------------------------------------------------- #
def test_boundary_row_to_dict() -> None:
    row = sync_witness_boundary_rows()[0]
    payload = row.to_dict()
    assert payload["status"] == "hard_gap"
    assert payload["claim_boundary"] == SYNC_WITNESS_CLAIM_BOUNDARY


@pytest.mark.parametrize(
    "overrides",
    [
        {"boundary_id": ""},
        {"status": "open"},
        {"reason": ""},
        {"claim_boundary": ""},
    ],
)
def test_boundary_row_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    row = sync_witness_boundary_rows()[0]
    with pytest.raises(ValueError):
        dataclasses.replace(row, **overrides)


# --------------------------------------------------------------------------- #
# SyncWitnessSuiteResult
# --------------------------------------------------------------------------- #
def test_suite_result_passed_and_lookup() -> None:
    result = run_sync_witness_suite()
    assert result.passed is True
    assert result.evidence_class == SYNC_WITNESS_EVIDENCE_CLASS
    assert len(result.records_for_regime("desynchronised")) == 1
    assert result.records_for_regime("clustered")[0].persistent_component_count == 3


def test_suite_result_to_dict_structure() -> None:
    payload = run_sync_witness_suite().to_dict()
    assert payload["passed"] is True
    assert len(cast(list[object], payload["records"])) == 3
    assert len(cast(list[object], payload["boundary_rows"])) == 2


def test_suite_result_reports_failure_when_a_record_fails() -> None:
    failing = phase_cloud_synchronisation_witness(
        np.zeros(4),
        thresholds=_default_thresholds(),
        reference_scale=0.5,
        expected_components=99,
    )
    result = SyncWitnessSuiteResult(records=(failing,), boundary_rows=sync_witness_boundary_rows())
    assert failing.passed is False
    assert result.passed is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"records": ()},
        {"boundary_rows": ()},
        {"evidence_class": ""},
        {"claim_boundary": ""},
    ],
)
def test_suite_result_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    result = run_sync_witness_suite()
    with pytest.raises(ValueError):
        dataclasses.replace(result, **overrides)


# --------------------------------------------------------------------------- #
# phase_cloud_synchronisation_witness + suite builders
# --------------------------------------------------------------------------- #
def test_witness_rejects_reference_scale_out_of_range() -> None:
    with pytest.raises(ValueError, match="reference_scale must lie within"):
        phase_cloud_synchronisation_witness(
            np.zeros(4), thresholds=_default_thresholds(), reference_scale=100.0
        )


def test_witness_desynchronised_regime_detects_loop() -> None:
    phases = np.linspace(0.0, 2.0 * PI, 8, endpoint=False)
    record = phase_cloud_synchronisation_witness(
        phases,
        thresholds=_default_thresholds(),
        reference_scale=0.5,
        regime="desynchronised",
        min_order_parameter=0.0,
        max_order_parameter=0.05,
        expected_components=8,
        min_dominant_h1=1.0,
        max_dominant_h1=PI,
    )
    assert record.passed
    assert record.dominant_h1_persistence > 1.0
    assert record.persistent_component_count == 8


def test_default_cases_and_suite_pass() -> None:
    cases = default_sync_witness_cases()
    assert len(cases) == 3
    assert run_sync_witness_suite(cases).passed


def test_run_suite_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="cases must be non-empty"):
        run_sync_witness_suite([])
