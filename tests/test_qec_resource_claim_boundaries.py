# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QEC decoder evidence, code scaffold and resource claim boundaries
"""Decoder samples, code structure, operation meaning and future resources stay separate."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.fault_tolerant_resource_product import (
    ADMITTED_TORIC_DISTANCES,
    EXCLUDED_FROM_BASE_ESTIMATE,
    LOGICAL_FAILURE_CONFIDENCE_METHOD,
    LOGICAL_OPERATION_SEMANTICS,
    QEC_CLAIM_QUALIFICATION_SCHEMA,
    DecoderExperiment,
    FaultTolerantResourceBoundaryError,
    OverheadAssumptions,
    OverheadScenarioEstimate,
    QecClaimQualification,
    SyncProblemResourceRequest,
    build_fault_tolerant_resource_product,
    build_qec_claim_qualification,
    enumerate_single_error_syndromes,
    estimate_ft_resources,
    estimate_overhead_scenario,
    logical_failure_confidence,
    run_toric_decoder_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "data/phase3_multicircuit_qec/qec_readiness_2026-05-07.json"


def _request(**overrides: Any) -> SyncProblemResourceRequest:
    values: dict[str, Any] = {
        "n_oscillators": 4,
        "evolution_time": 1.0,
        "target_precision": 0.01,
        "coupling_density": 0.5,
        "trotter_steps": 8,
    }
    values.update(overrides)
    return SyncProblemResourceRequest(**values)


def _scenario(**overrides: Any) -> OverheadAssumptions:
    values: dict[str, Any] = {
        "label": "compact",
        "routing_overhead_factor": 1.5,
        "factory_count": 2,
        "factory_physical_qubits": 1000,
        "factory_cycles_per_t_state": 10,
        "decoder_latency_seconds_per_round": 1e-6,
    }
    values.update(overrides)
    return OverheadAssumptions(**values)


def _toric_edge_endpoints(qubit: int, distance: int) -> tuple[int, ...]:
    """Return the two vertices an edge touches, from lattice geometry alone."""
    cell, vertical = divmod(qubit, 2)
    row, column = divmod(cell, distance)
    if vertical:
        neighbour = ((row + 1) % distance) * distance + column
    else:
        neighbour = row * distance + (column + 1) % distance
    return tuple(sorted({cell, neighbour}))


def _exact_coverage(trials: int, probability: float, interval: Any) -> float:
    """Sum the binomial mass of every outcome whose interval contains the truth."""
    total = 0.0
    for failures in range(trials + 1):
        low, high = interval(failures, trials)
        if low <= probability <= high:
            total += (
                math.comb(trials, failures)
                * probability**failures
                * (1.0 - probability) ** (trials - failures)
            )
    return total


def _exact_interval(failures: int, trials: int) -> tuple[float, float]:
    statement = logical_failure_confidence(failures, trials, sample_model="binomial oracle")
    return statement.lower, statement.upper


def _wald_interval(failures: int, trials: int) -> tuple[float, float]:
    rate = failures / trials
    half = 1.96 * math.sqrt(rate * (1.0 - rate) / trials)
    return max(0.0, rate - half), min(1.0, rate + half)


def test_qec_resource_claim_boundaries_01() -> None:
    """Every single data-qubit flip fires exactly its two vertices and is corrected."""
    for distance in ADMITTED_TORIC_DISTANCES:
        rows = enumerate_single_error_syndromes(distance)
        assert [row.data_qubit for row in rows] == list(range(2 * distance**2))
        for row in rows:
            assert row.defects == _toric_edge_endpoints(row.data_qubit, distance)
            assert row.corrected is True
        assert len({row.defects for row in rows}) == len(rows)
    for refused in (1, 4, 7, True):
        with pytest.raises(FaultTolerantResourceBoundaryError, match="toric distance"):
            enumerate_single_error_syndromes(refused)


def test_qec_resource_claim_boundaries_02() -> None:
    """Zero sampled failures still bound the failure probability above zero."""
    experiment = run_toric_decoder_experiment(3, 0.001, 300, seed=7)
    confidence = experiment.confidence
    assert confidence.failures == 0 and confidence.rate == 0.0
    assert confidence.method == LOGICAL_FAILURE_CONFIDENCE_METHOD
    assert confidence.lower == 0.0
    assert confidence.upper == pytest.approx(1.0 - 0.025 ** (1.0 / 300), rel=1e-12)
    assert confidence.upper > 0.0
    assert "d=3" in confidence.sample_model and "seed=7" in confidence.sample_model
    assert "p=0.001" in confidence.sample_model
    certain = logical_failure_confidence(40, 40, sample_model="all failed")
    assert certain.upper == 1.0
    assert certain.lower == pytest.approx(0.025 ** (1.0 / 40), rel=1e-12)
    for probability in (0.01, 0.05, 0.1, 0.3, 0.5):
        assert _exact_coverage(20, probability, _exact_interval) >= 0.95
    assert _exact_coverage(20, 0.01, _wald_interval) < 0.95
    committed = json.loads(READINESS.read_text(encoding="utf-8"))
    zero_rows = [row for row in committed["decoder_aggregates"] if row["failures"] == 0]
    assert zero_rows and all(row["logical_failure_ci_high"] == 0.0 for row in zero_rows)
    for row in zero_rows:
        bound = logical_failure_confidence(
            row["failures"], row["trials"], sample_model=row["noise_model"]
        )
        assert bound.upper == pytest.approx(1.0 - 0.025 ** (1.0 / row["trials"]), rel=1e-12)


def test_decoder_experiment_is_seeded_and_counts_real_failures() -> None:
    """The same seed repeats the sample; a noisy sample records its failures honestly."""
    first = run_toric_decoder_experiment(3, 0.05, 400, seed=11)
    assert run_toric_decoder_experiment(3, 0.05, 400, seed=11) == first
    assert first.confidence.failures > 0
    assert first.confidence.lower < first.confidence.rate < first.confidence.upper
    assert first.measured_logical_error_reduction_claim_allowed is False
    record = first.to_dict()
    assert json.loads(json.dumps(record))["confidence"]["failures"] == first.confidence.failures
    with pytest.raises(ValueError, match="measured error reduction"):
        dataclasses.replace(first, measured_logical_error_reduction_claim_allowed=True)
    with pytest.raises(ValueError, match="exactly the sampled shots"):
        dataclasses.replace(first, shots=401)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((3, 1.0, 10, 0), "physical_error_rate"),
        ((3, float("nan"), 10, 0), "physical_error_rate"),
        ((3, True, 10, 0), "physical_error_rate"),
        ((3, 0.1, 0, 0), "shots must be a positive integer"),
        ((3, 0.1, True, 0), "shots must be a positive integer"),
        ((3, 0.1, 10, -1), "seed must be a non-negative integer"),
    ],
)
def test_malformed_decoder_experiments_are_refused(
    arguments: tuple[Any, ...], message: str
) -> None:
    """A malformed experiment is refused before any shot is sampled."""
    with pytest.raises(ValueError, match=message):
        run_toric_decoder_experiment(*arguments)
    with pytest.raises(FaultTolerantResourceBoundaryError):
        run_toric_decoder_experiment(4, 0.1, 10, 0)


@pytest.mark.parametrize(
    ("failures", "trials", "level", "model", "message"),
    [
        (1.0, 10, 0.95, "m", "failures must be an integer"),
        (1, True, 0.95, "m", "trials must be an integer"),
        (0, 0, 0.95, "m", "counts must satisfy"),
        (11, 10, 0.95, "m", "counts must satisfy"),
        (-1, 10, 0.95, "m", "counts must satisfy"),
        (1, 10, 1.0, "m", "confidence_level"),
        (1, 10, True, "m", "confidence_level"),
        (1, 10, 0.95, " ", "sample_model"),
    ],
)
def test_malformed_confidence_statements_are_refused(
    failures: Any, trials: Any, level: Any, model: str, message: str
) -> None:
    """A confidence statement needs real counts, a real level and a named sample."""
    with pytest.raises(ValueError, match=message):
        logical_failure_confidence(failures, trials, sample_model=model, confidence_level=level)


def test_qec_resource_claim_boundaries_03() -> None:
    """Changing factory or routing assumptions changes the resource result exactly."""
    request = _request()
    base = estimate_ft_resources(request)
    compact = estimate_overhead_scenario(request, _scenario())
    assert compact.data_patch_physical_qubits == base.surface_code_physical_qubits
    assert compact.routed_patch_physical_qubits == math.ceil(
        base.surface_code_physical_qubits * 1.5
    )
    assert compact.factory_physical_qubits == 2000
    assert compact.total_physical_qubits == compact.routed_patch_physical_qubits + 2000
    t_floor = math.ceil(base.total_t_count / 2) * 10 * request.syndrome_cycle_seconds
    assert compact.t_state_time_floor_seconds == pytest.approx(t_floor, rel=1e-15)
    assert compact.decoding_time_floor_seconds == pytest.approx(base.qec_rounds * 1e-6)
    assert compact.scenario_time_floor_seconds == pytest.approx(
        max(base.syndrome_time_floor_seconds, t_floor) + base.qec_rounds * 1e-6
    )
    more_factories = estimate_overhead_scenario(request, _scenario(factory_count=8))
    assert more_factories.t_state_time_floor_seconds < compact.t_state_time_floor_seconds
    assert more_factories.factory_physical_qubits == 8000
    wider_routing = estimate_overhead_scenario(request, _scenario(routing_overhead_factor=3.0))
    assert wider_routing.total_physical_qubits > compact.total_physical_qubits
    record = build_qec_claim_qualification(
        request,
        (_scenario(), _scenario(label="wide", routing_overhead_factor=3.0, factory_count=8)),
    )
    totals = [row.total_physical_qubits for row in record.scenarios]
    assert record.total_physical_qubit_range == (min(totals), max(totals))
    assert record.total_physical_qubit_range[0] < record.total_physical_qubit_range[1]
    floors = [row.scenario_time_floor_seconds for row in record.scenarios]
    assert record.scenario_time_floor_range_seconds == (min(floors), max(floors))
    assert record.excluded_from_base_estimate == EXCLUDED_FROM_BASE_ESTIMATE


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", " ", "label must be non-empty"),
        ("routing_overhead_factor", 0.9, "routing_overhead_factor"),
        ("routing_overhead_factor", float("inf"), "routing_overhead_factor"),
        ("routing_overhead_factor", True, "routing_overhead_factor"),
        ("factory_count", 0, "factory_count must be an integer"),
        ("factory_physical_qubits", 2.0, "factory_physical_qubits must be an integer"),
        ("factory_cycles_per_t_state", True, "factory_cycles_per_t_state"),
        ("decoder_latency_seconds_per_round", -1e-9, "decoder_latency"),
        ("decoder_latency_seconds_per_round", True, "decoder_latency"),
    ],
)
def test_malformed_overhead_assumptions_are_refused(
    field: str, value: object, message: str
) -> None:
    """Assumptions that would shrink or invent resources are refused."""
    with pytest.raises(ValueError, match=message):
        _scenario(**{field: value})


def test_qec_resource_claim_boundaries_04() -> None:
    """No estimate, scenario or qualification record can claim an available device."""
    request = _request()
    experiment = run_toric_decoder_experiment(3, 0.01, 50, seed=3)
    record = build_qec_claim_qualification(request, (_scenario(),), decoder_experiment=experiment)
    assert record.schema == QEC_CLAIM_QUALIFICATION_SCHEMA
    assert record.device_available_claim_allowed is False
    assert all(row.hardware_availability_claim_allowed is False for row in record.scenarios)
    assert all(row.total_runtime_claim_allowed is False for row in record.scenarios)
    assert record.logical_operations == LOGICAL_OPERATION_SEMANTICS
    assert {row.status for row in record.logical_operations} == {
        "counted_not_validated",
        "modelled_ideal",
        "not_modelled",
    }
    assert record.resource_payload_sha256 == (
        build_fault_tolerant_resource_product(request).payload_sha256
    )
    payload = record.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    experiment_record = payload["decoder_experiment"]
    assert isinstance(experiment_record, dict)
    assert experiment_record["measured_logical_error_reduction_claim_allowed"] is False
    assert record == build_qec_claim_qualification(
        request, (_scenario(),), decoder_experiment=experiment
    )
    assert (
        build_qec_claim_qualification(request, (_scenario(),)).to_dict()["decoder_experiment"]
        is None
    )
    with pytest.raises(ValueError, match="never establish device availability"):
        dataclasses.replace(record, device_available_claim_allowed=True)
    with pytest.raises(ValueError, match="availability and runtime claims blocked"):
        dataclasses.replace(record.scenarios[0], hardware_availability_claim_allowed=True)
    with pytest.raises(ValueError, match="availability and runtime claims blocked"):
        dataclasses.replace(record.scenarios[0], total_runtime_claim_allowed=True)
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        dataclasses.replace(record.scenarios[0], total_physical_qubits=1)
    with pytest.raises(ValueError, match="unknown qualification schema"):
        dataclasses.replace(record, schema="qec_claim_qualification.v0")
    with pytest.raises(ValueError, match="at least one overhead scenario"):
        dataclasses.replace(record, scenarios=())


def test_qualification_refuses_without_side_effects(tmp_path: Path) -> None:
    """Refused qualification requests change nothing and write nothing."""
    before = sorted(tmp_path.iterdir())
    readiness_bytes = READINESS.read_bytes()
    with pytest.raises(ValueError, match="at least one overhead scenario"):
        build_qec_claim_qualification(_request(), ())
    with pytest.raises(ValueError, match="labels must be unique"):
        build_qec_claim_qualification(_request(), (_scenario(), _scenario()))
    with pytest.raises(FaultTolerantResourceBoundaryError):
        build_qec_claim_qualification(_request(physical_error_rate=0.02), (_scenario(),))
    assert sorted(tmp_path.iterdir()) == before
    assert READINESS.read_bytes() == readiness_bytes


def test_scenario_rows_are_their_own_type() -> None:
    """Scenario rows carry the claim boundary and their caller-stated assumptions."""
    row = estimate_overhead_scenario(_request(), _scenario())
    assert isinstance(row, OverheadScenarioEstimate)
    assert row.assumptions == _scenario()
    assert "not a runtime" in row.claim_boundary
    assert isinstance(
        build_qec_claim_qualification(_request(), (_scenario(),)), QecClaimQualification
    )
    assert isinstance(run_toric_decoder_experiment(3, 0.0, 5, seed=0), DecoderExperiment)
    assert np.isfinite(row.scenario_time_floor_seconds)
