# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping analog mapping tests
"""Focused tests for the bounded analog-mapping analog mapping product."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.analog_mapping.platforms as platforms_module
from scpn_quantum_control.analog_mapping import (
    ANALOG_MAPPING_EVIDENCE_SCHEMA,
    ANALOG_MAPPING_SCHEMA,
    AnalogMappingEvidenceBundle,
    CalibrationEvaluation,
    FeasibilityDiagnostic,
    FeasibilityReport,
    MappingRequest,
    MappingResult,
    analog_mapping_markdown,
    assess_mapping_feasibility,
    build_analog_mapping_evidence,
    calibration_sensitivity,
    classify_topology,
    compare_analog_model_to_trotter,
    coupling_scale_objective,
    load_platform_profiles,
    platform_profile,
    reconstruct_compiled_couplings,
    write_analog_mapping_evidence,
)


class _CatalogueResource:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def joinpath(self, _name: str) -> _CatalogueResource:
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return json.dumps(self._payload)


def _install_catalogue_payload(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    resource = _CatalogueResource(payload)

    def fake_files(_package: object) -> _CatalogueResource:
        return resource

    monkeypatch.setattr(platforms_module, "files", fake_files)
    platforms_module.load_platform_profiles.cache_clear()


def _ring_request(*, tolerance: float = 1e-4) -> MappingRequest:
    couplings = np.array(
        [
            [0.0, 0.30, 0.0, -0.20],
            [0.30, 0.0, 0.25, 0.0],
            [0.0, 0.25, 0.0, 0.15],
            [-0.20, 0.0, 0.15, 0.0],
        ],
        dtype=np.float64,
    )
    detunings = np.array([0.10, -0.15, 0.20, -0.05], dtype=np.float64)
    return MappingRequest.from_arrays(
        couplings,
        detunings,
        topology="ring",
        measurement="phase_proxy",
        duration=0.2,
        coupling_scale=1.25,
        comparison_tolerance=tolerance,
    )


def test_static_catalogue_is_source_dated_and_non_promotional() -> None:
    """Keep all packaged profiles unique, sourced, and non-promotional."""
    profiles = load_platform_profiles()
    assert len(profiles) == 5
    assert len({profile.profile_id for profile in profiles}) == 5
    assert all(profile.source_url.startswith("https://") for profile in profiles)
    assert all(profile.verified_at_source_utc.endswith("Z") for profile in profiles)
    assert sum(profile.posture == "internal_compiler_model" for profile in profiles) == 1
    assert platform_profile("scpn_circuit_qed_design_v1").to_dict()["max_nodes"] == 64


def test_unknown_profile_fails_with_catalogue_context() -> None:
    """Report known catalogue identifiers for an unknown profile."""
    with pytest.raises(KeyError, match="known: scpn_circuit_qed_design_v1"):
        platform_profile("missing")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "unknown schema"),
        ({"schema": "wrong"}, "unknown schema"),
        (
            {"schema": platforms_module.PLATFORM_CATALOGUE_SCHEMA, "profiles": "invalid"},
            "must contain profiles",
        ),
        (
            {"schema": platforms_module.PLATFORM_CATALOGUE_SCHEMA, "profiles": []},
            "must contain profiles",
        ),
    ],
)
def test_catalogue_rejects_invalid_envelopes(
    monkeypatch: pytest.MonkeyPatch, payload: object, match: str
) -> None:
    """The public catalogue loader rejects malformed envelopes."""
    _install_catalogue_payload(monkeypatch, payload)
    try:
        with pytest.raises(ValueError, match=match):
            platforms_module.load_platform_profiles()
    finally:
        platforms_module.load_platform_profiles.cache_clear()


@pytest.mark.parametrize("row", [None, {}])
def test_catalogue_rejects_malformed_profile_rows(
    monkeypatch: pytest.MonkeyPatch, row: object
) -> None:
    """The public catalogue loader rejects non-object and incomplete rows."""
    payload = {"schema": platforms_module.PLATFORM_CATALOGUE_SCHEMA, "profiles": [row]}
    _install_catalogue_payload(monkeypatch, payload)
    try:
        with pytest.raises(ValueError, match="profile row"):
            platforms_module.load_platform_profiles()
    finally:
        platforms_module.load_platform_profiles.cache_clear()


def test_catalogue_rejects_duplicate_profile_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public catalogue loader requires stable profile identifiers."""
    row = load_platform_profiles()[0].to_dict()
    payload = {
        "schema": platforms_module.PLATFORM_CATALOGUE_SCHEMA,
        "profiles": [row, row],
    }
    _install_catalogue_payload(monkeypatch, payload)
    try:
        with pytest.raises(ValueError, match="profile ids must be unique"):
            platforms_module.load_platform_profiles()
    finally:
        platforms_module.load_platform_profiles.cache_clear()


def test_mapping_request_is_immutable_and_digest_is_deterministic() -> None:
    """Detach request arrays and keep their digest deterministic."""
    source = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    request = MappingRequest.from_arrays(
        source,
        np.array([0.1, -0.1]),
        topology="all_to_all",
        measurement="phase_proxy",
    )
    original_digest = request.digest
    source[0, 1] = 9.0
    assert request.couplings[0][1] == pytest.approx(0.5)
    assert request.digest == original_digest
    assert request.n_nodes == 2
    assert request.to_dict()["topology"] == "all_to_all"


@pytest.mark.parametrize(
    ("couplings", "detunings", "match"),
    [
        (np.zeros(2), np.zeros(2), "rank-2"),
        (np.zeros((2, 2)), np.zeros((2, 1)), "rank-1"),
        (np.zeros((2, 3)), np.zeros(2), "square"),
        (np.zeros((2, 2)), np.zeros(3), "detunings length"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros(2), "symmetric"),
        (np.eye(2), np.zeros(2), "diagonal"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), np.zeros(2), "finite"),
    ],
)
def test_mapping_request_rejects_invalid_arrays(
    couplings: np.ndarray[Any, Any],
    detunings: np.ndarray[Any, Any],
    match: str,
) -> None:
    """Reject malformed, asymmetric, or non-finite request arrays."""
    with pytest.raises(ValueError, match=match):
        MappingRequest.from_arrays(
            couplings,
            detunings,
            topology="sparse",
            measurement="phase_proxy",
        )


@pytest.mark.parametrize("field", ["duration", "coupling_scale", "comparison_tolerance"])
def test_mapping_request_rejects_non_positive_scales(field: str) -> None:
    """Reject non-positive duration, scale, and tolerance values."""
    request = _ring_request()
    with pytest.raises(ValueError, match=field):
        if field == "duration":
            replace(request, duration=0.0)
        elif field == "coupling_scale":
            replace(request, coupling_scale=0.0)
        else:
            replace(request, comparison_tolerance=0.0)


def test_topology_classifier_covers_ring_complete_and_sparse() -> None:
    """Classify ring, complete, sparse, and two-node graphs."""
    ring = _ring_request().coupling_matrix
    complete = np.ones((4, 4), dtype=np.float64) - np.eye(4)
    sparse = np.zeros((4, 4), dtype=np.float64)
    sparse[0, 2] = sparse[2, 0] = 0.3
    assert classify_topology(ring) == "ring"
    assert classify_topology(complete) == "all_to_all"
    assert classify_topology(sparse) == "sparse"
    assert classify_topology(np.array([[0.0, 1.0], [1.0, 0.0]])) == "all_to_all"


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((1, 1)), np.zeros((2, 3)), np.array([[0.0, 1.0], [0.0, 0.0]])],
)
def test_topology_classifier_rejects_invalid_matrices(matrix: np.ndarray[Any, Any]) -> None:
    """Reject invalid matrices before topology classification."""
    with pytest.raises(ValueError, match="topology classification"):
        classify_topology(matrix)


def test_internal_compiler_profile_admits_parameter_mapping_only() -> None:
    """Admit bounded internal compilation without hardware claims."""
    report = assess_mapping_feasibility(_ring_request(), "scpn_circuit_qed_design_v1")
    assert report.schema == ANALOG_MAPPING_SCHEMA
    assert report.supported is True
    assert report.mapping_result is not None
    assert report.mapping_result.n_nodes == 4
    assert report.mapping_result.n_couplers == 4
    assert report.mapping_result.reconstructed_coupling_rmse < 1e-12
    assert len(report.mapping_result.compiled_program_digest) == 64
    assert report.hardware_submission_allowed is False
    assert report.hardware_support_claim_allowed is False
    assert report.analog_advantage_claim_allowed is False
    assert report.to_dict()["mapping_result"]["compiler_platform"] == "circuit_qed"


def test_declared_topology_mismatch_fails_closed() -> None:
    """Block a request whose declared topology disagrees with its matrix."""
    request = replace(_ring_request(), topology="sparse")
    report = assess_mapping_feasibility(request, "scpn_circuit_qed_design_v1")
    assert report.supported is False
    assert report.mapping_result is None
    assert "declared_topology_mismatch" in {item.code for item in report.diagnostics}


def test_provider_sketch_fails_closed_on_control_measurement_and_posture() -> None:
    """Block a provider sketch across all unsupported request dimensions."""
    report = assess_mapping_feasibility(_ring_request(), "pulser_analogdevice_sketch_2026_07")
    codes = {item.code for item in report.diagnostics}
    assert report.supported is False
    assert {
        "signed_coupling_unsupported",
        "local_detuning_unsupported",
        "measurement_mismatch",
        "pairwise_control_unverified",
        "profile_not_executable_mapping_evidence",
    } <= codes


def test_custom_profile_range_capacity_and_ledger_diagnostics() -> None:
    """Report capacity, range, and ledger mismatches together."""
    base = platform_profile("scpn_circuit_qed_design_v1")
    constrained = replace(
        base,
        max_nodes=3,
        coupling_abs_min=0.19,
        coupling_abs_max=0.30,
        ledger_ref="docs/another_ledger.md",
    )
    report = assess_mapping_feasibility(_ring_request(), constrained)
    codes = {item.code for item in report.diagnostics}
    assert {
        "node_capacity_exceeded",
        "coupling_below_profile_range",
        "coupling_above_profile_range",
        "ledger_reference_mismatch",
    } <= codes


def test_unsupported_vendor_topology_is_diagnostic() -> None:
    """Expose an unsupported vendor topology as a diagnostic."""
    report = assess_mapping_feasibility(_ring_request(), "ionq_native_gate_sketch_2026_07")
    assert "unsupported_topology" in {item.code for item in report.diagnostics}


def test_reconstruct_compiled_couplings_recovers_sign_phase() -> None:
    """Recover signed symmetric couplings from compiler phases."""
    payload: dict[str, object] = {
        "coupling_terms": [
            {"source": 0, "target": 1, "strength": 0.5, "phase": 0.0},
            {"source": 1, "target": 2, "strength": 0.25, "phase": np.pi},
        ]
    }
    matrix = reconstruct_compiled_couplings(payload, 3)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[1, 2] == pytest.approx(-0.25)
    np.testing.assert_allclose(matrix, matrix.T)


@pytest.mark.parametrize(
    ("payload", "n_nodes", "match"),
    [
        ({}, 2, "missing coupling_terms"),
        ({"coupling_terms": [1]}, 2, "must be objects"),
        ({"coupling_terms": [{"source": "0"}]}, 2, "invalid fields"),
        (
            {"coupling_terms": [{"source": 0, "target": 2, "strength": 1.0, "phase": 0.0}]},
            2,
            "indices",
        ),
    ],
)
def test_reconstruct_compiled_couplings_rejects_malformed_payloads(
    payload: dict[str, object], n_nodes: int, match: str
) -> None:
    """Reject malformed compiled coupling payloads and invalid sizes."""
    with pytest.raises(ValueError, match=match):
        reconstruct_compiled_couplings(payload, n_nodes)
    with pytest.raises(ValueError, match="at least two"):
        reconstruct_compiled_couplings({"coupling_terms": []}, 1)


def test_bounded_model_comparison_reports_math_not_hardware() -> None:
    """Report bounded model numerics without hardware-equivalence claims."""
    comparison = compare_analog_model_to_trotter(_ring_request(tolerance=5e-3), trotter_steps=32)
    assert comparison.n_nodes == 4
    assert comparison.parameter_rmse < 1e-12
    assert comparison.compiler_model_state_fidelity == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= comparison.digital_trotter_infidelity < 1.0
    assert comparison.hardware_equivalence_claim_allowed is False
    assert comparison.analog_advantage_claim_allowed is False
    assert "not physical analog dynamics" in comparison.comparison_boundary
    assert comparison.to_dict()["trotter_steps"] == 32


def test_bounded_model_comparison_rejects_size_and_steps() -> None:
    """Reject oversized bounded comparisons and invalid Trotter steps."""
    request = MappingRequest.from_arrays(
        np.zeros((7, 7)),
        np.zeros(7),
        topology="sparse",
        measurement="phase_proxy",
    )
    with pytest.raises(ValueError, match="bounded"):
        compare_analog_model_to_trotter(request)
    with pytest.raises(ValueError, match="positive integer"):
        compare_analog_model_to_trotter(_ring_request(), trotter_steps=0)


def test_calibration_objective_gradient_matches_central_difference() -> None:
    """Match the analytic calibration gradient to central differences."""
    native = _ring_request().coupling_matrix
    target = 1.4 * native
    evaluation = coupling_scale_objective(native, target, scale=1.2)
    step = 1e-6
    plus = coupling_scale_objective(native, target, scale=1.2 + step)
    minus = coupling_scale_objective(native, target, scale=1.2 - step)
    finite_difference = (plus.loss - minus.loss) / (2.0 * step)
    assert evaluation.gradient == pytest.approx(finite_difference, rel=1e-8, abs=1e-10)
    assert evaluation.to_dict()["boundary"].startswith("Analytic design-unit")


def test_calibration_sensitivity_has_zero_nominal_and_symmetric_drift() -> None:
    """Keep nominal calibration exact and symmetric under scale drift."""
    native = _ring_request().coupling_matrix
    sensitivity = calibration_sensitivity(
        native,
        1.25 * native,
        nominal_scale=1.25,
        relative_drift=0.1,
    )
    assert sensitivity.nominal.loss == pytest.approx(0.0)
    assert sensitivity.minus_drift.loss == pytest.approx(sensitivity.plus_drift.loss)
    assert sensitivity.worst_case_loss == pytest.approx(sensitivity.plus_drift.loss)
    assert sensitivity.to_dict()["relative_drift"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("native", "target", "scale", "match"),
    [
        (np.zeros(2), np.zeros((2, 2)), 1.0, "square"),
        (np.zeros((2, 2)), np.zeros((3, 3)), 1.0, "match"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), np.zeros((2, 2)), 1.0, "finite"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((2, 2)), 1.0, "symmetric"),
        (np.zeros((2, 2)), np.zeros((2, 2)), np.inf, "scale"),
    ],
)
def test_calibration_objective_rejects_invalid_inputs(
    native: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    scale: float,
    match: str,
) -> None:
    """Reject invalid calibration arrays, scales, and drift bounds."""
    with pytest.raises(ValueError, match=match):
        coupling_scale_objective(native, target, scale=scale)
    with pytest.raises(ValueError, match="relative_drift"):
        calibration_sensitivity(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            nominal_scale=1.0,
            relative_drift=0.0,
        )


def test_evidence_bundle_is_deterministic_writable_and_renderable(tmp_path: Any) -> None:
    """Write and render deterministic supported-profile evidence."""
    bundle = build_analog_mapping_evidence(
        _ring_request(tolerance=5e-3),
        "scpn_circuit_qed_design_v1",
        trotter_steps=32,
    )
    assert bundle.schema == ANALOG_MAPPING_EVIDENCE_SCHEMA
    assert bundle.report.supported is True
    assert bundle.comparison is not None
    assert bundle.calibration is not None
    assert (
        bundle.digest
        == build_analog_mapping_evidence(
            _ring_request(tolerance=5e-3),
            "scpn_circuit_qed_design_v1",
            trotter_steps=32,
        ).digest
    )
    target = write_analog_mapping_evidence(tmp_path / "nested" / "evidence.json", bundle)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["digest"] == bundle.digest
    markdown = analog_mapping_markdown(bundle)
    assert "Hardware submission allowed: `False`" in markdown
    assert "Bounded Model Comparison" in markdown


def test_blocked_profile_evidence_omits_comparison_and_calibration() -> None:
    """Omit numerical evidence when the selected profile is blocked."""
    bundle = build_analog_mapping_evidence(
        _ring_request(),
        "pulser_analogdevice_sketch_2026_07",
    )
    assert bundle.report.supported is False
    assert bundle.comparison is None
    assert bundle.calibration is None
    assert "Bounded Model Comparison" not in analog_mapping_markdown(bundle)


def test_report_and_bundle_invariants_refuse_promotion() -> None:
    """Reject report or bundle states that imply unsupported promotion."""
    diagnostic = FeasibilityDiagnostic("blocked", "blocker", "blocked")
    with pytest.raises(ValueError, match="supported"):
        FeasibilityReport(
            schema=ANALOG_MAPPING_SCHEMA,
            request_digest="x",
            profile_id="x",
            observed_topology="sparse",
            supported=True,
            diagnostics=(diagnostic,),
            mapping_result=None,
            source_url="https://example.test",
            verified_at_source_utc="2026-07-25T00:00:00Z",
        )
    mapping = MappingResult("model", 2, 1, 0.0, "digest", ("bounded",))
    report = FeasibilityReport(
        schema=ANALOG_MAPPING_SCHEMA,
        request_digest="x",
        profile_id="x",
        observed_topology="sparse",
        supported=True,
        diagnostics=(),
        mapping_result=mapping,
        source_url="https://example.test",
        verified_at_source_utc="2026-07-25T00:00:00Z",
    )
    with pytest.raises(
        ValueError,
        match="analog-mapping reports must keep hardware and advantage claims blocked",
    ):
        replace(report, hardware_support_claim_allowed=True)
    with pytest.raises(ValueError, match="match report support"):
        AnalogMappingEvidenceBundle(
            schema=ANALOG_MAPPING_EVIDENCE_SCHEMA,
            request=_ring_request(),
            report=report,
            comparison=None,
            calibration=None,
            profile_ledger_ref="docs/qpu_provider_readiness.md",
        )
    bundle = build_analog_mapping_evidence(
        _ring_request(tolerance=5e-3),
        "scpn_circuit_qed_design_v1",
    )
    with pytest.raises(
        ValueError,
        match="analog-mapping evidence must remain local and non-promotional",
    ):
        replace(bundle, no_provider_contact=False)
    evaluation = CalibrationEvaluation(scale=1.0, loss=0.0, gradient=0.0)
    assert evaluation.to_dict()["scale"] == pytest.approx(1.0)
