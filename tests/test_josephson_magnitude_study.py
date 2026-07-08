# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Josephson K_nm Magnitude Study
"""Tests for the QWC-5.2 Josephson K_nm magnitude-study design."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control import applications
from scpn_quantum_control.applications.josephson_array import JosephsonArrayParameters
from scpn_quantum_control.applications.josephson_magnitude_study import (
    DEFAULT_CANDIDATE_N,
    JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY,
    JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA,
    JosephsonKnmCandidate,
    JosephsonMagnitudeGate,
    JosephsonMagnitudeStudyDesign,
    build_josephson_knm_magnitude_study_design,
    render_josephson_knm_magnitude_study_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export_josephson_knm_magnitude_study.py"


def _load_export_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_export_josephson_knm_magnitude_study",
        EXPORT_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_script = _load_export_script()


def test_design_records_rounded_rho_candidate_without_magnitude_claim() -> None:
    """The QWC-5.2 manifest pins the rounded rho candidate and blocks promotion."""
    design = build_josephson_knm_magnitude_study_design()
    payload = design.to_dict()

    assert design.schema == JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA
    assert design.claim_boundary == JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY
    assert design.candidate.n_junctions == DEFAULT_CANDIDATE_N
    assert design.candidate.rounded_topology_correlation == pytest.approx(0.990)
    assert design.candidate.topology_correlation == pytest.approx(0.989947153717553)
    assert design.candidate.claim_status == "topology_candidate_magnitude_blocked"
    assert design.candidate.frequency_source == "canonical_OMEGA_N_16_prefix"
    assert design.extension_targets == (20, 30, 40)
    assert design.measured_calibration_available is False
    assert design.magnitude_claim_allowed is False
    assert design.hardware_submission_required is False
    candidate_payload = payload["candidate"]
    assert isinstance(candidate_payload, dict)
    assert candidate_payload["rounded_topology_correlation"] == pytest.approx(0.990)
    assert "No K_nm measured-magnitude validation from topology correlation alone." in (
        design.blocked_claims
    )
    assert any("hardware-device coupling-map" in claim for claim in design.blocked_claims)


def test_custom_parameters_keep_the_same_fail_closed_gate_shape() -> None:
    """Measured-parameter placeholders do not bypass the required evidence gates."""
    params = JosephsonArrayParameters(
        ej_ghz=18.0,
        ec_ghz=0.3,
        coupling_ghz=0.02,
        parameter_source="unit-test-calibration-placeholder",
    )
    design = build_josephson_knm_magnitude_study_design(n_junctions=12, parameters=params)

    assert design.candidate.n_junctions == 12
    assert design.candidate.parameter_source == "unit-test-calibration-placeholder"
    assert {gate.name for gate in design.gates} == {
        "calibrated_coupling_units",
        "locked_normalisation_and_uncertainty",
        "direct_magnitude_fit",
        "spectral_response",
        "null_models",
    }
    assert all(gate.required for gate in design.gates)
    assert design.required_calibration_fields[:3] == (
        "system_id",
        "device_or_array_source",
        "coupling_edges_0_indexed",
    )


def test_design_supports_larger_than_sixteen_node_targets() -> None:
    """N > 16 uses a Josephson frequency placeholder instead of OMEGA_N_16."""
    design = build_josephson_knm_magnitude_study_design(
        n_junctions=30,
        extension_targets=(30, 40, 50),
    )

    assert design.candidate.n_junctions == 30
    assert design.candidate.frequency_source == "uniform_josephson_charging_energy_placeholder"
    assert design.extension_targets == (30, 40, 50)
    assert design.magnitude_claim_allowed is False


def test_below_floor_candidate_keeps_non_promotional_status() -> None:
    """Weak topology matches stay below the candidate floor."""
    design = build_josephson_knm_magnitude_study_design(
        n_junctions=4,
        topology="linear",
        extension_targets=(20,),
    )

    assert design.candidate.topology_correlation < 0.98
    assert design.candidate.claim_status == "below_topology_candidate_floor"


def test_design_rejects_invalid_candidate_shapes() -> None:
    """Invalid node counts, topology strings, and target grids fail before export."""
    with pytest.raises(ValueError, match=">= 2"):
        build_josephson_knm_magnitude_study_design(n_junctions=1)
    with pytest.raises(ValueError, match="topology"):
        build_josephson_knm_magnitude_study_design(topology="")
    with pytest.raises(ValueError, match="Unknown topology"):
        build_josephson_knm_magnitude_study_design(topology="bogus")
    with pytest.raises(ValueError, match="extension_targets"):
        build_josephson_knm_magnitude_study_design(extension_targets=())
    with pytest.raises(ValueError, match="sorted"):
        build_josephson_knm_magnitude_study_design(extension_targets=(40, 30))
    with pytest.raises(ValueError, match="unique"):
        build_josephson_knm_magnitude_study_design(extension_targets=(30, 30))


def test_manifest_dataclasses_fail_closed_on_empty_fields() -> None:
    """Direct construction validates every public manifest row."""
    with pytest.raises(ValueError, match="topology"):
        JosephsonKnmCandidate(
            n_junctions=14,
            topology="",
            topology_correlation=0.99,
            rounded_topology_correlation=0.99,
            coupling_ratio=0.03,
            parameter_source="source",
            topology_source="source",
            ej_ec_ratio=60.0,
            is_transmon_regime=True,
            frequency_source="source",
            claim_status="candidate",
        )
    with pytest.raises(ValueError, match="coupling_ratio"):
        JosephsonKnmCandidate(
            n_junctions=14,
            topology="all_to_all",
            topology_correlation=0.99,
            rounded_topology_correlation=0.99,
            coupling_ratio=0.0,
            parameter_source="source",
            topology_source="source",
            ej_ec_ratio=60.0,
            is_transmon_regime=True,
            frequency_source="source",
            claim_status="candidate",
        )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        JosephsonKnmCandidate(
            n_junctions=14,
            topology="all_to_all",
            topology_correlation=1.01,
            rounded_topology_correlation=0.99,
            coupling_ratio=0.03,
            parameter_source="source",
            topology_source="source",
            ej_ec_ratio=60.0,
            is_transmon_regime=True,
            frequency_source="source",
            claim_status="candidate",
        )
    with pytest.raises(ValueError, match="evidence_required"):
        JosephsonMagnitudeGate(
            name="gate",
            required=True,
            current_status="blocked",
            evidence_required=(),
        )
    with pytest.raises(ValueError, match="gates"):
        JosephsonMagnitudeStudyDesign(
            schema=JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA,
            candidate=build_josephson_knm_magnitude_study_design().candidate,
            calibration_artifact_schema="schema",
            extension_targets=(20,),
            required_calibration_fields=("field",),
            gates=(),
            blocked_claims=("claim",),
            next_actions=("action",),
        )


def test_markdown_report_and_public_exports_are_wired() -> None:
    """The design renders and is exported from the applications package."""
    design = build_josephson_knm_magnitude_study_design()
    markdown = render_josephson_knm_magnitude_study_markdown(design)

    assert "# Josephson K_nm Magnitude Study" in markdown
    assert "scpn-bench knm-josephson-magnitude-study" in markdown
    assert "0.990" in markdown
    assert "No K_nm measured-magnitude validation" in markdown
    assert applications.JosephsonMagnitudeStudyDesign is JosephsonMagnitudeStudyDesign
    assert (
        applications.build_josephson_knm_magnitude_study_design
        is build_josephson_knm_magnitude_study_design
    )
    assert "build_josephson_knm_magnitude_study_design" in applications.__all__


def test_export_script_writes_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The QWC-5.2 artifacts regenerate through the export script."""
    out_dir = tmp_path / "data"
    doc_path = tmp_path / "josephson_knm_magnitude_study.md"
    monkeypatch.setattr(
        export_script,
        "parse_args",
        lambda: Namespace(out_dir=out_dir, doc_path=doc_path),
    )

    assert export_script.main() == 0
    json_files = list(out_dir.glob("josephson_knm_magnitude_study_*.json"))

    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["schema"] == JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA
    assert payload["magnitude_claim_allowed"] is False
    assert payload["candidate"]["rounded_topology_correlation"] == pytest.approx(0.990)
    assert "Josephson K_nm Magnitude Study" in doc_path.read_text(encoding="utf-8")
