# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 foundational viability postulate spec tests
"""Tests for Paper 0 foundational viability and Psi-field postulate promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_foundational_viability_postulate_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_foundational_viability_postulate_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_foundational_viability_postulate_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 foundational viability postulate spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.2 Foundational Viability",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        464: "1.2 Foundational Viability: Internal Consistency & Postulates",
        465: "three core pillars: ontological postulate, derived physical interactions, multiscale architecture",
        466: "Psi-field primitive ontology, Hierarchical Field Monism, generative hypothesis, SSB",
        467: "UPDE multi-scale generalisation of Kuramoto, information-geometric lift, FIM coherence",
        468: "15-layer hierarchy, bidirectional causality, Renormalisation Group flows",
        470: "Pillar 1: The Core Material (The Psi-Field)",
        472: "complex scalar field",
        474: "force of information emerges from symmetries",
        476: "UPDE supercharged version of a famous model",
        477: "Quasicriticality edge of chaos",
        478: "MS-QEC energy shield 63 times stronger than thermal chaos",
        479: "15 layers integrated self-correcting torus",
        483: "U(1) gauge + FIM structure",
        487: "H_int = -lambda * Psi_s * sigma",
        489: "H_int interaction is not a postulate but derivation from U(1) gauge symmetry",
        490: "sigma_info informational geometry",
        491: "sigma_phys stable coherent protected collective state variable",
        496: "irreducible ontological primitive, Hierarchical Field Monism",
        497: "not dogmatic assertion, generative hypothesis, primitive ontology",
        500: "complex scalar field permeating spacetime",
        502: "spin-0 bosons, global U(1) phase symmetry",
        503: "Psi = |Psi| e^{i theta}, Psi-Higgs boson, phase component",
        504: "Derivation of Interactions from a U(1) Gauge Principle and FIM",
        505: "derivation of interactions from a gauge principle",
    }
    return [
        _record(f"P0R{number:05d}", number, source_text.get(number, "source text"))
        for number in range(464, 506)
    ]


def test_foundational_viability_postulate_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_foundational_viability_postulate_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 42
    assert bundle.summary["consumed_source_record_count"] == 42
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00464", "P0R00505"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["pillar_count"] == 3
    assert bundle.summary["physics_postulate_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R00506"


def test_foundational_viability_postulate_specs_preserve_math_physics_boundaries() -> None:
    module = _load_module()

    bundle = module.build_foundational_viability_postulate_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(464, 506))
        for spec in specs.values()
    )
    assert specs[
        "foundational_viability_postulate.psi_complex_scalar_field"
    ].source_equation_ids == (
        "P0R00500:psi_complex_scalar_field",
        "P0R00502:spin0_u1_phase_symmetry",
        "P0R00503:psi_magnitude_phase_decomposition",
    )
    assert (
        "Psi = |Psi| e^{i theta}"
        in specs["foundational_viability_postulate.psi_complex_scalar_field"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["foundational_viability_postulate.hint_component_viability"].source_formulae
    )
    assert (
        "UPDE multi-scale generalisation of the Kuramoto model"
        in specs["foundational_viability_postulate.dynamic_spine_viability"].source_formulae
    )


def test_foundational_viability_postulate_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00503"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_foundational_viability_postulate_specs(incomplete)


def test_foundational_viability_postulate_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_foundational_viability_postulate_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_foundational_viability_postulate_validation_spec(
        "foundational_viability_postulate.psi_complex_scalar_field",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00464"
    assert loaded["source_ledger_ids"][-1] == "P0R00505"
    assert "Paper 0 Foundational Viability Postulate Specs" in report
    assert "U(1) gauge + FIM" in report
