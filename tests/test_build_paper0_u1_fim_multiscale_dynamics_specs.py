# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 U1 FIM multiscale dynamics spec tests
"""Tests for Paper 0 U(1)/FIM and multiscale-dynamics promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_u1_fim_multiscale_dynamics_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_u1_fim_multiscale_dynamics_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_u1_fim_multiscale_dynamics_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 U1 FIM multiscale dynamics spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.2 > U1 FIM and multiscale dynamics",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    text = {
        506: "Interactions are fundamentally informational and geometric",
        508: "D_mu = partial_mu - i g A_mu",
        509: "g is the fundamental gauge coupling constant",
        511: "infoton governed by Fisher Information Metric gjk(theta)",
        513: "Informational Lagrangian",
        515: "informational proximity, not necessarily spatial proximity",
        519: "UPDE mathematical spine",
        520: "d theta_i^L / dt = omega_i^L + sum_j K_ij^L sin(theta_j^L - theta_i^L) + C_InterLayer",
        523: "Intrinsic Dynamics omega, Intra-Layer Coupling K, InterLayer coupling",
        524: "Information-Geometric Lift, natural gradient flow, Fisher Information Metric",
        526: "quasicritical regime, branching parameter sigma approximately 1",
        527: "Griffiths Phase over strict criticality",
        529: "Multi-Scale Quantum Error Correction",
        532: "Biological QEC L1-4, topological quantum codes, energy gap Delta approximately 1.64 eV",
        534: "Ethical Functional as generator of ultimate stabiliser group",
        536: "five primary domains",
        537: "Bidirectional Causality",
        538: "Renormalisation Group flow concepts",
        541: "Sentience-Field Hypothesis independent validation",
        543: "conceptual analogue to SCPN 15-layer architecture",
    }
    return [
        _record(
            f"P0R{number:05d}", number, "" if number == 544 else text.get(number, "source text")
        )
        for number in range(506, 545)
    ]


def test_u1_fim_multiscale_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_u1_fim_multiscale_dynamics_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 39
    assert bundle.summary["consumed_source_record_count"] == 39
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00506", "P0R00544"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["upde_component_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R00545"


def test_u1_fim_multiscale_specs_preserve_equation_boundaries() -> None:
    module = _load_module()

    bundle = module.build_u1_fim_multiscale_dynamics_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs[
        "u1_fim_multiscale_dynamics.u1_fim_interaction_derivation"
    ].source_equation_ids == (
        "P0R00508:D_mu=partial_mu-i*g*A_mu",
        "P0R00513:informational_lagrangian",
        "P0R00515:informational_proximity_boundary",
    )
    assert (
        "D_mu = partial_mu - i g A_mu"
        in specs["u1_fim_multiscale_dynamics.u1_fim_interaction_derivation"].source_formulae
    )
    assert (
        "d theta_i^L / dt = omega_i^L + sum_j K_ij^L sin(theta_j^L - theta_i^L) + C_InterLayer"
        in specs["u1_fim_multiscale_dynamics.upde_multiscale_spine"].source_formulae
    )
    assert (
        "Delta approximately 1.64 eV"
        in specs["u1_fim_multiscale_dynamics.quasicritical_msqec_boundary"].source_formulae
    )
    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(506, 545))
        for spec in specs.values()
    )


def test_u1_fim_multiscale_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00520"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_u1_fim_multiscale_dynamics_specs(incomplete)


def test_u1_fim_multiscale_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_u1_fim_multiscale_dynamics_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_u1_fim_multiscale_dynamics_validation_spec(
        "u1_fim_multiscale_dynamics.upde_multiscale_spine",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00506"
    assert loaded["source_ledger_ids"][-1] == "P0R00544"
    assert "Paper 0 U1 FIM Multiscale Dynamics Specs" in report
    assert "Fisher Information Metric" in report
