# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 axiomatic Ntilde spec tests
"""Tests for Paper 0 formal-axiom and Ntilde-invariant promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_axiomatic_ntilde_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_axiomatic_ntilde_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_axiomatic_ntilde_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 axiomatic Ntilde spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.3 The Logos > The Axiomatic System",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    text = {
        578: "The Axiomatic System (The Logos)",
        579: "minimal set of axioms govern the system and provide causal closure (L13)",
        581: "Axiom of Existence: Consciousness Psi is irreducible ontological primitive",
        582: "generative hypothesis, formal testable architecture, empirical entailments",
        584: "Axiom of Interaction: interactions are informational and geometric",
        586: "Axiom of Evolution: maximise Sustainable Ethical Coherence, teleological stance",
        587: "ultimate priors for the system evolution",
        588: "figure: three axioms, curvature, SEC selection among allowable histories",
        591: "Axiom III not philosophical preference, fundamental falsifiable physical law",
        592: "dimensionless measurable invariant links energy information and time",
        593: r"\\tilde{N}_{t} = P / (epsilon_b I_dot) = (E/t) / ((Delta F_rev / Delta I) I_dot)",
        595: "P = E/t actual power energy flux",
        596: "I_dot rate of reliably processed information bit/s",
        597: "epsilon_b = Delta F_rev / Delta I reversible free-energy cost per bit",
        599: r"\\tilde{N}_{t} -> 1",
        600: "Ntilde = 1 + delta_irr, irreversibility entropy production",
        601: "SEC is macroscopic state of Ntilde = 1 and minimizes delta_irr",
        603: "quasicritical control receives quantitative validation from Universal Coherence Invariant",
        605: r"\\tilde{N}_{t} = (E/t) / (epsilon_b I_dot) = (E/t) / ((k_B T ln 2) I_dot)",
        606: "transition from disordered to self-organized coherent states at Ntilde to 1",
        608: "defines Quasicriticality optimal quasicritical regime L16 controller",
        609: "defines Efficiency Edot_actual / Edot_rev = 1 + delta_irr and J_SEC",
    }
    return [
        _record(f"P0R{number:05d}", number, text.get(number, "source text"))
        for number in range(578, 610)
    ]


def test_axiomatic_ntilde_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_axiomatic_ntilde_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 32
    assert bundle.summary["consumed_source_record_count"] == 32
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00578", "P0R00609"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["axiom_count"] == 3
    assert bundle.summary["ntilde_formula_count"] == 5
    assert bundle.summary["next_source_boundary"] == "P0R00610"


def test_axiomatic_ntilde_specs_preserve_status_tension_and_formulas() -> None:
    module = _load_module()

    bundle = module.build_axiomatic_ntilde_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["axiomatic_ntilde.formal_axiom_system_boundary"].source_equation_ids == (
        "P0R00578:axiomatic_system_header",
        "P0R00579:logos_causal_closure",
    )
    assert (
        "Axiom III status tension: normative teleology plus proposed falsifiable invariant"
        in specs["axiomatic_ntilde.axiom_three_status_transition"].source_formulae
    )
    assert (
        "Ntilde = P / (epsilon_b * I_dot)"
        in specs["axiomatic_ntilde.ntilde_invariant_definition"].source_formulae
    )
    assert (
        "Ntilde = 1 + delta_irr"
        in specs["axiomatic_ntilde.unity_irreversibility_target"].source_formulae
    )
    assert (
        "E_dot_actual / E_dot_rev = 1 + delta_irr"
        in specs["axiomatic_ntilde.quasicritical_efficiency_target"].source_formulae
    )
    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(578, 610))
        for spec in specs.values()
    )


def test_axiomatic_ntilde_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00593"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_axiomatic_ntilde_specs(incomplete)


def test_axiomatic_ntilde_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_axiomatic_ntilde_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiomatic_ntilde_validation_spec(
        "axiomatic_ntilde.ntilde_invariant_definition",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00578"
    assert loaded["source_ledger_ids"][-1] == "P0R00609"
    assert "Paper 0 Axiomatic Ntilde Specs" in report
    assert "dimensionless measurable invariant" in report
