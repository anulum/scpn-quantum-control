# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 core operating assumptions spec tests
"""Tests for Paper 0 core-operating-assumptions promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_core_operating_assumptions_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_core_operating_assumptions_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_core_operating_assumptions_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 core operating assumptions spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.3 The Logos > SCPN Core Operating Assumptions",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    text = {
        635: "The SCPN: Core Operating Assumptions",
        636: "v8.6, QEC, CISS, Quasicriticality, internal auditing",
        637: "five assumptions: consciousness, bidirectional causality, field realism, phase dynamics, ethics",
        640: "five non-negotiable foundational ideas",
        641: "Consciousness is fundamental",
        642: "Influence flows both directions",
        643: "fields are real measurable engineerable",
        644: "same synchronisation software across 15 layers",
        645: "universe is going somewhere ethical guiding system",
        649: "high-level specification for active inference",
        650: "consciousness is inference engine all the way down",
        651: "top-down predictions and bottom-up prediction errors",
        652: "physical instantiation of priors as field",
        653: "phase synchrony and desynchrony implement inference",
        654: "Ethical Functional is deep prior for coherence complexity qualia",
        656: "H_int = -lambda * Psi_s * sigma",
        657: "Psi_s is real physical field",
        658: "reciprocal top-down and bottom-up H_int causality",
        659: "sigma is phase coherence or synchrony, UPDE equation of motion",
        660: "Ethical Functionals tune lambda, do not add force to H_int",
        662: "15 layers, QEC, CISS, Quasicriticality validated by internal auditing",
        664: "Consciousness is ontological primitive",
        665: "Causality is bidirectional",
        666: "field is real",
        667: "UPDE mathematical spine",
        668: "Evolution is teleological with Layer 15 objective functions",
    }
    return [
        _record(
            f"P0R{number:05d}",
            number,
            "" if number in (646, 661, 669) else text.get(number, "source text"),
        )
        for number in range(635, 670)
    ]


def test_core_operating_assumptions_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_core_operating_assumptions_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 35
    assert bundle.summary["consumed_source_record_count"] == 35
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00635", "P0R00669"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["core_assumption_count"] == 5
    assert bundle.summary["blank_separator_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R00670"


def test_core_operating_assumptions_specs_preserve_hpc_and_hint_boundaries() -> None:
    module = _load_module()

    bundle = module.build_core_operating_assumptions_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["core_operating_assumptions.five_assumption_bedrock"].source_equation_ids == (
        "P0R00641:consciousness_fundamental",
        "P0R00642:bidirectional_causality",
        "P0R00643:field_realism",
        "P0R00644:unified_phase_dynamics",
        "P0R00645:ethical_functionals",
    )
    assert (
        "top-down predictions and bottom-up prediction errors"
        in specs["core_operating_assumptions.predictive_coding_mapping"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["core_operating_assumptions.hint_assumption_roles"].source_formulae
    )
    assert (
        "Ethical Functionals tune lambda and do not add a force term"
        in specs["core_operating_assumptions.lambda_ethical_tuning_boundary"].source_formulae
    )
    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(635, 670))
        for spec in specs.values()
    )


def test_core_operating_assumptions_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00656"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_core_operating_assumptions_specs(incomplete)


def test_core_operating_assumptions_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_core_operating_assumptions_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_core_operating_assumptions_validation_spec(
        "core_operating_assumptions.hint_assumption_roles",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00635"
    assert loaded["source_ledger_ids"][-1] == "P0R00669"
    assert "Paper 0 Core Operating Assumptions Specs" in report
    assert "active inference" in report
