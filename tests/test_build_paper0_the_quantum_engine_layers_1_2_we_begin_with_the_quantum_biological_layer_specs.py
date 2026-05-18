# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. builder tests
"""Tests for Paper 0 The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05314", "P0R05322"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R05323"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer",
        "ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
        "the_quantum_classical_bridge_selection_and_amplification",
        "i_guided_einselection_the_emergence_of_classicality",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded the quantum engine layers 1 2 we begin with the quantum biological layer source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded the quantum engine layers 1 2 we begin with the quantum biological layer source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system."
        + " Specs"
        in report
    )
    assert "P0R05314 - P0R05322" in report
