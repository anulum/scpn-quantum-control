# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II FIM solution builder tests
"""Tests for Paper 0 Axiom II FIM-solution spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_ii_fim_solution_validation_spec,
)
from scripts.build_paper0_axiom_ii_fim_solution_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_ii_fim_solution_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_fim_solution_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00775", "P0R00781"]
    assert bundle.summary["source_record_count"] == 7
    assert bundle.summary["consumed_source_record_count"] == 7
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["metric_definition_count"] == 1
    assert bundle.summary["physical_statement_count"] == 2
    assert bundle.summary["synthesis_statement_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R00782"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_ii_fim_solution.metric_definition",
        "axiom_ii_fim_solution.informational_interaction",
        "axiom_ii_fim_solution.complexity_coupling",
        "axiom_ii_fim_solution.fep_hpc_upde_synthesis",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_fim_solution_builder_keeps_source_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "natural unique Riemannian metric on a statistical manifold"
        in specs["axiom_ii_fim_solution.metric_definition"].source_formulae
    )
    assert (
        "distinguishability between nearby probability states"
        in specs["axiom_ii_fim_solution.metric_definition"].source_formulae
    )
    assert (
        "infoton propagates through the geometry of information itself"
        in specs["axiom_ii_fim_solution.informational_interaction"].source_formulae
    )
    assert (
        "coupling is proportional to informational complexity"
        in specs["axiom_ii_fim_solution.complexity_coupling"].source_formulae
    )
    assert (
        "brain at quasicriticality has large highly curved FIM"
        in specs["axiom_ii_fim_solution.complexity_coupling"].source_formulae
    )
    assert (
        "FEP is gradient descent on a manifold whose geometry is the FIM"
        in specs["axiom_ii_fim_solution.fep_hpc_upde_synthesis"].source_formulae
    )
    assert (
        "fundamental physics g_FIM algorithm HPC/FEP and dynamics UPDE share information geometry"
        in specs["axiom_ii_fim_solution.fep_hpc_upde_synthesis"].source_formulae
    )


def test_fim_solution_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00779":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom II > FIM Solution",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_ii_fim_solution_specs(records)


def test_fim_solution_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_ii_fim_solution_validation_spec(
        "axiom_ii_fim_solution.fep_hpc_upde_synthesis",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom II FIM Solution Specs" in report
    assert loaded["key"] == "axiom_ii_fim_solution.fep_hpc_upde_synthesis"
    assert (
        "FEP is gradient descent on a manifold whose geometry is the FIM"
        in loaded["source_formulae"]
    )
    assert "FIM Solution" in render_report(bundle)
