# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD alias contracts tests
# scpn-quantum-control -- Program AD alias contract tests
"""Tests for Program AD alias, lattice, and provenance result contracts."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import program_ad_alias_analysis as alias_analysis_module
from scpn_quantum_control.differentiable import (
    PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY,
    PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY,
    ProgramADAliasEdge,
    ProgramADAliasEffectAnalysis,
    ProgramADAliasSet,
    ProgramADControlPathAliasProvenance,
    ProgramADListAliasProvenance,
    ProgramADLoopCarriedStateProvenance,
    ProgramADRebindingAliasProvenance,
    ProgramADStaticAliasLatticeComponent,
    ProgramADStaticAliasLatticeReport,
    ProgramADUnknownAliasEdge,
    ProgramADViewAliasProvenance,
    WholeProgramUnsupportedSemanticDiagnostic,
    analyze_program_ad_alias_effects,
    program_ad_static_alias_lattice_report,
)

DOCSTRING_SECTION_TARGETS: tuple[tuple[str, object, tuple[str, ...]], ...] = (
    ("ProgramADAliasSet", ProgramADAliasSet, ("Parameters", "Raises")),
    ("ProgramADAliasSet.as_dict", ProgramADAliasSet.as_dict, ("Returns",)),
    ("ProgramADAliasEffectAnalysis", ProgramADAliasEffectAnalysis, ("Parameters", "Raises")),
    (
        "ProgramADAliasEffectAnalysis.as_dict",
        ProgramADAliasEffectAnalysis.as_dict,
        ("Returns",),
    ),
    (
        "ProgramADStaticAliasLatticeComponent",
        ProgramADStaticAliasLatticeComponent,
        ("Parameters", "Raises"),
    ),
    (
        "ProgramADStaticAliasLatticeComponent.as_dict",
        ProgramADStaticAliasLatticeComponent.as_dict,
        ("Returns",),
    ),
    ("ProgramADUnknownAliasEdge", ProgramADUnknownAliasEdge, ("Parameters", "Raises")),
    ("ProgramADUnknownAliasEdge.as_dict", ProgramADUnknownAliasEdge.as_dict, ("Returns",)),
    ("ProgramADViewAliasProvenance", ProgramADViewAliasProvenance, ("Parameters", "Raises")),
    (
        "ProgramADViewAliasProvenance.as_dict",
        ProgramADViewAliasProvenance.as_dict,
        ("Returns",),
    ),
    ("ProgramADListAliasProvenance", ProgramADListAliasProvenance, ("Parameters", "Raises")),
    (
        "ProgramADListAliasProvenance.as_dict",
        ProgramADListAliasProvenance.as_dict,
        ("Returns",),
    ),
    (
        "ProgramADLoopCarriedStateProvenance",
        ProgramADLoopCarriedStateProvenance,
        ("Parameters", "Raises"),
    ),
    (
        "ProgramADLoopCarriedStateProvenance.as_dict",
        ProgramADLoopCarriedStateProvenance.as_dict,
        ("Returns",),
    ),
    (
        "ProgramADControlPathAliasProvenance",
        ProgramADControlPathAliasProvenance,
        ("Parameters", "Raises"),
    ),
    (
        "ProgramADControlPathAliasProvenance.as_dict",
        ProgramADControlPathAliasProvenance.as_dict,
        ("Returns",),
    ),
    (
        "ProgramADRebindingAliasProvenance",
        ProgramADRebindingAliasProvenance,
        ("Parameters", "Raises"),
    ),
    (
        "ProgramADRebindingAliasProvenance.as_dict",
        ProgramADRebindingAliasProvenance.as_dict,
        ("Returns",),
    ),
    (
        "ProgramADStaticAliasLatticeReport",
        ProgramADStaticAliasLatticeReport,
        ("Parameters", "Raises"),
    ),
    (
        "ProgramADStaticAliasLatticeReport.as_dict",
        ProgramADStaticAliasLatticeReport.as_dict,
        ("Returns",),
    ),
    (
        "analyze_program_ad_alias_effects",
        analyze_program_ad_alias_effects,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "program_ad_static_alias_lattice_report",
        program_ad_static_alias_lattice_report,
        ("Parameters", "Returns", "Raises"),
    ),
)


def _alias_lattice_report(**overrides: object) -> ProgramADStaticAliasLatticeReport:
    values: dict[str, object] = {
        "components": (),
        "mutation_effects": (),
        "non_executed_phi_nodes": (),
        "non_executed_control_alias_edges": (),
        "unknown_alias_edge_kinds": (),
        "blocker_reasons": (),
        "complete": False,
        "claim_boundary": "static_alias_lattice_over_emitted_program_ad_ir",
    }
    values.update(overrides)
    return ProgramADStaticAliasLatticeReport(**cast(Any, values))


def test_program_ad_alias_public_docstrings_define_contract_sections() -> None:
    """Alias-analysis public exports should document construction and failure contracts."""
    for qualified_name, target, required_sections in DOCSTRING_SECTION_TARGETS:
        docstring = inspect.getdoc(target)
        assert docstring is not None, qualified_name
        for section in required_sections:
            assert f"{section}\n" in docstring, qualified_name


def test_program_ad_alias_analysis_exports_stable_facade_identities() -> None:
    """Program AD alias-analysis exports should stay stable across public surfaces."""
    assert PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY == (
        alias_analysis_module.PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY
    )
    assert PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY == (
        alias_analysis_module.PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY
    )
    assert ProgramADAliasSet is alias_analysis_module.ProgramADAliasSet
    assert ProgramADAliasEffectAnalysis is alias_analysis_module.ProgramADAliasEffectAnalysis
    assert (
        ProgramADControlPathAliasProvenance
        is alias_analysis_module.ProgramADControlPathAliasProvenance
    )
    assert (
        ProgramADStaticAliasLatticeComponent
        is alias_analysis_module.ProgramADStaticAliasLatticeComponent
    )
    assert (
        ProgramADStaticAliasLatticeReport
        is alias_analysis_module.ProgramADStaticAliasLatticeReport
    )
    assert ProgramADListAliasProvenance is alias_analysis_module.ProgramADListAliasProvenance
    assert (
        ProgramADLoopCarriedStateProvenance
        is alias_analysis_module.ProgramADLoopCarriedStateProvenance
    )
    assert (
        ProgramADRebindingAliasProvenance
        is alias_analysis_module.ProgramADRebindingAliasProvenance
    )
    assert ProgramADUnknownAliasEdge is alias_analysis_module.ProgramADUnknownAliasEdge
    assert ProgramADViewAliasProvenance is alias_analysis_module.ProgramADViewAliasProvenance
    assert (
        analyze_program_ad_alias_effects is alias_analysis_module.analyze_program_ad_alias_effects
    )
    assert (
        program_ad_static_alias_lattice_report
        is alias_analysis_module.program_ad_static_alias_lattice_report
    )
    assert scpn.ProgramADAliasSet is ProgramADAliasSet
    assert scpn.ProgramADAliasEffectAnalysis is ProgramADAliasEffectAnalysis
    assert scpn.ProgramADControlPathAliasProvenance is ProgramADControlPathAliasProvenance
    assert scpn.ProgramADStaticAliasLatticeComponent is ProgramADStaticAliasLatticeComponent
    assert scpn.ProgramADStaticAliasLatticeReport is ProgramADStaticAliasLatticeReport
    assert scpn.ProgramADListAliasProvenance is ProgramADListAliasProvenance
    assert scpn.ProgramADLoopCarriedStateProvenance is ProgramADLoopCarriedStateProvenance
    assert scpn.ProgramADRebindingAliasProvenance is ProgramADRebindingAliasProvenance
    assert scpn.ProgramADUnknownAliasEdge is ProgramADUnknownAliasEdge
    assert scpn.ProgramADViewAliasProvenance is ProgramADViewAliasProvenance
    assert scpn.analyze_program_ad_alias_effects is analyze_program_ad_alias_effects
    assert scpn.program_ad_static_alias_lattice_report is program_ad_static_alias_lattice_report


def test_program_ad_alias_analysis_validation_paths() -> None:
    """Program AD alias-analysis records should reject malformed metadata."""
    alias_set = ProgramADAliasSet(
        index=0,
        members=("a", "b"),
        versions=(0, 1),
        mutation_versions=(1,),
    )
    edge = ProgramADAliasEdge(source="a", target="b", kind="source_alias", version=0)
    component = ProgramADStaticAliasLatticeComponent(
        index=0,
        members=("a", "b"),
        edge_kinds=("source_alias",),
        versions=(0, 1),
        mutation_versions=(1,),
    )
    unsupported_diagnostic = WholeProgramUnsupportedSemanticDiagnostic(
        semantic="object_attribute",
        detail="object_attribute:captured",
        line_number=2,
        absolute_line_number=42,
        region_ids=("root",),
        bytecode_offsets=(8,),
    )

    assert alias_set.as_dict()["members"] == ["a", "b"]
    assert component.as_dict()["edge_kinds"] == ["source_alias"]
    unknown_edge = ProgramADUnknownAliasEdge(
        source="runtime:dynamic_object",
        target="%0",
        kind="runtime_unknown_alias",
        version=0,
    )
    view_provenance = ProgramADViewAliasProvenance(
        source="%array[1]",
        target="view:getitem:12[0]",
        operation="getitem",
        view_id=12,
        output_index=0,
        version=12,
    )
    list_provenance = ProgramADListAliasProvenance(
        source="list:scratch",
        target="name:alias",
        list_name="scratch",
        target_kind="local_name",
        version=4,
    )
    loop_provenance = ProgramADLoopCarriedStateProvenance(
        source="loop:carry:entry",
        target="loop:carry:backedge",
        state_name="carry",
        entry_label="entry",
        backedge_label="backedge",
        version=6,
    )
    control_provenance = ProgramADControlPathAliasProvenance(
        source="control:if:42:body",
        target="control:attr:scratch.value",
        branch_line=42,
        branch_arm="body",
        target_label="attr:scratch.value",
        version=7,
    )
    rebinding_provenance = ProgramADRebindingAliasProvenance(
        source="expr:42:scratch.left+2.0*scratch.right",
        target="name:combined",
        binding_kind="expression",
        source_name=None,
        expression_line=42,
        expression_label="scratch.left+2.0*scratch.right",
        target_name="combined",
        version=9,
    )
    assert unknown_edge.as_dict() == {
        "source": "runtime:dynamic_object",
        "target": "%0",
        "kind": "runtime_unknown_alias",
        "version": 0,
    }
    assert view_provenance.as_dict() == {
        "source": "%array[1]",
        "target": "view:getitem:12[0]",
        "operation": "getitem",
        "view_id": 12,
        "output_index": 0,
        "version": 12,
    }
    assert list_provenance.as_dict() == {
        "source": "list:scratch",
        "target": "name:alias",
        "list_name": "scratch",
        "target_kind": "local_name",
        "version": 4,
    }
    assert loop_provenance.as_dict() == {
        "source": "loop:carry:entry",
        "target": "loop:carry:backedge",
        "state_name": "carry",
        "entry_label": "entry",
        "backedge_label": "backedge",
        "version": 6,
    }
    assert control_provenance.as_dict() == {
        "source": "control:if:42:body",
        "target": "control:attr:scratch.value",
        "branch_line": 42,
        "branch_arm": "body",
        "target_label": "attr:scratch.value",
        "version": 7,
    }
    assert rebinding_provenance.as_dict() == {
        "source": "expr:42:scratch.left+2.0*scratch.right",
        "target": "name:combined",
        "binding_kind": "expression",
        "source_name": None,
        "expression_line": 42,
        "expression_label": "scratch.left+2.0*scratch.right",
        "target_name": "combined",
        "version": 9,
    }
    for control_kwargs, match in (
        (
            {
                "source": "",
                "target": "control:attr:scratch.value",
                "branch_line": 42,
                "branch_arm": "body",
                "target_label": "attr:scratch.value",
                "version": 7,
            },
            "source",
        ),
        (
            {
                "source": "control:while:42:body",
                "target": "control:attr:scratch.value",
                "branch_line": 42,
                "branch_arm": "body",
                "target_label": "attr:scratch.value",
                "version": 7,
            },
            "source",
        ),
        (
            {
                "source": "control:if:42:body",
                "target": "attr:scratch.value",
                "branch_line": 42,
                "branch_arm": "body",
                "target_label": "attr:scratch.value",
                "version": 7,
            },
            "target",
        ),
        (
            {
                "source": "control:if:42:body",
                "target": "control:attr:scratch.value",
                "branch_line": -1,
                "branch_arm": "body",
                "target_label": "attr:scratch.value",
                "version": 7,
            },
            "branch_line",
        ),
        (
            {
                "source": "control:if:42:then",
                "target": "control:attr:scratch.value",
                "branch_line": 42,
                "branch_arm": "then",
                "target_label": "attr:scratch.value",
                "version": 7,
            },
            "branch_arm",
        ),
        (
            {
                "source": "control:if:42:body",
                "target": "control:attr:scratch.value",
                "branch_line": 43,
                "branch_arm": "body",
                "target_label": "attr:scratch.value",
                "version": 7,
            },
            "source",
        ),
        (
            {
                "source": "control:if:42:body",
                "target": "control:attr:scratch.value",
                "branch_line": 42,
                "branch_arm": "body",
                "target_label": "name:other",
                "version": 7,
            },
            "target_label",
        ),
        (
            {
                "source": "control:if:42:body",
                "target": "control:attr:scratch.value",
                "branch_line": 42,
                "branch_arm": "body",
                "target_label": "attr:scratch.value",
                "version": -1,
            },
            "version",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADControlPathAliasProvenance(**cast(Any, control_kwargs))
    for list_kwargs, match in (
        (
            {
                "source": "",
                "target": "name:alias",
                "list_name": "scratch",
                "target_kind": "local_name",
                "version": 4,
            },
            "source",
        ),
        (
            {
                "source": "scratch",
                "target": "name:alias",
                "list_name": "scratch",
                "target_kind": "local_name",
                "version": 4,
            },
            "source",
        ),
        (
            {
                "source": "list:scratch",
                "target": "alias",
                "list_name": "scratch",
                "target_kind": "local_name",
                "version": 4,
            },
            "target",
        ),
        (
            {
                "source": "list:scratch",
                "target": "name:alias",
                "list_name": "",
                "target_kind": "local_name",
                "version": 4,
            },
            "list_name",
        ),
        (
            {
                "source": "list:scratch",
                "target": "name:alias",
                "list_name": "other",
                "target_kind": "local_name",
                "version": 4,
            },
            "list_name",
        ),
        (
            {
                "source": "list:scratch",
                "target": "name:alias",
                "list_name": "scratch",
                "target_kind": "dynamic",
                "version": 4,
            },
            "target_kind",
        ),
        (
            {
                "source": "list:scratch",
                "target": "name:alias",
                "list_name": "scratch",
                "target_kind": "local_name",
                "version": -1,
            },
            "version",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADListAliasProvenance(**cast(Any, list_kwargs))
    for loop_kwargs, match in (
        (
            {
                "source": "",
                "target": "loop:carry:backedge",
                "state_name": "carry",
                "entry_label": "entry",
                "backedge_label": "backedge",
                "version": 6,
            },
            "source",
        ),
        (
            {
                "source": "carry:entry",
                "target": "loop:carry:backedge",
                "state_name": "carry",
                "entry_label": "entry",
                "backedge_label": "backedge",
                "version": 6,
            },
            "source",
        ),
        (
            {
                "source": "loop:carry:entry",
                "target": "carry:backedge",
                "state_name": "carry",
                "entry_label": "entry",
                "backedge_label": "backedge",
                "version": 6,
            },
            "target",
        ),
        (
            {
                "source": "loop:carry:entry",
                "target": "loop:carry:backedge",
                "state_name": "",
                "entry_label": "entry",
                "backedge_label": "backedge",
                "version": 6,
            },
            "state_name",
        ),
        (
            {
                "source": "loop:carry:start",
                "target": "loop:carry:backedge",
                "state_name": "carry",
                "entry_label": "start",
                "backedge_label": "backedge",
                "version": 6,
            },
            "entry_label",
        ),
        (
            {
                "source": "loop:carry:entry",
                "target": "loop:carry:end",
                "state_name": "carry",
                "entry_label": "entry",
                "backedge_label": "end",
                "version": 6,
            },
            "backedge_label",
        ),
        (
            {
                "source": "loop:carry:entry",
                "target": "loop:other:backedge",
                "state_name": "carry",
                "entry_label": "entry",
                "backedge_label": "backedge",
                "version": 6,
            },
            "state_name",
        ),
        (
            {
                "source": "loop:carry:entry",
                "target": "loop:carry:backedge",
                "state_name": "carry",
                "entry_label": "entry",
                "backedge_label": "backedge",
                "version": -1,
            },
            "version",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADLoopCarriedStateProvenance(**cast(Any, loop_kwargs))
    for view_kwargs, match in (
        (
            {
                "source": "",
                "target": "view:getitem:12[0]",
                "operation": "getitem",
                "view_id": 12,
                "output_index": 0,
                "version": 12,
            },
            "source",
        ),
        (
            {
                "source": "%array[1]",
                "target": "bad:getitem:12[0]",
                "operation": "getitem",
                "view_id": 12,
                "output_index": 0,
                "version": 12,
            },
            "target",
        ),
        (
            {
                "source": "%array[1]",
                "target": "view:getitem:12[0]",
                "operation": "",
                "view_id": 12,
                "output_index": 0,
                "version": 12,
            },
            "operation",
        ),
        (
            {
                "source": "%array[1]",
                "target": "view:getitem:12[0]",
                "operation": "getitem",
                "view_id": -1,
                "output_index": 0,
                "version": 12,
            },
            "view_id",
        ),
        (
            {
                "source": "%array[1]",
                "target": "view:getitem:12[0]",
                "operation": "getitem",
                "view_id": 12,
                "output_index": -1,
                "version": 12,
            },
            "output_index",
        ),
        (
            {
                "source": "%array[1]",
                "target": "view:getitem:12[0]",
                "operation": "getitem",
                "view_id": 12,
                "output_index": 0,
                "version": -1,
            },
            "version",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADViewAliasProvenance(**cast(Any, view_kwargs))
    for rebinding_kwargs, match in (
        (
            {
                "source": "name:seed",
                "target": "name:rebound",
                "binding_kind": "dynamic",
                "source_name": "seed",
                "expression_line": None,
                "expression_label": None,
                "target_name": "rebound",
                "version": 0,
            },
            "binding_kind",
        ),
        (
            {
                "source": "seed",
                "target": "name:rebound",
                "binding_kind": "local",
                "source_name": "seed",
                "expression_line": None,
                "expression_label": None,
                "target_name": "rebound",
                "version": 0,
            },
            "source",
        ),
        (
            {
                "source": "expr:42:seed+values[1]",
                "target": "name:rebound",
                "binding_kind": "expression",
                "source_name": "seed",
                "expression_line": 42,
                "expression_label": "seed+values[1]",
                "target_name": "rebound",
                "version": 0,
            },
            "source_name",
        ),
        (
            {
                "source": "expr:42:seed+values[1]",
                "target": "name:rebound",
                "binding_kind": "expression",
                "source_name": None,
                "expression_line": 41,
                "expression_label": "seed+values[1]",
                "target_name": "rebound",
                "version": 0,
            },
            "source",
        ),
        (
            {
                "source": "name:seed",
                "target": "name:rebound",
                "binding_kind": "local",
                "source_name": "seed",
                "expression_line": None,
                "expression_label": None,
                "target_name": "other",
                "version": 0,
            },
            "target_name",
        ),
        (
            {
                "source": "name:seed",
                "target": "name:rebound",
                "binding_kind": "local",
                "source_name": "seed",
                "expression_line": None,
                "expression_label": None,
                "target_name": "rebound",
                "version": -1,
            },
            "version",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADRebindingAliasProvenance(**cast(Any, rebinding_kwargs))
    with pytest.raises(ValueError, match="source"):
        ProgramADUnknownAliasEdge(
            source="",
            target="%0",
            kind="runtime_unknown_alias",
            version=0,
        )
    with pytest.raises(ValueError, match="target"):
        ProgramADUnknownAliasEdge(
            source="runtime:dynamic_object",
            target="",
            kind="runtime_unknown_alias",
            version=0,
        )
    with pytest.raises(ValueError, match="kind"):
        ProgramADUnknownAliasEdge(
            source="runtime:dynamic_object",
            target="%0",
            kind="",
            version=0,
        )
    with pytest.raises(ValueError, match="unsupported"):
        ProgramADUnknownAliasEdge(
            source="a",
            target="b",
            kind="source_alias",
            version=0,
        )
    with pytest.raises(ValueError, match="version"):
        ProgramADUnknownAliasEdge(
            source="runtime:dynamic_object",
            target="%0",
            kind="runtime_unknown_alias",
            version=-1,
        )
    with pytest.raises(ValueError, match="index"):
        ProgramADAliasSet(index=-1, members=("a",), versions=(), mutation_versions=())
    with pytest.raises(ValueError, match="members"):
        ProgramADAliasSet(index=0, members=("",), versions=(), mutation_versions=())
    with pytest.raises(ValueError, match="sorted"):
        ProgramADAliasSet(index=0, members=("b", "a"), versions=(), mutation_versions=())
    with pytest.raises(ValueError, match="versions"):
        ProgramADAliasSet(index=0, members=("a",), versions=(-1,), mutation_versions=())
    with pytest.raises(ValueError, match="mutation_versions"):
        ProgramADAliasSet(index=0, members=("a",), versions=(), mutation_versions=(-1,))
    with pytest.raises(ValueError, match="alias_sets"):
        ProgramADAliasEffectAnalysis(
            alias_sets=cast(tuple[ProgramADAliasSet, ...], (object(),)),
            mutation_effects=(),
            alias_edges=(),
            unknown_aliasing=False,
            claim_boundary="metadata_only_no_general_alias_lattice",
        )
    with pytest.raises(ValueError, match="sorted"):
        ProgramADAliasEffectAnalysis(
            alias_sets=(alias_set,),
            mutation_effects=(1, 0),
            alias_edges=(),
            unknown_aliasing=False,
            claim_boundary="metadata_only_no_general_alias_lattice",
        )
    with pytest.raises(ValueError, match="non-negative"):
        ProgramADAliasEffectAnalysis(
            alias_sets=(alias_set,),
            mutation_effects=(-1,),
            alias_edges=(),
            unknown_aliasing=False,
            claim_boundary="metadata_only_no_general_alias_lattice",
        )
    with pytest.raises(ValueError, match="alias_edges"):
        ProgramADAliasEffectAnalysis(
            alias_sets=(alias_set,),
            mutation_effects=(),
            alias_edges=cast(tuple[ProgramADAliasEdge, ...], (object(),)),
            unknown_aliasing=False,
            claim_boundary="metadata_only_no_general_alias_lattice",
        )
    with pytest.raises(ValueError, match="unknown_aliasing"):
        ProgramADAliasEffectAnalysis(
            alias_sets=(alias_set,),
            mutation_effects=(),
            alias_edges=(edge,),
            unknown_aliasing=cast(bool, "no"),
            claim_boundary="metadata_only_no_general_alias_lattice",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        ProgramADAliasEffectAnalysis(
            alias_sets=(alias_set,),
            mutation_effects=(),
            alias_edges=(edge,),
            unknown_aliasing=False,
            claim_boundary="",
        )

    with pytest.raises(ValueError, match="index"):
        ProgramADStaticAliasLatticeComponent(
            index=-1,
            members=("a",),
            edge_kinds=(),
            versions=(),
            mutation_versions=(),
        )
    with pytest.raises(ValueError, match="members"):
        ProgramADStaticAliasLatticeComponent(
            index=0,
            members=("",),
            edge_kinds=(),
            versions=(),
            mutation_versions=(),
        )
    with pytest.raises(ValueError, match="sorted"):
        ProgramADStaticAliasLatticeComponent(
            index=0,
            members=("b", "a"),
            edge_kinds=(),
            versions=(),
            mutation_versions=(),
        )
    with pytest.raises(ValueError, match="edge_kinds"):
        ProgramADStaticAliasLatticeComponent(
            index=0,
            members=("a",),
            edge_kinds=("",),
            versions=(),
            mutation_versions=(),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        ProgramADStaticAliasLatticeComponent(
            index=0,
            members=("a",),
            edge_kinds=("view_alias", "source_alias"),
            versions=(),
            mutation_versions=(),
        )
    with pytest.raises(ValueError, match="versions"):
        ProgramADStaticAliasLatticeComponent(
            index=0,
            members=("a",),
            edge_kinds=(),
            versions=(-1,),
            mutation_versions=(),
        )
    with pytest.raises(ValueError, match="mutation_versions"):
        ProgramADStaticAliasLatticeComponent(
            index=0,
            members=("a",),
            edge_kinds=(),
            versions=(),
            mutation_versions=(-1,),
        )

    valid_report = ProgramADStaticAliasLatticeReport(
        components=(component,),
        mutation_effects=(),
        non_executed_phi_nodes=(),
        non_executed_control_alias_edges=(),
        unknown_alias_edge_kinds=(),
        blocker_reasons=(),
        complete=True,
        claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
    )
    assert valid_report.as_dict()["components"]
    assert valid_report.as_dict()["view_alias_provenance"] == []
    assert valid_report.as_dict()["malformed_view_alias_edges"] == []
    assert valid_report.as_dict()["list_alias_provenance"] == []
    assert valid_report.as_dict()["malformed_list_alias_edges"] == []
    assert valid_report.as_dict()["loop_carried_state_provenance"] == []
    assert valid_report.as_dict()["malformed_loop_carried_state_edges"] == []
    assert valid_report.as_dict()["unsupported_python_semantics"] == []
    assert valid_report.as_dict()["unsupported_object_attribute_roots"] == []
    assert valid_report.as_dict()["unsupported_object_attribute_details"] == []
    mutation_blocked_report = ProgramADStaticAliasLatticeReport(
        components=(component,),
        mutation_effects=(0,),
        non_executed_phi_nodes=(),
        non_executed_control_alias_edges=(),
        unknown_alias_edge_kinds=(),
        blocker_reasons=("mutation_effects_require_versioned_alias_semantics",),
        complete=False,
        claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
    )
    assert mutation_blocked_report.complete is False
    assert mutation_blocked_report.mutation_effects == (0,)
    unsupported_blocked_report = ProgramADStaticAliasLatticeReport(
        components=(component,),
        mutation_effects=(),
        non_executed_phi_nodes=(),
        non_executed_control_alias_edges=(),
        unknown_alias_edge_kinds=(),
        blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
        complete=False,
        claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        unsupported_python_semantics=("filtered_comprehension",),
    )
    assert unsupported_blocked_report.complete is False
    assert unsupported_blocked_report.unsupported_python_semantics == ("filtered_comprehension",)
    unsupported_diagnostic_report = ProgramADStaticAliasLatticeReport(
        components=(component,),
        mutation_effects=(),
        non_executed_phi_nodes=(),
        non_executed_control_alias_edges=(),
        unknown_alias_edge_kinds=(),
        blocker_reasons=(
            "object_attributes_require_static_object_model",
            "unsupported_python_semantics_require_frontend_lowering",
        ),
        complete=False,
        claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        unsupported_python_semantics=("object_attribute",),
        unsupported_semantic_diagnostics=(unsupported_diagnostic,),
        unsupported_object_attribute_roots=("captured",),
        unsupported_object_attribute_details=("object_attribute:captured",),
    )
    assert unsupported_diagnostic_report.unsupported_semantic_diagnostics == (
        unsupported_diagnostic,
    )
    assert unsupported_diagnostic_report.as_dict()["unsupported_semantic_diagnostics"] == [
        unsupported_diagnostic.to_dict()
    ]
    with pytest.raises(ValueError, match="components"):
        ProgramADStaticAliasLatticeReport(
            components=cast(tuple[ProgramADStaticAliasLatticeComponent, ...], (object(),)),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="dense"):
        ProgramADStaticAliasLatticeReport(
            components=(
                ProgramADStaticAliasLatticeComponent(
                    index=1,
                    members=("a",),
                    edge_kinds=(),
                    versions=(),
                    mutation_versions=(),
                ),
            ),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    for field_name in ("mutation_effects", "non_executed_phi_nodes"):
        kwargs: dict[str, object] = {
            "components": (),
            "mutation_effects": (),
            "non_executed_phi_nodes": (),
            "non_executed_control_alias_edges": (),
            "unknown_alias_edge_kinds": (),
            "blocker_reasons": (),
            "complete": False,
            "claim_boundary": "static_alias_lattice_over_emitted_program_ad_ir",
        }
        kwargs[field_name] = (1, 0)
        with pytest.raises(ValueError, match=field_name):
            ProgramADStaticAliasLatticeReport(**cast(Any, kwargs))
    with pytest.raises(ValueError, match="control_alias_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=("",),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=("b->a", "a->b"),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="unknown_alias_edge_kinds"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=("",),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="sorted unique"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=("z", "a"),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="unknown_alias_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=("runtime_unknown_alias",),
            blocker_reasons=("unknown_alias_edge_kinds",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unknown_alias_edges=cast(tuple[ProgramADUnknownAliasEdge, ...], (object(),)),
        )
    with pytest.raises(ValueError, match="view_alias_provenance"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            view_alias_provenance=cast(tuple[ProgramADViewAliasProvenance, ...], (object(),)),
        )
    with pytest.raises(ValueError, match="list_alias_provenance"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            list_alias_provenance=cast(tuple[ProgramADListAliasProvenance, ...], (object(),)),
        )
    with pytest.raises(ValueError, match="loop_carried_state_provenance"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            loop_carried_state_provenance=cast(
                tuple[ProgramADLoopCarriedStateProvenance, ...],
                (object(),),
            ),
        )
    with pytest.raises(ValueError, match="rebinding_alias_provenance"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            rebinding_alias_provenance=cast(
                tuple[ProgramADRebindingAliasProvenance, ...],
                (object(),),
            ),
        )
    with pytest.raises(ValueError, match="sorted unique"):
        ProgramADStaticAliasLatticeReport(
            components=(
                ProgramADStaticAliasLatticeComponent(
                    index=0,
                    members=("%array[1]", "view:getitem:12[0]"),
                    edge_kinds=("view_alias",),
                    versions=(12,),
                    mutation_versions=(),
                ),
            ),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            view_alias_provenance=(
                ProgramADViewAliasProvenance(
                    source="%array[2]",
                    target="view:getitem:12[1]",
                    operation="getitem",
                    view_id=12,
                    output_index=1,
                    version=12,
                ),
                view_provenance,
            ),
        )
    with pytest.raises(ValueError, match="view alias components require"):
        ProgramADStaticAliasLatticeReport(
            components=(
                ProgramADStaticAliasLatticeComponent(
                    index=0,
                    members=("%array[1]", "view:getitem:12[0]"),
                    edge_kinds=("view_alias",),
                    versions=(12,),
                    mutation_versions=(),
                ),
            ),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="view alias provenance requires"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            view_alias_provenance=(view_provenance,),
        )
    with pytest.raises(ValueError, match="list alias components require"):
        ProgramADStaticAliasLatticeReport(
            components=(
                ProgramADStaticAliasLatticeComponent(
                    index=0,
                    members=("list:scratch", "name:alias"),
                    edge_kinds=("list_alias",),
                    versions=(4,),
                    mutation_versions=(),
                ),
            ),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="list alias provenance requires"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            list_alias_provenance=(list_provenance,),
        )
    with pytest.raises(ValueError, match="rebinding alias components require"):
        ProgramADStaticAliasLatticeReport(
            components=(
                ProgramADStaticAliasLatticeComponent(
                    index=0,
                    members=("name:rebound", "name:seed"),
                    edge_kinds=("local_rebinding_alias",),
                    versions=(4,),
                    mutation_versions=(),
                ),
            ),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="rebinding alias provenance requires"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            rebinding_alias_provenance=(rebinding_provenance,),
        )
    with pytest.raises(ValueError, match="malformed_view_alias_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_view_alias_edges=("",),
        )
    with pytest.raises(ValueError, match="malformed_list_alias_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_list_alias_edges=("",),
        )
    with pytest.raises(ValueError, match="malformed_loop_carried_state_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_loop_carried_state_edges=("",),
        )
    with pytest.raises(ValueError, match="malformed_rebinding_alias_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_rebinding_alias_edges=("",),
        )
    with pytest.raises(ValueError, match="malformed view-alias"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_view_alias_edges=("bad-edge",),
        )
    with pytest.raises(ValueError, match="malformed list-alias"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_list_alias_edges=("bad-edge",),
        )
    with pytest.raises(ValueError, match="malformed loop-carried state"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_loop_carried_state_edges=("bad-edge",),
        )
    with pytest.raises(ValueError, match="malformed rebinding-alias"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_rebinding_alias_edges=("bad-edge",),
        )
    with pytest.raises(ValueError, match="sorted unique"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=("runtime_unknown_alias",),
            blocker_reasons=("unknown_alias_edge_kinds",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unknown_alias_edges=(
                ProgramADUnknownAliasEdge(
                    source="runtime:z",
                    target="%0",
                    kind="runtime_unknown_alias",
                    version=0,
                ),
                unknown_edge,
            ),
        )
    with pytest.raises(ValueError, match="must match unknown_alias_edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=("runtime_unknown_alias",),
            blocker_reasons=("unknown_alias_edge_kinds",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="unknown_alias_edges require a blocker"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=("runtime_unknown_alias",),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unknown_alias_edges=(unknown_edge,),
        )
    with pytest.raises(ValueError, match="unknown alias blocker requires"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("unknown_alias_edge_kinds",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="blocker_reasons"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("z", "a"),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="unsupported_python_semantics"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("",),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("set_or_dict_comprehension", "filtered_comprehension"),
        )
    with pytest.raises(ValueError, match="unsupported_semantic_diagnostics"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("object_attribute",),
            unsupported_semantic_diagnostics=cast(
                tuple[WholeProgramUnsupportedSemanticDiagnostic, ...],
                (object(),),
            ),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("object_attribute",),
            unsupported_semantic_diagnostics=(
                WholeProgramUnsupportedSemanticDiagnostic(
                    semantic="object_attribute",
                    detail="object_attribute:later",
                    line_number=3,
                    absolute_line_number=43,
                    region_ids=("root",),
                    bytecode_offsets=(10,),
                ),
                unsupported_diagnostic,
            ),
        )
    with pytest.raises(ValueError, match="must match unsupported_python_semantics"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("filtered_comprehension",),
            unsupported_semantic_diagnostics=(unsupported_diagnostic,),
        )
    with pytest.raises(ValueError, match="unsupported_object_attribute_roots"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(
                "object_attributes_require_static_object_model",
                "unsupported_python_semantics_require_frontend_lowering",
            ),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("object_attribute",),
            unsupported_semantic_diagnostics=(unsupported_diagnostic,),
            unsupported_object_attribute_roots=("",),
            unsupported_object_attribute_details=("object_attribute:captured",),
        )
    with pytest.raises(ValueError, match="unsupported_object_attribute_details"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(
                "object_attributes_require_static_object_model",
                "unsupported_python_semantics_require_frontend_lowering",
            ),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("object_attribute",),
            unsupported_semantic_diagnostics=(unsupported_diagnostic,),
            unsupported_object_attribute_roots=("captured",),
            unsupported_object_attribute_details=("object_attribute:other",),
        )
    with pytest.raises(ValueError, match="complete must be boolean"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=cast(bool, "no"),
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="cannot carry mutation_effects"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(0,),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="cannot carry unsupported_python_semantics"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("filtered_comprehension",),
        )
    with pytest.raises(ValueError, match="cannot carry malformed list-alias edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_list_alias_edges=("bad-edge",),
        )
    with pytest.raises(ValueError, match="cannot carry malformed loop-carried state edges"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            malformed_loop_carried_state_edges=("bad-edge",),
        )
    with pytest.raises(ValueError, match="mutation_effects require a blocker"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(0,),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="mutation blocker requires mutation_effects"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("mutation_effects_require_versioned_alias_semantics",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="unsupported_python_semantics require"):
        ProgramADStaticAliasLatticeReport(
            components=(component,),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("filtered_comprehension",),
        )
    with pytest.raises(ValueError, match="unsupported semantics blocker requires"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="unsupported object attributes require"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
            unsupported_python_semantics=("object_attribute",),
            unsupported_semantic_diagnostics=(unsupported_diagnostic,),
            unsupported_object_attribute_roots=("captured",),
            unsupported_object_attribute_details=("object_attribute:captured",),
        )
    with pytest.raises(ValueError, match="object attribute blocker requires"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("object_attributes_require_static_object_model",),
            complete=False,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=(),
            complete=False,
            claim_boundary="",
        )


def test_program_ad_alias_provenance_rejects_every_malformed_marker_shape() -> None:
    """Typed provenance should fail closed for every supported marker component."""
    with pytest.raises(ValueError, match="must match operation"):
        ProgramADViewAliasProvenance("%a", "view:getitem:9[0]", "getitem", 8, 0, 0)

    for kwargs, match in (
        (
            dict(
                source="list:",
                target="name:alias",
                list_name="scratch",
                target_kind="local_name",
                version=0,
            ),
            "include a list name",
        ),
        (
            dict(
                source="list:scratch",
                target="name:",
                list_name="scratch",
                target_kind="local_name",
                version=0,
            ),
            "must name a local",
        ),
        (
            dict(
                source="list:scratch",
                target="name:alias",
                list_name="scratch",
                target_kind="indexed_mutation_source",
                version=0,
            ),
            "mutation target",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADListAliasProvenance(**cast(Any, kwargs))

    with pytest.raises(ValueError, match="source must match state_name"):
        ProgramADLoopCarriedStateProvenance(
            "loop:other:entry", "loop:carry:backedge", "carry", "entry", "backedge", 0
        )

    for kwargs, match in (
        (
            dict(
                source="control:if:4:body",
                target="control:",
                branch_line=4,
                branch_arm="body",
                target_label="target",
                version=0,
            ),
            "include a label",
        ),
        (
            dict(
                source="control:if:4:body",
                target="control:target",
                branch_line=4,
                branch_arm="body",
                target_label="",
                version=0,
            ),
            "target_label",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            ProgramADControlPathAliasProvenance(**cast(Any, kwargs))

    local_defaults: dict[str, object] = dict(
        source="name:seed",
        target="name:result",
        binding_kind="local",
        source_name="seed",
        expression_line=None,
        expression_label=None,
        target_name="result",
        version=0,
    )
    expression_defaults: dict[str, object] = dict(
        source="expr:4:seed+1",
        target="name:result",
        binding_kind="expression",
        source_name=None,
        expression_line=4,
        expression_label="seed+1",
        target_name="result",
        version=0,
    )
    rebinding_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({**local_defaults, "target": "result"}, "target must be a local name"),
        ({**local_defaults, "target": "name:"}, "target must name a local"),
        ({**local_defaults, "target_name": ""}, "target_name must be non-empty"),
        ({**local_defaults, "source": "name:"}, "source must name a local"),
        ({**local_defaults, "source_name": ""}, "source_name must be non-empty"),
        ({**local_defaults, "source_name": "other"}, "source_name must match source"),
        ({**local_defaults, "expression_line": 4}, "cannot carry expression metadata"),
        ({**expression_defaults, "expression_line": 0}, "expression_line must be positive"),
        ({**expression_defaults, "expression_label": ""}, "expression_label must be non-empty"),
    )
    for kwargs, match in rebinding_cases:
        with pytest.raises(ValueError, match=match):
            ProgramADRebindingAliasProvenance(**cast(Any, kwargs))


def test_program_ad_alias_lattice_rejects_all_cross_field_inconsistencies() -> None:
    """The lattice report should reject malformed ordering, provenance, and blockers."""
    control_a = ProgramADControlPathAliasProvenance(
        "control:if:4:body", "control:name:a", 4, "body", "name:a", 0
    )
    control_b = ProgramADControlPathAliasProvenance(
        "control:if:5:body", "control:name:b", 5, "body", "name:b", 0
    )
    list_a = ProgramADListAliasProvenance("list:a", "name:a", "a", "local_name", 0)
    list_b = ProgramADListAliasProvenance("list:b", "name:b", "b", "local_name", 0)
    loop_a = ProgramADLoopCarriedStateProvenance(
        "loop:a:entry", "loop:a:backedge", "a", "entry", "backedge", 0
    )
    loop_b = ProgramADLoopCarriedStateProvenance(
        "loop:b:entry", "loop:b:backedge", "b", "entry", "backedge", 0
    )
    rebinding_a = ProgramADRebindingAliasProvenance(
        "name:a", "name:c", "local", "a", None, None, "c", 0
    )
    rebinding_b = ProgramADRebindingAliasProvenance(
        "name:b", "name:d", "local", "b", None, None, "d", 0
    )

    invalid_reports = (
        (dict(control_path_alias_provenance=cast(Any, (object(),))), "control_path_alias"),
        (
            dict(control_path_alias_provenance=(control_b, control_a)),
            "control_path_alias_provenance must be sorted unique",
        ),
        (
            dict(
                control_path_alias_provenance=(control_a,),
                non_executed_control_alias_edges=("wrong->edge",),
            ),
            "must match non_executed_control_alias_edges",
        ),
        (dict(malformed_control_path_alias_edges=("",)), "must contain non-empty strings"),
        (dict(malformed_control_path_alias_edges=("z", "a")), "must be sorted and unique"),
        (dict(malformed_view_alias_edges=("z", "a")), "must be sorted and unique"),
        (dict(list_alias_provenance=(list_b, list_a)), "list_alias_provenance must be sorted"),
        (dict(malformed_list_alias_edges=("z", "a")), "must be sorted and unique"),
        (
            dict(loop_carried_state_provenance=(loop_b, loop_a)),
            "loop_carried_state_provenance must be sorted",
        ),
        (
            dict(malformed_loop_carried_state_edges=("z", "a")),
            "must be sorted and unique",
        ),
        (
            dict(rebinding_alias_provenance=(rebinding_b, rebinding_a)),
            "rebinding_alias_provenance must be sorted",
        ),
        (
            dict(malformed_rebinding_alias_edges=("z", "a")),
            "must be sorted and unique",
        ),
        (dict(unsupported_object_attribute_roots=("z", "a")), "must be sorted and unique"),
        (dict(unsupported_object_attribute_roots=("captured",)), "must match"),
    )
    for overrides, match in invalid_reports:
        with pytest.raises(ValueError, match=match):
            _alias_lattice_report(**overrides)

    control_component = ProgramADStaticAliasLatticeComponent(
        0, ("control:if:4:body", "control:name:a"), ("control_path_alias",), (0,), ()
    )
    loop_component = ProgramADStaticAliasLatticeComponent(
        0, ("loop:a:backedge", "loop:a:entry"), ("loop_carried_state",), (0,), ()
    )
    component_cases = (
        (dict(components=(control_component,)), "control-path alias components require"),
        (
            dict(
                control_path_alias_provenance=(control_a,),
                non_executed_control_alias_edges=("control:if:4:body->control:name:a",),
            ),
            "control-path alias provenance requires",
        ),
        (dict(components=(loop_component,)), "loop-carried state components require"),
        (
            dict(loop_carried_state_provenance=(loop_a,)),
            "loop-carried state provenance requires",
        ),
    )
    for overrides, match in component_cases:
        with pytest.raises(ValueError, match=match):
            _alias_lattice_report(**overrides)

    unknown = ProgramADUnknownAliasEdge("runtime:a", "%0", "runtime_unknown_alias", 0)
    complete_cases = (
        (
            dict(
                components=(control_component,),
                non_executed_control_alias_edges=("control:if:4:body->control:name:a",),
                control_path_alias_provenance=(control_a,),
                complete=True,
            ),
            "cannot carry control-path aliases",
        ),
        (
            dict(malformed_control_path_alias_edges=("bad",), complete=True),
            "cannot carry malformed control-path",
        ),
        (
            dict(
                unknown_alias_edge_kinds=("runtime_unknown_alias",),
                unknown_alias_edges=(unknown,),
                complete=True,
            ),
            "cannot carry unknown_alias_edges",
        ),
        (
            dict(malformed_view_alias_edges=("bad",), complete=True),
            "cannot carry malformed view-alias",
        ),
        (
            dict(malformed_rebinding_alias_edges=("bad",), complete=True),
            "cannot carry malformed rebinding-alias",
        ),
    )
    for overrides, match in complete_cases:
        with pytest.raises(ValueError, match=match):
            _alias_lattice_report(**overrides)

    blocker_cases = (
        (
            dict(non_executed_control_alias_edges=("a->b",)),
            "control-path aliases require a blocker",
        ),
        (
            dict(blocker_reasons=("control_path_aliases_require_branch_semantics",)),
            "control-path blocker requires",
        ),
        (
            dict(malformed_control_path_alias_edges=("bad",)),
            "malformed control-path alias edges require",
        ),
        (
            dict(blocker_reasons=("control_path_alias_provenance_requires_parseable_targets",)),
            "malformed control-path alias blocker requires",
        ),
        (
            dict(blocker_reasons=("view_alias_provenance_requires_parseable_targets",)),
            "malformed view-alias blocker requires",
        ),
        (
            dict(blocker_reasons=("list_alias_provenance_requires_parseable_targets",)),
            "malformed list-alias blocker requires",
        ),
        (
            dict(blocker_reasons=("loop_carried_state_provenance_requires_parseable_targets",)),
            "malformed loop-carried state blocker requires",
        ),
        (
            dict(blocker_reasons=("rebinding_alias_provenance_requires_parseable_targets",)),
            "malformed rebinding-alias blocker requires",
        ),
    )
    for overrides, match in blocker_cases:
        with pytest.raises(ValueError, match=match):
            _alias_lattice_report(**overrides)

    diagnostic = WholeProgramUnsupportedSemanticDiagnostic(
        semantic="object_attribute",
        detail="captured",
        line_number=1,
        absolute_line_number=1,
        region_ids=("root",),
        bytecode_offsets=(0,),
    )
    report = _alias_lattice_report(
        unsupported_python_semantics=("object_attribute",),
        unsupported_semantic_diagnostics=(diagnostic,),
        unsupported_object_attribute_roots=("captured",),
        unsupported_object_attribute_details=("captured",),
        blocker_reasons=(
            "object_attributes_require_static_object_model",
            "unsupported_python_semantics_require_frontend_lowering",
        ),
    )
    assert report.unsupported_object_attribute_roots == ("captured",)
