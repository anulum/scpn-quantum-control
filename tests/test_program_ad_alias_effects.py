# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD alias/effect tests
"""Tests for Program AD alias/effect analysis and static alias-lattice reports."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import program_ad_alias_analysis as alias_analysis_module
from scpn_quantum_control.differentiable import (
    PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY,
    PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY,
    Parameter,
    ProgramADAliasEdge,
    ProgramADAliasEffectAnalysis,
    ProgramADAliasSet,
    ProgramADControlPathAliasProvenance,
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADListAliasProvenance,
    ProgramADLoopCarriedStateProvenance,
    ProgramADRebindingAliasProvenance,
    ProgramADSSAValue,
    ProgramADStaticAliasLatticeComponent,
    ProgramADStaticAliasLatticeReport,
    ProgramADUnknownAliasEdge,
    ProgramADViewAliasProvenance,
    WholeProgramUnsupportedSemanticDiagnostic,
    analyze_program_ad_alias_effects,
    compile_whole_program_frontend,
    program_ad_static_alias_lattice_report,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


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


def test_program_ad_alias_analysis_fail_closed_entry_points() -> None:
    """Alias-analysis entry points should reject non-effect-IR inputs."""

    with pytest.raises(ValueError, match="alias analysis requires"):
        analyze_program_ad_alias_effects(cast(ProgramADEffectIR, object()))
    with pytest.raises(ValueError, match="static alias lattice requires"):
        program_ad_static_alias_lattice_report(cast(ProgramADEffectIR, object()))
    valid_ir = ProgramADEffectIR(
        ssa_values=(
            ProgramADSSAValue(
                "%0",
                producer=0,
                version=0,
                shape=(),
                dtype="float64",
                effect=0,
            ),
        ),
        effects=(),
        alias_edges=(),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )
    with pytest.raises(ValueError, match="unsupported_semantic_diagnostics"):
        program_ad_static_alias_lattice_report(
            valid_ir,
            unsupported_semantic_diagnostics=cast(
                tuple[WholeProgramUnsupportedSemanticDiagnostic, ...],
                (object(),),
            ),
        )


def test_program_ad_static_alias_lattice_tracks_mutation_versions_directly() -> None:
    """Static alias lattice reports should retain mutation-version component metadata."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="mutation",
        target="%0",
        inputs=("%0",),
        version=2,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="%0",
                target="source:mutation",
                kind="mutation_version",
                version=2,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.mutation_effects == (0,)
    assert report.complete is False
    assert "mutation_effects_require_versioned_alias_semantics" in report.blocker_reasons
    assert len(report.components) == 1
    assert report.components[0].mutation_versions == (2,)
    assert report.components[0].edge_kinds == ("mutation_version",)
    assert report.as_dict()["blocker_reasons"] == list(report.blocker_reasons)


def test_program_ad_static_alias_lattice_blocks_frontend_unsupported_semantics() -> None:
    """Static alias lattice reports should consume frontend unsupported semantics."""

    def supported_alias_objective(values: Any) -> object:
        view = values.reshape((2, 2)).T.ravel()
        return view[0] + 2.0 * view[3]

    def unsupported_dynamic_objective(values: Any) -> object:
        selected = [value for value in values if value > 0.0]
        return selected[0]

    result = whole_program_value_and_grad(
        supported_alias_objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    frontend_report = compile_whole_program_frontend(unsupported_dynamic_objective)
    assert result.program_ir is not None
    assert frontend_report.semantics_report.unsupported_python_semantics == (
        "filtered_comprehension",
    )
    assert frontend_report.unsupported_semantic_diagnostic_count == 1

    report = program_ad_static_alias_lattice_report(
        result.program_ir,
        unsupported_python_semantics=(
            frontend_report.semantics_report.unsupported_python_semantics
        ),
    )

    assert report.complete is False
    assert report.unsupported_python_semantics == ("filtered_comprehension",)
    assert report.mutation_effects == ()
    assert "unsupported_python_semantics_require_frontend_lowering" in report.blocker_reasons
    assert report.as_dict()["unsupported_python_semantics"] == ["filtered_comprehension"]


def test_program_ad_static_alias_lattice_preserves_unsupported_diagnostics() -> None:
    """Static alias lattice reports should retain frontend diagnostic provenance."""

    class CapturedState:
        """Captured object whose attribute access remains a frontend blocker."""

        value: float

    captured = CapturedState()
    captured.value = 2.0

    def supported_alias_objective(values: Any) -> object:
        view = values.reshape((2, 2)).T.ravel()
        return view[0] + view[3]

    def unsupported_attribute_objective(values: Any) -> object:
        return captured.value + values[0]

    result = whole_program_value_and_grad(
        supported_alias_objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    frontend_report = compile_whole_program_frontend(unsupported_attribute_objective)
    assert result.program_ir is not None
    assert frontend_report.semantics_report.unsupported_python_semantics == ("object_attribute",)
    assert frontend_report.unsupported_semantic_diagnostic_count == 1

    report = program_ad_static_alias_lattice_report(
        result.program_ir,
        unsupported_semantic_diagnostics=(frontend_report.unsupported_semantic_diagnostics),
    )

    assert report.complete is False
    assert report.unsupported_python_semantics == ("object_attribute",)
    assert report.unsupported_object_attribute_roots == ("captured",)
    assert report.unsupported_object_attribute_details == ("object_attribute:captured",)
    assert len(report.unsupported_semantic_diagnostics) == 1
    diagnostic = report.unsupported_semantic_diagnostics[0]
    assert diagnostic.semantic == "object_attribute"
    assert diagnostic.detail == "object_attribute:captured"
    assert diagnostic.line_number > 0
    assert diagnostic.absolute_line_number is not None
    assert diagnostic.region_ids
    assert diagnostic.bytecode_offsets
    payload = report.as_dict()
    assert payload["unsupported_semantic_diagnostics"] == [diagnostic.to_dict()]
    assert payload["unsupported_object_attribute_roots"] == ["captured"]
    assert payload["unsupported_object_attribute_details"] == ["object_attribute:captured"]
    assert "unsupported_python_semantics_require_frontend_lowering" in report.blocker_reasons
    assert "object_attributes_require_static_object_model" in report.blocker_reasons


def test_program_ad_alias_effect_analysis_summarizes_alias_sets_and_mutations() -> None:
    """Program AD alias/effect analysis should expose deterministic alias sets."""

    def objective(values: Any) -> object:
        view = values.copy()
        total = values[0]
        for index in range(1, 3):
            total = total + view[index] * float(index)
        view[0] = total
        return view[0] + values[2]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None
    analysis = analyze_program_ad_alias_effects(result.program_ir)

    assert isinstance(analysis, ProgramADAliasEffectAnalysis)
    assert all(isinstance(alias_set, ProgramADAliasSet) for alias_set in analysis.alias_sets)
    assert analysis.claim_boundary == "metadata_only_no_general_alias_lattice"
    assert analysis.unknown_aliasing is False
    assert analysis.mutation_effects
    assert analysis.mutation_effects == tuple(sorted(analysis.mutation_effects))
    assert any(edge.kind == "mutation_version" for edge in analysis.alias_edges)
    assert any(
        "%array[0]" in alias_set.members
        and any(member.startswith("%") and member != "%array[0]" for member in alias_set.members)
        for alias_set in analysis.alias_sets
    )
    payload = analysis.as_dict()
    assert payload["claim_boundary"] == "metadata_only_no_general_alias_lattice"
    assert payload["unknown_aliasing"] is False
    assert payload["mutation_effects"] == list(analysis.mutation_effects)

    unsupported = ProgramADEffectIR(
        ssa_values=result.program_ir.ssa_values,
        effects=result.program_ir.effects,
        alias_edges=(
            ProgramADAliasEdge(
                source="dynamic_object",
                target="%0",
                kind="runtime_unknown_alias",
                version=0,
            ),
        ),
        control_regions=result.program_ir.control_regions,
        serialization="program_ad_effect_ir.v1",
        phi_nodes=result.program_ir.phi_nodes,
    )
    with pytest.raises(ValueError, match="unknown alias"):
        analyze_program_ad_alias_effects(unsupported)


def test_program_ad_static_alias_lattice_reports_complete_emitted_ir() -> None:
    """Static alias lattice reports should classify emitted Program AD alias components."""

    def objective(values: Any) -> object:
        view = values.reshape((2, 2)).T.ravel()
        return view[0] + 2.0 * view[3]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    assert result.program_ir is not None

    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert isinstance(report, ProgramADStaticAliasLatticeReport)
    assert all(
        isinstance(component, ProgramADStaticAliasLatticeComponent)
        for component in report.components
    )
    assert report.complete is True
    assert report.blocker_reasons == ()
    assert report.non_executed_phi_nodes == ()
    assert report.unknown_alias_edge_kinds == ()
    assert report.malformed_view_alias_edges == ()
    assert report.control_path_alias_provenance == ()
    assert report.malformed_control_path_alias_edges == ()
    assert report.claim_boundary == (
        "static_alias_lattice_over_emitted_program_ad_ir_no_non_executed_branch_semantics"
    )
    assert any(
        "view_alias" in component.edge_kinds
        and "%array[0]" in component.members
        and any(member.startswith("view:transpose") for member in component.members)
        for component in report.components
    )
    view_rows = {
        (row.operation, row.source, row.target, row.view_id, row.output_index)
        for row in report.view_alias_provenance
    }
    assert ("reshape", "%array[0]", "view:reshape:0[0]", 0, 0) in view_rows
    assert ("transpose", "%array[0]", "view:transpose:8[0]", 8, 0) in view_rows
    assert ("ravel", "%array[0]", "view:ravel:16[0]", 16, 0) in view_rows
    assert all(
        isinstance(row, ProgramADViewAliasProvenance) for row in report.view_alias_provenance
    )
    payload = report.as_dict()
    assert payload["complete"] is True
    assert payload["blocker_reasons"] == []
    assert payload["components"]
    assert payload["control_path_alias_provenance"] == []
    assert payload["malformed_control_path_alias_edges"] == []
    view_payload = cast(list[dict[str, object]], payload["view_alias_provenance"])
    assert {
        "source": "%array[0]",
        "target": "view:reshape:0[0]",
        "operation": "reshape",
        "view_id": 0,
        "output_index": 0,
        "version": 0,
    } in view_payload
    assert payload["malformed_view_alias_edges"] == []


def test_program_ad_static_alias_lattice_tracks_local_object_attribute_aliases() -> None:
    """Static alias lattice reports should include bounded local object attributes."""

    class Scratch:
        """Mutable local container used to exercise source-level attribute aliases."""

        left: Any
        right: Any
        total: Any

    def objective(values: Any) -> object:
        scratch = Scratch()
        scratch.left = values[0]
        scratch.right = values[1]
        alias = scratch.left
        combined = alias + 2.0 * scratch.right
        scratch.total = combined
        return scratch.total + values[2]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None

    analysis = analyze_program_ad_alias_effects(result.program_ir)
    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert result.semantics_report is not None
    assert result.semantics_report.aliasing_observed is True
    assert report.complete is True
    assert report.blocker_reasons == ()
    assert any(edge.kind == "object_attribute_alias" for edge in analysis.alias_edges)
    assert any(edge.kind == "expression_rebinding_alias" for edge in analysis.alias_edges)
    assert any(
        "object_attribute_alias" in component.edge_kinds
        and "object:scratch" in component.members
        and "attr:scratch.left" in component.members
        and "attr:scratch.total" in component.members
        for component in report.components
    )
    assert any(
        "expression_rebinding_alias" in component.edge_kinds
        and "name:combined" in component.members
        for component in report.components
    )
    payload = report.as_dict()
    assert payload["complete"] is True
    np.testing.assert_allclose(result.gradient, [1.0, 2.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_alias_lattice_reports_rebinding_provenance() -> None:
    """Static alias lattice reports should retain parseable rebinding provenance."""

    def objective(values: Any) -> object:
        seed = values[0]
        rebound = seed
        combined = rebound + 2.0 * values[1]
        return combined + values[2]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None

    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert report.complete is True
    assert report.malformed_rebinding_alias_edges == ()
    assert any(
        "local_rebinding_alias" in component.edge_kinds
        and "name:seed" in component.members
        and "name:rebound" in component.members
        for component in report.components
    )
    assert any(
        "expression_rebinding_alias" in component.edge_kinds
        and "name:combined" in component.members
        for component in report.components
    )
    local_rows = {
        (row.binding_kind, row.source_name, row.target_name, row.source, row.target)
        for row in report.rebinding_alias_provenance
        if row.binding_kind == "local"
    }
    assert ("local", "seed", "rebound", "name:seed", "name:rebound") in local_rows
    expression_rows = tuple(
        row for row in report.rebinding_alias_provenance if row.binding_kind == "expression"
    )
    assert any(
        row.target_name == "combined"
        and row.target == "name:combined"
        and row.source.startswith("expr:")
        and row.expression_line is not None
        and row.expression_label
        for row in expression_rows
    )
    assert all(
        isinstance(row, ProgramADRebindingAliasProvenance)
        for row in report.rebinding_alias_provenance
    )
    payload = report.as_dict()
    assert payload["complete"] is True
    assert payload["malformed_rebinding_alias_edges"] == []
    assert {
        "source": "name:seed",
        "target": "name:rebound",
        "binding_kind": "local",
        "source_name": "seed",
        "expression_line": None,
        "expression_label": None,
        "target_name": "rebound",
        "version": next(
            row.version
            for row in report.rebinding_alias_provenance
            if row.source == "name:seed" and row.target == "name:rebound"
        ),
    } in cast(list[dict[str, object]], payload["rebinding_alias_provenance"])
    np.testing.assert_allclose(result.gradient, [1.0, 2.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_alias_lattice_reports_malformed_rebinding_aliases() -> None:
    """Static alias lattice reports should block malformed rebinding aliases."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=(),
        version=0,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="seed",
                target="name:rebound",
                kind="local_rebinding_alias",
                version=0,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.complete is False
    assert report.rebinding_alias_provenance == ()
    assert report.malformed_rebinding_alias_edges == (
        "seed->name:rebound:local_rebinding_alias@0",
    )
    assert "rebinding_alias_provenance_requires_parseable_targets" in report.blocker_reasons
    payload = report.as_dict()
    assert payload["rebinding_alias_provenance"] == []
    assert payload["malformed_rebinding_alias_edges"] == [
        "seed->name:rebound:local_rebinding_alias@0"
    ]


def test_program_ad_static_alias_lattice_records_non_executed_phi_blockers() -> None:
    """Static alias lattice reports should keep non-executed branch inputs blocked."""

    def objective(values: Any) -> object:
        total = values[0]
        if values[1] > 0.0:
            total = total + values[2]
        else:
            total = total - values[3]
        return total

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    assert result.program_ir is not None

    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert report.complete is False
    assert report.non_executed_phi_nodes
    assert report.non_executed_control_alias_edges
    assert report.control_path_alias_provenance
    assert report.malformed_control_path_alias_edges == ()
    assert "non_executed_phi_inputs_require_branch_semantics" in report.blocker_reasons
    assert "control_path_aliases_require_branch_semantics" in report.blocker_reasons
    assert report.unknown_alias_edge_kinds == ()
    assert report.as_dict()["non_executed_phi_nodes"] == list(report.non_executed_phi_nodes)
    assert report.as_dict()["non_executed_control_alias_edges"] == list(
        report.non_executed_control_alias_edges
    )
    rows = {(row.branch_arm, row.target_label) for row in report.control_path_alias_provenance}
    assert ("body", "name:total") in rows
    assert ("orelse", "name:total") in rows
    payload = report.as_dict()
    assert {
        "source": next(
            row.source for row in report.control_path_alias_provenance if row.branch_arm == "body"
        ),
        "target": "control:name:total",
        "branch_line": next(
            row.branch_line
            for row in report.control_path_alias_provenance
            if row.branch_arm == "body"
        ),
        "branch_arm": "body",
        "target_label": "name:total",
        "version": next(
            row.version for row in report.control_path_alias_provenance if row.branch_arm == "body"
        ),
    } in cast(list[dict[str, object]], payload["control_path_alias_provenance"])
    assert payload["malformed_control_path_alias_edges"] == []


def test_program_ad_static_alias_lattice_blocks_non_executed_attribute_paths() -> None:
    """Static alias lattice reports should not promote branch-local attribute paths."""

    class Scratch:
        """Mutable local container used to exercise branch-local attribute writes."""

        value: object

    def objective(values: Any) -> object:
        scratch = Scratch()
        if values[2] > 0.0:
            scratch.value = values[0]
        else:
            scratch.value = values[1]
        alias = scratch.value
        return alias + values[3]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    assert result.program_ir is not None

    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert report.complete is False
    assert report.non_executed_phi_nodes
    assert report.non_executed_control_alias_edges
    assert report.control_path_alias_provenance
    assert report.malformed_control_path_alias_edges == ()
    assert "non_executed_phi_inputs_require_branch_semantics" in report.blocker_reasons
    assert "control_path_aliases_require_branch_semantics" in report.blocker_reasons
    assert any(
        row.target_label == "attr:scratch.value" for row in report.control_path_alias_provenance
    )
    assert any(
        "object_attribute_alias" in component.edge_kinds
        and "attr:scratch.value" in component.members
        for component in report.components
    )
    np.testing.assert_allclose(result.gradient, [1.0, 0.0, 0.0, 1.0], atol=1.0e-12)


def test_program_ad_static_alias_lattice_reports_unknown_alias_edges() -> None:
    """Static alias lattice reports should expose unknown edge kinds without promoting them."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=(),
        version=0,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="dynamic_object",
                target="%0",
                kind="runtime_unknown_alias",
                version=0,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.complete is False
    assert report.unknown_alias_edge_kinds == ("runtime_unknown_alias",)
    assert len(report.unknown_alias_edges) == 1
    unknown_edge = report.unknown_alias_edges[0]
    assert unknown_edge.kind == "runtime_unknown_alias"
    assert unknown_edge.source == "dynamic_object"
    assert unknown_edge.target == "%0"
    assert unknown_edge.version == 0
    assert report.as_dict()["unknown_alias_edges"] == [
        {
            "source": "dynamic_object",
            "target": "%0",
            "kind": "runtime_unknown_alias",
            "version": 0,
        }
    ]
    assert "unknown_alias_edge_kinds" in report.blocker_reasons

    with pytest.raises(ValueError, match="complete"):
        ProgramADStaticAliasLatticeReport(
            components=(),
            mutation_effects=(),
            non_executed_phi_nodes=(),
            non_executed_control_alias_edges=(),
            unknown_alias_edge_kinds=(),
            blocker_reasons=("blocked",),
            complete=True,
            claim_boundary="static_alias_lattice_over_emitted_program_ad_ir",
        )


def test_program_ad_static_alias_lattice_reports_malformed_view_aliases() -> None:
    """Static alias lattice reports should block malformed view-alias markers."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=(),
        version=0,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="%0",
                target="view:reshape:missing_index",
                kind="view_alias",
                version=0,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.complete is False
    assert report.view_alias_provenance == ()
    assert report.malformed_view_alias_edges == ("%0->view:reshape:missing_index:view_alias@0",)
    assert "view_alias_provenance_requires_parseable_targets" in report.blocker_reasons
    payload = report.as_dict()
    assert payload["view_alias_provenance"] == []
    assert payload["malformed_view_alias_edges"] == ["%0->view:reshape:missing_index:view_alias@0"]


def test_program_ad_static_alias_lattice_reports_list_alias_provenance() -> None:
    """Static alias lattice reports should preserve local list-alias provenance."""

    def objective(values: Any) -> object:
        scratch = [values[0], values[1]]
        alias = scratch
        alias[0] = values[2]
        return scratch[0] + 2.0 * alias[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None

    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert report.complete is True
    assert report.malformed_list_alias_edges == ()
    assert any(
        "list_alias" in component.edge_kinds
        and "list:scratch" in component.members
        and "name:alias" in component.members
        and "source:list_mutation" in component.members
        for component in report.components
    )
    list_rows = {
        (row.list_name, row.target_kind, row.source, row.target, row.version)
        for row in report.list_alias_provenance
    }
    assert ("scratch", "local_name", "list:scratch", "name:scratch", 2) in list_rows
    assert ("scratch", "local_name", "list:scratch", "name:alias", 3) in list_rows
    assert (
        "scratch",
        "indexed_mutation_source",
        "list:scratch",
        "source:list_mutation",
        4,
    ) in list_rows
    assert all(
        isinstance(row, ProgramADListAliasProvenance) for row in report.list_alias_provenance
    )
    payload = report.as_dict()
    assert payload["complete"] is True
    assert payload["malformed_list_alias_edges"] == []
    assert {
        "source": "list:scratch",
        "target": "name:alias",
        "list_name": "scratch",
        "target_kind": "local_name",
        "version": 3,
    } in cast(list[dict[str, object]], payload["list_alias_provenance"])
    np.testing.assert_allclose(result.gradient, [0.0, 2.0, 1.0], atol=1.0e-12)


def test_program_ad_static_alias_lattice_reports_malformed_list_aliases() -> None:
    """Static alias lattice reports should block malformed list-alias markers."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=(),
        version=0,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="scratch",
                target="name:alias",
                kind="list_alias",
                version=0,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.complete is False
    assert report.list_alias_provenance == ()
    assert report.malformed_list_alias_edges == ("scratch->name:alias:list_alias@0",)
    assert "list_alias_provenance_requires_parseable_targets" in report.blocker_reasons
    payload = report.as_dict()
    assert payload["list_alias_provenance"] == []
    assert payload["malformed_list_alias_edges"] == ["scratch->name:alias:list_alias@0"]


def test_program_ad_static_alias_lattice_reports_loop_carried_state_provenance() -> None:
    """Static alias lattice reports should preserve typed loop-carried state provenance."""

    def objective(values: Any) -> object:
        carry = values[0]
        for index in range(1, 4):
            carry = carry + float(index) * values[index]
        return carry + values[4]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64),
        parameters=(
            Parameter("seed"),
            Parameter("step1"),
            Parameter("step2"),
            Parameter("step3"),
            Parameter("tail"),
        ),
    )
    assert result.program_ir is not None

    report = program_ad_static_alias_lattice_report(result.program_ir)

    assert report.complete is True
    assert report.malformed_loop_carried_state_edges == ()
    assert any(
        "loop_carried_state" in component.edge_kinds
        and "loop:carry:entry" in component.members
        and "loop:carry:backedge" in component.members
        for component in report.components
    )
    assert all(
        isinstance(row, ProgramADLoopCarriedStateProvenance)
        for row in report.loop_carried_state_provenance
    )
    assert len(report.loop_carried_state_provenance) == 1
    row = report.loop_carried_state_provenance[0]
    assert row.source == "loop:carry:entry"
    assert row.target == "loop:carry:backedge"
    assert row.state_name == "carry"
    assert row.entry_label == "entry"
    assert row.backedge_label == "backedge"
    assert row.version >= 0
    payload = report.as_dict()
    assert payload["complete"] is True
    assert payload["malformed_loop_carried_state_edges"] == []
    assert {
        "source": "loop:carry:entry",
        "target": "loop:carry:backedge",
        "state_name": "carry",
        "entry_label": "entry",
        "backedge_label": "backedge",
        "version": row.version,
    } in cast(list[dict[str, object]], payload["loop_carried_state_provenance"])
    np.testing.assert_allclose(result.gradient, [1.0, 1.0, 2.0, 3.0, 1.0], atol=1.0e-12)


def test_program_ad_static_alias_lattice_reports_malformed_loop_carried_state() -> None:
    """Static alias lattice reports should block malformed loop-carried state markers."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=(),
        version=0,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="loop:carry:start",
                target="loop:carry:backedge",
                kind="loop_carried_state",
                version=0,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.complete is False
    assert report.loop_carried_state_provenance == ()
    assert report.malformed_loop_carried_state_edges == (
        "loop:carry:start->loop:carry:backedge:loop_carried_state@0",
    )
    assert "loop_carried_state_provenance_requires_parseable_targets" in report.blocker_reasons
    payload = report.as_dict()
    assert payload["loop_carried_state_provenance"] == []
    assert payload["malformed_loop_carried_state_edges"] == [
        "loop:carry:start->loop:carry:backedge:loop_carried_state@0"
    ]


def test_program_ad_static_alias_lattice_reports_malformed_control_path_aliases() -> None:
    """Static alias lattice reports should block malformed control-path aliases."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=(),
        version=0,
        ordering=0,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(
            ProgramADAliasEdge(
                source="control:if:bad:body",
                target="control:name:total",
                kind="control_path_alias",
                version=0,
            ),
        ),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )

    report = program_ad_static_alias_lattice_report(ir)

    assert report.complete is False
    assert report.control_path_alias_provenance == ()
    assert report.malformed_control_path_alias_edges == (
        "control:if:bad:body->control:name:total:control_path_alias@0",
    )
    assert "control_path_aliases_require_branch_semantics" in report.blocker_reasons
    assert "control_path_alias_provenance_requires_parseable_targets" in report.blocker_reasons
    payload = report.as_dict()
    assert payload["control_path_alias_provenance"] == []
    assert payload["malformed_control_path_alias_edges"] == [
        "control:if:bad:body->control:name:total:control_path_alias@0"
    ]


def test_program_ad_alias_effect_analysis_tracks_array_view_aliases() -> None:
    """Program AD alias metadata should distinguish view aliases from mutations."""

    def objective(values: Any) -> object:
        matrix = values.reshape((2, 3))
        trailing = matrix[:, 1:]
        transposed = trailing.T
        flat = transposed.ravel()
        tensor = values.reshape((1, 2, 1, 3))
        squeezed = np.squeeze(tensor, axis=(0, 2))
        expanded = np.expand_dims(squeezed, axis=0)
        swapped = np.swapaxes(expanded, 0, 1)
        moved = np.moveaxis(swapped, source=2, destination=0)
        repeated = np.repeat(moved, repeats=(1, 2, 1), axis=0)
        promoted = np.atleast_3d(squeezed[0])
        base = values.reshape((2, 3))
        tiled = np.tile(base, (2, 1))
        rolled = np.roll(base, shift=1, axis=1)
        rotated = np.rot90(base, k=1)
        flipped = np.flip(base, axis=0)
        flipped_ud = np.flipud(base)
        flipped_lr = np.fliplr(base)
        return (
            flat[0]
            + 2.0 * flat[2]
            + np.sum(repeated)
            + np.sum(promoted)
            + np.sum(tiled)
            + np.sum(rolled)
            + np.sum(rotated)
            + np.sum(flipped)
            + np.sum(flipped_ud)
            + np.sum(flipped_lr)
        )

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
    )
    assert result.program_ir is not None
    analysis = analyze_program_ad_alias_effects(result.program_ir)
    view_edges = tuple(edge for edge in analysis.alias_edges if edge.kind == "view_alias")

    assert view_edges
    assert analysis.mutation_effects == ()
    assert all(alias_set.mutation_versions == () for alias_set in analysis.alias_sets)
    assert any(
        edge.source == "%array[1]" and edge.target.startswith("view:getitem")
        for edge in view_edges
    )
    assert any(edge.target.startswith("view:transpose") for edge in view_edges)
    assert any(edge.target.startswith("view:ravel") for edge in view_edges)
    assert any(edge.target.startswith("view:squeeze") for edge in view_edges)
    assert any(edge.target.startswith("view:expand_dims") for edge in view_edges)
    assert any(edge.target.startswith("view:swapaxes") for edge in view_edges)
    assert any(edge.target.startswith("view:moveaxis") for edge in view_edges)
    assert any(edge.target.startswith("view:repeat") for edge in view_edges)
    assert any(edge.target.startswith("view:atleast_3d") for edge in view_edges)
    assert any(edge.target.startswith("view:tile") for edge in view_edges)
    assert any(edge.target.startswith("view:roll") for edge in view_edges)
    assert any(edge.target.startswith("view:rot90") for edge in view_edges)
    assert any(edge.target.startswith("view:flip") for edge in view_edges)
    assert any(edge.target.startswith("view:flipud") for edge in view_edges)
    assert any(edge.target.startswith("view:fliplr") for edge in view_edges)
    assert any(
        "%array[1]" in alias_set.members
        and any(member.startswith("view:getitem") for member in alias_set.members)
        and any(member.startswith("view:ravel") for member in alias_set.members)
        for alias_set in analysis.alias_sets
    )
    assert any(
        "%array[1]" in alias_set.members
        and any(member.startswith("view:repeat") for member in alias_set.members)
        for alias_set in analysis.alias_sets
    )
    assert any(
        "%array[1]" in alias_set.members
        and any(member.startswith("view:roll") for member in alias_set.members)
        and any(member.startswith("view:fliplr") for member in alias_set.members)
        for alias_set in analysis.alias_sets
    )


def test_program_ad_alias_effect_analysis_tracks_static_slice_mutation() -> None:
    """Program AD slice mutation should preserve gradients and source metadata."""

    def objective(values: Any) -> object:
        window = values.reshape((6,))[1:5]
        window[1:3] = np.array([2.0 * values[0], values[5] + 1.0])
        return window[0] + window[1] + window[2] + window[3]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
    )
    assert result.program_ir is not None
    analysis = analyze_program_ad_alias_effects(result.program_ir)

    mutation_edges = tuple(
        edge for edge in analysis.alias_edges if edge.kind == "mutation_version"
    )
    assert len(mutation_edges) == 2
    assert any(edge.source == "%array[2]" for edge in mutation_edges)
    assert any(edge.source == "%array[3]" for edge in mutation_edges)
    assert any(effect.kind == "mutation" for effect in result.program_ir.effects)
    assert any(edge.target.startswith("view:getitem") for edge in analysis.alias_edges)
    assert analysis.mutation_effects == tuple(sorted(analysis.mutation_effects))
    assert analysis.claim_boundary == "metadata_only_no_general_alias_lattice"
    np.testing.assert_allclose(result.gradient, [2.0, 1.0, 0.0, 0.0, 1.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_slice_mutation_fails_closed_for_unsupported_shapes() -> None:
    """Program AD slice mutation should reject non-rank-1 and length-mismatched writes."""

    with pytest.raises(ValueError, match="rank-1"):
        whole_program_value_and_grad(
            lambda values: _program_ad_rank_two_slice_write(values),
            np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="value length"):
        whole_program_value_and_grad(
            lambda values: _program_ad_slice_write_length_mismatch(values),
            np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        )


def _program_ad_rank_two_slice_write(values: Any) -> object:
    matrix = values.reshape((2, 2))
    matrix[0:1] = np.array([values[0]])
    return np.sum(matrix)


def _program_ad_slice_write_length_mismatch(values: Any) -> object:
    work = values.reshape((4,))
    work[1:3] = np.array([values[0]])
    return np.sum(work)


def test_program_ad_alias_effect_analysis_tracks_list_alias_rebinding() -> None:
    """Program AD source metadata should expose local list alias rebinding."""

    def objective(values: Any) -> object:
        scratch = [values[0], values[1]]
        alias = scratch
        alias[0] = values[2]
        return scratch[0] + 2.0 * alias[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    assert result.program_ir is not None
    analysis = analyze_program_ad_alias_effects(result.program_ir)
    list_edges = tuple(edge for edge in analysis.alias_edges if edge.kind == "list_alias")

    assert list_edges
    assert any(
        edge.source == "list:scratch" and edge.target == "name:alias" for edge in list_edges
    )
    assert any(
        edge.source == "list:scratch" and edge.target == "source:list_mutation"
        for edge in list_edges
    )
    assert any(
        "list:scratch" in alias_set.members
        and "name:alias" in alias_set.members
        and "source:list_mutation" in alias_set.members
        for alias_set in analysis.alias_sets
    )
    assert analysis.claim_boundary == "metadata_only_no_general_alias_lattice"
    np.testing.assert_allclose(result.gradient, [0.0, 2.0, 1.0], atol=1.0e-12)


def test_program_ad_alias_effect_analysis_tracks_local_scalar_rebinding() -> None:
    """Program AD source metadata should expose local scalar rebinding aliases."""

    def objective(values: Any) -> object:
        seed = values[0]
        rebound = seed
        return rebound + 3.0 * values[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b")),
    )
    assert result.program_ir is not None

    analysis = analyze_program_ad_alias_effects(result.program_ir)
    rebinding_edges = tuple(
        edge for edge in analysis.alias_edges if edge.kind == "local_rebinding_alias"
    )

    assert any(
        edge.source == "name:seed" and edge.target == "name:rebound" for edge in rebinding_edges
    )
    assert any(
        "name:seed" in alias_set.members and "name:rebound" in alias_set.members
        for alias_set in analysis.alias_sets
    )
    assert analysis.claim_boundary == "metadata_only_no_general_alias_lattice"
    np.testing.assert_allclose(result.gradient, [1.0, 3.0], atol=1.0e-12)


def test_program_ad_alias_effect_analysis_tracks_loop_carried_state() -> None:
    """Program AD source metadata should expose derivative-carrying loop state."""

    def objective(values: Any) -> object:
        carry = values[0]
        for index in range(1, 4):
            carry = carry + float(index) * values[index]
        return carry + values[4]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64),
        parameters=(
            Parameter("seed"),
            Parameter("step1"),
            Parameter("step2"),
            Parameter("step3"),
            Parameter("tail"),
        ),
    )
    assert result.program_ir is not None

    analysis = analyze_program_ad_alias_effects(result.program_ir)
    loop_edges = tuple(edge for edge in analysis.alias_edges if edge.kind == "loop_carried_state")

    assert loop_edges
    assert any(edge.source == "loop:carry:entry" for edge in loop_edges)
    assert any(edge.target == "loop:carry:backedge" for edge in loop_edges)
    assert any(
        "loop:carry:entry" in alias_set.members and "loop:carry:backedge" in alias_set.members
        for alias_set in analysis.alias_sets
    )
    assert any(
        phi.selected == "executed_loop_trace"
        and "loop_entry" in phi.incoming
        and "loop_backedge" in phi.incoming
        for phi in result.program_ir.phi_nodes
    )
    assert analysis.claim_boundary == "metadata_only_no_general_alias_lattice"
    np.testing.assert_allclose(result.gradient, [1.0, 1.0, 2.0, 3.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)
