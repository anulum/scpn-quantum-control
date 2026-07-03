# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable programming benchmark alias edge tests
"""Alias and static-lattice edge tests for differentiable-programming benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differentiable_programming_benchmark_edge_helpers import (
    _fake_whole_program,
    _program_ir,
    _whole_program_result,
)
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks import differentiable_programming as dp
from scpn_quantum_control.differentiable import (
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADSSAValue,
)


def _static_lattice_program_ir() -> ProgramADEffectIR:
    """Return real minimal Program AD IR for static-lattice benchmark guards."""

    return ProgramADEffectIR(
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
        effects=(
            ProgramADEffect(
                index=0,
                kind="pure",
                target="%0",
                inputs=(),
                version=0,
                ordering=0,
            ),
        ),
        alias_edges=(),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )


def test_alias_metadata_benchmarks_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alias metadata benchmarks should reject missing or incomplete alias evidence."""

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=None)),
    )
    with pytest.raises(ValueError, match="shape-view alias benchmark"):
        dp._shape_view_alias_metadata_case()
    with pytest.raises(ValueError, match="slice-mutation alias benchmark"):
        dp._slice_mutation_alias_metadata_case()
    with pytest.raises(ValueError, match="loop-carried state alias benchmark"):
        dp._loop_carried_state_alias_metadata_case()

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=_static_lattice_program_ir())),
    )
    monkeypatch.setattr(
        dp,
        "analyze_program_ad_alias_effects",
        lambda _ir: SimpleNamespace(alias_edges=(), mutation_effects=()),
    )
    with pytest.raises(ValueError, match="missing aliases"):
        dp._shape_view_alias_metadata_case()
    with pytest.raises(ValueError, match="source-index mutations"):
        dp._slice_mutation_alias_metadata_case()
    with pytest.raises(ValueError, match="carry backedge"):
        dp._loop_carried_state_alias_metadata_case()

    monkeypatch.setattr(
        dp,
        "analyze_program_ad_alias_effects",
        lambda _ir: SimpleNamespace(
            alias_edges=(
                SimpleNamespace(kind="mutation_version", source="%array[2]", target="%array[2]"),
                SimpleNamespace(kind="mutation_version", source="%array[3]", target="%array[3]"),
            ),
            mutation_effects=(0, 1),
        ),
    )
    with pytest.raises(ValueError, match="view alias metadata"):
        dp._slice_mutation_alias_metadata_case()

    monkeypatch.setattr(
        dp,
        "analyze_program_ad_alias_effects",
        lambda _ir: SimpleNamespace(
            alias_edges=(
                SimpleNamespace(kind="mutation_version", source="%array[2]", target="%array[2]"),
                SimpleNamespace(kind="mutation_version", source="%array[3]", target="%array[3]"),
                SimpleNamespace(kind="view_alias", source="x", target="view:getitem:0"),
            ),
            mutation_effects=(0,),
        ),
    )
    with pytest.raises(ValueError, match="two mutation effects"):
        dp._slice_mutation_alias_metadata_case()

    monkeypatch.setattr(
        dp,
        "analyze_program_ad_alias_effects",
        lambda _ir: SimpleNamespace(
            alias_edges=(
                SimpleNamespace(
                    kind="loop_carried_state",
                    source="loop:carry:entry",
                    target="loop:carry:backedge",
                ),
            ),
            mutation_effects=(),
        ),
    )
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(
            _whole_program_result(
                program_ir=_program_ir(
                    phi_nodes=(
                        SimpleNamespace(
                            target="phi:runtime_branch:0",
                            selected="executed_true",
                            control_region=0,
                            incoming=("true", "false"),
                        ),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="loop phi metadata"):
        dp._loop_carried_state_alias_metadata_case()


def test_static_alias_lattice_benchmark_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static alias-lattice benchmark should reject incomplete lattice evidence."""

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=None)),
    )
    with pytest.raises(ValueError, match="requires Program AD IR"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=_static_lattice_program_ir())),
    )
    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(complete=False),
    )
    with pytest.raises(ValueError, match="complete emitted-IR report"):
        dp._static_alias_lattice_report_case()

    report_without_view = SimpleNamespace(
        complete=True,
        components=(
            SimpleNamespace(
                edge_kinds=("object_attribute_alias",),
                members=("attr:scratch.left", "attr:scratch.total", "object:scratch"),
            ),
            SimpleNamespace(
                edge_kinds=("expression_rebinding_alias",), members=("name:combined",)
            ),
        ),
    )
    monkeypatch.setattr(
        dp, "program_ad_static_alias_lattice_report", lambda _ir: report_without_view
    )
    with pytest.raises(ValueError, match="view-alias component"):
        dp._static_alias_lattice_report_case()

    view_component = SimpleNamespace(
        edge_kinds=("view_alias",),
        members=("%array[0]", "view:transpose:0"),
    )
    view_provenance = SimpleNamespace(
        source="%array[0]",
        target="view:transpose:0[0]",
        operation="transpose",
        view_id=0,
        output_index=0,
        version=0,
    )
    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(complete=True, components=(view_component,)),
    )
    with pytest.raises(ValueError, match="object-attribute component"):
        dp._static_alias_lattice_report_case()

    object_component = SimpleNamespace(
        edge_kinds=("object_attribute_alias",),
        members=("attr:scratch.left", "attr:scratch.total", "object:scratch"),
    )
    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(view_component, object_component),
        ),
    )
    with pytest.raises(ValueError, match="expression-rebinding component"):
        dp._static_alias_lattice_report_case()

    expression_component = SimpleNamespace(
        edge_kinds=("expression_rebinding_alias",),
        members=("name:combined",),
    )
    local_rebinding_component = SimpleNamespace(
        edge_kinds=("local_rebinding_alias",),
        members=("name:combined", "name:direct"),
    )
    rebinding_provenance = SimpleNamespace(
        source="expr:42:scratch.left+2.0*scratch.right",
        target="name:combined",
        binding_kind="expression",
        source_name=None,
        expression_line=42,
        expression_label="scratch.left+2.0*scratch.right",
        target_name="combined",
        version=0,
    )
    local_rebinding_provenance = SimpleNamespace(
        source="name:combined",
        target="name:direct",
        binding_kind="local",
        source_name="combined",
        expression_line=None,
        expression_label=None,
        target_name="direct",
        version=1,
    )
    list_component = SimpleNamespace(
        edge_kinds=("list_alias",),
        members=("list:scratch_list", "name:list_alias", "source:list_mutation"),
    )
    list_provenance = SimpleNamespace(
        source="list:scratch_list",
        target="name:list_alias",
        list_name="scratch_list",
        target_kind="local_name",
        version=0,
    )
    list_mutation_provenance = SimpleNamespace(
        source="list:scratch_list",
        target="source:list_mutation",
        list_name="scratch_list",
        target_kind="indexed_mutation_source",
        version=1,
    )
    loop_component = SimpleNamespace(
        edge_kinds=("loop_carried_state",),
        members=("loop:carry:backedge", "loop:carry:entry"),
    )
    loop_provenance = SimpleNamespace(
        source="loop:carry:entry",
        target="loop:carry:backedge",
        state_name="carry",
        entry_label="entry",
        backedge_label="backedge",
        version=0,
    )
    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(view_component, object_component, expression_component),
        ),
    )
    with pytest.raises(ValueError, match="local-rebinding component"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(
                view_component,
                object_component,
                expression_component,
                local_rebinding_component,
            ),
            view_alias_provenance=(),
        ),
    )
    with pytest.raises(ValueError, match="view-alias provenance"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(
                view_component,
                object_component,
                expression_component,
                local_rebinding_component,
                list_component,
                loop_component,
            ),
            view_alias_provenance=(view_provenance,),
            malformed_view_alias_edges=(),
            rebinding_alias_provenance=(),
            malformed_rebinding_alias_edges=(),
            list_alias_provenance=(list_provenance, list_mutation_provenance),
            malformed_list_alias_edges=(),
            loop_carried_state_provenance=(loop_provenance,),
            malformed_loop_carried_state_edges=(),
        ),
    )
    with pytest.raises(ValueError, match="rebinding provenance"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(
                view_component,
                object_component,
                expression_component,
                local_rebinding_component,
            ),
            view_alias_provenance=(view_provenance,),
            malformed_view_alias_edges=(),
            rebinding_alias_provenance=(rebinding_provenance, local_rebinding_provenance),
            malformed_rebinding_alias_edges=(),
            list_alias_provenance=(),
            malformed_list_alias_edges=(),
        ),
    )
    with pytest.raises(ValueError, match="list-alias component"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(
                view_component,
                object_component,
                expression_component,
                local_rebinding_component,
                list_component,
            ),
            view_alias_provenance=(view_provenance,),
            malformed_view_alias_edges=(),
            rebinding_alias_provenance=(rebinding_provenance, local_rebinding_provenance),
            malformed_rebinding_alias_edges=(),
            list_alias_provenance=(),
            malformed_list_alias_edges=(),
        ),
    )
    with pytest.raises(ValueError, match="list-alias provenance"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(
                view_component,
                object_component,
                expression_component,
                local_rebinding_component,
                list_component,
            ),
            view_alias_provenance=(view_provenance,),
            malformed_view_alias_edges=(),
            rebinding_alias_provenance=(rebinding_provenance, local_rebinding_provenance),
            malformed_rebinding_alias_edges=(),
            list_alias_provenance=(list_provenance, list_mutation_provenance),
            malformed_list_alias_edges=(),
            loop_carried_state_provenance=(),
            malformed_loop_carried_state_edges=(),
        ),
    )
    with pytest.raises(ValueError, match="loop-carried state component"):
        dp._static_alias_lattice_report_case()

    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir: SimpleNamespace(
            complete=True,
            components=(
                view_component,
                object_component,
                expression_component,
                local_rebinding_component,
                list_component,
                loop_component,
            ),
            view_alias_provenance=(view_provenance,),
            malformed_view_alias_edges=(),
            rebinding_alias_provenance=(rebinding_provenance, local_rebinding_provenance),
            malformed_rebinding_alias_edges=(),
            list_alias_provenance=(list_provenance, list_mutation_provenance),
            malformed_list_alias_edges=(),
            loop_carried_state_provenance=(),
            malformed_loop_carried_state_edges=(),
        ),
    )
    with pytest.raises(ValueError, match="loop-carried state provenance"):
        dp._static_alias_lattice_report_case()

    unsupported_semantics = ("filtered_comprehension",)
    unsupported_diagnostic = SimpleNamespace(
        semantic="filtered_comprehension",
        detail="filtered_comprehension",
        region_ids=("body",),
        bytecode_offsets=(8,),
    )
    object_attribute_semantics = ("object_attribute",)
    object_attribute_diagnostic = SimpleNamespace(
        semantic="object_attribute",
        detail="object_attribute:captured",
        region_ids=("body",),
        bytecode_offsets=(12,),
    )
    unknown_alias_edge = SimpleNamespace(
        source="runtime:dynamic_object",
        target="%0",
        kind="runtime_unknown_alias",
        version=0,
    )
    control_provenance = SimpleNamespace(
        source="control:if:42:body",
        target="control:attr:scratch.value",
        branch_line=42,
        branch_arm="body",
        target_label="attr:scratch.value",
        version=0,
    )

    def fake_frontend(objective: Callable[[Any], object]) -> SimpleNamespace:
        if objective.__name__ == "unsupported_object_attribute_boundary":
            return SimpleNamespace(
                semantics_report=SimpleNamespace(
                    unsupported_python_semantics=object_attribute_semantics,
                ),
                unsupported_semantic_diagnostics=(object_attribute_diagnostic,),
            )
        return SimpleNamespace(
            semantics_report=SimpleNamespace(
                unsupported_python_semantics=unsupported_semantics,
            ),
            unsupported_semantic_diagnostics=(unsupported_diagnostic,),
        )

    monkeypatch.setattr(dp, "compile_whole_program_frontend", fake_frontend)
    complete_report = SimpleNamespace(
        complete=True,
        components=(
            view_component,
            object_component,
            expression_component,
            local_rebinding_component,
            list_component,
            loop_component,
        ),
        view_alias_provenance=(view_provenance,),
        malformed_view_alias_edges=(),
        rebinding_alias_provenance=(rebinding_provenance, local_rebinding_provenance),
        malformed_rebinding_alias_edges=(),
        list_alias_provenance=(list_provenance, list_mutation_provenance),
        malformed_list_alias_edges=(),
        loop_carried_state_provenance=(loop_provenance,),
        malformed_loop_carried_state_edges=(),
    )
    mutation_blocked_report = SimpleNamespace(
        complete=False,
        mutation_effects=(0,),
        blocker_reasons=("mutation_effects_require_versioned_alias_semantics",),
    )
    unsupported_blocked_report = SimpleNamespace(
        complete=False,
        unsupported_python_semantics=unsupported_semantics,
        unsupported_semantic_diagnostics=(unsupported_diagnostic,),
        blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
    )
    object_attribute_blocked_report = SimpleNamespace(
        complete=False,
        unsupported_python_semantics=object_attribute_semantics,
        unsupported_semantic_diagnostics=(object_attribute_diagnostic,),
        unsupported_object_attribute_roots=("captured",),
        unsupported_object_attribute_details=("object_attribute:captured",),
        blocker_reasons=(
            "object_attributes_require_static_object_model",
            "unsupported_python_semantics_require_frontend_lowering",
        ),
    )
    unknown_alias_blocked_report = SimpleNamespace(
        complete=False,
        unknown_alias_edge_kinds=("runtime_unknown_alias",),
        unknown_alias_edges=(unknown_alias_edge,),
        blocker_reasons=("unknown_alias_edge_kinds",),
    )
    malformed_list_blocked_report = SimpleNamespace(
        complete=False,
        malformed_list_alias_edges=("scratch_list->name:list_alias:list_alias@0",),
        blocker_reasons=("list_alias_provenance_requires_parseable_targets",),
    )
    malformed_rebinding_blocked_report = SimpleNamespace(
        complete=False,
        malformed_rebinding_alias_edges=("combined->name:direct:local_rebinding_alias@0",),
        blocker_reasons=("rebinding_alias_provenance_requires_parseable_targets",),
    )
    malformed_loop_blocked_report = SimpleNamespace(
        complete=False,
        malformed_loop_carried_state_edges=(
            "loop:carry:start->loop:carry:backedge:loop_carried_state@0",
        ),
        blocker_reasons=("loop_carried_state_provenance_requires_parseable_targets",),
    )
    for unsupported_report, match in (
        (
            SimpleNamespace(
                complete=True,
                unsupported_python_semantics=unsupported_semantics,
                unsupported_semantic_diagnostics=(unsupported_diagnostic,),
                blocker_reasons=(),
            ),
            "must not promote unsupported semantics",
        ),
        (
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=(),
                unsupported_semantic_diagnostics=(unsupported_diagnostic,),
                blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            ),
            "lost unsupported frontend semantics",
        ),
        (
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=unsupported_semantics,
                unsupported_semantic_diagnostics=(),
                blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            ),
            "lost unsupported frontend diagnostics",
        ),
        (
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=unsupported_semantics,
                unsupported_semantic_diagnostics=(unsupported_diagnostic,),
                blocker_reasons=(),
            ),
            "unsupported-semantics blocker",
        ),
    ):
        reports = iter((complete_report, unsupported_report))
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()

    for object_attribute_report, match in (
        (
            SimpleNamespace(
                complete=True,
                unsupported_python_semantics=object_attribute_semantics,
                unsupported_semantic_diagnostics=(object_attribute_diagnostic,),
                unsupported_object_attribute_roots=("captured",),
                unsupported_object_attribute_details=("object_attribute:captured",),
                blocker_reasons=(),
            ),
            "must not promote captured/global object attributes",
        ),
        (
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=object_attribute_semantics,
                unsupported_semantic_diagnostics=(object_attribute_diagnostic,),
                unsupported_object_attribute_roots=(),
                unsupported_object_attribute_details=("object_attribute:captured",),
                blocker_reasons=(
                    "object_attributes_require_static_object_model",
                    "unsupported_python_semantics_require_frontend_lowering",
                ),
            ),
            "lost captured/global object-attribute roots",
        ),
        (
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=object_attribute_semantics,
                unsupported_semantic_diagnostics=(object_attribute_diagnostic,),
                unsupported_object_attribute_roots=("captured",),
                unsupported_object_attribute_details=(),
                blocker_reasons=(
                    "object_attributes_require_static_object_model",
                    "unsupported_python_semantics_require_frontend_lowering",
                ),
            ),
            "lost captured/global object-attribute diagnostics",
        ),
        (
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=object_attribute_semantics,
                unsupported_semantic_diagnostics=(object_attribute_diagnostic,),
                unsupported_object_attribute_roots=("captured",),
                unsupported_object_attribute_details=("object_attribute:captured",),
                blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            ),
            "object-attribute blocker",
        ),
    ):
        reports = iter((complete_report, unsupported_blocked_report, object_attribute_report))
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()

    for unknown_alias_report, match in (
        (
            SimpleNamespace(
                complete=True,
                unknown_alias_edge_kinds=("runtime_unknown_alias",),
                unknown_alias_edges=(unknown_alias_edge,),
                blocker_reasons=(),
            ),
            "must not promote unknown alias edges",
        ),
        (
            SimpleNamespace(
                complete=False,
                unknown_alias_edge_kinds=(),
                unknown_alias_edges=(unknown_alias_edge,),
                blocker_reasons=("unknown_alias_edge_kinds",),
            ),
            "lost unknown alias edge kinds",
        ),
        (
            SimpleNamespace(
                complete=False,
                unknown_alias_edge_kinds=("runtime_unknown_alias",),
                unknown_alias_edges=(),
                blocker_reasons=("unknown_alias_edge_kinds",),
            ),
            "lost unknown alias edge provenance",
        ),
        (
            SimpleNamespace(
                complete=False,
                unknown_alias_edge_kinds=("runtime_unknown_alias",),
                unknown_alias_edges=(unknown_alias_edge,),
                blocker_reasons=(),
            ),
            "unknown-alias blocker",
        ),
    ):
        reports = iter(
            (
                complete_report,
                unsupported_blocked_report,
                object_attribute_blocked_report,
                unknown_alias_report,
            )
        )
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()

    for malformed_list_report, match in (
        (
            SimpleNamespace(
                complete=True,
                malformed_list_alias_edges=("scratch_list->name:list_alias:list_alias@0",),
                blocker_reasons=(),
            ),
            "must not promote malformed list aliases",
        ),
        (
            SimpleNamespace(
                complete=False,
                malformed_list_alias_edges=(),
                blocker_reasons=("list_alias_provenance_requires_parseable_targets",),
            ),
            "lost malformed list-alias provenance",
        ),
        (
            SimpleNamespace(
                complete=False,
                malformed_list_alias_edges=("scratch_list->name:list_alias:list_alias@0",),
                blocker_reasons=(),
            ),
            "malformed-list blocker",
        ),
    ):
        reports = iter(
            (
                complete_report,
                unsupported_blocked_report,
                object_attribute_blocked_report,
                unknown_alias_blocked_report,
                malformed_list_report,
            )
        )
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()

    for malformed_rebinding_report, match in (
        (
            SimpleNamespace(
                complete=True,
                malformed_rebinding_alias_edges=("combined->name:direct:local_rebinding_alias@0",),
                blocker_reasons=(),
            ),
            "must not promote malformed rebinding aliases",
        ),
        (
            SimpleNamespace(
                complete=False,
                malformed_rebinding_alias_edges=(),
                blocker_reasons=("rebinding_alias_provenance_requires_parseable_targets",),
            ),
            "lost malformed rebinding-alias provenance",
        ),
        (
            SimpleNamespace(
                complete=False,
                malformed_rebinding_alias_edges=("combined->name:direct:local_rebinding_alias@0",),
                blocker_reasons=(),
            ),
            "malformed-rebinding blocker",
        ),
    ):
        reports = iter(
            (
                complete_report,
                unsupported_blocked_report,
                object_attribute_blocked_report,
                unknown_alias_blocked_report,
                malformed_list_blocked_report,
                malformed_rebinding_report,
            )
        )
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()

    for branch_report, match in (
        (
            SimpleNamespace(
                complete=True,
                blocker_reasons=(),
                non_executed_control_alias_edges=(),
                control_path_alias_provenance=(),
                malformed_control_path_alias_edges=(),
                components=(),
            ),
            "must not promote branch phi blockers",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=("control_path_aliases_require_branch_semantics",),
                non_executed_control_alias_edges=("edge",),
                control_path_alias_provenance=(control_provenance,),
                malformed_control_path_alias_edges=(),
                components=(object_component,),
            ),
            "non-executed phi blocker",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=("non_executed_phi_inputs_require_branch_semantics",),
                non_executed_control_alias_edges=("edge",),
                control_path_alias_provenance=(control_provenance,),
                malformed_control_path_alias_edges=(),
                components=(object_component,),
            ),
            "control-path alias blocker",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=(
                    "non_executed_phi_inputs_require_branch_semantics",
                    "control_path_aliases_require_branch_semantics",
                ),
                non_executed_control_alias_edges=(),
                control_path_alias_provenance=(control_provenance,),
                malformed_control_path_alias_edges=(),
                components=(object_component,),
            ),
            "control-path alias edges",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=(
                    "non_executed_phi_inputs_require_branch_semantics",
                    "control_path_aliases_require_branch_semantics",
                ),
                non_executed_control_alias_edges=("edge",),
                control_path_alias_provenance=(),
                malformed_control_path_alias_edges=(),
                components=(object_component,),
            ),
            "control-path alias provenance",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=(
                    "non_executed_phi_inputs_require_branch_semantics",
                    "control_path_aliases_require_branch_semantics",
                ),
                non_executed_control_alias_edges=("edge",),
                control_path_alias_provenance=(),
                malformed_control_path_alias_edges=(
                    "control:if:bad:body->control:attr:scratch.value:control_path_alias@0",
                ),
                components=(object_component,),
            ),
            "malformed-control blocker",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=(
                    "control_path_alias_provenance_requires_parseable_targets",
                    "non_executed_phi_inputs_require_branch_semantics",
                    "control_path_aliases_require_branch_semantics",
                ),
                non_executed_control_alias_edges=("edge",),
                control_path_alias_provenance=(control_provenance,),
                malformed_control_path_alias_edges=(
                    "control:if:bad:body->control:attr:scratch.value:control_path_alias@0",
                ),
                components=(object_component,),
            ),
            "malformed control-path aliases",
        ),
        (
            SimpleNamespace(
                complete=False,
                blocker_reasons=(
                    "non_executed_phi_inputs_require_branch_semantics",
                    "control_path_aliases_require_branch_semantics",
                ),
                non_executed_control_alias_edges=("edge",),
                control_path_alias_provenance=(control_provenance,),
                malformed_control_path_alias_edges=(),
                components=(),
            ),
            "attribute-path metadata",
        ),
    ):
        reports = iter(
            (
                complete_report,
                unsupported_blocked_report,
                object_attribute_blocked_report,
                unknown_alias_blocked_report,
                malformed_list_blocked_report,
                malformed_rebinding_blocked_report,
                malformed_loop_blocked_report,
                mutation_blocked_report,
                branch_report,
            )
        )
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()

    for malformed_loop_report, match in (
        (
            SimpleNamespace(
                complete=True,
                malformed_loop_carried_state_edges=(
                    "loop:carry:start->loop:carry:backedge:loop_carried_state@0",
                ),
                blocker_reasons=(),
            ),
            "must not promote malformed loop-carried state",
        ),
        (
            SimpleNamespace(
                complete=False,
                malformed_loop_carried_state_edges=(),
                blocker_reasons=("loop_carried_state_provenance_requires_parseable_targets",),
            ),
            "lost malformed loop-carried state provenance",
        ),
        (
            SimpleNamespace(
                complete=False,
                malformed_loop_carried_state_edges=(
                    "loop:carry:start->loop:carry:backedge:loop_carried_state@0",
                ),
                blocker_reasons=(),
            ),
            "malformed-loop-carried-state blocker",
        ),
    ):
        reports = iter(
            (
                complete_report,
                unsupported_blocked_report,
                object_attribute_blocked_report,
                unknown_alias_blocked_report,
                malformed_list_blocked_report,
                malformed_rebinding_blocked_report,
                malformed_loop_report,
            )
        )
        monkeypatch.setattr(
            dp,
            "program_ad_static_alias_lattice_report",
            lambda _ir, reports=reports, **_kwargs: next(reports),
        )
        with pytest.raises(ValueError, match=match):
            dp._static_alias_lattice_report_case()


def test_static_alias_lattice_branch_ir_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static alias-lattice benchmark should reject missing branch IR evidence."""

    view_component = SimpleNamespace(
        edge_kinds=("view_alias",),
        members=("%array[0]", "view:transpose:0"),
    )
    view_provenance = SimpleNamespace(
        source="%array[0]",
        target="view:transpose:0[0]",
        operation="transpose",
        view_id=0,
        output_index=0,
        version=0,
    )
    object_component = SimpleNamespace(
        edge_kinds=("object_attribute_alias",),
        members=("attr:scratch.left", "attr:scratch.total", "object:scratch"),
    )
    expression_component = SimpleNamespace(
        edge_kinds=("expression_rebinding_alias",),
        members=("name:combined",),
    )
    local_rebinding_component = SimpleNamespace(
        edge_kinds=("local_rebinding_alias",),
        members=("name:combined", "name:direct"),
    )
    rebinding_provenance = SimpleNamespace(
        source="expr:42:scratch.left+2.0*scratch.right",
        target="name:combined",
        binding_kind="expression",
        source_name=None,
        expression_line=42,
        expression_label="scratch.left+2.0*scratch.right",
        target_name="combined",
        version=0,
    )
    local_rebinding_provenance = SimpleNamespace(
        source="name:combined",
        target="name:direct",
        binding_kind="local",
        source_name="combined",
        expression_line=None,
        expression_label=None,
        target_name="direct",
        version=1,
    )
    list_component = SimpleNamespace(
        edge_kinds=("list_alias",),
        members=("list:scratch_list", "name:list_alias", "source:list_mutation"),
    )
    list_provenance = SimpleNamespace(
        source="list:scratch_list",
        target="name:list_alias",
        list_name="scratch_list",
        target_kind="local_name",
        version=0,
    )
    list_mutation_provenance = SimpleNamespace(
        source="list:scratch_list",
        target="source:list_mutation",
        list_name="scratch_list",
        target_kind="indexed_mutation_source",
        version=1,
    )
    loop_component = SimpleNamespace(
        edge_kinds=("loop_carried_state",),
        members=("loop:carry:backedge", "loop:carry:entry"),
    )
    loop_provenance = SimpleNamespace(
        source="loop:carry:entry",
        target="loop:carry:backedge",
        state_name="carry",
        entry_label="entry",
        backedge_label="backedge",
        version=0,
    )
    results = iter(
        (
            _whole_program_result(program_ir=_static_lattice_program_ir()),
            _whole_program_result(program_ir=_static_lattice_program_ir()),
            _whole_program_result(program_ir=None),
        ),
    )
    callback_values = iter(
        (
            None,
            None,
            np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float64),
        ),
    )

    def fake_whole_program(
        objective: Callable[[Any], object],
        _values: NDArray[np.float64],
        **_kwargs: object,
    ) -> SimpleNamespace:
        values = next(callback_values)
        if values is not None:
            objective(values)
        return next(results)

    monkeypatch.setattr(dp, "whole_program_value_and_grad", fake_whole_program)

    def fake_frontend(objective: Callable[[Any], object]) -> SimpleNamespace:
        if objective.__name__ == "unsupported_object_attribute_boundary":
            return SimpleNamespace(
                semantics_report=SimpleNamespace(
                    unsupported_python_semantics=("object_attribute",),
                ),
                unsupported_semantic_diagnostics=(
                    SimpleNamespace(
                        semantic="object_attribute",
                        detail="object_attribute:captured",
                        region_ids=("body",),
                        bytecode_offsets=(12,),
                    ),
                ),
            )
        return SimpleNamespace(
            semantics_report=SimpleNamespace(
                unsupported_python_semantics=("filtered_comprehension",),
            ),
            unsupported_semantic_diagnostics=(
                SimpleNamespace(
                    semantic="filtered_comprehension",
                    detail="filtered_comprehension",
                    region_ids=("body",),
                    bytecode_offsets=(8,),
                ),
            ),
        )

    monkeypatch.setattr(dp, "compile_whole_program_frontend", fake_frontend)
    reports = iter(
        (
            SimpleNamespace(
                complete=True,
                components=(
                    view_component,
                    object_component,
                    expression_component,
                    local_rebinding_component,
                    list_component,
                    loop_component,
                ),
                view_alias_provenance=(view_provenance,),
                malformed_view_alias_edges=(),
                rebinding_alias_provenance=(
                    rebinding_provenance,
                    local_rebinding_provenance,
                ),
                malformed_rebinding_alias_edges=(),
                list_alias_provenance=(list_provenance, list_mutation_provenance),
                malformed_list_alias_edges=(),
                loop_carried_state_provenance=(loop_provenance,),
                malformed_loop_carried_state_edges=(),
            ),
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=("filtered_comprehension",),
                unsupported_semantic_diagnostics=(
                    SimpleNamespace(
                        semantic="filtered_comprehension",
                        detail="filtered_comprehension",
                        region_ids=("body",),
                        bytecode_offsets=(8,),
                    ),
                ),
                blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
            ),
            SimpleNamespace(
                complete=False,
                unsupported_python_semantics=("object_attribute",),
                unsupported_semantic_diagnostics=(
                    SimpleNamespace(
                        semantic="object_attribute",
                        detail="object_attribute:captured",
                        region_ids=("body",),
                        bytecode_offsets=(12,),
                    ),
                ),
                unsupported_object_attribute_roots=("captured",),
                unsupported_object_attribute_details=("object_attribute:captured",),
                blocker_reasons=(
                    "object_attributes_require_static_object_model",
                    "unsupported_python_semantics_require_frontend_lowering",
                ),
            ),
            SimpleNamespace(
                complete=False,
                unknown_alias_edge_kinds=("runtime_unknown_alias",),
                unknown_alias_edges=(
                    SimpleNamespace(
                        source="runtime:dynamic_object",
                        target="%0",
                        kind="runtime_unknown_alias",
                        version=0,
                    ),
                ),
                blocker_reasons=("unknown_alias_edge_kinds",),
            ),
            SimpleNamespace(
                complete=False,
                malformed_list_alias_edges=("scratch_list->name:list_alias:list_alias@0",),
                blocker_reasons=("list_alias_provenance_requires_parseable_targets",),
            ),
            SimpleNamespace(
                complete=False,
                malformed_rebinding_alias_edges=("combined->name:direct:local_rebinding_alias@0",),
                blocker_reasons=("rebinding_alias_provenance_requires_parseable_targets",),
            ),
            SimpleNamespace(
                complete=False,
                malformed_loop_carried_state_edges=(
                    "loop:carry:start->loop:carry:backedge:loop_carried_state@0",
                ),
                blocker_reasons=("loop_carried_state_provenance_requires_parseable_targets",),
            ),
            SimpleNamespace(
                complete=False,
                mutation_effects=(0,),
                blocker_reasons=("mutation_effects_require_versioned_alias_semantics",),
            ),
        )
    )
    monkeypatch.setattr(
        dp,
        "program_ad_static_alias_lattice_report",
        lambda _ir, **_kwargs: next(reports),
    )

    with pytest.raises(ValueError, match="branch benchmark requires Program AD IR"):
        dp._static_alias_lattice_report_case()
