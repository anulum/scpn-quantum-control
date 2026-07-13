# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Static Alias-Lattice Provenance Guard Tests
"""Fail-closed static alias-lattice provenance and metadata guard tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _differentiable_programming_benchmark_edge_helpers import (
    _static_lattice_program_ir,
    _whole_program_result,
)
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks import differentiable_programming as dp


def _complete_report(scenario: str) -> SimpleNamespace:
    """Return complete primary lattice evidence, with one optional defect."""
    view_component = SimpleNamespace(
        edge_kinds=("view_alias",),
        members=("%array[0]", "view:transpose:0"),
    )
    object_component = SimpleNamespace(
        edge_kinds=("object_attribute_alias",),
        members=("object:scratch", "attr:scratch.left", "attr:scratch.total"),
    )
    expression_component = SimpleNamespace(
        edge_kinds=("expression_rebinding_alias",),
        members=("name:combined",),
    )
    local_component = SimpleNamespace(
        edge_kinds=("local_rebinding_alias",),
        members=("name:combined", "name:direct"),
    )
    list_component = SimpleNamespace(
        edge_kinds=("list_alias",),
        members=("list:scratch_list", "name:list_alias", "source:list_mutation"),
    )
    loop_component = SimpleNamespace(
        edge_kinds=("loop_carried_state",),
        members=("loop:carry:entry", "loop:carry:backedge"),
    )
    list_provenance = [
        SimpleNamespace(
            list_name="scratch_list",
            target_kind="local_name",
            source="list:scratch_list",
            target="name:list_alias",
        ),
        SimpleNamespace(
            list_name="scratch_list",
            target_kind="indexed_mutation_source",
            source="list:scratch_list",
            target="source:list_mutation",
        ),
    ]
    if scenario == "missing_list_mutation":
        list_provenance.pop()
    rebinding_provenance = [
        SimpleNamespace(
            binding_kind="expression",
            source_name=None,
            target_name="combined",
            source="expr:combined",
            target="name:combined",
        ),
        SimpleNamespace(
            binding_kind="local",
            source_name="combined",
            target_name="direct",
            source="name:combined",
            target="name:direct",
        ),
    ]
    if scenario == "missing_local_rebinding":
        rebinding_provenance.pop()
    return SimpleNamespace(
        complete=True,
        components=(
            view_component,
            object_component,
            expression_component,
            local_component,
            list_component,
            loop_component,
        ),
        view_alias_provenance=(
            SimpleNamespace(
                operation="transpose",
                source="%array[0]",
                target="view:transpose:0",
            ),
        ),
        malformed_view_alias_edges=("bad-view",) if scenario == "malformed_view" else (),
        list_alias_provenance=tuple(list_provenance),
        malformed_list_alias_edges=("bad-list",) if scenario == "malformed_list" else (),
        loop_carried_state_provenance=(
            SimpleNamespace(
                state_name="carry",
                entry_label="entry",
                backedge_label="backedge",
                source="loop:carry:entry",
                target="loop:carry:backedge",
            ),
        ),
        malformed_loop_carried_state_edges=(("bad-loop",) if scenario == "malformed_loop" else ()),
        rebinding_alias_provenance=tuple(rebinding_provenance),
        malformed_rebinding_alias_edges=(
            ("bad-rebinding",) if scenario == "malformed_rebinding" else ()
        ),
    )


def _install_lattice_scenario(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    """Install deterministic dependency shims for one fail-closed guard."""
    program_ir = _static_lattice_program_ir()
    alias_ir = replace(program_ir, ssa_values=()) if scenario == "missing_ssa" else program_ir
    mutation_ir = None if scenario == "missing_mutation_ir" else program_ir
    whole_program_results = iter(
        (
            _whole_program_result(program_ir=alias_ir),
            _whole_program_result(program_ir=mutation_ir),
            _whole_program_result(program_ir=program_ir),
        )
    )
    trace_values = np.array([0.25, -0.5, 0.75, 1.0], dtype=np.float64)

    def fake_whole_program(
        objective: Callable[[Any], object],
        _values: NDArray[np.float64],
        **_kwargs: object,
    ) -> SimpleNamespace:
        objective(trace_values)
        return next(whole_program_results)

    monkeypatch.setattr(dp, "whole_program_value_and_grad", fake_whole_program)

    unsupported_semantics = (
        () if scenario == "missing_unsupported_semantics" else ("filtered_comprehension",)
    )
    unsupported_diagnostics = (
        ()
        if scenario == "missing_unsupported_diagnostic"
        else (
            SimpleNamespace(
                semantic="filtered_comprehension",
                detail="filtered_comprehension",
                region_ids=("body",),
                bytecode_offsets=(8,),
            ),
        )
    )
    object_semantics = () if scenario == "missing_object_semantics" else ("object_attribute",)
    object_diagnostics = (
        ()
        if scenario == "missing_object_diagnostic"
        else (
            SimpleNamespace(
                semantic="object_attribute",
                detail="object_attribute:captured",
                region_ids=("body",),
                bytecode_offsets=(12,),
            ),
        )
    )

    def fake_frontend(objective: Callable[[Any], object]) -> SimpleNamespace:
        objective(trace_values)
        is_object_boundary = objective.__name__ == "unsupported_object_attribute_boundary"
        semantics = object_semantics if is_object_boundary else unsupported_semantics
        diagnostics = object_diagnostics if is_object_boundary else unsupported_diagnostics
        return SimpleNamespace(
            semantics_report=SimpleNamespace(unsupported_python_semantics=semantics),
            unsupported_semantic_diagnostics=diagnostics,
        )

    monkeypatch.setattr(dp, "compile_whole_program_frontend", fake_frontend)

    unsupported_report = SimpleNamespace(
        complete=False,
        unsupported_python_semantics=("filtered_comprehension",),
        unsupported_semantic_diagnostics=unsupported_diagnostics,
        blocker_reasons=("unsupported_python_semantics_require_frontend_lowering",),
    )
    object_report = SimpleNamespace(
        complete=False,
        unsupported_python_semantics=("object_attribute",),
        unsupported_semantic_diagnostics=object_diagnostics,
        unsupported_object_attribute_roots=("captured",),
        unsupported_object_attribute_details=("object_attribute:captured",),
        blocker_reasons=(
            "object_attributes_require_static_object_model",
            "unsupported_python_semantics_require_frontend_lowering",
        ),
    )
    unknown_edge = SimpleNamespace(
        source="runtime:dynamic_object",
        target="wrong-target" if scenario == "corrupt_unknown" else "%0",
        kind="runtime_unknown_alias",
        version=0,
    )
    unknown_report = SimpleNamespace(
        complete=False,
        unknown_alias_edge_kinds=("runtime_unknown_alias",),
        unknown_alias_edges=(unknown_edge,),
        blocker_reasons=("unknown_alias_edge_kinds",),
    )
    malformed_list_report = SimpleNamespace(
        complete=False,
        malformed_list_alias_edges=("scratch_list->name:list_alias:list_alias@0",),
        blocker_reasons=("list_alias_provenance_requires_parseable_targets",),
    )
    malformed_rebinding_report = SimpleNamespace(
        complete=False,
        malformed_rebinding_alias_edges=("combined->name:direct:local_rebinding_alias@0",),
        blocker_reasons=("rebinding_alias_provenance_requires_parseable_targets",),
    )
    malformed_loop_report = SimpleNamespace(
        complete=False,
        malformed_loop_carried_state_edges=(
            "loop:carry:start->loop:carry:backedge:loop_carried_state@0",
        ),
        blocker_reasons=("loop_carried_state_provenance_requires_parseable_targets",),
    )
    mutation_report = SimpleNamespace(
        complete=scenario == "promoted_mutation",
        mutation_effects=() if scenario == "missing_mutation_effects" else (0,),
        blocker_reasons=(
            ()
            if scenario == "missing_mutation_blocker"
            else ("mutation_effects_require_versioned_alias_semantics",)
        ),
    )
    control_provenance = SimpleNamespace(
        target_label=(
            "attr:other.value" if scenario == "missing_branch_attribute" else "attr:scratch.value"
        ),
    )
    branch_report = SimpleNamespace(
        complete=False,
        blocker_reasons=(
            "non_executed_phi_inputs_require_branch_semantics",
            "control_path_aliases_require_branch_semantics",
        ),
        non_executed_control_alias_edges=("edge",),
        control_path_alias_provenance=(control_provenance,),
        malformed_control_path_alias_edges=(),
        components=(
            SimpleNamespace(
                edge_kinds=("object_attribute_alias",),
                members=("attr:scratch.value",),
            ),
        ),
    )
    reports = iter(
        (
            _complete_report(scenario),
            unsupported_report,
            object_report,
            unknown_report,
            malformed_list_report,
            malformed_rebinding_report,
            malformed_loop_report,
            mutation_report,
            branch_report,
        )
    )

    def fake_lattice_report(_program_ir: object, **_kwargs: object) -> SimpleNamespace:
        return next(reports)

    monkeypatch.setattr(dp, "program_ad_static_alias_lattice_report", fake_lattice_report)


@pytest.mark.parametrize(
    ("scenario", "match"),
    (
        ("malformed_view", "malformed view-alias edges"),
        ("missing_list_mutation", "list-mutation provenance"),
        ("malformed_list", "malformed list-alias edges"),
        ("malformed_loop", "malformed loop-carried state edges"),
        ("missing_local_rebinding", "local rebinding provenance"),
        ("malformed_rebinding", "malformed rebinding-alias edges"),
        ("missing_unsupported_semantics", "unsupported frontend semantics"),
        ("missing_unsupported_diagnostic", "unsupported frontend diagnostics"),
        ("missing_object_semantics", "object-attribute frontend semantics"),
        ("missing_object_diagnostic", "object-attribute frontend diagnostics"),
        ("missing_ssa", "missing SSA values"),
        ("corrupt_unknown", "corrupted unknown alias edge provenance"),
        ("missing_mutation_ir", "mutation benchmark requires Program AD IR"),
        ("promoted_mutation", "must not promote mutation effects"),
        ("missing_mutation_effects", "missing mutation effects"),
        ("missing_mutation_blocker", "missing mutation blocker"),
        ("missing_branch_attribute", "missing attribute-path metadata"),
    ),
)
def test_static_alias_lattice_provenance_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    match: str,
) -> None:
    """Every lattice provenance guard rejects its incomplete evidence shape."""
    _install_lattice_scenario(monkeypatch, scenario)

    with pytest.raises(ValueError, match=match):
        dp._static_alias_lattice_report_case()
