# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Static Alias Branch-IR Benchmark Edge Tests
"""Fail-closed static alias branch-ir evidence tests for differentiable benchmarks."""

from __future__ import annotations

from collections.abc import Callable
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
