# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable programming benchmark alias edges tests
# scpn-quantum-control -- differentiable programming benchmark alias edge tests
"""Alias and static-lattice edge tests for differentiable-programming benchmarks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _differentiable_programming_benchmark_edge_helpers import (
    _fake_whole_program,
    _program_ir,
    _static_lattice_program_ir,
    _whole_program_result,
)

from scpn_quantum_control.benchmarks import differentiable_programming as dp


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
