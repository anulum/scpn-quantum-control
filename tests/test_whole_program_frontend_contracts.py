# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program frontend contract tests
"""Validation and public-identity tests for whole-program frontend contracts."""

from __future__ import annotations

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable import (
    compile_whole_program_frontend as facade_compile_whole_program_frontend,
)
from scpn_quantum_control.whole_program_frontend import (
    WholeProgramBytecodeBasicBlock,
    WholeProgramCompilerFrontendReport,
    WholeProgramSemanticsReport,
    WholeProgramSourceBytecodeLineMap,
    WholeProgramSourceRegion,
    WholeProgramSymbolScopeEntry,
    WholeProgramUnsupportedSemanticDiagnostic,
    compile_whole_program_frontend,
)


def test_whole_program_frontend_dataclasses_fail_closed() -> None:
    """Frontend value objects should reject inconsistent static metadata."""

    with pytest.raises(ValueError, match="instruction_offsets must be sorted"):
        WholeProgramBytecodeBasicBlock(
            label="bb",
            start_offset=1,
            end_offset=2,
            instruction_offsets=(2, 1),
            successor_offsets=(),
            terminating_opname="RETURN_VALUE",
        )

    with pytest.raises(ValueError, match="feature_kinds must be sorted and unique"):
        WholeProgramSourceRegion(
            region_id="region:bad",
            kind="entry",
            detail="module",
            line_start=1,
            line_end=1,
            parent_region_id=None,
            feature_kinds=("loop", "loop"),
        )

    with pytest.raises(ValueError, match="instruction_offsets must be sorted"):
        WholeProgramSourceBytecodeLineMap(
            line_number=1,
            absolute_line_number=1,
            instruction_offsets=(4, 2),
            region_ids=("region:entry",),
            feature_kinds=(),
        )

    with pytest.raises(ValueError, match="roles must be sorted and unique"):
        WholeProgramSymbolScopeEntry(
            symbol="values",
            roles=("parameter", "parameter"),
            line_numbers=(1,),
            bytecode_offsets=(),
            region_ids=("region:entry",),
        )

    with pytest.raises(ValueError, match="bytecode_offsets must be sorted"):
        WholeProgramUnsupportedSemanticDiagnostic(
            semantic="filtered_comprehension",
            detail="filtered_comprehension",
            line_number=1,
            absolute_line_number=10,
            region_ids=("region:entry",),
            bytecode_offsets=(4, 2),
        )

    with pytest.raises(ValueError, match="accepted_python_semantics entries"):
        WholeProgramSemanticsReport(
            bytecode_frontend=True,
            source_frontend=True,
            graph_capture=True,
            aliasing_observed=False,
            mutation_observed=False,
            loop_observed=False,
            control_flow_observed=False,
            numpy_observed=False,
            differentiation_semantics="bounded",
            accepted_python_semantics=("",),
            unsupported_python_semantics=(),
        )


def test_whole_program_frontend_exports_stay_crosswired() -> None:
    """Package-root, facade, and module exports should share object identity."""

    assert scpn.compile_whole_program_frontend is compile_whole_program_frontend
    assert facade_compile_whole_program_frontend is compile_whole_program_frontend
    assert scpn.WholeProgramBytecodeBasicBlock is WholeProgramBytecodeBasicBlock
    assert scpn.WholeProgramCompilerFrontendReport is WholeProgramCompilerFrontendReport
    assert scpn.WholeProgramSourceBytecodeLineMap is WholeProgramSourceBytecodeLineMap
    assert scpn.WholeProgramSourceRegion is WholeProgramSourceRegion
    assert scpn.WholeProgramSymbolScopeEntry is WholeProgramSymbolScopeEntry
    assert (
        scpn.WholeProgramUnsupportedSemanticDiagnostic is WholeProgramUnsupportedSemanticDiagnostic
    )
