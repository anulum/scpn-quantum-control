# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program AD frontend inspection
"""Static bytecode/source frontend inspection for whole-program Program AD.

This module owns no-execution compiler-fronted metadata for Python objectives:
bytecode rows, source-region summaries, symbol scopes, semantic diagnostics,
and deterministic report digests. Runtime operator interception remains in
``scpn_quantum_control.differentiable``; this module is the static preflight
boundary used by that facade and by package-root exports.

Module size note: this module is intentionally kept whole. Its top-level definitions form a single connected compiler-frontend cluster, so it is sized by responsibility rather than line count. See ``docs/architecture.md`` ("Module size and single-responsibility policy").
"""

from __future__ import annotations

import ast
import dis
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WholeProgramBytecodeInstruction:
    """One Python bytecode instruction captured for whole-program AD frontend IR."""

    offset: int
    opname: str
    argrepr: str
    line_number: int | None
    jump_target_offset: int | None = None

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("bytecode instruction offset must be non-negative")
        if not self.opname:
            raise ValueError("bytecode instruction opname must be non-empty")
        if not isinstance(self.argrepr, str):
            raise ValueError("bytecode instruction argrepr must be a string")
        if self.line_number is not None and self.line_number <= 0:
            raise ValueError("bytecode instruction line_number must be positive or None")
        if self.jump_target_offset is not None and self.jump_target_offset < 0:
            raise ValueError("bytecode instruction jump_target_offset must be non-negative")


@dataclass(frozen=True)
class _ObjectiveSourceMetadata:
    """Source text and absolute file-line bounds for a Python callable."""

    source: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("objective source metadata source must be non-empty")
        if self.start_line <= 0:
            raise ValueError("objective source metadata start_line must be positive")
        if self.end_line < self.start_line:
            raise ValueError("objective source metadata end_line must be >= start_line")


@dataclass(frozen=True)
class WholeProgramSourceIRFeature:
    """One source-level semantic feature captured for whole-program AD."""

    kind: str
    detail: str
    line_number: int

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("source IR feature kind must be non-empty")
        if not self.detail:
            raise ValueError("source IR feature detail must be non-empty")
        if self.line_number <= 0:
            raise ValueError("source IR feature line_number must be positive")


@dataclass(frozen=True)
class WholeProgramBytecodeBasicBlock:
    """One static Python-bytecode basic block for frontend planning.

    The block is derived from normalized ``dis`` instructions without executing
    the objective. Successors are bytecode offsets only; they are a static
    control-flow skeleton for audits and later lowerings, not executable
    compiler evidence.
    """

    label: str
    start_offset: int
    end_offset: int
    instruction_offsets: tuple[int, ...]
    successor_offsets: tuple[int, ...]
    terminating_opname: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("bytecode basic block label must be non-empty")
        if self.start_offset < 0 or self.end_offset < 0:
            raise ValueError("bytecode basic block offsets must be non-negative")
        if self.end_offset < self.start_offset:
            raise ValueError("bytecode basic block end_offset must be >= start_offset")
        if not self.instruction_offsets:
            raise ValueError("bytecode basic block instruction_offsets must be non-empty")
        if tuple(sorted(self.instruction_offsets)) != self.instruction_offsets:
            raise ValueError("bytecode basic block instruction_offsets must be sorted")
        if self.instruction_offsets[0] != self.start_offset:
            raise ValueError("bytecode basic block start_offset must match first instruction")
        if self.instruction_offsets[-1] != self.end_offset:
            raise ValueError("bytecode basic block end_offset must match last instruction")
        if any(offset < 0 for offset in self.successor_offsets):
            raise ValueError("bytecode basic block successor_offsets must be non-negative")
        if tuple(sorted(set(self.successor_offsets))) != self.successor_offsets:
            raise ValueError("bytecode basic block successor_offsets must be sorted and unique")
        if not self.terminating_opname:
            raise ValueError("bytecode basic block terminating_opname must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready bytecode block."""

        return {
            "label": self.label,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "instruction_offsets": list(self.instruction_offsets),
            "successor_offsets": list(self.successor_offsets),
            "terminating_opname": self.terminating_opname,
        }


@dataclass(frozen=True)
class WholeProgramSourceRegion:
    """One static source-region node for frontend planning.

    Regions summarize bounded AST constructs such as function entry, control
    flow, loops, alias bindings, and mutations. They are deterministic source
    metadata for the bytecode/source frontend and do not imply non-executed
    branch adjoints or executable compiler lowering.
    """

    region_id: str
    kind: str
    detail: str
    line_start: int
    line_end: int
    parent_region_id: str | None
    feature_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("source region region_id must be non-empty")
        if not self.kind:
            raise ValueError("source region kind must be non-empty")
        if not self.detail:
            raise ValueError("source region detail must be non-empty")
        if self.line_start <= 0 or self.line_end <= 0:
            raise ValueError("source region line numbers must be positive")
        if self.line_end < self.line_start:
            raise ValueError("source region line_end must be >= line_start")
        if self.parent_region_id is not None and not self.parent_region_id:
            raise ValueError("source region parent_region_id must be non-empty or None")
        if any(not feature for feature in self.feature_kinds):
            raise ValueError("source region feature_kinds entries must be non-empty")
        if tuple(sorted(set(self.feature_kinds))) != self.feature_kinds:
            raise ValueError("source region feature_kinds must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready source region."""

        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "detail": self.detail,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "parent_region_id": self.parent_region_id,
            "feature_kinds": list(self.feature_kinds),
        }


@dataclass(frozen=True)
class WholeProgramSourceBytecodeLineMap:
    """One source-line to bytecode crosswalk row for frontend planning.

    The row links a Python source line to normalized bytecode offsets, source
    regions, and feature kinds. It is static inspection metadata only; it does
    not assert executable compiler lowering or non-executed branch adjoints.
    """

    line_number: int
    absolute_line_number: int | None
    instruction_offsets: tuple[int, ...]
    region_ids: tuple[str, ...]
    feature_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.line_number <= 0:
            raise ValueError("source-bytecode line map line_number must be positive")
        if self.absolute_line_number is not None and self.absolute_line_number <= 0:
            raise ValueError(
                "source-bytecode line map absolute_line_number must be positive or None"
            )
        if not self.instruction_offsets:
            raise ValueError("source-bytecode line map instruction_offsets must be non-empty")
        if tuple(sorted(self.instruction_offsets)) != self.instruction_offsets:
            raise ValueError("source-bytecode line map instruction_offsets must be sorted")
        if any(offset < 0 for offset in self.instruction_offsets):
            raise ValueError("source-bytecode line map instruction_offsets must be non-negative")
        if any(not region_id for region_id in self.region_ids):
            raise ValueError("source-bytecode line map region_ids entries must be non-empty")
        if tuple(sorted(set(self.region_ids))) != self.region_ids:
            raise ValueError("source-bytecode line map region_ids must be sorted and unique")
        if any(not feature for feature in self.feature_kinds):
            raise ValueError("source-bytecode line map feature_kinds entries must be non-empty")
        if tuple(sorted(set(self.feature_kinds))) != self.feature_kinds:
            raise ValueError("source-bytecode line map feature_kinds must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready source-bytecode crosswalk row."""

        return {
            "line_number": self.line_number,
            "absolute_line_number": self.absolute_line_number,
            "instruction_offsets": list(self.instruction_offsets),
            "region_ids": list(self.region_ids),
            "feature_kinds": list(self.feature_kinds),
        }


@dataclass(frozen=True)
class WholeProgramSymbolScopeEntry:
    """One static symbol-scope entry for whole-program frontend diagnostics.

    Entries merge source names, bytecode operands, function parameters, locals,
    globals, closure variables, and cell variables into a deterministic symbol
    table. The table is a compiler-frontend diagnostic and does not execute the
    objective or prove runtime alias safety.
    """

    symbol: str
    roles: tuple[str, ...]
    line_numbers: tuple[int, ...]
    bytecode_offsets: tuple[int, ...]
    region_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol-scope entry symbol must be non-empty")
        if not self.roles:
            raise ValueError("symbol-scope entry roles must be non-empty")
        if tuple(sorted(set(self.roles))) != self.roles:
            raise ValueError("symbol-scope entry roles must be sorted and unique")
        if any(not role for role in self.roles):
            raise ValueError("symbol-scope entry roles entries must be non-empty")
        if any(line_number <= 0 for line_number in self.line_numbers):
            raise ValueError("symbol-scope entry line_numbers must be positive")
        if tuple(sorted(set(self.line_numbers))) != self.line_numbers:
            raise ValueError("symbol-scope entry line_numbers must be sorted and unique")
        if any(offset < 0 for offset in self.bytecode_offsets):
            raise ValueError("symbol-scope entry bytecode_offsets must be non-negative")
        if tuple(sorted(set(self.bytecode_offsets))) != self.bytecode_offsets:
            raise ValueError("symbol-scope entry bytecode_offsets must be sorted and unique")
        if any(not region_id for region_id in self.region_ids):
            raise ValueError("symbol-scope entry region_ids entries must be non-empty")
        if tuple(sorted(set(self.region_ids))) != self.region_ids:
            raise ValueError("symbol-scope entry region_ids must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready symbol-scope entry."""

        return {
            "symbol": self.symbol,
            "roles": list(self.roles),
            "line_numbers": list(self.line_numbers),
            "bytecode_offsets": list(self.bytecode_offsets),
            "region_ids": list(self.region_ids),
        }


@dataclass(frozen=True)
class WholeProgramUnsupportedSemanticDiagnostic:
    """One fail-closed Python-semantics diagnostic for frontend audits.

    The diagnostic binds an unsupported source construct to source-relative
    lines, optional CPython/file lines, source regions, and bytecode offsets.
    It is static preflight metadata only and does not execute or lower the
    blocked construct.
    """

    semantic: str
    detail: str
    line_number: int
    absolute_line_number: int | None
    region_ids: tuple[str, ...]
    bytecode_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.semantic:
            raise ValueError("unsupported semantic diagnostic semantic must be non-empty")
        if not self.detail:
            raise ValueError("unsupported semantic diagnostic detail must be non-empty")
        if self.line_number <= 0:
            raise ValueError("unsupported semantic diagnostic line_number must be positive")
        if self.absolute_line_number is not None and self.absolute_line_number <= 0:
            raise ValueError(
                "unsupported semantic diagnostic absolute_line_number must be positive or None"
            )
        if any(not region_id for region_id in self.region_ids):
            raise ValueError(
                "unsupported semantic diagnostic region_ids entries must be non-empty"
            )
        if tuple(sorted(set(self.region_ids))) != self.region_ids:
            raise ValueError(
                "unsupported semantic diagnostic region_ids must be sorted and unique"
            )
        if any(offset < 0 for offset in self.bytecode_offsets):
            raise ValueError(
                "unsupported semantic diagnostic bytecode_offsets must be non-negative"
            )
        if tuple(sorted(set(self.bytecode_offsets))) != self.bytecode_offsets:
            raise ValueError(
                "unsupported semantic diagnostic bytecode_offsets must be sorted and unique"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready unsupported-semantics diagnostic."""

        return {
            "semantic": self.semantic,
            "detail": self.detail,
            "line_number": self.line_number,
            "absolute_line_number": self.absolute_line_number,
            "region_ids": list(self.region_ids),
            "bytecode_offsets": list(self.bytecode_offsets),
        }


@dataclass(frozen=True)
class WholeProgramSemanticsReport:
    """Static semantics summary for whole-program AD graph capture."""

    bytecode_frontend: bool
    source_frontend: bool
    graph_capture: bool
    aliasing_observed: bool
    mutation_observed: bool
    loop_observed: bool
    control_flow_observed: bool
    numpy_observed: bool
    differentiation_semantics: str
    accepted_python_semantics: tuple[str, ...] = ()
    unsupported_python_semantics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "bytecode_frontend",
            "source_frontend",
            "graph_capture",
            "aliasing_observed",
            "mutation_observed",
            "loop_observed",
            "control_flow_observed",
            "numpy_observed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not self.differentiation_semantics:
            raise ValueError("differentiation_semantics must be non-empty")
        for name in ("accepted_python_semantics", "unsupported_python_semantics"):
            semantics = getattr(self, name)
            if not isinstance(semantics, tuple):
                raise ValueError(f"{name} must be a tuple")
            if any(not isinstance(item, str) or not item for item in semantics):
                raise ValueError(f"{name} entries must be non-empty strings")


@dataclass(frozen=True)
class WholeProgramCompilerFrontendReport:
    """Static bytecode/source frontend report for whole-program AD objectives.

    The report inspects Python bytecode and source-derived AST features without
    executing the objective. It is a compiler-frontend preflight artefact for
    accepted whole-program AD semantics, not executable Rust, LLVM, JIT,
    provider, hardware, or benchmark evidence.

    Parameters
    ----------
    function_name:
        Python callable name used for diagnostics.
    bytecode_instructions:
        Normalised Python bytecode instruction rows from ``dis``.
    bytecode_basic_blocks:
        Static control-flow block skeleton derived from the bytecode stream.
    source_ir_features:
        Source-level AST feature rows used by the Program AD frontend.
    source_regions:
        Static source-region graph derived from bounded AST constructs.
    source_bytecode_line_map:
        Static crosswalk from source lines to bytecode offsets, source regions,
        and source feature kinds.
    symbol_scope_entries:
        Static symbol-scope table derived from source names, bytecode operands,
        and function code-object scope metadata.
    unsupported_semantic_diagnostics:
        Static fail-closed diagnostics for unsupported Python constructs.
    semantics_report:
        Static semantics summary derived from bytecode and source features.
    source_available:
        Whether source text could be obtained through introspection.
    source_sha256:
        SHA-256 digest of the dedented source when available.
    source_start_line:
        Absolute file line where the inspected source snippet starts.
    source_end_line:
        Absolute file line where the inspected source snippet ends.
    bytecode_digest:
        SHA-256 digest over the normalised bytecode instruction stream.
    frontend_digest:
        SHA-256 digest over bytecode, source, source features, source regions,
        and semantic support metadata.
    ast_node_count:
        Number of AST nodes in the parsed source tree when available.
    hard_gaps:
        Named blockers that prevent the report from being accepted as a
        complete bytecode/source frontend preflight.
    claim_boundary:
        Boundary preventing this static report from becoming an execution or
        performance claim.
    """

    function_name: str
    bytecode_instructions: tuple[WholeProgramBytecodeInstruction, ...]
    bytecode_basic_blocks: tuple[WholeProgramBytecodeBasicBlock, ...]
    source_ir_features: tuple[WholeProgramSourceIRFeature, ...]
    source_regions: tuple[WholeProgramSourceRegion, ...]
    source_bytecode_line_map: tuple[WholeProgramSourceBytecodeLineMap, ...]
    symbol_scope_entries: tuple[WholeProgramSymbolScopeEntry, ...]
    unsupported_semantic_diagnostics: tuple[WholeProgramUnsupportedSemanticDiagnostic, ...]
    semantics_report: WholeProgramSemanticsReport
    source_available: bool
    source_sha256: str | None
    source_start_line: int | None
    source_end_line: int | None
    bytecode_digest: str
    frontend_digest: str
    ast_node_count: int
    hard_gaps: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not isinstance(self.function_name, str) or not self.function_name:
            raise ValueError("compiler frontend function_name must be non-empty")
        if any(
            not isinstance(instruction, WholeProgramBytecodeInstruction)
            for instruction in self.bytecode_instructions
        ):
            raise ValueError(
                "compiler frontend bytecode_instructions must contain "
                "WholeProgramBytecodeInstruction entries"
            )
        if any(
            not isinstance(block, WholeProgramBytecodeBasicBlock)
            for block in self.bytecode_basic_blocks
        ):
            raise ValueError(
                "compiler frontend bytecode_basic_blocks must contain "
                "WholeProgramBytecodeBasicBlock entries"
            )
        instruction_offsets = {instruction.offset for instruction in self.bytecode_instructions}
        for block in self.bytecode_basic_blocks:
            if any(offset not in instruction_offsets for offset in block.instruction_offsets):
                raise ValueError(
                    "compiler frontend bytecode_basic_blocks must reference known instructions"
                )
            if any(offset not in instruction_offsets for offset in block.successor_offsets):
                raise ValueError(
                    "compiler frontend bytecode_basic_blocks must reference known successors"
                )
        if any(
            not isinstance(feature, WholeProgramSourceIRFeature)
            for feature in self.source_ir_features
        ):
            raise ValueError(
                "compiler frontend source_ir_features must contain "
                "WholeProgramSourceIRFeature entries"
            )
        if any(not isinstance(region, WholeProgramSourceRegion) for region in self.source_regions):
            raise ValueError(
                "compiler frontend source_regions must contain WholeProgramSourceRegion entries"
            )
        region_ids = {region.region_id for region in self.source_regions}
        for region in self.source_regions:
            if region.parent_region_id is not None and region.parent_region_id not in region_ids:
                raise ValueError("compiler frontend source_regions must reference known parents")
        if any(
            not isinstance(line_map, WholeProgramSourceBytecodeLineMap)
            for line_map in self.source_bytecode_line_map
        ):
            raise ValueError(
                "compiler frontend source_bytecode_line_map must contain "
                "WholeProgramSourceBytecodeLineMap entries"
            )
        for line_map in self.source_bytecode_line_map:
            if self.source_regions and not line_map.region_ids:
                raise ValueError(
                    "compiler frontend source_bytecode_line_map must attach source regions"
                )
            if any(offset not in instruction_offsets for offset in line_map.instruction_offsets):
                raise ValueError(
                    "compiler frontend source_bytecode_line_map must reference known instructions"
                )
            if any(region_id not in region_ids for region_id in line_map.region_ids):
                raise ValueError(
                    "compiler frontend source_bytecode_line_map must reference known regions"
                )
        if any(
            not isinstance(entry, WholeProgramSymbolScopeEntry)
            for entry in self.symbol_scope_entries
        ):
            raise ValueError(
                "compiler frontend symbol_scope_entries must contain "
                "WholeProgramSymbolScopeEntry entries"
            )
        for entry in self.symbol_scope_entries:
            if entry.line_numbers and self.source_regions and not entry.region_ids:
                raise ValueError(
                    "compiler frontend symbol_scope_entries with lines must attach source regions"
                )
            if any(offset not in instruction_offsets for offset in entry.bytecode_offsets):
                raise ValueError(
                    "compiler frontend symbol_scope_entries must reference known instructions"
                )
            if any(region_id not in region_ids for region_id in entry.region_ids):
                raise ValueError(
                    "compiler frontend symbol_scope_entries must reference known regions"
                )
        if any(
            not isinstance(diagnostic, WholeProgramUnsupportedSemanticDiagnostic)
            for diagnostic in self.unsupported_semantic_diagnostics
        ):
            raise ValueError(
                "compiler frontend unsupported_semantic_diagnostics must contain "
                "WholeProgramUnsupportedSemanticDiagnostic entries"
            )
        for diagnostic in self.unsupported_semantic_diagnostics:
            if self.source_regions and not diagnostic.region_ids:
                raise ValueError(
                    "compiler frontend unsupported_semantic_diagnostics must attach regions"
                )
            if any(region_id not in region_ids for region_id in diagnostic.region_ids):
                raise ValueError(
                    "compiler frontend unsupported_semantic_diagnostics must reference known regions"
                )
            if any(offset not in instruction_offsets for offset in diagnostic.bytecode_offsets):
                raise ValueError(
                    "compiler frontend unsupported_semantic_diagnostics must reference known "
                    "instructions"
                )
        if not isinstance(self.semantics_report, WholeProgramSemanticsReport):
            raise ValueError(
                "compiler frontend semantics_report must be WholeProgramSemanticsReport"
            )
        if not isinstance(self.source_available, bool):
            raise ValueError("compiler frontend source_available must be boolean")
        if self.source_available:
            if self.source_sha256 is None or len(self.source_sha256) != 64:
                raise ValueError("available compiler frontend source requires sha256 digest")
            if self.source_start_line is None or self.source_start_line <= 0:
                raise ValueError("available compiler frontend source requires start line")
            if self.source_end_line is None or self.source_end_line < self.source_start_line:
                raise ValueError("available compiler frontend source requires valid end line")
        elif self.source_sha256 is not None:
            raise ValueError("unavailable compiler frontend source must not carry sha256")
        elif self.source_start_line is not None or self.source_end_line is not None:
            raise ValueError("unavailable compiler frontend source must not carry line bounds")
        if len(self.bytecode_digest) != 64:
            raise ValueError("compiler frontend bytecode_digest must be a sha256 digest")
        if len(self.frontend_digest) != 64:
            raise ValueError("compiler frontend frontend_digest must be a sha256 digest")
        if self.ast_node_count < 0:
            raise ValueError("compiler frontend ast_node_count must be non-negative")
        if any(not isinstance(gap, str) or not gap for gap in self.hard_gaps):
            raise ValueError("compiler frontend hard_gaps entries must be non-empty strings")
        if self.semantics_report.unsupported_python_semantics and not any(
            gap.startswith("unsupported_python_semantics:") for gap in self.hard_gaps
        ):
            raise ValueError(
                "unsupported compiler frontend semantics must be recorded as hard gaps"
            )
        diagnostic_semantics = {
            diagnostic.semantic for diagnostic in self.unsupported_semantic_diagnostics
        }
        report_semantics = set(self.semantics_report.unsupported_python_semantics)
        if diagnostic_semantics != report_semantics:
            raise ValueError("unsupported compiler frontend semantics must match diagnostics")
        if not self.claim_boundary:
            raise ValueError("compiler frontend claim_boundary must be non-empty")

    @property
    def bytecode_instruction_count(self) -> int:
        """Return the number of bytecode instructions in the frontend report."""

        return len(self.bytecode_instructions)

    @property
    def source_feature_count(self) -> int:
        """Return the number of source IR feature rows in the frontend report."""

        return len(self.source_ir_features)

    @property
    def frontend_ready(self) -> bool:
        """Return whether bytecode and source frontend metadata passed preflight."""

        return (
            self.source_available
            and self.bytecode_instruction_count > 0
            and self.bytecode_basic_block_count > 0
            and self.source_region_count > 0
            and self.source_bytecode_line_map_count > 0
            and self.symbol_scope_entry_count > 0
            and self.semantics_report.bytecode_frontend
            and self.semantics_report.source_frontend
            and not self.hard_gaps
        )

    @property
    def bytecode_basic_block_count(self) -> int:
        """Return the number of bytecode basic blocks in the frontend report."""

        return len(self.bytecode_basic_blocks)

    @property
    def source_region_count(self) -> int:
        """Return the number of source regions in the frontend report."""

        return len(self.source_regions)

    @property
    def source_bytecode_line_map_count(self) -> int:
        """Return the number of source-bytecode crosswalk rows in the report."""

        return len(self.source_bytecode_line_map)

    @property
    def symbol_scope_entry_count(self) -> int:
        """Return the number of static symbol-scope entries in the report."""

        return len(self.symbol_scope_entries)

    @property
    def unsupported_semantic_diagnostic_count(self) -> int:
        """Return the number of unsupported-semantics diagnostics in the report."""

        return len(self.unsupported_semantic_diagnostics)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready compiler frontend report."""

        return {
            "function_name": self.function_name,
            "frontend_ready": self.frontend_ready,
            "source_available": self.source_available,
            "source_sha256": self.source_sha256,
            "source_start_line": self.source_start_line,
            "source_end_line": self.source_end_line,
            "bytecode_digest": self.bytecode_digest,
            "frontend_digest": self.frontend_digest,
            "bytecode_instruction_count": self.bytecode_instruction_count,
            "bytecode_basic_block_count": self.bytecode_basic_block_count,
            "source_feature_count": self.source_feature_count,
            "source_region_count": self.source_region_count,
            "source_bytecode_line_map_count": self.source_bytecode_line_map_count,
            "symbol_scope_entry_count": self.symbol_scope_entry_count,
            "unsupported_semantic_diagnostic_count": (self.unsupported_semantic_diagnostic_count),
            "ast_node_count": self.ast_node_count,
            "bytecode_instructions": [
                {
                    "offset": instruction.offset,
                    "opname": instruction.opname,
                    "argrepr": instruction.argrepr,
                    "line_number": instruction.line_number,
                    "jump_target_offset": instruction.jump_target_offset,
                }
                for instruction in self.bytecode_instructions
            ],
            "bytecode_basic_blocks": [block.to_dict() for block in self.bytecode_basic_blocks],
            "source_ir_features": [
                {
                    "kind": feature.kind,
                    "detail": feature.detail,
                    "line_number": feature.line_number,
                }
                for feature in self.source_ir_features
            ],
            "source_regions": [region.to_dict() for region in self.source_regions],
            "source_bytecode_line_map": [
                line_map.to_dict() for line_map in self.source_bytecode_line_map
            ],
            "symbol_scope_entries": [entry.to_dict() for entry in self.symbol_scope_entries],
            "unsupported_semantic_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.unsupported_semantic_diagnostics
            ],
            "semantics_report": {
                "bytecode_frontend": self.semantics_report.bytecode_frontend,
                "source_frontend": self.semantics_report.source_frontend,
                "graph_capture": self.semantics_report.graph_capture,
                "aliasing_observed": self.semantics_report.aliasing_observed,
                "mutation_observed": self.semantics_report.mutation_observed,
                "loop_observed": self.semantics_report.loop_observed,
                "control_flow_observed": self.semantics_report.control_flow_observed,
                "numpy_observed": self.semantics_report.numpy_observed,
                "differentiation_semantics": self.semantics_report.differentiation_semantics,
                "accepted_python_semantics": list(self.semantics_report.accepted_python_semantics),
                "unsupported_python_semantics": list(
                    self.semantics_report.unsupported_python_semantics
                ),
            },
            "hard_gaps": list(self.hard_gaps),
            "claim_boundary": self.claim_boundary,
        }


def _objective_source_metadata(
    objective: Callable[..., object],
) -> _ObjectiveSourceMetadata | None:
    """Return dedented source and file-line bounds when introspection permits."""

    try:
        source_lines, start_line = inspect.getsourcelines(objective)
    except (OSError, TypeError):
        return None
    source = textwrap.dedent("".join(source_lines)).strip()
    if not source:
        return None
    source_line_count = max(1, len(source.splitlines()))
    return _ObjectiveSourceMetadata(
        source=source,
        start_line=int(start_line),
        end_line=int(start_line) + source_line_count - 1,
    )


def _objective_source(objective: Callable[..., object]) -> str | None:
    """Return dedented source for a Python callable when introspection permits."""

    metadata = _objective_source_metadata(objective)
    return None if metadata is None else metadata.source


def _objective_bytecode(
    objective: Callable[..., object],
) -> tuple[WholeProgramBytecodeInstruction, ...]:
    """Return bytecode frontend IR for a Python objective when available."""

    try:
        instructions = dis.get_instructions(objective)
    except TypeError:
        return ()

    return tuple(
        WholeProgramBytecodeInstruction(
            offset=int(instruction.offset),
            opname=instruction.opname,
            argrepr=instruction.argrepr,
            line_number=_instruction_line_number(instruction),
            jump_target_offset=(
                int(instruction.argval)
                if isinstance(instruction.argval, int)
                and ("JUMP" in instruction.opname or instruction.opname == "FOR_ITER")
                else None
            ),
        )
        for instruction in instructions
    )


def _normalise_positive_line_number(value: int | None) -> int | None:
    """Return a positive source line number or ``None`` for missing metadata."""

    if value is None:
        return None
    line_number = int(value)
    return line_number if line_number > 0 else None


def _instruction_line_number(instruction: dis.Instruction) -> int | None:
    """Return the CPython-version-stable source line for a bytecode instruction."""

    starts_line = instruction.starts_line
    if isinstance(starts_line, bool):
        positions = instruction.positions
        return None if positions is None else _normalise_positive_line_number(positions.lineno)
    return _normalise_positive_line_number(starts_line)


def _source_ir_features(
    source: str | None,
    *,
    accepted_python_semantics: tuple[str, ...] = (),
    unsupported_python_semantics: tuple[str, ...] = (),
    unsupported_semantic_diagnostics: Sequence[WholeProgramUnsupportedSemanticDiagnostic] = (),
) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return source-level control, alias, mutation, and loop features."""

    features: list[WholeProgramSourceIRFeature] = []
    for detail in accepted_python_semantics:
        features.append(WholeProgramSourceIRFeature("python_semantics", detail, 1))
    if unsupported_semantic_diagnostics:
        for diagnostic in unsupported_semantic_diagnostics:
            features.append(
                WholeProgramSourceIRFeature(
                    "unsupported_python_semantics",
                    diagnostic.semantic,
                    diagnostic.line_number,
                )
            )
    else:
        for detail in unsupported_python_semantics:
            features.append(WholeProgramSourceIRFeature("unsupported_python_semantics", detail, 1))
    if source is None:
        return tuple(features)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return tuple(features)

    def add(node: ast.AST, kind: str, detail: str) -> None:
        line_number = int(getattr(node, "lineno", 1) or 1)
        features.append(WholeProgramSourceIRFeature(kind, detail, line_number))

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            add(node, "control_flow", "if")
        elif isinstance(node, ast.IfExp):
            add(node, "control_flow", "if_expression")
        elif isinstance(node, ast.For):
            add(node, "loop", "for")
        elif isinstance(node, ast.While):
            add(node, "loop", "while")
        elif isinstance(node, ast.Break):
            add(node, "loop", "break")
        elif isinstance(node, ast.Continue):
            add(node, "loop", "continue")
        elif isinstance(node, ast.Assign):
            if len(node.targets) > 1 or any(
                isinstance(target, ast.Name) for target in node.targets
            ):
                add(node, "alias_analysis", "assignment_binding")
            if any(_is_mutation_target(target) for target in node.targets):
                add(node, "mutation", "indexed_or_attribute_assignment")
        elif isinstance(node, ast.AugAssign):
            add(node, "mutation", "augmented_assignment")
        elif isinstance(node, ast.Delete):
            add(node, "mutation", "delete")
        elif isinstance(node, ast.Call):
            name = _ast_call_name(node.func)
            if name.startswith("np.") or name.startswith("numpy."):
                add(node, "numpy", name)
            if name.rsplit(".", 1)[-1] in {
                "append",
                "extend",
                "insert",
                "pop",
                "remove",
                "clear",
                "sort",
                "update",
                "add",
            }:
                add(node, "mutation", name)
    features.extend(_source_list_alias_features(tree))
    features.extend(_source_local_rebinding_alias_features(tree))
    features.extend(_source_expression_rebinding_alias_features(tree))
    features.extend(_source_object_attribute_alias_features(tree))
    features.extend(_source_control_path_alias_features(tree))
    features.extend(_source_loop_carried_state_features(tree))
    return tuple(features)


def _source_list_alias_features(tree: ast.AST) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return bounded source-level list alias metadata for local bindings."""

    features: list[WholeProgramSourceIRFeature] = []
    list_roots: set[str] = set()
    aliases: dict[str, str] = {}

    def root_for(name: str) -> str | None:
        if name in list_roots:
            return name
        return aliases.get(name)

    assignments = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Assign)),
        key=lambda node: int(getattr(node, "lineno", 1) or 1),
    )
    for node in assignments:
        line_number = int(getattr(node, "lineno", 1) or 1)
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
                list_roots.add(target.id)
                features.append(
                    WholeProgramSourceIRFeature(
                        "list_alias",
                        f"list:{target.id}->name:{target.id}",
                        line_number,
                    )
                )
            elif isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                root = root_for(node.value.id)
                if root is not None:
                    aliases[target.id] = root
                    features.append(
                        WholeProgramSourceIRFeature(
                            "list_alias",
                            f"list:{root}->name:{target.id}",
                            line_number,
                        )
                    )
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                root = root_for(target.value.id)
                if root is not None:
                    features.append(
                        WholeProgramSourceIRFeature(
                            "list_alias",
                            f"list:{root}->source:list_mutation",
                            line_number,
                        )
                    )
    return tuple(features)


def _source_local_rebinding_alias_features(
    tree: ast.AST,
) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return bounded source-level metadata for local name rebinding."""

    features: list[WholeProgramSourceIRFeature] = []
    assignments = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Assign)),
        key=lambda node: int(getattr(node, "lineno", 1) or 1),
    )
    for node in assignments:
        line_number = int(getattr(node, "lineno", 1) or 1)
        if not isinstance(node.value, ast.Name):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                features.append(
                    WholeProgramSourceIRFeature(
                        "local_rebinding_alias",
                        f"name:{node.value.id}->name:{target.id}",
                        line_number,
                    )
                )
    return tuple(features)


def _source_expression_rebinding_alias_features(
    tree: ast.AST,
) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return bounded source-level metadata for local expression rebinding."""

    features: list[WholeProgramSourceIRFeature] = []
    assignments = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Assign)),
        key=lambda node: int(getattr(node, "lineno", 1) or 1),
    )
    for node in assignments:
        if isinstance(node.value, ast.Name):
            continue
        if not _expression_may_rebind_trace_value(node.value):
            continue
        line_number = int(getattr(node, "lineno", 1) or 1)
        expression = _stable_ast_expression_label(node.value, line_number)
        for target in node.targets:
            if isinstance(target, ast.Name):
                features.append(
                    WholeProgramSourceIRFeature(
                        "expression_rebinding_alias",
                        f"{expression}->name:{target.id}",
                        line_number,
                    )
                )
    return tuple(features)


def _source_object_attribute_alias_features(
    tree: ast.AST,
) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return bounded source-level metadata for local object attribute aliases."""

    features: list[WholeProgramSourceIRFeature] = []
    local_objects: set[str] = set()
    assignments = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Assign)),
        key=lambda node: int(getattr(node, "lineno", 1) or 1),
    )
    for node in assignments:
        line_number = int(getattr(node, "lineno", 1) or 1)
        for target in node.targets:
            if isinstance(target, ast.Name) and _is_local_object_constructor_call(node.value):
                local_objects.add(target.id)
                features.append(
                    WholeProgramSourceIRFeature(
                        "object_attribute_alias",
                        f"object:{target.id}->name:{target.id}",
                        line_number,
                    )
                )
            elif isinstance(target, ast.Attribute):
                root_name = _ast_attribute_root(target)
                if root_name in local_objects:
                    features.append(
                        WholeProgramSourceIRFeature(
                            "object_attribute_alias",
                            f"object:{root_name}->attr:{root_name}.{target.attr}",
                            line_number,
                        )
                    )
            elif isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute):
                root_name = _ast_attribute_root(node.value)
                if root_name in local_objects:
                    features.append(
                        WholeProgramSourceIRFeature(
                            "object_attribute_alias",
                            f"attr:{root_name}.{node.value.attr}->name:{target.id}",
                            line_number,
                        )
                    )
    return tuple(features)


def _source_control_path_alias_features(
    tree: ast.AST,
) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return branch-local alias metadata that needs non-executed semantics."""

    features: list[WholeProgramSourceIRFeature] = []

    def target_label(target: ast.AST) -> str | None:
        if isinstance(target, ast.Name):
            return f"name:{target.id}"
        if isinstance(target, ast.Attribute):
            root_name = _ast_attribute_root(target)
            return f"attr:{root_name}.{target.attr}" if root_name else None
        if isinstance(target, ast.Subscript):
            root_name = _ast_subscript_root(target)
            return f"subscript:{root_name}" if root_name else None
        return None

    def add_targets(statement: ast.stmt, source: str) -> None:
        targets: Sequence[ast.AST]
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
            targets = (statement.target,)
        else:
            return
        line_number = int(getattr(statement, "lineno", 1) or 1)
        for target in targets:
            label = target_label(target)
            if label is not None:
                features.append(
                    WholeProgramSourceIRFeature(
                        "control_path_alias",
                        f"{source}->control:{label}",
                        line_number,
                    )
                )

    for node in sorted(
        (candidate for candidate in ast.walk(tree) if isinstance(candidate, ast.If)),
        key=lambda candidate: int(getattr(candidate, "lineno", 1) or 1),
    ):
        branch_line = int(getattr(node, "lineno", 1) or 1)
        for statement in node.body:
            add_targets(statement, f"control:if:{branch_line}:body")
        for statement in node.orelse:
            add_targets(statement, f"control:if:{branch_line}:orelse")
    return tuple(features)


def _expression_may_rebind_trace_value(node: ast.AST) -> bool:
    """Return whether an expression can carry a trace value into a local binding."""

    return any(
        isinstance(child, ast.Name | ast.Attribute | ast.Subscript) for child in ast.walk(node)
    )


def _stable_ast_expression_label(node: ast.AST, line_number: int) -> str:
    """Return a deterministic short label for a source expression alias."""

    try:
        expression = ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is available on supported Python.
        expression = node.__class__.__name__
    compact = "_".join(expression.split())
    if len(compact) > 80:
        compact = f"{compact[:77]}..."
    return f"expr:{line_number}:{compact}"


def _is_local_object_constructor_call(node: ast.AST) -> bool:
    """Return whether an assignment value looks like a local object constructor."""

    if not isinstance(node, ast.Call):
        return False
    name = _ast_call_name(node.func)
    if not name:
        return False
    root = name.split(".", 1)[0]
    return root not in {"math", "np", "numpy"}


def _source_loop_carried_state_features(
    tree: ast.AST,
) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return bounded source metadata for local loop-carried scalar state."""

    features: list[WholeProgramSourceIRFeature] = []

    def assigned_names_in_target(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for element in target.elts:
                names.update(assigned_names_in_target(element))
            return names
        return set()

    def assigned_names_in_statement(statement: ast.stmt) -> set[str]:
        if isinstance(statement, ast.Assign):
            names: set[str] = set()
            for target in statement.targets:
                names.update(assigned_names_in_target(target))
            return names
        if isinstance(statement, ast.AnnAssign):
            return assigned_names_in_target(statement.target)
        if isinstance(statement, ast.AugAssign):
            return assigned_names_in_target(statement.target)
        if isinstance(statement, (ast.For, ast.While, ast.If)):
            names = (
                assigned_names_in_target(statement.target)
                if isinstance(statement, ast.For)
                else set()
            )
            for child in (*statement.body, *statement.orelse):
                names.update(assigned_names_in_statement(child))
            return names
        return set()

    def scan_statement_block(statements: Sequence[ast.stmt]) -> set[str]:
        assigned_before: set[str] = set()
        for statement in statements:
            if isinstance(statement, (ast.For, ast.While)):
                loop_assigned: set[str] = set()
                for child in statement.body:
                    loop_assigned.update(assigned_names_in_statement(child))
                line_number = int(getattr(statement, "lineno", 1) or 1)
                for name in sorted(assigned_before.intersection(loop_assigned)):
                    features.append(
                        WholeProgramSourceIRFeature(
                            "loop_carried_state",
                            f"loop:{name}:entry->loop:{name}:backedge",
                            line_number,
                        )
                    )
                scan_statement_block(statement.body)
                scan_statement_block(statement.orelse)
            elif isinstance(statement, ast.If):
                scan_statement_block(statement.body)
                scan_statement_block(statement.orelse)
            assigned_before.update(assigned_names_in_statement(statement))
        return assigned_before

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_statement_block(node.body)
    return tuple(features)


def _accepted_python_semantics(
    objective: Callable[..., object],
    source: str | None,
) -> tuple[str, ...]:
    """Return Python calling semantics supported by whole-program AD preflight."""

    accepted: set[str] = set()
    code = getattr(objective, "__code__", None)
    if code is not None and code.co_freevars:
        accepted.add("closure")
    try:
        signature = inspect.signature(objective)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for parameter in signature.parameters.values():
            if parameter.default is not inspect.Signature.empty:
                accepted.add("default_argument")
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                accepted.add("keyword_only_parameter")
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                accepted.add("var_keyword_parameter")
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                accepted.add("var_positional_parameter")
    if _source_has_node(source, ast.ListComp):
        accepted.add("list_comprehension")
    if _source_has_node(source, ast.GeneratorExp):
        accepted.add("generator_expression")
    return tuple(sorted(accepted))


def _unsupported_python_semantics(
    objective: Callable[..., object],
    source: str | None,
) -> tuple[str, ...]:
    """Return unsupported Python semantics detected before objective execution."""

    return tuple(
        sorted(
            {
                diagnostic.semantic
                for diagnostic in _unsupported_python_semantic_diagnostics(
                    objective=objective,
                    source=source,
                    source_start_line=None,
                    bytecode_instructions=(),
                    source_regions=(),
                )
            }
        )
    )


def _unsupported_python_semantic_diagnostics(
    *,
    objective: Callable[..., object],
    source: str | None,
    source_start_line: int | None,
    bytecode_instructions: Sequence[WholeProgramBytecodeInstruction],
    source_regions: Sequence[WholeProgramSourceRegion],
) -> tuple[WholeProgramUnsupportedSemanticDiagnostic, ...]:
    """Return located unsupported Python-semantics diagnostics."""

    if source is None:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    objective_name = getattr(objective, "__name__", "")
    captured_attribute_roots = _captured_or_global_names(objective)
    allowed_attribute_roots = {"math", "np", "numpy"}
    diagnostics: dict[tuple[str, int, str], WholeProgramUnsupportedSemanticDiagnostic] = {}

    def add(node: ast.AST, semantic: str, detail: str | None = None) -> None:
        line_number = int(getattr(node, "lineno", 1) or 1)
        absolute_line = None if source_start_line is None else source_start_line + line_number - 1
        bytecode_offsets = tuple(
            instruction.offset
            for instruction in bytecode_instructions
            if instruction.line_number == absolute_line
        )
        diagnostic = WholeProgramUnsupportedSemanticDiagnostic(
            semantic=semantic,
            detail=detail or semantic,
            line_number=line_number,
            absolute_line_number=absolute_line,
            region_ids=_source_region_ids_for_line(source_regions, line_number),
            bytecode_offsets=bytecode_offsets,
        )
        diagnostics[(semantic, line_number, diagnostic.detail)] = diagnostic

    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            if any(generator.ifs for generator in node.generators):
                add(node, "filtered_comprehension")
        elif isinstance(node, ast.SetComp | ast.DictComp):
            add(node, "set_or_dict_comprehension")
        elif isinstance(node, ast.Yield | ast.YieldFrom):
            add(node, "generator")
        elif isinstance(node, ast.With | ast.AsyncWith):
            add(node, "context_manager")
        elif isinstance(node, ast.Try | ast.Raise | ast.Assert):
            add(node, "exception_control_flow")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.decorator_list:
                add(node, "decorator")
        elif isinstance(node, ast.Call) and objective_name:
            call_name = _ast_call_name(node.func)
            if call_name.rsplit(".", 1)[-1] == objective_name:
                add(node, "recursion")
        elif isinstance(node, ast.Attribute):
            root_name = _ast_attribute_root(node)
            if root_name in captured_attribute_roots and root_name not in allowed_attribute_roots:
                add(node, "object_attribute", detail=f"object_attribute:{root_name}")
    return tuple(
        diagnostics[key]
        for key in sorted(diagnostics, key=lambda item: (item[1], item[0], item[2]))
    )


def _captured_or_global_names(objective: Callable[..., object]) -> set[str]:
    """Return names whose attributes would resolve through closure/global objects."""

    code = getattr(objective, "__code__", None)
    if code is None:
        return set()
    names = set(code.co_freevars)
    globals_mapping = getattr(objective, "__globals__", {})
    if isinstance(globals_mapping, Mapping):
        names.update(name for name in code.co_names if name in globals_mapping)
    return names


def _ast_attribute_root(node: ast.Attribute) -> str:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return ""


def _ast_subscript_root(node: ast.Subscript) -> str:
    current: ast.AST = node.value
    while isinstance(current, ast.Subscript):
        current = current.value
    if isinstance(current, ast.Attribute):
        return _ast_attribute_root(current)
    if isinstance(current, ast.Name):
        return current.id
    return ""


def _is_mutation_target(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript | ast.Attribute)


def _ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _whole_program_semantics_report(
    *,
    bytecode_instructions: tuple[WholeProgramBytecodeInstruction, ...],
    source_ir_features: tuple[WholeProgramSourceIRFeature, ...],
    trace_events: tuple[object, ...],
    source: str | None,
    accepted_python_semantics: tuple[str, ...],
    unsupported_python_semantics: tuple[str, ...],
    numpy_observed: bool,
    differentiation_semantics: str,
) -> WholeProgramSemanticsReport:
    feature_kinds = {feature.kind for feature in source_ir_features}
    jump_ops = {
        instruction.opname
        for instruction in bytecode_instructions
        if "JUMP" in instruction.opname or instruction.opname in {"FOR_ITER"}
    }
    return WholeProgramSemanticsReport(
        bytecode_frontend=bool(bytecode_instructions),
        source_frontend=source is not None,
        graph_capture=bool(trace_events or bytecode_instructions or source_ir_features),
        aliasing_observed=any("alias" in kind for kind in feature_kinds),
        mutation_observed="mutation" in feature_kinds,
        loop_observed="loop" in feature_kinds or "FOR_ITER" in jump_ops,
        control_flow_observed=_source_has_control_flow(source)
        or "control_flow" in feature_kinds
        or bool(jump_ops),
        numpy_observed=numpy_observed or "numpy" in feature_kinds,
        differentiation_semantics=differentiation_semantics,
        accepted_python_semantics=accepted_python_semantics,
        unsupported_python_semantics=unsupported_python_semantics,
    )


def compile_whole_program_frontend(
    objective: Callable[..., object],
) -> WholeProgramCompilerFrontendReport:
    """Inspect a Python callable as a bounded whole-program AD frontend.

    Parameters
    ----------
    objective:
        Callable to inspect. The callable is not executed; only bytecode,
        source text, AST features, and supported/unsupported Python semantics
        are inspected.

    Returns
    -------
    WholeProgramCompilerFrontendReport
        Static compiler-frontend preflight report with bytecode/source
        metadata, deterministic digests, hard gaps, and claim boundary.

    Raises
    ------
    ValueError
        If ``objective`` is not callable.
    """

    if not callable(objective):
        raise ValueError("whole-program compiler frontend objective must be callable")
    source_metadata = _objective_source_metadata(objective)
    source = None if source_metadata is None else source_metadata.source
    bytecode_instructions = _objective_bytecode(objective)
    accepted_python_semantics = _accepted_python_semantics(objective, source)
    preliminary_unsupported_diagnostics = _unsupported_python_semantic_diagnostics(
        objective=objective,
        source=source,
        source_start_line=None if source_metadata is None else source_metadata.start_line,
        bytecode_instructions=bytecode_instructions,
        source_regions=(),
    )
    unsupported_python_semantics = tuple(
        sorted({diagnostic.semantic for diagnostic in preliminary_unsupported_diagnostics})
    )
    source_ir_features = _source_ir_features(
        source,
        accepted_python_semantics=accepted_python_semantics,
        unsupported_python_semantics=unsupported_python_semantics,
        unsupported_semantic_diagnostics=preliminary_unsupported_diagnostics,
    )
    bytecode_basic_blocks = _bytecode_basic_blocks(bytecode_instructions)
    source_regions = _source_regions(source, source_ir_features)
    unsupported_semantic_diagnostics = _unsupported_python_semantic_diagnostics(
        objective=objective,
        source=source,
        source_start_line=None if source_metadata is None else source_metadata.start_line,
        bytecode_instructions=bytecode_instructions,
        source_regions=source_regions,
    )
    source_bytecode_line_map = _source_bytecode_line_map(
        bytecode_instructions=bytecode_instructions,
        source_ir_features=source_ir_features,
        source_regions=source_regions,
        source_start_line=None if source_metadata is None else source_metadata.start_line,
    )
    symbol_scope_entries = _symbol_scope_entries(
        objective=objective,
        source=source,
        bytecode_instructions=bytecode_instructions,
        source_regions=source_regions,
        source_start_line=None if source_metadata is None else source_metadata.start_line,
    )
    source_parse_failed = _source_parse_failed(source)
    semantics_report = _whole_program_semantics_report(
        bytecode_instructions=bytecode_instructions,
        source_ir_features=source_ir_features,
        trace_events=(),
        source=source,
        accepted_python_semantics=accepted_python_semantics,
        unsupported_python_semantics=unsupported_python_semantics,
        numpy_observed=_source_mentions_numpy(source),
        differentiation_semantics=(
            "static bytecode/source frontend preflight for whole-program AD; "
            "no objective execution, no finite-difference fallback, and no "
            "executable Rust, LLVM, JIT, provider, hardware, or performance claim"
        ),
    )
    hard_gaps: list[str] = []
    if not bytecode_instructions:
        hard_gaps.append("bytecode_frontend_missing")
    if not bytecode_basic_blocks:
        hard_gaps.append("bytecode_basic_blocks_missing")
    if source is None:
        hard_gaps.append("source_frontend_missing")
    elif not source_regions:
        hard_gaps.append("source_regions_missing")
    elif not source_bytecode_line_map:
        hard_gaps.append("source_bytecode_line_map_missing")
    if not symbol_scope_entries:
        hard_gaps.append("symbol_scope_entries_missing")
    if source_parse_failed:
        hard_gaps.append("source_ast_parse_failed")
    hard_gaps.extend(
        f"unsupported_python_semantics:{item}" for item in unsupported_python_semantics
    )
    return WholeProgramCompilerFrontendReport(
        function_name=_objective_name(objective),
        bytecode_instructions=bytecode_instructions,
        bytecode_basic_blocks=bytecode_basic_blocks,
        source_ir_features=source_ir_features,
        source_regions=source_regions,
        source_bytecode_line_map=source_bytecode_line_map,
        symbol_scope_entries=symbol_scope_entries,
        unsupported_semantic_diagnostics=unsupported_semantic_diagnostics,
        semantics_report=semantics_report,
        source_available=source is not None,
        source_sha256=None
        if source is None
        else hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source_start_line=None if source_metadata is None else source_metadata.start_line,
        source_end_line=None if source_metadata is None else source_metadata.end_line,
        bytecode_digest=_bytecode_instruction_digest(bytecode_instructions),
        frontend_digest=_frontend_digest(
            source=source,
            source_start_line=None if source_metadata is None else source_metadata.start_line,
            source_end_line=None if source_metadata is None else source_metadata.end_line,
            bytecode_instructions=bytecode_instructions,
            bytecode_basic_blocks=bytecode_basic_blocks,
            source_ir_features=source_ir_features,
            source_regions=source_regions,
            source_bytecode_line_map=source_bytecode_line_map,
            symbol_scope_entries=symbol_scope_entries,
            unsupported_semantic_diagnostics=unsupported_semantic_diagnostics,
            semantics_report=semantics_report,
            hard_gaps=tuple(hard_gaps),
        ),
        ast_node_count=_source_ast_node_count(source),
        hard_gaps=tuple(hard_gaps),
        claim_boundary=(
            "static bytecode/source compiler frontend preflight for supported "
            "whole-program AD Python semantics only; does not execute objectives "
            "and is not executable Rust, LLVM, JIT, provider, hardware, or "
            "benchmark evidence"
        ),
    )


def _objective_name(objective: Callable[..., object]) -> str:
    name = getattr(objective, "__qualname__", None) or getattr(objective, "__name__", "")
    return str(name) if name else "<callable>"


def _bytecode_basic_blocks(
    instructions: Sequence[WholeProgramBytecodeInstruction],
) -> tuple[WholeProgramBytecodeBasicBlock, ...]:
    """Return a deterministic static basic-block skeleton for bytecode."""

    if not instructions:
        return ()
    offsets = [instruction.offset for instruction in instructions]
    offset_set = set(offsets)
    offset_to_index = {offset: index for index, offset in enumerate(offsets)}
    leaders: set[int] = {offsets[0]}
    jump_targets: dict[int, int] = {}
    terminators = {"RETURN_VALUE", "RAISE_VARARGS", "RERAISE"}

    for index, instruction in enumerate(instructions):
        jump_target = _bytecode_jump_target(instruction)
        if jump_target is not None and jump_target in offset_set:
            leaders.add(jump_target)
            jump_targets[instruction.offset] = jump_target
        if (
            "JUMP" in instruction.opname
            or instruction.opname == "FOR_ITER"
            or instruction.opname in terminators
        ) and index + 1 < len(instructions):
            leaders.add(instructions[index + 1].offset)

    sorted_leaders = sorted(leaders)
    blocks: list[WholeProgramBytecodeBasicBlock] = []
    for block_index, start_offset in enumerate(sorted_leaders):
        start_index = offset_to_index[start_offset]
        if block_index + 1 < len(sorted_leaders):
            next_start_index = offset_to_index[sorted_leaders[block_index + 1]]
            block_instructions = tuple(instructions[start_index:next_start_index])
        else:
            block_instructions = tuple(instructions[start_index:])
        if not block_instructions:
            continue
        terminator = block_instructions[-1]
        successors: set[int] = set()
        jump_target = jump_targets.get(terminator.offset)
        if jump_target is not None:
            successors.add(jump_target)
        next_block_offset = (
            sorted_leaders[block_index + 1] if block_index + 1 < len(sorted_leaders) else None
        )
        if (
            next_block_offset is not None
            and terminator.opname not in terminators
            and not _bytecode_is_unconditional_jump(terminator.opname)
        ):
            successors.add(next_block_offset)
        blocks.append(
            WholeProgramBytecodeBasicBlock(
                label=f"bb{block_index}",
                start_offset=block_instructions[0].offset,
                end_offset=block_instructions[-1].offset,
                instruction_offsets=tuple(
                    instruction.offset for instruction in block_instructions
                ),
                successor_offsets=tuple(sorted(successors)),
                terminating_opname=terminator.opname,
            )
        )
    return tuple(blocks)


def _bytecode_is_unconditional_jump(opname: str) -> bool:
    """Return true when a bytecode operation jumps without a fallthrough path."""

    return opname.startswith("JUMP") and "IF" not in opname and opname != "JUMP_IF_NOT_EXC_MATCH"


def _bytecode_jump_target(instruction: WholeProgramBytecodeInstruction) -> int | None:
    """Return a jump target offset encoded by CPython's ``dis`` argrepr."""

    if "JUMP" not in instruction.opname and instruction.opname != "FOR_ITER":
        return None
    if instruction.jump_target_offset is not None:
        return instruction.jump_target_offset
    pieces = instruction.argrepr.replace("(", " ").replace(")", " ").split()
    for index, piece in enumerate(pieces):
        if piece == "to" and index + 1 < len(pieces):
            try:
                target = int(pieces[index + 1])
            except ValueError:
                return None
            return target if target >= 0 else None
    return None


def _bytecode_instruction_digest(
    instructions: Sequence[WholeProgramBytecodeInstruction],
) -> str:
    payload = [
        {
            "offset": instruction.offset,
            "opname": instruction.opname,
            "argrepr": instruction.argrepr,
            "line_number": instruction.line_number,
            "jump_target_offset": instruction.jump_target_offset,
        }
        for instruction in instructions
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_regions(
    source: str | None,
    source_ir_features: Sequence[WholeProgramSourceIRFeature],
) -> tuple[WholeProgramSourceRegion, ...]:
    """Return deterministic bounded source-region metadata for frontend planning."""

    if source is None:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    features_by_line: dict[int, set[str]] = {}
    for feature in source_ir_features:
        features_by_line.setdefault(feature.line_number, set()).add(feature.kind)

    source_line_count = max(1, len(source.splitlines()))
    regions: list[WholeProgramSourceRegion] = [
        WholeProgramSourceRegion(
            region_id="region:entry",
            kind="entry",
            detail="module",
            line_start=1,
            line_end=source_line_count,
            parent_region_id=None,
            feature_kinds=tuple(sorted({feature.kind for feature in source_ir_features})),
        )
    ]
    stack: list[str] = ["region:entry"]
    counter = 0

    def region_descriptor(node: ast.AST) -> tuple[str, str] | None:
        if isinstance(node, ast.FunctionDef):
            return ("function", node.name)
        if isinstance(node, ast.AsyncFunctionDef):
            return ("function", node.name)
        if isinstance(node, ast.If):
            return ("control_flow", "if")
        if isinstance(node, ast.IfExp):
            return ("control_flow", "if_expression")
        if isinstance(node, ast.For):
            return ("loop", "for")
        if isinstance(node, ast.While):
            return ("loop", "while")
        if isinstance(node, ast.Assign):
            return ("effect", "assignment")
        if isinstance(node, ast.AnnAssign):
            return ("effect", "annotated_assignment")
        if isinstance(node, ast.AugAssign):
            return ("effect", "augmented_assignment")
        if isinstance(node, ast.Delete):
            return ("effect", "delete")
        if isinstance(node, ast.Call):
            name = _ast_call_name(node.func)
            if name.startswith(("np.", "numpy.")):
                return ("numpy_call", name)
            if name.rsplit(".", 1)[-1] in {
                "append",
                "extend",
                "insert",
                "pop",
                "remove",
                "clear",
                "sort",
                "update",
                "add",
            }:
                return ("effect", name)
        return None

    def line_features(line_start: int, line_end: int) -> tuple[str, ...]:
        kinds: set[str] = set()
        for line_number in range(line_start, line_end + 1):
            kinds.update(features_by_line.get(line_number, set()))
        return tuple(sorted(kinds))

    def visit(node: ast.AST) -> None:
        nonlocal counter
        descriptor = region_descriptor(node)
        pushed = False
        if descriptor is not None:
            kind, detail = descriptor
            counter += 1
            line_start = int(getattr(node, "lineno", 1) or 1)
            line_end = int(getattr(node, "end_lineno", line_start) or line_start)
            region_id = f"region:{counter}:{kind}:{line_start}"
            regions.append(
                WholeProgramSourceRegion(
                    region_id=region_id,
                    kind=kind,
                    detail=detail,
                    line_start=line_start,
                    line_end=line_end,
                    parent_region_id=stack[-1],
                    feature_kinds=line_features(line_start, line_end),
                )
            )
            stack.append(region_id)
            pushed = True
        for child in ast.iter_child_nodes(node):
            visit(child)
        if pushed:
            stack.pop()

    visit(tree)
    return tuple(regions)


def _source_bytecode_line_map(
    *,
    bytecode_instructions: Sequence[WholeProgramBytecodeInstruction],
    source_ir_features: Sequence[WholeProgramSourceIRFeature],
    source_regions: Sequence[WholeProgramSourceRegion],
    source_start_line: int | None,
) -> tuple[WholeProgramSourceBytecodeLineMap, ...]:
    """Return deterministic source-line to bytecode crosswalk metadata."""

    offsets_by_line: dict[int, set[int]] = {}
    absolute_by_line: dict[int, set[int]] = {}
    for instruction in bytecode_instructions:
        if instruction.line_number is not None:
            absolute_line = instruction.line_number
            source_line = _source_relative_line(absolute_line, source_start_line)
            if source_start_line is not None and instruction.line_number < source_start_line:
                absolute_line = source_start_line + source_line - 1
            offsets_by_line.setdefault(source_line, set()).add(instruction.offset)
            absolute_by_line.setdefault(source_line, set()).add(absolute_line)

    features_by_line: dict[int, set[str]] = {}
    for feature in source_ir_features:
        features_by_line.setdefault(feature.line_number, set()).add(feature.kind)

    rows: list[WholeProgramSourceBytecodeLineMap] = []
    for line_number in sorted(offsets_by_line):
        region_ids = tuple(
            sorted(
                region.region_id
                for region in source_regions
                if region.line_start <= line_number <= region.line_end
            )
        )
        rows.append(
            WholeProgramSourceBytecodeLineMap(
                line_number=line_number,
                absolute_line_number=_single_absolute_line(absolute_by_line[line_number]),
                instruction_offsets=tuple(sorted(offsets_by_line[line_number])),
                region_ids=region_ids,
                feature_kinds=tuple(sorted(features_by_line.get(line_number, set()))),
            )
        )
    return tuple(rows)


def _symbol_scope_entries(
    *,
    objective: Callable[..., object],
    source: str | None,
    bytecode_instructions: Sequence[WholeProgramBytecodeInstruction],
    source_regions: Sequence[WholeProgramSourceRegion],
    source_start_line: int | None,
) -> tuple[WholeProgramSymbolScopeEntry, ...]:
    """Return deterministic static symbol-scope metadata for a callable."""

    symbol_roles: dict[str, set[str]] = {}
    symbol_lines: dict[str, set[int]] = {}
    symbol_offsets: dict[str, set[int]] = {}
    symbol_regions: dict[str, set[str]] = {}

    def ensure_symbol(symbol: str) -> None:
        symbol_roles.setdefault(symbol, set())
        symbol_lines.setdefault(symbol, set())
        symbol_offsets.setdefault(symbol, set())
        symbol_regions.setdefault(symbol, set())

    def add_symbol(
        symbol: str,
        role: str,
        *,
        line_number: int | None = None,
        bytecode_offset: int | None = None,
        line_is_absolute: bool = True,
    ) -> None:
        if not symbol or not role:
            return
        ensure_symbol(symbol)
        symbol_roles[symbol].add(role)
        if line_number is not None and line_number > 0:
            source_line = (
                _source_relative_line(line_number, source_start_line)
                if line_is_absolute
                else line_number
            )
            symbol_lines[symbol].add(source_line)
            symbol_regions[symbol].update(_source_region_ids_for_line(source_regions, source_line))
        if bytecode_offset is not None and bytecode_offset >= 0:
            symbol_offsets[symbol].add(bytecode_offset)

    try:
        signature = inspect.signature(objective)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for parameter in signature.parameters:
            add_symbol(parameter, "parameter")

    code = getattr(objective, "__code__", None)
    if code is not None:
        argcount = int(code.co_argcount) + int(code.co_kwonlyargcount)
        for symbol in code.co_varnames[:argcount]:
            add_symbol(str(symbol), "parameter")
        for symbol in code.co_varnames[argcount:]:
            add_symbol(str(symbol), "local")
        for symbol in code.co_freevars:
            add_symbol(str(symbol), "closure")
        for symbol in code.co_cellvars:
            add_symbol(str(symbol), "cell")
        for symbol in code.co_names:
            add_symbol(str(symbol), "global_or_attribute")

    for instruction in bytecode_instructions:
        symbol = _bytecode_symbol_name(instruction)
        if symbol is not None:
            add_symbol(
                symbol,
                _bytecode_symbol_role(instruction.opname),
                line_number=instruction.line_number,
                bytecode_offset=instruction.offset,
            )

    if source is not None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.arg):
                    add_symbol(
                        node.arg,
                        "parameter",
                        line_number=getattr(node, "lineno", None),
                        line_is_absolute=False,
                    )
                elif isinstance(node, ast.Name):
                    add_symbol(
                        node.id,
                        _ast_name_role(node.ctx),
                        line_number=getattr(node, "lineno", None),
                        line_is_absolute=False,
                    )

    entries: list[WholeProgramSymbolScopeEntry] = []
    for symbol in sorted(symbol_roles):
        roles = tuple(sorted(symbol_roles[symbol]))
        if not roles:
            continue
        entries.append(
            WholeProgramSymbolScopeEntry(
                symbol=symbol,
                roles=roles,
                line_numbers=tuple(sorted(symbol_lines[symbol])),
                bytecode_offsets=tuple(sorted(symbol_offsets[symbol])),
                region_ids=tuple(sorted(symbol_regions[symbol])),
            )
        )
    return tuple(entries)


def _source_region_ids_for_line(
    source_regions: Sequence[WholeProgramSourceRegion],
    line_number: int,
) -> tuple[str, ...]:
    """Return source-region identifiers covering a source line."""

    return tuple(
        sorted(
            region.region_id
            for region in source_regions
            if region.line_start <= line_number <= region.line_end
        )
    )


def _source_relative_line(line_number: int, source_start_line: int | None) -> int:
    """Return a source-relative line for a CPython absolute line number."""

    if source_start_line is None:
        return line_number
    relative_line = line_number - source_start_line + 1
    return relative_line if relative_line > 0 else line_number


def _single_absolute_line(line_numbers: set[int]) -> int | None:
    """Return one absolute line when all bytecode offsets share it."""

    sorted_lines = sorted(line_numbers)
    if len(sorted_lines) != 1:
        return None
    return sorted_lines[0]


def _bytecode_symbol_name(instruction: WholeProgramBytecodeInstruction) -> str | None:
    """Return the static symbol operand for a bytecode instruction."""

    if not (
        instruction.opname.endswith("_FAST")
        or instruction.opname.endswith("_DEREF")
        or instruction.opname.endswith("_GLOBAL")
        or instruction.opname.endswith("_NAME")
        or instruction.opname in {"LOAD_ATTR", "STORE_ATTR", "DELETE_ATTR"}
    ):
        return None
    pieces = instruction.argrepr.replace("(", " ").replace(")", " ").replace("+", " ").split()
    if not pieces:
        return None
    for piece in pieces:
        if piece.isidentifier() and piece != "NULL":
            return piece
    return None


def _bytecode_symbol_role(opname: str) -> str:
    """Return a source-like symbol role for a bytecode operation name."""

    if opname.startswith("LOAD"):
        return "bytecode_load"
    if opname.startswith("STORE"):
        return "bytecode_store"
    if opname.startswith("DELETE"):
        return "bytecode_delete"
    return "bytecode_reference"


def _ast_name_role(context: ast.expr_context) -> str:
    """Return a symbol role for an AST name context."""

    if isinstance(context, ast.Load):
        return "source_load"
    if isinstance(context, ast.Store):
        return "source_store"
    if isinstance(context, ast.Del):
        return "source_delete"
    return "source_reference"


def _frontend_digest(
    *,
    source: str | None,
    source_start_line: int | None,
    source_end_line: int | None,
    bytecode_instructions: Sequence[WholeProgramBytecodeInstruction],
    bytecode_basic_blocks: Sequence[WholeProgramBytecodeBasicBlock],
    source_ir_features: Sequence[WholeProgramSourceIRFeature],
    source_regions: Sequence[WholeProgramSourceRegion],
    source_bytecode_line_map: Sequence[WholeProgramSourceBytecodeLineMap],
    symbol_scope_entries: Sequence[WholeProgramSymbolScopeEntry],
    unsupported_semantic_diagnostics: Sequence[WholeProgramUnsupportedSemanticDiagnostic],
    semantics_report: WholeProgramSemanticsReport,
    hard_gaps: tuple[str, ...],
) -> str:
    """Return a stable digest for the complete static frontend payload."""

    payload = {
        "source_sha256": None
        if source is None
        else hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_start_line": source_start_line,
        "source_end_line": source_end_line,
        "bytecode_instructions": [
            {
                "offset": instruction.offset,
                "opname": instruction.opname,
                "argrepr": instruction.argrepr,
                "line_number": instruction.line_number,
            }
            for instruction in bytecode_instructions
        ],
        "bytecode_basic_blocks": [block.to_dict() for block in bytecode_basic_blocks],
        "source_ir_features": [
            {
                "kind": feature.kind,
                "detail": feature.detail,
                "line_number": feature.line_number,
            }
            for feature in source_ir_features
        ],
        "source_regions": [region.to_dict() for region in source_regions],
        "source_bytecode_line_map": [line_map.to_dict() for line_map in source_bytecode_line_map],
        "symbol_scope_entries": [entry.to_dict() for entry in symbol_scope_entries],
        "unsupported_semantic_diagnostics": [
            diagnostic.to_dict() for diagnostic in unsupported_semantic_diagnostics
        ],
        "semantics": {
            "accepted": list(semantics_report.accepted_python_semantics),
            "unsupported": list(semantics_report.unsupported_python_semantics),
            "bytecode_frontend": semantics_report.bytecode_frontend,
            "source_frontend": semantics_report.source_frontend,
        },
        "hard_gaps": list(hard_gaps),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_ast_node_count(source: str | None) -> int:
    if source is None:
        return 0
    try:
        return sum(1 for _node in ast.walk(ast.parse(source)))
    except SyntaxError:
        return 0


def _source_parse_failed(source: str | None) -> bool:
    if source is None:
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return True
    return False


def _source_has_control_flow(source: str | None) -> bool:
    """Return whether source contains explicit Python control-flow nodes."""

    return _source_has_node(source, ast.If, ast.For, ast.While, ast.IfExp)


def _source_has_node(source: str | None, *node_types: type[ast.AST]) -> bool:
    """Return whether source contains any AST node of the requested types."""

    if source is None:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tokens = {
            ast.If: "if ",
            ast.For: "for ",
            ast.While: "while ",
            ast.IfExp: " if ",
            ast.GeneratorExp: " for ",
        }
        return any(
            token in source
            for node_type in node_types
            for token in (tokens.get(node_type),)
            if token
        )
    return any(isinstance(node, node_types) for node in ast.walk(tree))


def _source_mentions_numpy(source: str | None) -> bool:
    """Return whether a source fragment visibly references NumPy."""

    if source is None:
        return False
    return "np." in source or "numpy." in source


__all__ = [
    "WholeProgramBytecodeInstruction",
    "WholeProgramBytecodeBasicBlock",
    "WholeProgramCompilerFrontendReport",
    "WholeProgramSemanticsReport",
    "WholeProgramSourceBytecodeLineMap",
    "WholeProgramSourceIRFeature",
    "WholeProgramSourceRegion",
    "WholeProgramSymbolScopeEntry",
    "WholeProgramUnsupportedSemanticDiagnostic",
    "compile_whole_program_frontend",
]
