# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program frontend contracts module
# scpn-quantum-control -- whole-program frontend contracts
"""Immutable IR and report contracts for static whole-program frontend inspection."""

from __future__ import annotations

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
]
