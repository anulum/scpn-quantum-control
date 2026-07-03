# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD alias and static-lattice analysis
"""Program AD alias/effect summaries and static alias-lattice reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .program_ad_effect_ir import ProgramADAliasEdge, ProgramADEffectIR
from .whole_program_frontend import WholeProgramUnsupportedSemanticDiagnostic

PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY = "metadata_only_no_general_alias_lattice"
PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY = (
    "static_alias_lattice_over_emitted_program_ad_ir_no_non_executed_branch_semantics"
)
_PROGRAM_AD_SUPPORTED_ALIAS_EDGE_KINDS = frozenset(
    {
        "alias_analysis",
        "control_path_alias",
        "expression_rebinding_alias",
        "list_alias",
        "local_rebinding_alias",
        "loop_carried_state",
        "mutation_version",
        "object_attribute_alias",
        "source_alias",
        "view_alias",
    }
)
_PROGRAM_AD_LIST_ALIAS_TARGET_KINDS = frozenset({"indexed_mutation_source", "local_name"})
_PROGRAM_AD_CONTROL_PATH_ALIAS_BRANCH_ARMS = frozenset({"body", "orelse"})
_PROGRAM_AD_REBINDING_ALIAS_EDGE_KINDS = frozenset(
    {"expression_rebinding_alias", "local_rebinding_alias"}
)


@dataclass(frozen=True)
class ProgramADAliasSet:
    """One deterministic alias component derived from Program AD effect IR."""

    index: int
    members: tuple[str, ...]
    versions: tuple[int, ...]
    mutation_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate alias-set metadata at construction time."""

        if self.index < 0:
            raise ValueError("program AD alias set index must be non-negative")
        if any(not isinstance(member, str) or not member for member in self.members):
            raise ValueError("program AD alias set members must be non-empty strings")
        if tuple(sorted(self.members)) != self.members:
            raise ValueError("program AD alias set members must be sorted deterministically")
        if any(version < 0 for version in self.versions):
            raise ValueError("program AD alias set versions must be non-negative")
        if any(version < 0 for version in self.mutation_versions):
            raise ValueError("program AD alias set mutation_versions must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready alias-set payload."""

        return {
            "index": self.index,
            "members": list(self.members),
            "versions": list(self.versions),
            "mutation_versions": list(self.mutation_versions),
        }


@dataclass(frozen=True)
class ProgramADAliasEffectAnalysis:
    """Deterministic metadata-only alias/effect analysis for Program AD IR."""

    alias_sets: tuple[ProgramADAliasSet, ...]
    mutation_effects: tuple[int, ...]
    alias_edges: tuple[ProgramADAliasEdge, ...]
    unknown_aliasing: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate alias/effect analysis contents at construction time."""

        if any(not isinstance(alias_set, ProgramADAliasSet) for alias_set in self.alias_sets):
            raise ValueError("program AD alias analysis alias_sets must contain ProgramADAliasSet")
        if tuple(sorted(self.mutation_effects)) != self.mutation_effects:
            raise ValueError("program AD alias analysis mutation_effects must be sorted")
        if any(effect < 0 for effect in self.mutation_effects):
            raise ValueError("program AD alias analysis mutation_effects must be non-negative")
        if any(not isinstance(edge, ProgramADAliasEdge) for edge in self.alias_edges):
            raise ValueError(
                "program AD alias analysis alias_edges must contain ProgramADAliasEdge"
            )
        if not isinstance(self.unknown_aliasing, bool):
            raise ValueError("program AD alias analysis unknown_aliasing must be boolean")
        _normalise_claim_boundary("program AD alias analysis", self.claim_boundary)

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready alias/effect analysis payload."""

        return {
            "alias_sets": [alias_set.as_dict() for alias_set in self.alias_sets],
            "mutation_effects": list(self.mutation_effects),
            "alias_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind,
                    "version": edge.version,
                }
                for edge in self.alias_edges
            ],
            "unknown_aliasing": self.unknown_aliasing,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ProgramADStaticAliasLatticeComponent:
    """One component in the static alias lattice derived from emitted Program AD IR."""

    index: int
    members: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    versions: tuple[int, ...]
    mutation_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate static alias-lattice component metadata at construction time."""

        if self.index < 0:
            raise ValueError("program AD static alias component index must be non-negative")
        if any(not isinstance(member, str) or not member for member in self.members):
            raise ValueError("program AD static alias component members must be non-empty strings")
        if tuple(sorted(self.members)) != self.members:
            raise ValueError(
                "program AD static alias component members must be sorted deterministically"
            )
        if any(not isinstance(kind, str) or not kind for kind in self.edge_kinds):
            raise ValueError(
                "program AD static alias component edge_kinds must be non-empty strings"
            )
        if tuple(sorted(set(self.edge_kinds))) != self.edge_kinds:
            raise ValueError(
                "program AD static alias component edge_kinds must be sorted and unique"
            )
        if any(version < 0 for version in self.versions):
            raise ValueError("program AD static alias component versions must be non-negative")
        if any(version < 0 for version in self.mutation_versions):
            raise ValueError(
                "program AD static alias component mutation_versions must be non-negative"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready static alias lattice component."""

        return {
            "index": self.index,
            "members": list(self.members),
            "edge_kinds": list(self.edge_kinds),
            "versions": list(self.versions),
            "mutation_versions": list(self.mutation_versions),
        }


@dataclass(frozen=True)
class ProgramADUnknownAliasEdge:
    """Unsupported alias edge preserved as fail-closed static-lattice provenance."""

    source: str
    target: str
    kind: str
    version: int

    def __post_init__(self) -> None:
        """Validate unknown alias-edge provenance at construction time."""

        if not isinstance(self.source, str) or not self.source:
            raise ValueError("program AD unknown alias edge source must be non-empty")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("program AD unknown alias edge target must be non-empty")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD unknown alias edge kind must be non-empty")
        if self.kind in _PROGRAM_AD_SUPPORTED_ALIAS_EDGE_KINDS:
            raise ValueError("program AD unknown alias edge kind must be unsupported")
        if self.version < 0:
            raise ValueError("program AD unknown alias edge version must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready unknown alias-edge provenance payload."""

        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "version": self.version,
        }


@dataclass(frozen=True)
class ProgramADViewAliasProvenance:
    """Parseable source-to-view alias edge preserved by the static lattice."""

    source: str
    target: str
    operation: str
    view_id: int
    output_index: int
    version: int

    def __post_init__(self) -> None:
        """Validate view-alias provenance at construction time."""

        if not isinstance(self.source, str) or not self.source:
            raise ValueError("program AD view alias provenance source must be non-empty")
        if not isinstance(self.target, str) or not self.target.startswith("view:"):
            raise ValueError("program AD view alias provenance target must be a view marker")
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("program AD view alias provenance operation must be non-empty")
        if self.view_id < 0:
            raise ValueError("program AD view alias provenance view_id must be non-negative")
        if self.output_index < 0:
            raise ValueError("program AD view alias provenance output_index must be non-negative")
        if self.version < 0:
            raise ValueError("program AD view alias provenance version must be non-negative")
        expected_target = f"view:{self.operation}:{self.view_id}[{self.output_index}]"
        if self.target != expected_target:
            raise ValueError(
                "program AD view alias provenance target must match operation/view_id/output_index"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view-alias provenance payload."""

        return {
            "source": self.source,
            "target": self.target,
            "operation": self.operation,
            "view_id": self.view_id,
            "output_index": self.output_index,
            "version": self.version,
        }


@dataclass(frozen=True)
class ProgramADListAliasProvenance:
    """Parseable local list-alias edge preserved by the static lattice."""

    source: str
    target: str
    list_name: str
    target_kind: str
    version: int

    def __post_init__(self) -> None:
        """Validate list-alias provenance at construction time."""

        if not isinstance(self.source, str) or not self.source.startswith("list:"):
            raise ValueError("program AD list alias provenance source must be a list marker")
        if self.source == "list:":
            raise ValueError("program AD list alias provenance source must include a list name")
        if not isinstance(self.list_name, str) or not self.list_name:
            raise ValueError("program AD list alias provenance list_name must be non-empty")
        if self.source != f"list:{self.list_name}":
            raise ValueError("program AD list alias provenance list_name must match source")
        if self.target_kind not in _PROGRAM_AD_LIST_ALIAS_TARGET_KINDS:
            raise ValueError("program AD list alias provenance target_kind is unsupported")
        if self.target_kind == "local_name" and (
            not isinstance(self.target, str) or not self.target.startswith("name:")
        ):
            raise ValueError("program AD list alias provenance target must be a local name")
        if self.target_kind == "local_name" and self.target == "name:":
            raise ValueError("program AD list alias provenance target must name a local")
        if self.target_kind == "indexed_mutation_source" and self.target != "source:list_mutation":
            raise ValueError(
                "program AD list alias provenance mutation target must be source:list_mutation"
            )
        if self.version < 0:
            raise ValueError("program AD list alias provenance version must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready list-alias provenance payload."""

        return {
            "source": self.source,
            "target": self.target,
            "list_name": self.list_name,
            "target_kind": self.target_kind,
            "version": self.version,
        }


@dataclass(frozen=True)
class ProgramADLoopCarriedStateProvenance:
    """Parseable loop-carried scalar state edge preserved by the static lattice."""

    source: str
    target: str
    state_name: str
    entry_label: str
    backedge_label: str
    version: int

    def __post_init__(self) -> None:
        """Validate loop-carried state provenance at construction time."""

        if not isinstance(self.source, str) or not self.source.startswith("loop:"):
            raise ValueError(
                "program AD loop-carried state provenance source must be loop:<state>:entry"
            )
        if not isinstance(self.target, str) or not self.target.startswith("loop:"):
            raise ValueError(
                "program AD loop-carried state provenance target must be loop:<state>:backedge"
            )
        if not isinstance(self.state_name, str) or not self.state_name:
            raise ValueError(
                "program AD loop-carried state provenance state_name must be non-empty"
            )
        if self.entry_label != "entry":
            raise ValueError("program AD loop-carried state provenance entry_label must be entry")
        if self.backedge_label != "backedge":
            raise ValueError(
                "program AD loop-carried state provenance backedge_label must be backedge"
            )
        if self.source != f"loop:{self.state_name}:{self.entry_label}":
            raise ValueError(
                "program AD loop-carried state provenance source must match state_name"
            )
        if self.target != f"loop:{self.state_name}:{self.backedge_label}":
            raise ValueError(
                "program AD loop-carried state provenance target must match state_name"
            )
        if self.version < 0:
            raise ValueError(
                "program AD loop-carried state provenance version must be non-negative"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready loop-carried state provenance payload."""

        return {
            "source": self.source,
            "target": self.target,
            "state_name": self.state_name,
            "entry_label": self.entry_label,
            "backedge_label": self.backedge_label,
            "version": self.version,
        }


@dataclass(frozen=True)
class ProgramADControlPathAliasProvenance:
    """Parseable branch-local control-path alias edge preserved by the static lattice."""

    source: str
    target: str
    branch_line: int
    branch_arm: str
    target_label: str
    version: int

    def __post_init__(self) -> None:
        """Validate control-path alias provenance at construction time."""

        if not isinstance(self.source, str) or not self.source.startswith("control:if:"):
            raise ValueError(
                "program AD control-path alias provenance source must be "
                "control:if:<line>:<body|orelse>"
            )
        if not isinstance(self.target, str) or not self.target.startswith("control:"):
            raise ValueError("program AD control-path alias provenance target must be control:*")
        if self.target == "control:":
            raise ValueError(
                "program AD control-path alias provenance target must include a label"
            )
        if self.branch_line <= 0:
            raise ValueError(
                "program AD control-path alias provenance branch_line must be positive"
            )
        if self.branch_arm not in _PROGRAM_AD_CONTROL_PATH_ALIAS_BRANCH_ARMS:
            raise ValueError("program AD control-path alias provenance branch_arm is unsupported")
        if not isinstance(self.target_label, str) or not self.target_label:
            raise ValueError(
                "program AD control-path alias provenance target_label must be non-empty"
            )
        expected_source = f"control:if:{self.branch_line}:{self.branch_arm}"
        if self.source != expected_source:
            raise ValueError(
                "program AD control-path alias provenance source must match "
                "branch_line and branch_arm"
            )
        if self.target != f"control:{self.target_label}":
            raise ValueError(
                "program AD control-path alias provenance target_label must match target"
            )
        if self.version < 0:
            raise ValueError(
                "program AD control-path alias provenance version must be non-negative"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready control-path alias provenance payload."""

        return {
            "source": self.source,
            "target": self.target,
            "branch_line": self.branch_line,
            "branch_arm": self.branch_arm,
            "target_label": self.target_label,
            "version": self.version,
        }


@dataclass(frozen=True)
class ProgramADRebindingAliasProvenance:
    """Parseable local or expression rebinding alias preserved by the static lattice."""

    source: str
    target: str
    binding_kind: str
    source_name: str | None
    expression_line: int | None
    expression_label: str | None
    target_name: str
    version: int

    def __post_init__(self) -> None:
        """Validate rebinding-alias provenance at construction time."""

        if self.binding_kind not in {"expression", "local"}:
            raise ValueError("program AD rebinding alias provenance binding_kind is unsupported")
        if not isinstance(self.target, str) or not self.target.startswith("name:"):
            raise ValueError("program AD rebinding alias provenance target must be a local name")
        if self.target == "name:":
            raise ValueError("program AD rebinding alias provenance target must name a local")
        if not isinstance(self.target_name, str) or not self.target_name:
            raise ValueError("program AD rebinding alias provenance target_name must be non-empty")
        if self.target != f"name:{self.target_name}":
            raise ValueError("program AD rebinding alias provenance target_name must match target")
        if self.binding_kind == "local":
            if not isinstance(self.source, str) or not self.source.startswith("name:"):
                raise ValueError(
                    "program AD local rebinding alias provenance source must be a local name"
                )
            if self.source == "name:":
                raise ValueError(
                    "program AD local rebinding alias provenance source must name a local"
                )
            if not isinstance(self.source_name, str) or not self.source_name:
                raise ValueError(
                    "program AD local rebinding alias provenance source_name must be non-empty"
                )
            if self.source != f"name:{self.source_name}":
                raise ValueError(
                    "program AD local rebinding alias provenance source_name must match source"
                )
            if self.expression_line is not None or self.expression_label is not None:
                raise ValueError(
                    "program AD local rebinding alias provenance cannot carry expression metadata"
                )
        if self.binding_kind == "expression":
            if self.source_name is not None:
                raise ValueError(
                    "program AD expression rebinding alias provenance cannot carry source_name"
                )
            if self.expression_line is None or self.expression_line <= 0:
                raise ValueError(
                    "program AD expression rebinding alias provenance "
                    "expression_line must be positive"
                )
            if not isinstance(self.expression_label, str) or not self.expression_label:
                raise ValueError(
                    "program AD expression rebinding alias provenance "
                    "expression_label must be non-empty"
                )
            if self.source != f"expr:{self.expression_line}:{self.expression_label}":
                raise ValueError(
                    "program AD expression rebinding alias provenance source must match "
                    "expression_line and expression_label"
                )
        if self.version < 0:
            raise ValueError("program AD rebinding alias provenance version must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready rebinding-alias provenance payload."""

        return {
            "source": self.source,
            "target": self.target,
            "binding_kind": self.binding_kind,
            "source_name": self.source_name,
            "expression_line": self.expression_line,
            "expression_label": self.expression_label,
            "target_name": self.target_name,
            "version": self.version,
        }


@dataclass(frozen=True)
class ProgramADStaticAliasLatticeReport:
    """Static alias-lattice readiness report for emitted Program AD IR.

    The report builds deterministic alias components from the emitted
    ``program_ad_effect_ir.v1`` metadata and records the exact blockers that
    prevent promotion to full static alias, mutation, or non-executed branch
    semantics. Unsupported whole-program frontend diagnostics are preserved as
    static source/region/bytecode provenance for the corresponding blockers.
    """

    components: tuple[ProgramADStaticAliasLatticeComponent, ...]
    mutation_effects: tuple[int, ...]
    non_executed_phi_nodes: tuple[int, ...]
    non_executed_control_alias_edges: tuple[str, ...]
    unknown_alias_edge_kinds: tuple[str, ...]
    blocker_reasons: tuple[str, ...]
    complete: bool
    claim_boundary: str
    control_path_alias_provenance: tuple[ProgramADControlPathAliasProvenance, ...] = ()
    malformed_control_path_alias_edges: tuple[str, ...] = ()
    unsupported_python_semantics: tuple[str, ...] = ()
    unsupported_semantic_diagnostics: tuple[WholeProgramUnsupportedSemanticDiagnostic, ...] = ()
    unsupported_object_attribute_roots: tuple[str, ...] = ()
    unsupported_object_attribute_details: tuple[str, ...] = ()
    unknown_alias_edges: tuple[ProgramADUnknownAliasEdge, ...] = ()
    view_alias_provenance: tuple[ProgramADViewAliasProvenance, ...] = ()
    malformed_view_alias_edges: tuple[str, ...] = ()
    list_alias_provenance: tuple[ProgramADListAliasProvenance, ...] = ()
    malformed_list_alias_edges: tuple[str, ...] = ()
    loop_carried_state_provenance: tuple[ProgramADLoopCarriedStateProvenance, ...] = ()
    malformed_loop_carried_state_edges: tuple[str, ...] = ()
    rebinding_alias_provenance: tuple[ProgramADRebindingAliasProvenance, ...] = ()
    malformed_rebinding_alias_edges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate static alias-lattice report contents at construction time."""

        if any(
            not isinstance(component, ProgramADStaticAliasLatticeComponent)
            for component in self.components
        ):
            raise ValueError(
                "program AD static alias lattice components must contain "
                "ProgramADStaticAliasLatticeComponent entries"
            )
        if tuple(component.index for component in self.components) != tuple(
            range(len(self.components))
        ):
            raise ValueError("program AD static alias lattice components must be dense")
        for name in ("mutation_effects", "non_executed_phi_nodes"):
            values = getattr(self, name)
            if tuple(sorted(values)) != values or any(value < 0 for value in values):
                raise ValueError(
                    f"program AD static alias lattice {name} must be sorted non-negative ints"
                )
        if any(
            not isinstance(edge, str) or not edge for edge in self.non_executed_control_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice non_executed_control_alias_edges "
                "must be non-empty strings"
            )
        if (
            tuple(sorted(set(self.non_executed_control_alias_edges)))
            != self.non_executed_control_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice non_executed_control_alias_edges "
                "must be sorted and unique"
            )
        if any(
            not isinstance(row, ProgramADControlPathAliasProvenance)
            for row in self.control_path_alias_provenance
        ):
            raise ValueError(
                "program AD static alias lattice control_path_alias_provenance must "
                "contain ProgramADControlPathAliasProvenance entries"
            )
        control_path_alias_order = tuple(
            (
                row.branch_line,
                row.branch_arm,
                row.target_label,
                row.source,
                row.target,
                row.version,
            )
            for row in self.control_path_alias_provenance
        )
        if tuple(sorted(set(control_path_alias_order))) != control_path_alias_order:
            raise ValueError(
                "program AD static alias lattice control_path_alias_provenance must "
                "be sorted unique"
            )
        control_path_alias_labels = set(self.non_executed_control_alias_edges)
        if any(
            f"{row.source}->{row.target}" not in control_path_alias_labels
            for row in self.control_path_alias_provenance
        ):
            raise ValueError(
                "program AD static alias lattice control_path_alias_provenance must "
                "match non_executed_control_alias_edges"
            )
        if any(
            not isinstance(edge, str) or not edge
            for edge in self.malformed_control_path_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice malformed_control_path_alias_edges "
                "must contain non-empty strings"
            )
        if (
            tuple(sorted(set(self.malformed_control_path_alias_edges)))
            != self.malformed_control_path_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice malformed_control_path_alias_edges "
                "must be sorted and unique"
            )
        if any(not isinstance(kind, str) or not kind for kind in self.unknown_alias_edge_kinds):
            raise ValueError(
                "program AD static alias lattice unknown_alias_edge_kinds must be strings"
            )
        if tuple(sorted(set(self.unknown_alias_edge_kinds))) != self.unknown_alias_edge_kinds:
            raise ValueError(
                "program AD static alias lattice unknown_alias_edge_kinds must be sorted unique"
            )
        if any(
            not isinstance(edge, ProgramADUnknownAliasEdge) for edge in self.unknown_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice unknown_alias_edges must contain "
                "ProgramADUnknownAliasEdge entries"
            )
        unknown_edge_order = tuple(
            (edge.kind, edge.source, edge.target, edge.version)
            for edge in self.unknown_alias_edges
        )
        if tuple(sorted(set(unknown_edge_order))) != unknown_edge_order:
            raise ValueError(
                "program AD static alias lattice unknown_alias_edges must be sorted unique"
            )
        expected_unknown_alias_edge_kinds = tuple(
            sorted({edge.kind for edge in self.unknown_alias_edges})
        )
        if self.unknown_alias_edge_kinds != expected_unknown_alias_edge_kinds:
            raise ValueError(
                "program AD static alias lattice unknown_alias_edge_kinds must match "
                "unknown_alias_edges"
            )
        if any(
            not isinstance(row, ProgramADViewAliasProvenance) for row in self.view_alias_provenance
        ):
            raise ValueError(
                "program AD static alias lattice view_alias_provenance must contain "
                "ProgramADViewAliasProvenance entries"
            )
        view_alias_order = tuple(
            (
                row.operation,
                row.view_id,
                row.output_index,
                row.source,
                row.target,
                row.version,
            )
            for row in self.view_alias_provenance
        )
        if tuple(sorted(set(view_alias_order))) != view_alias_order:
            raise ValueError(
                "program AD static alias lattice view_alias_provenance must be sorted unique"
            )
        if any(not isinstance(edge, str) or not edge for edge in self.malformed_view_alias_edges):
            raise ValueError(
                "program AD static alias lattice malformed_view_alias_edges "
                "must contain non-empty strings"
            )
        if tuple(sorted(set(self.malformed_view_alias_edges))) != self.malformed_view_alias_edges:
            raise ValueError(
                "program AD static alias lattice malformed_view_alias_edges "
                "must be sorted and unique"
            )
        if any(
            not isinstance(row, ProgramADListAliasProvenance) for row in self.list_alias_provenance
        ):
            raise ValueError(
                "program AD static alias lattice list_alias_provenance must contain "
                "ProgramADListAliasProvenance entries"
            )
        list_alias_order = tuple(
            (
                row.list_name,
                row.target_kind,
                row.source,
                row.target,
                row.version,
            )
            for row in self.list_alias_provenance
        )
        if tuple(sorted(set(list_alias_order))) != list_alias_order:
            raise ValueError(
                "program AD static alias lattice list_alias_provenance must be sorted unique"
            )
        if any(not isinstance(edge, str) or not edge for edge in self.malformed_list_alias_edges):
            raise ValueError(
                "program AD static alias lattice malformed_list_alias_edges "
                "must contain non-empty strings"
            )
        if tuple(sorted(set(self.malformed_list_alias_edges))) != self.malformed_list_alias_edges:
            raise ValueError(
                "program AD static alias lattice malformed_list_alias_edges "
                "must be sorted and unique"
            )
        if any(
            not isinstance(row, ProgramADLoopCarriedStateProvenance)
            for row in self.loop_carried_state_provenance
        ):
            raise ValueError(
                "program AD static alias lattice loop_carried_state_provenance must "
                "contain ProgramADLoopCarriedStateProvenance entries"
            )
        loop_carried_state_order = tuple(
            (
                row.state_name,
                row.entry_label,
                row.backedge_label,
                row.source,
                row.target,
                row.version,
            )
            for row in self.loop_carried_state_provenance
        )
        if tuple(sorted(set(loop_carried_state_order))) != loop_carried_state_order:
            raise ValueError(
                "program AD static alias lattice loop_carried_state_provenance "
                "must be sorted unique"
            )
        if any(
            not isinstance(edge, str) or not edge
            for edge in self.malformed_loop_carried_state_edges
        ):
            raise ValueError(
                "program AD static alias lattice malformed_loop_carried_state_edges "
                "must contain non-empty strings"
            )
        if (
            tuple(sorted(set(self.malformed_loop_carried_state_edges)))
            != self.malformed_loop_carried_state_edges
        ):
            raise ValueError(
                "program AD static alias lattice malformed_loop_carried_state_edges "
                "must be sorted and unique"
            )
        if any(
            not isinstance(row, ProgramADRebindingAliasProvenance)
            for row in self.rebinding_alias_provenance
        ):
            raise ValueError(
                "program AD static alias lattice rebinding_alias_provenance must contain "
                "ProgramADRebindingAliasProvenance entries"
            )
        rebinding_alias_order = tuple(
            (
                row.binding_kind,
                row.source,
                row.target,
                row.source_name or "",
                row.expression_line or 0,
                row.expression_label or "",
                row.version,
            )
            for row in self.rebinding_alias_provenance
        )
        if tuple(sorted(set(rebinding_alias_order))) != rebinding_alias_order:
            raise ValueError(
                "program AD static alias lattice rebinding_alias_provenance must be sorted unique"
            )
        if any(
            not isinstance(edge, str) or not edge for edge in self.malformed_rebinding_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice malformed_rebinding_alias_edges "
                "must contain non-empty strings"
            )
        if (
            tuple(sorted(set(self.malformed_rebinding_alias_edges)))
            != self.malformed_rebinding_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice malformed_rebinding_alias_edges "
                "must be sorted and unique"
            )
        has_view_alias_component = any(
            "view_alias" in component.edge_kinds for component in self.components
        )
        has_list_alias_component = any(
            "list_alias" in component.edge_kinds for component in self.components
        )
        has_loop_carried_state_component = any(
            "loop_carried_state" in component.edge_kinds for component in self.components
        )
        has_rebinding_alias_component = any(
            bool(_PROGRAM_AD_REBINDING_ALIAS_EDGE_KINDS.intersection(component.edge_kinds))
            for component in self.components
        )
        has_control_path_alias_component = any(
            "control_path_alias" in component.edge_kinds for component in self.components
        )
        if (
            has_control_path_alias_component
            and not self.control_path_alias_provenance
            and not self.malformed_control_path_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice control-path alias components require "
                "control_path_alias_provenance"
            )
        if self.control_path_alias_provenance and not has_control_path_alias_component:
            raise ValueError(
                "program AD static alias lattice control-path alias provenance requires "
                "a control_path_alias component"
            )
        if (
            has_view_alias_component
            and not self.view_alias_provenance
            and not self.malformed_view_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice view alias components require "
                "view_alias_provenance"
            )
        if self.view_alias_provenance and not has_view_alias_component:
            raise ValueError(
                "program AD static alias lattice view alias provenance requires "
                "a view_alias component"
            )
        if (
            has_list_alias_component
            and not self.list_alias_provenance
            and not self.malformed_list_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice list alias components require "
                "list_alias_provenance"
            )
        if self.list_alias_provenance and not has_list_alias_component:
            raise ValueError(
                "program AD static alias lattice list alias provenance requires "
                "a list_alias component"
            )
        if (
            has_loop_carried_state_component
            and not self.loop_carried_state_provenance
            and not self.malformed_loop_carried_state_edges
        ):
            raise ValueError(
                "program AD static alias lattice loop-carried state components require "
                "loop_carried_state_provenance"
            )
        if self.loop_carried_state_provenance and not has_loop_carried_state_component:
            raise ValueError(
                "program AD static alias lattice loop-carried state provenance requires "
                "a loop_carried_state component"
            )
        if (
            has_rebinding_alias_component
            and not self.rebinding_alias_provenance
            and not self.malformed_rebinding_alias_edges
        ):
            raise ValueError(
                "program AD static alias lattice rebinding alias components require "
                "rebinding_alias_provenance"
            )
        if self.rebinding_alias_provenance and not has_rebinding_alias_component:
            raise ValueError(
                "program AD static alias lattice rebinding alias provenance requires "
                "a rebinding alias component"
            )
        if any(
            not isinstance(semantic, str) or not semantic
            for semantic in self.unsupported_python_semantics
        ):
            raise ValueError(
                "program AD static alias lattice unsupported_python_semantics "
                "must be non-empty strings"
            )
        if (
            tuple(sorted(set(self.unsupported_python_semantics)))
            != self.unsupported_python_semantics
        ):
            raise ValueError(
                "program AD static alias lattice unsupported_python_semantics "
                "must be sorted and unique"
            )
        if any(
            not isinstance(diagnostic, WholeProgramUnsupportedSemanticDiagnostic)
            for diagnostic in self.unsupported_semantic_diagnostics
        ):
            raise ValueError(
                "program AD static alias lattice unsupported_semantic_diagnostics "
                "must contain WholeProgramUnsupportedSemanticDiagnostic entries"
            )
        diagnostic_order = tuple(
            (diagnostic.line_number, diagnostic.semantic, diagnostic.detail)
            for diagnostic in self.unsupported_semantic_diagnostics
        )
        if tuple(sorted(set(diagnostic_order))) != diagnostic_order:
            raise ValueError(
                "program AD static alias lattice unsupported_semantic_diagnostics "
                "must be sorted and unique"
            )
        unsupported_semantics = set(self.unsupported_python_semantics)
        if any(
            diagnostic.semantic not in unsupported_semantics
            for diagnostic in self.unsupported_semantic_diagnostics
        ):
            raise ValueError(
                "program AD static alias lattice unsupported_semantic_diagnostics "
                "must match unsupported_python_semantics"
            )
        for name in ("unsupported_object_attribute_roots", "unsupported_object_attribute_details"):
            values = getattr(self, name)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(
                    f"program AD static alias lattice {name} must contain non-empty strings"
                )
            if tuple(sorted(set(values))) != values:
                raise ValueError(
                    f"program AD static alias lattice {name} must be sorted and unique"
                )
        expected_object_attribute_details = _unsupported_object_attribute_details(
            self.unsupported_semantic_diagnostics
        )
        if self.unsupported_object_attribute_details != expected_object_attribute_details:
            raise ValueError(
                "program AD static alias lattice unsupported_object_attribute_details "
                "must match object-attribute diagnostics"
            )
        expected_object_attribute_roots = _unsupported_object_attribute_roots(
            expected_object_attribute_details
        )
        if self.unsupported_object_attribute_roots != expected_object_attribute_roots:
            raise ValueError(
                "program AD static alias lattice unsupported_object_attribute_roots "
                "must match object-attribute diagnostics"
            )
        if any(not isinstance(reason, str) or not reason for reason in self.blocker_reasons):
            raise ValueError(
                "program AD static alias lattice blocker_reasons must be non-empty strings"
            )
        if tuple(sorted(set(self.blocker_reasons))) != self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice blocker_reasons must be sorted and unique"
            )
        if not isinstance(self.complete, bool):
            raise ValueError("program AD static alias lattice complete must be boolean")
        if self.complete and self.blocker_reasons:
            raise ValueError(
                "complete program AD static alias lattice cannot carry blocker reasons"
            )
        mutation_blocker = "mutation_effects_require_versioned_alias_semantics"
        unknown_alias_blocker = "unknown_alias_edge_kinds"
        unsupported_semantics_blocker = "unsupported_python_semantics_require_frontend_lowering"
        control_path_alias_blocker = "control_path_aliases_require_branch_semantics"
        malformed_control_path_alias_blocker = (
            "control_path_alias_provenance_requires_parseable_targets"
        )
        malformed_view_alias_blocker = "view_alias_provenance_requires_parseable_targets"
        malformed_list_alias_blocker = "list_alias_provenance_requires_parseable_targets"
        malformed_loop_carried_state_blocker = (
            "loop_carried_state_provenance_requires_parseable_targets"
        )
        malformed_rebinding_alias_blocker = "rebinding_alias_provenance_requires_parseable_targets"
        if self.complete and self.control_path_alias_provenance:
            raise ValueError(
                "complete program AD static alias lattice cannot carry control-path aliases"
            )
        if self.complete and self.malformed_control_path_alias_edges:
            raise ValueError(
                "complete program AD static alias lattice cannot carry malformed "
                "control-path alias edges"
            )
        if self.complete and self.unknown_alias_edges:
            raise ValueError(
                "complete program AD static alias lattice cannot carry unknown_alias_edges"
            )
        if self.complete and self.malformed_view_alias_edges:
            raise ValueError(
                "complete program AD static alias lattice cannot carry malformed view-alias edges"
            )
        if self.complete and self.malformed_list_alias_edges:
            raise ValueError(
                "complete program AD static alias lattice cannot carry malformed list-alias edges"
            )
        if self.complete and self.malformed_loop_carried_state_edges:
            raise ValueError(
                "complete program AD static alias lattice cannot carry malformed "
                "loop-carried state edges"
            )
        if self.complete and self.malformed_rebinding_alias_edges:
            raise ValueError(
                "complete program AD static alias lattice cannot carry malformed "
                "rebinding-alias edges"
            )
        if self.complete and self.mutation_effects:
            raise ValueError(
                "complete program AD static alias lattice cannot carry mutation_effects"
            )
        if (
            self.non_executed_control_alias_edges
            and control_path_alias_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice control-path aliases require a blocker reason"
            )
        if (
            not self.non_executed_control_alias_edges
            and control_path_alias_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice control-path blocker requires "
                "non_executed_control_alias_edges"
            )
        if (
            self.malformed_control_path_alias_edges
            and malformed_control_path_alias_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed control-path alias edges "
                "require a blocker reason"
            )
        if (
            not self.malformed_control_path_alias_edges
            and malformed_control_path_alias_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed control-path alias blocker "
                "requires malformed_control_path_alias_edges"
            )
        if self.unknown_alias_edges and unknown_alias_blocker not in self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice unknown_alias_edges require a blocker reason"
            )
        if not self.unknown_alias_edges and unknown_alias_blocker in self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice unknown alias blocker requires "
                "unknown_alias_edges"
            )
        if (
            self.malformed_view_alias_edges
            and malformed_view_alias_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed view-alias edges require "
                "a blocker reason"
            )
        if (
            not self.malformed_view_alias_edges
            and malformed_view_alias_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed view-alias blocker requires "
                "malformed_view_alias_edges"
            )
        if (
            self.malformed_list_alias_edges
            and malformed_list_alias_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed list-alias edges require "
                "a blocker reason"
            )
        if (
            not self.malformed_list_alias_edges
            and malformed_list_alias_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed list-alias blocker requires "
                "malformed_list_alias_edges"
            )
        if (
            self.malformed_loop_carried_state_edges
            and malformed_loop_carried_state_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed loop-carried state edges "
                "require a blocker reason"
            )
        if (
            not self.malformed_loop_carried_state_edges
            and malformed_loop_carried_state_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed loop-carried state blocker "
                "requires malformed_loop_carried_state_edges"
            )
        if (
            self.malformed_rebinding_alias_edges
            and malformed_rebinding_alias_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed rebinding-alias edges require "
                "a blocker reason"
            )
        if (
            not self.malformed_rebinding_alias_edges
            and malformed_rebinding_alias_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice malformed rebinding-alias blocker requires "
                "malformed_rebinding_alias_edges"
            )
        if self.complete and self.unsupported_python_semantics:
            raise ValueError(
                "complete program AD static alias lattice cannot carry "
                "unsupported_python_semantics"
            )
        if self.complete and self.unsupported_object_attribute_roots:
            raise ValueError(
                "complete program AD static alias lattice cannot carry "
                "unsupported_object_attribute_roots"
            )
        if self.mutation_effects and mutation_blocker not in self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice mutation_effects require a blocker reason"
            )
        if not self.mutation_effects and mutation_blocker in self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice mutation blocker requires mutation_effects"
            )
        if (
            self.unsupported_python_semantics
            and unsupported_semantics_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice unsupported_python_semantics require "
                "a blocker reason"
            )
        if (
            not self.unsupported_python_semantics
            and unsupported_semantics_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice unsupported semantics blocker requires "
                "unsupported_python_semantics"
            )
        object_attribute_blocker = "object_attributes_require_static_object_model"
        if (
            self.unsupported_object_attribute_roots
            and object_attribute_blocker not in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice unsupported object attributes require "
                "a blocker reason"
            )
        if (
            not self.unsupported_object_attribute_roots
            and object_attribute_blocker in self.blocker_reasons
        ):
            raise ValueError(
                "program AD static alias lattice object attribute blocker requires "
                "unsupported object-attribute diagnostics"
            )
        _normalise_claim_boundary("program AD static alias lattice", self.claim_boundary)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready static alias lattice readiness payload."""

        return {
            "components": [component.as_dict() for component in self.components],
            "mutation_effects": list(self.mutation_effects),
            "non_executed_phi_nodes": list(self.non_executed_phi_nodes),
            "non_executed_control_alias_edges": list(self.non_executed_control_alias_edges),
            "control_path_alias_provenance": [
                row.as_dict() for row in self.control_path_alias_provenance
            ],
            "malformed_control_path_alias_edges": list(self.malformed_control_path_alias_edges),
            "unknown_alias_edge_kinds": list(self.unknown_alias_edge_kinds),
            "unknown_alias_edges": [edge.as_dict() for edge in self.unknown_alias_edges],
            "view_alias_provenance": [row.as_dict() for row in self.view_alias_provenance],
            "malformed_view_alias_edges": list(self.malformed_view_alias_edges),
            "list_alias_provenance": [row.as_dict() for row in self.list_alias_provenance],
            "malformed_list_alias_edges": list(self.malformed_list_alias_edges),
            "loop_carried_state_provenance": [
                row.as_dict() for row in self.loop_carried_state_provenance
            ],
            "malformed_loop_carried_state_edges": list(self.malformed_loop_carried_state_edges),
            "rebinding_alias_provenance": [
                row.as_dict() for row in self.rebinding_alias_provenance
            ],
            "malformed_rebinding_alias_edges": list(self.malformed_rebinding_alias_edges),
            "unsupported_python_semantics": list(self.unsupported_python_semantics),
            "unsupported_semantic_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.unsupported_semantic_diagnostics
            ],
            "unsupported_object_attribute_roots": list(self.unsupported_object_attribute_roots),
            "unsupported_object_attribute_details": list(
                self.unsupported_object_attribute_details
            ),
            "blocker_reasons": list(self.blocker_reasons),
            "complete": self.complete,
            "claim_boundary": self.claim_boundary,
        }


def analyze_program_ad_alias_effects(
    program_ir: ProgramADEffectIR,
) -> ProgramADAliasEffectAnalysis:
    """Summarize deterministic alias/effect metadata from captured Program AD IR.

    This helper is intentionally metadata-only. It does not promote the current
    runtime trace evidence to a complete alias lattice or static compiler IR.
    """

    if not isinstance(program_ir, ProgramADEffectIR):
        raise ValueError("program AD alias analysis requires ProgramADEffectIR")

    parent: dict[str, str] = {}
    versions_by_member: dict[str, set[int]] = {}
    mutation_versions_by_member: dict[str, set[int]] = {}

    def find(member: str) -> str:
        parent.setdefault(member, member)
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for value in program_ir.ssa_values:
        find(value.name)
        versions_by_member.setdefault(value.name, set()).add(value.version)
    for edge in program_ir.alias_edges:
        if edge.kind not in _PROGRAM_AD_SUPPORTED_ALIAS_EDGE_KINDS:
            raise ValueError(
                f"unknown alias edge kind {edge.kind!r}; program AD alias analysis fails closed"
            )
        find(edge.source)
        find(edge.target)
        union(edge.source, edge.target)
        versions_by_member.setdefault(edge.source, set()).add(edge.version)
        versions_by_member.setdefault(edge.target, set()).add(edge.version)
        if edge.kind == "mutation_version":
            mutation_versions_by_member.setdefault(edge.source, set()).add(edge.version)
            mutation_versions_by_member.setdefault(edge.target, set()).add(edge.version)

    components: dict[str, list[str]] = {}
    for member in sorted(parent):
        components.setdefault(find(member), []).append(member)

    alias_sets: list[ProgramADAliasSet] = []
    for index, members in enumerate(sorted(components.values(), key=lambda values: tuple(values))):
        component_versions: set[int] = set()
        component_mutation_versions: set[int] = set()
        for member in members:
            component_versions.update(versions_by_member.get(member, set()))
            component_mutation_versions.update(mutation_versions_by_member.get(member, set()))
        alias_sets.append(
            ProgramADAliasSet(
                index=index,
                members=tuple(members),
                versions=tuple(sorted(component_versions)),
                mutation_versions=tuple(sorted(component_mutation_versions)),
            )
        )

    mutation_effects = tuple(
        sorted(effect.index for effect in program_ir.effects if effect.kind == "mutation")
    )
    return ProgramADAliasEffectAnalysis(
        alias_sets=tuple(alias_sets),
        mutation_effects=mutation_effects,
        alias_edges=program_ir.alias_edges,
        unknown_aliasing=False,
        claim_boundary=PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY,
    )


def program_ad_static_alias_lattice_report(
    program_ir: ProgramADEffectIR,
    *,
    unsupported_python_semantics: Sequence[str] = (),
    unsupported_semantic_diagnostics: Sequence[WholeProgramUnsupportedSemanticDiagnostic] = (),
) -> ProgramADStaticAliasLatticeReport:
    """Build a static alias-lattice readiness report from emitted Program AD IR.

    The report is complete only for the alias metadata actually emitted in
    ``program_ad_effect_ir.v1``. Mutation effects, unknown alias edge kinds,
    non-selected phi inputs, and unsupported whole-program frontend semantics
    and diagnostics are recorded as hard blockers instead of being promoted
    into full mutation, non-executed branch, or arbitrary Python compiler
    semantics.
    """

    if not isinstance(program_ir, ProgramADEffectIR):
        raise ValueError("program AD static alias lattice requires ProgramADEffectIR")

    parent: dict[str, str] = {}
    versions_by_member: dict[str, set[int]] = {}
    mutation_versions_by_member: dict[str, set[int]] = {}
    edge_kinds_by_member: dict[str, set[str]] = {}
    unknown_alias_edges = _unknown_alias_edges(program_ir.alias_edges)
    unknown_alias_edge_kinds = tuple(sorted({edge.kind for edge in unknown_alias_edges}))
    control_path_alias_provenance, malformed_control_path_alias_edges = (
        _control_path_alias_provenance(program_ir.alias_edges)
    )
    view_alias_provenance, malformed_view_alias_edges = _view_alias_provenance(
        program_ir.alias_edges
    )
    list_alias_provenance, malformed_list_alias_edges = _list_alias_provenance(
        program_ir.alias_edges
    )
    loop_carried_state_provenance, malformed_loop_carried_state_edges = (
        _loop_carried_state_provenance(program_ir.alias_edges)
    )
    rebinding_alias_provenance, malformed_rebinding_alias_edges = _rebinding_alias_provenance(
        program_ir.alias_edges
    )
    unsupported_diagnostics = tuple(unsupported_semantic_diagnostics)
    if any(
        not isinstance(diagnostic, WholeProgramUnsupportedSemanticDiagnostic)
        for diagnostic in unsupported_diagnostics
    ):
        raise ValueError(
            "program AD static alias lattice unsupported_semantic_diagnostics "
            "must contain WholeProgramUnsupportedSemanticDiagnostic entries"
        )
    unsupported_semantics = tuple(
        sorted(
            set(unsupported_python_semantics).union(
                diagnostic.semantic for diagnostic in unsupported_diagnostics
            )
        )
    )
    unsupported_object_attribute_details = _unsupported_object_attribute_details(
        unsupported_diagnostics
    )
    unsupported_object_attribute_roots = _unsupported_object_attribute_roots(
        unsupported_object_attribute_details
    )

    def find(member: str) -> str:
        parent.setdefault(member, member)
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for value in program_ir.ssa_values:
        find(value.name)
        versions_by_member.setdefault(value.name, set()).add(value.version)
    for edge in program_ir.alias_edges:
        find(edge.source)
        find(edge.target)
        union(edge.source, edge.target)
        versions_by_member.setdefault(edge.source, set()).add(edge.version)
        versions_by_member.setdefault(edge.target, set()).add(edge.version)
        edge_kinds_by_member.setdefault(edge.source, set()).add(edge.kind)
        edge_kinds_by_member.setdefault(edge.target, set()).add(edge.kind)
        if edge.kind == "mutation_version":
            mutation_versions_by_member.setdefault(edge.source, set()).add(edge.version)
            mutation_versions_by_member.setdefault(edge.target, set()).add(edge.version)

    components_by_root: dict[str, list[str]] = {}
    for member in sorted(parent):
        components_by_root.setdefault(find(member), []).append(member)

    components: list[ProgramADStaticAliasLatticeComponent] = []
    for index, members in enumerate(
        sorted(components_by_root.values(), key=lambda values: tuple(values))
    ):
        edge_kinds: set[str] = set()
        versions: set[int] = set()
        mutation_versions: set[int] = set()
        for member in members:
            edge_kinds.update(edge_kinds_by_member.get(member, set()))
            versions.update(versions_by_member.get(member, set()))
            mutation_versions.update(mutation_versions_by_member.get(member, set()))
        components.append(
            ProgramADStaticAliasLatticeComponent(
                index=index,
                members=tuple(members),
                edge_kinds=tuple(sorted(edge_kinds)),
                versions=tuple(sorted(versions)),
                mutation_versions=tuple(sorted(mutation_versions)),
            )
        )

    non_executed_phi_nodes = tuple(
        sorted(
            phi_node.index
            for phi_node in program_ir.phi_nodes
            if phi_node.selected is not None
            and phi_node.selected != "executed_loop_trace"
            and any(incoming != phi_node.selected for incoming in phi_node.incoming)
        )
    )
    control_alias_edges = tuple(
        sorted(
            f"{edge.source}->{edge.target}"
            for edge in program_ir.alias_edges
            if edge.kind == "control_path_alias"
        )
    )
    mutation_effects = tuple(
        sorted(effect.index for effect in program_ir.effects if effect.kind == "mutation")
    )
    blocker_reasons: set[str] = set()
    if unknown_alias_edge_kinds:
        blocker_reasons.add("unknown_alias_edge_kinds")
    if mutation_effects:
        blocker_reasons.add("mutation_effects_require_versioned_alias_semantics")
    if non_executed_phi_nodes:
        blocker_reasons.add("non_executed_phi_inputs_require_branch_semantics")
    if control_alias_edges:
        blocker_reasons.add("control_path_aliases_require_branch_semantics")
    if malformed_control_path_alias_edges:
        blocker_reasons.add("control_path_alias_provenance_requires_parseable_targets")
    if malformed_view_alias_edges:
        blocker_reasons.add("view_alias_provenance_requires_parseable_targets")
    if malformed_list_alias_edges:
        blocker_reasons.add("list_alias_provenance_requires_parseable_targets")
    if malformed_loop_carried_state_edges:
        blocker_reasons.add("loop_carried_state_provenance_requires_parseable_targets")
    if malformed_rebinding_alias_edges:
        blocker_reasons.add("rebinding_alias_provenance_requires_parseable_targets")
    if unsupported_semantics:
        blocker_reasons.add("unsupported_python_semantics_require_frontend_lowering")
    if unsupported_object_attribute_roots:
        blocker_reasons.add("object_attributes_require_static_object_model")
    complete = not blocker_reasons
    return ProgramADStaticAliasLatticeReport(
        components=tuple(components),
        mutation_effects=mutation_effects,
        non_executed_phi_nodes=non_executed_phi_nodes,
        non_executed_control_alias_edges=control_alias_edges,
        unknown_alias_edge_kinds=unknown_alias_edge_kinds,
        blocker_reasons=tuple(sorted(blocker_reasons)),
        complete=complete,
        claim_boundary=PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY,
        control_path_alias_provenance=control_path_alias_provenance,
        malformed_control_path_alias_edges=malformed_control_path_alias_edges,
        unsupported_python_semantics=unsupported_semantics,
        unsupported_semantic_diagnostics=unsupported_diagnostics,
        unsupported_object_attribute_roots=unsupported_object_attribute_roots,
        unsupported_object_attribute_details=unsupported_object_attribute_details,
        unknown_alias_edges=unknown_alias_edges,
        view_alias_provenance=view_alias_provenance,
        malformed_view_alias_edges=malformed_view_alias_edges,
        list_alias_provenance=list_alias_provenance,
        malformed_list_alias_edges=malformed_list_alias_edges,
        loop_carried_state_provenance=loop_carried_state_provenance,
        malformed_loop_carried_state_edges=malformed_loop_carried_state_edges,
        rebinding_alias_provenance=rebinding_alias_provenance,
        malformed_rebinding_alias_edges=malformed_rebinding_alias_edges,
    )


def _unknown_alias_edges(
    alias_edges: Sequence[ProgramADAliasEdge],
) -> tuple[ProgramADUnknownAliasEdge, ...]:
    """Return sorted unsupported alias-edge provenance from emitted IR."""

    edges = {
        ProgramADUnknownAliasEdge(
            source=edge.source,
            target=edge.target,
            kind=edge.kind,
            version=edge.version,
        )
        for edge in alias_edges
        if edge.kind not in _PROGRAM_AD_SUPPORTED_ALIAS_EDGE_KINDS
    }
    return tuple(
        sorted(edges, key=lambda edge: (edge.kind, edge.source, edge.target, edge.version))
    )


def _control_path_alias_provenance(
    alias_edges: Sequence[ProgramADAliasEdge],
) -> tuple[tuple[ProgramADControlPathAliasProvenance, ...], tuple[str, ...]]:
    """Return sorted parseable control-path alias provenance and malformed labels."""

    rows: set[ProgramADControlPathAliasProvenance] = set()
    malformed: set[str] = set()
    for edge in alias_edges:
        if edge.kind != "control_path_alias":
            continue
        try:
            rows.add(_parse_control_path_alias_provenance(edge))
        except ValueError:
            malformed.add(_alias_edge_label(edge))
    return (
        tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.branch_line,
                    row.branch_arm,
                    row.target_label,
                    row.source,
                    row.target,
                    row.version,
                ),
            )
        ),
        tuple(sorted(malformed)),
    )


def _parse_control_path_alias_provenance(
    edge: ProgramADAliasEdge,
) -> ProgramADControlPathAliasProvenance:
    """Parse one branch-local control-path alias edge into typed provenance."""

    source_parts = edge.source.split(":")
    if len(source_parts) != 4 or source_parts[0] != "control" or source_parts[1] != "if":
        raise ValueError("program AD control-path alias source must be control:if:<line>:<arm>")
    branch_line_raw = source_parts[2]
    branch_arm = source_parts[3]
    if not branch_line_raw.isdecimal():
        raise ValueError("program AD control-path alias branch line must be positive")
    branch_line = int(branch_line_raw)
    if branch_line <= 0:
        raise ValueError("program AD control-path alias branch line must be positive")
    if branch_arm not in _PROGRAM_AD_CONTROL_PATH_ALIAS_BRANCH_ARMS:
        raise ValueError("program AD control-path alias branch arm is unsupported")
    if not edge.target.startswith("control:") or edge.target == "control:":
        raise ValueError("program AD control-path alias target must be control:<label>")
    return ProgramADControlPathAliasProvenance(
        source=edge.source,
        target=edge.target,
        branch_line=branch_line,
        branch_arm=branch_arm,
        target_label=edge.target.removeprefix("control:"),
        version=edge.version,
    )


def _view_alias_provenance(
    alias_edges: Sequence[ProgramADAliasEdge],
) -> tuple[tuple[ProgramADViewAliasProvenance, ...], tuple[str, ...]]:
    """Return sorted parseable view-alias provenance and malformed edge labels."""

    rows: set[ProgramADViewAliasProvenance] = set()
    malformed: set[str] = set()
    for edge in alias_edges:
        if edge.kind != "view_alias":
            continue
        if edge.target.startswith("view:"):
            try:
                rows.add(_parse_view_alias_provenance(edge))
            except ValueError:
                malformed.add(_alias_edge_label(edge))
            continue
        if not edge.source.startswith("view:"):
            malformed.add(_alias_edge_label(edge))
    return (
        tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.operation,
                    row.view_id,
                    row.output_index,
                    row.source,
                    row.target,
                    row.version,
                ),
            )
        ),
        tuple(sorted(malformed)),
    )


def _parse_view_alias_provenance(edge: ProgramADAliasEdge) -> ProgramADViewAliasProvenance:
    """Parse one source-to-view marker alias edge into typed provenance."""

    marker = edge.target.removeprefix("view:")
    try:
        operation, identifier = marker.split(":", maxsplit=1)
        view_id_raw, output_index_raw = identifier.removesuffix("]").split("[", maxsplit=1)
    except ValueError as exc:
        raise ValueError("program AD view alias target must be view:<op>:<id>[<index>]") from exc
    if not edge.target.endswith("]"):
        raise ValueError("program AD view alias target must end with an output index")
    return ProgramADViewAliasProvenance(
        source=edge.source,
        target=edge.target,
        operation=operation,
        view_id=_parse_view_alias_int("view_id", view_id_raw),
        output_index=_parse_view_alias_int("output_index", output_index_raw),
        version=edge.version,
    )


def _parse_view_alias_int(name: str, value: str) -> int:
    """Parse a non-negative integer token from a view-alias marker."""

    if not value.isdecimal():
        raise ValueError(f"program AD view alias {name} must be a non-negative integer")
    return int(value)


def _list_alias_provenance(
    alias_edges: Sequence[ProgramADAliasEdge],
) -> tuple[tuple[ProgramADListAliasProvenance, ...], tuple[str, ...]]:
    """Return sorted parseable list-alias provenance and malformed edge labels."""

    rows: set[ProgramADListAliasProvenance] = set()
    malformed: set[str] = set()
    for edge in alias_edges:
        if edge.kind != "list_alias":
            continue
        try:
            rows.add(_parse_list_alias_provenance(edge))
        except ValueError:
            malformed.add(_alias_edge_label(edge))
    return (
        tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.list_name,
                    row.target_kind,
                    row.source,
                    row.target,
                    row.version,
                ),
            )
        ),
        tuple(sorted(malformed)),
    )


def _parse_list_alias_provenance(edge: ProgramADAliasEdge) -> ProgramADListAliasProvenance:
    """Parse one local list-alias edge into typed provenance."""

    if not edge.source.startswith("list:") or edge.source == "list:":
        raise ValueError("program AD list alias source must be list:<name>")
    if edge.target.startswith("name:") and edge.target != "name:":
        target_kind = "local_name"
    elif edge.target == "source:list_mutation":
        target_kind = "indexed_mutation_source"
    else:
        raise ValueError("program AD list alias target must be local name or list mutation")
    return ProgramADListAliasProvenance(
        source=edge.source,
        target=edge.target,
        list_name=edge.source.removeprefix("list:"),
        target_kind=target_kind,
        version=edge.version,
    )


def _loop_carried_state_provenance(
    alias_edges: Sequence[ProgramADAliasEdge],
) -> tuple[tuple[ProgramADLoopCarriedStateProvenance, ...], tuple[str, ...]]:
    """Return sorted parseable loop-carried state provenance and malformed labels."""

    rows: set[ProgramADLoopCarriedStateProvenance] = set()
    malformed: set[str] = set()
    for edge in alias_edges:
        if edge.kind != "loop_carried_state":
            continue
        try:
            rows.add(_parse_loop_carried_state_provenance(edge))
        except ValueError:
            malformed.add(_alias_edge_label(edge))
    return (
        tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.state_name,
                    row.entry_label,
                    row.backedge_label,
                    row.source,
                    row.target,
                    row.version,
                ),
            )
        ),
        tuple(sorted(malformed)),
    )


def _parse_loop_carried_state_provenance(
    edge: ProgramADAliasEdge,
) -> ProgramADLoopCarriedStateProvenance:
    """Parse one loop-carried state edge into typed provenance."""

    source_parts = edge.source.split(":")
    target_parts = edge.target.split(":")
    if len(source_parts) != 3 or source_parts[0] != "loop":
        raise ValueError("program AD loop-carried state source must be loop:<state>:entry")
    if len(target_parts) != 3 or target_parts[0] != "loop":
        raise ValueError("program AD loop-carried state target must be loop:<state>:backedge")
    source_state = source_parts[1]
    target_state = target_parts[1]
    entry_label = source_parts[2]
    backedge_label = target_parts[2]
    if not source_state or source_state != target_state:
        raise ValueError("program AD loop-carried state markers must share one state name")
    return ProgramADLoopCarriedStateProvenance(
        source=edge.source,
        target=edge.target,
        state_name=source_state,
        entry_label=entry_label,
        backedge_label=backedge_label,
        version=edge.version,
    )


def _rebinding_alias_provenance(
    alias_edges: Sequence[ProgramADAliasEdge],
) -> tuple[tuple[ProgramADRebindingAliasProvenance, ...], tuple[str, ...]]:
    """Return sorted parseable rebinding-alias provenance and malformed labels."""

    rows: set[ProgramADRebindingAliasProvenance] = set()
    malformed: set[str] = set()
    for edge in alias_edges:
        if edge.kind not in _PROGRAM_AD_REBINDING_ALIAS_EDGE_KINDS:
            continue
        try:
            rows.add(_parse_rebinding_alias_provenance(edge))
        except ValueError:
            malformed.add(_alias_edge_label(edge))
    return (
        tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.binding_kind,
                    row.source,
                    row.target,
                    row.source_name or "",
                    row.expression_line or 0,
                    row.expression_label or "",
                    row.version,
                ),
            )
        ),
        tuple(sorted(malformed)),
    )


def _parse_rebinding_alias_provenance(
    edge: ProgramADAliasEdge,
) -> ProgramADRebindingAliasProvenance:
    """Parse one local or expression rebinding alias edge into typed provenance."""

    if not edge.target.startswith("name:") or edge.target == "name:":
        raise ValueError("program AD rebinding alias target must be name:<local>")
    target_name = edge.target.removeprefix("name:")
    if edge.kind == "local_rebinding_alias":
        if not edge.source.startswith("name:") or edge.source == "name:":
            raise ValueError("program AD local rebinding alias source must be name:<local>")
        return ProgramADRebindingAliasProvenance(
            source=edge.source,
            target=edge.target,
            binding_kind="local",
            source_name=edge.source.removeprefix("name:"),
            expression_line=None,
            expression_label=None,
            target_name=target_name,
            version=edge.version,
        )
    if edge.kind == "expression_rebinding_alias":
        try:
            prefix, line_raw, expression_label = edge.source.split(":", maxsplit=2)
        except ValueError as exc:
            raise ValueError(
                "program AD expression rebinding alias source must be expr:<line>:<label>"
            ) from exc
        if prefix != "expr" or not line_raw.isdecimal() or not expression_label:
            raise ValueError(
                "program AD expression rebinding alias source must be expr:<line>:<label>"
            )
        expression_line = int(line_raw)
        if expression_line <= 0:
            raise ValueError("program AD expression rebinding alias line must be positive")
        return ProgramADRebindingAliasProvenance(
            source=edge.source,
            target=edge.target,
            binding_kind="expression",
            source_name=None,
            expression_line=expression_line,
            expression_label=expression_label,
            target_name=target_name,
            version=edge.version,
        )
    raise ValueError("program AD rebinding alias kind is unsupported")


def _alias_edge_label(edge: ProgramADAliasEdge) -> str:
    """Return a deterministic label for malformed alias-edge provenance."""

    return f"{edge.source}->{edge.target}:{edge.kind}@{edge.version}"


def _unsupported_object_attribute_details(
    diagnostics: Sequence[WholeProgramUnsupportedSemanticDiagnostic],
) -> tuple[str, ...]:
    """Return sorted frontend object-attribute diagnostic detail strings."""

    return tuple(
        sorted(
            {
                diagnostic.detail
                for diagnostic in diagnostics
                if diagnostic.semantic == "object_attribute"
            }
        )
    )


def _unsupported_object_attribute_roots(details: Sequence[str]) -> tuple[str, ...]:
    """Return captured or global object roots from object-attribute details."""

    roots: set[str] = set()
    prefix = "object_attribute:"
    for detail in details:
        if detail.startswith(prefix) and detail != prefix:
            roots.add(detail.removeprefix(prefix))
        else:
            roots.add(detail)
    return tuple(sorted(roots))


def _normalise_claim_boundary(label: str, claim_boundary: str) -> str:
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise ValueError(f"{label} claim_boundary must be a non-empty string")
    return claim_boundary


__all__ = [
    "PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY",
    "PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY",
    "ProgramADAliasEffectAnalysis",
    "ProgramADAliasSet",
    "ProgramADControlPathAliasProvenance",
    "ProgramADListAliasProvenance",
    "ProgramADLoopCarriedStateProvenance",
    "ProgramADRebindingAliasProvenance",
    "ProgramADStaticAliasLatticeComponent",
    "ProgramADStaticAliasLatticeReport",
    "ProgramADUnknownAliasEdge",
    "ProgramADViewAliasProvenance",
    "analyze_program_ad_alias_effects",
    "program_ad_static_alias_lattice_report",
]
