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
    unsupported_python_semantics: tuple[str, ...] = ()
    unsupported_semantic_diagnostics: tuple[WholeProgramUnsupportedSemanticDiagnostic, ...] = ()
    unsupported_object_attribute_roots: tuple[str, ...] = ()
    unsupported_object_attribute_details: tuple[str, ...] = ()
    unknown_alias_edges: tuple[ProgramADUnknownAliasEdge, ...] = ()
    view_alias_provenance: tuple[ProgramADViewAliasProvenance, ...] = ()
    malformed_view_alias_edges: tuple[str, ...] = ()
    list_alias_provenance: tuple[ProgramADListAliasProvenance, ...] = ()
    malformed_list_alias_edges: tuple[str, ...] = ()

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
        has_view_alias_component = any(
            "view_alias" in component.edge_kinds for component in self.components
        )
        has_list_alias_component = any(
            "list_alias" in component.edge_kinds for component in self.components
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
        malformed_view_alias_blocker = "view_alias_provenance_requires_parseable_targets"
        malformed_list_alias_blocker = "list_alias_provenance_requires_parseable_targets"
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
        if self.complete and self.mutation_effects:
            raise ValueError(
                "complete program AD static alias lattice cannot carry mutation_effects"
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
            "unknown_alias_edge_kinds": list(self.unknown_alias_edge_kinds),
            "unknown_alias_edges": [edge.as_dict() for edge in self.unknown_alias_edges],
            "view_alias_provenance": [row.as_dict() for row in self.view_alias_provenance],
            "malformed_view_alias_edges": list(self.malformed_view_alias_edges),
            "list_alias_provenance": [row.as_dict() for row in self.list_alias_provenance],
            "malformed_list_alias_edges": list(self.malformed_list_alias_edges),
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
    view_alias_provenance, malformed_view_alias_edges = _view_alias_provenance(
        program_ir.alias_edges
    )
    list_alias_provenance, malformed_list_alias_edges = _list_alias_provenance(
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
    if malformed_view_alias_edges:
        blocker_reasons.add("view_alias_provenance_requires_parseable_targets")
    if malformed_list_alias_edges:
        blocker_reasons.add("list_alias_provenance_requires_parseable_targets")
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
        unsupported_python_semantics=unsupported_semantics,
        unsupported_semantic_diagnostics=unsupported_diagnostics,
        unsupported_object_attribute_roots=unsupported_object_attribute_roots,
        unsupported_object_attribute_details=unsupported_object_attribute_details,
        unknown_alias_edges=unknown_alias_edges,
        view_alias_provenance=view_alias_provenance,
        malformed_view_alias_edges=malformed_view_alias_edges,
        list_alias_provenance=list_alias_provenance,
        malformed_list_alias_edges=malformed_list_alias_edges,
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
    "ProgramADListAliasProvenance",
    "ProgramADStaticAliasLatticeComponent",
    "ProgramADStaticAliasLatticeReport",
    "ProgramADUnknownAliasEdge",
    "ProgramADViewAliasProvenance",
    "analyze_program_ad_alias_effects",
    "program_ad_static_alias_lattice_report",
]
