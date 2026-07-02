# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD alias and static-lattice analysis
"""Program AD alias/effect summaries and static alias-lattice reports."""

from __future__ import annotations

from dataclasses import dataclass

from .program_ad_effect_ir import ProgramADAliasEdge, ProgramADEffectIR

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
class ProgramADStaticAliasLatticeReport:
    """Static alias-lattice readiness report for emitted Program AD IR.

    The report builds deterministic alias components from the emitted
    ``program_ad_effect_ir.v1`` metadata and records the exact blockers that
    prevent promotion to full static alias, mutation, or non-executed branch
    semantics.
    """

    components: tuple[ProgramADStaticAliasLatticeComponent, ...]
    mutation_effects: tuple[int, ...]
    non_executed_phi_nodes: tuple[int, ...]
    non_executed_control_alias_edges: tuple[str, ...]
    unknown_alias_edge_kinds: tuple[str, ...]
    blocker_reasons: tuple[str, ...]
    complete: bool
    claim_boundary: str

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
        if self.complete and self.mutation_effects:
            raise ValueError(
                "complete program AD static alias lattice cannot carry mutation_effects"
            )
        if self.mutation_effects and mutation_blocker not in self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice mutation_effects require a blocker reason"
            )
        if not self.mutation_effects and mutation_blocker in self.blocker_reasons:
            raise ValueError(
                "program AD static alias lattice mutation blocker requires mutation_effects"
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
) -> ProgramADStaticAliasLatticeReport:
    """Build a static alias-lattice readiness report from emitted Program AD IR.

    The report is complete only for the alias metadata actually emitted in
    ``program_ad_effect_ir.v1``. Mutation effects, unknown alias edge kinds,
    and non-selected phi inputs are recorded as hard blockers instead of being
    promoted into full mutation or non-executed branch semantics.
    """

    if not isinstance(program_ir, ProgramADEffectIR):
        raise ValueError("program AD static alias lattice requires ProgramADEffectIR")

    parent: dict[str, str] = {}
    versions_by_member: dict[str, set[int]] = {}
    mutation_versions_by_member: dict[str, set[int]] = {}
    edge_kinds_by_member: dict[str, set[str]] = {}
    unknown_alias_edge_kinds = tuple(
        sorted(
            {
                edge.kind
                for edge in program_ir.alias_edges
                if edge.kind not in _PROGRAM_AD_SUPPORTED_ALIAS_EDGE_KINDS
            }
        )
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
    )


def _normalise_claim_boundary(label: str, claim_boundary: str) -> str:
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise ValueError(f"{label} claim_boundary must be a non-empty string")
    return claim_boundary


__all__ = [
    "PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY",
    "PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY",
    "ProgramADAliasEffectAnalysis",
    "ProgramADAliasSet",
    "ProgramADStaticAliasLatticeComponent",
    "ProgramADStaticAliasLatticeReport",
    "analyze_program_ad_alias_effects",
    "program_ad_static_alias_lattice_report",
]
