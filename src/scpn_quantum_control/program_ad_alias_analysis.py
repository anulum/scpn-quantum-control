# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD alias and static-lattice analysis
"""Program AD alias/effect analysis drivers and static-lattice provenance parsing."""

from __future__ import annotations

from collections.abc import Sequence

from .program_ad_alias_contracts import (
    _PROGRAM_AD_CONTROL_PATH_ALIAS_BRANCH_ARMS,
    _PROGRAM_AD_REBINDING_ALIAS_EDGE_KINDS,
    _PROGRAM_AD_SUPPORTED_ALIAS_EDGE_KINDS,
    PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY,
    PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY,
    ProgramADAliasEffectAnalysis,
    ProgramADAliasSet,
    ProgramADControlPathAliasProvenance,
    ProgramADListAliasProvenance,
    ProgramADLoopCarriedStateProvenance,
    ProgramADRebindingAliasProvenance,
    ProgramADStaticAliasLatticeComponent,
    ProgramADStaticAliasLatticeReport,
    ProgramADUnknownAliasEdge,
    ProgramADViewAliasProvenance,
    _unsupported_object_attribute_details,
    _unsupported_object_attribute_roots,
)
from .program_ad_effect_ir import ProgramADAliasEdge, ProgramADEffectIR
from .whole_program_frontend import WholeProgramUnsupportedSemanticDiagnostic


def analyze_program_ad_alias_effects(
    program_ir: ProgramADEffectIR,
) -> ProgramADAliasEffectAnalysis:
    """Summarize deterministic alias/effect metadata from captured Program AD IR.

    This helper is intentionally metadata-only. It does not promote the current
    runtime trace evidence to a complete alias lattice or static compiler IR.

    Parameters
    ----------
    program_ir:
        Emitted ``program_ad_effect_ir.v1`` payload from whole-program Program
        AD capture.

    Returns
    -------
    ProgramADAliasEffectAnalysis
        Deterministic alias components, mutation-effect indices, raw alias
        edges, and the metadata-only claim boundary.

    Raises
    ------
    ValueError
        Raised when ``program_ir`` is not a Program AD effect IR or contains an
        alias-edge kind outside the supported metadata-only vocabulary.
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
    for index, members in enumerate(sorted(components.values(), key=tuple)):
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

    Parameters
    ----------
    program_ir:
        Emitted ``program_ad_effect_ir.v1`` payload from whole-program Program
        AD capture.
    unsupported_python_semantics:
        Optional frontend semantic names that remain outside the static
        alias-lattice contract.
    unsupported_semantic_diagnostics:
        Optional source-level diagnostics for unsupported frontend semantics.

    Returns
    -------
    ProgramADStaticAliasLatticeReport
        Deterministic static alias-lattice readiness report with explicit
        blockers and typed provenance for unpromoted paths.

    Raises
    ------
    ValueError
        Raised when ``program_ir`` is not a Program AD effect IR, diagnostics
        have the wrong type, or emitted alias metadata violates the bounded
        static-lattice contract.
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
    for index, members in enumerate(sorted(components_by_root.values(), key=tuple)):
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
