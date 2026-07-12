# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — alias activity evidence module
# scpn-quantum-control -- compiler alias-activity evidence
"""Compiler alias-activity evidence assembled from Program AD lattice reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..differentiable_parameter_contracts import Parameter
from ..program_ad_alias_analysis import (
    ProgramADStaticAliasLatticeReport,
    program_ad_static_alias_lattice_report,
)
from ..whole_program_ad_api import whole_program_value_and_grad

COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA = "scpn_qc_compiler_alias_activity_evidence_v1"
COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID = "compiler-alias-activity-evidence-20260706"
COMPILER_ALIAS_ACTIVITY_CLASSIFICATION = "functional_non_isolated"
COMPILER_ALIAS_ACTIVITY_CLAIM_BOUNDARY = (
    "Compiler alias-activity evidence only: Program AD static alias-lattice "
    "reports are executed locally for bounded view, list, scalar-rebinding, "
    "object-attribute, loop-carried, branch-control, and mutation-version "
    "cases; this does not promote general compiler AD, isolated benchmark, "
    "provider, hardware, GPU, or performance claim."
)
_SUPPORTED_CASE_STATUSES = frozenset({"complete_lattice", "blocked_lattice"})
_REQUIRED_ALIAS_EDGE_KINDS = frozenset(
    {
        "control_path_alias",
        "expression_rebinding_alias",
        "list_alias",
        "local_rebinding_alias",
        "loop_carried_state",
        "mutation_version",
        "object_attribute_alias",
        "view_alias",
    }
)


@dataclass(frozen=True)
class CompilerAliasActivityCase:
    """One bounded Program AD alias-lattice activity case."""

    case_id: str
    status: str
    complete: bool
    alias_edge_kinds: tuple[str, ...]
    blocker_reasons: tuple[str, ...]
    component_count: int
    provenance_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate alias-activity case metadata."""
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("compiler alias-activity case_id must be non-empty")
        if self.status not in _SUPPORTED_CASE_STATUSES:
            raise ValueError("compiler alias-activity case status is unsupported")
        if not isinstance(self.complete, bool):
            raise ValueError("compiler alias-activity case complete must be boolean")
        if (self.status == "complete_lattice") != self.complete:
            raise ValueError("compiler alias-activity case status must match completeness")
        if not isinstance(self.alias_edge_kinds, tuple):
            raise ValueError("compiler alias-activity case alias_edge_kinds must be a tuple")
        if any(not isinstance(kind, str) or not kind for kind in self.alias_edge_kinds):
            raise ValueError("compiler alias-activity case alias_edge_kinds must be non-empty")
        if not self.alias_edge_kinds:
            raise ValueError("compiler alias-activity case alias_edge_kinds must be non-empty")
        if tuple(sorted(set(self.alias_edge_kinds))) != self.alias_edge_kinds:
            raise ValueError("compiler alias-activity case alias_edge_kinds must be sorted unique")
        if not isinstance(self.blocker_reasons, tuple):
            raise ValueError("compiler alias-activity case blocker_reasons must be a tuple")
        if any(not isinstance(reason, str) or not reason for reason in self.blocker_reasons):
            raise ValueError("compiler alias-activity case blocker_reasons must be non-empty")
        if tuple(sorted(set(self.blocker_reasons))) != self.blocker_reasons:
            raise ValueError("compiler alias-activity case blocker_reasons must be sorted unique")
        if self.complete and self.blocker_reasons:
            raise ValueError("complete compiler alias-activity cases cannot carry blockers")
        if not self.complete and not self.blocker_reasons:
            raise ValueError("blocked compiler alias-activity cases must carry blockers")
        if (
            not isinstance(self.component_count, int)
            or isinstance(self.component_count, bool)
            or self.component_count <= 0
        ):
            raise ValueError(
                "compiler alias-activity case component_count must be a positive integer"
            )
        if not isinstance(self.provenance_counts, Mapping):
            raise ValueError("compiler alias-activity case provenance_counts must be a mapping")
        if not self.provenance_counts:
            raise ValueError("compiler alias-activity case provenance_counts must be non-empty")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for name, count in self.provenance_counts.items()
        ):
            raise ValueError(
                "compiler alias-activity case provenance_counts must map names to non-negative ints"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready alias-activity case."""
        return {
            "case_id": self.case_id,
            "status": self.status,
            "complete": self.complete,
            "alias_edge_kinds": list(self.alias_edge_kinds),
            "blocker_reasons": list(self.blocker_reasons),
            "component_count": self.component_count,
            "provenance_counts": dict(sorted(self.provenance_counts.items())),
        }


@dataclass(frozen=True)
class CompilerAliasActivityEvidence:
    """Evidence bundle for the compiler alias-activity promotion requirement."""

    source_commit: str
    cases: tuple[CompilerAliasActivityCase, ...]
    test_ids: tuple[str, ...]
    artifact_id: str = COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID
    schema: str = COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA
    classification: str = COMPILER_ALIAS_ACTIVITY_CLASSIFICATION
    promotion_ready: bool = False
    claim_boundary: str = COMPILER_ALIAS_ACTIVITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate alias-activity evidence metadata."""
        if self.artifact_id != COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID:
            raise ValueError("compiler alias-activity evidence artifact_id drifted")
        if self.schema != COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA:
            raise ValueError("compiler alias-activity evidence schema drifted")
        if self.classification != COMPILER_ALIAS_ACTIVITY_CLASSIFICATION:
            raise ValueError("compiler alias-activity evidence classification drifted")
        if self.promotion_ready:
            raise ValueError("compiler alias-activity evidence cannot be promotion-ready")
        if not isinstance(self.source_commit, str) or not self.source_commit.strip():
            raise ValueError("compiler alias-activity evidence source_commit must be non-empty")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("compiler alias-activity evidence cases must be a non-empty tuple")
        if any(not isinstance(case, CompilerAliasActivityCase) for case in self.cases):
            raise ValueError(
                "compiler alias-activity evidence cases must contain CompilerAliasActivityCase"
            )
        if tuple(sorted(self.cases, key=lambda case: case.case_id)) != self.cases:
            raise ValueError("compiler alias-activity evidence cases must be sorted by case_id")
        case_ids = tuple(case.case_id for case in self.cases)
        if tuple(sorted(set(case_ids))) != case_ids:
            raise ValueError("compiler alias-activity evidence case_ids must be unique")
        if not self.complete_lattice_case_count:
            raise ValueError("compiler alias-activity evidence requires a complete lattice case")
        if not self.blocked_lattice_case_count:
            raise ValueError("compiler alias-activity evidence requires a blocked lattice case")
        missing_alias_kinds = _REQUIRED_ALIAS_EDGE_KINDS.difference(self.observed_alias_edge_kinds)
        if missing_alias_kinds:
            missing = ", ".join(sorted(missing_alias_kinds))
            raise ValueError(f"compiler alias-activity evidence missing alias kinds: {missing}")
        if not isinstance(self.test_ids, tuple) or not self.test_ids:
            raise ValueError("compiler alias-activity evidence test_ids must be a non-empty tuple")
        if any(not isinstance(test_id, str) or "::" not in test_id for test_id in self.test_ids):
            raise ValueError("compiler alias-activity evidence test_ids must name pytest nodes")
        if tuple(sorted(set(self.test_ids))) != self.test_ids:
            raise ValueError("compiler alias-activity evidence test_ids must be sorted unique")
        for phrase in (
            "general compiler AD",
            "isolated benchmark",
            "provider, hardware, GPU, or performance claim",
        ):
            if phrase not in self.claim_boundary:
                raise ValueError("compiler alias-activity evidence claim boundary is incomplete")

    @property
    def complete_lattice_case_count(self) -> int:
        """Return the number of complete lattice cases."""
        return sum(1 for case in self.cases if case.status == "complete_lattice")

    @property
    def blocked_lattice_case_count(self) -> int:
        """Return the number of fail-closed lattice cases."""
        return sum(1 for case in self.cases if case.status == "blocked_lattice")

    @property
    def observed_alias_edge_kinds(self) -> tuple[str, ...]:
        """Return the sorted alias-edge kinds observed across all cases."""
        return tuple(sorted({kind for case in self.cases for kind in case.alias_edge_kinds}))

    @property
    def alias_activity_verified(self) -> bool:
        """Return whether the bounded alias-activity requirement has evidence."""
        return (
            self.complete_lattice_case_count > 0
            and self.blocked_lattice_case_count > 0
            and _REQUIRED_ALIAS_EDGE_KINDS.issubset(self.observed_alias_edge_kinds)
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready alias-activity evidence payload."""
        return {
            "artifact_id": self.artifact_id,
            "schema": self.schema,
            "source_commit": self.source_commit,
            "classification": self.classification,
            "promotion_ready": self.promotion_ready,
            "alias_activity_verified": self.alias_activity_verified,
            "complete_lattice_case_count": self.complete_lattice_case_count,
            "blocked_lattice_case_count": self.blocked_lattice_case_count,
            "observed_alias_edge_kinds": list(self.observed_alias_edge_kinds),
            "test_ids": list(self.test_ids),
            "cases": [case.as_dict() for case in self.cases],
            "claim_boundary": self.claim_boundary,
        }


def build_compiler_alias_activity_evidence(
    *,
    source_commit: str,
) -> CompilerAliasActivityEvidence:
    """Build alias-activity evidence from real Program AD lattice reports."""
    cases = tuple(
        sorted(
            (
                _view_alias_case(),
                _list_alias_case(),
                _scalar_rebinding_case(),
                _object_attribute_case(),
                _loop_carried_state_case(),
                _non_executed_branch_case(),
                _branch_attribute_case(),
                _slice_mutation_case(),
            ),
            key=lambda case: case.case_id,
        )
    )
    return CompilerAliasActivityEvidence(
        source_commit=source_commit,
        cases=cases,
        test_ids=_ALIAS_ACTIVITY_TEST_IDS,
    )


def render_compiler_alias_activity_evidence_markdown(
    evidence: CompilerAliasActivityEvidence,
) -> str:
    """Render alias-activity evidence as reviewer-facing Markdown."""
    lines = [
        "# Compiler Alias-Activity Evidence",
        "",
        f"- artifact_id: `{evidence.artifact_id}`",
        f"- source_commit: `{evidence.source_commit}`",
        f"- classification: `{evidence.classification}`",
        f"- promotion_ready: `{evidence.promotion_ready}`",
        f"- alias_activity_verified: `{evidence.alias_activity_verified}`",
        f"- complete lattice cases: `{evidence.complete_lattice_case_count}`",
        f"- blocked lattice cases: `{evidence.blocked_lattice_case_count}`",
        "",
        "Observed alias-edge kinds:",
        "",
        *[f"- `{kind}`" for kind in evidence.observed_alias_edge_kinds],
        "",
        "| Case | Status | Alias kinds | Blockers |",
        "|---|---|---|---|",
    ]
    for case in evidence.cases:
        alias_kinds = ", ".join(f"`{kind}`" for kind in case.alias_edge_kinds)
        blockers = ", ".join(f"`{reason}`" for reason in case.blocker_reasons) or "none"
        lines.append(f"| `{case.case_id}` | `{case.status}` | {alias_kinds} | {blockers} |")
    lines.extend(
        [
            "",
            "Focused test evidence:",
            "",
            *[f"- `{test_id}`" for test_id in evidence.test_ids],
            "",
            f"Claim boundary: {evidence.claim_boundary}",
            "",
        ]
    )
    return "\n".join(lines)


Objective = Callable[[Any], object]


def _case_from_objective(
    case_id: str,
    objective: Objective,
    values: NDArray[np.float64],
    parameter_names: Sequence[str],
) -> CompilerAliasActivityCase:
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(name) for name in parameter_names),
    )
    if result.program_ir is None:
        raise RuntimeError("whole-program AD did not emit Program AD IR")
    return _case_from_report(case_id, program_ad_static_alias_lattice_report(result.program_ir))


def _case_from_report(
    case_id: str,
    report: ProgramADStaticAliasLatticeReport,
) -> CompilerAliasActivityCase:
    edge_kinds = tuple(
        sorted({kind for component in report.components for kind in component.edge_kinds})
    )
    status = "complete_lattice" if report.complete else "blocked_lattice"
    return CompilerAliasActivityCase(
        case_id=case_id,
        status=status,
        complete=report.complete,
        alias_edge_kinds=edge_kinds,
        blocker_reasons=report.blocker_reasons,
        component_count=len(report.components),
        provenance_counts=_provenance_counts(report),
    )


def _provenance_counts(report: ProgramADStaticAliasLatticeReport) -> dict[str, int]:
    return {
        "control_path_alias": len(report.control_path_alias_provenance),
        "malformed_control_path_alias": len(report.malformed_control_path_alias_edges),
        "malformed_list_alias": len(report.malformed_list_alias_edges),
        "malformed_loop_carried_state": len(report.malformed_loop_carried_state_edges),
        "malformed_rebinding_alias": len(report.malformed_rebinding_alias_edges),
        "malformed_view_alias": len(report.malformed_view_alias_edges),
        "list_alias": len(report.list_alias_provenance),
        "loop_carried_state": len(report.loop_carried_state_provenance),
        "rebinding_alias": len(report.rebinding_alias_provenance),
        "unknown_alias_edge": len(report.unknown_alias_edges),
        "view_alias": len(report.view_alias_provenance),
    }


def _view_alias_case() -> CompilerAliasActivityCase:
    def objective(values: Any) -> object:
        view = values.reshape((2, 2)).T.ravel()
        return view[0] + 2.0 * view[3]

    return _case_from_objective(
        "complete_view_alias",
        objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        ("a", "b", "c", "d"),
    )


def _list_alias_case() -> CompilerAliasActivityCase:
    def objective(values: Any) -> object:
        scratch = [values[0], values[1]]
        alias = scratch
        alias[0] = values[2]
        return scratch[0] + 2.0 * alias[1]

    return _case_from_objective(
        "complete_list_alias",
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        ("a", "b", "c"),
    )


def _scalar_rebinding_case() -> CompilerAliasActivityCase:
    def objective(values: Any) -> object:
        seed = values[0]
        rebound = seed
        combined = rebound + 2.0 * values[1]
        return combined + values[2]

    return _case_from_objective(
        "complete_scalar_rebinding_alias",
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        ("a", "b", "c"),
    )


def _object_attribute_case() -> CompilerAliasActivityCase:
    class Scratch:
        """Mutable local container for bounded object-attribute alias metadata."""

        left: Any
        right: Any
        total: Any

    def objective(values: Any) -> object:
        scratch = Scratch()
        scratch.left = values[0]
        scratch.right = values[1]
        alias = scratch.left
        combined = alias + 2.0 * scratch.right
        scratch.total = combined
        return scratch.total + values[2]

    return _case_from_objective(
        "complete_object_attribute_alias",
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        ("a", "b", "c"),
    )


def _loop_carried_state_case() -> CompilerAliasActivityCase:
    def objective(values: Any) -> object:
        carry = values[0]
        for index in range(1, 4):
            carry = carry + float(index) * values[index]
        return carry + values[4]

    return _case_from_objective(
        "complete_loop_carried_state_alias",
        objective,
        np.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64),
        ("seed", "step1", "step2", "step3", "tail"),
    )


def _non_executed_branch_case() -> CompilerAliasActivityCase:
    def objective(values: Any) -> object:
        total = values[0]
        if values[1] > 0.0:
            total = total + values[2]
        else:
            total = total - values[3]
        return total

    return _case_from_objective(
        "blocked_non_executed_branch_alias",
        objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        ("a", "b", "c", "d"),
    )


def _branch_attribute_case() -> CompilerAliasActivityCase:
    class Scratch:
        """Mutable local container for branch-local attribute alias blockers."""

        value: object

    def objective(values: Any) -> object:
        scratch = Scratch()
        if values[2] > 0.0:
            scratch.value = values[0]
        else:
            scratch.value = values[1]
        alias = scratch.value
        return alias + values[3]

    return _case_from_objective(
        "blocked_branch_attribute_alias",
        objective,
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        ("a", "b", "c", "d"),
    )


def _slice_mutation_case() -> CompilerAliasActivityCase:
    def objective(values: Any) -> object:
        window = values.reshape((6,))[1:5]
        window[1:3] = np.array([2.0 * values[0], values[5] + 1.0])
        return window[0] + window[1] + window[2] + window[3]

    return _case_from_objective(
        "blocked_static_slice_mutation_alias",
        objective,
        np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
        ("a", "b", "c", "d", "e", "f"),
    )


_ALIAS_ACTIVITY_TEST_IDS = tuple(
    sorted(
        {
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_complete_emitted_ir",
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_list_alias_provenance",
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_loop_carried_state_provenance",
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_rebinding_provenance",
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_tracks_local_object_attribute_aliases",
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_records_non_executed_phi_blockers",
            "tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_blocks_non_executed_attribute_paths",
            "tests/test_program_ad_alias_effects.py::test_program_ad_alias_effect_analysis_tracks_static_slice_mutation",
        }
    )
)


__all__ = [
    "COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID",
    "COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA",
    "CompilerAliasActivityCase",
    "CompilerAliasActivityEvidence",
    "build_compiler_alias_activity_evidence",
    "render_compiler_alias_activity_evidence_markdown",
]
