# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — gradient-plan explanation committed artefact.
"""Committed gradient-plan explanations for the Studio cockpit.

The planner surface in :mod:`scpn_quantum_control.phase.gradient_support_matrix`
answers whether a gate/observable/backend/transform/adapter request may execute,
which method it selects, and why unsupported combinations fail closed. This
module serialises the built-in executable planner audit into a committed JSON
artefact and a reviewer-facing markdown table for the Studio panel.

Emission is fail-closed. A planner audit that no longer passes, carries no
supported routes, or carries no blocked routes is not serialised; validation
compares the committed files against a fresh audit regeneration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .differentiable_claim_ledger import REPO_ROOT
from .phase.gradient_support_matrix import (
    GradientSupportMatrixAuditResult,
    GradientSupportPlan,
    run_gradient_support_matrix_audit,
)

GRADIENT_PLAN_EXPLANATION_SCHEMA: Final[str] = "scpn_qc_gradient_plan_explanations_v1"
"""Schema identifier stamped into the committed gradient-plan artefact."""

GRADIENT_PLAN_EXPLANATION_ARTIFACT_ID: Final[str] = "gradient-plan-explanations-20260709"
"""Artifact identifier of the committed gradient-plan explanation emission."""

DEFAULT_GRADIENT_PLAN_EXPLANATION_JSON_PATH: Final[Path] = Path(
    "data/differentiable_phase_qnode/gradient_plan_explanations_20260709.json"
)
"""Repository-relative path of the committed JSON artefact."""

DEFAULT_GRADIENT_PLAN_EXPLANATION_MARKDOWN_PATH: Final[Path] = Path(
    "data/differentiable_phase_qnode/gradient_plan_explanations_20260709.md"
)
"""Repository-relative path of the committed markdown rendering."""

_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.gradient_plan_explanation_artifact --write"
)

_MARKDOWN_HEADER: Final[str] = """<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Gradient-Plan Explanations
-->"""


@dataclass(frozen=True)
class GradientPlanExplanationArtifactValidation:
    """Validation verdict for the committed gradient-plan artefact.

    Parameters
    ----------
    passed
        Whether the committed payload and markdown match a fresh planner audit.
    errors
        Human-readable mismatch descriptions, empty when ``passed``.
    """

    passed: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that the verdict agrees with its error list."""
        if self.passed and self.errors:
            raise ValueError("a passed validation must not carry errors")
        if not self.passed and not self.errors:
            raise ValueError("a failed validation must explain its errors")


def build_gradient_plan_explanation_artifact(
    audit: GradientSupportMatrixAuditResult | None = None,
) -> dict[str, object]:
    """Build the committed gradient-plan explanation payload.

    Parameters
    ----------
    audit
        Planner audit to serialise, or ``None`` to run
        :func:`run_gradient_support_matrix_audit` on the current tree.

    Returns
    -------
    dict[str, object]
        JSON-ready payload with one explanation row per planner audit case.

    Raises
    ------
    ValueError
        If the planner audit is not passing, or no supported/blocked contrast
        exists for the Studio explanation view.
    """
    resolved = run_gradient_support_matrix_audit() if audit is None else audit
    if not resolved.passed:
        failures = ", ".join(_cell_id(plan) for plan in resolved.failing_plans)
        raise ValueError(f"gradient support audit failed and is not serialised: {failures}")
    supported = resolved.supported_plans
    blocked = resolved.blocked_plans
    if not supported:
        raise ValueError("gradient support audit has no supported plans to explain")
    if not blocked:
        raise ValueError("gradient support audit has no fail-closed plans to explain")
    explanations = [_plan_explanation(plan, index) for index, plan in enumerate(resolved.plans, 1)]
    method_families = sorted(
        {value for row in explanations if isinstance((value := row.get("method_family")), str)}
    )
    return {
        "schema": GRADIENT_PLAN_EXPLANATION_SCHEMA,
        "artifact_id": GRADIENT_PLAN_EXPLANATION_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "claim_boundary": resolved.claim_boundary,
        "plan_count": len(resolved.plans),
        "supported_plan_count": len(supported),
        "blocked_plan_count": len(blocked),
        "method_families": method_families,
        "explanations": explanations,
    }


def validate_gradient_plan_explanation_artifact(
    payload: dict[str, object],
    *,
    audit: GradientSupportMatrixAuditResult | None = None,
) -> GradientPlanExplanationArtifactValidation:
    """Validate a committed payload against a fresh planner regeneration.

    Parameters
    ----------
    payload
        Parsed committed JSON payload.
    audit
        Optional audit to regenerate from; defaults to a fresh run.

    Returns
    -------
    GradientPlanExplanationArtifactValidation
        Verdict with per-field mismatch descriptions.
    """
    errors: list[str] = []
    reference = build_gradient_plan_explanation_artifact(audit)
    for key, expected in reference.items():
        if payload.get(key) != expected:
            errors.append(f"field {key!r} does not match the regenerated artefact")
    return GradientPlanExplanationArtifactValidation(passed=not errors, errors=tuple(errors))


def render_gradient_plan_explanation_markdown(payload: dict[str, object]) -> str:
    """Render a gradient-plan payload as a reviewer-facing markdown table.

    Parameters
    ----------
    payload
        Artefact payload from :func:`build_gradient_plan_explanation_artifact`.

    Returns
    -------
    str
        Markdown document with artefact metadata and one row per planner case.
    """
    rows = _explanation_rows(payload)
    lines = [
        _MARKDOWN_HEADER,
        "",
        "# Gradient-Plan Explanations",
        "",
        f"- Schema: `{payload.get('schema')}`",
        f"- Artifact ID: `{payload.get('artifact_id')}`",
        f"- Plans: `{payload.get('supported_plan_count')}` supported, `{payload.get('blocked_plan_count')}` fail-closed",
        f"- Method families: `{', '.join(_string_list(payload.get('method_families')))}`",
        f"- Claim boundary: {payload.get('claim_boundary')}",
        "",
        "| Cell | Framework | Backend | Transform | Selected method | Status | Why | Boundaries |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_markdown_row(row) for row in rows)
    lines.extend(
        [
            "",
            "Supported rows explain bounded planner choices. Blocked rows are",
            "fail-closed planning boundaries and do not permit derivative execution.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for emission, validation, and JSON preview.

    Parameters
    ----------
    argv
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` when the requested operation succeeds, ``1`` when check mode
        detects drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the JSON and markdown files")
    parser.add_argument("--check", action="store_true", help="validate committed files")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_GRADIENT_PLAN_EXPLANATION_JSON_PATH,
        help="JSON artefact path",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=DEFAULT_GRADIENT_PLAN_EXPLANATION_MARKDOWN_PATH,
        help="markdown artefact path",
    )
    args = parser.parse_args(argv)
    payload = build_gradient_plan_explanation_artifact()
    rendered = render_gradient_plan_explanation_markdown(payload)
    json_path = _repo_path(args.json_path)
    markdown_path = _repo_path(args.markdown_path)
    if args.write:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(rendered, encoding="utf-8")
        return 0
    if args.check:
        errors: list[str] = []
        if not json_path.exists():
            errors.append(f"missing JSON artefact: {json_path}")
        else:
            committed = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(committed, dict):
                errors.append("committed JSON artefact must be an object")
            else:
                validation = validate_gradient_plan_explanation_artifact(committed)
                errors.extend(validation.errors)
        if not markdown_path.exists():
            errors.append(f"missing markdown artefact: {markdown_path}")
        elif markdown_path.read_text(encoding="utf-8") != rendered:
            errors.append("markdown artefact does not match the regenerated rendering")
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("gradient-plan explanation artefact is current")
        return 0
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cell_id(plan: GradientSupportPlan) -> str:
    """Return the stable cell identifier for one planner request."""
    return "::".join((plan.gate, plan.observable, plan.backend, plan.transform, plan.adapter))


def _plan_explanation(plan: GradientSupportPlan, index: int) -> dict[str, object]:
    """Return the JSON-ready explanation row for one planner decision."""
    return {
        "index": index,
        "cell_id": _cell_id(plan),
        "operation": f"{plan.gate}.{plan.observable}.{plan.transform}",
        "framework": plan.adapter,
        "backend": plan.backend,
        "transform": plan.transform,
        "supported": plan.supported,
        "status": "supported" if plan.supported else "fail_closed",
        "selected_method": plan.recommended_method,
        "method_family": _method_family(plan),
        "evaluation_mode": plan.evaluation_mode,
        "backend_family": plan.backend_plan.family,
        "backend_evaluations": plan.backend_plan.evaluations,
        "shots": plan.backend_plan.shots,
        "requires_finite_shot_variance": plan.requires_finite_shot_variance,
        "requires_hardware_policy": plan.requires_hardware_policy,
        "why": _decision_reasons(plan),
        "fail_closed_boundaries": list(plan.blocked_reasons),
        "warnings": list(plan.warnings),
        "alternatives": list(plan.alternatives),
        "claim_boundary": plan.claim_boundary,
    }


def _method_family(plan: GradientSupportPlan) -> str:
    """Classify a planner method into the coarse cockpit vocabulary."""
    method = plan.recommended_method
    if not plan.supported:
        return "unsupported"
    if "adjoint" in method:
        return "adjoint"
    if "finite_difference" in method or "fd" in method:
        return "finite-difference"
    if "parameter_shift" in method or "shifted_circuit" in method:
        return "parameter-shift"
    if plan.backend_plan.family == "statevector":
        return "exact-local"
    return method


def _decision_reasons(plan: GradientSupportPlan) -> list[str]:
    """Build human-readable reasons for one supported or blocked plan."""
    if plan.blocked_reasons:
        return list(plan.blocked_reasons)
    reasons = [
        f"{capability.category}:{capability.name} supports {', '.join(capability.gradient_methods)}"
        for capability in plan.capabilities
        if capability.supported and capability.gradient_methods
    ]
    reasons.append(
        f"backend planner selected {plan.backend_plan.method} with {plan.backend_plan.evaluations} evaluations"
    )
    return reasons


def _explanation_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return explanation rows, failing closed on malformed payloads."""
    rows = payload.get("explanations")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("gradient-plan payload must carry a list of explanation objects")
    return rows


def _markdown_row(row: dict[str, object]) -> str:
    """Render one planner explanation row as markdown."""
    why = "; ".join(_string_list(row.get("why")))
    boundaries = "; ".join(_string_list(row.get("fail_closed_boundaries"))) or "-"
    return (
        f"| `{row.get('cell_id')}` | `{row.get('framework')}` | `{row.get('backend')}` | "
        f"`{row.get('transform')}` | `{row.get('selected_method')}` | `{row.get('status')}` | "
        f"{why} | {boundaries} |"
    )


def _string_list(value: object) -> list[str]:
    """Return string items from a JSON list, or an empty list for malformed cells."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _repo_path(path: Path) -> Path:
    """Resolve repository-relative paths under the working tree."""
    return path if path.is_absolute() else REPO_ROOT / path


__all__ = [
    "DEFAULT_GRADIENT_PLAN_EXPLANATION_JSON_PATH",
    "DEFAULT_GRADIENT_PLAN_EXPLANATION_MARKDOWN_PATH",
    "GRADIENT_PLAN_EXPLANATION_ARTIFACT_ID",
    "GRADIENT_PLAN_EXPLANATION_SCHEMA",
    "GradientPlanExplanationArtifactValidation",
    "build_gradient_plan_explanation_artifact",
    "main",
    "render_gradient_plan_explanation_markdown",
    "validate_gradient_plan_explanation_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
