# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — gradient-plan explanation artefact tests
"""Tests for the committed gradient-plan explanation artefact (ST-17)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.gradient_plan_explanation_artifact import (
    DEFAULT_GRADIENT_PLAN_EXPLANATION_JSON_PATH,
    DEFAULT_GRADIENT_PLAN_EXPLANATION_MARKDOWN_PATH,
    GRADIENT_PLAN_EXPLANATION_ARTIFACT_ID,
    GRADIENT_PLAN_EXPLANATION_SCHEMA,
    GradientPlanExplanationArtifactValidation,
    build_gradient_plan_explanation_artifact,
    main,
    render_gradient_plan_explanation_markdown,
    validate_gradient_plan_explanation_artifact,
)
from scpn_quantum_control.phase import run_gradient_support_matrix_audit

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_JSON = REPO_ROOT / DEFAULT_GRADIENT_PLAN_EXPLANATION_JSON_PATH
COMMITTED_MARKDOWN = REPO_ROOT / DEFAULT_GRADIENT_PLAN_EXPLANATION_MARKDOWN_PATH


def test_payload_explains_supported_and_blocked_planner_cells() -> None:
    """The payload mirrors every built-in planner audit row."""
    audit = run_gradient_support_matrix_audit()
    payload = build_gradient_plan_explanation_artifact(audit)
    rows = cast(list[dict[str, object]], payload["explanations"])

    assert payload["schema"] == GRADIENT_PLAN_EXPLANATION_SCHEMA
    assert payload["artifact_id"] == GRADIENT_PLAN_EXPLANATION_ARTIFACT_ID
    assert payload["claim_boundary"] == audit.claim_boundary
    assert payload["plan_count"] == len(audit.plans)
    assert payload["supported_plan_count"] == 5
    assert payload["blocked_plan_count"] == 5
    assert len(rows) == 10
    assert rows[0]["cell_id"] == "ry::pauli_expectation::statevector::grad::native"
    assert rows[0]["selected_method"] == "parameter_shift"
    assert rows[0]["method_family"] == "parameter-shift"
    assert rows[0]["status"] == "supported"
    assert rows[-1]["status"] == "fail_closed"
    assert rows[-1]["method_family"] == "unsupported"
    assert rows[-1]["fail_closed_boundaries"]


def test_payload_refuses_failed_or_one_sided_audits() -> None:
    """Failed audits and one-sided planner samples are not serialised."""
    audit = run_gradient_support_matrix_audit()
    failed = dataclasses.replace(audit, passed=False)
    supported_only = dataclasses.replace(audit, plans=audit.supported_plans)
    blocked_only = dataclasses.replace(audit, plans=audit.blocked_plans)

    with pytest.raises(ValueError, match="audit failed"):
        build_gradient_plan_explanation_artifact(failed)
    with pytest.raises(ValueError, match="no fail-closed plans"):
        build_gradient_plan_explanation_artifact(supported_only)
    with pytest.raises(ValueError, match="no supported plans"):
        build_gradient_plan_explanation_artifact(blocked_only)


def test_markdown_renders_rows_and_boundaries() -> None:
    """The markdown table carries supported and blocked planner rows."""
    payload = build_gradient_plan_explanation_artifact()
    rendered = render_gradient_plan_explanation_markdown(payload)

    assert "# Gradient-Plan Explanations" in rendered
    assert "| `ry::pauli_expectation::statevector::grad::native` |" in rendered
    assert "| `ry::pauli_expectation::qasm_simulator::hessian::native` |" in rendered
    assert "provider_callback_stochastic_parameter_shift" in rendered
    assert "hessian support is limited to deterministic local backends" in rendered
    assert rendered.endswith(
        "fail-closed planning boundaries and do not permit derivative execution.\n"
    )


def test_markdown_fails_closed_on_malformed_payload() -> None:
    """Malformed payload rows refuse to render."""
    with pytest.raises(ValueError, match="list of explanation objects"):
        render_gradient_plan_explanation_markdown({"explanations": "bad"})


def test_markdown_renders_malformed_cells_as_empty() -> None:
    """Malformed list cells render as empty cells rather than crashing."""
    payload = build_gradient_plan_explanation_artifact()
    rows = cast(list[dict[str, object]], payload["explanations"])
    rows[0]["why"] = "not-a-list"
    rendered = render_gradient_plan_explanation_markdown(payload)
    assert rendered.count("| `ry::pauli_expectation::statevector::grad::native` |") == 1


def test_committed_artifact_is_current() -> None:
    """The committed JSON and markdown match fresh planner regeneration."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    validation = validate_gradient_plan_explanation_artifact(committed)
    assert validation.passed, validation.errors
    rendered = render_gradient_plan_explanation_markdown(committed)
    assert COMMITTED_MARKDOWN.read_text(encoding="utf-8") == rendered


def test_validation_flags_tampered_payload() -> None:
    """Tampered fields are reported as validation drift."""
    committed = build_gradient_plan_explanation_artifact()
    committed["supported_plan_count"] = 99
    rows = cast(list[dict[str, object]], committed["explanations"])
    rows[0]["selected_method"] = "finite_difference"

    validation = validate_gradient_plan_explanation_artifact(committed)
    assert not validation.passed
    assert any("supported_plan_count" in error for error in validation.errors)
    assert any("explanations" in error for error in validation.errors)


def test_method_family_classifies_non_parameter_shift_routes() -> None:
    """The cockpit vocabulary covers adjoint, finite-difference, exact, and custom methods."""
    audit = run_gradient_support_matrix_audit()
    base = audit.supported_plans[0]
    backend_plan = base.backend_plan
    variants = (
        dataclasses.replace(base, recommended_method="adjoint_gradient"),
        dataclasses.replace(base, recommended_method="finite_difference"),
        dataclasses.replace(base, recommended_method="exact_value", backend_plan=backend_plan),
        dataclasses.replace(
            base,
            recommended_method="custom_rule",
            backend_plan=dataclasses.replace(backend_plan, family="custom_backend"),
        ),
    )
    rebuilt = dataclasses.replace(audit, plans=(*variants, *audit.blocked_plans))
    payload = build_gradient_plan_explanation_artifact(rebuilt)
    rows = cast(list[dict[str, object]], payload["explanations"])
    assert [row["method_family"] for row in rows[:4]] == [
        "adjoint",
        "finite-difference",
        "exact-local",
        "custom_rule",
    ]


def test_validation_verdict_invariants_hold() -> None:
    """The validation dataclass rejects inconsistent verdicts."""
    with pytest.raises(ValueError, match="must not carry errors"):
        GradientPlanExplanationArtifactValidation(passed=True, errors=("x",))
    with pytest.raises(ValueError, match="must explain its errors"):
        GradientPlanExplanationArtifactValidation(passed=False, errors=())


def test_main_write_and_check_round_trip(tmp_path: Path) -> None:
    """Write mode emits both files and check mode accepts them."""
    json_path = tmp_path / "plans.json"
    markdown_path = tmp_path / "plans.md"
    args = ["--json-path", str(json_path), "--markdown-path", str(markdown_path)]

    assert main(["--write", *args]) == 0
    assert json_path.exists()
    assert markdown_path.exists()
    assert main(["--check", *args]) == 0
    markdown_path.write_text("stale\n", encoding="utf-8")
    assert main(["--check", *args]) == 1
    tampered = json.loads(json_path.read_text(encoding="utf-8"))
    tampered["plan_count"] = 0
    json_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert main(["--check", *args]) == 1


def test_main_check_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    """Check mode reports absent files and non-object JSON artefacts."""
    missing_json = tmp_path / "missing.json"
    missing_markdown = tmp_path / "missing.md"
    assert (
        main(
            [
                "--check",
                "--json-path",
                str(missing_json),
                "--markdown-path",
                str(missing_markdown),
            ]
        )
        == 1
    )
    json_path = tmp_path / "scalar.json"
    markdown_path = tmp_path / "present.md"
    json_path.write_text("[]\n", encoding="utf-8")
    markdown_path.write_text(
        render_gradient_plan_explanation_markdown(build_gradient_plan_explanation_artifact()),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--check",
                "--json-path",
                str(json_path),
                "--markdown-path",
                str(markdown_path),
            ]
        )
        == 1
    )


def test_main_default_mode_prints_payload(capsys: pytest.CaptureFixture[str]) -> None:
    """Without flags the CLI prints the regenerated JSON payload."""
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == GRADIENT_PLAN_EXPLANATION_SCHEMA
    assert len(payload["explanations"]) == 10
