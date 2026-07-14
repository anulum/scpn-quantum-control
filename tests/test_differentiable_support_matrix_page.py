# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generated differentiable support-matrix page tests
"""Real-surface contracts for the generated differentiable support-matrix page."""

from __future__ import annotations

import copy
import dataclasses
import json
import runpy
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.phase import run_gradient_support_matrix_audit
from scpn_quantum_control.program_ad_registry import (
    program_ad_registry_dispatch_coverage_report,
)
from tools.differentiable_support_matrix_page import (
    DEFAULT_CAPABILITY_MANIFEST_PATH,
    DEFAULT_PAGE_PATH,
    SUPPORT_MATRIX_PAGE_SCHEMA,
    SupportMatrixCapabilityManifestValidation,
    audit_differentiable_support_matrix_page,
    build_differentiable_support_matrix_payload,
    load_capability_manifest,
    main,
    render_differentiable_support_matrix_page,
    validate_support_matrix_capability_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools/differentiable_support_matrix_page.py"


def _mapping(container: dict[str, object], key: str) -> dict[str, object]:
    """Return one mutable nested mapping from a test payload."""
    value = container[key]
    assert isinstance(value, dict)
    assert all(isinstance(item, str) for item in value)
    return cast(dict[str, object], value)


def _list(container: dict[str, object], key: str) -> list[object]:
    """Return one mutable nested list from a test payload."""
    value = container[key]
    assert isinstance(value, list)
    return value


def _row(value: object) -> dict[str, object]:
    """Return one mutable string-keyed row mapping."""
    assert isinstance(value, dict)
    assert all(isinstance(item, str) for item in value)
    return cast(dict[str, object], value)


def _manifest_copy() -> dict[str, object]:
    """Return a mutable copy of the committed capability manifest."""
    return copy.deepcopy(load_capability_manifest())


def test_payload_comes_from_live_registry_and_planner_surfaces() -> None:
    """The generated payload mirrors both executable owners row for row."""
    registry = program_ad_registry_dispatch_coverage_report()
    planner = run_gradient_support_matrix_audit()

    payload = build_differentiable_support_matrix_payload(
        registry_report=registry,
        planner_audit=planner,
    )
    registry_payload = _mapping(payload, "registry")
    planner_payload = _mapping(payload, "planner")

    assert payload["schema"] == SUPPORT_MATRIX_PAGE_SCHEMA
    assert registry_payload["covered_primitives"] == registry.covered_primitives == 118
    assert registry_payload["total_primitives"] == registry.total_primitives == 118
    assert registry_payload["rows"] == [row.to_dict() for row in registry.rows]
    assert planner.passed
    assert planner_payload["supported_plan_count"] == len(planner.supported_plans) == 5
    assert planner_payload["blocked_plan_count"] == len(planner.blocked_plans) == 5
    assert planner_payload["rows"] == [plan.to_dict() for plan in planner.plans]


def test_payload_rejects_a_failed_planner_audit_with_named_cells() -> None:
    """A planner invariant failure cannot be serialised as public support."""
    audit = run_gradient_support_matrix_audit()
    first = dataclasses.replace(
        audit.plans[0],
        supported=False,
        blocked_reasons=("forced regression",),
    )
    failed = dataclasses.replace(audit, plans=(first, *audit.plans[1:]), passed=False)

    with pytest.raises(ValueError, match="ry::pauli_expectation::statevector::grad::native"):
        build_differentiable_support_matrix_payload(planner_audit=failed)


def test_payload_rejects_false_or_one_sided_planner_evidence() -> None:
    """False verdicts and audits without both support classes fail closed."""
    audit = run_gradient_support_matrix_audit()
    false_without_failing_rows = dataclasses.replace(audit, passed=False)
    supported_only = dataclasses.replace(audit, plans=audit.supported_plans)
    blocked_only = dataclasses.replace(audit, plans=audit.blocked_plans)

    with pytest.raises(ValueError, match="audit verdict is false"):
        build_differentiable_support_matrix_payload(planner_audit=false_without_failing_rows)
    with pytest.raises(ValueError, match="no fail-closed cases"):
        build_differentiable_support_matrix_payload(planner_audit=supported_only)
    with pytest.raises(ValueError, match="no supported cases"):
        build_differentiable_support_matrix_payload(planner_audit=blocked_only)


def test_manifest_validation_accepts_the_committed_inventory() -> None:
    """The real manifest inventories every source, test, export, and page owner."""
    validation = validate_support_matrix_capability_manifest(load_capability_manifest())

    assert validation.passed, validation.errors
    assert validation.errors == ()


def test_manifest_validation_verdict_invariants_are_fail_closed() -> None:
    """Contradictory validation records are rejected at construction."""
    with pytest.raises(ValueError, match="must not carry errors"):
        SupportMatrixCapabilityManifestValidation(passed=True, errors=("unexpected",))
    with pytest.raises(ValueError, match="must explain"):
        SupportMatrixCapabilityManifestValidation(passed=False, errors=())


def test_manifest_validation_names_missing_required_surfaces() -> None:
    """Every required export, module, class, test, and page is checked explicitly."""
    manifest = _manifest_copy()
    exports = _list(manifest, "package_exports")
    exports.remove("program_ad_registry_dispatch_coverage_report")
    models = _mapping(manifest, "models")
    sources = _list(models, "python_source_modules")
    sources.remove("src/scpn_quantum_control/program_ad_registry.py")
    classes = _list(models, "python_classes")
    classes[:] = [
        item for item in classes if _row(item).get("name") != "GradientSupportMatrixAuditResult"
    ]
    quality = _mapping(manifest, "quality_gates")
    tests = _list(quality, "test_files")
    tests.remove("tests/test_differentiable_support_matrix_page.py")
    documentation = _mapping(manifest, "documentation")
    pages = _list(documentation, "public_pages")
    pages.remove("docs/differentiable_support_matrix.md")

    validation = validate_support_matrix_capability_manifest(manifest)

    assert not validation.passed
    joined = "\n".join(validation.errors)
    assert "required package export" in joined
    assert "required Python source module" in joined
    assert "GradientSupportMatrixAuditResult" in joined
    assert "required focused test" in joined
    assert "required public documentation page" in joined


def test_manifest_validation_rejects_identity_schema_and_count_drift() -> None:
    """Manifest identity and list-derived counts cannot drift independently."""
    manifest = _manifest_copy()
    manifest["schema_version"] = "foreign.v1"
    _mapping(manifest, "generated_from")["generator"] = "other.py"
    _mapping(manifest, "project")["name"] = "other-project"
    counts = _mapping(manifest, "counts")
    for name in (
        "public_api_exports",
        "python_model_source_modules",
        "python_model_classes",
        "test_files",
        "public_documentation_pages",
    ):
        value = counts[name]
        assert isinstance(value, int)
        counts[name] = value + 1

    validation = validate_support_matrix_capability_manifest(manifest)

    assert not validation.passed
    joined = "\n".join(validation.errors)
    assert "schema_version" in joined
    assert "generator identity" in joined
    assert "project identity" in joined
    assert joined.count("does not match its surface length") == 5


@pytest.mark.parametrize(
    ("section", "replacement", "expected"),
    [
        ("generated_from", "invalid", "generator identity"),
        ("project", "invalid", "project identity"),
        ("models", "invalid", "models object is missing"),
        ("quality_gates", "invalid", "quality_gates object is missing"),
        ("documentation", "invalid", "documentation object is missing"),
        ("counts", "invalid", "counts object is missing"),
    ],
)
def test_manifest_validation_reports_malformed_top_level_sections(
    section: str,
    replacement: object,
    expected: str,
) -> None:
    """Malformed manifest sections yield diagnostics rather than exceptions."""
    manifest = _manifest_copy()
    manifest[section] = replacement

    validation = validate_support_matrix_capability_manifest(manifest)

    assert not validation.passed
    assert expected in "\n".join(validation.errors)


def test_manifest_validation_reports_malformed_nested_inventory_rows() -> None:
    """Malformed source, class, test, and page lists remain fail-closed."""
    manifest = _manifest_copy()
    models = _mapping(manifest, "models")
    models["python_source_modules"] = [1]
    models["python_classes"] = ["not-an-object", {"name": 1, "path": 2}]
    _mapping(manifest, "quality_gates")["test_files"] = [1]
    _mapping(manifest, "documentation")["public_pages"] = [1]

    validation = validate_support_matrix_capability_manifest(manifest)

    joined = "\n".join(validation.errors)
    assert "python_source_modules must be a string list" in joined
    assert "Python class rows must be objects" in joined
    assert "Python class rows require string name/path" in joined
    assert "quality_gates.test_files must be a string list" in joined
    assert "documentation.public_pages must be a string list" in joined


def test_manifest_validation_reports_non_list_class_inventory() -> None:
    """The manifest class inventory must remain an explicit object list."""
    manifest = _manifest_copy()
    _mapping(manifest, "models")["python_classes"] = "invalid"

    validation = validate_support_matrix_capability_manifest(manifest)

    assert "models.python_classes must be an object list" in "\n".join(validation.errors)


def test_manifest_validation_rejects_malformed_page_requirements() -> None:
    """A malformed generator requirement block cannot bypass the manifest gate."""
    payload = build_differentiable_support_matrix_payload()
    payload["capability_manifest_requirements"] = "invalid"

    validation = validate_support_matrix_capability_manifest(
        load_capability_manifest(),
        payload=payload,
    )

    assert not validation.passed
    assert validation.errors == (
        "capability_manifest_requirements must be an object with string keys",
    )


def test_manifest_validation_reports_malformed_required_surface_lists() -> None:
    """Required export, module, test, and page lists fail closed by name."""
    payload = build_differentiable_support_matrix_payload()
    requirements = _mapping(payload, "capability_manifest_requirements")
    for name in ("package_exports", "python_source_modules", "test_files", "public_pages"):
        requirements[name] = "invalid"

    validation = validate_support_matrix_capability_manifest(
        load_capability_manifest(),
        payload=payload,
    )

    joined = "\n".join(validation.errors)
    for name in ("package_exports", "python_source_modules", "test_files", "public_pages"):
        assert f"{name} must be a list of strings" in joined


def test_manifest_validation_reports_malformed_required_class_rows() -> None:
    """Class requirements must remain explicit name/path objects."""
    payload = build_differentiable_support_matrix_payload()
    requirements = _mapping(payload, "capability_manifest_requirements")
    requirements["python_classes"] = ["invalid", {"name": 1, "path": 2}]

    validation = validate_support_matrix_capability_manifest(
        load_capability_manifest(),
        payload=payload,
    )

    joined = "\n".join(validation.errors)
    assert "requirement rows must be objects" in joined
    assert "requirement rows require string name/path" in joined


def test_manifest_validation_reports_non_list_required_classes() -> None:
    """The required-class contract must remain an explicit object list."""
    payload = build_differentiable_support_matrix_payload()
    _mapping(payload, "capability_manifest_requirements")["python_classes"] = "invalid"

    validation = validate_support_matrix_capability_manifest(
        load_capability_manifest(),
        payload=payload,
    )

    assert "python_classes requirements must be an object list" in "\n".join(validation.errors)


def test_manifest_loader_rejects_non_object_json(tmp_path: Path) -> None:
    """Only string-keyed JSON objects are admitted as manifests."""
    path = tmp_path / "manifest.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_capability_manifest(path)


def test_renderer_emits_every_live_registry_and_planner_row() -> None:
    """The page contains every executable row and its conservative boundaries."""
    registry = program_ad_registry_dispatch_coverage_report()
    planner = run_gradient_support_matrix_audit()
    rendered = render_differentiable_support_matrix_page(
        build_differentiable_support_matrix_payload(
            registry_report=registry,
            planner_audit=planner,
        )
    )

    assert rendered.startswith("<!-- Generated by tools/differentiable_support_matrix_page.py")
    assert "# Differentiable Support Matrix" in rendered
    assert "118/118" in rendered
    for plan in planner.plans:
        cell = "::".join((plan.gate, plan.observable, plan.backend, plan.transform, plan.adapter))
        assert f"`{cell}`" in rendered
    for row in registry.rows:
        assert f"`{row.identity}`" in rendered
    assert registry.claim_boundary in rendered
    assert planner.claim_boundary in rendered
    assert rendered.endswith("performance claim.\n")


def test_renderer_escapes_table_controls_in_runtime_text() -> None:
    """Runtime strings cannot break generated Markdown table structure."""
    payload = build_differentiable_support_matrix_payload()
    planner = _mapping(payload, "planner")
    first = _row(_list(planner, "rows")[0])
    first["warnings"] = ["line one|line two\ncontinued`code`"]

    rendered = render_differentiable_support_matrix_page(payload)

    assert "line one\\|line two continued\\`code\\`" in rendered


def test_renderer_preserves_future_incomplete_registry_rows() -> None:
    """An incomplete future registry row stays visible with its exact gap."""
    payload = build_differentiable_support_matrix_payload()
    first = _row(_list(_mapping(payload, "registry"), "rows")[0])
    first["complete"] = False
    first["derivative_rule"] = None
    first["blocked_reasons"] = ["missing derivative rule"]

    rendered = render_differentiable_support_matrix_page(payload)

    assert "`n/a`" in rendered
    assert "`fail-closed: missing derivative rule`" in rendered


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("schema", "unexpected schema"),
        ("registry", "registry must be an object"),
        ("planner_rows", "planner.rows must be a list"),
        ("family_counts", "positive integers"),
        ("empty_family_counts", "must not be empty"),
        ("planner_supported", "supported must be boolean"),
        ("planner_reasons", "blocked_reasons must be a list"),
        ("planner_identity", "require non-empty"),
        ("registry_complete", "complete must be boolean"),
        ("registry_facet", "has_batching_rule must be boolean"),
    ],
)
def test_renderer_rejects_malformed_runtime_payloads(mutation: str, expected: str) -> None:
    """Malformed generated rows fail closed with field-specific diagnostics."""
    payload = build_differentiable_support_matrix_payload()
    registry = _mapping(payload, "registry")
    planner = _mapping(payload, "planner")
    if mutation == "schema":
        payload["schema"] = "foreign"
    elif mutation == "registry":
        payload["registry"] = "invalid"
    elif mutation == "planner_rows":
        planner["rows"] = "invalid"
    elif mutation == "family_counts":
        _mapping(registry, "family_counts")["array"] = 0
    elif mutation == "empty_family_counts":
        registry["family_counts"] = {}
    elif mutation == "planner_supported":
        _row(_list(planner, "rows")[0])["supported"] = "yes"
    elif mutation == "planner_reasons":
        _row(_list(planner, "rows")[0])["blocked_reasons"] = "invalid"
    elif mutation == "planner_identity":
        _row(_list(planner, "rows")[0])["gate"] = ""
    elif mutation == "registry_complete":
        _row(_list(registry, "rows")[0])["complete"] = "yes"
    else:
        _row(_list(registry, "rows")[0])["has_batching_rule"] = "yes"

    with pytest.raises(ValueError, match=expected):
        render_differentiable_support_matrix_page(payload)


def test_committed_page_is_current_and_manifest_aligned() -> None:
    """The checked-in page equals a fresh rendering from both live owners."""
    assert audit_differentiable_support_matrix_page() == ()
    assert DEFAULT_PAGE_PATH.read_text(encoding="utf-8") == (
        render_differentiable_support_matrix_page(build_differentiable_support_matrix_payload())
    )


def test_page_audit_reports_missing_stale_and_invalid_inputs(tmp_path: Path) -> None:
    """The audit names missing/stale pages and unreadable manifest shapes."""
    missing_page = tmp_path / "missing.md"
    assert (
        "is missing"
        in audit_differentiable_support_matrix_page(
            page_path=missing_page,
        )[0]
    )
    stale_page = tmp_path / "stale.md"
    stale_page.write_text("stale\n", encoding="utf-8")
    assert "is stale" in audit_differentiable_support_matrix_page(page_path=stale_page)[0]
    invalid_manifest = tmp_path / "manifest.json"
    invalid_manifest.write_text("not-json\n", encoding="utf-8")
    assert audit_differentiable_support_matrix_page(
        page_path=stale_page,
        manifest_path=invalid_manifest,
    )


def test_cli_write_preview_and_check_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI writes, previews, validates, and detects stale output."""
    output = tmp_path / "support.md"
    arguments = ["--output", str(output), "--manifest", str(DEFAULT_CAPABILITY_MANIFEST_PATH)]

    assert main(["--write", *arguments]) == 0
    assert output.is_file()
    assert main(["--check", *arguments]) == 0
    assert "current" in capsys.readouterr().out
    output.write_text("stale\n", encoding="utf-8")
    assert main(["--check", *arguments]) == 1
    assert "is stale" in capsys.readouterr().err
    assert main(["--manifest", str(DEFAULT_CAPABILITY_MANIFEST_PATH)]) == 0
    assert "# Differentiable Support Matrix" in capsys.readouterr().out


def test_cli_fails_closed_on_invalid_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write mode refuses a manifest that cannot prove surface alignment."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "foreign"}), encoding="utf-8")

    assert main(["--write", "--manifest", str(manifest)]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_script_entry_point_executes_the_real_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct script execution runs the same committed-page audit."""
    monkeypatch.setattr(sys, "argv", [str(TOOL_PATH), "--check"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(TOOL_PATH), run_name="__main__")

    assert exc_info.value.code == 0
