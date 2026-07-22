# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable competitive baseline tests
"""Tests for differentiable competitive-baseline refresh governance."""

from __future__ import annotations

import copy
import json
import subprocess  # noqa: S404
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control import (
    audit_competitive_baseline_promotion_gate,
    differentiable_api,
    load_competitive_baseline_refresh,
    render_competitive_baseline_refresh_markdown,
    run_competitive_baseline_refresh,
)
from scpn_quantum_control.differentiable_baseline_scorecard import REQUIRED_BASELINE_CATEGORIES
from scpn_quantum_control.differentiable_competitive_baselines import (
    MAX_BASELINE_AGE_DAYS,
    REQUIRED_BASELINE_IDS,
    CompetitiveBaselineRefresh,
    validate_competitive_baseline_refresh,
)

ROOT = Path(__file__).resolve().parent.parent


def _write_refresh_payload(tmp_path: Path, payload: object) -> Path:
    """Write a JSON payload for the production refresh loader."""
    path = tmp_path / "competitive-baseline-refresh.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_competitive_baseline_refresh_records_required_sources() -> None:
    """The baseline refresh artifact must cover all required competitor sources."""
    refresh = run_competitive_baseline_refresh()
    validation = validate_competitive_baseline_refresh(refresh, as_of=date(2026, 6, 27))

    assert refresh.schema == "scpn_qc_differentiable_competitive_baseline_refresh_v1"
    assert refresh.artifact_id == "diff-competitive-baseline-refresh-20260627"
    assert refresh.max_age_days == MAX_BASELINE_AGE_DAYS
    assert {row.baseline_id for row in refresh.rows} == set(REQUIRED_BASELINE_IDS)
    assert set(validation.checked_categories) == set(REQUIRED_BASELINE_CATEGORIES)
    assert validation.passed
    assert validation.errors == ()
    assert all(row.source_url.startswith("https://") for row in refresh.rows)
    assert validation.to_dict()["passed"] is True
    assert refresh.rows[0].age_days(as_of=date(2026, 6, 28)) == 1


def test_committed_competitive_baseline_refresh_matches_current_builder() -> None:
    """The committed JSON refresh must match the deterministic builder."""
    committed = load_competitive_baseline_refresh()
    generated = run_competitive_baseline_refresh()

    assert committed.to_dict() == generated.to_dict()
    assert validate_competitive_baseline_refresh(
        committed,
        as_of=date(2026, 6, 27),
    ).passed


def test_competitive_baseline_refresh_rejects_stale_rows() -> None:
    """Baseline validation must fail once the freshness window expires."""
    refresh = run_competitive_baseline_refresh()

    validation = validate_competitive_baseline_refresh(
        refresh,
        as_of=date(2026, 6, 27) + timedelta(days=MAX_BASELINE_AGE_DAYS + 1),
    )

    assert not validation.passed
    assert any("baseline source is stale" in error for error in validation.errors)


def test_competitive_baseline_refresh_rejects_missing_baseline_and_category() -> None:
    """Baseline validation must reject incomplete competitor/category coverage."""
    refresh = run_competitive_baseline_refresh()
    incomplete = CompetitiveBaselineRefresh(
        schema=refresh.schema,
        artifact_id=refresh.artifact_id,
        generated_on=refresh.generated_on,
        max_age_days=refresh.max_age_days,
        rows=refresh.rows[1:],
        claim_boundary=refresh.claim_boundary,
    )

    validation = validate_competitive_baseline_refresh(
        incomplete,
        as_of=date(2026, 6, 27),
    )

    assert not validation.passed
    assert "missing competitive baseline rows: jax" in validation.errors
    assert any("jax_native_transforms" in error for error in validation.errors)


def test_competitive_baseline_row_rejects_invalid_source() -> None:
    """Rows must fail closed on non-official or non-HTTPS source metadata."""
    row = run_competitive_baseline_refresh().rows[0]

    for source_url in (
        "http://docs.jax.dev/",
        "https://",
        "https://user:secret@docs.jax.dev/",
        "https://docs.jax.dev:notaport/",
        "https://docs.jax.dev:0/",
        "https://docs.jax.dev/a bad path",
    ):
        with pytest.raises(ValueError, match="absolute credential-free HTTPS URL"):
            replace(row, source_url=source_url)

    with pytest.raises(ValueError, match="source_url must be a non-empty string"):
        replace(row, source_url=cast(Any, 7))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"baseline_id": "unknown"}, "unknown competitive baseline"),
        ({"source_kind": "blog"}, "unknown baseline source kind"),
        ({"checked_on": datetime(2026, 6, 27)}, "checked_on must be a date"),
        ({"refresh_due_on": "2026-08-11"}, "refresh_due_on must be a date"),
        ({"max_age_days": True}, "max_age_days must be a positive integer"),
        ({"max_age_days": 0}, "max_age_days must be a positive integer"),
        ({"refresh_due_on": date(2026, 8, 12)}, "refresh_due_on must match"),
        ({"display_name": 12}, "display_name must be a non-empty string"),
        ({"upstream_version": "  "}, "upstream_version must be a non-empty string"),
        ({"scorecard_categories": []}, "scorecard_categories must contain non-empty strings"),
        ({"scorecard_categories": ()}, "scorecard_categories must contain at least one value"),
        (
            {"scorecard_categories": ("jax_native_transforms", "")},
            "scorecard_categories must contain non-empty strings",
        ),
        (
            {"scorecard_categories": ("jax_native_transforms", "jax_native_transforms")},
            "scorecard_categories must contain unique values",
        ),
        ({"scorecard_categories": ("unknown",)}, "unknown baseline categories"),
        ({"required_capabilities": ("grad", 1)}, "must contain non-empty strings"),
        (
            {"hardening_implications": ("same", "same")},
            "hardening_implications must contain unique values",
        ),
        ({"claim_boundary": "no promotion"}, "claim_boundary is not canonical"),
    ),
)
def test_competitive_baseline_row_rejects_ambiguous_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    """Row construction must reject coercion, duplicates, and boundary drift."""
    row = run_competitive_baseline_refresh().rows[0]

    with pytest.raises(ValueError, match=message):
        replace(row, **changes)


def test_competitive_baseline_age_rejects_datetime_subclass() -> None:
    """Freshness arithmetic must accept exact dates rather than datetime subclasses."""
    row = run_competitive_baseline_refresh().rows[0]

    with pytest.raises(ValueError, match="as_of must be a date"):
        row.age_days(as_of=cast(Any, datetime(2026, 6, 28)))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema": "unknown"}, "schema is not canonical"),
        ({"artifact_id": "unknown"}, "artifact_id is not canonical"),
        ({"generated_on": datetime(2026, 6, 27)}, "generated_on must be a date"),
        ({"max_age_days": True}, "max_age_days must be a positive integer"),
        ({"max_age_days": 44}, "max_age_days is not canonical"),
        ({"rows": []}, "rows must be a non-empty row tuple"),
        ({"rows": ()}, "rows must be a non-empty row tuple"),
        ({"rows": ("not-a-row",)}, "rows must be a non-empty row tuple"),
        ({"claim_boundary": "fresh"}, "claim_boundary is not canonical"),
    ),
)
def test_competitive_baseline_refresh_rejects_structural_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    """Bundle construction must preserve canonical identity and exact runtime types."""
    refresh = run_competitive_baseline_refresh()

    with pytest.raises(ValueError, match=message):
        replace(refresh, **changes)


def test_competitive_baseline_refresh_rejects_conflicting_rows() -> None:
    """Bundle rows must have unique identities and matching refresh metadata."""
    refresh = run_competitive_baseline_refresh()
    first = refresh.rows[0]
    wrong_day = run_competitive_baseline_refresh(
        generated_on=refresh.generated_on + timedelta(days=1)
    ).rows[0]
    wrong_age = replace(
        first,
        max_age_days=MAX_BASELINE_AGE_DAYS - 1,
        refresh_due_on=first.checked_on + timedelta(days=MAX_BASELINE_AGE_DAYS - 1),
    )

    with pytest.raises(ValueError, match="identities must contain unique values"):
        replace(refresh, rows=(first, first))
    with pytest.raises(ValueError, match="rows must preserve canonical order"):
        replace(refresh, rows=tuple(reversed(refresh.rows)))
    with pytest.raises(ValueError, match="checked_on must match generated_on"):
        replace(refresh, rows=(wrong_day,))
    with pytest.raises(ValueError, match="max_age_days must match bundle"):
        replace(refresh, rows=(wrong_age,))


def test_competitive_baseline_validation_rejects_future_rows() -> None:
    """Semantic validation must report future-dated rows as non-fresh evidence."""
    refresh = run_competitive_baseline_refresh()
    validation = validate_competitive_baseline_refresh(refresh, as_of=date(2026, 6, 26))

    assert not validation.passed
    assert any("checked_on is in the future" in error for error in validation.errors)
    assert any("baseline source is stale" in error for error in validation.errors)

    with pytest.raises(ValueError, match="validation as_of must be a date"):
        validate_competitive_baseline_refresh(
            refresh,
            as_of=cast(Any, datetime(2026, 6, 27)),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"passed": 1}, "passed must be boolean"),
        ({"passed": False}, "passed must be true exactly when errors are empty"),
        ({"errors": ["error"]}, "errors must contain non-empty strings"),
        ({"errors": ("",)}, "errors must contain non-empty strings"),
        ({"checked_baselines": ["jax"]}, "checked_baselines must contain non-empty strings"),
        (
            {"checked_baselines": ("jax", "jax")},
            "checked_baselines must contain unique values",
        ),
        ({"checked_baselines": ("unknown",)}, "unknown baseline identity"),
        ({"checked_categories": ("",)}, "checked_categories must contain non-empty strings"),
        (
            {"checked_categories": ("jax_native_transforms", "jax_native_transforms")},
            "checked_categories must contain unique values",
        ),
        ({"checked_categories": ("unknown",)}, "contains unknown category"),
        ({"checked_urls": ("",)}, "checked_urls must contain non-empty strings"),
        (
            {"checked_urls": ("https://example.com", "https://example.com")},
            "checked_urls must contain unique values",
        ),
        ({"checked_urls": ("http://example.com",)}, "absolute credential-free HTTPS URL"),
        ({"as_of": datetime(2026, 6, 27)}, "as_of must be a date"),
        ({"claim_boundary": "validated"}, "claim_boundary is not canonical"),
    ),
)
def test_competitive_baseline_validation_result_rejects_incoherence(
    changes: dict[str, object],
    message: str,
) -> None:
    """Validation result objects must not admit ambiguous success evidence."""
    validation = validate_competitive_baseline_refresh(
        run_competitive_baseline_refresh(),
        as_of=date(2026, 6, 27),
    )

    with pytest.raises(ValueError, match=message):
        replace(validation, **changes)


def test_competitive_baseline_promotion_gate_combines_language_and_freshness() -> None:
    """The combined gate must reject unbacked promotional wording and require freshness."""
    audit = audit_competitive_baseline_promotion_gate(
        refresh=run_competitive_baseline_refresh(),
        as_of=date(2026, 6, 27),
        public_texts={
            "README.md": ("The differentiable stack is world-leading for PyTorch autograd.")
        },
    )

    assert not audit.passed
    assert audit.baseline_validation.passed
    assert audit.language_audit.checked_promotional_categories == ("pytorch_autograd_compile",)
    assert any("pytorch_autograd_compile" in error for error in audit.errors)
    assert audit.to_dict()["passed"] is False


def test_competitive_baseline_promotion_gate_requires_category_baseline() -> None:
    """Public promotion wording must have a fresh baseline for its exact category."""
    refresh = run_competitive_baseline_refresh()
    without_pytorch_category = replace(
        refresh,
        rows=tuple(
            row
            for row in refresh.rows
            if "pytorch_autograd_compile" not in row.scorecard_categories
        ),
    )

    audit = audit_competitive_baseline_promotion_gate(
        refresh=without_pytorch_category,
        as_of=date(2026, 6, 27),
        public_texts={"README.md": "The stack is world-leading for PyTorch autograd."},
    )

    assert any(
        error == "pytorch_autograd_compile: public promotion wording lacks fresh baseline evidence"
        for error in audit.errors
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"passed": 1}, "passed must be boolean"),
        ({"passed": False}, "passed must be true exactly when errors are empty"),
        ({"errors": ["error"]}, "errors must contain non-empty strings"),
        ({"baseline_validation": "invalid"}, "must be CompetitiveBaselineValidation"),
        ({"language_audit": "invalid"}, "must be DifferentiablePromotionLanguageAudit"),
        ({"checked_paths": ["README.md"]}, "checked_paths must contain non-empty strings"),
        (
            {"checked_paths": ("README.md", "README.md")},
            "checked_paths must contain unique values",
        ),
        ({"checked_categories": ("",)}, "checked_categories must contain non-empty strings"),
        (
            {"checked_categories": ("benchmark_promotion", "benchmark_promotion")},
            "checked_categories must contain unique values",
        ),
        ({"checked_categories": ("unknown",)}, "contains unknown category"),
        (
            {"checked_categories": ("benchmark_promotion",)},
            "checked_categories are inconsistent",
        ),
        ({"checked_paths": ("OTHER.md",)}, "checked_paths are inconsistent"),
        ({"claim_boundary": "promoted"}, "claim_boundary is not canonical"),
    ),
)
def test_competitive_baseline_promotion_result_rejects_incoherence(
    changes: dict[str, object],
    message: str,
) -> None:
    """Combined-gate objects must preserve typed components and fail-closed state."""
    audit = audit_competitive_baseline_promotion_gate(
        refresh=run_competitive_baseline_refresh(),
        as_of=date(2026, 6, 27),
        public_texts={"README.md": "Bounded differentiable evidence."},
    )
    assert audit.passed

    with pytest.raises(ValueError, match=message):
        replace(audit, **changes)


def test_competitive_baseline_promotion_rejects_incoherent_components() -> None:
    """A combined result must not conceal failed or malformed component results."""
    audit = audit_competitive_baseline_promotion_gate(
        refresh=run_competitive_baseline_refresh(),
        as_of=date(2026, 6, 27),
        public_texts={"README.md": "Bounded differentiable evidence."},
    )
    failed_validation = replace(
        audit.baseline_validation,
        passed=False,
        errors=("failed",),
    )

    with pytest.raises(ValueError, match="passed must equal component results"):
        replace(audit, baseline_validation=failed_validation)

    malformed_language_audit = replace(audit.language_audit, passed=cast(Any, 1))
    with pytest.raises(ValueError, match="language audit passed must be boolean"):
        replace(audit, language_audit=malformed_language_audit)

    incoherent_language_audit = replace(audit.language_audit, passed=False)
    with pytest.raises(ValueError, match="true exactly when errors are empty"):
        replace(audit, language_audit=incoherent_language_audit)

    malformed_language_errors = replace(audit.language_audit, errors=cast(Any, ["error"]))
    with pytest.raises(ValueError, match="language audit errors must contain non-empty strings"):
        replace(audit, language_audit=malformed_language_errors)


def test_competitive_baseline_loader_rejects_schema_smuggling(tmp_path: Path) -> None:
    """The JSON boundary must reject duplicate, missing, and unknown object members."""
    payload = run_competitive_baseline_refresh().to_dict()
    missing = copy.deepcopy(payload)
    cast(dict[str, object], missing).pop("schema")
    with pytest.raises(ValueError, match=r"missing: schema; unexpected: none"):
        load_competitive_baseline_refresh(_write_refresh_payload(tmp_path, missing))

    unexpected = copy.deepcopy(payload)
    cast(dict[str, object], unexpected)["promoted"] = True
    with pytest.raises(ValueError, match=r"missing: none; unexpected: promoted"):
        load_competitive_baseline_refresh(_write_refresh_payload(tmp_path, unexpected))

    row_extra = copy.deepcopy(payload)
    rows = cast(list[dict[str, object]], cast(dict[str, object], row_extra)["rows"])
    rows[0]["unreviewed"] = "accepted"
    with pytest.raises(ValueError, match="unexpected: unreviewed"):
        load_competitive_baseline_refresh(_write_refresh_payload(tmp_path, row_extra))

    duplicate_path = tmp_path / "duplicate-key.json"
    duplicate_path.write_text('{"schema": "first", "schema": "second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key: schema"):
        load_competitive_baseline_refresh(duplicate_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("classification", "candidate", "classification is not canonical"),
        ("baseline_id", "other", "unknown competitive baseline"),
        ("source_kind", "blog", "unknown baseline source kind"),
        ("scorecard_categories", ["other"], "unknown baseline category"),
        ("display_name", 4, "display_name must be a non-empty string"),
        ("max_age_days", True, "max_age_days must be an integer"),
        ("checked_on", "not-a-date", "checked_on must be an ISO date"),
        ("required_capabilities", {}, "required_capabilities must be a JSON array"),
        ("required_capabilities", [], "required_capabilities must contain at least one value"),
        (
            "required_capabilities",
            ["grad", ""],
            "required_capabilities must contain non-empty strings",
        ),
    ),
)
def test_competitive_baseline_loader_rejects_coerced_row_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """The committed-artifact loader must validate every row field without coercion."""
    payload = run_competitive_baseline_refresh().to_dict()
    rows = cast(list[dict[str, object]], cast(dict[str, object], payload)["rows"])
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        load_competitive_baseline_refresh(_write_refresh_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("rows", {}, "rows must be a JSON array"),
        ("rows", [42], "competitive baseline row must be a JSON object"),
        ("schema", 7, "schema must be a non-empty string"),
        ("generated_on", "not-a-date", "generated_on must be an ISO date"),
        ("max_age_days", True, "max_age_days must be an integer"),
    ),
)
def test_competitive_baseline_loader_rejects_coerced_bundle_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """The JSON loader must reject scalar and container drift at the bundle boundary."""
    payload = cast(dict[str, object], run_competitive_baseline_refresh().to_dict())
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        load_competitive_baseline_refresh(_write_refresh_payload(tmp_path, payload))


def test_competitive_baseline_loader_rejects_non_object_root(tmp_path: Path) -> None:
    """A refresh artifact must have a JSON-object root."""
    with pytest.raises(ValueError, match="refresh must be a JSON object"):
        load_competitive_baseline_refresh(_write_refresh_payload(tmp_path, []))


def test_competitive_baseline_markdown_and_api_dispatch() -> None:
    """Baseline refresh evidence must render and dispatch through the unified API."""
    refresh = run_competitive_baseline_refresh()
    markdown = render_competitive_baseline_refresh_markdown(refresh)
    result = differentiable_api("competitive_baseline_refresh")

    assert "# Differentiable Competitive Baseline Refresh" in markdown
    assert "jax_native_transforms" in markdown
    assert "does not promote" in markdown
    assert result.operation == "competitive_baseline_refresh"
    assert result.supported is False
    assert result.payload["artifact_id"] == refresh.artifact_id
    assert "does not promote" in result.claim_boundary


def test_competitive_baseline_cli_gate_passes_on_committed_artifact() -> None:
    """The preflight CLI must validate the real committed baseline artifact."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "tools/check_differentiable_competitive_baselines.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "differentiable competitive-baseline gate: PASS" in completed.stdout
