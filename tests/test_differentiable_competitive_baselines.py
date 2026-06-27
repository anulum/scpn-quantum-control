# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable competitive baseline tests
"""Tests for differentiable competitive-baseline refresh governance."""

from __future__ import annotations

import subprocess  # noqa: S404
import sys
from datetime import date, timedelta
from pathlib import Path

from scpn_quantum_control import (
    CompetitiveBaselineRow,
    audit_competitive_baseline_promotion_gate,
    differentiable_api,
    load_competitive_baseline_refresh,
    render_competitive_baseline_refresh_markdown,
    run_competitive_baseline_refresh,
)
from scpn_quantum_control.differentiable_competitive_baselines import (
    MAX_BASELINE_AGE_DAYS,
    REQUIRED_BASELINE_IDS,
    CompetitiveBaselineRefresh,
    validate_competitive_baseline_refresh,
)
from scpn_quantum_control.differentiable_sota_scorecard import REQUIRED_SOTA_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent


def test_competitive_baseline_refresh_records_required_sources() -> None:
    """The baseline refresh artifact must cover all required competitor sources."""
    refresh = run_competitive_baseline_refresh()
    validation = validate_competitive_baseline_refresh(refresh, as_of=date(2026, 6, 27))

    assert refresh.schema == "scpn_qc_differentiable_competitive_baseline_refresh_v1"
    assert refresh.artifact_id == "diff-competitive-baseline-refresh-20260627"
    assert refresh.max_age_days == MAX_BASELINE_AGE_DAYS
    assert {row.baseline_id for row in refresh.rows} == set(REQUIRED_BASELINE_IDS)
    assert set(validation.checked_categories) == set(REQUIRED_SOTA_CATEGORIES)
    assert validation.passed
    assert validation.errors == ()
    assert all(row.source_url.startswith("https://") for row in refresh.rows)


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
    due = date(2026, 6, 27) + timedelta(days=MAX_BASELINE_AGE_DAYS)

    try:
        CompetitiveBaselineRow(
            baseline_id="jax",
            display_name="JAX",
            upstream_version="bad",
            source_url="http://docs.jax.dev/",
            source_kind="official_docs",
            checked_on=date(2026, 6, 27),
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("jax_native_transforms",),
            required_capabilities=("grad",),
            hardening_implications=("reject stale baselines",),
            claim_boundary="no promotion",
        )
    except ValueError as exc:
        assert str(exc) == "competitive baseline source_url must be an HTTPS URL"
    else:  # pragma: no cover - defensive failure path.
        raise AssertionError("invalid baseline row was accepted")


def test_competitive_baseline_promotion_gate_combines_language_and_freshness() -> None:
    """The combined gate must reject unbacked SOTA wording and require freshness."""
    audit = audit_competitive_baseline_promotion_gate(
        refresh=run_competitive_baseline_refresh(),
        as_of=date(2026, 6, 27),
        public_texts={
            "README.md": ("The differentiable stack is state-of-the-art for PyTorch autograd.")
        },
    )

    assert not audit.passed
    assert audit.baseline_validation.passed
    assert audit.language_audit.checked_promotional_categories == ("pytorch_autograd_compile",)
    assert any("pytorch_autograd_compile" in error for error in audit.errors)


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
