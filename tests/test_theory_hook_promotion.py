# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-98 theory-hook promotion tests
"""Tests for evidence-gated theory-hook promotion and custody artifacts."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import scpn_quantum_control.analysis.theory_hook_promotion as promotion
from scpn_quantum_control.analysis import (
    THEORY_HOOK_PROMOTION_BOUNDARY,
    THEORY_HOOK_PROMOTION_SCHEMA,
    TheoryHookEvidenceRecord,
    TheoryHookPromotionRecord,
    TheoryHookPromotionReport,
    TheoryHookRole,
    TheoryHookStatus,
    TheoryHookTier,
    build_theory_hook_promotion_report,
    get_theory_hook_promotion,
    list_theory_hook_promotions,
    render_theory_hook_promotion_markdown,
    run_theory_hook_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample_record() -> TheoryHookPromotionRecord:
    """Return a minimal internally valid policy record."""
    return TheoryHookPromotionRecord(
        hook_id="sample",
        title="Sample hook",
        module="package.sample",
        tier=TheoryHookTier.BOUNDED,
        role=TheoryHookRole.SPECTRAL_DIAGNOSTIC,
        status=TheoryHookStatus.DIAGNOSTIC_ONLY,
        differentiable=False,
        evidence_fixture="small fixture",
        allowed_claims=("bounded diagnostic",),
        forbidden_claims=("production claim",),
        promotion_requirements=("held-out evidence",),
        references=("doi:10.0000/example",),
    )


def _sample_evidence(*, passed: bool = True) -> TheoryHookEvidenceRecord:
    """Return a minimal evidence record whose aggregate matches its check."""
    return TheoryHookEvidenceRecord(
        hook_id="sample",
        passed=passed,
        fixture="small fixture",
        checks=(("invariant", passed),),
        metrics=(("value", 1.0),),
    )


def _load_runner() -> ModuleType:
    """Load the evidence CLI as a module without mutating import paths."""
    path = ROOT / "scripts" / "run_theory_hook_promotion_evidence.py"
    spec = importlib.util.spec_from_file_location("theory_hook_evidence_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_has_one_fail_closed_record_per_reviewed_hook() -> None:
    """The canonical registry covers all six BL-98 theory families."""
    records = list_theory_hook_promotions()

    assert tuple(record.hook_id for record in records) == (
        "quantum_speed_limit",
        "hamiltonian_learning",
        "koopman_local_closure",
        "bipartite_mutual_information",
        "stabilizer_renyi_entropy",
        "spectral_form_factor",
    )
    assert all(not record.differentiable for record in records)
    assert all(not record.admitted_for_control for record in records)
    assert all(not record.admitted_for_publication_claim for record in records)
    assert all(record.allowed_claims for record in records)
    assert all(record.forbidden_claims for record in records)
    assert get_theory_hook_promotion("quantum_speed_limit") is records[0]


def test_phi_policy_is_tier_d_and_forbids_consciousness_claims() -> None:
    """Minimum QMI cannot silently promote into IIT or consciousness language."""
    record = get_theory_hook_promotion("bipartite_mutual_information")
    forbidden = " ".join(record.forbidden_claims).lower()

    assert record.tier is TheoryHookTier.RESEARCH_ONLY
    assert record.status is TheoryHookStatus.RESEARCH_ONLY
    assert record.role is TheoryHookRole.MUTUAL_INFORMATION_DIAGNOSTIC
    assert "consciousness" in forbidden
    assert "integrated information theory" in forbidden


def test_koopman_policy_names_local_closure_and_blocks_bqp_claim() -> None:
    """The finite closure stays a classical local diagnostic."""
    record = get_theory_hook_promotion("koopman_local_closure")

    assert record.role is TheoryHookRole.CLASSICAL_LOCAL_BASELINE
    assert "BQP-completeness" in record.forbidden_claims
    assert "full nonlinear dynamics" in record.forbidden_claims


def test_unknown_hook_fails_closed() -> None:
    """Registry lookups never synthesize a permissive default."""
    with pytest.raises(KeyError, match="unknown theory hook"):
        get_theory_hook_promotion("missing")


@pytest.mark.parametrize("field", ["hook_id", "title", "module", "evidence_fixture"])
def test_promotion_record_rejects_blank_identity_fields(field: str) -> None:
    """Every displayed identity field must be non-empty."""
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        replace(_sample_record(), **{field: " "})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    ["allowed_claims", "forbidden_claims", "promotion_requirements", "references"],
)
def test_promotion_record_rejects_empty_policy_sequences(field: str) -> None:
    """No policy record may omit positive, negative, or evidentiary context."""
    with pytest.raises(ValueError, match=f"{field} must contain non-empty strings"):
        replace(_sample_record(), **{field: ()})  # type: ignore[arg-type]


def test_promotion_record_rejects_blank_and_duplicate_policy_entries() -> None:
    """Policy sequences remain display-safe and unambiguous."""
    with pytest.raises(ValueError, match="allowed_claims must contain non-empty strings"):
        replace(_sample_record(), allowed_claims=(" ",))
    with pytest.raises(ValueError, match="forbidden_claims must not contain duplicates"):
        replace(_sample_record(), forbidden_claims=("blocked", "blocked"))


def test_promotion_record_rejects_unsupported_derivative_and_tier_combinations() -> None:
    """BL-98 cannot imply differentiability or promote a tier-D hook."""
    with pytest.raises(ValueError, match="no admitted differentiable contract"):
        replace(_sample_record(), differentiable=True)
    with pytest.raises(ValueError, match="tier-D hooks must remain research_only"):
        replace(_sample_record(), tier=TheoryHookTier.RESEARCH_ONLY)


def test_promotion_record_serialization_keeps_negative_capabilities() -> None:
    """Machine-readable policy includes both control and publication denials."""
    payload = _sample_record().as_dict()

    assert payload["tier"] == "B"
    assert payload["role"] == "spectral_diagnostic"
    assert payload["status"] == "diagnostic_only"
    assert payload["admitted_for_control"] is False
    assert payload["admitted_for_publication_claim"] is False


def test_evidence_record_validates_identity_checks_and_uniqueness() -> None:
    """Evidence rows cannot carry blank identities or ambiguous maps."""
    with pytest.raises(ValueError, match="hook_id and fixture must be non-empty"):
        replace(_sample_evidence(), hook_id="")
    with pytest.raises(ValueError, match="hook_id and fixture must be non-empty"):
        replace(_sample_evidence(), fixture="")
    with pytest.raises(ValueError, match="checks must not be empty"):
        replace(_sample_evidence(), checks=())
    with pytest.raises(ValueError, match="check and metric names must be non-empty"):
        replace(_sample_evidence(), checks=(("", True),))
    with pytest.raises(ValueError, match="check and metric names must be non-empty"):
        replace(_sample_evidence(), metrics=(("", 1),))
    with pytest.raises(ValueError, match="check names must be unique"):
        replace(_sample_evidence(), checks=(("same", True), ("same", True)))
    with pytest.raises(ValueError, match="metric names must be unique"):
        replace(_sample_evidence(), metrics=(("same", 1), ("same", 2)))
    with pytest.raises(ValueError, match="passed must equal"):
        replace(_sample_evidence(), passed=False)


def test_evidence_record_serialization_preserves_named_maps() -> None:
    """Tuple storage becomes simple deterministic JSON maps."""
    assert _sample_evidence().as_dict() == {
        "hook_id": "sample",
        "passed": True,
        "fixture": "small fixture",
        "checks": {"invariant": True},
        "metrics": {"value": 1.0},
    }


def test_evidence_record_rejects_numpy_scalar_metrics() -> None:
    """Evidence custody requires JSON-native scalars at construction time."""
    with pytest.raises(ValueError, match="JSON-native"):
        replace(_sample_evidence(), metrics=(("value", np.int64(5)),))


def test_all_real_local_fixtures_pass_without_granting_promotion() -> None:
    """Execute QSL, inverse, closure, QMI, magic, and SFF probes end to end."""
    evidence = run_theory_hook_evidence()

    assert len(evidence) == 6
    assert all(item.passed for item in evidence)
    assert dict(evidence[3].metrics) == {
        "mutual_information_bits": 2.0,
        "iit_phi": "not_computed",
    }
    assert dict(evidence[4].checks)["t_state_positive"] is True
    assert dict(evidence[5].metrics)["sff_at_zero"] == 1.0


def test_report_digest_and_markdown_are_stable_and_explicit() -> None:
    """The aggregate report is complete, digest-locked, and non-promotional."""
    report = build_theory_hook_promotion_report()
    payload = report.as_dict()
    markdown = render_theory_hook_promotion_markdown(report)

    assert report.passed
    assert report.schema == THEORY_HOOK_PROMOTION_SCHEMA
    assert report.claim_boundary == THEORY_HOOK_PROMOTION_BOUNDARY
    assert len(report.content_digest) == 64
    assert payload["passed"] is True
    assert len(payload["records"]) == len(payload["evidence"]) == 6
    assert markdown.endswith("\n")
    assert report.content_digest in markdown
    assert "All local fixtures passed: **true**" in markdown
    assert "bounded local synthetic evidence only" in markdown


def test_report_passed_property_tracks_failed_evidence() -> None:
    """A single failed row makes the aggregate report fail closed."""
    report = TheoryHookPromotionReport(
        schema="test.v1",
        claim_boundary="test only",
        records=(_sample_record(),),
        evidence=(_sample_evidence(passed=False),),
        content_digest="0" * 64,
    )

    assert not report.passed
    assert report.as_dict()["passed"] is False


def test_evidence_order_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registry/evidence drift cannot emit a mislabelled report."""
    identifiers = tuple(record.hook_id for record in promotion._RECORDS)

    def evidence_for(hook_id: str) -> TheoryHookEvidenceRecord:
        return TheoryHookEvidenceRecord(
            hook_id=hook_id,
            passed=True,
            fixture="fixture",
            checks=(("ok", True),),
            metrics=(),
        )

    monkeypatch.setattr(promotion, "_qsl_evidence", lambda: evidence_for(identifiers[0]))
    monkeypatch.setattr(
        promotion, "_hamiltonian_learning_evidence", lambda: evidence_for(identifiers[1])
    )
    monkeypatch.setattr(promotion, "_koopman_evidence", lambda: evidence_for(identifiers[2]))
    monkeypatch.setattr(
        promotion, "_mutual_information_evidence", lambda: evidence_for(identifiers[3])
    )
    monkeypatch.setattr(promotion, "_magic_evidence", lambda: evidence_for(identifiers[4]))
    monkeypatch.setattr(promotion, "_spectral_evidence", lambda: evidence_for(identifiers[5]))
    monkeypatch.setattr(promotion, "_RECORDS", tuple(reversed(promotion._RECORDS)))

    with pytest.raises(RuntimeError, match="evidence order"):
        promotion.run_theory_hook_evidence()


def test_evidence_runner_writes_checks_and_rejects_stale_bytes(tmp_path: Path) -> None:
    """The CLI writes canonical evidence and fails closed on custody drift."""
    runner = _load_runner()
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    args = ["--json", str(json_path), "--markdown", str(markdown_path)]

    assert runner.main(args) == 0
    assert runner.main([*args, "--check"]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == THEORY_HOOK_PROMOTION_SCHEMA
    assert payload["passed"] is True

    json_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale evidence file"):
        runner.main([*args, "--check"])

    json_path.unlink()
    with pytest.raises(SystemExit, match="missing evidence file"):
        runner.main([*args, "--check"])
