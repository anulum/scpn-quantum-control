# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for metamorphic AD verification
"""Real-surface tests for ``scpn_quantum_control.metamorphic_ad_verification``."""

from __future__ import annotations

import pytest

import scpn_quantum_control.metamorphic_ad_verification as metamorphic_ad_verification
from scpn_quantum_control.metamorphic_ad_verification import (
    METAMORPHIC_AD_CLAIM_BOUNDARY,
    METAMORPHIC_AD_VERIFICATION_SCHEMA,
    MetamorphicCheckResult,
    MetamorphicLawRecord,
    assert_metamorphic_registry_integrity,
    build_metamorphic_ad_registry,
    evaluate_chain_rule_residual,
    evaluate_linearity_residual,
    get_metamorphic_law,
    iter_metamorphic_laws,
    list_metamorphic_law_ids,
    probe_metamorphic_law,
)


def test_list_ids_stable() -> None:
    """Expose unique law ids in stable canonical order."""
    ids = list_metamorphic_law_ids()
    assert ids
    assert len(ids) == len(set(ids))
    assert ids == list_metamorphic_law_ids()
    assert "law:metamorphic.linearity" in ids


def test_get_executable_and_boundary_laws() -> None:
    """Resolve executable, evidence-gated, and permanent-boundary laws."""
    lin = get_metamorphic_law("law:metamorphic.linearity")
    assert lin.expected_outcome == "executable_local"
    assert not lin.reason
    assert lin.claim_boundary == METAMORPHIC_AD_CLAIM_BOUNDARY

    hw = get_metamorphic_law("law:formal.hardware_interactive_proof")
    assert hw.expected_outcome == "refuse_invent_green"
    assert hw.reason
    assert "hardware" in hw.reason.lower()


def test_get_rejects_blank_and_unknown() -> None:
    """Reject blank and unknown law identifiers fail closed."""
    with pytest.raises(ValueError, match="non-empty"):
        get_metamorphic_law("  ")
    with pytest.raises(ValueError, match="unknown metamorphic law_id"):
        get_metamorphic_law("law:invent.green")


def test_iter_filters() -> None:
    """Return the full catalogue or deterministic outcome subsets."""
    all_rows = iter_metamorphic_laws()
    assert len(all_rows) == len(list_metamorphic_law_ids())
    anti = iter_metamorphic_laws(kind="anti_silent_wrong")
    assert anti
    assert all(row.kind == "anti_silent_wrong" for row in anti)
    permanent = iter_metamorphic_laws(expected_outcome="permanent_boundary")
    assert permanent
    assert all(row.expected_outcome == "permanent_boundary" for row in permanent)


def test_build_registry_zero_blanks() -> None:
    """Build a schema-tagged registry with complete outcome counts."""
    registry = build_metamorphic_ad_registry()
    assert registry["schema"] == METAMORPHIC_AD_VERIFICATION_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["law_count"] == len(registry["laws"])  # type: ignore[arg-type]
    validated = assert_metamorphic_registry_integrity(registry)
    assert validated["blank_entry_count"] == 0


def test_probe_known_paths() -> None:
    """Probe executable, gated, and refused catalogue paths."""
    local = probe_metamorphic_law("law:metamorphic.linearity")
    assert local.passed is True
    assert local.refused is False

    gated = probe_metamorphic_law("law:metamorphic.grad_vmap_composition")
    assert gated.passed is False
    assert gated.refused is False
    assert "evidence_gated" in gated.message

    boundary = probe_metamorphic_law("law:anti_silent.di_jl_compiled_tape")
    assert boundary.refused is True
    assert boundary.passed is False

    invent = probe_metamorphic_law("law:formal.hardware_interactive_proof")
    assert invent.refused is True
    assert "refuse_invent_green" in invent.message or "invent" in invent.message.lower()


def test_probe_unknown_policies() -> None:
    """Raise or refuse unknown laws according to explicit policy."""
    with pytest.raises(ValueError, match="unknown metamorphic law_id"):
        probe_metamorphic_law("law:missing")
    refused = probe_metamorphic_law("law:missing", unknown_policy="refuse")
    assert refused.refused is True
    assert refused.passed is False
    with pytest.raises(ValueError, match="unknown_policy"):
        probe_metamorphic_law(
            "law:missing",
            unknown_policy="invent",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty"):
        probe_metamorphic_law("")


def test_evaluate_linearity_pass_and_fail() -> None:
    """Evaluate passing and failing additive-linearity residuals."""
    ok = evaluate_linearity_residual(1.0, 2.0, 3.0)
    assert ok.passed is True
    assert ok.residual == 0.0
    bad = evaluate_linearity_residual(1.0, 2.0, 4.0, tolerance=1e-9)
    assert bad.passed is False
    assert bad.residual == pytest.approx(1.0)


def test_evaluate_chain_rule_pass_and_fail() -> None:
    """Evaluate passing and failing scalar chain-rule residuals."""
    # g'(f)=2, f'=3 => (g∘f)'=6
    ok = evaluate_chain_rule_residual(2.0, 3.0, 6.0)
    assert ok.passed is True
    bad = evaluate_chain_rule_residual(2.0, 3.0, 5.0, tolerance=1e-12)
    assert bad.passed is False


def test_evaluate_rejects_wrong_law_and_bad_values() -> None:
    """Reject mismatched laws, non-finite inputs, and invalid bands."""
    with pytest.raises(ValueError, match="linearity"):
        evaluate_linearity_residual(1.0, 1.0, 2.0, law_id="law:metamorphic.chain_rule_scalar")
    with pytest.raises(ValueError, match="chain_rule"):
        evaluate_chain_rule_residual(1.0, 1.0, 1.0, law_id="law:metamorphic.linearity")
    with pytest.raises(ValueError, match="finite"):
        evaluate_linearity_residual(float("nan"), 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_chain_rule_residual(1.0, float("inf"), 1.0)
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_linearity_residual(1.0, 1.0, 2.0, tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_chain_rule_residual(1.0, 1.0, 1.0, tolerance=-1.0)


def test_record_validation() -> None:
    """Enforce law-record identifiers, outcomes, evidence, and reasons."""
    with pytest.raises(ValueError, match="law_id"):
        MetamorphicLawRecord(
            law_id="",
            kind="metamorphic_identity",
            expected_outcome="executable_local",
            relation="r",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="unknown law kind"):
        MetamorphicLawRecord(
            law_id="l",
            kind="nope",  # type: ignore[arg-type]
            expected_outcome="executable_local",
            relation="r",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="unknown expected_outcome"):
        MetamorphicLawRecord(
            law_id="l",
            kind="metamorphic_identity",
            expected_outcome="green",  # type: ignore[arg-type]
            relation="r",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="relation"):
        MetamorphicLawRecord(
            law_id="l",
            kind="metamorphic_identity",
            expected_outcome="executable_local",
            relation="  ",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="evidence_modules"):
        MetamorphicLawRecord(
            law_id="l",
            kind="metamorphic_identity",
            expected_outcome="executable_local",
            relation="r",
            evidence_modules=("",),
        )
    with pytest.raises(ValueError, match="default_tolerance"):
        MetamorphicLawRecord(
            law_id="l",
            kind="metamorphic_identity",
            expected_outcome="executable_local",
            relation="r",
            evidence_modules=("m",),
            default_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="must not carry"):
        MetamorphicLawRecord(
            law_id="l",
            kind="metamorphic_identity",
            expected_outcome="executable_local",
            relation="r",
            evidence_modules=("m",),
            reason="nope",
        )
    with pytest.raises(ValueError, match="require a non-empty reason"):
        MetamorphicLawRecord(
            law_id="l",
            kind="formal_boundary",
            expected_outcome="refuse_invent_green",
            relation="r",
            evidence_modules=("m",),
            reason="",
        )


def test_check_result_validation() -> None:
    """Enforce check-result residual, refusal, and message invariants."""
    with pytest.raises(ValueError, match="law_id"):
        MetamorphicCheckResult(
            law_id="",
            passed=True,
            residual=None,
            tolerance=None,
            message="m",
        )
    with pytest.raises(ValueError, match="message"):
        MetamorphicCheckResult(
            law_id="l",
            passed=True,
            residual=None,
            tolerance=None,
            message="  ",
        )
    with pytest.raises(ValueError, match="cannot be marked passed"):
        MetamorphicCheckResult(
            law_id="l",
            passed=True,
            residual=None,
            tolerance=None,
            message="m",
            refused=True,
        )
    with pytest.raises(ValueError, match="residual"):
        MetamorphicCheckResult(
            law_id="l",
            passed=False,
            residual=-1.0,
            tolerance=1.0,
            message="m",
        )
    with pytest.raises(ValueError, match="tolerance"):
        MetamorphicCheckResult(
            law_id="l",
            passed=False,
            residual=0.0,
            tolerance=0.0,
            message="m",
        )


def test_assert_integrity_rejects_invalid() -> None:
    """Reject malformed registries, blanks, missing reasons, and drift."""
    with pytest.raises(ValueError, match="non-empty laws"):
        assert_metamorphic_registry_integrity({"laws": []})
    with pytest.raises(ValueError, match="blank"):
        assert_metamorphic_registry_integrity(
            {
                "laws": [{"law_id": "", "expected_outcome": "executable_local"}],
                "blank_entry_count": 0,
                "law_count": 1,
            }
        )
    with pytest.raises(ValueError, match="without reason"):
        assert_metamorphic_registry_integrity(
            {
                "laws": [
                    {
                        "law_id": "l",
                        "expected_outcome": "permanent_boundary",
                        "reason": "",
                    }
                ],
                "blank_entry_count": 0,
                "law_count": 1,
            }
        )
    good = get_metamorphic_law("law:metamorphic.linearity").to_dict()
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_metamorphic_registry_integrity(
            {"laws": [good], "blank_entry_count": 1, "law_count": 1}
        )
    with pytest.raises(ValueError, match="law_count"):
        assert_metamorphic_registry_integrity(
            {"laws": [good], "blank_entry_count": 0, "law_count": 99}
        )
    with pytest.raises(ValueError, match="mapping"):
        assert_metamorphic_registry_integrity(
            {"laws": ["x"], "blank_entry_count": 0, "law_count": 1}
        )
    with pytest.raises(ValueError, match="blank"):
        assert_metamorphic_registry_integrity(
            {
                "laws": [
                    {
                        "law_id": "l",
                        "expected_outcome": "not-an-outcome",
                        "reason": "r",
                    }
                ],
                "blank_entry_count": 0,
                "law_count": 1,
            }
        )


def test_catalogue_map_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when canonical law identifiers are duplicated."""
    row = get_metamorphic_law("law:metamorphic.linearity")
    monkeypatch.setattr(metamorphic_ad_verification, "_CANONICAL_LAWS", (row, row))
    with pytest.raises(RuntimeError, match="duplicate law_id"):
        metamorphic_ad_verification._catalogue_map()


def test_to_dict_round_trips() -> None:
    """Serialise law records and check results to JSON-ready maps."""
    row = get_metamorphic_law("law:fd.agreement_band.parameter_shift")
    payload = row.to_dict()
    assert payload["law_id"] == row.law_id
    assert isinstance(payload["evidence_modules"], list)
    result = evaluate_linearity_residual(0.0, 0.0, 0.0)
    assert result.to_dict()["passed"] is True
