# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive validate handler tests
"""Tests for the claim-ledger reference-validation ``validate`` handler."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio.executive import (  # noqa: E402
    ActionRegistry,
    ExecutiveRequest,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from scpn_quantum_control.studio.executive_validate import (  # noqa: E402
    VALIDATE_VERB,
    ValidateActionHandler,
    _as_positive_int,
    _normalise_validate,
    _safe_slug,
)


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(ValidateActionHandler())
    return registry


def _request(*, backend: str | None = None, **parameters: Any) -> ExecutiveRequest:
    return ExecutiveRequest(
        verb=VALIDATE_VERB, action_id="validate-ledger", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_validate_committed_registry_against_committed_ledger() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["validation_schema"] == "studio.physics-validation.v1"
    assert outputs["validation_passed"] is True
    assert outputs["validation_errors"] == []
    assert outputs["certificate_count"] >= 0
    assert outputs["total_claims"] >= 1
    assert outputs["minimum_total_claims_met"] is True
    assert 0.0 <= outputs["answer_rate"] <= 1.0
    assert sum(outputs["grade_distribution"].values()) == outputs["total_claims"]
    assert record.script is not None


def test_validate_reports_unmet_minimum_claim_bound() -> None:
    record = run_action(_request(minimum_total_claims=4096), registry=_registry())
    outputs = record.result.outputs
    assert outputs["minimum_total_claims"] == 4096
    assert outputs["minimum_total_claims_met"] is False


def test_validate_plan_defaults_backend_read_only() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "python"
    assert plan.requires_approval is False
    assert len(plan.steps) == 5
    assert plan.parameters["minimum_total_claims"] == 1


def test_validate_rejects_undeclared_backend() -> None:
    handler = ValidateActionHandler()
    contract = resolve_verb_contract(VALIDATE_VERB)
    with pytest.raises(ValueError, match="is not declared for the validate verb"):
        handler.plan(_request(backend="abacus"), contract)


def test_generated_validate_script_embeds_verdict_and_compiles() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert "load_reference_validation_registry" in source
    assert f"EXPECTED_TOTAL_CLAIMS = {record.result.outputs['total_claims']!r}" in source


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "two", 0, 5000])
def test_as_positive_int_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_positive_int("v", bad, maximum=4096)


def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("validate-ledger.1") == "validate_ledger_1"
    assert _safe_slug("!!!") == "action"


@pytest.mark.parametrize(
    "parameters",
    [
        {"unexpected": 1},
        {"minimum_total_claims": 0},
        {"minimum_total_claims": True},
        {"minimum_total_claims": "many"},
    ],
)
def test_normalise_validate_rejects_invalid(parameters: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _normalise_validate(parameters)


def test_normalise_validate_defaults_minimum() -> None:
    assert _normalise_validate({}) == {"minimum_total_claims": 1}
    assert _normalise_validate({"minimum_total_claims": 5}) == {"minimum_total_claims": 5}
