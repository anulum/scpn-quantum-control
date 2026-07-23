# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive spine tests
"""Tests for the verb-agnostic executive action spine."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio.executive import (  # noqa: E402
    ActionHandler,
    ActionRegistry,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRecord,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
    preview_action,
    resolve_verb_contract,
    run_action,
)

_SCRIPT_SOURCE = "print('reproduced')\n"


def _script() -> GeneratedScript:
    return build_generated_script(
        filename="reproduce.py", entrypoint="python reproduce.py", source=_SCRIPT_SOURCE
    )


class _StubHandler(ActionHandler):
    """A configurable handler for exercising the spine."""

    def __init__(
        self,
        verb: str,
        *,
        result: ExecutionResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._verb = verb
        self._result = result if result is not None else ExecutionResult("succeeded", {"ok": True})
        self._raises = raises

    @property
    def verb(self) -> str:
        return self._verb

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        return ExecutionPlan(
            verb=self._verb,
            action_id=request.action_id,
            backend=contract.backends[0],
            contract=contract,
            claim_boundary="stub boundary",
            steps=("do the thing",),
            parameters=dict(request.parameters),
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        if self._raises is not None:
            raise self._raises
        return self._result

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        return _script()


def _request(verb: str = "differentiate", *, approved: bool = False) -> ExecutiveRequest:
    return ExecutiveRequest(verb=verb, action_id="unit", parameters={"k": 1}, approved=approved)


# --------------------------------------------------------------------------- #
# resolve_verb_contract
# --------------------------------------------------------------------------- #
def test_resolve_contract_read_only_verb_needs_no_approval() -> None:
    """Read-only verbs must preserve their ungated evidence contract."""
    contract = resolve_verb_contract("differentiate")
    assert contract.requires_approval is False
    assert contract.side_effect == "READ_ONLY"
    assert "studio.differentiation-evidence.v1" in contract.produces
    assert contract.to_dict()["verb"] == "differentiate"


def test_resolve_contract_live_hardware_verb_requires_approval() -> None:
    """Live-hardware verbs must require explicit approval."""
    contract = resolve_verb_contract("execute")
    assert contract.requires_approval is True
    assert contract.side_effect == "LIVE_HARDWARE"


def test_resolve_contract_rejects_unknown_verb() -> None:
    """Unknown verbs must fail closed during contract resolution."""
    with pytest.raises(KeyError, match="no studio verb"):
        resolve_verb_contract("teleport")


# --------------------------------------------------------------------------- #
# ActionRegistry
# --------------------------------------------------------------------------- #
def test_registry_register_resolve_and_list() -> None:
    """Registered handlers must resolve and list deterministically."""
    registry = ActionRegistry()
    handler = _StubHandler("differentiate")
    registry.register(handler)
    assert registry.resolve("differentiate") is handler
    assert registry.verbs() == ("differentiate",)


def test_registry_rejects_unknown_verb() -> None:
    """The registry must reject handlers for undeclared verbs."""
    with pytest.raises(ValueError, match="unknown verb"):
        ActionRegistry().register(_StubHandler("teleport"))


def test_registry_rejects_duplicate() -> None:
    """The registry must reject duplicate handlers for one verb."""
    registry = ActionRegistry()
    registry.register(_StubHandler("differentiate"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_StubHandler("differentiate"))


def test_registry_resolve_missing() -> None:
    """Resolving an unregistered handler must fail closed."""
    with pytest.raises(KeyError, match="no executive handler"):
        ActionRegistry().resolve("differentiate")


# --------------------------------------------------------------------------- #
# ExecutiveRequest
# --------------------------------------------------------------------------- #
def test_request_to_dict() -> None:
    """Requests must expose their complete JSON-ready representation."""
    payload = _request().to_dict()
    assert payload["verb"] == "differentiate"
    assert payload["approved"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"verb": ""},
        {"action_id": ""},
        {"parameters": ["not", "mapping"]},
        {"backend": ""},
        {"approved": 1},
    ],
)
def test_request_rejects_invalid_fields(kwargs: dict[str, Any]) -> None:
    """Requests must reject every malformed field boundary."""
    base: dict[str, Any] = {"verb": "differentiate", "action_id": "unit", "parameters": {}}
    base.update(kwargs)
    with pytest.raises(ValueError):
        ExecutiveRequest(**base)


# --------------------------------------------------------------------------- #
# ExecutionPlan
# --------------------------------------------------------------------------- #
def _plan(**overrides: Any) -> ExecutionPlan:
    contract = resolve_verb_contract("differentiate")
    kwargs: dict[str, Any] = {
        "verb": "differentiate",
        "action_id": "unit",
        "backend": contract.backends[0],
        "contract": contract,
        "claim_boundary": "boundary",
        "steps": ("step",),
        "parameters": {"k": 1},
    }
    kwargs.update(overrides)
    return ExecutionPlan(**kwargs)


def test_plan_valid_and_to_dict() -> None:
    """Valid plans must expose approval state and backend evidence."""
    plan = _plan()
    assert plan.requires_approval is False
    assert plan.to_dict()["backend"] == plan.backend


@pytest.mark.parametrize(
    "overrides",
    [
        {"verb": "compile"},
        {"backend": ""},
        {"backend": "nonexistent-backend"},
        {"claim_boundary": ""},
        {"steps": ()},
        {"parameters": ["x"]},
    ],
)
def test_plan_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    """Plans must reject every contract-inconsistent field boundary."""
    with pytest.raises(ValueError):
        _plan(**overrides)


# --------------------------------------------------------------------------- #
# ExecutionResult
# --------------------------------------------------------------------------- #
def test_result_valid_variants_and_to_dict() -> None:
    """Valid results must retain status and error evidence."""
    ok = ExecutionResult("succeeded", {"value": 1})
    gated = ExecutionResult("gated", {}, error="needs approval")
    assert ok.to_dict()["status"] == "succeeded"
    assert gated.to_dict()["error"] == "needs approval"


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (("weird", {}), {"error": "x"}),
        (("succeeded", ["x"]), {}),
        (("succeeded", {}), {"error": "should not be here"}),
        (("failed", {}), {}),
    ],
)
def test_result_rejects_invalid(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Results must reject invalid status, output, and error coupling."""
    with pytest.raises(ValueError):
        ExecutionResult(*args, **kwargs)


# --------------------------------------------------------------------------- #
# GeneratedScript
# --------------------------------------------------------------------------- #
def test_generated_script_valid_and_to_dict() -> None:
    """Generated scripts must expose their content-addressed record."""
    script = _script()
    assert script.to_dict()["digest"] == script.digest
    assert script.digest.startswith("sha256:")


def test_generated_script_rejects_wrong_digest() -> None:
    """Generated scripts must reject a mismatched content digest."""
    with pytest.raises(ValueError, match="digest must be the sha256"):
        GeneratedScript(
            language="python",
            filename="a.py",
            entrypoint="python a.py",
            source=_SCRIPT_SOURCE,
            digest="sha256:deadbeef",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"language": "ruby"},
        {"filename": "a.txt"},
        {"entrypoint": ""},
        {"source": "   "},
    ],
)
def test_generated_script_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    """Generated scripts must reject every invalid field boundary."""
    kwargs: dict[str, Any] = {
        "filename": "a.py",
        "entrypoint": "python a.py",
        "source": _SCRIPT_SOURCE,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        build_generated_script(**kwargs)


# --------------------------------------------------------------------------- #
# run_action / preview_action / ExecutiveRecord
# --------------------------------------------------------------------------- #
def _registry(handler: ActionHandler) -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(handler)
    return registry


def test_preview_returns_plan_without_executing() -> None:
    """Preview must return a plan without invoking execution."""
    registry = _registry(_StubHandler("differentiate"))
    plan = preview_action(_request(), registry=registry)
    assert plan.verb == "differentiate"


def test_run_action_succeeds_and_seals_a_script() -> None:
    """Successful actions must seal their reproduction script and claims."""
    registry = _registry(_StubHandler("differentiate"))
    record = run_action(_request(), registry=registry)
    assert record.result.status == "succeeded"
    assert record.script is not None
    assert record.produced_schemas == resolve_verb_contract("differentiate").produces
    assert record.claim_boundary == "stub boundary"
    assert record.to_dict()["digest"] == record.digest


def test_run_action_gates_live_hardware_without_approval() -> None:
    """Unapproved live-hardware actions must gate before execution."""
    registry = _registry(_StubHandler("execute"))
    record = run_action(_request("execute"), registry=registry)
    assert record.result.status == "gated"
    assert record.script is None
    assert "requires an explicit approval" in (record.result.error or "")


def test_run_action_executes_gated_verb_when_approved() -> None:
    """Explicit approval must admit an otherwise gated verb."""
    registry = _registry(_StubHandler("execute"))
    record = run_action(_request("execute", approved=True), registry=registry)
    assert record.result.status == "succeeded"
    assert record.script is not None


def test_run_action_marks_failure_when_handler_raises() -> None:
    """Handler exceptions must become fail-closed records."""
    registry = _registry(_StubHandler("differentiate", raises=RuntimeError("boom")))
    record = run_action(_request(), registry=registry)
    assert record.result.status == "failed"
    assert record.script is None
    assert "RuntimeError: boom" in (record.result.error or "")


def test_run_action_non_succeeded_execute_result_is_not_scripted() -> None:
    """Non-success results must never receive reproduction scripts."""
    gated = ExecutionResult("gated", {}, error="handler declined")
    registry = _registry(_StubHandler("differentiate", result=gated))
    record = run_action(_request(), registry=registry)
    assert record.result.status == "gated"
    assert record.script is None


def test_record_rejects_tampered_digest() -> None:
    """Executive records must reject tampered seal digests."""
    request = _request()
    plan = _plan()
    result = ExecutionResult("gated", {}, error="x")
    with pytest.raises(ValueError, match="digest must seal"):
        ExecutiveRecord(request=request, plan=plan, result=result, script=None, digest="sha256:x")


def test_record_requires_script_for_success() -> None:
    """Successful executive records must carry a reproduction script."""
    request = _request()
    plan = _plan()
    result = ExecutionResult("succeeded", {"ok": True})
    from scpn_quantum_control.studio.executive import _digest

    digest = _digest(
        {
            "request": request.to_dict(),
            "plan": plan.to_dict(),
            "result": result.to_dict(),
            "script": None,
        }
    )
    with pytest.raises(ValueError, match="must carry a reproduction script"):
        ExecutiveRecord(request=request, plan=plan, result=result, script=None, digest=digest)


def test_record_rejects_script_on_non_success() -> None:
    """Non-success executive records must reject attached scripts."""
    request = _request()
    plan = _plan()
    result = ExecutionResult("gated", {}, error="x")
    script = _script()
    from scpn_quantum_control.studio.executive import _digest

    digest = _digest(
        {
            "request": request.to_dict(),
            "plan": plan.to_dict(),
            "result": result.to_dict(),
            "script": script.to_dict(),
        }
    )
    with pytest.raises(ValueError, match="only a succeeded action"):
        ExecutiveRecord(request=request, plan=plan, result=result, script=script, digest=digest)


def test_abstract_handler_methods_are_defined() -> None:
    """Abstract handler defaults must remain callable no-op contracts."""

    class _Concrete(ActionHandler):
        @property
        def verb(self) -> str:
            return super().verb  # type: ignore[safe-super]

        def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
            return super().plan(request, contract)  # type: ignore[safe-super]

        def execute(self, plan: ExecutionPlan) -> ExecutionResult:
            return super().execute(plan)  # type: ignore[safe-super]

        def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
            return super().generate_script(plan, result)  # type: ignore[safe-super]

    handler = _Concrete()
    contract = resolve_verb_contract("differentiate")
    request = _request()
    assert cast(object, handler.verb) is None
    assert cast(object, handler.plan(request, contract)) is None
    assert cast(object, handler.execute(_plan())) is None
    assert cast(object, handler.generate_script(_plan(), ExecutionResult("succeeded", {}))) is None
