# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive execute handler tests
"""Tests for the approval-gated QPU deployment ``execute`` handler."""

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
from scpn_quantum_control.studio.executive_execute import (  # noqa: E402
    EXECUTE_VERB,
    ExecuteActionHandler,
    _normalise_deployment,
    _safe_slug,
)

_DEPLOYMENT: dict[str, Any] = {
    "provider": "ibm-quantum",
    "endpoint": "ibm_brisbane",
    "circuit_digest": "sha256:abc123",
    "circuit_ref": "data/studio/xy_compile_recompute_unit_20260708.json",
    "shots": 4096,
}


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(ExecuteActionHandler())
    return registry


def _request(
    *, approved: bool = False, backend: str | None = None, **overrides: Any
) -> ExecutiveRequest:
    parameters = dict(_DEPLOYMENT)
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=EXECUTE_VERB,
        action_id="deploy-brisbane",
        parameters=parameters,
        backend=backend,
        approved=approved,
    )


# --------------------------------------------------------------------------- #
# approval gate
# --------------------------------------------------------------------------- #
def test_execute_gates_closed_without_approval() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "gated"
    assert record.script is None
    assert "requires an explicit approval" in (record.result.error or "")


def test_execute_runs_a_no_submit_deployment_when_approved() -> None:
    record = run_action(_request(approved=True), registry=_registry())
    assert record.result.status == "succeeded"
    outputs = record.result.outputs
    assert outputs["submitted"] is False
    assert outputs["result_status"] == "unverifiable"
    assert outputs["verifiability_mode"] == "attestation"
    assert outputs["result_schema"] == "studio.qpu-result-pack.v1"
    assert outputs["shots"] == 4096
    assert record.script is not None


def test_execute_plan_defaults_backend_and_is_gated() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "qiskit-runtime"
    assert plan.requires_approval is True
    assert len(plan.steps) == 4


def test_execute_accepts_declared_provider_hal_backend() -> None:
    plan = preview_action(_request(backend="provider-hal"), registry=_registry())
    assert plan.backend == "provider-hal"


def test_execute_rejects_undeclared_backend() -> None:
    handler = ExecuteActionHandler()
    contract = resolve_verb_contract(EXECUTE_VERB)
    with pytest.raises(ValueError, match="is not declared for the execute verb"):
        handler.plan(_request(backend="my-laptop"), contract)


# --------------------------------------------------------------------------- #
# generated submission script
# --------------------------------------------------------------------------- #
def test_generated_deploy_script_is_guarded_and_compiles() -> None:
    record = run_action(_request(approved=True), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert "--confirm" in source
    assert "refusing to submit a live QPU job without --confirm" in source
    assert "build_qpu_result_pack_unit" in source
    assert "sha256:abc123" in source
    assert record.script.entrypoint.endswith("--confirm")


def test_deployment_dossier_carries_optional_calibration_ref() -> None:
    record = run_action(
        _request(approved=True, calibration_ref="calibration/brisbane_2026-07-09"),
        registry=_registry(),
    )
    assert record.result.outputs["calibration_ref"] == "calibration/brisbane_2026-07-09"


# --------------------------------------------------------------------------- #
# _normalise_deployment validation branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": ""},
        {"provider": 3},
        {"endpoint": "  "},
        {"circuit_digest": "abc123"},
        {"circuit_digest": ""},
        {"circuit_ref": ""},
        {"shots": 0},
        {"shots": -5},
        {"shots": True},
        {"shots": "many"},
        {"shots": 10_000_000},
        {"calibration_ref": ""},
    ],
)
def test_normalise_deployment_rejects_invalid(overrides: dict[str, Any]) -> None:
    parameters = dict(_DEPLOYMENT)
    parameters.update(overrides)
    with pytest.raises(ValueError):
        _normalise_deployment(parameters)


def test_normalise_deployment_accepts_bounded_spec() -> None:
    deployment = _normalise_deployment(_DEPLOYMENT)
    assert deployment["provider"] == "ibm-quantum"
    assert deployment["shots"] == 4096
    assert "calibration_ref" not in deployment


def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("deploy-brisbane.1") == "deploy_brisbane_1"
    assert _safe_slug("///") == "action"
