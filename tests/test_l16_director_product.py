# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded-director bounded L16 director tests
"""Real-simulator, policy, contract, and evidence tests for bounded-director."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.codesign.contracts import SafetyAction
from scpn_quantum_control.codesign.evidence import build_demo_loop, demo_inputs
from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.l16.director_contracts import (
    L16_DIRECTOR_CLAIM_BOUNDARY,
    L16DirectorEvidence,
    L16IndicatorCertificate,
    L16RouteEvidence,
    L16ScenarioSpec,
)
from scpn_quantum_control.l16.director_evidence import (
    canonical_l16_json,
    l16_evidence_payload,
    main,
    render_l16_evidence_markdown,
    validate_l16_evidence,
    write_l16_evidence,
)
from scpn_quantum_control.l16.director_product import (
    L16DirectorPolicyError,
    frozen_l16_scenarios,
    informative_l16_indicators,
    l16_promotion_blockers,
    observer_inputs_from_l16,
    run_l16_director_suite,
    run_l16_indicator_scenario,
)
from scpn_quantum_control.l16.quantum_director import L16Result


@pytest.fixture(scope="module")
def evidence() -> L16DirectorEvidence:
    """Run the bounded exact-simulator suite once for contract assertions."""
    return run_l16_director_suite()


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    """Run the digest-bound public evidence path once."""
    return l16_evidence_payload()


def test_frozen_suite_is_functional_but_not_promotion_ready(
    evidence: L16DirectorEvidence,
) -> None:
    """Retain exact scenarios, deterministic replay, and honest negative findings."""
    assert tuple(item.scenario.scenario_id for item in evidence.certificates) == (
        "paper27_baseline",
        "susceptibility_probe",
        "weak_coupling_probe",
    )
    assert evidence.functional_passed is True
    assert evidence.promotion_ready is False
    assert evidence.action_diversity is False
    assert {item.heuristic_action for item in evidence.certificates} == {"continue"}
    assert {item.codesign_action for item in evidence.certificates} == {"allow"}
    assert all(item.deterministic_replay and item.passed for item in evidence.certificates)
    assert all(item.hardware_execution is False for item in evidence.certificates)
    assert any(item.informative_indicators for item in evidence.certificates)
    assert any("action diversity" in blocker for blocker in evidence.promotion_blockers)
    assert any("fewer than two" in blocker for blocker in evidence.promotion_blockers)


def test_indicator_classification_and_promotion_policy_cover_both_outcomes(
    evidence: L16DirectorEvidence,
) -> None:
    """Classify all raw indicators and retain only fixed blockers for diverse evidence."""
    changed = L16Result(0.5, 0.2, 0.3, 0.4, 0.6, "adjust")
    assert informative_l16_indicators(changed) == (
        "loschmidt_echo",
        "energy_variance",
        "fidelity_susceptibility",
        "order_parameter",
    )

    rich_certificates = tuple(
        replace(
            certificate,
            heuristic_action=("continue", "adjust", "halt")[index],
            codesign_action=("allow", "hold", "abort")[index],
            informative_indicators=("energy_variance", "fidelity_susceptibility"),
        )
        for index, certificate in enumerate(evidence.certificates)
    )
    blockers = l16_promotion_blockers(rich_certificates)
    assert len(blockers) == 3
    assert all("action diversity" not in blocker for blocker in blockers)


def test_l16_actions_drive_real_codesign_safety_interlocks() -> None:
    """Route all legacy labels through a real conservative co-design loop step."""
    loop = build_demo_loop()
    step_input = demo_inputs()[0]

    continued = loop.step(step_input, observers=observer_inputs_from_l16("continue"))
    adjusted = loop.step(
        step_input,
        observers=observer_inputs_from_l16("adjust", reason="local indicator warning"),
    )
    halted = loop.step(step_input, observers=observer_inputs_from_l16("halt"))

    assert continued.safety.action in {SafetyAction.ALLOW, SafetyAction.CLAMP}
    assert adjusted.safety.action is SafetyAction.HOLD
    assert adjusted.safety.reason == "local indicator warning"
    assert halted.safety.action is SafetyAction.ABORT
    with pytest.raises(ValueError, match="continue, adjust, or halt"):
        observer_inputs_from_l16("invented")


def test_bl67_policy_refuses_unauthorised_and_hardware_execution() -> None:
    """Allow local simulation but refuse both incomplete and ticketed hardware modes."""
    scenario = frozen_l16_scenarios()[0]
    local = run_l16_indicator_scenario(
        scenario,
        policy=ClosedLoopExecutionPolicy(round_budget=1),
    )
    assert local.policy_authorised is True
    assert local.route_id == "adapter:l16.local_indicator"

    with pytest.raises(L16DirectorPolicyError, match="refused"):
        run_l16_indicator_scenario(
            scenario,
            policy=ClosedLoopExecutionPolicy(allow_hardware=True, round_budget=1),
        )
    with pytest.raises(L16DirectorPolicyError, match="local-simulator only"):
        run_l16_indicator_scenario(
            scenario,
            policy=ClosedLoopExecutionPolicy(
                allow_hardware=True,
                live_ticket="owner-ticket",
                backend_allowlist=("simulated-qpu",),
                round_budget=1,
            ),
            backend="simulated-qpu",
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: L16ScenarioSpec("", 2, 1.0, 1.0, 0.5), "scenario_id"),
        (lambda: L16ScenarioSpec("x", True, 1.0, 1.0, 0.5), "oscillators"),
        (lambda: L16ScenarioSpec("x", 7, 1.0, 1.0, 0.5), "oscillators"),
        (lambda: L16ScenarioSpec("x", 2, -1.0, 1.0, 0.5), "coupling_scale"),
        (lambda: L16ScenarioSpec("x", 2, 1.0, 0.0, 0.5), "frequency_scale"),
        (lambda: L16ScenarioSpec("x", 2, 1.0, 1.0, float("nan")), "evolution_time"),
        (lambda: L16RouteEvidence("", "supported", ""), "route_id"),
        (lambda: L16RouteEvidence("x", "open", ""), "status"),
        (lambda: L16RouteEvidence("x", "supported", "reason"), "cannot carry"),
        (lambda: L16RouteEvidence("x", "permanent_boundary", ""), "require"),
    ],
)
def test_scenario_and_route_contracts_fail_closed(
    factory: Callable[[], object],
    message: str,
) -> None:
    """Reject malformed bounded scenarios and route records."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"loschmidt_echo": 2.0}, "loschmidt_echo"),
        ({"energy_variance": -1.0}, "energy_variance"),
        ({"fidelity_susceptibility": float("inf")}, "fidelity_susceptibility"),
        ({"order_parameter": -2.0}, "order_parameter"),
        ({"heuristic_score": True}, "heuristic_score"),
        ({"heuristic_action": "unknown"}, "heuristic_action"),
        ({"heuristic_action": "adjust"}, "codesign_action"),
        ({"informative_indicators": ("energy_variance", "energy_variance")}, "unique"),
        ({"informative_indicators": ("unknown",)}, "unknown indicator"),
        ({"route_id": ""}, "route_id"),
        ({"hardware_execution": True}, "simulator-only"),
        ({"claim_boundary": "promoted"}, "claim boundary"),
    ],
)
def test_indicator_certificate_contract_fails_closed(
    evidence: L16DirectorEvidence,
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject promoted, inconsistent, and non-finite indicator certificates."""
    unchecked_replace = cast(Callable[..., L16IndicatorCertificate], replace)
    with pytest.raises(ValueError, match=message):
        unchecked_replace(evidence.certificates[0], **changes)


def test_evidence_contract_rejects_incomplete_or_promoted_records(
    evidence: L16DirectorEvidence,
) -> None:
    """Require exact scenario/routes, blockers, schema, and local execution flags."""
    with pytest.raises(ValueError, match="scenario order"):
        replace(evidence, certificates=tuple(reversed(evidence.certificates)))
    with pytest.raises(ValueError, match="local and autonomous"):
        replace(evidence, routes=evidence.routes[:1])
    with pytest.raises(ValueError, match="ordered local"):
        replace(evidence, routes=tuple(reversed(evidence.routes)))
    with pytest.raises(ValueError, match="promotion_blockers"):
        replace(evidence, promotion_blockers=())
    with pytest.raises(ValueError, match="schema and claim boundary"):
        replace(evidence, schema="invented")
    with pytest.raises(ValueError, match="cannot promote"):
        replace(evidence, provider_execution=True)

    failed_certificate = replace(evidence.certificates[0], policy_authorised=False)
    adjusted_certificate = replace(
        evidence.certificates[1],
        heuristic_action="adjust",
        codesign_action="hold",
    )
    mixed = replace(
        evidence,
        certificates=(failed_certificate, adjusted_certificate, evidence.certificates[2]),
    )
    assert mixed.functional_passed is False
    assert mixed.action_diversity is True
    assert mixed.to_payload()["functional_passed"] is False


def test_digest_render_validation_and_atomic_write(
    payload: dict[str, object],
    tmp_path: Path,
) -> None:
    """Validate real evidence, render it, and reproduce exact atomic outputs."""
    assert validate_l16_evidence(payload) == ()
    unsigned = {key: value for key, value in payload.items() if key != "content_digest"}
    assert (
        payload["content_digest"]
        == hashlib.sha256(canonical_l16_json(unsigned).encode("utf-8")).hexdigest()
    )
    markdown = render_l16_evidence_markdown(payload)
    assert "Functional passed: `true`" in markdown
    assert "Promotion ready: `false`" in markdown
    assert L16_DIRECTOR_CLAIM_BOUNDARY in markdown

    json_path = tmp_path / "nested" / "evidence.json"
    markdown_path = tmp_path / "nested" / "evidence.md"
    written = write_l16_evidence(json_path, markdown_path, payload=payload)
    assert written == payload
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert markdown_path.read_text(encoding="utf-8") == markdown

    broken = copy.deepcopy(payload)
    broken["promotion_ready"] = True
    with pytest.raises(RuntimeError, match="promotion_ready"):
        write_l16_evidence(json_path, markdown_path, payload=broken)


def test_validator_reports_structural_and_integrity_failures(
    payload: dict[str, object],
) -> None:
    """Fail closed on every evidence class without relying on test doubles."""
    assert validate_l16_evidence([]) == ("payload must be a JSON object",)

    broken = copy.deepcopy(payload)
    broken.update(
        {
            "schema": "invented",
            "claim_boundary": "promoted",
            "functional_passed": False,
            "promotion_ready": True,
            "provider_execution": True,
            "hardware_execution": True,
            "certificates": "not-an-array",
            "routes": [],
            "promotion_blockers": [""],
            "content_digest": "bad",
        }
    )
    findings = validate_l16_evidence(broken)
    assert len(findings) >= 10

    malformed_certificates = copy.deepcopy(payload)
    malformed_certificates["certificates"] = [{"scenario": {}}, "invalid"]
    malformed_certificates["routes"] = [
        {"route_id": "adapter:l16.local_indicator", "closure_status": "permanent_boundary"},
        {
            "route_id": "adapter:l16.autonomous_hardware_control",
            "closure_status": "supported",
        },
    ]
    findings = validate_l16_evidence(malformed_certificates)
    assert any("three frozen scenarios" in item for item in findings)
    assert any("every L16 certificate" in item for item in findings)
    assert any("blocked-hardware" in item for item in findings)
    assert any("content_digest" in item for item in findings)

    noncanonical = copy.deepcopy(payload)
    noncanonical["invented"] = {"not-json"}
    findings = validate_l16_evidence(noncanonical)
    assert "payload contains a non-canonical JSON value" in findings


def test_markdown_rejects_non_numeric_indicator(payload: dict[str, object]) -> None:
    """Reject type-confused numeric fields on the public render path."""
    broken = copy.deepcopy(payload)
    certificates = broken["certificates"]
    assert isinstance(certificates, list)
    certificate = certificates[0]
    assert isinstance(certificate, dict)
    certificate["loschmidt_echo"] = True
    with pytest.raises(ValueError, match="loschmidt_echo"):
        render_l16_evidence_markdown(broken)


def test_cli_writes_default_contract_to_requested_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the real CLI path with bounded temporary outputs."""
    json_path = tmp_path / "cli.json"
    markdown_path = tmp_path / "cli.md"

    assert main(["--json-output", str(json_path), "--markdown-output", str(markdown_path)]) == 0
    output = capsys.readouterr().out
    assert str(json_path) in output
    assert "functional_passed=true" in output
    assert "promotion_ready=false" in output
    assert validate_l16_evidence(json.loads(json_path.read_text(encoding="utf-8"))) == ()
    assert markdown_path.is_file()
