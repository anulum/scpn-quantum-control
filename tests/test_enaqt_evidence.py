# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded ENAQT evidence tests
"""Evidence contracts, replay, CLI, and negative tests for ENAQT."""

from __future__ import annotations

import copy
import json
import runpy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.analysis.enaqt_evidence as evidence_module
from scpn_quantum_control.analysis.enaqt_evidence import (
    ENAQT_CLAIM_BOUNDARY,
    ENAQT_EVIDENCE_SCHEMA,
    ENAQT_LITERATURE,
    ENAQTScenario,
    canonical_enaqt_json,
    enaqt_evidence_payload,
    frozen_enaqt_scenarios,
    main,
    render_enaqt_evidence_markdown,
    validate_enaqt_evidence,
    write_enaqt_evidence,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    """Generate the deterministic three-scenario evidence once."""
    return enaqt_evidence_payload()


def test_frozen_scenarios_cover_positive_and_negative_controls() -> None:
    """Keep exactly one expected intermediate case and two controls."""
    scenarios = frozen_enaqt_scenarios()
    assert tuple(scenario.scenario_id for scenario in scenarios) == (
        "disordered_chain_intermediate",
        "uniform_chain_coherent_control",
        "disconnected_target_control",
    )
    assert [item.expected_intermediate_optimum for item in scenarios] == [
        True,
        False,
        False,
    ]
    coupling, omega = scenarios[0].arrays()
    assert coupling.shape == (4, 4)
    np.testing.assert_allclose(coupling, coupling.T)
    assert omega.tolist() == [0.0, 3.0, -2.0, 1.0]
    record = scenarios[0].to_dict()
    assert record["edges"] == [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]]


def test_functional_payload_is_bounded_and_digest_valid(
    payload: dict[str, object],
) -> None:
    """Prove real replay, classifications, literature pins, and claim limits."""
    assert payload["schema"] == ENAQT_EVIDENCE_SCHEMA
    assert payload["claim_boundary"] == ENAQT_CLAIM_BOUNDARY
    assert payload["literature"] == [dict(item) for item in ENAQT_LITERATURE]
    assert payload["functional_passed"] is True
    assert payload["intermediate_scenario_count"] == 1
    assert payload["negative_control_count"] == 2
    assert payload["bounded_claim_ready"] is True
    assert payload["universal_optimum_claimed"] is False
    assert payload["setpoint_policy_available"] is False
    assert payload["provider_execution"] is False
    assert payload["hardware_execution"] is False
    assert validate_enaqt_evidence(payload) == ()

    scenarios = cast(list[dict[str, object]], payload["scenarios"])
    assert all(item["passed"] is True for item in scenarios)
    results = {
        cast(dict[str, object], item["scenario"])["scenario_id"]: cast(
            dict[str, object], item["result"]
        )
        for item in scenarios
    }
    positive = results["disordered_chain_intermediate"]
    assert positive["optimal_gamma"] == 3.0
    assert positive["has_intermediate_optimum"] is True
    assert results["uniform_chain_coherent_control"]["optimal_gamma"] == 0.0
    assert results["disconnected_target_control"]["optimal_efficiency"] == pytest.approx(
        0.0, abs=1e-14
    )


def test_canonical_json_is_order_independent() -> None:
    """Use stable sorted compact bytes for integrity digests."""
    assert canonical_enaqt_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_markdown_reports_all_scenarios_and_boundary(payload: dict[str, object]) -> None:
    """Render measured rows without universal or actuation promotion."""
    markdown = render_enaqt_evidence_markdown(payload)
    assert "disordered_chain_intermediate" in markdown
    assert "uniform_chain_coherent_control" in markdown
    assert "disconnected_target_control" in markdown
    assert "3.37769137" in markdown
    assert ENAQT_CLAIM_BOUNDARY in markdown
    assert "universal optimum" in markdown


def test_writer_and_cli_emit_valid_atomic_files(
    payload: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Write supplied evidence and exercise the public CLI with real generation."""
    json_path = tmp_path / "supplied" / "evidence.json"
    markdown_path = tmp_path / "supplied" / "evidence.md"
    written = write_enaqt_evidence(json_path, markdown_path, payload=payload)
    assert written == payload
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# ENAQT bounded transport evidence"
    )
    assert not json_path.with_suffix(".json.tmp").exists()

    generated_json = tmp_path / "generated.json"
    generated_markdown = tmp_path / "generated.md"
    assert (
        main(
            [
                "--json-output",
                str(generated_json),
                "--markdown-output",
                str(generated_markdown),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert str(generated_json) in output
    assert "functional_passed=true" in output
    assert "universal/setpoint/hardware claims=false" in output
    assert validate_enaqt_evidence(json.loads(generated_json.read_text(encoding="utf-8"))) == ()


def test_default_cli_paths_are_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the repository evidence destinations fixed for deterministic replay."""
    seen: list[tuple[Path, Path]] = []

    def fake_write(json_path: Path, markdown_path: Path) -> dict[str, object]:
        seen.append((json_path, markdown_path))
        return {"functional_passed": True}

    monkeypatch.setattr(evidence_module, "write_enaqt_evidence", fake_write)
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0
    assert seen == [
        (
            Path("data/enaqt_product/enaqt_evidence.json"),
            Path("data/enaqt_product/enaqt_evidence.md"),
        )
    ]


def test_repository_runner_delegates_to_evidence_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the script entry point without rerunning the numerical suite."""
    monkeypatch.setattr(evidence_module, "main", lambda: 17)
    runner = Path(__file__).parents[1] / "scripts" / "run_enaqt_evidence.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(runner), run_name="__main__")
    assert error.value.code == 17


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"scenario_id": ""}, "scenario_id"),
        ({"site_energies": (0.0,)}, "site_energies"),
        ({"site_energies": (0.0, float("nan"))}, "site_energies"),
        ({"gamma_values": ()}, "gamma_values"),
        ({"gamma_values": (0.0, -1.0)}, "gamma_values"),
        ({"t_evolve": 0.0}, "t_evolve"),
        ({"source_site": -1}, "index"),
        ({"target_site": 0}, "must differ"),
        ({"edges": ((0, 4, 1.0),)}, "edges"),
        ({"edges": ((0, 0, 1.0),)}, "edges"),
        ({"edges": ((0, 1, float("inf")),)}, "edges"),
    ],
)
def test_scenario_contract_fails_closed(changes: dict[str, object], message: str) -> None:
    """Reject incomplete, non-finite, and invalid frozen scenarios."""
    unchecked_replace = cast(Callable[..., ENAQTScenario], replace)
    with pytest.raises(ValueError, match=message):
        unchecked_replace(frozen_enaqt_scenarios()[0], **changes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema": "wrong"}, "schema"),
        ({"claim_boundary": "promoted"}, "claim_boundary"),
        ({"functional_passed": False}, "functional_passed"),
        ({"intermediate_scenario_count": 2}, "intermediate_scenario_count"),
        ({"negative_control_count": 1}, "negative_control_count"),
        ({"bounded_claim_ready": False}, "bounded_claim_ready"),
        ({"universal_optimum_claimed": True}, "universal_optimum_claimed"),
        ({"setpoint_policy_available": True}, "setpoint_policy_available"),
        ({"provider_execution": True}, "provider_execution"),
        ({"hardware_execution": True}, "hardware_execution"),
        ({"literature": []}, "literature"),
        ({"scenarios": "wrong"}, "scenarios"),
        ({"content_digest": "0" * 64}, "content_digest"),
    ],
)
def test_evidence_validator_rejects_top_level_drift(
    payload: dict[str, object], mutation: dict[str, object], message: str
) -> None:
    """Reject promotion, execution, source, structure, and digest drift."""
    changed = copy.deepcopy(payload)
    changed.update(mutation)
    assert any(message in finding for finding in validate_enaqt_evidence(changed))


def test_evidence_validator_rejects_scenario_drift(payload: dict[str, object]) -> None:
    """Require exact scenario membership and passing replay classifications."""
    missing = copy.deepcopy(payload)
    cast(list[object], missing["scenarios"]).pop()
    findings = validate_enaqt_evidence(missing)
    assert any("three frozen" in finding for finding in findings)

    failed = copy.deepcopy(payload)
    first = cast(list[dict[str, object]], failed["scenarios"])[0]
    first["deterministic_replay"] = False
    assert any("classification and replay" in item for item in validate_enaqt_evidence(failed))


def test_evidence_validator_rejects_non_object_and_noncanonical_payload(
    payload: dict[str, object],
) -> None:
    """Fail closed on wrong roots and JSON-incompatible values."""
    assert validate_enaqt_evidence([]) == ("payload must be a JSON object",)
    changed = copy.deepcopy(payload)
    changed["unexpected"] = {"not", "json"}
    assert any("non-canonical" in item for item in validate_enaqt_evidence(changed))


def test_writer_rejects_invalid_supplied_payload(
    payload: dict[str, object], tmp_path: Path
) -> None:
    """Refuse to write evidence whose digest or boundary was altered."""
    changed = copy.deepcopy(payload)
    changed["bounded_claim_ready"] = False
    with pytest.raises(RuntimeError, match="invalid ENAQT evidence"):
        write_enaqt_evidence(tmp_path / "bad.json", tmp_path / "bad.md", payload=changed)


def test_markdown_rejects_non_numeric_result_field(payload: dict[str, object]) -> None:
    """Fail closed when a human-readable numeric row is malformed."""
    changed = copy.deepcopy(payload)
    scenarios = cast(list[dict[str, object]], changed["scenarios"])
    result = cast(dict[str, object], scenarios[0]["result"])
    result["optimal_gamma"] = True
    with pytest.raises(ValueError, match="optimal_gamma must be numeric"):
        render_enaqt_evidence_markdown(changed)
