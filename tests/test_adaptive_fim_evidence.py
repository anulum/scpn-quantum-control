# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive FIM evidence tests
"""Replay, digest, CLI, and negative tests for adaptive-FIM evidence."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from typing import cast

import pytest

import scpn_quantum_control.analysis.adaptive_fim_evidence as evidence_module
from scpn_quantum_control.analysis.adaptive_fim_evidence import (
    ADAPTIVE_FIM_EVIDENCE_SCHEMA,
    ADAPTIVE_FIM_LITERATURE,
    HISTORICAL_SOURCE,
    REPLAY_CIRCUIT_INDICES,
    adaptive_fim_evidence_payload,
    canonical_adaptive_fim_json,
    historical_replay_witnesses,
    main,
    render_adaptive_fim_evidence_markdown,
    synthetic_calibration_witnesses,
    validate_adaptive_fim_evidence,
    write_adaptive_fim_evidence,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return adaptive_fim_evidence_payload()


def test_historical_custody_replay_extracts_exact_disjoint_counts() -> None:
    witnesses = historical_replay_witnesses()

    assert len(witnesses) == 3
    assert all(witness.count_bound for witness in witnesses)
    assert all(witness.source == "hardware_replay" for witness in witnesses)
    assert [witness.shots for witness in witnesses] == [2048, 2048, 2048]
    assert witnesses[0].leakage_events == 211
    assert witnesses[0].retention_events == 1440
    assert witnesses[0].artifact_id is not None
    assert "circuit-0" in witnesses[0].artifact_id


def test_synthetic_controls_freeze_signal_boundary_and_power_cases() -> None:
    witnesses = synthetic_calibration_witnesses()

    assert [witness.leakage_events for witness in witnesses] == [60, 25, 3]
    assert [witness.shots for witness in witnesses] == [512, 512, 32]
    assert all(witness.source == "synthetic" for witness in witnesses)


def test_payload_is_functional_bounded_and_digest_valid(payload: dict[str, object]) -> None:
    assert payload["schema"] == ADAPTIVE_FIM_EVIDENCE_SCHEMA
    assert payload["literature"] == [dict(item) for item in ADAPTIVE_FIM_LITERATURE]
    assert payload["functional_passed"] is True
    assert payload["provider_submission"] is False
    assert payload["hardware_execution"] is False
    assert payload["closed_loop_validated"] is False
    assert payload["fim_protection_claimed"] is False
    assert payload["optimal_policy_claimed"] is False
    assert payload["quantum_advantage_claimed"] is False
    assert validate_adaptive_fim_evidence(payload) == ()

    source = cast(dict[str, object], payload["historical_source"])
    assert source["circuit_indices"] == list(REPLAY_CIRCUIT_INDICES)
    assert source["sha256"] == "13948b12223dbc64f659cb26de393bd9894dba37c2a3787ce15d3b6aad4089d2"
    calibration = cast(dict[str, object], payload["synthetic_calibration"])
    replay = cast(dict[str, object], payload["historical_offline_replay"])
    assert [row["decision"] for row in cast(list[dict[str, object]], calibration["steps"])] == [
        "decrease",
        "hold",
        "hold",
    ]
    assert [row["decision"] for row in cast(list[dict[str, object]], replay["steps"])] == [
        "decrease",
        "decrease",
        "decrease",
    ]


def test_canonical_json_is_order_independent() -> None:
    assert canonical_adaptive_fim_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_markdown_reports_calibration_custody_and_boundary(payload: dict[str, object]) -> None:
    markdown = render_adaptive_fim_evidence_markdown(payload)

    assert "decrease -> hold -> hold" in markdown
    assert "decrease -> decrease -> decrease" in markdown
    assert "ibm-run-cf4835290f607387" in markdown
    assert "does not validate" in markdown


def test_writer_and_cli_emit_valid_atomic_files(
    payload: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_path = tmp_path / "nested" / "evidence.json"
    markdown_path = tmp_path / "nested" / "evidence.md"
    assert write_adaptive_fim_evidence(json_path, markdown_path, payload=payload) == payload
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert markdown_path.read_text(encoding="utf-8").startswith("# Adaptive FIM proposal evidence")
    assert not json_path.with_suffix(".json.tmp").exists()

    generated_json = tmp_path / "generated.json"
    generated_markdown = tmp_path / "generated.md"
    assert (
        main(["--json-output", str(generated_json), "--markdown-output", str(generated_markdown)])
        == 0
    )
    assert "functional_passed=true" in capsys.readouterr().out
    assert (
        validate_adaptive_fim_evidence(json.loads(generated_json.read_text(encoding="utf-8")))
        == ()
    )


def test_default_cli_paths_are_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Path, Path, Path]] = []

    def fake_write(
        json_path: Path,
        markdown_path: Path,
        *,
        source_path: Path,
    ) -> dict[str, object]:
        seen.append((json_path, markdown_path, source_path))
        return {"functional_passed": True}

    monkeypatch.setattr(evidence_module, "write_adaptive_fim_evidence", fake_write)
    assert main([]) == 0
    assert seen == [
        (
            Path("data/adaptive_fim_product/adaptive_fim_evidence.json"),
            Path("data/adaptive_fim_product/adaptive_fim_evidence.md"),
            HISTORICAL_SOURCE,
        )
    ]


def test_repository_runner_delegates_to_evidence_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evidence_module, "main", lambda: 19)
    runner = Path(__file__).parents[1] / "scripts" / "run_adaptive_fim_evidence.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(runner), run_name="__main__")
    assert error.value.code == 19


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema": "adaptive_fim_evidence.v1"}, "schema"),
        ({"claim_boundary": "promoted"}, "claim_boundary"),
        ({"functional_passed": False}, "functional_passed"),
        ({"provider_submission": True}, "provider_submission"),
        ({"hardware_execution": True}, "hardware_execution"),
        ({"closed_loop_validated": True}, "closed_loop_validated"),
        ({"fim_protection_claimed": True}, "fim_protection_claimed"),
        ({"optimal_policy_claimed": True}, "optimal_policy_claimed"),
        ({"quantum_advantage_claimed": True}, "quantum_advantage_claimed"),
        ({"literature": []}, "literature"),
        ({"historical_source": "bad"}, "historical_source"),
        ({"synthetic_calibration": "bad"}, "synthetic calibration"),
        ({"historical_offline_replay": "bad"}, "historical_offline_replay"),
        ({"budget_refusal": "bad"}, "budget_refusal"),
        ({"hardware_refusal": "bad"}, "hardware_refusal"),
        ({"content_digest": "0" * 64}, "content_digest"),
    ],
)
def test_validator_rejects_top_level_drift(
    payload: dict[str, object], mutation: dict[str, object], message: str
) -> None:
    changed = copy.deepcopy(payload)
    changed.update(mutation)
    assert any(message in finding for finding in validate_adaptive_fim_evidence(changed))


def test_validator_rejects_nested_promotion_and_action_drift(payload: dict[str, object]) -> None:
    source = copy.deepcopy(payload)
    cast(dict[str, object], source["historical_source"])["job_id"] = "wrong"
    assert any("job_id" in item for item in validate_adaptive_fim_evidence(source))
    source_indices = copy.deepcopy(payload)
    cast(dict[str, object], source_indices["historical_source"])["circuit_indices"] = []
    assert any(
        "circuit_indices" in item for item in validate_adaptive_fim_evidence(source_indices)
    )
    source_use = copy.deepcopy(payload)
    cast(dict[str, object], source_use["historical_source"])["use"] = "promotion"
    assert any("offline replay" in item for item in validate_adaptive_fim_evidence(source_use))

    calibration = copy.deepcopy(payload)
    steps = cast(
        list[dict[str, object]],
        cast(dict[str, object], calibration["synthetic_calibration"])["steps"],
    )
    steps[0]["decision"] = "hold"
    assert any(
        "synthetic calibration" in item for item in validate_adaptive_fim_evidence(calibration)
    )
    calibration_schema = copy.deepcopy(payload)
    cast(dict[str, object], calibration_schema["synthetic_calibration"])["schema"] = (
        "adaptive_fim_feedback.v2"
    )
    assert any(
        "synthetic calibration plan contracts" in item
        for item in validate_adaptive_fim_evidence(calibration_schema)
    )
    calibration_observers = copy.deepcopy(payload)
    cast(dict[str, object], calibration_observers["synthetic_calibration"])["observers"] = "bad"
    assert any(
        "synthetic calibration plan contracts" in item
        for item in validate_adaptive_fim_evidence(calibration_observers)
    )

    replay = copy.deepcopy(payload)
    cast(dict[str, object], replay["historical_offline_replay"])["closed_loop_efficacy_tested"] = (
        True
    )
    assert any("efficacy" in item for item in validate_adaptive_fim_evidence(replay))
    replay_actions = copy.deepcopy(payload)
    replay_steps = cast(
        list[dict[str, object]],
        cast(dict[str, object], replay_actions["historical_offline_replay"])["steps"],
    )
    replay_steps[0]["decision"] = "hold"
    assert any(
        "three decreases" in item for item in validate_adaptive_fim_evidence(replay_actions)
    )
    replay_contract = copy.deepcopy(payload)
    contract_steps = cast(
        list[dict[str, object]],
        cast(dict[str, object], replay_contract["historical_offline_replay"])["steps"],
    )
    contract_steps[0]["claim_boundary"] = "broader replay claims"
    assert any(
        "historical replay step contracts" in item
        for item in validate_adaptive_fim_evidence(replay_contract)
    )
    replay_nondeterministic = copy.deepcopy(payload)
    cast(dict[str, object], replay_nondeterministic["historical_offline_replay"])[
        "deterministic_replay"
    ] = False
    assert any(
        "deterministic" in item for item in validate_adaptive_fim_evidence(replay_nondeterministic)
    )
    refusal_steps = copy.deepcopy(payload)
    cast(dict[str, object], refusal_steps["budget_refusal"])["steps"] = [{}]
    assert any("no proposals" in item for item in validate_adaptive_fim_evidence(refusal_steps))
    refusal_schema = copy.deepcopy(payload)
    cast(dict[str, object], refusal_schema["budget_refusal"])["schema"] = (
        "adaptive_fim_feedback.v2"
    )
    assert any(
        "budget_refusal plan contracts" in item
        for item in validate_adaptive_fim_evidence(refusal_schema)
    )
    invalid_steps = copy.deepcopy(payload)
    cast(dict[str, object], invalid_steps["synthetic_calibration"])["steps"] = "bad"
    assert any(
        "synthetic calibration" in item for item in validate_adaptive_fim_evidence(invalid_steps)
    )


def test_loader_rejects_invalid_custody_shapes(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        historical_replay_witnesses(invalid)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"status": "pending", "job_id": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="custody identifiers"):
        historical_replay_witnesses(wrong)


def test_loader_rejects_row_and_count_drift(tmp_path: Path) -> None:
    source = json.loads(HISTORICAL_SOURCE.read_text(encoding="utf-8"))

    no_rows = copy.deepcopy(source)
    no_rows["result_rows"] = "bad"
    no_rows_path = tmp_path / "no_rows.json"
    no_rows_path.write_text(json.dumps(no_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="result_rows"):
        historical_replay_witnesses(no_rows_path)

    missing = copy.deepcopy(source)
    missing["result_rows"] = []
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        historical_replay_witnesses(missing_path)

    wrong_lambda = copy.deepcopy(source)
    rows = cast(list[dict[str, object]], wrong_lambda["result_rows"])
    selected = next(
        row for row in rows if cast(dict[str, object], row["metadata"])["circuit_index"] == 0
    )
    cast(dict[str, object], selected["metadata"])["lambda_fim"] = 1.0
    wrong_lambda_path = tmp_path / "wrong_lambda.json"
    wrong_lambda_path.write_text(json.dumps(wrong_lambda), encoding="utf-8")
    with pytest.raises(ValueError, match="lambda/depth"):
        historical_replay_witnesses(wrong_lambda_path)

    duplicate = copy.deepcopy(source)
    duplicate_rows = cast(list[dict[str, object]], duplicate["result_rows"])
    duplicate_rows.append(copy.deepcopy(selected))
    cast(dict[str, object], duplicate_rows[-1]["metadata"])["lambda_fim"] = 4.0
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        historical_replay_witnesses(duplicate_path)

    malformed = copy.deepcopy(source)
    cast(list[object], malformed["result_rows"]).insert(0, "skip-me")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    assert len(historical_replay_witnesses(malformed_path)) == 3


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({}, "metadata and counts"),
        ({"metadata": {}, "counts": {}}, "metadata"),
        (
            {
                "metadata": {
                    "shots": 1,
                    "initial_bitstring": "0",
                    "popcount": 0,
                    "circuit_index": 0,
                    "depth": 2,
                },
                "counts": {1: 1},
            },
            "bitstrings",
        ),
        (
            {
                "metadata": {
                    "shots": 2,
                    "initial_bitstring": "0",
                    "popcount": 0,
                    "circuit_index": 0,
                    "depth": 2,
                },
                "counts": {"0": 1},
            },
            "sum",
        ),
    ],
)
def test_historical_row_parser_rejects_invalid_shapes(
    row: dict[object, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evidence_module._witness_from_historical_row(cast(dict[str, object], row))


def test_renderer_and_writer_reject_invalid_payload(
    payload: dict[str, object], tmp_path: Path
) -> None:
    changed = copy.deepcopy(payload)
    changed["hardware_execution"] = True
    with pytest.raises(ValueError, match="invalid adaptive FIM evidence"):
        render_adaptive_fim_evidence_markdown(changed)
    with pytest.raises(ValueError, match="invalid adaptive FIM evidence"):
        write_adaptive_fim_evidence(tmp_path / "bad.json", tmp_path / "bad.md", payload=changed)


def test_validator_rejects_non_object() -> None:
    assert validate_adaptive_fim_evidence([]) == ("payload must be a JSON object",)
