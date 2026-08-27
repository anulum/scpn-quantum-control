# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — entanglement-sync Entangled Initial-State Evidence Tests
"""Evidence, replay, CLI, and fail-closed tests for entanglement-sync."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.analysis.entanglement_sync_evidence import (
    ENTANGLEMENT_SYNC_CLAIM_BOUNDARY,
    ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA,
    ENTANGLEMENT_SYNC_LITERATURE,
    canonical_entanglement_sync_json,
    entanglement_sync_evidence_payload,
    frozen_entanglement_sync_scenario,
    main,
    render_entanglement_sync_evidence_markdown,
    validate_entanglement_sync_evidence,
    write_entanglement_sync_evidence,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    """Generate the deterministic frozen study once for evidence assertions."""
    return entanglement_sync_evidence_payload()


def test_frozen_scenario_has_exact_model_and_control() -> None:
    """Keep the four-qubit Hamiltonian, grid, observables, and control immutable."""
    scenario = frozen_entanglement_sync_scenario()
    assert scenario["scenario_id"] == "paper27_four_qubit_initial_state_controls"
    assert scenario["omega"] == [1.329, 2.61, 0.844, 1.52]
    coupling = cast(list[list[float]], scenario["coupling"])
    assert len(coupling) == 4
    assert all(len(row) == 4 for row in coupling)
    assert scenario["t_max"] == 2.0
    assert scenario["n_steps"] == 20
    assert scenario["control"] == ("computational_basis_dephasing_with_identical_populations")


def test_payload_passes_all_bounded_classifications(payload: dict[str, object]) -> None:
    """Verify replay, controls, literature, language, and execution boundaries."""
    assert payload["schema"] == ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA
    assert payload["claim_boundary"] == ENTANGLEMENT_SYNC_CLAIM_BOUNDARY
    assert payload["literature"] == [dict(item) for item in ENTANGLEMENT_SYNC_LITERATURE]
    assert payload["deterministic_replay"] is True
    assert payload["functional_passed"] is True
    assert payload["state_family_count"] == 4
    assert payload["population_matched_controls"] is True
    assert payload["separable_attribution_control_present"] is True
    assert payload["entanglement_specific_effect_supported"] is False
    assert payload["critical_coupling_claimed"] is False
    assert payload["quantum_advantage_claimed"] is False
    assert payload["provider_execution"] is False
    assert payload["hardware_execution"] is False
    assert all(cast(dict[str, bool], payload["classification"]).values())
    assert validate_entanglement_sync_evidence(payload) == ()


def test_payload_records_positive_negative_and_attribution_controls(
    payload: dict[str, object],
) -> None:
    """Keep Bell/W observations, GHZ negative control, and separable confounder."""
    comparisons = cast(dict[str, dict[str, object]], payload["comparisons"])
    assert set(comparisons) == {"product", "bell_pairs", "ghz", "w_state"}
    assert comparisons["product"]["delta_mean_exchange_coherence"] == pytest.approx(0.265045126908)
    assert comparisons["bell_pairs"]["delta_mean_exchange_coherence"] == pytest.approx(
        0.0410195954932
    )
    assert comparisons["ghz"]["delta_mean_exchange_coherence"] == 0.0
    assert comparisons["w_state"]["delta_mean_exchange_coherence"] == pytest.approx(0.366405311833)
    assert all(
        row["entanglement_specific_effect_supported"] is False for row in comparisons.values()
    )


def test_payload_replay_and_canonical_json_are_stable(payload: dict[str, object]) -> None:
    """Reproduce canonical comparison bytes and the top-level digest."""
    replay = entanglement_sync_evidence_payload()
    assert canonical_entanglement_sync_json(replay) == canonical_entanglement_sync_json(payload)
    assert canonical_entanglement_sync_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_markdown_reports_measured_rows_and_boundary(payload: dict[str, object]) -> None:
    """Render all controls without causal, critical-coupling, or advantage promotion."""
    markdown = render_entanglement_sync_evidence_markdown(payload)
    assert "| product | 0 |" in markdown
    assert "| bell_pairs | 1 |" in markdown
    assert "| ghz | 1 | 0 | 0 | 0 | 0 |" in markdown
    assert "| w_state | 0.75 |" in markdown
    assert "not attributable uniquely to entanglement" in markdown
    assert (
        "cannot establish spontaneous synchronisation or a shifted critical coupling" in markdown
    )
    assert ENTANGLEMENT_SYNC_CLAIM_BOUNDARY in markdown


def test_writer_and_main_emit_atomic_valid_files(
    payload: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write supplied evidence and execute the public CLI generation path."""
    supplied_json = tmp_path / "supplied" / "evidence.json"
    supplied_markdown = tmp_path / "supplied" / "evidence.md"
    written = write_entanglement_sync_evidence(
        supplied_json,
        supplied_markdown,
        payload=payload,
    )
    assert written == payload
    assert json.loads(supplied_json.read_text(encoding="utf-8")) == payload
    assert supplied_markdown.read_text(encoding="utf-8").startswith("# Bounded entanglement-sync")
    assert not supplied_json.with_suffix(".json.tmp").exists()

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
    assert "entanglement-specific/critical-coupling/advantage/hardware claims=false" in output
    generated = json.loads(generated_json.read_text(encoding="utf-8"))
    assert validate_entanglement_sync_evidence(generated) == ()


def test_repository_runner_executes_real_cli(tmp_path: Path) -> None:
    """Exercise the repository script in a clean output directory."""
    repository = Path(__file__).parents[1]
    runner = repository / "scripts" / "run_entanglement_sync_evidence.py"
    json_path = tmp_path / "runner.json"
    markdown_path = tmp_path / "runner.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "functional_passed=true" in completed.stdout
    assert (
        validate_entanglement_sync_evidence(json.loads(json_path.read_text(encoding="utf-8")))
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema": "wrong"}, "schema"),
        ({"claim_boundary": "promoted"}, "claim_boundary"),
        ({"deterministic_replay": False}, "deterministic_replay"),
        ({"functional_passed": False}, "functional_passed"),
        ({"state_family_count": 3}, "state_family_count"),
        ({"population_matched_controls": False}, "population_matched_controls"),
        ({"separable_attribution_control_present": False}, "separable_attribution_control"),
        ({"entanglement_specific_effect_supported": True}, "entanglement_specific"),
        ({"critical_coupling_claimed": True}, "critical_coupling_claimed"),
        ({"quantum_advantage_claimed": True}, "quantum_advantage_claimed"),
        ({"provider_execution": True}, "provider_execution"),
        ({"hardware_execution": True}, "hardware_execution"),
        ({"literature": []}, "literature"),
        ({"scenario": {}}, "scenario"),
        ({"comparisons": {}}, "four initial-state"),
        ({"content_digest": "0" * 64}, "content_digest"),
    ],
)
def test_validator_rejects_top_level_drift(
    payload: dict[str, object],
    mutation: dict[str, object],
    message: str,
) -> None:
    """Reject promotion, execution, source, scenario, structure, and digest drift."""
    changed = copy.deepcopy(payload)
    changed.update(mutation)
    assert any(message in finding for finding in validate_entanglement_sync_evidence(changed))


def test_validator_rejects_stale_coded_schema(payload: dict[str, object]) -> None:
    """Reject the superseded schema instead of retaining a compatibility alias."""
    stale = copy.deepcopy(payload)
    stale["schema"] = "entanglement_initial_state_evidence.v1"
    assert validate_entanglement_sync_evidence(stale) == (
        f"schema must equal {ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA!r}",
        "content_digest does not match canonical payload bytes",
    )


def test_validator_rejects_malformed_and_reclassified_comparisons(
    payload: dict[str, object],
) -> None:
    """Fail closed when measured rows or their classifications are altered."""
    malformed = copy.deepcopy(payload)
    comparisons = cast(dict[str, dict[str, object]], malformed["comparisons"])
    comparisons["product"]["delta_mean_exchange_coherence"] = "wrong"
    assert any(
        "malformed classification" in finding
        for finding in validate_entanglement_sync_evidence(malformed)
    )

    reclassified = copy.deepcopy(payload)
    cast(dict[str, bool], reclassified["classification"])["product_is_separable"] = False
    assert any(
        "classification must match" in finding
        for finding in validate_entanglement_sync_evidence(reclassified)
    )

    failed = copy.deepcopy(payload)
    failed_comparisons = cast(dict[str, dict[str, object]], failed["comparisons"])
    failed_comparisons["product"]["initial_mean_single_qubit_linear_entropy"] = 1.0
    assert any(
        "every preregistered classification" in finding
        for finding in validate_entanglement_sync_evidence(failed)
    )


def test_validator_rejects_wrong_root_and_noncanonical_value(
    payload: dict[str, object],
) -> None:
    """Reject non-object roots and values that canonical JSON cannot encode."""
    assert validate_entanglement_sync_evidence([]) == ("payload must be a JSON object",)
    changed = copy.deepcopy(payload)
    changed["unexpected"] = {"not", "json"}
    assert any(
        "non-canonical" in finding for finding in validate_entanglement_sync_evidence(changed)
    )


def test_writer_rejects_invalid_payload(payload: dict[str, object], tmp_path: Path) -> None:
    """Refuse to persist evidence after a claim-boundary mutation."""
    changed = copy.deepcopy(payload)
    changed["entanglement_specific_effect_supported"] = True
    with pytest.raises(RuntimeError, match="invalid entanglement-sync evidence"):
        write_entanglement_sync_evidence(
            tmp_path / "bad.json",
            tmp_path / "bad.md",
            payload=changed,
        )


def test_markdown_rejects_non_numeric_row(payload: dict[str, object]) -> None:
    """Reject malformed human-readable numeric fields."""
    changed = copy.deepcopy(payload)
    comparisons = cast(dict[str, dict[str, object]], changed["comparisons"])
    comparisons["product"]["mean_exchange_coherence"] = True
    with pytest.raises(ValueError, match="mean_exchange_coherence must be numeric"):
        render_entanglement_sync_evidence_markdown(changed)
