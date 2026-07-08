# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed Kuramoto scenario artefact tests
"""Tests for the committed Kuramoto Play scenario artefact (ST-11)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scpn_quantum_control.studio import kuramoto_scenario_artifact as artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_JSON = REPO_ROOT / artifact.DEFAULT_KURAMOTO_SCENARIO_JSON_PATH


def test_build_has_the_expected_shape() -> None:
    """The payload carries scenario, packed input, and expectation blocks."""
    payload = artifact.build_kuramoto_scenario_artifact()
    assert payload["schema"] == artifact.KURAMOTO_SCENARIO_SCHEMA
    assert payload["artifact_id"] == artifact.KURAMOTO_SCENARIO_ARTIFACT_ID
    assert payload["boundaries"] == {"max_oscillators": 128, "max_steps": 4096}
    assert payload["scenario"]["mode"] == "mean-field"
    assert isinstance(payload["input_hex"], str) and payload["input_hex"]
    r_series = payload["expected"]["order_parameter"]
    assert len(r_series) == payload["scenario"]["steps"] + 1
    assert len(payload["expected"]["theta_final"]) == payload["scenario"]["n"]


def test_build_is_deterministic() -> None:
    """Two builds on the same host are byte-identical."""
    assert artifact.build_kuramoto_scenario_artifact() == (
        artifact.build_kuramoto_scenario_artifact()
    )


def test_committed_artifact_is_current() -> None:
    """The committed JSON validates against a fresh build within tolerance."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    assert artifact.validate_kuramoto_scenario_artifact(committed)


def test_validation_rejects_reproducible_block_drift() -> None:
    """A change to a byte-reproducible field (the scenario) fails validation."""
    payload = artifact.build_kuramoto_scenario_artifact()
    payload["scenario"]["coupling"] = 9.9
    assert not artifact.validate_kuramoto_scenario_artifact(payload)


def test_validation_rejects_a_non_dict_expectation() -> None:
    """A payload whose expectation is not a mapping fails validation."""
    payload = artifact.build_kuramoto_scenario_artifact()
    payload["expected"] = "not-a-dict"
    assert not artifact.validate_kuramoto_scenario_artifact(payload)


def test_validation_rejects_a_wrong_length_expectation() -> None:
    """A truncated expectation array fails validation."""
    payload = artifact.build_kuramoto_scenario_artifact()
    payload["expected"]["order_parameter"] = payload["expected"]["order_parameter"][:-1]
    assert not artifact.validate_kuramoto_scenario_artifact(payload)


def test_validation_rejects_an_expectation_that_left_tolerance() -> None:
    """An order-parameter sample off by far more than the ULP floor fails."""
    payload = artifact.build_kuramoto_scenario_artifact()
    payload["expected"]["order_parameter"][0] += 1e-3
    assert not artifact.validate_kuramoto_scenario_artifact(payload)


def test_validation_tolerates_sub_tolerance_drift() -> None:
    """A last-ULP-scale expectation drift (the cross-platform sin/cos case) passes."""
    payload = artifact.build_kuramoto_scenario_artifact()
    payload["expected"]["order_parameter"][0] += 1e-15
    payload["expected"]["theta_final"][0] += 1e-15
    assert artifact.validate_kuramoto_scenario_artifact(payload)


def test_main_check_passes_on_the_committed_artifact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI check mode confirms the committed artefact is current."""
    assert artifact.main(["--check"]) == 0
    captured = capsys.readouterr()
    assert "current" in captured.out
    assert captured.err == ""


def test_main_check_fails_on_drift(tmp_path: Path) -> None:
    """The CLI check mode reports drift and exits 1."""
    drifted = tmp_path / "scenario.json"
    payload = copy.deepcopy(artifact.build_kuramoto_scenario_artifact())
    payload["expected"]["order_parameter"][5] += 1.0
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(drifted)]) == 1


def test_main_write_and_default_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write mode emits the artefact and default mode prints it."""
    json_path = tmp_path / "nested" / "scenario.json"
    assert artifact.main(["--write", "--json-path", str(json_path)]) == 0
    assert json_path.exists()
    assert artifact.main(["--check", "--json-path", str(json_path)]) == 0
    capsys.readouterr()
    assert artifact.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["artifact_id"] == artifact.KURAMOTO_SCENARIO_ARTIFACT_ID
