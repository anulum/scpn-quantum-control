# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the committed pulse instrument data
"""The pulse instrument data comes from its source owner and stays current."""

from __future__ import annotations

import json
import runpy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.hardware.pulse_feasibility import DesignPulse
from scpn_quantum_control.studio import pulse_instrument_artifact as artifact


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build the instrument data once.

    Returns
    -------
    dict
        Freshly built payload.

    """
    return cast(dict[str, Any], artifact.build_pulse_instrument_artifact())


def _cases(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the cases by identifier."""
    return {case["case_id"]: case for case in payload["cases"]}


def _instant(text: str) -> datetime:
    """Parse an ISO UTC second independently of the owner's parser."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_pulse_analog_instrument_01_violations_name_their_source_span(
    payload: dict[str, Any],
) -> None:
    """A half-grid start and an over-amplitude segment are tied to their pulses."""
    case = _cases(payload)["half-grid-and-over-amplitude"]
    limits = case["limits"]
    starts = [pulse["start_seconds"] / limits["dt_seconds"] for pulse in case["design"]]
    off_grid = [
        pulse["index"]
        for pulse, start in zip(case["design"], starts, strict=True)
        if round(start) % limits["granularity_samples"] != 0
    ]
    over = [
        pulse["index"]
        for pulse in case["design"]
        if max(pulse["rates_rad_per_second"]) / limits["rabi_rate_per_amplitude"]
        > limits["max_amplitude"]
    ]
    assert off_grid == [0] and over == [1]
    by_code = {violation["code"]: violation for violation in case["violations"]}
    assert by_code["off_grid"]["pulse"] == 0
    assert by_code["amplitude_over_limit"]["pulse"] == 1
    assert all(
        violation["text"].startswith(violation["code"] + ": ") for violation in case["violations"]
    )
    assert case["feasible"] is False and case["native"] == []
    assert case["dossier"] is None and "violations" in case["dossier_refusal"]
    disconnected = _cases(payload)["disconnected-coupler"]
    pair = sorted(item["physical"] for item in disconnected["placement"])
    assert pair not in [sorted(edge) for edge in disconnected["limits"]["coupling_map"]]
    assert [violation["code"] for violation in disconnected["violations"]] == [
        "disconnected_placement"
    ]


def test_pulse_analog_instrument_02_expired_target_invalidates_the_dossier(
    payload: dict[str, Any],
) -> None:
    """The same feasible design has a dossier for a current snapshot and none for an expired one."""
    cases = _cases(payload)
    assessed = _instant(payload["assessed_at"])
    for case_id, valid in (("current-snapshot", True), ("expired-snapshot", False)):
        case = cases[case_id]
        snapshot = case["snapshot"]
        independent = (
            _instant(snapshot["observed_at"]) <= assessed < _instant(snapshot["expires_at"])
        )
        assert independent is valid and snapshot["valid_at_assessment"] is valid
        assert case["feasible"] is True and case["limits"]["source"] == "provider_metadata"
        assert (case["dossier"] is not None) is valid
    assert cases["expired-snapshot"]["dossier_refusal"] == (
        "the capability snapshot expired at 2026-10-30T21:00:00Z, before the assessment at "
        "2026-10-31T12:00:00Z"
    )
    dossier = cases["current-snapshot"]["dossier"]
    assert dossier["provider_snapshot"] == {
        "provider": "example-provider",
        "backend_name": "example-pulse-backend",
    }


def test_pulse_analog_instrument_03_dossiers_submit_nothing(payload: dict[str, Any]) -> None:
    """Every dossier and case records no submission and no device authority."""
    for case in payload["cases"]:
        assert case["hardware_submission"] is False and case["device_authority"] is False
        if case["dossier"] is not None:
            assert case["dossier"]["submissions"] == 0
            assert case["dossier"]["hardware_submission"] is False
            assert case["dossier"]["device_authority"] is False
            assert case["dossier"]["claim_boundary"] == payload["design_claim_boundary"]


def test_pulse_analog_instrument_04_native_samples_follow_the_design_units(
    payload: dict[str, Any],
) -> None:
    """Native samples are the design rates divided by the declared conversions."""
    case = _cases(payload)["feasible-design"]
    limits = case["limits"]
    placement = {item["logical"]: item["physical"] for item in case["placement"]}
    for pulse, native in zip(case["design"], case["native"], strict=True):
        samples = round(pulse["duration_seconds"] / limits["dt_seconds"])
        per_segment = samples // len(pulse["rates_rad_per_second"])
        conversion = (
            limits["rabi_rate_per_amplitude"]
            if len(pulse["qubits"]) == 1
            else limits["coupler_rate_per_amplitude"]
        )
        expected = [
            value / conversion
            for value in pulse["rates_rad_per_second"]
            for _ in range(per_segment)
        ]
        assert native["samples"] == pytest.approx(expected, rel=1e-12)
        assert native["t0"] == round(pulse["start_seconds"] / limits["dt_seconds"])
        physical = [placement[qubit] for qubit in pulse["qubits"]]
        assert native["physical_qubits"] == physical
        channel = f"d{physical[0]}" if len(physical) == 1 else f"u{physical[0]}_{physical[1]}"
        assert native["channel"] == channel
    assert payload["claim_boundary"] == artifact.PULSE_INSTRUMENT_CLAIM_BOUNDARY
    assert "not a device-executable" in payload["design_claim_boundary"]


def test_violations_must_start_with_a_code() -> None:
    """A violation the owner did not write as ``code: detail`` is refused."""
    with pytest.raises(ValueError, match="does not start with a code"):
        artifact._violation("Unexpected text")
    assert artifact._violation("slew_over_limit: d0 steps 0.7 at sample 4")["channel"] == "d0"


def test_a_case_without_limits_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A case must declare limits or a snapshot."""
    monkeypatch.setattr(
        artifact,
        "PULSE_CASES",
        (
            artifact.PulseCase(
                "bare", "No limits", (DesignPulse((0,), 0.0, 8e-9, (1.0,)),), {0: 0}
            ),
        ),
    )
    with pytest.raises(ValueError, match="neither limits nor a snapshot"):
        artifact.build_pulse_instrument_artifact()


def test_main_writes_and_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    """The command line writes the payload and checks it as current.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    capsys
        Captured output.
    monkeypatch
        Replaces the build with the module's payload.
    payload
        Freshly built payload.

    """
    monkeypatch.setattr(artifact, "build_pulse_instrument_artifact", lambda: payload)
    target = tmp_path / "instrument.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    assert "pulse instrument data: current" in capsys.readouterr().out


def test_committed_data_is_current(payload: dict[str, Any]) -> None:
    """The committed file equals a fresh build within the budget."""
    committed = json.loads(
        (artifact.REPO_ROOT / artifact.DEFAULT_PULSE_INSTRUMENT_JSON_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert artifact.differences(committed, payload) == []


def test_module_runs_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m ... --check`` exits with the check's status.

    Parameters
    ----------
    monkeypatch
        Sets the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["pulse_instrument_artifact", "--check"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(
            "scpn_quantum_control.studio.pulse_instrument_artifact", run_name="__main__"
        )
    assert exited.value.code == 0
