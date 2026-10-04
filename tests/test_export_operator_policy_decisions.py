# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native synthetic policy export boundary
"""Qualify actual core decisions, exclusive exports and check-mode custody."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from scpn_quantum_control.studio.workspace import read_json
from tools.export_operator_policy_decisions import build_operator_policy_example, main


@pytest.mark.parametrize(
    "case,reason",
    [
        ("unknown_price", "price_unknown"),
        ("at_ceiling", None),
        ("over_shots", "shots_ceiling"),
        ("over_cost", "cost_ceiling"),
        ("wrong_region", "region_forbidden"),
        ("expired_policy", "policy_expired"),
        ("over_concurrency", "concurrency_ceiling"),
        ("over_time", "time_limit_ms_ceiling"),
    ],
)
def test_native_export_literal_oracles(case: str, reason: str | None) -> None:
    """Independent expected refusals agree with the actual public native assessment."""
    envelope = build_operator_policy_example(case)
    body = envelope["body"]
    assert isinstance(body, Mapping)
    decision = body["decision"]
    assert isinstance(decision, Mapping)
    reasons = decision["reasons"]
    assert isinstance(reasons, list)
    assert decision["allowed"] is (reason is None)
    assert reason is None and reasons == [] or reason in reasons
    assert body["no_submit"] is True


def test_exclusive_cli_export_and_readonly_check(tmp_path: Path) -> None:
    """Exact bytes survive check, stale checks and refused overwrite attempts."""
    output = tmp_path / "decision.json"
    args = ["--output", str(output)]
    assert main([*args, "--check"]) == 1
    assert not output.exists()
    assert main(args) == 0
    original = output.read_bytes()
    assert main([*args, "--check"]) == 0
    assert main([*args, "--case", "at_ceiling", "--check"]) == 1
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == original
    restored = read_json(original.decode())
    assert isinstance(restored, Mapping)
    assert restored["schema"] == "studio.operator-policy-decision.v1"


def test_unknown_example_refuses() -> None:
    """No unknown named scenario silently becomes a passing default."""
    with pytest.raises(ValueError, match="unknown"):
        build_operator_policy_example("future")


def test_actual_cli_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The script entry invokes its native producer and writes one exact snapshot."""
    output = tmp_path / "from-command.json"
    monkeypatch.setattr(
        sys, "argv", ["export_operator_policy_decisions.py", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as exit_value:
        runpy.run_path("tools/export_operator_policy_decisions.py", run_name="__main__")
    assert exit_value.value.code == 0
    assert main(["--output", str(output), "--check"]) == 0
