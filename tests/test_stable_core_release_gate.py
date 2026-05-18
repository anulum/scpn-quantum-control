# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable-core release gate tests
"""Tests for stable-core release gate command order and fail-closed behavior."""

from __future__ import annotations

import sys

from scripts.run_stable_core_release_gate import (
    CAPABILITY_GATE_SCRIPT,
    CONTRACT_GATE_SCRIPT,
    build_stable_core_release_gate_commands,
    main,
)


def test_gate_helper_builds_commands_in_stable_order() -> None:
    """The release gate should run the capability gate before contract gate."""

    commands = build_stable_core_release_gate_commands()

    assert commands == (
        (sys.executable, str(CAPABILITY_GATE_SCRIPT)),
        (sys.executable, str(CONTRACT_GATE_SCRIPT)),
    )


def test_gate_main_calls_release_steps_in_order(monkeypatch) -> None:
    """Release gate commands must execute in fixed, ordered sequence."""

    executed_commands: list[tuple[str, ...]] = []

    def stub_run_command(command: tuple[str, ...]) -> None:
        executed_commands.append(command)

    monkeypatch.setattr("scripts.run_stable_core_release_gate.run_command", stub_run_command)

    result = main()

    assert result == 0
    assert tuple(executed_commands) == build_stable_core_release_gate_commands()


def test_gate_main_fails_closed_on_first_failure(monkeypatch) -> None:
    """A non-zero exit in the first command must stop and propagate as SystemExit."""

    executed_commands: list[tuple[str, ...]] = []

    def stub_run_command(command: tuple[str, ...]) -> None:
        executed_commands.append(command)
        if command[1] == str(CAPABILITY_GATE_SCRIPT):
            raise SystemExit(7)
        raise RuntimeError("unexpected command")

    monkeypatch.setattr("scripts.run_stable_core_release_gate.run_command", stub_run_command)

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 7
    else:
        raise AssertionError("expected SystemExit")

    assert tuple(executed_commands) == ((sys.executable, str(CAPABILITY_GATE_SCRIPT)),)
