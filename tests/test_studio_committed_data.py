# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the shared committed Studio data checks
"""Committed instrument data is serialised exactly and checked within the budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scpn_quantum_control.studio import committed_data


@pytest.mark.parametrize(
    ("committed", "fresh", "expected"),
    [
        (0.1 + 1e-12, 0.1, []),
        (1, 1.0, []),
        (0.2, 0.1, ["$: 0.2 is outside the budget of 0.1"]),
        (True, 1.0, ["$: True is outside the budget of 1.0"]),
        ("0.1", 0.1, ["$: '0.1' != 0.1"]),
        ({"a": 1}, {"b": 1}, ["$: keys ['a'] != ['b']"]),
        ([1], [1, 2], ["$: length 1 != 2"]),
        ({"a": [1, "x"]}, {"a": [1, "y"]}, ["$.a[1]: 'x' != 'y'"]),
        ({"a": [0.5, 2]}, {"a": [0.5, 2]}, []),
    ],
    ids=[
        "within-budget",
        "integer-for-float",
        "outside-budget",
        "bool-for-float",
        "text-for-float",
        "keys",
        "length",
        "nested",
        "equal",
    ],
)
def test_differences_name_every_departure(
    committed: object, fresh: object, expected: list[str]
) -> None:
    """Floats compare within the frozen budget; everything else exactly.

    Parameters
    ----------
    committed
        Committed value.
    fresh
        Freshly built value.
    expected
        Messages the comparison must produce.

    """
    assert committed_data.differences(committed, fresh) == expected


def test_serialise_is_sorted_indented_and_keeps_unicode() -> None:
    """The committed form is stable whatever the key order of the build."""
    text = committed_data.serialise({"b": "c0←q2", "a": [1, 2]})
    assert text == '{\n  "a": [\n    1,\n    2\n  ],\n  "b": "c0←q2"\n}\n'


def test_cli_prints_writes_and_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Print, write, then check current, drifted and unreadable files."""
    payload: dict[str, object] = {"value": 0.25, "label": "x"}
    target = tmp_path / "nested" / "data.json"

    def run(*argv: str) -> int:
        return committed_data.run_committed_data_cli(
            list(argv),
            build=lambda: payload,
            default_path=target,
            label="test data",
            description="test",
        )

    assert run() == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert not target.exists()
    assert run("--write") == 0
    assert target.read_text(encoding="utf-8") == committed_data.serialise(payload)
    assert "wrote" in capsys.readouterr().out
    assert run("--check") == 0
    assert capsys.readouterr().out == "test data: current\n"
    target.write_text(json.dumps({"value": 0.5, "label": "x"}), encoding="utf-8")
    assert run("--check") == 1
    assert "test data: drift: $.value: 0.5 is outside the budget of 0.25" in (
        capsys.readouterr().err
    )
    assert run("--check", "--json-path", str(tmp_path / "missing.json")) == 2
    assert "test data: unreadable" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run("--check", "--write")


def test_cli_reads_the_process_arguments_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an argument vector the process arguments decide the mode."""
    monkeypatch.setattr(sys, "argv", ["instrument", "--write"])
    target = tmp_path / "data.json"
    assert (
        committed_data.run_committed_data_cli(
            None, build=lambda: {"a": 1}, default_path=target, label="l", description=None
        )
        == 0
    )
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    capsys.readouterr()
