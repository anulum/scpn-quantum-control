# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed Atlas snapshot tool tests
"""Tests for the committed Atlas snapshot that the Studio browser loads."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.atlas import atlas_snapshot_digest, parse_atlas_snapshot

_ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    """Load the snapshot tool from its file.

    Returns
    -------
    ModuleType
        The loaded ``tools/studio_atlas_snapshot.py`` module.

    """
    spec = importlib.util.spec_from_file_location(
        "studio_atlas_snapshot_for_tests", _ROOT / "tools/studio_atlas_snapshot.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL = _tool()
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_the_committed_pin_is_current_and_valid() -> None:
    """The committed pin equals a fresh extraction and its snapshot passes the contract."""
    assert _TOOL.main(["--check"]) == 0
    pin = json.loads(_TOOL.DEFAULT_PATH.read_text(encoding="utf-8"))
    snapshot = parse_atlas_snapshot(pin["snapshot"])

    assert pin["schema"] == "scpn_qc_studio_atlas_snapshot_pin_v1"
    assert atlas_snapshot_digest(snapshot) == pin["snapshot_digest"]


def test_write_then_check_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A written pin checks clean, and a tampered one reports its difference."""
    target = tmp_path / "pin.json"

    assert _TOOL.main(["--write", "--source-commit", _COMMIT, "--json-path", str(target)]) == 0
    assert _TOOL.main(["--check", "--json-path", str(target)]) == 0
    pin = json.loads(target.read_text(encoding="utf-8"))
    assert pin["snapshot"]["source_commit"] == _COMMIT
    pin["snapshot_digest"] = "0" * 64
    target.write_text(json.dumps(pin), encoding="utf-8")
    capsys.readouterr()

    assert _TOOL.main(["--check", "--json-path", str(target)]) == 1
    assert "snapshot_digest" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--write"],
        ["--write", "--source-commit", "abc"],
        ["--check", "--source-commit", _COMMIT],
    ],
    ids=["write-without-commit", "write-short-commit", "commit-outside-write"],
)
def test_source_commit_rules_are_enforced(arguments: list[str], tmp_path: Path) -> None:
    """Writing needs a full commit; the commit option is refused outside writing."""
    assert _TOOL.main([*arguments, "--json-path", str(tmp_path / "pin.json")]) == 2


@pytest.mark.parametrize(
    "content",
    ["not json", json.dumps({"snapshot": {"source_commit": "short"}}), json.dumps([1])],
    ids=["not-json", "short-commit", "not-a-pin"],
)
def test_check_refuses_an_unreadable_pin(content: str, tmp_path: Path) -> None:
    """A pin without a readable full source commit cannot be checked."""
    target = tmp_path / "pin.json"
    target.write_text(content, encoding="utf-8")

    assert _TOOL.main(["--check", "--json-path", str(target)]) == 2
    assert _TOOL.main(["--check", "--json-path", str(tmp_path / "absent.json")]) == 2


def test_the_script_entry_point_checks_the_committed_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the file as a script checks the committed pin and exits with its status."""
    monkeypatch.setattr(sys, "argv", ["studio_atlas_snapshot.py", "--check"])

    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(_ROOT / "tools/studio_atlas_snapshot.py"), run_name="__main__")

    assert exited.value.code == 0
