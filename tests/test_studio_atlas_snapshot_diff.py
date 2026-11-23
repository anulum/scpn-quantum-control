# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — the Atlas snapshot diff record tool
"""Check the diff record tool on the committed pin and on every refusal."""

from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.atlas import (
    AtlasSnapshot,
    atlas_snapshot_digest,
    parse_atlas_snapshot,
    snapshot_to_wire,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from tools.studio_atlas_snapshot_diff import (
    PIN_SCHEMA,
    RECORD_DIGEST_SCHEMA,
    build_diff_record,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "data/studio/atlas_snapshot_20261122.json"
Json = dict[str, object]


def committed() -> AtlasSnapshot:
    """Parse the committed snapshot.

    Returns
    -------
    AtlasSnapshot
        The snapshot of the committed pin.

    """
    return parse_atlas_snapshot(json.loads(COMMITTED.read_text(encoding="utf-8"))["snapshot"])


def write_pin(path: Path, snapshot: AtlasSnapshot) -> Path:
    """Write a valid pin of ``snapshot``.

    Parameters
    ----------
    path : Path
        Output file.
    snapshot : AtlasSnapshot
        The snapshot.

    Returns
    -------
    Path
        The written pin.

    """
    pin = {
        "schema": PIN_SCHEMA,
        "snapshot": snapshot_to_wire(snapshot),
        "snapshot_digest": atlas_snapshot_digest(snapshot),
    }
    path.write_text(json.dumps(pin), encoding="utf-8")
    return path


def changed(tmp_path: Path) -> tuple[Path, str]:
    """Write a pin whose first sourced definition has another digest.

    Parameters
    ----------
    tmp_path : Path
        Output directory.

    Returns
    -------
    tuple[Path, str]
        The pin and the changed node's identifier.

    """
    wire = copy.deepcopy(snapshot_to_wire(committed()))
    nodes = cast(list[Json], wire["nodes"])
    target = next(item for item in nodes if item["kind"] == "definition" and item["source_digest"])
    target["source_digest"] = "0" * 64
    return write_pin(tmp_path / "after.json", parse_atlas_snapshot(wire)), str(target["id"])


def test_the_record_names_both_snapshots_and_seals_the_diff(tmp_path: Path) -> None:
    """The record holds both digests, the diff and a canonical digest over the rest.

    Parameters
    ----------
    tmp_path : Path
        Output directory.

    """
    after, node_id = changed(tmp_path)
    record = build_diff_record(COMMITTED, after)

    sealed = dict(record)
    digest = sealed.pop("record_digest")
    assert digest == canonical_digest(RECORD_DIGEST_SCHEMA, sealed)
    assert record["schema"] == "scpn_qc_studio_atlas_snapshot_diff_v1"
    before = cast(Json, record["before"])
    assert (
        before["snapshot_digest"]
        == json.loads(COMMITTED.read_text(encoding="utf-8"))["snapshot_digest"]
    )
    diff = cast(Json, record["diff"])
    nodes = cast(list[Json], diff["nodes"])
    assert [(item["id"], item["change"]) for item in nodes] == [(node_id, "changed")]
    assert diff["impact"]
    assert diff["complete"] is False


def test_the_cli_prints_or_writes_and_keeps_an_existing_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints the record, writes it once, and keeps an existing output.

    Parameters
    ----------
    tmp_path : Path
        Output directory.
    capsys : pytest.CaptureFixture[str]
        Captured output.

    """
    after, _ = changed(tmp_path)
    assert main(["--before", str(COMMITTED), "--after", str(after)]) == 0
    printed = json.loads(capsys.readouterr().out)
    output = tmp_path / "diff.json"
    assert main(["--before", str(COMMITTED), "--after", str(after), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == printed
    capsys.readouterr()
    assert main(["--before", str(COMMITTED), "--after", str(after), "--output", str(output)]) == 2
    assert "exists and is kept" in capsys.readouterr().err


def test_every_unadmitted_pin_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-JSON file, another schema, a broken contract and a wrong digest are refused.

    Parameters
    ----------
    tmp_path : Path
        Output directory.
    capsys : pytest.CaptureFixture[str]
        Captured output.

    """
    pin = json.loads(COMMITTED.read_text(encoding="utf-8"))
    cases = {
        "missing.json": None,
        "text.json": "{not json",
        "schema.json": json.dumps({**pin, "schema": "other"}),
        "contract.json": json.dumps({**pin, "snapshot": {}}),
        "digest.json": json.dumps({**pin, "snapshot_digest": "0" * 64}),
    }
    messages = {
        "missing.json": "not a readable JSON pin",
        "text.json": "not a readable JSON pin",
        "schema.json": f"not a {PIN_SCHEMA} pin",
        "contract.json": "the snapshot breaks the contract",
        "digest.json": "does not match its recorded digest",
    }
    for name, text in cases.items():
        path = tmp_path / name
        if text is not None:
            path.write_text(text, encoding="utf-8")
        assert main(["--before", str(path), "--after", str(COMMITTED)]) == 2
        assert messages[name] in capsys.readouterr().err, name


def test_snapshots_of_different_scopes_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A public pin is not compared with a private one.

    Parameters
    ----------
    tmp_path : Path
        Output directory.
    capsys : pytest.CaptureFixture[str]
        Captured output.

    """
    wire = copy.deepcopy(snapshot_to_wire(committed()))
    wire["access_scope"] = "private"
    private = write_pin(tmp_path / "private.json", parse_atlas_snapshot(wire))
    assert main(["--before", str(COMMITTED), "--after", str(private)]) == 2
    assert "cannot compare a public snapshot with a private one" in capsys.readouterr().err


def test_the_script_entry_point_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the tool as a script exits with the CLI status.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the process arguments.
    capsys : pytest.CaptureFixture[str]
        Captured output.

    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["studio_atlas_snapshot_diff.py", "--before", str(COMMITTED), "--after", str(COMMITTED)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / "tools/studio_atlas_snapshot_diff.py"), run_name="__main__")
    assert stopped.value.code == 0
    record = json.loads(capsys.readouterr().out)
    assert cast(Json, record["diff"])["nodes"] == []
