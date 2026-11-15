# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark databank pin against independent digests
"""Check the benchmark databank pin, its refusals and its CLI."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import studio_benchmark_databank as databank


def _copy_sources(root: Path) -> None:
    """Copy both committed tier artefacts under an owned repository root."""
    for relative in databank.SOURCES.values():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((databank.ROOT / relative).read_bytes())


def _rewrite(root: Path, name: str, change: dict[str, Any]) -> None:
    """Rewrite one copied artefact with top-level fields replaced."""
    path = root / databank.SOURCES[name]
    artifact = json.loads(path.read_bytes())
    artifact.update(change)
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_benchmark_databank_01_pin_matches_independent_digests() -> None:
    """File and payload digests equal an independent recomputation of both artefacts."""
    built: dict[str, Any] = databank.build_benchmark_databank()
    sources: dict[str, dict[str, Any]] = built["sources"]
    assert set(sources) == {"ci", "local"}
    for name, pin in sources.items():
        raw = (databank.ROOT / databank.SOURCES[name]).read_bytes()
        artifact = json.loads(raw)
        body = {key: value for key, value in artifact.items() if key != "payload_sha256"}
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        assert pin["sha256"] == hashlib.sha256(raw).hexdigest()
        assert pin["payload_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert pin["generated_utc"] == artifact["generated_utc"]
        assert artifact["production_claim_allowed"] is False
    assert "re-baseline is pending" in built["boundary"]
    for row in built["ascii_oracle"]:
        assert row["python_text"] == json.dumps(
            row["content"], sort_keys=True, separators=(",", ":")
        )
        assert row["python_text"].isascii()
    assert "\\u007f" in built["ascii_oracle"][0]["python_text"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": "other.v1"}, "is not a scpn-quantum-control.tier-benchmark.v1"),
        ({"production_claim_allowed": True}, "allows a production claim"),
        ({"generated_utc": "2026-06-27T00:00:00Z"}, "payload digest does not match"),
    ],
)
def test_benchmark_databank_02_refuses_a_changed_artefact(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    """A foreign schema, a production claim or an edit under the old digest is refused.

    Parameters
    ----------
    tmp_path
        Owned repository root with copies of both artefacts.
    change
        Top-level fields replaced in the CI artefact.
    message
        Expected refusal text.

    """
    _copy_sources(tmp_path)
    assert databank.build_benchmark_databank(tmp_path) == databank.build_benchmark_databank()
    _rewrite(tmp_path, "ci", change)
    with pytest.raises(ValueError, match=message):
        databank.build_benchmark_databank(tmp_path)


def test_committed_data_and_command_line_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, check and refuse a changed pin through the public command line.

    Parameters
    ----------
    tmp_path
        Owned scratch output.
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert databank.main(["--check"]) == 0
    target = tmp_path / "databank.json"
    assert databank.main(["--write", "--json-path", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["sources"]["local"]["sha256"] = "0" * 64
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert databank.main(["--check", "--json-path", str(target)]) == 1
    monkeypatch.setattr(sys, "argv", ["databank", "--check"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(
            str(databank.ROOT / "tools/studio_benchmark_databank.py"), run_name="__main__"
        )
    assert exit_status.value.code == 0
