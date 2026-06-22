# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the symmetry-sector mitigation fixtures
"""Branch tests for the symmetry-sector mitigation fixture helpers.

Covers the JSON and text artifact writers and the fail-closed assertion that the
blocked replay fixture truly blocks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scpn_quantum_control.mitigation import symmetry_sector_fixtures as fixtures
from scpn_quantum_control.mitigation.symmetry_sector_fixtures import (
    normalised_json,
    replay_fixture_rows,
    write_json,
    write_text,
)


def test_write_json_persists_and_digests(tmp_path: Path) -> None:
    """The JSON writer persists deterministic content and returns its digest."""
    path = tmp_path / "nested" / "fixture.json"
    digest = write_json(path, {"b": 2, "a": 1})
    encoded = normalised_json({"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == encoded
    assert digest == hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_write_text_persists_and_digests(tmp_path: Path) -> None:
    """The text writer persists content and returns its digest."""
    path = tmp_path / "nested" / "fixture.md"
    digest = write_text(path, "hello\n")
    assert path.read_text(encoding="utf-8") == "hello\n"
    assert digest == hashlib.sha256(b"hello\n").hexdigest()


def test_replay_fixture_rows_asserts_blocked_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the blocked replay fixture unexpectedly succeeds the builder fails closed."""

    def _always_succeeds(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(status="ok", to_dict=lambda: {})

    monkeypatch.setattr(fixtures, "replay_symmetry_sector_counts", _always_succeeds)
    with pytest.raises(AssertionError, match="blocked replay fixture unexpectedly succeeded"):
        replay_fixture_rows()
