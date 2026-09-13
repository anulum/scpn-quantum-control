# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control - Zenodo DOI registry checker tests
"""Tests for the Zenodo DOI registry checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "check_zenodo_dois.py"
_SPEC = importlib.util.spec_from_file_location("check_zenodo_dois", _TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

collect_zenodo_dois = _MODULE.collect_zenodo_dois
extract_zenodo_dois = _MODULE.extract_zenodo_dois
main = _MODULE.main
validate_datacite_record = _MODULE.validate_datacite_record


def _record(doi: str, *, active: bool = True, state: str = "findable") -> dict[str, Any]:
    return {"data": {"attributes": {"doi": doi, "isActive": active, "state": state}}}


def test_extract_zenodo_dois_normalizes_supported_urls() -> None:
    text = """
    https://doi.org/10.5281/zenodo.20382000
    https://zenodo.org/doi/10.5281/ZENODO.20382000
    https://zenodo.org/badge/DOI/10.5281/zenodo.18821929.svg
    https://doi.org/10.1000/not-zenodo
    """

    assert extract_zenodo_dois(text) == {
        "10.5281/zenodo.18821929",
        "10.5281/zenodo.20382000",
    }


def test_collect_zenodo_dois_skips_non_link_checked_directories(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text(
        "https://doi.org/10.5281/zenodo.20382000", encoding="utf-8"
    )
    for directory in ("paper", "site", ".coordination", ".venv-linux"):
        target = tmp_path / directory
        target.mkdir()
        (target / "ignored.md").write_text(
            "https://doi.org/10.5281/zenodo.99999999", encoding="utf-8"
        )

    assert collect_zenodo_dois(tmp_path) == ("10.5281/zenodo.20382000",)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "no attributes"),
        (_record("10.5281/zenodo.2"), "returned DOI"),
        (_record("10.5281/zenodo.1", active=False), "not active"),
        (_record("10.5281/zenodo.1", state="registered"), "not 'findable'"),
    ],
)
def test_validate_datacite_record_fails_closed(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_datacite_record("10.5281/zenodo.1", payload)


def test_main_reports_successful_registry_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "README.md").write_text(
        "https://doi.org/10.5281/zenodo.20382000", encoding="utf-8"
    )
    monkeypatch.setattr(
        _MODULE,
        "_fetch_datacite_record",
        lambda doi, **_kwargs: _record(doi),
    )

    assert main([str(tmp_path), "--retries", "1"]) == 0
    output = capsys.readouterr()
    assert "[ok] 10.5281/zenodo.20382000" in output.out
    assert output.err == ""


def test_main_returns_nonzero_for_unfindable_registry_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "README.md").write_text(
        "https://doi.org/10.5281/zenodo.20382000", encoding="utf-8"
    )
    monkeypatch.setattr(
        _MODULE,
        "_fetch_datacite_record",
        lambda doi, **_kwargs: _record(doi, state="registered"),
    )

    assert main([str(tmp_path), "--retries", "1"]) == 1
    assert "not 'findable'" in capsys.readouterr().err
