# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Zenodo DOI registry checker tests
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
    """Supported Zenodo URLs are normalised to DOIs."""
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
    """Directories outside the link check are skipped."""
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
    """An invalid registry record is refused with its reason."""
    with pytest.raises(ValueError, match=message):
        validate_datacite_record("10.5281/zenodo.1", payload)


def test_main_reports_successful_registry_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command reports each DOI the registry finds."""
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
    """The command fails for a DOI that is not findable."""
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


class _FakeResponse:
    """A DataCite response with a status and a raw body."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        """Return the raw body."""
        return self._body


class _FakeConnection:
    """An HTTPS connection that replays scripted outcomes and records every request."""

    outcomes: list[_FakeResponse | Exception] = []
    opened: list[tuple[str, float]] = []
    requested: list[tuple[str, str, dict[str, str]]] = []
    closed: int = 0

    def __init__(self, host: str, timeout: float) -> None:
        _FakeConnection.opened.append((host, timeout))

    def request(self, method: str, path: str, headers: dict[str, str]) -> None:
        """Record one request."""
        _FakeConnection.requested.append((method, path, headers))

    def getresponse(self) -> _FakeResponse:
        """Return the next scripted response, or raise the next scripted error."""
        outcome = _FakeConnection.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        """Count one closed connection."""
        _FakeConnection.closed += 1


@pytest.fixture
def fake_https(monkeypatch: pytest.MonkeyPatch) -> type[_FakeConnection]:
    """Route the checker's HTTPS connections to the scripted fake and skip the retry sleep."""
    _FakeConnection.outcomes = []
    _FakeConnection.opened = []
    _FakeConnection.requested = []
    _FakeConnection.closed = 0
    monkeypatch.setattr(_MODULE.http.client, "HTTPSConnection", _FakeConnection)
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _seconds: None)
    return _FakeConnection


def test_fetch_asks_the_fixed_host_over_https(fake_https: type[_FakeConnection]) -> None:
    """A good answer is returned, from the DataCite host, for the DOI's path, and closed."""
    fake_https.outcomes = [_FakeResponse(200, b'{"data": {"id": "10.5281/zenodo.1"}}')]
    payload = _MODULE._fetch_datacite_record("10.5281/zenodo.1", timeout=7, retries=3)
    assert payload == {"data": {"id": "10.5281/zenodo.1"}}
    assert fake_https.opened == [("api.datacite.org", 7)]
    method, path, headers = fake_https.requested[0]
    assert (method, path) == ("GET", "/dois/10.5281/zenodo.1")
    assert headers["Accept"] == "application/vnd.api+json"
    assert fake_https.closed == 1


def test_fetch_retries_bad_status_and_protocol_errors(fake_https: type[_FakeConnection]) -> None:
    """A non-200 status and a protocol error are retried until a good answer arrives."""
    fake_https.outcomes = [
        _FakeResponse(503, b"busy"),
        _MODULE.http.client.RemoteDisconnected("dropped"),
        _FakeResponse(200, b'{"data": {}}'),
    ]
    assert _MODULE._fetch_datacite_record("10.5281/zenodo.2", timeout=1, retries=3) == {"data": {}}
    assert fake_https.closed == 3


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (_FakeResponse(404, b"missing"), "HTTP 404"),
        (_FakeResponse(200, b"[1, 2]"), "not a JSON object"),
        (_FakeResponse(200, b"not json"), "Expecting value"),
        (TimeoutError("slow"), "slow"),
    ],
)
def test_fetch_fails_after_its_retries(
    fake_https: type[_FakeConnection], outcome: _FakeResponse | Exception, message: str
) -> None:
    """Every attempt failing the same way ends in one error that names the last cause."""
    fake_https.outcomes = [outcome, outcome]
    with pytest.raises(RuntimeError, match=f"after 2 attempt\\(s\\): .*{message}"):
        _MODULE._fetch_datacite_record("10.5281/zenodo.3", timeout=1, retries=2)
    assert fake_https.closed == 2


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--timeout", "0"], "--timeout must be positive"),
        (["--retries", "0"], "--retries must be at least 1"),
    ],
)
def test_main_refuses_unusable_limits(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-positive timeout or a retry count below one is refused before any request."""
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
