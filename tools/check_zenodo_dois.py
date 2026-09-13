# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control - Zenodo DOI registry checker
"""Verify Markdown Zenodo DOI references against the DataCite registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DATACITE_ENDPOINT = "https://api.datacite.org/dois/{doi}"
ZENODO_DOI_RE = re.compile(
    r"https://(?:doi\.org/|zenodo\.org/(?:badge/DOI/|doi/))"
    r"(10\.5281/zenodo\.\d+)(?:\.svg)?",
    re.IGNORECASE,
)
SKIPPED_DIRECTORIES = frozenset({".git", ".coordination", "paper", "site"})


def extract_zenodo_dois(text: str) -> set[str]:
    """Return normalized Zenodo DOI identifiers referenced by URL in text."""
    return {match.group(1).lower() for match in ZENODO_DOI_RE.finditer(text)}


def collect_zenodo_dois(root: Path) -> tuple[str, ...]:
    """Collect unique Zenodo DOI identifiers from link-checked Markdown files."""
    dois: set[str] = set()
    for path in root.rglob("*.md"):
        relative = path.relative_to(root)
        if any(
            part in SKIPPED_DIRECTORIES or part.startswith(".venv") for part in relative.parts[:-1]
        ):
            continue
        dois.update(extract_zenodo_dois(path.read_text(encoding="utf-8")))
    return tuple(sorted(dois))


def _fetch_datacite_record(doi: str, *, timeout: float, retries: int) -> Mapping[str, Any]:
    """Fetch a DataCite DOI record with bounded retries."""
    request = Request(
        DATACITE_ENDPOINT.format(doi=doi),
        headers={"Accept": "application/vnd.api+json", "User-Agent": "scpn-link-check/1"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
                payload = json.load(response)
            if not isinstance(payload, Mapping):
                raise ValueError("DataCite response is not a JSON object")
            return payload
        except (OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 5))
    raise RuntimeError(f"DataCite request failed after {retries} attempt(s): {last_error}")


def validate_datacite_record(doi: str, payload: Mapping[str, Any]) -> None:
    """Require the requested DOI to be active and publicly findable."""
    data = payload.get("data")
    attributes = data.get("attributes") if isinstance(data, Mapping) else None
    if not isinstance(attributes, Mapping):
        raise ValueError("DataCite response has no attributes object")

    registered_doi = str(attributes.get("doi", "")).lower()
    if registered_doi != doi.lower():
        raise ValueError(f"DataCite returned DOI {registered_doi!r}")
    if attributes.get("isActive") is not True:
        raise ValueError("DOI is not active")
    if attributes.get("state") != "findable":
        raise ValueError(f"DOI state is {attributes.get('state')!r}, not 'findable'")


def check_zenodo_dois(root: Path, *, timeout: float = 20, retries: int = 3) -> list[str]:
    """Check all discovered DOI records and return human-readable failures."""
    failures: list[str] = []
    dois = collect_zenodo_dois(root)
    for doi in dois:
        try:
            payload = _fetch_datacite_record(doi, timeout=timeout, retries=retries)
            validate_datacite_record(doi, payload)
        except (RuntimeError, ValueError) as exc:
            failures.append(f"{doi}: {exc}")
        else:
            print(f"[ok] {doi}")
    print(f"Checked {len(dois)} unique Zenodo DOI record(s).")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 1:
        parser.error("--retries must be at least 1")

    failures = check_zenodo_dois(args.root, timeout=args.timeout, retries=args.retries)
    for failure in failures:
        print(f"[error] {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
