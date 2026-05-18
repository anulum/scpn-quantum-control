#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 lane registry comparator
"""Compare regenerated Paper 0 lane registry artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.lane_registry import (
    PAPER0_LANE_REGISTRY_SCHEMA,
    paper0_lane_registry_json,
    paper0_lane_registry_markdown,
    paper0_lane_registry_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "paper0_lane_registry.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "paper0_lane_registry.md"
DEFAULT_ACTUAL_JSON_NAME = "paper0_lane_registry.json"
DEFAULT_ACTUAL_MARKDOWN_NAME = "paper0_lane_registry.md"


def write_paper0_lane_registry(*, json_path: Path, markdown_path: Path) -> dict[str, str]:
    """Write deterministic Paper 0 lane registry artefacts."""

    payload = paper0_lane_registry_payload()
    json_text = paper0_lane_registry_json(payload)
    markdown_text = paper0_lane_registry_markdown(payload)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return {
        "json_sha256": _sha256(json_text),
        "markdown_sha256": _sha256(markdown_text),
    }


def compare_paper0_lane_registry(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path | None = None,
    actual_markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Compare committed Paper 0 lane registry artefacts with generated output."""

    blockers: list[str] = []
    if (actual_json_path is None) != (actual_markdown_path is None):
        raise ValueError("--actual-json and --actual-markdown must be provided together")

    if actual_json_path is not None and actual_markdown_path is not None:
        actual_json = actual_json_path
        actual_markdown = actual_markdown_path
        actual_json_digest = _sha256_path(actual_json)
        actual_markdown_digest = _sha256_path(actual_markdown)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            actual_json = tmp_dir / DEFAULT_ACTUAL_JSON_NAME
            actual_markdown = tmp_dir / DEFAULT_ACTUAL_MARKDOWN_NAME
            generated_digests = write_paper0_lane_registry(
                json_path=actual_json,
                markdown_path=actual_markdown,
            )
            return _comparison_payload(
                expected_json_path=expected_json_path,
                expected_markdown_path=expected_markdown_path,
                actual_json_path=actual_json,
                actual_markdown_path=actual_markdown,
                actual_json_digest=generated_digests["json_sha256"],
                actual_markdown_digest=generated_digests["markdown_sha256"],
                blockers=blockers,
            )

    return _comparison_payload(
        expected_json_path=expected_json_path,
        expected_markdown_path=expected_markdown_path,
        actual_json_path=actual_json,
        actual_markdown_path=actual_markdown,
        actual_json_digest=actual_json_digest,
        actual_markdown_digest=actual_markdown_digest,
        blockers=blockers,
    )


def _comparison_payload(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path,
    actual_markdown_path: Path,
    actual_json_digest: str,
    actual_markdown_digest: str,
    blockers: list[str],
) -> dict[str, Any]:
    valid = _comparison_result(
        expected_json_path=expected_json_path,
        expected_markdown_path=expected_markdown_path,
        actual_json_path=actual_json_path,
        actual_markdown_path=actual_markdown_path,
        blockers=blockers,
    )
    return {
        "schema": PAPER0_LANE_REGISTRY_SCHEMA,
        "valid": valid,
        "blockers": tuple(blockers),
        "digests": {
            "expected_json_sha256": _sha256_path(expected_json_path),
            "expected_markdown_sha256": _sha256_path(expected_markdown_path),
            "actual_json_sha256": actual_json_digest,
            "actual_markdown_sha256": actual_markdown_digest,
        },
        "expected_paths": {
            "json": str(expected_json_path),
            "markdown": str(expected_markdown_path),
        },
        "actual_paths": {
            "json": str(actual_json_path),
            "markdown": str(actual_markdown_path),
        },
    }


def _comparison_result(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path,
    actual_markdown_path: Path,
    blockers: list[str],
) -> bool:
    expected_json = _load_json(expected_json_path, blockers)
    actual_json = _load_json(actual_json_path, blockers)
    expected_markdown = _normalise_markdown(_load_text(expected_markdown_path, blockers))
    actual_markdown = _normalise_markdown(_load_text(actual_markdown_path, blockers))

    if expected_json is not None and actual_json is not None:
        if expected_json.get("schema") != PAPER0_LANE_REGISTRY_SCHEMA:
            blockers.append("Paper 0 lane registry schema mismatch")
        if _normalised_json(expected_json) != _normalised_json(actual_json):
            blockers.append("Paper 0 lane registry JSON artifacts differ from committed version")

    if expected_markdown != actual_markdown:
        blockers.append("Paper 0 lane registry Markdown artifacts differ from committed version")

    if not expected_json_path.exists() or not actual_json_path.exists():
        blockers.append(
            f"missing committed or generated Paper 0 lane registry JSON: {expected_json_path} or {actual_json_path}"
        )
    if not expected_markdown_path.exists() or not actual_markdown_path.exists():
        blockers.append(
            "missing committed or generated Paper 0 lane registry Markdown: "
            f"{expected_markdown_path} or {actual_markdown_path}"
        )
    return not blockers


def _normalised_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _normalise_markdown(text: str) -> str:
    return text.replace("\r\n", "\n")


def _load_text(path: Path, blockers: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        blockers.append(f"unable to read {path}: {exc}")
        return ""


def _load_json(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(_load_text(path, blockers))
    except json.JSONDecodeError as exc:
        blockers.append(f"{path} must contain valid JSON: {exc}")
        return None


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return _sha256(path.read_text(encoding="utf-8"))
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Paper 0 lane registry comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--expected-markdown", type=Path, default=DEFAULT_EXPECTED_MARKDOWN)
    parser.add_argument("--actual-json", type=Path)
    parser.add_argument("--actual-markdown", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON payload")
    args = parser.parse_args(argv)

    payload = compare_paper0_lane_registry(
        expected_json_path=args.expected_json,
        expected_markdown_path=args.expected_markdown,
        actual_json_path=args.actual_json,
        actual_markdown_path=args.actual_markdown,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Paper 0 lane registry comparison valid: {payload['valid']}")
        for blocker in payload["blockers"]:
            print(f"  blocker: {blocker}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
