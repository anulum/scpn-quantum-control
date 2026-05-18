#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core capability matrix comparator
"""Compare regenerated stable core capability matrix artefacts.

The comparator is deterministic: it regenerates stable-core capability JSON and
Markdown rows into a temporary directory (unless explicit `--actual-*` paths are
provided) and compares canonical payloads against committed artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scpn_quantum_control.stable_core import (
    normalised_stable_core_json,
    stable_core_capability_markdown,
    stable_core_capability_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "stable_core" / "backend_capability_matrix.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "stable_core_backend_capability_matrix.md"
DEFAULT_ACTUAL_JSON_NAME = "backend_capability_matrix.json"
DEFAULT_ACTUAL_MARKDOWN_NAME = "stable_core_backend_capability_matrix.md"
SCHEMA_VERSION = "stable_core_capability_matrix_comparison_v1"


def write_stable_core_capability_artifacts(
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, str]:
    """Write deterministic stable-core capability artefacts to the target paths."""

    payload = stable_core_capability_payload()
    json_text = normalised_stable_core_json(payload)
    markdown_text = stable_core_capability_markdown(payload)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return {
        "json_sha256": _sha256(json_text),
        "markdown_sha256": _sha256(markdown_text),
    }


def compare_stable_core_capability_matrix(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path | None = None,
    actual_markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Compare committed stable-core capability artefacts with generated ones."""

    blockers: list[str] = []

    if (actual_json_path is None) != (actual_markdown_path is None):
        raise ValueError("--actual-json and --actual-markdown must be provided together")

    if actual_json_path is not None and actual_markdown_path is not None:
        actual_json = actual_json_path
        actual_markdown = actual_markdown_path
        generated_digests = None
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            actual_json = tmp_dir / DEFAULT_ACTUAL_JSON_NAME
            actual_markdown = tmp_dir / DEFAULT_ACTUAL_MARKDOWN_NAME
            generated_digests = write_stable_core_capability_artifacts(
                json_path=actual_json,
                markdown_path=actual_markdown,
            )
            return {
                "schema": SCHEMA_VERSION,
                "valid": _comparison_result(
                    expected_json_path=expected_json_path,
                    expected_markdown_path=expected_markdown_path,
                    actual_json_path=actual_json,
                    actual_markdown_path=actual_markdown,
                    blockers=blockers,
                ),
                "blockers": tuple(blockers),
                "digests": {
                    "expected_json_sha256": _sha256_path(expected_json_path),
                    "expected_markdown_sha256": _sha256_path(expected_markdown_path),
                    "actual_json_sha256": generated_digests["json_sha256"],
                    "actual_markdown_sha256": generated_digests["markdown_sha256"],
                },
                "expected_paths": {
                    "json": str(expected_json_path),
                    "markdown": str(expected_markdown_path),
                },
                "actual_paths": {
                    "json": str(actual_json),
                    "markdown": str(actual_markdown),
                },
            }

    # Branch for explicit actual paths (mostly for tests and offline workflows).
    return {
        "schema": SCHEMA_VERSION,
        "valid": _comparison_result(
            expected_json_path=expected_json_path,
            expected_markdown_path=expected_markdown_path,
            actual_json_path=actual_json,
            actual_markdown_path=actual_markdown,
            blockers=blockers,
        ),
        "blockers": tuple(blockers),
        "digests": {
            "expected_json_sha256": _sha256_path(expected_json_path),
            "expected_markdown_sha256": _sha256_path(expected_markdown_path),
            "actual_json_sha256": _sha256_path(actual_json),
            "actual_markdown_sha256": _sha256_path(actual_markdown),
        },
        "expected_paths": {
            "json": str(expected_json_path),
            "markdown": str(expected_markdown_path),
        },
        "actual_paths": {
            "json": str(actual_json),
            "markdown": str(actual_markdown),
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
    """Populate blockers and return whether the artefacts match exactly."""

    valid = True
    expected_json = _load_json(expected_json_path, blockers)
    actual_json = _load_json(actual_json_path, blockers)
    expected_markdown = _normalise_markdown(_load_text(expected_markdown_path, blockers))
    actual_markdown = _normalise_markdown(_load_text(actual_markdown_path, blockers))

    if (
        expected_json is not None
        and actual_json is not None
        and _normalised_json(expected_json) != _normalised_json(actual_json)
    ):
        blockers.append("stable-core JSON matrix artefacts differ from committed version")
        valid = False

    if (
        expected_markdown is not None
        and actual_markdown is not None
        and expected_markdown != actual_markdown
    ):
        blockers.append("stable-core Markdown matrix artefacts differ from committed version")
        valid = False

    if not expected_json_path.exists() or not actual_json_path.exists():
        valid = False
    if not expected_markdown_path.exists() or not actual_markdown_path.exists():
        valid = False

    if blockers:
        valid = False

    return valid


def _normalise_markdown(text: str) -> str:
    """Normalise markdown line endings for deterministic comparisons."""

    return text.replace("\r\n", "\n")


def _normalised_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON canonical form for drift checks."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    """Load and decode JSON with blocker recording."""

    try:
        return json.loads(_load_text(path, blockers))
    except json.JSONDecodeError as exc:
        blockers.append(f"{path} must contain valid JSON: {exc}")
        return None


def _load_text(path: Path, blockers: list[str]) -> str:
    """Load file text with deterministic blocker reporting."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        blockers.append(f"unable to read {path}: {exc}")
        return ""


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for stable core capability matrix comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--expected-markdown", type=Path, default=DEFAULT_EXPECTED_MARKDOWN)
    parser.add_argument("--actual-json", type=Path)
    parser.add_argument("--actual-markdown", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON payload")
    args = parser.parse_args(argv)

    payload = compare_stable_core_capability_matrix(
        expected_json_path=args.expected_json,
        expected_markdown_path=args.expected_markdown,
        actual_json_path=args.actual_json,
        actual_markdown_path=args.actual_markdown,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"stable core capability matrix comparison valid: {payload['valid']}")
        for blocker in payload["blockers"]:
            print(f"  blocker: {blocker}")

    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
