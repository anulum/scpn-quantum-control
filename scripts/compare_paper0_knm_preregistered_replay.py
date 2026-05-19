#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm preregistered replay comparator
"""Compare regenerated Paper 0 K_nm preregistered replay artefacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "run_paper0_knm_preregistered_replay.py"
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "paper0_knm_preregistered_replay.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "paper0_knm_preregistered_replay.md"
DEFAULT_ACTUAL_JSON_NAME = "paper0_knm_preregistered_replay.json"
DEFAULT_ACTUAL_MARKDOWN_NAME = "paper0_knm_preregistered_replay.md"
_GENERATOR = None


def _generator_module():
    """Load the replay generator without requiring scripts to be a package."""

    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    spec = importlib.util.spec_from_file_location(
        "_paper0_knm_preregistered_replay_generator",
        GENERATOR_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load replay generator: {GENERATOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _GENERATOR = module
    return module


def compare_paper0_knm_preregistered_replay(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path | None = None,
    actual_markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Compare committed Paper 0 K_nm replay artefacts with generated output."""

    blockers: list[str] = []
    if (actual_json_path is None) != (actual_markdown_path is None):
        raise ValueError("--actual-json and --actual-markdown must be provided together")

    if actual_json_path is not None and actual_markdown_path is not None:
        return _comparison_payload(
            expected_json_path=expected_json_path,
            expected_markdown_path=expected_markdown_path,
            actual_json_path=actual_json_path,
            actual_markdown_path=actual_markdown_path,
            actual_json_digest=_sha256_path(actual_json_path),
            actual_markdown_digest=_sha256_path(actual_markdown_path),
            blockers=blockers,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        actual_json = tmp_dir / DEFAULT_ACTUAL_JSON_NAME
        actual_markdown = tmp_dir / DEFAULT_ACTUAL_MARKDOWN_NAME
        _generator_module().write_replay_artifacts(
            output_json=actual_json,
            output_doc=actual_markdown,
        )
        return _comparison_payload(
            expected_json_path=expected_json_path,
            expected_markdown_path=expected_markdown_path,
            actual_json_path=actual_json,
            actual_markdown_path=actual_markdown,
            actual_json_digest=_sha256_path(actual_json),
            actual_markdown_digest=_sha256_path(actual_markdown),
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
        "schema": _generator_module().SCHEMA,
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
        if expected_json.get("schema") != _generator_module().SCHEMA:
            blockers.append("Paper 0 K_nm replay schema mismatch")
        _validate_replay_payload(expected_json, blockers, label="expected")
        _validate_replay_payload(actual_json, blockers, label="actual")
        if _normalised_json(expected_json) != _normalised_json(actual_json):
            blockers.append("Paper 0 K_nm replay JSON artifacts differ from committed version")

    if expected_markdown != actual_markdown:
        blockers.append("Paper 0 K_nm replay Markdown artifacts differ from committed version")

    if not expected_json_path.exists() or not actual_json_path.exists():
        blockers.append(
            "missing committed or generated Paper 0 K_nm replay JSON: "
            f"{expected_json_path} or {actual_json_path}"
        )
    if not expected_markdown_path.exists() or not actual_markdown_path.exists():
        blockers.append(
            "missing committed or generated Paper 0 K_nm replay Markdown: "
            f"{expected_markdown_path} or {actual_markdown_path}"
        )
    return not blockers


def _validate_replay_payload(payload: dict[str, Any], blockers: list[str], *, label: str) -> None:
    """Validate non-promotional replay invariants independent of byte equality."""

    if payload.get("status") != "blocked_non_closing_preregistered_replay":
        blockers.append(f"{label} replay status must remain blocked and non-closing")

    _validate_input_manifest(payload, blockers, label=label)

    claim_boundary = str(payload.get("claim_boundary", ""))
    if "does not authorise hardware submission" not in claim_boundary:
        blockers.append(f"{label} replay claim boundary must block hardware submission")

    decision = payload.get("promotion_decision")
    if not isinstance(decision, dict):
        blockers.append(f"{label} replay promotion_decision must be present")
        return
    if decision.get("decision") != "do_not_promote":
        blockers.append(f"{label} replay promotion decision must be do_not_promote")
    if decision.get("hardware_submission_authorised") is not False:
        blockers.append(f"{label} replay must not authorise hardware submission")
    if decision.get("claim_promotion_authorised") is not False:
        blockers.append(f"{label} replay must not authorise claim promotion")

    blocking_gates = decision.get("blocking_gates")
    if not isinstance(blocking_gates, dict) or "qpu_submission" not in blocking_gates:
        blockers.append(f"{label} replay decision must preserve qpu_submission blocking gate")
    required = decision.get("required_evidence_before_reconsideration")
    if not isinstance(required, list) or len(required) < 4:
        blockers.append(
            f"{label} replay decision must list required evidence before reconsideration"
        )
    falsifiers = decision.get("falsifiers")
    if not isinstance(falsifiers, list) or len(falsifiers) < 4:
        blockers.append(f"{label} replay decision must list falsifiers")


def _validate_input_manifest(payload: dict[str, Any], blockers: list[str], *, label: str) -> None:
    """Validate replay input digests against repository files."""

    reproducibility = payload.get("reproducibility")
    if not isinstance(reproducibility, dict):
        blockers.append(f"{label} replay reproducibility block must be present")
        return
    manifest = reproducibility.get("input_manifest")
    if not isinstance(manifest, dict):
        blockers.append(f"{label} replay input_manifest must be present")
        return

    required_inputs = {
        "primary_candidate",
        "negative_control",
        "negative_measured_couplings",
    }
    if set(manifest) != required_inputs:
        blockers.append(f"{label} replay input_manifest keys must match locked inputs")
        return

    for name, entry in manifest.items():
        if not isinstance(entry, dict):
            blockers.append(f"{label} replay input manifest entry {name} must be an object")
            continue
        relative_path = entry.get("path")
        expected_digest = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            blockers.append(
                f"{label} replay input manifest entry {name} must expose path and sha256"
            )
            continue
        input_path = (REPO_ROOT / relative_path).resolve()
        if not input_path.is_relative_to(REPO_ROOT):
            blockers.append(
                f"{label} replay input manifest path escapes repository: {relative_path}"
            )
            continue
        if not input_path.exists():
            blockers.append(f"{label} replay input manifest path is missing: {relative_path}")
            continue
        actual_digest = _sha256_path(input_path)
        if actual_digest != expected_digest:
            blockers.append(f"{label} replay input manifest digest mismatch for {name}")


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
    """CLI entry point for Paper 0 K_nm replay comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--expected-markdown", type=Path, default=DEFAULT_EXPECTED_MARKDOWN)
    parser.add_argument("--actual-json", type=Path)
    parser.add_argument("--actual-markdown", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON payload")
    args = parser.parse_args(argv)

    payload = compare_paper0_knm_preregistered_replay(
        expected_json_path=args.expected_json,
        expected_markdown_path=args.expected_markdown,
        actual_json_path=args.actual_json,
        actual_markdown_path=args.actual_markdown,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Paper 0 K_nm preregistered replay comparison valid: {payload['valid']}")
        for blocker in payload["blockers"]:
            print(f"  blocker: {blocker}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
