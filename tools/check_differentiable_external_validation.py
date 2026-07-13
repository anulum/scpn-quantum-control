# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external-validation manifest gate
"""Validate or refresh the paired differentiable external-validation manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scpn_quantum_control.differentiable_external_validation import (  # noqa: E402
    DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH,
    DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH,
    build_external_validation_artifact_bundle,
    build_external_validation_environment_lock,
    load_external_validation_artifact_bundle,
    load_external_validation_environment_lock,
    render_external_validation_artifact_bundle_markdown,
    render_external_validation_environment_lock_markdown,
    validate_external_validation_artifact_bundle,
    validate_external_validation_environment_lock,
)

ENVIRONMENT_PATH = DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH
ENVIRONMENT_MARKDOWN_PATH = ENVIRONMENT_PATH.with_suffix(".md")
BUNDLE_PATH = DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH
BUNDLE_MARKDOWN_PATH = BUNDLE_PATH.with_suffix(".md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one deterministic JSON manifest."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refresh_manifests() -> None:
    """Refresh the environment pair before the bundle that hashes that pair."""
    environment = build_external_validation_environment_lock(repo_root=ROOT)
    _write_json(ENVIRONMENT_PATH, environment.to_dict())
    ENVIRONMENT_MARKDOWN_PATH.write_text(
        render_external_validation_environment_lock_markdown(environment) + "\n",
        encoding="utf-8",
    )
    bundle = build_external_validation_artifact_bundle(repo_root=ROOT)
    _write_json(BUNDLE_PATH, bundle.to_dict())
    BUNDLE_MARKDOWN_PATH.write_text(
        render_external_validation_artifact_bundle_markdown(bundle) + "\n",
        encoding="utf-8",
    )


def audit_manifests() -> tuple[str, ...]:
    """Return prefixed drift findings from both committed manifest pairs."""
    environment = load_external_validation_environment_lock(ENVIRONMENT_PATH)
    environment_result = validate_external_validation_environment_lock(
        environment,
        repo_root=ROOT,
    )
    bundle = load_external_validation_artifact_bundle(BUNDLE_PATH)
    bundle_result = validate_external_validation_artifact_bundle(bundle, repo_root=ROOT)
    return (
        *(f"environment: {error}" for error in environment_result.errors),
        *(f"bundle: {error}" for error in bundle_result.errors),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest gate, optionally refreshing both dependency-ordered pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh environment JSON/Markdown, then the dependent bundle pair.",
    )
    args = parser.parse_args(argv)
    if args.write:
        refresh_manifests()
    errors = audit_manifests()
    if errors:
        print("differentiable external-validation manifest gate: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("differentiable external-validation manifest gate: PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
