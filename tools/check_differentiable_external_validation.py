# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external-validation manifest gate
"""Validate or refresh the differentiable external-validation manifests and dependency map.

The environment pair pins the lockfiles and ``pyproject.toml``; the dependency map
pins the environment pair and further sources; the bundle pins the map. A refresh
therefore writes them in that order.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scpn_quantum_control.differentiable_dependency_environment_map import (
    render_differentiable_dependency_environment_map_markdown,
    run_differentiable_dependency_environment_map,
)
from scpn_quantum_control.differentiable_external_validation import (
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

ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT_PATH = DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH
ENVIRONMENT_MARKDOWN_PATH = ENVIRONMENT_PATH.with_suffix(".md")
BUNDLE_PATH = DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH
BUNDLE_MARKDOWN_PATH = BUNDLE_PATH.with_suffix(".md")
MAP_PATH = (
    ROOT
    / "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.json"
)
MAP_MARKDOWN_PATH = MAP_PATH.with_suffix(".md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one deterministic JSON manifest."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_dependency_map() -> None:
    """Write the dependency map pair, built from the environment pair on disk."""
    environment_map = run_differentiable_dependency_environment_map()
    MAP_PATH.write_text(json.dumps(environment_map.to_dict(), indent=2) + "\n", encoding="utf-8")
    MAP_MARKDOWN_PATH.write_text(
        render_differentiable_dependency_environment_map_markdown(environment_map),
        encoding="utf-8",
    )


def refresh_manifests() -> None:
    """Refresh the environment pair, then the dependency map, then the bundle that hashes both."""
    environment = build_external_validation_environment_lock(repo_root=ROOT)
    _write_json(ENVIRONMENT_PATH, environment.to_dict())
    ENVIRONMENT_MARKDOWN_PATH.write_text(
        render_external_validation_environment_lock_markdown(environment) + "\n",
        encoding="utf-8",
    )
    _write_dependency_map()
    bundle = build_external_validation_artifact_bundle(repo_root=ROOT)
    _write_json(BUNDLE_PATH, bundle.to_dict())
    BUNDLE_MARKDOWN_PATH.write_text(
        render_external_validation_artifact_bundle_markdown(bundle) + "\n",
        encoding="utf-8",
    )


def audit_dependency_map() -> tuple[str, ...]:
    """Return drift findings of the committed dependency map against the live map."""
    environment_map = run_differentiable_dependency_environment_map()
    findings: list[str] = []
    if json.loads(MAP_PATH.read_text(encoding="utf-8")) != environment_map.to_dict():
        findings.append(f"committed JSON differs from the live map: {MAP_PATH.name}")
    if MAP_MARKDOWN_PATH.read_text(
        encoding="utf-8"
    ) != render_differentiable_dependency_environment_map_markdown(environment_map):
        findings.append(f"committed Markdown differs from the live map: {MAP_MARKDOWN_PATH.name}")
    return tuple(findings)


def audit_manifests() -> tuple[str, ...]:
    """Return prefixed drift findings from both manifest pairs and the dependency map."""
    environment = load_external_validation_environment_lock(ENVIRONMENT_PATH)
    environment_result = validate_external_validation_environment_lock(
        environment,
        repo_root=ROOT,
    )
    bundle = load_external_validation_artifact_bundle(BUNDLE_PATH)
    bundle_result = validate_external_validation_artifact_bundle(bundle, repo_root=ROOT)
    return (
        *(f"environment: {error}" for error in environment_result.errors),
        *(f"dependency map: {error}" for error in audit_dependency_map()),
        *(f"bundle: {error}" for error in bundle_result.errors),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest gate, optionally refreshing the dependency-ordered artefacts first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh the environment pair, then the dependency map, then the bundle pair.",
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
