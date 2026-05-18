#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core preflight fixture comparator
"""Compare regenerated stable-core preflight fixtures.

The comparator is deterministic: it regenerates fixture JSON and Markdown
into a temporary directory (unless explicit ``--actual-*`` paths are
provided) and compares deterministic outputs against committed artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "stable_core" / "stable_core_preflight_fixtures.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "stable_core_preflight_fixtures.md"
DEFAULT_ACTUAL_JSON_NAME = "stable_core_preflight_fixtures.json"
DEFAULT_ACTUAL_MARKDOWN_NAME = "stable_core_preflight_fixtures.md"
SCHEMA_VERSION = "stable_core_preflight_fixtures_v1"

_PRE_FLIGHT_MODULE = "scpn_quantum_control.stable_core_preflight"
_PRE_FLIGHT_SOURCE_PATH = REPO_ROOT / "src" / "scpn_quantum_control" / "stable_core_preflight.py"
_PRE_FLIGHT_FALLBACK_MODULE = "_stable_core_preflight_fixtures_optional"


def stable_core_preflight_fixtures_payload() -> dict[str, Any]:
    """Return stable-core preflight fixture payload."""

    module = _load_stable_core_preflight_module()
    if module is not None:
        candidate = _resolve_callable(
            module,
            (
                "stable_core_preflight_fixtures_payload",
                "stable_core_preflight_payload",
                "preflight_fixtures_payload",
            ),
        )
        if candidate is not None:
            payload = candidate()
            if isinstance(payload, dict):
                return _normalised_json_payload(payload)

    return _fallback_stable_core_preflight_fixtures_payload()


def stable_core_preflight_fixtures_json(payload: dict[str, Any]) -> str:
    """Return deterministic JSON text for stable-core preflight fixtures."""

    module = _load_stable_core_preflight_module()
    if module is not None:
        candidate = _resolve_callable(
            module,
            (
                "stable_core_preflight_fixtures_json",
                "stable_core_preflight_json",
                "preflight_fixtures_json",
            ),
        )
        if candidate is not None:
            return candidate(payload)

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def stable_core_preflight_fixtures_markdown(payload: dict[str, Any]) -> str:
    """Return a deterministic markdown summary for stable-core preflight fixtures."""

    module = _load_stable_core_preflight_module()
    if module is not None:
        candidate = _resolve_callable(
            module,
            (
                "stable_core_preflight_fixtures_markdown",
                "stable_core_preflight_markdown",
                "preflight_fixtures_markdown",
            ),
        )
        if candidate is not None:
            return candidate(payload)

    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- stable core preflight fixtures -->",
        "",
        "# Stable Core Preflight Fixtures",
        "",
        "These no-QPU, no-network fixtures lock stable core preflight branches.",
        "",
        "## Fixture summary",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Hardware submission enabled in fixtures: `{payload['hardware_submission']}`",
        "",
        "## Preflight fixtures",
        "",
        "| Fixture ID | Status | Backend ID | Objective | Blockers | Primitives |",
        "|---|---|---|---|---|---|",
    ]

    for row in payload.get("fixtures", ()):
        lines.append(
            "| `{fixture_id}` | `{status}` | `{backend_id}` | `{objective}` | {blockers} | {primitives} |".format(
                fixture_id=row.get("fixture_id", ""),
                status=row.get("status", ""),
                backend_id=row.get("backend", {}).get("backend_id", ""),
                objective=row.get("objective", ""),
                blockers=", ".join(row.get("blockers", ())) or "`none`",
                primitives=", ".join(row.get("primitives", ())) or "`none`",
            )
        )

    lines.extend(
        [
            "",
            "## Reproducibility gate",
            "",
            "Regenerate and compare these fixtures with:",
            "",
            "```bash",
            "scpn-bench stable-core-preflight-gate",
            "```",
            "",
            "## Claim boundary",
            "",
            str(payload.get("claim_boundary", "")),
        ]
    )
    return "\n".join(lines) + "\n"


def write_stable_core_preflight_fixtures(
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, str]:
    """Write deterministic stable-core preflight fixture artifacts."""

    payload = stable_core_preflight_fixtures_payload()
    json_text = stable_core_preflight_fixtures_json(payload)
    markdown_text = stable_core_preflight_fixtures_markdown(payload)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")

    return {
        "json_sha256": _sha256(json_text),
        "markdown_sha256": _sha256(markdown_text),
    }


def compare_stable_core_preflight_fixtures(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path | None = None,
    actual_markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Compare committed stable-core preflight artifacts with generated ones."""

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
            generated_digests = write_stable_core_preflight_fixtures(
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
    expected_markdown = _normalise_markdown(
        _load_text(expected_markdown_path, blockers),
    )
    actual_markdown = _normalise_markdown(_load_text(actual_markdown_path, blockers))

    if expected_json is not None and actual_json is not None:
        if expected_json.get("schema") != SCHEMA_VERSION:
            blockers.append("stable core preflight schema mismatch")
            valid = False
        if _normalised_json(expected_json) != _normalised_json(actual_json):
            blockers.append("stable-core preflight JSON artifacts differ from committed version")
            valid = False

    if (
        expected_markdown is not None
        and actual_markdown is not None
        and expected_markdown != actual_markdown
    ):
        blockers.append("stable-core preflight Markdown artifacts differ from committed version")
        valid = False

    if not expected_json_path.exists() or not actual_json_path.exists():
        blockers.append(
            f"missing committed or generated preflight JSON: {expected_json_path} or {actual_json_path}"
        )
        valid = False
    if not expected_markdown_path.exists() or not actual_markdown_path.exists():
        blockers.append(
            f"missing committed or generated preflight Markdown: {expected_markdown_path} or {actual_markdown_path}"
        )
        valid = False

    if blockers:
        valid = False
    return valid


def _normalised_json(payload: dict[str, Any]) -> str:
    """Canonical JSON text for deterministic drift checks."""

    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _normalised_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return canonical payload shape for stable comparisons."""

    return json.loads(json.dumps(payload, sort_keys=True))


def _normalise_markdown(text: str) -> str:
    """Normalise markdown line endings for deterministic comparisons."""

    return text.replace("\r\n", "\n")


def _load_text(path: Path, blockers: list[str]) -> str:
    """Load text with deterministic blocker recording."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        blockers.append(f"unable to read {path}: {exc}")
        return ""


def _load_json(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    """Load JSON with deterministic blocker recording."""

    try:
        return json.loads(_load_text(path, blockers))
    except json.JSONDecodeError as exc:
        blockers.append(f"{path} must contain valid JSON: {exc}")
        return None


def _sha256(payload: str) -> str:
    """Return sha256 digest for stable text."""

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    """Return sha256 digest for existing path text."""

    try:
        return _sha256(path.read_text(encoding="utf-8"))
    except OSError:
        return ""


def _load_stable_core_preflight_module() -> ModuleType | None:
    """Import optional stable_core_preflight module without hard dependency."""

    try:
        return importlib.import_module(_PRE_FLIGHT_MODULE)
    except ImportError:
        # Optional module may be unavailable while worker 1 work is still in flight.
        pass

    try:
        if not _PRE_FLIGHT_SOURCE_PATH.exists():
            return None

        spec = importlib.util.spec_from_file_location(
            _PRE_FLIGHT_FALLBACK_MODULE,
            _PRE_FLIGHT_SOURCE_PATH,
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ImportError:
        return None

    return None


def _resolve_callable(module: ModuleType, candidates: tuple[str, ...]) -> Any:
    """Return first callable from candidate attributes on module."""

    for candidate in candidates:
        value = getattr(module, candidate, None)
        if callable(value):
            return value
    return None


def _fallback_stable_core_preflight_fixtures_payload() -> dict[str, Any]:
    """Fallback fixture payload before shared preflight module is added."""

    return {
        "schema": SCHEMA_VERSION,
        "hardware_submission": False,
        "fixtures": [
            {
                "fixture_id": "eligible_classical_reference",
                "status": "eligible",
                "backend": {
                    "backend_id": "classical-reference",
                    "kind": "classical_reference",
                    "capabilities": ("order_parameter", "parity", "fim", "control"),
                    "hardware_submission_allowed": False,
                },
                "objective": "order_parameter",
                "blockers": (),
                "primitives": (
                    "dependency_probe",
                    "capability_guard",
                    "preregistration_guard",
                    "eligible",
                ),
                "metadata": {
                    "scenario": "eligible classical/reference",
                },
            },
            {
                "fixture_id": "blocked_missing_dependency",
                "status": "blocked",
                "backend": {
                    "backend_id": "qiskit-runtime",
                    "kind": "qiskit",
                    "capabilities": ("order_parameter", "parity", "mitigation_replay"),
                    "hardware_submission_allowed": False,
                },
                "objective": "order_parameter",
                "blockers": ("missing dependency: qiskit-runtime provider package",),
                "primitives": (),
                "metadata": {
                    "scenario": "blocked missing dependency",
                },
            },
            {
                "fixture_id": "blocked_hardware_preregistration_or_boundary",
                "status": "blocked",
                "backend": {
                    "backend_id": "qiskit-runtime-live",
                    "kind": "qiskit",
                    "capabilities": ("order_parameter", "parity", "mitigation_replay"),
                    "hardware_submission_allowed": True,
                },
                "objective": "order_parameter",
                "blockers": (
                    "hardware preregistration required for live submission",
                    "hardware boundary blocks run-path in stable fixture mode",
                ),
                "primitives": (),
                "metadata": {
                    "scenario": "blocked hardware preregistration or boundary",
                },
            },
            {
                "fixture_id": "blocked_missing_capability",
                "status": "blocked",
                "backend": {
                    "backend_id": "qutip-dynamics",
                    "kind": "qutip",
                    "capabilities": ("order_parameter", "hamiltonian_dynamics", "lindblad"),
                    "hardware_submission_allowed": False,
                },
                "objective": "control_cost",
                "blockers": ("backend qutip-dynamics does not declare control capability",),
                "primitives": (),
                "metadata": {
                    "scenario": "blocked missing capability",
                    "required_capability": "control",
                },
            },
        ],
        "claim_boundary": (
            "Preflight fixtures are offline and do not prove runtime execution, "
            "hardware registration, or external dependency readiness. "
            "They only lock shape checks for deterministic guard branches."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for stable-core preflight fixture comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--expected-markdown", type=Path, default=DEFAULT_EXPECTED_MARKDOWN)
    parser.add_argument("--actual-json", type=Path)
    parser.add_argument("--actual-markdown", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON payload")
    args = parser.parse_args(argv)

    payload = compare_stable_core_preflight_fixtures(
        expected_json_path=args.expected_json,
        expected_markdown_path=args.expected_markdown,
        actual_json_path=args.actual_json,
        actual_markdown_path=args.actual_markdown,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"stable core preflight fixture comparison valid: {payload['valid']}")
        for blocker in payload["blockers"]:
            print(f"  blocker: {blocker}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
